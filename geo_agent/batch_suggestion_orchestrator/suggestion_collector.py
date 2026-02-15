"""
Batch GEO V2 Suggestion Collector
Collects optimization suggestions using two-phase analysis (diagnosis + strategy selection)

V2.1 Updates:
- Implements retry validation mechanism identical to GEO Agent
- Tests modifications on temporary structure, checks if effective
- Only returns suggestions that finally took effect
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
GEO_AGENT_ROOT = REPO_ROOT / "geo_agent"
if str(GEO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_AGENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geo_agent.core.models import CitationCheckResult
from geo_agent.core.memory import parse_tool_output
from geo_agent.utils.structural_parser import ContentChunk, StructuralHtmlParser
from geo_agent.tools import registry

# Reuse geo_agent core modules (aligned with geo_agent)
from geo_agent.agent.content_auditor import audit_content_truncation, TruncationAuditResult
from geo_agent.agent.policy_engine import PolicyEngine as GeoAgentPolicyEngine
from geo_agent.core.telemetry import (
    TelemetryStore,
    FailureCategory,
    ToolInvocationSpan,
    IterationMetrics,
    ToolOutcome,
    compute_args_hash,
)

from .failure_analysis import analyze_failure_async, DiagnosisResult, regenerate_tool_args_async
from .memory_manager import OptimizationMemoryV2, PolicyEngine as BatchPolicyEngine, HistoryManagerV2
from .models import DiagnosisInfo, QueryResultV2, SuggestionV2
from .tool_executor import ToolExecutor
from .citation_checker import compute_geo_score, GEOScoreInfo

logger = logging.getLogger(__name__)


class SuggestionCollectorV2:
    """
    V2 Suggestion Collector

    Uses two-phase analysis:
    1. Diagnose - Identify failure root cause
    2. Select Tool Strategy - Choose optimal tool

    Supports:
    - Parallel processing of multiple queries
    - Multiple retries per query
    - Diagnostic information recording
    - Strategy injection
    """

    def __init__(
        self,
        llm,
        generator: Any,  # AsyncInContextGenerator
        chunks: List[ContentChunk],
        core_ideas: Dict[int, str],
        history_context: str = "",
        chunks_per_orchestra: int = 2,
        history_manager: Optional[HistoryManagerV2] = None,
        enable_policy_injection: bool = True,
        enable_memory: bool = True,
        enable_history: bool = True,
        excluded_tools: Optional[List[str]] = None,
    ):
        self.llm = llm
        self.generator = generator
        self.chunks = chunks
        self.core_ideas = core_ideas
        self.history_context = history_context
        self.chunks_per_orchestra = chunks_per_orchestra
        self.tool_executor = ToolExecutor()
        self.history_manager = history_manager or HistoryManagerV2()
        self.enable_policy_injection = enable_policy_injection
        self.enable_memory = enable_memory
        self.enable_history = enable_history
        self.excluded_tools = excluded_tools or []

        # Memory for each query (for tracking retries)
        self._query_memories: Dict[str, OptimizationMemoryV2] = {}

    def _get_chunks_summary(self) -> str:
        """Generate chunks summary (for LLM to select target chunk)"""
        summary_parts = []
        for i, chunk in enumerate(self.chunks):
            text_preview = chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
            summary_parts.append(f"[CHUNK {i}]: {text_preview}")
        return "\n\n".join(summary_parts)

    def _format_indexed_content(self) -> str:
        """Generate full content with indexes"""
        return "\n\n".join(
            [f">> [CHUNK_ID: {i}]\n{chunk.text}" for i, chunk in enumerate(self.chunks)]
        )

    def _sanitize_for_logging(self, tool_args: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize tool arguments, remove lengthy content (aligned with geo_agent)"""
        sanitized = {}
        for key, value in tool_args.items():
            if key in ["target_content", "context_before", "context_after"]:
                sanitized[key] = f"<{len(str(value))} chars>" if value else ""
            elif key == "previous_modifications":
                lines = str(value).strip().split('\n') if value else []
                sanitized[key] = f"<{len(lines)} rules>"
            else:
                sanitized[key] = value
        return sanitized

    def _str_to_failure_category(self, diagnosis_str: str) -> FailureCategory:
        """Convert string diagnosis to FailureCategory enum"""
        try:
            return FailureCategory(diagnosis_str.lower())
        except ValueError:
            return FailureCategory.UNKNOWN

    def _get_orchestra_core_idea(self, chunk_index: int) -> str:
        """Get core idea for the orchestra that chunk belongs to"""
        orchestra_id = chunk_index // self.chunks_per_orchestra
        return self.core_ideas.get(orchestra_id, "")

    async def collect_for_query(
        self,
        query: str,
        original_content: str,
        original_html: str,
        retrieved_docs: List[Any],
        competitor_contents: List[str],
        max_retries: int = 3,
        check_citation_func: Optional[Callable] = None,
    ) -> QueryResultV2:
        """
        Collect suggestions for a single query (retry validation mechanism identical to GEO Agent)

        V2.2 Updates (aligned with geo_agent):
        - Reuse truncation audit audit_content_truncation()
        - Reuse policy engine PolicyEngine and telemetry store TelemetryStore
        - Two-phase strategy evaluation and tool override
        - Fast fail detection

        Core logic:
        1. Create temporary HTML structure for testing modifications
        2. Each iteration: export HTML -> reparse -> get plain text -> check citation
        3. If failed, execute tool, update temporary structure
        4. If successful, exit and return final suggestion

        Args:
            query: User query
            original_content: Original document content (plain text, unchanged)
            original_html: Original HTML (for creating temporary structure)
            retrieved_docs: Retrieved competitor documents
            competitor_contents: Competitor full contents
            max_retries: Maximum retry count
            check_citation_func: Citation check function

        Returns:
            QueryResultV2: Query result (containing final effective suggestion and final_html)
        """
        import time

        is_cited = False
        generated_answer = ""
        final_diagnosis: Optional[DiagnosisInfo] = None
        final_suggestion: Optional[SuggestionV2] = None
        all_suggestions: List[SuggestionV2] = []  # Accumulate all iteration suggestions
        iterations_used = 0
        # GEO Score info (V2.3 new)
        final_geo_score: Optional[GEOScoreInfo] = None

        # Create memory for this query
        memory = OptimizationMemoryV2()
        self._query_memories[query] = memory

        # ========== Initialize telemetry store and policy engine (reuse geo_agent) ==========
        telemetry = TelemetryStore(url="", core_idea=self.core_ideas.get(0, ""))
        geo_policy_engine = GeoAgentPolicyEngine(telemetry)

        # Keep original BatchPolicyEngine for strategy injection prompt generation
        batch_policy_engine = BatchPolicyEngine(
            self.history_manager,
            memory,
            enable_memory=self.enable_memory,
            enable_history=self.enable_history,
        )

        # ========== Dual structure strategy: frozen_structure (stable index) + temp_structure (latest content) ==========
        struct_parser = StructuralHtmlParser(min_length=50)

        # frozen_structure: calculate chunks only once, used for stable index mapping
        frozen_structure = struct_parser.parse(original_html)
        frozen_structure.calculate_chunks(max_chunk_length=2000)
        frozen_num_chunks = len(frozen_structure._chunks)

        # JS Fallback: when chunks=0, create virtual chunk containing entire HTML
        # This allows static_rendering tool to try extracting content from JS/JSON
        js_fallback_mode = False
        if frozen_num_chunks == 0 and original_html:
            from geo_agent.utils.structural_parser import ContentChunk
            # Note: use 'id' key name (not 'geo_id') to match apply_modification_to_live expectations
            virtual_element = {
                'text_content': original_content if original_content else '',
                'original_html': original_html,
                'id': 'virtual-js-chunk-0',  # For identifying virtual chunk
            }
            virtual_chunk = ContentChunk(index=0, elements=[virtual_element])
            frozen_structure._chunks = [virtual_chunk]
            frozen_num_chunks = 1
            js_fallback_mode = True
            logger.info(f"[JS Fallback] Created virtual chunk from raw HTML ({len(original_html)} chars)")

        # temp_structure: dynamically updated DOM, for latest content state
        temp_structure = struct_parser.parse(original_html)
        if js_fallback_mode:
            # Sync temp_structure's virtual chunk
            temp_structure._chunks = frozen_structure._chunks.copy()

        # Truncation audit info (shared across iterations)
        truncation_summary: Optional[str] = None
        has_truncation_alert = False

        for iteration in range(max_retries):
            iterations_used = iteration + 1
            tool_start_time = time.time()
            tool_outcome = ToolOutcome.SKIPPED
            tool_error_msg: Optional[str] = None
            args_hash = ""

            logger.info(f"Query '{query[:50]}...' - Iteration {iteration}/{max_retries}")

            try:
                # ========== 1. Refresh structure (identical to GEO Agent) ==========
                current_raw_html = temp_structure.export_html()
                temp_structure = struct_parser.parse(current_raw_html)  # Reparse
                temp_content = temp_structure.get_clean_text()  # HTML -> plain text

                # If parsed content is empty, use original content and skip citation check
                content_empty = not temp_content or not temp_content.strip()
                if content_empty:
                    logger.warning(f"Empty content after parsing at iteration {iteration}, using original for analysis")
                    temp_content = original_content  # For subsequent truncation audit
                    is_cited = False
                    cited_indices = []
                elif check_citation_func:
                    citation_result: CitationCheckResult = await check_citation_func(
                        query, temp_content, retrieved_docs, competitor_contents
                    )
                    is_cited = citation_result.is_cited
                    generated_answer = citation_result.generated_answer
                    cited_indices = citation_result.citations_found_idx

                    # Calculate GEO Score (V2.3 new)
                    # target_idx defaults to competitor_contents count + 1 (target document at the end)
                    num_sources = len(competitor_contents) + 1
                    target_idx = num_sources  # Assume target document is the last one
                    final_geo_score = compute_geo_score(generated_answer, target_idx, num_sources)

                    if is_cited:
                        logger.info(f"✅ Query '{query[:50]}...' - Cited at iteration {iteration}!")
                        logger.info(f"📊 GEO Score: overall={final_geo_score.overall:.4f}, word={final_geo_score.word:.4f}, pos={final_geo_score.position:.4f}")
                        break
                else:
                    cited_indices = []

                # ========== 3. Prepare content needed for analysis ==========
                # Use dual structure strategy: index from frozen_structure (stable), content from temp_structure (latest)
                indexed_content = frozen_structure.format_indexed_content_with_live_dom(temp_structure)
                num_chunks = frozen_num_chunks  # Always use frozen chunk count

                # ========== 3.1 Truncation audit (reuse geo_agent) ==========
                if iteration == 0:  # Only execute truncation audit on first iteration
                    try:
                        audit_res = audit_content_truncation(
                            self.llm,
                            query,
                            full_text=temp_content,
                            visible_chunks_text=indexed_content
                        )
                        if audit_res.has_hidden_relevant_content:
                            logger.info(f"⚠️ Truncation Alert: {audit_res.summary_of_hidden_info}")
                            truncation_summary = audit_res.summary_of_hidden_info
                            has_truncation_alert = True
                    except Exception as e:
                        logger.warning(f"Truncation audit failed: {e}")

                # Prepare competitor content
                # Note: cited_indices is 1-based (LLM generates [1], [2], etc.)
                if cited_indices and competitor_contents:
                    valid_indices = [i for i in cited_indices if 1 <= i <= len(competitor_contents)]
                    if valid_indices:
                        competitor_content = "\n---\n".join(
                            [competitor_contents[i - 1][:3000] for i in valid_indices[:3]]  # i-1 converts to 0-based
                        )
                    else:
                        competitor_content = competitor_contents[0][:3000] if competitor_contents else ""
                else:
                    competitor_content = competitor_contents[0][:3000] if competitor_contents else ""

                # ========== 4. Phase 1 strategy evaluation (before diagnosis, based on truncation info) ==========
                # JS Fallback mode: force use PARSING_FAILURE diagnosis
                if js_fallback_mode:
                    pre_diagnosis_category = FailureCategory.PARSING_FAILURE
                    logger.info("[JS Fallback] Using PARSING_FAILURE diagnosis for JS/JSON content")
                elif has_truncation_alert:
                    pre_diagnosis_category = FailureCategory.CONTENT_TRUNCATED
                else:
                    pre_diagnosis_category = FailureCategory.UNKNOWN

                pre_policy_eval = geo_policy_engine.evaluate(
                    diagnosis_category=pre_diagnosis_category,
                    diagnosis_explanation="",
                    has_truncation_alert=has_truncation_alert,
                    hidden_content_summary=truncation_summary or ""
                )

                # ========== 4.1 Generate strategy injection (merge geo_agent strategy and batch strategy) ==========
                policy_injection = ""
                if self.enable_policy_injection:
                    # Prioritize geo_agent policy engine injection
                    if pre_policy_eval.injection_prompt:
                        policy_injection = pre_policy_eval.injection_prompt
                    elif self.enable_memory:
                        # Fallback to batch_policy_engine (only when enable_memory=True)
                        policy_injection = batch_policy_engine.generate_policy_injection(
                            current_diagnosis=final_diagnosis,
                            current_chunk_index=None,
                        )

                # ========== 5. Two-phase analysis ==========
                analysis, diagnosis = await analyze_failure_async(
                    llm=self.llm,
                    query=query,
                    indexed_target_doc=indexed_content,
                    competitor_doc=competitor_content,
                    memory=memory,
                    truncation_audit_summary=truncation_summary,  # Pass truncation info
                    policy_injection=policy_injection,
                    num_chunks=num_chunks,
                    excluded_tools=self.excluded_tools,
                )

                # Record diagnosis
                diagnosis_info = diagnosis.to_diagnosis_info()
                final_diagnosis = diagnosis_info
                logger.info(f"Diagnosis: {diagnosis.root_cause} - {diagnosis.key_deficiency}")

                # JS Fallback mode: force use static_rendering tool (first iteration only)
                if js_fallback_mode and iteration == 0 and "static_rendering" not in self.excluded_tools:
                    original_tool = analysis.selected_tool_name
                    analysis.selected_tool_name = "static_rendering"
                    analysis.tool_arguments = {}  # static_rendering doesn't need extra parameters
                    logger.info(f"[JS Fallback] Overriding tool: {original_tool} -> static_rendering")

                logger.info(f"Tool Selected: {analysis.selected_tool_name}")

                # ========== 5.1 Phase 2 strategy evaluation (after diagnosis) ==========
                diagnosis_category = self._str_to_failure_category(diagnosis.root_cause)

                policy_eval = geo_policy_engine.evaluate(
                    diagnosis_category=diagnosis_category,
                    diagnosis_explanation=diagnosis.explanation,
                    has_truncation_alert=has_truncation_alert,
                    hidden_content_summary=truncation_summary or "",
                    severity=diagnosis.severity
                )

                # ========== 5.2 Apply forced tool override (fix: regenerate arguments) ==========
                # Note: static_rendering tool in JS Fallback mode should not be overridden
                original_tool = analysis.selected_tool_name
                skip_policy_override = (js_fallback_mode and analysis.selected_tool_name == "static_rendering")
                if policy_eval.forced_tool and policy_eval.forced_tool != analysis.selected_tool_name and not skip_policy_override:
                    logger.info(f"🎯 Policy Override: {analysis.selected_tool_name} -> {policy_eval.forced_tool}")

                    # Regenerate arguments adapted to new tool (fix argument mismatch issue)
                    try:
                        history_context = memory.get_history_summary() if self.enable_memory and memory else ""
                        analysis = await regenerate_tool_args_async(
                            llm=self.llm,
                            forced_tool=policy_eval.forced_tool,
                            diagnosis=diagnosis,
                            query=query,
                            target_content_indexed=indexed_content,
                            history_context=history_context,
                            num_chunks=num_chunks,
                        )
                        logger.info(f"✅ Regenerated args for {policy_eval.forced_tool}")
                    except Exception as e:
                        logger.error(f"Failed to regenerate args for {policy_eval.forced_tool}: {e}")
                        # Fall back to original tool
                        analysis.selected_tool_name = original_tool
                        logger.warning(f"⚠️ Falling back to original tool: {original_tool}")

                # ========== 5.3 Check if tool is blocked ==========
                if analysis.selected_tool_name in policy_eval.blocked_tools:
                    logger.warning(f"Tool {analysis.selected_tool_name} is blocked by policy, trying next iteration")
                    tool_outcome = ToolOutcome.SKIPPED
                    continue

                # ========== 5.4 Fast fail detection (reuse geo_agent judgment logic) ==========
                is_fixable, fixable_reason = geo_policy_engine.is_category_fixable(diagnosis_category)
                if not is_fixable and diagnosis.severity == "critical":
                    logger.warning(f"⚡ Unfixable diagnosis: {diagnosis.root_cause} - {fixable_reason}")
                    break

                if policy_eval.should_skip:
                    logger.warning(f"⚡ Policy suggests skip: {policy_eval.skip_reason}")
                    if iteration >= 1:  # Already tried at least once
                        break

                # ========== 6. Prepare tool arguments (dual structure strategy) ==========
                target_chunk_index = analysis.target_chunk_index or 0
                if target_chunk_index >= num_chunks:
                    target_chunk_index = num_chunks - 1

                # Use dual structure strategy: frozen_structure index for positioning, temp_structure's latest content
                tool_args = analysis.tool_arguments.copy()
                tool_args.update(frozen_structure.get_chunk_tool_args_from_live(temp_structure, target_chunk_index))

                # Get orchestra's core idea
                orchestra_id = target_chunk_index // self.chunks_per_orchestra
                core_idea = self.core_ideas.get(orchestra_id, "")
                tool_args['core_idea'] = core_idea
                tool_args['previous_modifications'] = memory.get_preservation_rules() if self.enable_memory else ""

                # Add required parameters for content_relocation tool (aligned with geo_agent)
                if analysis.selected_tool_name == "content_relocation":
                    # Only override when truncation_summary has actual content, otherwise keep LLM's possibly generated value
                    if truncation_summary:
                        tool_args["hidden_content_summary"] = truncation_summary
                    elif "hidden_content_summary" not in tool_args:
                        tool_args["hidden_content_summary"] = ""
                    tool_args["query"] = query

                # Add required user_query parameter for intent_realignment tool
                if analysis.selected_tool_name == "intent_realignment":
                    tool_args["user_query"] = query

                # Add required target_query parameter for historical_redteam tool
                if analysis.selected_tool_name == "historical_redteam":
                    tool_args["target_query"] = query

                # ========== 6.1 Check duplicate invocation (identical to geo_agent) ==========
                args_hash = compute_args_hash(tool_args)
                is_dup, dup_msg = geo_policy_engine.check_duplicate_invocation(analysis.selected_tool_name, args_hash)
                if is_dup:
                    logger.warning(dup_msg)
                    tool_outcome = ToolOutcome.SKIPPED
                    continue

                # ========== 7. Execute tool ==========
                tool = registry.get_tool(analysis.selected_tool_name)
                if not tool:
                    logger.error(f"Tool {analysis.selected_tool_name} not found.")
                    tool_outcome = ToolOutcome.FAILED
                    tool_error_msg = f"Tool {analysis.selected_tool_name} not found"
                    continue

                try:
                    # Save original chunk content before modification
                    original_chunk_content = tool_args.get('target_content', '')

                    raw_output = await asyncio.to_thread(tool.run, tool_args)
                    modified_chunk_html, key_changes = parse_tool_output(raw_output)
                    tool_outcome = ToolOutcome.SUCCESS

                    # ========== 8. Update temporary structure (dual structure strategy: locate via frozen index, modify on temp) ==========
                    if js_fallback_mode:
                        # JS Fallback mode: virtual chunk has no real DOM anchor
                        # Replace entire structure with tool output's new HTML directly
                        wrapped_html = f"<html><body>{modified_chunk_html}</body></html>"
                        temp_structure = struct_parser.parse(wrapped_html)
                        temp_structure.calculate_chunks(max_chunk_length=2000)
                        logger.info(f"[JS Fallback] Replaced temp_structure with tool output ({len(modified_chunk_html)} chars)")
                    elif frozen_structure.apply_modification_to_live(temp_structure, target_chunk_index, modified_chunk_html, highlight=False):
                        logger.info(f"DOM updated successfully at frozen chunk index {target_chunk_index}")
                    else:
                        logger.warning(f"Failed to update DOM at frozen chunk index {target_chunk_index}")
                        tool_outcome = ToolOutcome.PARTIAL

                    # Create suggestion record (including new fields)
                    final_suggestion = SuggestionV2(
                        suggestion_id=str(uuid.uuid4())[:8],
                        query=query,
                        tool_name=analysis.selected_tool_name,
                        tool_arguments=analysis.tool_arguments,  # LLM original output
                        target_segment_index=target_chunk_index,
                        reasoning=analysis.reasoning,
                        proposed_content=modified_chunk_html,
                        original_content=original_chunk_content,
                        key_changes=key_changes,
                        diagnosis=diagnosis_info,
                        iteration=iteration,
                        confidence=self._calculate_confidence(diagnosis_info),
                        executed_arguments=self._sanitize_for_logging(tool_args),  # Actual executed arguments (sanitized)
                        truncation_info={
                            "has_alert": has_truncation_alert,
                            "summary": truncation_summary
                        } if has_truncation_alert else None,
                    )
                    all_suggestions.append(final_suggestion)

                    # Update memory (only when enable_memory=True)
                    if self.enable_memory:
                        from .memory_manager import ModificationRecordV2
                        record = ModificationRecordV2(
                            query=query,
                            tool_name=analysis.selected_tool_name,
                            reasoning=analysis.reasoning,
                            key_changes=key_changes,
                            diagnosis=diagnosis_info,
                            chunk_index=target_chunk_index,
                        )
                        memory.add_modification(record)

                    logger.info(f"Tool '{analysis.selected_tool_name}' executed, changes: {key_changes}")

                except Exception as e:
                    logger.error(f"Tool execution failed: {e}")
                    import traceback
                    traceback.print_exc()
                    tool_outcome = ToolOutcome.FAILED
                    tool_error_msg = str(e)

                # ========== 9. Record telemetry data (identical to geo_agent) ==========
                tool_duration = (time.time() - tool_start_time) * 1000
                tool_span = ToolInvocationSpan(
                    tool_name=analysis.selected_tool_name,
                    target_chunk_index=target_chunk_index,
                    arguments_hash=args_hash,
                    outcome=tool_outcome,
                    reasoning=analysis.reasoning,
                    duration_ms=tool_duration,
                    error_message=tool_error_msg
                )

                iteration_metrics = IterationMetrics(
                    iteration_index=iteration,
                    query=query,
                    full_doc_length=len(temp_content),
                    visible_chunk_length=len(indexed_content),
                    truncation_ratio=1 - len(indexed_content) / max(len(temp_content), 1),
                    chunk_count=num_chunks,
                    diagnosis_category=diagnosis_category,
                    diagnosis_explanation=diagnosis.explanation,
                    has_hidden_relevant_content=has_truncation_alert,
                    hidden_content_summary=truncation_summary or "",
                    tool_span=tool_span,
                    was_cited=is_cited
                )
                telemetry.record_iteration(iteration_metrics)

            except Exception as e:
                logger.error(f"Analysis failed for query '{query[:50]}...': {e}")
                import traceback
                traceback.print_exc()
                continue

        # Get final HTML
        final_html = temp_structure.export_html() if final_suggestion else None

        return QueryResultV2(
            query=query,
            is_cited=is_cited,
            generated_answer=generated_answer,
            suggestions=all_suggestions,
            diagnosis=final_diagnosis,
            iterations_used=iterations_used,
            final_html=final_html,
            # GEO Score fields (V2.3 new)
            geo_score_word=final_geo_score.word if final_geo_score else 0.0,
            geo_score_position=final_geo_score.position if final_geo_score else 0.0,
            geo_score_wordpos=final_geo_score.wordpos if final_geo_score else 0.0,
            geo_score_overall=final_geo_score.overall if final_geo_score else 0.0,
            has_valid_citations=final_geo_score.has_valid_citations if final_geo_score else False,
        )

    def _calculate_confidence(self, diagnosis: DiagnosisInfo) -> float:
        """Calculate confidence based on diagnosis"""
        severity_scores = {
            "critical": 0.9,
            "high": 0.8,
            "medium": 0.7,
            "low": 0.6,
        }
        return severity_scores.get(diagnosis.severity, 0.7)

    async def collect_batch(
        self,
        queries: List[str],
        current_content: str,
        current_html: str,
        retrieved_docs_func: Callable,
        competitor_contents_func: Callable,
        check_citation_func: Callable,
        max_concurrency: int = 4,
        max_retries_per_query: int = 3,
    ) -> List[QueryResultV2]:
        """
        Batch collect suggestions

        V2.1 Updates:
        - Pass original_html to each query's processing
        - Each query independently tests modifications on temporary structure

        Args:
            queries: List of queries
            current_content: Current content (plain text)
            current_html: Current HTML (required, for creating temporary structure)
            retrieved_docs_func: Function to get retrieved documents (query) -> List[SearchResult]
            competitor_contents_func: Function to get competitor contents (docs) -> List[str]
            check_citation_func: Citation check function
            max_concurrency: Maximum concurrency
            max_retries_per_query: Maximum retries per query

        Returns:
            List[QueryResultV2]: List of query results (each containing final_html)
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def process_query(query: str) -> QueryResultV2:
            async with semaphore:
                try:
                    # Get retrieved documents
                    retrieved_docs = await retrieved_docs_func(query)
                    if not retrieved_docs:
                        return QueryResultV2(
                            query=query,
                            is_cited=False,
                            generated_answer="",
                            suggestions=[],
                            error="No retrieved documents",
                        )

                    # Get competitor contents (returns filtered docs and contents)
                    result = await competitor_contents_func(retrieved_docs)
                    if isinstance(result, tuple) and len(result) == 2:
                        retrieved_docs, competitor_contents = result
                    else:
                        competitor_contents = result
                    if not competitor_contents:
                        return QueryResultV2(
                            query=query,
                            is_cited=False,
                            generated_answer="",
                            suggestions=[],
                            error="No competitor contents",
                        )

                    # Collect suggestions (pass HTML)
                    return await self.collect_for_query(
                        query=query,
                        original_content=current_content,
                        original_html=current_html,
                        retrieved_docs=retrieved_docs,
                        competitor_contents=competitor_contents,
                        max_retries=max_retries_per_query,
                        check_citation_func=check_citation_func,
                    )
                except Exception as e:
                    logger.error(f"Failed to process query '{query[:50]}...': {e}")
                    import traceback
                    traceback.print_exc()
                    return QueryResultV2(
                        query=query,
                        is_cited=False,
                        generated_answer="",
                        suggestions=[],
                        error=str(e),
                    )

        tasks = [asyncio.create_task(process_query(q)) for q in queries]
        results = await asyncio.gather(*tasks)

        return results

    def get_all_suggestions(self) -> List[SuggestionV2]:
        """Get all collected suggestions"""
        all_suggestions = []
        for memory in self._query_memories.values():
            # Extract suggestions from memory
            pass
        return all_suggestions

    def get_diagnosis_stats(self) -> Dict[str, int]:
        """Get diagnosis statistics"""
        stats: Dict[str, int] = {}
        for memory in self._query_memories.values():
            for cause, count in memory.diagnosis_stats.items():
                stats[cause] = stats.get(cause, 0) + count
        return stats
