"""
AgentGEO V2 Suggestion Processing Coordinator
Coordinates complete batch processing workflow, fully based on geo_agent architecture
"""
import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
GEO_AGENT_ROOT = REPO_ROOT / "geo_agent"
if str(GEO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_AGENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geo_agent.utils.structural_parser import StructuralHtmlParser, ContentChunk

from .memory_manager import HistoryManagerV2, OptimizationMemoryV2
from .models import (
    AppliedToolInfo,
    AgentGEOConfigV2,
    OptimizationResultV2,
    OrchestraGroupV2,
    QueryResultV2,
    SuggestionV2,
)
from .segment_orchestra import SegmentOrchestraV2, create_orchestras_from_chunks
from .suggestion_collector import SuggestionCollectorV2

logger = logging.getLogger(__name__)


class CoreIdeaManagerV2:
    """V2 Core idea manager"""

    def __init__(self, llm):
        self.llm = llm
        self.core_ideas: Dict[int, str] = {}

    async def extract_core_idea(self, content: str) -> str:
        """Extract core idea"""
        from langchain_core.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_template(
            """
Analyze this document and extract its CORE IDEA in 2-3 sentences.

Focus on:
- What is the PRIMARY topic/subject of this document?
- What is the main purpose or theme?
- What key concepts must be preserved?

Document:
{content}

CORE IDEA (2-3 sentences):
"""
        )

        try:
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(prompt.format(content=content[:3000]))
            else:
                response = await asyncio.to_thread(
                    self.llm.invoke, prompt.format(content=content[:3000])
                )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Core idea extraction failed: {e}")
            return ""

    async def extract_all_orchestra_core_ideas(
        self,
        chunks: List[ContentChunk],
        orchestras: List[OrchestraGroupV2],
    ) -> None:
        """Extract core ideas for all orchestras"""

        async def extract_one(orch: OrchestraGroupV2) -> Tuple[int, str]:
            content = "\n".join(
                chunks[idx].text for idx in orch.segment_indices if idx < len(chunks)
            )
            core_idea = await self.extract_core_idea(content)
            return orch.orchestra_id, core_idea

        tasks = [asyncio.create_task(extract_one(orch)) for orch in orchestras]
        results = await asyncio.gather(*tasks)

        for orch_id, core_idea in results:
            self.core_ideas[orch_id] = core_idea
            # Update orchestra
            for orch in orchestras:
                if orch.orchestra_id == orch_id:
                    orch.core_idea = core_idea

    def get_all_core_ideas(self) -> Dict[int, str]:
        return self.core_ideas


class SuggestionProcessorV2:
    """
    V2 Suggestion Processing Coordinator - AgentGEO core processing workflow

    Fully based on geo_agent architecture, uses two-phase failure analysis

    Workflow:
    1. Parse HTML and get chunks
    2. Dynamically create Orchestras
    3. Extract core ideas in parallel
    4. Collect suggestions in parallel (using two-phase analysis)
    5. Synthesize decisions through Orchestra in parallel
    6. Apply modifications
    7. Update history records
    """

    def __init__(
        self,
        llm,
        generator,  # AsyncInContextGenerator
        config: AgentGEOConfigV2,
        history_manager: Optional[HistoryManagerV2] = None,
        search_func: Optional[Callable] = None,
        competitor_content_func: Optional[Callable] = None,
    ):
        self.llm = llm
        self.generator = generator
        self.config = config
        self.struct_parser = StructuralHtmlParser(min_length=50)
        self.core_idea_manager = CoreIdeaManagerV2(llm)

        # History manager
        if history_manager:
            self.history_manager = history_manager
        elif config.history_persist_path:
            self.history_manager = HistoryManagerV2(
                persist_path=Path(config.history_persist_path)
            )
        else:
            self.history_manager = HistoryManagerV2()

        # Search and content retrieval functions
        self.search_func = search_func
        self.competitor_content_func = competitor_content_func

    def _create_orchestras(
        self, chunks: List[ContentChunk]
    ) -> List[OrchestraGroupV2]:
        """Dynamically create Orchestras from ContentChunks"""
        n = len(chunks)
        orchestras = []
        ppo = self.config.chunks_per_orchestra

        for i in range(0, n, ppo):
            indices = list(range(i, min(i + ppo, n)))
            original_content = "\n".join(chunks[idx].text for idx in indices)
            original_html = "\n".join(chunks[idx].html for idx in indices)

            orchestras.append(
                OrchestraGroupV2(
                    orchestra_id=len(orchestras),
                    segment_indices=indices,
                    original_content=original_content,
                    original_html=original_html,
                )
            )

        return orchestras

    async def process_batch(
        self,
        content: str,
        queries: List[str],
        raw_html: Optional[str] = None,
    ) -> OptimizationResultV2:
        """
        Process a batch of queries

        Args:
            content: Current document content (plain text)
            queries: List of queries to process
            raw_html: HTML content (used preferentially)

        Returns:
            OptimizationResultV2: Batch result
        """
        batch_id = str(uuid.uuid4())[:8]
        print(f"\n=== Processing Batch {batch_id} (V2) ===")
        print(f"Queries: {len(queries)}")

        # 1. Parse HTML and get chunks
        js_fallback_mode = False  # Flag for JS fallback mode
        if raw_html:
            current_structure = self.struct_parser.parse(raw_html)
            current_structure.calculate_chunks(max_chunk_length=2000)
            chunks = current_structure._chunks
            current_text = current_structure.get_clean_text()

            # JS Fallback: when chunks=0, create virtual chunk containing entire HTML
            # This allows static_rendering tool to try extracting content from JS/JSON
            if not chunks and raw_html:
                from geo_agent.utils.structural_parser import ContentChunk
                # Create virtual chunk containing entire HTML
                # Note: use 'id' key name (not 'geo_id') to match apply_modification_to_live expectations
                virtual_element = {
                    'text_content': content if content else '',  # Use passed content as text
                    'original_html': raw_html,  # Keep complete HTML for tool processing
                    'id': 'virtual-js-chunk-0',  # Identifier for virtual chunk
                }
                virtual_chunk = ContentChunk(index=0, elements=[virtual_element])
                chunks = [virtual_chunk]
                current_structure._chunks = chunks  # Update chunks in structure
                js_fallback_mode = True
                print(f"[JS Fallback] Created virtual chunk from raw HTML ({len(raw_html)} chars)")
        else:
            wrapped_html = f"<html><body><p>{content}</p></body></html>"
            current_structure = self.struct_parser.parse(wrapped_html)
            current_structure.calculate_chunks(max_chunk_length=2000)
            chunks = current_structure._chunks
            current_text = content

        print(f"Chunks: {len(chunks)}{' (JS fallback)' if js_fallback_mode else ''}")

        # 2. Dynamically create Orchestras
        orchestras = self._create_orchestras(chunks)
        print(f"Orchestras: {len(orchestras)}")

        # 3. Extract core ideas in parallel
        await self.core_idea_manager.extract_all_orchestra_core_ideas(chunks, orchestras)
        core_ideas = self.core_idea_manager.get_all_core_ideas()

        # 4. Get history context (only when enable_history=True)
        history_context = self.history_manager.get_preservation_rules() if self.config.enable_history else ""

        # 5. Create SuggestionCollector
        # Decide whether to exclude autogeo_rephrase tool based on config
        excluded_tools = [] if self.config.enable_autogeo_rephrase else ["autogeo_rephrase"]

        collector = SuggestionCollectorV2(
            llm=self.llm,
            generator=self.generator,
            chunks=chunks,
            core_ideas=core_ideas,
            history_context=history_context,
            chunks_per_orchestra=self.config.chunks_per_orchestra,
            history_manager=self.history_manager,
            enable_policy_injection=self.config.enable_policy_injection,
            enable_memory=self.config.enable_memory,
            enable_history=self.config.enable_history,
            excluded_tools=excluded_tools,
        )

        # 6. Define retrieval and citation check functions
        async def get_retrieved_docs(query: str):
            if self.search_func:
                return await self.search_func(query)
            return []

        async def get_competitor_contents(docs):
            if self.competitor_content_func:
                result = await self.competitor_content_func(docs)
                # Compatible with tuple return (filtered_docs, contents)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
                return (docs, result)  # Compatible with old format
            return ([], [])

        async def check_citation(query, current_content, retrieved_docs, competitor_contents):
            from geo_agent.core.models import WebPage
            temp_page = WebPage(
                url="",
                raw_html="",
                cleaned_content=current_content,
            )
            return await self.generator.generate_and_check(
                query, temp_page, retrieved_docs, competitor_contents
            )

        # 7. Collect suggestions in parallel (pass HTML for temporary structure testing)
        print(f"Collecting suggestions for {len(queries)} queries...")
        # V2.1: Always provide HTML for temporary structure testing per query
        current_html_for_test = current_structure.export_html()
        query_results = await collector.collect_batch(
            queries=queries,
            current_content=current_text,
            current_html=current_html_for_test,
            retrieved_docs_func=get_retrieved_docs,
            competitor_contents_func=get_competitor_contents,
            check_citation_func=check_citation,
            max_concurrency=self.config.max_concurrency,
            max_retries_per_query=self.config.max_retries_per_query,
        )

        # 8. Collect all suggestions
        all_suggestions: List[SuggestionV2] = []
        for qr in query_results:
            all_suggestions.extend(qr.suggestions)

        print(f"Collected {len(all_suggestions)} suggestions")

        # 9. Create SegmentOrchestras and assign suggestions
        segment_orchestras: List[SegmentOrchestraV2] = []
        for orch in orchestras:
            segment_orch = SegmentOrchestraV2(
                orchestra_group=orch,
                chunks=chunks,
                history_context=history_context,
                llm=self.llm,
            )

            # Filter suggestions belonging to this partition
            relevant = [
                s
                for s in all_suggestions
                if s.target_segment_index in orch.segment_indices
            ]
            segment_orch.add_suggestions(relevant)
            segment_orchestras.append(segment_orch)

        # 10. Synthesize decisions in parallel to generate final modifications
        print("Synthesizing modifications...")
        synthesis_tasks = [
            orch.synthesize_modifications(
                strategy=self.config.suggestion_merge_strategy,
                max_per_segment=self.config.max_suggestions_per_segment,
            )
            for orch in segment_orchestras
        ]
        synthesis_results = await asyncio.gather(*synthesis_tasks)

        # Merge all modifications
        all_modifications: Dict[int, str] = {}
        applied_suggestion_ids: List[str] = []

        for orch, modifications in zip(segment_orchestras, synthesis_results):
            all_modifications.update(modifications)
            applied_suggestion_ids.extend(orch.orchestra_group.applied_suggestions)

        # V2.4: Collect applied tool details
        suggestion_map = {s.suggestion_id: s for s in all_suggestions}
        applied_tools: List[AppliedToolInfo] = []
        for sid in applied_suggestion_ids:
            if sid in suggestion_map:
                applied_tools.append(AppliedToolInfo.from_suggestion(suggestion_map[sid]))

        # 11. Apply modifications to HTML structure
        if js_fallback_mode:
            # JS Fallback mode: directly use final_html from query_results
            # Virtual chunks have no real DOM anchors, cannot do chunk replacement
            final_html_from_results = None
            for qr in query_results:
                if qr.final_html:
                    final_html_from_results = qr.final_html
                    break  # Only one query, take first valid one

            if final_html_from_results:
                current_structure = self.struct_parser.parse(final_html_from_results)
                current_structure.calculate_chunks(max_chunk_length=2000)
                chunks = current_structure._chunks
                print(f"[JS Fallback] Using final_html from suggestion_collector")
            else:
                print(f"[JS Fallback] No final_html available, content unchanged")
        else:
            # Normal mode: reverse traversal to avoid index drift
            for seg_idx in sorted(all_modifications.keys(), reverse=True):
                if seg_idx < len(chunks):
                    modified_html = all_modifications[seg_idx]
                    current_structure.replace_chunk_by_index(
                        seg_idx, modified_html, highlight=True
                    )

            # After all modifications, re-parse HTML to update extracted_elements
            # This is critical: replace_chunk_by_index only modifies DOM, doesn't update extracted_elements
            # Must re-parse for get_clean_text() to return latest content
            current_structure = self.struct_parser.parse(current_structure.export_html())
            current_structure.calculate_chunks(max_chunk_length=2000)
            chunks = current_structure._chunks

        # Get final content
        new_html = current_structure.export_html()
        new_content = current_structure.get_clean_text()

        # 12. Calculate success rate
        cited_count = sum(1 for qr in query_results if qr.is_cited)
        success_before = cited_count / len(query_results) if query_results else 0

        # 13. Create result
        result = OptimizationResultV2(
            batch_id=batch_id,
            queries=queries,
            query_results=query_results,
            all_suggestions=all_suggestions,
            applied_modifications=applied_suggestion_ids,
            content_before=content,
            content_after=new_content,
            success_rate_before=success_before,
            html_after=new_html,
            applied_tools=applied_tools,  # V2.4: Add tool details
        )

        # Calculate diagnosis statistics
        result.compute_diagnosis_stats()
        # Calculate GEO Score statistics (V2.3 addition)
        result.compute_geo_score_stats()

        # 14. Update history (only when enable_history=True)
        if self.config.enable_cross_batch_history and self.config.enable_history:
            self.history_manager.add_batch_result(result)

        print(f"Batch {batch_id} completed:")
        print(f"  - Suggestions collected: {len(all_suggestions)}")
        print(f"  - Modifications applied: {len(applied_suggestion_ids)}")
        print(f"  - Success rate before: {success_before:.2%}")
        print(f"  - Diagnosis stats: {result.diagnosis_stats}")
        print(f"  - GEO Score: overall={result.avg_geo_score_overall:.4f}, word={result.avg_geo_score_word:.4f}, pos={result.avg_geo_score_position:.4f}")
        print(f"  - Valid citation rate: {result.valid_citation_rate:.2%}")

        return result

    async def process_all_batches(
        self,
        content: str,
        all_queries: List[str],
        raw_html: Optional[str] = None,
    ) -> List[OptimizationResultV2]:
        """
        Process all queries with automatic batching

        Args:
            content: Plain text content
            all_queries: All query list
            raw_html: HTML content (used preferentially)

        Returns:
            List[OptimizationResultV2]: All batch results
        """
        results: List[OptimizationResultV2] = []
        current_content = content
        current_html = raw_html

        total_batches = (
            len(all_queries) + self.config.batch_size - 1
        ) // self.config.batch_size

        print(f"\n{'='*50}")
        print(f"Starting Batch GEO V2 Processing")
        print(f"Total queries: {len(all_queries)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Total batches: {total_batches}")
        print(f"Using HTML mode: {raw_html is not None}")
        print(f"Two-phase analysis: {self.config.use_two_phase_analysis}")
        print(f"{'='*50}")

        for i in range(0, len(all_queries), self.config.batch_size):
            batch_queries = all_queries[i : i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1

            print(f"\n>>> Batch {batch_num}/{total_batches}")

            result = await self.process_batch(
                current_content, batch_queries, raw_html=current_html
            )
            results.append(result)

            # Update content for next batch
            current_content = result.content_after
            if result.html_after:
                current_html = result.html_after

        # Aggregate statistics
        total_suggestions = sum(len(r.all_suggestions) for r in results)
        total_applied = sum(len(r.applied_modifications) for r in results)
        overall_diagnosis_stats: Dict[str, int] = {}
        for r in results:
            for cause, count in r.diagnosis_stats.items():
                overall_diagnosis_stats[cause] = overall_diagnosis_stats.get(cause, 0) + count

        # GEO Score aggregation (V2.3 addition)
        total_queries = sum(len(r.query_results) for r in results)
        if total_queries > 0:
            avg_geo_overall = sum(r.avg_geo_score_overall * len(r.query_results) for r in results) / total_queries
            avg_geo_word = sum(r.avg_geo_score_word * len(r.query_results) for r in results) / total_queries
            avg_geo_pos = sum(r.avg_geo_score_position * len(r.query_results) for r in results) / total_queries
            valid_count = sum(sum(1 for qr in r.query_results if qr.has_valid_citations) for r in results)
            overall_valid_rate = valid_count / total_queries
        else:
            avg_geo_overall = avg_geo_word = avg_geo_pos = overall_valid_rate = 0.0

        print(f"\n{'='*50}")
        print(f"Batch GEO V2 Processing Complete")
        print(f"Total suggestions: {total_suggestions}")
        print(f"Total applied: {total_applied}")
        print(f"Overall diagnosis stats: {overall_diagnosis_stats}")
        print(f"Overall GEO Score: overall={avg_geo_overall:.4f}, word={avg_geo_word:.4f}, pos={avg_geo_pos:.4f}")
        print(f"Overall valid citation rate: {overall_valid_rate:.2%}")
        print(f"{'='*50}")

        return results
