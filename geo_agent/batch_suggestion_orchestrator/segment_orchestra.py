"""
Batch GEO V2 Segment Orchestrator
Intelligent suggestion merging and modification synthesis based on diagnostic results
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

# Setup paths
REPO_ROOT = Path(__file__).resolve().parents[1]
GEO_AGENT_ROOT = REPO_ROOT / "geo_agent"
if str(GEO_AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(GEO_AGENT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))

from geo_agent.utils.structural_parser import ContentChunk

from .models import OrchestraGroupV2, SuggestionV2
from .tool_executor import ToolExecutor

logger = logging.getLogger(__name__)


class SegmentOrchestraV2:
    """
    V2 Segment Orchestrator

    Responsible for:
    1. Managing modification suggestions for a group of chunks
    2. Intelligent merging of suggestions based on diagnostic results
    3. Executing synthesized modifications

    New features:
    - Diagnosis-aware suggestion priority ranking
    - Conflict detection and resolution
    - Incremental modification support
    """

    def __init__(
        self,
        orchestra_group: OrchestraGroupV2,
        chunks: List[ContentChunk],
        history_context: str = "",
        config_path: str = "geo_agent/config.yaml",
        llm = None,
    ):
        self.orchestra_group = orchestra_group
        self.chunks = chunks
        self.history_context = history_context
        self.config_path = config_path
        self.llm = llm
        self.tool_executor = ToolExecutor()

        # Suggestions organized by chunk index
        self._suggestions_by_chunk: Dict[int, List[SuggestionV2]] = {}

    def add_suggestions(self, suggestions: List[SuggestionV2]) -> None:
        """Add suggestions to orchestrator"""
        for suggestion in suggestions:
            # Only accept suggestions belonging to this partition
            if suggestion.target_segment_index in self.orchestra_group.segment_indices:
                self.orchestra_group.add_suggestion(suggestion)

                # Organize by chunk index
                idx = suggestion.target_segment_index
                if idx not in self._suggestions_by_chunk:
                    self._suggestions_by_chunk[idx] = []
                self._suggestions_by_chunk[idx].append(suggestion)

    def get_suggestions_for_chunk(self, chunk_index: int) -> List[SuggestionV2]:
        """Get all suggestions for specific chunk"""
        return self._suggestions_by_chunk.get(chunk_index, [])

    def _rank_suggestions_by_diagnosis(
        self, suggestions: List[SuggestionV2]
    ) -> List[SuggestionV2]:
        """
        Rank suggestions based on diagnostic results

        Ranking criteria:
        1. Severity
        2. Confidence
        3. Iteration count (later preferred, as earlier approaches already tried)
        """
        def priority_key(s: SuggestionV2) -> float:
            return s.get_priority_score()

        return sorted(suggestions, key=priority_key, reverse=True)

    def _detect_conflicts(
        self, suggestions: List[SuggestionV2]
    ) -> Dict[str, List[SuggestionV2]]:
        """
        Detect conflicting suggestions

        Conflict types:
        - same_tool: Same tool selected multiple times
        - opposite_diagnosis: Opposite diagnostic results (e.g., MISSING_INFO vs LOW_INFO_DENSITY)
        """
        conflicts: Dict[str, List[SuggestionV2]] = {
            "same_tool": [],
            "same_chunk": [],
        }

        # Group by tool
        by_tool: Dict[str, List[SuggestionV2]] = {}
        for s in suggestions:
            if s.tool_name not in by_tool:
                by_tool[s.tool_name] = []
            by_tool[s.tool_name].append(s)

        for tool, group in by_tool.items():
            if len(group) > 1:
                conflicts["same_tool"].extend(group)

        # Multiple suggestions for same chunk
        for idx, group in self._suggestions_by_chunk.items():
            if len(group) > 1:
                conflicts["same_chunk"].extend(group)

        return conflicts

    async def synthesize_modifications(
        self,
        strategy: str = "diagnosis_aware",
        max_per_segment: int = 3,
    ) -> Dict[int, str]:
        """
        Synthesize multiple suggestions to generate final modifications

        Args:
            strategy: Merge strategy
                - "diagnosis_aware": Intelligent merging based on diagnostic results
                - "priority": Select highest priority
                - "vote": Vote for most common tool
                - "llm_merge": Use LLM to merge
            max_per_segment: Max suggestions to apply per segment

        Returns:
            Dict[int, str]: chunk_index -> modified content
        """
        modifications: Dict[int, str] = {}
        applied_ids: List[str] = []

        for chunk_idx in self.orchestra_group.segment_indices:
            chunk_suggestions = self.get_suggestions_for_chunk(chunk_idx)

            if not chunk_suggestions:
                continue

            # 1. Rank suggestions
            ranked = self._rank_suggestions_by_diagnosis(chunk_suggestions)

            # 2. Select suggestions to apply based on strategy
            if strategy == "diagnosis_aware":
                selected = self._select_diagnosis_aware(ranked, max_per_segment)
            elif strategy == "priority":
                selected = ranked[:1]  # Only select highest priority
            elif strategy == "vote":
                selected = self._select_by_vote(ranked)
            elif strategy == "llm_merge":
                selected = await self._select_by_llm(ranked, chunk_idx)
            else:
                selected = ranked[:max_per_segment]

            if not selected:
                continue

            # 3. Apply selected suggestions
            # Use content from first (highest priority) suggestion
            best = selected[0]
            if best.proposed_content:
                modifications[chunk_idx] = best.proposed_content
                applied_ids.append(best.suggestion_id)
                logger.info(
                    f"Applied suggestion {best.suggestion_id} to chunk {chunk_idx}: "
                    f"tool={best.tool_name}, diagnosis={best.diagnosis.root_cause if best.diagnosis else 'N/A'}"
                )

        self.orchestra_group.applied_suggestions = applied_ids
        return modifications

    def _select_diagnosis_aware(
        self, suggestions: List[SuggestionV2], max_count: int
    ) -> List[SuggestionV2]:
        """
        Intelligent selection based on diagnostic results

        Rules:
        1. Prioritize high severity suggestions
        2. Avoid selecting multiple suggestions with same diagnosis type
        3. Consider tool diversity
        """
        selected: List[SuggestionV2] = []
        seen_diagnoses: set = set()
        seen_tools: set = set()

        for s in suggestions:
            if len(selected) >= max_count:
                break

            diagnosis = s.diagnosis.root_cause if s.diagnosis else "UNKNOWN"
            tool = s.tool_name

            # Avoid duplicate diagnosis types (unless high severity)
            if diagnosis in seen_diagnoses:
                if s.diagnosis and s.diagnosis.severity not in ["critical", "high"]:
                    continue

            # Avoid duplicate tools (unless diagnosis is different)
            if tool in seen_tools and diagnosis in seen_diagnoses:
                continue

            selected.append(s)
            seen_diagnoses.add(diagnosis)
            seen_tools.add(tool)

        return selected

    def _select_by_vote(self, suggestions: List[SuggestionV2]) -> List[SuggestionV2]:
        """Select most common tool by voting"""
        tool_counts: Dict[str, int] = {}
        tool_suggestions: Dict[str, List[SuggestionV2]] = {}

        for s in suggestions:
            if s.tool_name not in tool_counts:
                tool_counts[s.tool_name] = 0
                tool_suggestions[s.tool_name] = []
            tool_counts[s.tool_name] += 1
            tool_suggestions[s.tool_name].append(s)

        if not tool_counts:
            return []

        # Select tool with most votes
        best_tool = max(tool_counts, key=tool_counts.get)
        return tool_suggestions[best_tool][:1]

    async def _select_by_llm(
        self, suggestions: List[SuggestionV2], chunk_idx: int
    ) -> List[SuggestionV2]:
        """Intelligent selection using LLM"""
        if not self.llm or len(suggestions) <= 1:
            return suggestions[:1]

        # Prepare suggestion descriptions
        suggestions_desc = []
        for i, s in enumerate(suggestions):
            diagnosis_info = ""
            if s.diagnosis:
                diagnosis_info = f"Diagnosis: {s.diagnosis.root_cause} ({s.diagnosis.severity})"
            suggestions_desc.append(
                f"[Suggestion {i}]\n"
                f"Tool: {s.tool_name}\n"
                f"Query: {s.query}\n"
                f"{diagnosis_info}\n"
                f"Reasoning: {s.reasoning}\n"
                f"Key Changes: {', '.join(s.key_changes)}"
            )

        prompt = ChatPromptTemplate.from_template(
            """
You are helping to select the best modification suggestion for a document chunk.

Chunk Index: {chunk_idx}

Available Suggestions:
{suggestions}

History Context:
{history}

Select the BEST suggestion index (0-{max_idx}) based on:
1. Severity of the diagnosed issue
2. Likelihood of improving citation
3. Avoiding conflicts with previous modifications

Output only the index number (e.g., "0" or "1").
"""
        )

        final_prompt = prompt.format(
            chunk_idx=chunk_idx,
            suggestions="\n\n".join(suggestions_desc),
            history=self.history_context,
            max_idx=len(suggestions) - 1,
        )

        try:
            if hasattr(self.llm, "ainvoke"):
                response = await self.llm.ainvoke(final_prompt)
            else:
                response = await asyncio.to_thread(self.llm.invoke, final_prompt)

            # Parse index
            idx_str = response.content.strip()
            idx = int(idx_str)
            if 0 <= idx < len(suggestions):
                return [suggestions[idx]]
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")

        return suggestions[:1]

    def get_diagnosis_summary(self) -> Dict[str, int]:
        """Get diagnosis summary for this partition"""
        return self.orchestra_group.diagnosis_summary

    def get_dominant_failure(self) -> Optional[str]:
        """Get most common failure type"""
        return self.orchestra_group.get_dominant_failure()


async def create_orchestras_from_chunks(
    chunks: List[ContentChunk],
    chunks_per_orchestra: int = 2,
    history_context: str = "",
    config_path: str = "geo_agent/config.yaml",
    llm = None,
) -> List[SegmentOrchestraV2]:
    """
    Create orchestras from chunks

    Args:
        chunks: ContentChunk list
        chunks_per_orchestra: Number of chunks per orchestra
        history_context: History context
        config_path: Config path
        llm: LLM instance

    Returns:
        List[SegmentOrchestraV2]: Orchestra list
    """
    orchestras = []
    n = len(chunks)

    for i in range(0, n, chunks_per_orchestra):
        indices = list(range(i, min(i + chunks_per_orchestra, n)))

        # Collect content
        original_content = "\n".join(chunks[idx].text for idx in indices)
        original_html = "\n".join(chunks[idx].html for idx in indices)

        group = OrchestraGroupV2(
            orchestra_id=len(orchestras),
            segment_indices=indices,
            original_content=original_content,
            original_html=original_html,
        )

        orchestra = SegmentOrchestraV2(
            orchestra_group=group,
            chunks=chunks,
            history_context=history_context,
            config_path=config_path,
            llm=llm,
        )

        orchestras.append(orchestra)

    return orchestras
