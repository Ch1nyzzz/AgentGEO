"""
Batch GEO V2 分段编排器
基于诊断结果的智能建议合并和修改综合
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate

# 设置路径
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
    V2 分段编排器

    负责：
    1. 管理一组 chunks 的修改建议
    2. 基于诊断结果智能合并建议
    3. 执行综合后的修改

    新增特性：
    - 诊断感知的建议优先级排序
    - 冲突检测和解决
    - 增量修改支持
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

        # 按 chunk 索引组织的建议
        self._suggestions_by_chunk: Dict[int, List[SuggestionV2]] = {}

    def add_suggestions(self, suggestions: List[SuggestionV2]) -> None:
        """添加建议到编排器"""
        for suggestion in suggestions:
            # 只接受属于该分区的建议
            if suggestion.target_segment_index in self.orchestra_group.segment_indices:
                self.orchestra_group.add_suggestion(suggestion)

                # 按 chunk 索引组织
                idx = suggestion.target_segment_index
                if idx not in self._suggestions_by_chunk:
                    self._suggestions_by_chunk[idx] = []
                self._suggestions_by_chunk[idx].append(suggestion)

    def get_suggestions_for_chunk(self, chunk_index: int) -> List[SuggestionV2]:
        """获取特定 chunk 的所有建议"""
        return self._suggestions_by_chunk.get(chunk_index, [])

    def _rank_suggestions_by_diagnosis(
        self, suggestions: List[SuggestionV2]
    ) -> List[SuggestionV2]:
        """
        根据诊断结果对建议进行排序

        排序依据：
        1. 严重程度 (severity)
        2. 置信度 (confidence)
        3. 迭代次数（越晚越优先，因为已经尝试过早期方案）
        """
        def priority_key(s: SuggestionV2) -> float:
            return s.get_priority_score()

        return sorted(suggestions, key=priority_key, reverse=True)

    def _detect_conflicts(
        self, suggestions: List[SuggestionV2]
    ) -> Dict[str, List[SuggestionV2]]:
        """
        检测冲突的建议

        冲突类型：
        - same_tool: 同一工具被多次选择
        - opposite_diagnosis: 诊断结果相反（例如 MISSING_INFO vs LOW_INFO_DENSITY）
        """
        conflicts: Dict[str, List[SuggestionV2]] = {
            "same_tool": [],
            "same_chunk": [],
        }

        # 按工具分组
        by_tool: Dict[str, List[SuggestionV2]] = {}
        for s in suggestions:
            if s.tool_name not in by_tool:
                by_tool[s.tool_name] = []
            by_tool[s.tool_name].append(s)

        for tool, group in by_tool.items():
            if len(group) > 1:
                conflicts["same_tool"].extend(group)

        # 同一 chunk 的多个建议
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
        综合多个建议，生成最终修改

        Args:
            strategy: 合并策略
                - "diagnosis_aware": 基于诊断结果的智能合并
                - "priority": 按优先级选择最高的
                - "vote": 投票选择最常见的工具
                - "llm_merge": 使用 LLM 合并
            max_per_segment: 每个段落最多应用的建议数

        Returns:
            Dict[int, str]: chunk_index -> 修改后的内容
        """
        modifications: Dict[int, str] = {}
        applied_ids: List[str] = []

        for chunk_idx in self.orchestra_group.segment_indices:
            chunk_suggestions = self.get_suggestions_for_chunk(chunk_idx)

            if not chunk_suggestions:
                continue

            # 1. 排序建议
            ranked = self._rank_suggestions_by_diagnosis(chunk_suggestions)

            # 2. 根据策略选择要应用的建议
            if strategy == "diagnosis_aware":
                selected = self._select_diagnosis_aware(ranked, max_per_segment)
            elif strategy == "priority":
                selected = ranked[:1]  # 只选最高优先级
            elif strategy == "vote":
                selected = self._select_by_vote(ranked)
            elif strategy == "llm_merge":
                selected = await self._select_by_llm(ranked, chunk_idx)
            else:
                selected = ranked[:max_per_segment]

            if not selected:
                continue

            # 3. 应用选中的建议
            # 使用第一个（最高优先级）建议的内容
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
        基于诊断结果的智能选择

        规则：
        1. 优先选择高严重程度的建议
        2. 避免选择相同诊断类型的多个建议
        3. 考虑工具多样性
        """
        selected: List[SuggestionV2] = []
        seen_diagnoses: set = set()
        seen_tools: set = set()

        for s in suggestions:
            if len(selected) >= max_count:
                break

            diagnosis = s.diagnosis.root_cause if s.diagnosis else "UNKNOWN"
            tool = s.tool_name

            # 避免重复诊断类型（除非是高严重程度）
            if diagnosis in seen_diagnoses:
                if s.diagnosis and s.diagnosis.severity not in ["critical", "high"]:
                    continue

            # 避免重复工具（除非诊断不同）
            if tool in seen_tools and diagnosis in seen_diagnoses:
                continue

            selected.append(s)
            seen_diagnoses.add(diagnosis)
            seen_tools.add(tool)

        return selected

    def _select_by_vote(self, suggestions: List[SuggestionV2]) -> List[SuggestionV2]:
        """通过投票选择最常见的工具"""
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

        # 选择票数最高的工具
        best_tool = max(tool_counts, key=tool_counts.get)
        return tool_suggestions[best_tool][:1]

    async def _select_by_llm(
        self, suggestions: List[SuggestionV2], chunk_idx: int
    ) -> List[SuggestionV2]:
        """使用 LLM 智能选择"""
        if not self.llm or len(suggestions) <= 1:
            return suggestions[:1]

        # 准备建议描述
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

            # 解析索引
            idx_str = response.content.strip()
            idx = int(idx_str)
            if 0 <= idx < len(suggestions):
                return [suggestions[idx]]
        except Exception as e:
            logger.warning(f"LLM selection failed: {e}")

        return suggestions[:1]

    def get_diagnosis_summary(self) -> Dict[str, int]:
        """获取该分区的诊断摘要"""
        return self.orchestra_group.diagnosis_summary

    def get_dominant_failure(self) -> Optional[str]:
        """获取最常见的失败类型"""
        return self.orchestra_group.get_dominant_failure()


async def create_orchestras_from_chunks(
    chunks: List[ContentChunk],
    chunks_per_orchestra: int = 2,
    history_context: str = "",
    config_path: str = "geo_agent/config.yaml",
    llm = None,
) -> List[SegmentOrchestraV2]:
    """
    从 chunks 创建 orchestras

    Args:
        chunks: ContentChunk 列表
        chunks_per_orchestra: 每个 orchestra 的 chunk 数量
        history_context: 历史上下文
        config_path: 配置路径
        llm: LLM 实例

    Returns:
        List[SegmentOrchestraV2]: orchestra 列表
    """
    orchestras = []
    n = len(chunks)

    for i in range(0, n, chunks_per_orchestra):
        indices = list(range(i, min(i + chunks_per_orchestra, n)))

        # 收集内容
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
