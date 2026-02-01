"""
Batch GEO V2 内存管理器
统一管理优化历史、跨 Batch 历史和策略注入
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    OptimizationResultV2,
    DiagnosisInfo,
    HistoryEntryV2,
    SuggestionV2,
)

logger = logging.getLogger(__name__)


class ModificationRecordV2:
    """V2 修改记录"""

    def __init__(
        self,
        query: str,
        tool_name: str,
        reasoning: str = "",
        key_changes: Optional[List[str]] = None,
        diagnosis: Optional[DiagnosisInfo] = None,
        chunk_index: Optional[int] = None,
    ):
        self.query = query
        self.tool_name = tool_name
        self.reasoning = reasoning
        self.key_changes = key_changes or []
        self.diagnosis = diagnosis
        self.chunk_index = chunk_index
        self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "tool_name": self.tool_name,
            "reasoning": self.reasoning,
            "key_changes": self.key_changes,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "chunk_index": self.chunk_index,
            "timestamp": self.timestamp,
        }


class OptimizationMemoryV2:
    """
    V2 优化记忆模块

    扩展功能：
    - 支持诊断信息记录
    - 支持 chunk 级别的修改追踪
    - 支持策略注入生成
    """

    def __init__(self, core_idea: str = ""):
        self.core_idea = core_idea
        self.modifications: List[ModificationRecordV2] = []
        # 按 chunk 索引组织的修改历史
        self.chunk_history: Dict[int, List[ModificationRecordV2]] = {}
        # 诊断统计
        self.diagnosis_stats: Dict[str, int] = {}

    def add_modification(self, record: ModificationRecordV2) -> None:
        """添加修改记录"""
        self.modifications.append(record)

        # 更新 chunk 历史
        if record.chunk_index is not None:
            if record.chunk_index not in self.chunk_history:
                self.chunk_history[record.chunk_index] = []
            self.chunk_history[record.chunk_index].append(record)

        # 更新诊断统计
        if record.diagnosis:
            cause = record.diagnosis.root_cause
            self.diagnosis_stats[cause] = self.diagnosis_stats.get(cause, 0) + 1

    def get_history_summary(self) -> str:
        """生成历史修改摘要"""
        if not self.modifications:
            return "No previous modifications."

        summary_parts = []
        for i, mod in enumerate(self.modifications, 1):
            changes = "\n".join(f"    - {c}" for c in mod.key_changes)
            diagnosis_info = ""
            if mod.diagnosis:
                diagnosis_info = f"\n  Diagnosis: {mod.diagnosis.root_cause} ({mod.diagnosis.severity})"

            summary_parts.append(
                f"[Modification {i}] Query: {mod.query}\n"
                f"  Tool: {mod.tool_name}\n"
                f"  Chunk: {mod.chunk_index}{diagnosis_info}\n"
                f"  Reason: {mod.reasoning}\n"
                f"  Key Changes:\n{changes}"
            )
        return "\n\n".join(summary_parts)

    def get_preservation_rules(self) -> str:
        """生成保护之前修改的规则"""
        if not self.modifications:
            return ""

        all_changes = []
        for mod in self.modifications:
            all_changes.extend(mod.key_changes)

        if not all_changes:
            return ""

        rules = "PREVIOUS MODIFICATIONS (MUST PRESERVE):\n"
        for change in all_changes:
            rules += f"- {change}\n"
        rules += "\nDO NOT remove, contradict, or undo any of these changes."
        return rules

    def get_chunk_history(self, chunk_index: int) -> List[ModificationRecordV2]:
        """获取特定 chunk 的修改历史"""
        return self.chunk_history.get(chunk_index, [])

    def get_failed_tools_for_diagnosis(self, diagnosis_cause: str) -> List[str]:
        """获取对某种诊断类型已经尝试过的工具"""
        tools = []
        for mod in self.modifications:
            if mod.diagnosis and mod.diagnosis.root_cause == diagnosis_cause:
                tools.append(mod.tool_name)
        return tools

    def to_dict(self) -> Dict:
        """序列化"""
        return {
            "core_idea": self.core_idea,
            "modifications": [m.to_dict() for m in self.modifications],
            "diagnosis_stats": self.diagnosis_stats,
        }


class HistoryManagerV2:
    """
    V2 跨 Batch 历史管理器

    扩展功能：
    - 诊断信息持久化
    - 策略注入生成
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self.persist_path = persist_path
        self.entries: List[HistoryEntryV2] = []
        self.batch_results: List[OptimizationResultV2] = []
        # 加载历史
        if persist_path and persist_path.exists():
            self._load()

    def _load(self) -> None:
        """从文件加载历史"""
        if not self.persist_path or not self.persist_path.exists():
            return

        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry_data in data.get("entries", []):
                diagnosis = None
                if entry_data.get("diagnosis"):
                    diagnosis = DiagnosisInfo(**entry_data["diagnosis"])

                self.entries.append(
                    HistoryEntryV2(
                        batch_id=entry_data["batch_id"],
                        orchestra_id=entry_data["orchestra_id"],
                        segment_index=entry_data["segment_index"],
                        tool_name=entry_data["tool_name"],
                        key_changes=entry_data["key_changes"],
                        applied_at=entry_data["applied_at"],
                        content_snapshot=entry_data.get("content_snapshot", ""),
                        diagnosis=diagnosis,
                    )
                )
        except Exception as e:
            logger.warning(f"Failed to load history: {e}")

    def _save(self) -> None:
        """保存历史到文件"""
        if not self.persist_path:
            return

        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "entries": [
                    {
                        "batch_id": e.batch_id,
                        "orchestra_id": e.orchestra_id,
                        "segment_index": e.segment_index,
                        "tool_name": e.tool_name,
                        "key_changes": e.key_changes,
                        "applied_at": e.applied_at,
                        "content_snapshot": e.content_snapshot,
                        "diagnosis": e.diagnosis.to_dict() if e.diagnosis else None,
                    }
                    for e in self.entries
                ]
            }
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save history: {e}")

    def add_entry(self, entry: HistoryEntryV2) -> None:
        """添加历史条目"""
        self.entries.append(entry)
        self._save()

    def add_batch_result(self, result: OptimizationResultV2) -> None:
        """从 BatchResult 添加历史条目"""
        self.batch_results.append(result)

        for suggestion in result.all_suggestions:
            if suggestion.suggestion_id in result.applied_modifications:
                entry = HistoryEntryV2(
                    batch_id=result.batch_id,
                    orchestra_id=0,  # 需要从外部传入
                    segment_index=suggestion.target_segment_index,
                    tool_name=suggestion.tool_name,
                    key_changes=suggestion.key_changes,
                    applied_at=suggestion.timestamp,
                    diagnosis=suggestion.diagnosis,
                )
                self.entries.append(entry)

        self._save()

    def get_preservation_rules(self) -> str:
        """生成保护规则"""
        if not self.entries:
            return ""

        all_changes = []
        for entry in self.entries:
            all_changes.extend(entry.key_changes)

        if not all_changes:
            return ""

        rules = "CROSS-BATCH HISTORY (MUST PRESERVE):\n"
        for change in all_changes[-20:]:  # 最近 20 条
            rules += f"- {change}\n"
        return rules

    def get_segment_history(self, segment_index: int) -> List[HistoryEntryV2]:
        """获取特定段落的历史"""
        return [e for e in self.entries if e.segment_index == segment_index]

    def get_diagnosis_stats(self) -> Dict[str, int]:
        """获取诊断统计"""
        stats: Dict[str, int] = {}
        for entry in self.entries:
            if entry.diagnosis:
                cause = entry.diagnosis.root_cause
                stats[cause] = stats.get(cause, 0) + 1
        return stats


class PolicyEngine:
    """
    策略引擎

    基于历史诊断结果生成策略注入

    消融实验支持：
    - enable_memory=False: 跳过单 query 修改历史相关的策略
    - enable_history=False: 跳过跨 batch 历史相关的策略
    """

    def __init__(
        self,
        history_manager: HistoryManagerV2,
        memory: OptimizationMemoryV2,
        enable_memory: bool = True,
        enable_history: bool = True,
    ):
        self.history_manager = history_manager
        self.memory = memory
        self.enable_memory = enable_memory
        self.enable_history = enable_history

    def generate_policy_injection(
        self,
        current_diagnosis: Optional[DiagnosisInfo] = None,
        current_chunk_index: Optional[int] = None,
    ) -> str:
        """
        生成策略注入

        基于：
        - 当前诊断
        - 历史诊断统计
        - 已尝试的工具

        消融实验：
        - enable_memory=False: 跳过基于 memory 的规则（tried_tools 检查）
        - enable_history=False: 跳过基于 history_manager 的规则
        """
        policy_parts = []

        # 1. 基于当前诊断的规则（需要 enable_memory）
        if current_diagnosis:
            cause = current_diagnosis.root_cause

            # 检查是否已经尝试过某些工具（仅当 enable_memory=True 时）
            if self.enable_memory:
                tried_tools = self.memory.get_failed_tools_for_diagnosis(cause)
                if tried_tools:
                    tools_str = ", ".join(tried_tools)
                    policy_parts.append(
                        f"### AVOID REPEATED TOOLS:\n"
                        f"For diagnosis '{cause}', these tools have been tried: {tools_str}.\n"
                        f"Try a DIFFERENT tool or approach."
                    )

            # 基于诊断类型的特定规则（不依赖 memory 或 history）
            diagnosis_rules = self._get_diagnosis_specific_rules(cause)
            if diagnosis_rules:
                policy_parts.append(diagnosis_rules)

        # 2. 基于历史的全局规则（仅当 enable_history=True 时）
        if self.enable_history:
            diagnosis_stats = self.history_manager.get_diagnosis_stats()
            if diagnosis_stats:
                # 找出最常见的失败类型
                most_common = max(diagnosis_stats, key=diagnosis_stats.get)
                count = diagnosis_stats[most_common]
                if count >= 3:
                    policy_parts.append(
                        f"### RECURRING ISSUE:\n"
                        f"'{most_common}' has occurred {count} times. "
                        f"Consider a more aggressive strategy for this issue type."
                    )

        # 3. 段落特定历史（仅当 enable_history=True 时）
        if self.enable_history and current_chunk_index is not None:
            segment_history = self.history_manager.get_segment_history(current_chunk_index)
            if len(segment_history) >= 2:
                recent_tools = [h.tool_name for h in segment_history[-3:]]
                policy_parts.append(
                    f"### SEGMENT HISTORY:\n"
                    f"Chunk {current_chunk_index} has been modified {len(segment_history)} times.\n"
                    f"Recent tools used: {', '.join(recent_tools)}.\n"
                    f"Consider if further modifications are necessary."
                )

        if not policy_parts:
            return ""

        return "\n\n".join(policy_parts)

    def _get_diagnosis_specific_rules(self, diagnosis_cause: str) -> str:
        """获取诊断特定规则"""
        rules = {
            "PARSING_FAILURE": (
                "### PARSING_FAILURE POLICY:\n"
                "The document contains JS code, JSON data, or dynamic content that wasn't rendered.\n"
                "Use 'static_rendering' to extract readable content from JS/JSON embedded data.\n"
                "Look for __NEXT_DATA__, application/json scripts, or window.* data objects."
            ),
            "CONTENT_TRUNCATED": (
                "### CONTENT_TRUNCATED POLICY:\n"
                "Use 'content_relocation' to surface hidden content.\n"
                "Move the most relevant information to the beginning of the document."
            ),
            "MISSING_INFO": (
                "### MISSING_INFO POLICY:\n"
                "Use 'entity_injection' to add specific facts, entities, or data points.\n"
                "If entity_injection was already tried, switch to 'persuasive_rewriting' or 'historical_redteam'."
            ),
            "STRUCTURAL_WEAKNESS": (
                "### STRUCTURAL_WEAKNESS POLICY:\n"
                "Use 'structure_optimization' to improve document organization.\n"
                "Alternatively, use 'intent_realignment' to reframe content for the query."
            ),
            "BURIED_ANSWER": (
                "### BURIED_ANSWER POLICY:\n"
                "Use 'content_relocation' to move relevant answers to prominent positions.\n"
                "Focus on making key information immediately visible."
            ),
            "LOW_INFO_DENSITY": (
                "### LOW_INFO_DENSITY POLICY:\n"
                "Use 'entity_injection' to add specific, actionable information.\n"
                "Focus on concrete facts rather than general statements."
            ),
            "WEB_NOISE": (
                "### WEB_NOISE POLICY:\n"
                "Use 'noise_isolation' to remove boilerplate and irrelevant content.\n"
                "Focus on retaining only substantive information."
            ),
            "SEMANTIC_IRRELEVANCE": (
                "### SEMANTIC_IRRELEVANCE POLICY:\n"
                "Use 'intent_realignment' to reframe content for query relevance.\n"
                "This may require significant content restructuring."
            ),
        }

        return rules.get(diagnosis_cause, "")
