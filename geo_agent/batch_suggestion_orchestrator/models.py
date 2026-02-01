"""
AgentGEO V2 数据模型
扩展的数据结构，包含诊断结果（DiagnosisResult）支持
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# 诊断结果类型（与 geo_agent 保持一致）
FAILURE_CATEGORY_LITERAL = Literal[
    "PARSING_FAILURE",
    "CONTENT_TRUNCATED",
    "DATA_INTEGRITY",
    "WEB_NOISE",
    "LOW_SIGNAL_RATIO",
    "LOW_INFO_DENSITY",
    "MISSING_INFO",
    "STRUCTURAL_WEAKNESS",
    "SEMANTIC_IRRELEVANCE",
    "ATTRIBUTE_MISMATCH",
    "BURIED_ANSWER",
    "NON_FACTUAL_CONTENT",
    "TRUST_CREDIBILITY",
    "OUTDATED_CONTENT",
    "UNKNOWN",
]

SEVERITY_LITERAL = Literal["low", "medium", "high", "critical"]


@dataclass
class DiagnosisInfo:
    """诊断信息（从 DiagnosisResult 提取的关键字段）"""

    root_cause: str
    explanation: str
    key_deficiency: str
    severity: str = "medium"

    def to_dict(self) -> Dict[str, str]:
        return {
            "root_cause": self.root_cause,
            "explanation": self.explanation,
            "key_deficiency": self.key_deficiency,
            "severity": self.severity,
        }


@dataclass
class SuggestionV2:
    """
    扩展的修改建议，包含诊断信息

    相比 V1 版本新增：
    - diagnosis: 诊断信息（root_cause, key_deficiency 等）
    - iteration: 迭代次数（用于跟踪重试）
    - executed_arguments: 实际执行的参数（脱敏后）
    - truncation_info: 截断审计信息
    """

    suggestion_id: str
    query: str
    tool_name: str
    tool_arguments: Dict[str, Any]  # LLM 原始输出
    target_segment_index: int
    reasoning: str
    proposed_content: str
    key_changes: List[str]
    # V2 新增字段
    diagnosis: Optional[DiagnosisInfo] = None
    iteration: int = 0
    confidence: float = 0.8
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # V2.2 新增：实际执行参数和截断信息（与 geo_agent 对齐）
    executed_arguments: Optional[Dict[str, Any]] = None  # 实际执行参数（脱敏）
    truncation_info: Optional[Dict[str, Any]] = None     # 截断审计信息

    def conflicts_with(self, other: "SuggestionV2") -> bool:
        """检查两个建议是否可能冲突（同一段落）"""
        return self.target_segment_index == other.target_segment_index

    def get_priority_score(self) -> float:
        """
        计算优先级分数（用于建议排序）

        基于：
        - 诊断严重程度 (severity)
        - 置信度 (confidence)
        - 迭代次数（越早的建议优先级越低，因为可能已被尝试）
        """
        severity_scores = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
        severity_score = 0.5
        if self.diagnosis:
            severity_score = severity_scores.get(self.diagnosis.severity, 0.5)

        # 迭代惩罚：每次迭代降低 10% 优先级
        iteration_penalty = max(0.5, 1.0 - self.iteration * 0.1)

        return severity_score * self.confidence * iteration_penalty


@dataclass
class OrchestraGroupV2:
    """
    V2 动态分区：每个 Orchestra 负责固定数量的 chunk

    相比 V1 新增：
    - diagnosis_summary: 该分区的诊断摘要（用于智能合并）
    """

    orchestra_id: int
    segment_indices: List[int]
    core_idea: str = ""
    original_content: str = ""
    original_html: str = ""
    current_content: str = ""
    current_html: str = ""
    suggestions: List[SuggestionV2] = field(default_factory=list)
    applied_suggestions: List[str] = field(default_factory=list)
    # V2 新增：诊断摘要
    diagnosis_summary: Dict[str, int] = field(default_factory=dict)

    def add_suggestion(self, suggestion: SuggestionV2) -> None:
        """添加建议并更新诊断摘要"""
        self.suggestions.append(suggestion)
        if suggestion.diagnosis:
            cause = suggestion.diagnosis.root_cause
            self.diagnosis_summary[cause] = self.diagnosis_summary.get(cause, 0) + 1

    def get_dominant_failure(self) -> Optional[str]:
        """获取最常见的失败类型"""
        if not self.diagnosis_summary:
            return None
        return max(self.diagnosis_summary, key=self.diagnosis_summary.get)


@dataclass
class QueryResultV2:
    """
    V2 单个 Query 的处理结果

    相比 V1 新增：
    - diagnosis: 诊断结果
    - iterations_used: 实际使用的迭代次数
    - final_html: 优化后的 HTML（用于汇总）
    - GEO Score 相关字段（V2.3 新增）
    """

    query: str
    is_cited: bool
    generated_answer: str
    suggestions: List[SuggestionV2]
    # V2 新增字段
    diagnosis: Optional[DiagnosisInfo] = None
    iterations_used: int = 0
    error: Optional[str] = None
    # V2.1 新增：存储优化后的 HTML 用于汇总
    final_html: Optional[str] = None
    # V2.3 新增：GEO Score 相关字段
    geo_score_word: float = 0.0
    geo_score_position: float = 0.0
    geo_score_wordpos: float = 0.0
    geo_score_overall: float = 0.0
    has_valid_citations: bool = False


@dataclass
class AppliedToolInfo:
    """记录应用的工具信息"""
    suggestion_id: str
    tool_name: str
    query: str
    target_segment_index: int
    diagnosis_type: Optional[str] = None  # root_cause
    reasoning: str = ""
    key_changes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "tool_name": self.tool_name,
            "query": self.query,
            "target_segment_index": self.target_segment_index,
            "diagnosis_type": self.diagnosis_type,
            "reasoning": self.reasoning,
            "key_changes": self.key_changes,
        }

    @classmethod
    def from_suggestion(cls, suggestion: "SuggestionV2") -> "AppliedToolInfo":
        """从 SuggestionV2 创建"""
        return cls(
            suggestion_id=suggestion.suggestion_id,
            tool_name=suggestion.tool_name,
            query=suggestion.query,
            target_segment_index=suggestion.target_segment_index,
            diagnosis_type=suggestion.diagnosis.root_cause if suggestion.diagnosis else None,
            reasoning=suggestion.reasoning,
            key_changes=suggestion.key_changes,
        )


@dataclass
class OptimizationResultV2:
    """V2 优化结果（原 BatchResultV2）"""

    batch_id: str
    queries: List[str]
    query_results: List[QueryResultV2]
    all_suggestions: List[SuggestionV2]
    applied_modifications: List[str]
    content_before: str
    content_after: str
    success_rate_before: float = 0.0
    success_rate_after: float = 0.0
    # V2 新增：诊断统计
    diagnosis_stats: Dict[str, int] = field(default_factory=dict)
    # V2.4 新增：应用的工具详细信息
    applied_tools: List[AppliedToolInfo] = field(default_factory=list)
    # HTML 输出（可选）
    html_after: str = ""
    # V2.3 新增：GEO Score 统计
    avg_geo_score_word: float = 0.0
    avg_geo_score_position: float = 0.0
    avg_geo_score_wordpos: float = 0.0
    avg_geo_score_overall: float = 0.0
    valid_citation_rate: float = 0.0

    def compute_diagnosis_stats(self) -> None:
        """计算诊断统计"""
        self.diagnosis_stats = {}
        for suggestion in self.all_suggestions:
            if suggestion.diagnosis:
                cause = suggestion.diagnosis.root_cause
                self.diagnosis_stats[cause] = self.diagnosis_stats.get(cause, 0) + 1

    def compute_geo_score_stats(self) -> None:
        """计算 GEO Score 统计"""
        if not self.query_results:
            return

        total_word = 0.0
        total_position = 0.0
        total_wordpos = 0.0
        total_overall = 0.0
        valid_count = 0

        for qr in self.query_results:
            total_word += qr.geo_score_word
            total_position += qr.geo_score_position
            total_wordpos += qr.geo_score_wordpos
            total_overall += qr.geo_score_overall
            if qr.has_valid_citations:
                valid_count += 1

        n = len(self.query_results)
        self.avg_geo_score_word = total_word / n
        self.avg_geo_score_position = total_position / n
        self.avg_geo_score_wordpos = total_wordpos / n
        self.avg_geo_score_overall = total_overall / n
        self.valid_citation_rate = valid_count / n


@dataclass
class HistoryEntryV2:
    """V2 历史修改条目"""

    batch_id: str
    orchestra_id: int
    segment_index: int
    tool_name: str
    key_changes: List[str]
    applied_at: str
    content_snapshot: str = ""
    # V2 新增：诊断信息
    diagnosis: Optional[DiagnosisInfo] = None


class AgentGEOConfigV2(BaseModel):
    """V2 AgentGEO 配置（原 BatchConfigV2）"""

    batch_size: int = Field(10, description="每个 batch 处理的 query 数量")
    max_concurrency: int = Field(4, description="并发数")
    max_retries_per_query: int = Field(3, description="每个 query 的最大重试次数")
    chunks_per_orchestra: int = Field(2, description="每个 orchestra 负责的 chunk 数")
    suggestion_merge_strategy: str = Field(
        "diagnosis_aware", description="建议合并策略: vote, llm_merge, diagnosis_aware"
    )
    max_suggestions_per_segment: int = Field(5, description="每个段落最多保留的建议数")
    enable_cross_batch_history: bool = Field(True, description="是否启用跨 batch 历史")
    enable_policy_injection: bool = Field(True, description="是否启用策略注入")
    history_persist_path: Optional[str] = Field(None, description="历史持久化路径")
    # V2 新增配置
    use_two_phase_analysis: bool = Field(True, description="是否使用两阶段分析（诊断 + 策略选择）")
    diagnosis_cache_enabled: bool = Field(True, description="是否缓存诊断结果")
    # 消融实验配置
    enable_memory: bool = Field(
        True,
        description="是否启用 Memory 模块（单 query 修改历史追踪）"
    )
    enable_history: bool = Field(
        True,
        description="是否启用 Historical 模块（跨 batch 历史追踪）"
    )

    # 引用检查配置（V2.2 新增）
    citation_method: str = Field(
        "llm",
        description="引用检查方式: llm, attr_evaluator, both"
    )
    citation_composite_strategy: str = Field(
        "any",
        description="复合模式策略: any, all, llm_primary, attr_primary"
    )
    attr_evaluator_config: Optional[str] = Field(
        None,
        description="Attribute Evaluator 配置文件路径"
    )
    use_fast_mode: bool = Field(
        True,
        description="是否使用快速模式的 Attribute Evaluator"
    )

    # 工具配置（V2.4 新增）
    enable_autogeo_rephrase: bool = Field(
        True,
        description="是否启用 autogeo_rephrase 工具（AutoGEO 论文方法）。"
                    "设为 False 时只使用 BatchGEO 自有工具。"
    )


class MultiDocConfigV2(BaseModel):
    """
    多文档并行处理配置

    用于控制文档级别的并发处理，与 AgentGEOConfigV2 配合使用。
    AgentGEOConfigV2 控制单文档内 query 的并发，MultiDocConfigV2 控制多文档间的并发。
    """

    max_doc_concurrency: int = Field(
        2, ge=1, le=10, description="同时处理的文档数（建议 2-4）"
    )
    batch_config: AgentGEOConfigV2 = Field(
        default_factory=AgentGEOConfigV2, description="单文档处理配置"
    )
    share_search_cache: bool = Field(True, description="跨文档共享搜索缓存")
    fail_fast: bool = Field(False, description="单文档失败是否中止全部处理")
    max_retries_per_doc: int = Field(1, ge=0, le=3, description="文档级重试次数")
    progress_interval_ms: int = Field(1000, description="进度回调最小间隔（毫秒）")


@dataclass
class DocumentOptimizationResult:
    """
    单个文档的优化结果

    封装 WebPage 的优化过程和结果，包括成功/失败状态、耗时等元信息。
    """

    webpage: "WebPage"  # 优化后的 WebPage
    batch_results: List[OptimizationResultV2]  # 该文档的所有批次结果
    success: bool  # 是否成功
    error: Optional[str] = None  # 错误信息（如果失败）
    duration_ms: float = 0.0  # 处理耗时（毫秒）
    queries_count: int = 0  # 处理的 query 数量
    suggestions_applied: int = 0  # 应用的建议数量

    def get_success_rate_improvement(self) -> float:
        """计算成功率提升"""
        if not self.batch_results:
            return 0.0
        first = self.batch_results[0].success_rate_before
        last = self.batch_results[-1].success_rate_after if self.batch_results[-1].success_rate_after else first
        return last - first


@dataclass
class MultiDocOptimizationResult:
    """
    多文档优化的汇总结果

    包含所有文档的优化结果和整体统计信息。
    """

    results: List[DocumentOptimizationResult]  # 各文档结果
    total_duration_ms: float  # 总耗时
    success_count: int = 0  # 成功文档数
    failure_count: int = 0  # 失败文档数

    def __post_init__(self):
        """自动计算成功/失败数量"""
        if self.success_count == 0 and self.failure_count == 0:
            self.success_count = sum(1 for r in self.results if r.success)
            self.failure_count = len(self.results) - self.success_count

    @property
    def total_count(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.success_count / len(self.results)

    def get_overall_diagnosis_stats(self) -> Dict[str, int]:
        """获取所有文档的诊断统计"""
        stats: Dict[str, int] = {}
        for doc_result in self.results:
            for batch_result in doc_result.batch_results:
                for cause, count in batch_result.diagnosis_stats.items():
                    stats[cause] = stats.get(cause, 0) + count
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """获取汇总信息"""
        return {
            "total_documents": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_per_doc_ms": self.total_duration_ms / self.total_count if self.total_count else 0,
            "diagnosis_stats": self.get_overall_diagnosis_stats(),
        }
