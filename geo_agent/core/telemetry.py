"""
Telemetry Store for GEO Agent
采集、存储、查询每次优化循环中的结构化信号
设计灵感来自 OpenTelemetry 的 Span/Metric 模型
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class FailureCategory(Enum):
    """
    失败根因分类 - 基于大规模文档分析总结
    分为技术问题、内容问题、相关性问题三大类
    """
    # === 技术/解析问题 (Technical Issues) ===
    PARSING_FAILURE = "parsing_failure"  # 技术上不可读，解析/渲染/格式化失败
    CONTENT_TRUNCATED = "content_truncated"  # 内容不完整、截断、部分缺失
    DATA_INTEGRITY = "data_integrity"  # 数据完整性问题，爬取/提取失败
    
    # === 内容噪音问题 (Noise Issues) ===
    WEB_NOISE = "web_noise"  # 主要是非信息性网页噪音、样板、导航元素
    LOW_SIGNAL_RATIO = "low_signal_ratio"  # 信噪比低，语义质量差
    
    # === 信息密度问题 (Density Issues) ===
    LOW_INFO_DENSITY = "low_info_density"  # 信息密度不足，内容稀疏、轶事性
    MISSING_INFO = "missing_info"  # 缺少关键信息（传统分类，保留兼容）
    
    # === 结构问题 (Structure Issues) ===
    STRUCTURAL_WEAKNESS = "structural_weakness"  # 结构不良，缺乏分段、逻辑流
    
    # === 语义相关性问题 (Relevance Issues) ===
    SEMANTIC_IRRELEVANCE = "semantic_irrelevance"  # 语义无关，离题，与查询不匹配
    ATTRIBUTE_MISMATCH = "attribute_mismatch"  # 实体正确但属性/指标不匹配
    
    # === 答案定位问题 (Answer Positioning) ===
    BURIED_ANSWER = "buried_answer"  # 答案埋在冗长内容深处，未直接呈现
    
    # === 内容质量问题 (Quality Issues) ===
    NON_FACTUAL_CONTENT = "non_factual_content"  # 仅含问题、猜测、意见、营销内容
    TRUST_CREDIBILITY = "trust_credibility"  # 权威性/可信度不足
    
    # === 时效性问题 (Temporal Issues) ===
    OUTDATED_CONTENT = "outdated_content"  # 内容过时，时间不匹配
    
    # === 其他 ===
    UNKNOWN = "unknown"


class ToolOutcome(Enum):
    """工具执行结果"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class ToolInvocationSpan:
    """单次工具调用的 Span 记录"""
    tool_name: str
    target_chunk_index: int
    arguments_hash: str  # 参数摘要的哈希，用于去重检测
    outcome: ToolOutcome
    reasoning: str
    duration_ms: Optional[float] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "tool_name": self.tool_name,
            "target_chunk_index": self.target_chunk_index,
            "arguments_hash": self.arguments_hash,
            "outcome": self.outcome.value,
            "reasoning": self.reasoning,
            "duration_ms": self.duration_ms,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


@dataclass
class IterationMetrics:
    """单次迭代的指标采集"""
    iteration_index: int
    query: str
    
    # 文档指标
    full_doc_length: int
    visible_chunk_length: int
    truncation_ratio: float  # 截断比例 = 1 - (visible/full)
    chunk_count: int
    
    # 诊断结果
    diagnosis_category: FailureCategory
    diagnosis_explanation: str
    
    # 截断审计
    has_hidden_relevant_content: bool = False
    hidden_content_summary: str = ""
    
    # 工具调用
    tool_span: Optional[ToolInvocationSpan] = None
    
    # 结果
    was_cited: bool = False
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        return {
            "iteration_index": self.iteration_index,
            "query": self.query,
            "full_doc_length": self.full_doc_length,
            "visible_chunk_length": self.visible_chunk_length,
            "truncation_ratio": self.truncation_ratio,
            "chunk_count": self.chunk_count,
            "diagnosis_category": self.diagnosis_category.value,
            "diagnosis_explanation": self.diagnosis_explanation,
            "has_hidden_relevant_content": self.has_hidden_relevant_content,
            "hidden_content_summary": self.hidden_content_summary,
            "tool_span": self.tool_span.to_dict() if self.tool_span else None,
            "was_cited": self.was_cited,
            "timestamp": self.timestamp
        }


class TelemetryStore:
    """
    Telemetry 存储器
    - 收集每次迭代的指标
    - 提供查询接口供 Policy 层使用
    """
    
    def __init__(self, url: str = "", core_idea: str = ""):
        self.url = url
        self.core_idea = core_idea
        self.iterations: List[IterationMetrics] = []
        self._tool_invocation_history: List[ToolInvocationSpan] = []
    
    def record_iteration(self, metrics: IterationMetrics) -> None:
        """记录一次迭代"""
        self.iterations.append(metrics)
        if metrics.tool_span:
            self._tool_invocation_history.append(metrics.tool_span)
    
    # ========== 查询接口 ==========
    
    def get_tool_usage_count(self, tool_name: str) -> int:
        """获取某工具被调用的次数"""
        return sum(1 for span in self._tool_invocation_history if span.tool_name == tool_name)
    
    def get_failed_tool_invocations(self) -> List[ToolInvocationSpan]:
        """获取所有失败的工具调用"""
        return [span for span in self._tool_invocation_history if span.outcome == ToolOutcome.FAILED]
    
    def get_recent_tools(self, n: int = 3) -> List[str]:
        """获取最近 n 次调用的工具名"""
        return [span.tool_name for span in self._tool_invocation_history[-n:]]
    
    def has_repeated_tool_args(self, tool_name: str, args_hash: str) -> bool:
        """检测是否有相同工具+参数的重复调用"""
        for span in self._tool_invocation_history:
            if span.tool_name == tool_name and span.arguments_hash == args_hash:
                return True
        return False
    
    def get_diagnosis_history(self) -> List[FailureCategory]:
        """获取诊断历史"""
        return [it.diagnosis_category for it in self.iterations]
    
    def get_truncation_alerts_count(self) -> int:
        """获取截断警报次数"""
        return sum(1 for it in self.iterations if it.has_hidden_relevant_content)
    
    def get_average_truncation_ratio(self) -> float:
        """获取平均截断比例"""
        if not self.iterations:
            return 0.0
        return sum(it.truncation_ratio for it in self.iterations) / len(self.iterations)
    
    # ========== 导出 ==========
    
    def to_dict(self) -> Dict:
        return {
            "url": self.url,
            "core_idea": self.core_idea,
            "total_iterations": len(self.iterations),
            "iterations": [it.to_dict() for it in self.iterations],
            "summary": {
                "avg_truncation_ratio": self.get_average_truncation_ratio(),
                "truncation_alerts": self.get_truncation_alerts_count(),
                "tool_usage": self._get_tool_usage_summary()
            }
        }
    
    def _get_tool_usage_summary(self) -> Dict[str, int]:
        """工具使用统计"""
        usage = {}
        for span in self._tool_invocation_history:
            usage[span.tool_name] = usage.get(span.tool_name, 0) + 1
        return usage
    
    def export_json(self, path: str) -> None:
        """导出为 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def compute_args_hash(tool_args: Dict[str, Any], exclude_keys: List[str] = None) -> str:
    """
    计算工具参数的哈希值，用于去重检测
    排除动态字段如 previous_modifications
    """
    import hashlib
    
    exclude_keys = exclude_keys or ["previous_modifications", "core_idea"]
    filtered_args = {k: v for k, v in tool_args.items() if k not in exclude_keys}
    
    # 只取 target_content 的前 500 字符
    if "target_content" in filtered_args:
        filtered_args["target_content"] = filtered_args["target_content"][:500]
    
    arg_str = json.dumps(filtered_args, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(arg_str.encode()).hexdigest()[:12]
