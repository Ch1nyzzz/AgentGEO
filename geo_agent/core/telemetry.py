"""
Telemetry Store for GEO Agent
Collect, store, and query structured signals from each optimization loop
Design inspired by OpenTelemetry's Span/Metric model
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import json


class FailureCategory(Enum):
    """
    Failure root cause classification - summarized from large-scale document analysis
    Divided into three main categories: technical issues, content issues, relevance issues
    """
    # === Technical/Parsing Issues ===
    PARSING_FAILURE = "parsing_failure"  # Technically unreadable, parsing/rendering/formatting failure
    CONTENT_TRUNCATED = "content_truncated"  # Incomplete content, truncated, partially missing
    DATA_INTEGRITY = "data_integrity"  # Data integrity issues, crawling/extraction failure

    # === Noise Issues ===
    WEB_NOISE = "web_noise"  # Mainly non-informational web noise, boilerplate, navigation elements
    LOW_SIGNAL_RATIO = "low_signal_ratio"  # Low signal-to-noise ratio, poor semantic quality

    # === Information Density Issues ===
    LOW_INFO_DENSITY = "low_info_density"  # Insufficient information density, sparse or anecdotal content
    MISSING_INFO = "missing_info"  # Missing key information (legacy classification, kept for compatibility)

    # === Structure Issues ===
    STRUCTURAL_WEAKNESS = "structural_weakness"  # Poor structure, lacking segmentation and logical flow

    # === Semantic Relevance Issues ===
    SEMANTIC_IRRELEVANCE = "semantic_irrelevance"  # Semantically irrelevant, off-topic, mismatched with query
    ATTRIBUTE_MISMATCH = "attribute_mismatch"  # Correct entity but mismatched attributes/metrics

    # === Answer Positioning Issues ===
    BURIED_ANSWER = "buried_answer"  # Answer buried deep in verbose content, not directly presented

    # === Content Quality Issues ===
    NON_FACTUAL_CONTENT = "non_factual_content"  # Contains only questions, speculation, opinions, marketing content
    TRUST_CREDIBILITY = "trust_credibility"  # Insufficient authority/credibility

    # === Temporal Issues ===
    OUTDATED_CONTENT = "outdated_content"  # Content is outdated, temporal mismatch

    # === Other ===
    UNKNOWN = "unknown"


class ToolOutcome(Enum):
    """Tool execution result"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    SKIPPED = "skipped"


@dataclass
class ToolInvocationSpan:
    """Span record for a single tool invocation"""
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
    """Metrics collection for a single iteration"""
    iteration_index: int
    query: str

    # Document metrics
    full_doc_length: int
    visible_chunk_length: int
    truncation_ratio: float  # Truncation ratio = 1 - (visible/full)
    chunk_count: int

    # Diagnosis results
    diagnosis_category: FailureCategory
    diagnosis_explanation: str

    # Truncation audit
    has_hidden_relevant_content: bool = False
    hidden_content_summary: str = ""

    # Tool invocation
    tool_span: Optional[ToolInvocationSpan] = None

    # Result
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
    Telemetry Store
    - Collect metrics for each iteration
    - Provide query interface for Policy layer
    """
    
    def __init__(self, url: str = "", core_idea: str = ""):
        self.url = url
        self.core_idea = core_idea
        self.iterations: List[IterationMetrics] = []
        self._tool_invocation_history: List[ToolInvocationSpan] = []
    
    def record_iteration(self, metrics: IterationMetrics) -> None:
        """Record one iteration"""
        self.iterations.append(metrics)
        if metrics.tool_span:
            self._tool_invocation_history.append(metrics.tool_span)
    
    # ========== Query Interface ==========

    def get_tool_usage_count(self, tool_name: str) -> int:
        """Get the number of times a tool has been called"""
        return sum(1 for span in self._tool_invocation_history if span.tool_name == tool_name)
    
    def get_failed_tool_invocations(self) -> List[ToolInvocationSpan]:
        """Get all failed tool invocations"""
        return [span for span in self._tool_invocation_history if span.outcome == ToolOutcome.FAILED]
    
    def get_recent_tools(self, n: int = 3) -> List[str]:
        """Get the names of the last n tools called"""
        return [span.tool_name for span in self._tool_invocation_history[-n:]]
    
    def has_repeated_tool_args(self, tool_name: str, args_hash: str) -> bool:
        """Detect if there are duplicate calls with the same tool and arguments"""
        for span in self._tool_invocation_history:
            if span.tool_name == tool_name and span.arguments_hash == args_hash:
                return True
        return False
    
    def get_diagnosis_history(self) -> List[FailureCategory]:
        """Get diagnosis history"""
        return [it.diagnosis_category for it in self.iterations]
    
    def get_truncation_alerts_count(self) -> int:
        """Get the count of truncation alerts"""
        return sum(1 for it in self.iterations if it.has_hidden_relevant_content)
    
    def get_average_truncation_ratio(self) -> float:
        """Get average truncation ratio"""
        if not self.iterations:
            return 0.0
        return sum(it.truncation_ratio for it in self.iterations) / len(self.iterations)
    
    # ========== Export ==========
    
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
        """Tool usage statistics"""
        usage = {}
        for span in self._tool_invocation_history:
            usage[span.tool_name] = usage.get(span.tool_name, 0) + 1
        return usage
    
    def export_json(self, path: str) -> None:
        """Export to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def compute_args_hash(tool_args: Dict[str, Any], exclude_keys: List[str] = None) -> str:
    """
    Compute hash of tool arguments for duplicate detection
    Excludes dynamic fields like previous_modifications
    """
    import hashlib

    exclude_keys = exclude_keys or ["previous_modifications", "core_idea"]
    filtered_args = {k: v for k, v in tool_args.items() if k not in exclude_keys}

    # Only take the first 500 characters of target_content
    if "target_content" in filtered_args:
        filtered_args["target_content"] = filtered_args["target_content"][:500]

    arg_str = json.dumps(filtered_args, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(arg_str.encode()).hexdigest()[:12]
