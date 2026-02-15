"""
AgentGEO V2 Data Models
Extended data structures with diagnostic result (DiagnosisResult) support
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# Diagnosis result types (consistent with geo_agent)
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
    """Diagnosis information (key fields extracted from DiagnosisResult)"""

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
    Extended modification suggestion with diagnostic information

    New fields compared to V1:
    - diagnosis: Diagnostic info (root_cause, key_deficiency, etc.)
    - iteration: Iteration count (for retry tracking)
    - executed_arguments: Actually executed parameters (sanitized)
    - truncation_info: Truncation audit information
    """

    suggestion_id: str
    query: str
    tool_name: str
    tool_arguments: Dict[str, Any]  # LLM raw output
    target_segment_index: int
    reasoning: str
    proposed_content: str
    key_changes: List[str]
    original_content: str = ""  # Original chunk content before modification
    # V2 new fields
    diagnosis: Optional[DiagnosisInfo] = None
    iteration: int = 0
    confidence: float = 0.8
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # V2.2 addition: Actually executed parameters and truncation info (aligned with geo_agent)
    executed_arguments: Optional[Dict[str, Any]] = None  # Actually executed parameters (sanitized)
    truncation_info: Optional[Dict[str, Any]] = None     # Truncation audit information

    def conflicts_with(self, other: "SuggestionV2") -> bool:
        """Check if two suggestions may conflict (same segment)"""
        return self.target_segment_index == other.target_segment_index

    def get_priority_score(self) -> float:
        """
        Calculate priority score (for suggestion sorting)

        Based on:
        - Diagnosis severity
        - Confidence
        - Iteration count (earlier suggestions have lower priority as they may have been tried)
        """
        severity_scores = {"critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.3}
        severity_score = 0.5
        if self.diagnosis:
            severity_score = severity_scores.get(self.diagnosis.severity, 0.5)

        # Iteration penalty: reduce priority by 10% per iteration
        iteration_penalty = max(0.5, 1.0 - self.iteration * 0.1)

        return severity_score * self.confidence * iteration_penalty


@dataclass
class OrchestraGroupV2:
    """
    V2 Dynamic partition: each Orchestra handles a fixed number of chunks

    New compared to V1:
    - diagnosis_summary: Diagnosis summary for this partition (for intelligent merging)
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
    # V2 addition: Diagnosis summary
    diagnosis_summary: Dict[str, int] = field(default_factory=dict)

    def add_suggestion(self, suggestion: SuggestionV2) -> None:
        """Add suggestion and update diagnosis summary"""
        self.suggestions.append(suggestion)
        if suggestion.diagnosis:
            cause = suggestion.diagnosis.root_cause
            self.diagnosis_summary[cause] = self.diagnosis_summary.get(cause, 0) + 1

    def get_dominant_failure(self) -> Optional[str]:
        """Get the most common failure type"""
        if not self.diagnosis_summary:
            return None
        return max(self.diagnosis_summary, key=self.diagnosis_summary.get)


@dataclass
class QueryResultV2:
    """
    V2 Single Query processing result

    New compared to V1:
    - diagnosis: Diagnostic result
    - iterations_used: Actual iterations used
    - final_html: Optimized HTML (for aggregation)
    - GEO Score related fields (V2.3 addition)
    """

    query: str
    is_cited: bool
    generated_answer: str
    suggestions: List[SuggestionV2]
    # V2 new fields
    diagnosis: Optional[DiagnosisInfo] = None
    iterations_used: int = 0
    error: Optional[str] = None
    # V2.1 addition: Store optimized HTML for aggregation
    final_html: Optional[str] = None
    # V2.3 addition: GEO Score related fields
    geo_score_word: float = 0.0
    geo_score_position: float = 0.0
    geo_score_wordpos: float = 0.0
    geo_score_overall: float = 0.0
    has_valid_citations: bool = False


@dataclass
class AppliedToolInfo:
    """Record applied tool information"""
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
        """Create from SuggestionV2"""
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
    """V2 Optimization result (formerly BatchResultV2)"""

    batch_id: str
    queries: List[str]
    query_results: List[QueryResultV2]
    all_suggestions: List[SuggestionV2]
    applied_modifications: List[str]
    content_before: str
    content_after: str
    success_rate_before: float = 0.0
    success_rate_after: float = 0.0
    # V2 addition: Diagnosis statistics
    diagnosis_stats: Dict[str, int] = field(default_factory=dict)
    # V2.4 addition: Applied tool details
    applied_tools: List[AppliedToolInfo] = field(default_factory=list)
    # HTML output (optional)
    html_after: str = ""
    # V2.3 addition: GEO Score statistics
    avg_geo_score_word: float = 0.0
    avg_geo_score_position: float = 0.0
    avg_geo_score_wordpos: float = 0.0
    avg_geo_score_overall: float = 0.0
    valid_citation_rate: float = 0.0

    def compute_diagnosis_stats(self) -> None:
        """Calculate diagnosis statistics"""
        self.diagnosis_stats = {}
        for suggestion in self.all_suggestions:
            if suggestion.diagnosis:
                cause = suggestion.diagnosis.root_cause
                self.diagnosis_stats[cause] = self.diagnosis_stats.get(cause, 0) + 1

    def compute_geo_score_stats(self) -> None:
        """Calculate GEO Score statistics"""
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
    """V2 History modification entry"""

    batch_id: str
    orchestra_id: int
    segment_index: int
    tool_name: str
    key_changes: List[str]
    applied_at: str
    content_snapshot: str = ""
    # V2 addition: Diagnostic information
    diagnosis: Optional[DiagnosisInfo] = None


class AgentGEOConfigV2(BaseModel):
    """V2 AgentGEO Configuration (formerly BatchConfigV2)"""

    batch_size: int = Field(10, description="Number of queries per batch")
    max_concurrency: int = Field(4, description="Concurrency level")
    max_retries_per_query: int = Field(3, description="Max retries per query")
    chunks_per_orchestra: int = Field(2, description="Chunks per orchestra")
    suggestion_merge_strategy: str = Field(
        "diagnosis_aware", description="Suggestion merge strategy: vote, llm_merge, diagnosis_aware"
    )
    max_suggestions_per_segment: int = Field(5, description="Max suggestions per segment")
    enable_cross_batch_history: bool = Field(True, description="Enable cross-batch history")
    enable_policy_injection: bool = Field(True, description="Enable policy injection")
    history_persist_path: Optional[str] = Field(None, description="History persistence path")
    # V2 new config
    use_two_phase_analysis: bool = Field(True, description="Use two-phase analysis (diagnosis + strategy selection)")
    diagnosis_cache_enabled: bool = Field(True, description="Cache diagnosis results")
    # Ablation experiment config
    enable_memory: bool = Field(
        True,
        description="Enable Memory module (single query modification history tracking)"
    )
    enable_history: bool = Field(
        True,
        description="Enable Historical module (cross-batch history tracking)"
    )

    # Citation check config (V2.2 addition)
    citation_method: str = Field(
        "llm",
        description="Citation check method: llm, attr_evaluator, both"
    )
    citation_composite_strategy: str = Field(
        "any",
        description="Composite mode strategy: any, all, llm_primary, attr_primary"
    )
    attr_evaluator_config: Optional[str] = Field(
        None,
        description="Attribute Evaluator config file path"
    )
    use_fast_mode: bool = Field(
        True,
        description="Use fast mode for Attribute Evaluator"
    )

    # Tool config (V2.4 addition)
    enable_autogeo_rephrase: bool = Field(
        True,
        description="Enable autogeo_rephrase tool (AutoGEO paper method). "
                    "Set to False to use only BatchGEO native tools."
    )


class MultiDocConfigV2(BaseModel):
    """
    Multi-document parallel processing configuration

    Controls document-level concurrent processing, used with AgentGEOConfigV2.
    AgentGEOConfigV2 controls query concurrency within a document, MultiDocConfigV2 controls cross-document concurrency.
    """

    max_doc_concurrency: int = Field(
        2, ge=1, le=10, description="Number of documents processed simultaneously (recommended 2-4)"
    )
    batch_config: AgentGEOConfigV2 = Field(
        default_factory=AgentGEOConfigV2, description="Single document processing config"
    )
    share_search_cache: bool = Field(True, description="Share search cache across documents")
    fail_fast: bool = Field(False, description="Abort all processing on single document failure")
    max_retries_per_doc: int = Field(1, ge=0, le=3, description="Document-level retry count")
    progress_interval_ms: int = Field(1000, description="Progress callback minimum interval (ms)")


@dataclass
class DocumentOptimizationResult:
    """
    Single document optimization result

    Encapsulates WebPage optimization process and result, including success/failure status, duration, etc.
    """

    webpage: "WebPage"  # Optimized WebPage
    batch_results: List[OptimizationResultV2]  # All batch results for this document
    success: bool  # Whether successful
    error: Optional[str] = None  # Error message (if failed)
    duration_ms: float = 0.0  # Processing duration (ms)
    queries_count: int = 0  # Number of queries processed
    suggestions_applied: int = 0  # Number of suggestions applied

    def get_success_rate_improvement(self) -> float:
        """Calculate success rate improvement"""
        if not self.batch_results:
            return 0.0
        first = self.batch_results[0].success_rate_before
        last = self.batch_results[-1].success_rate_after if self.batch_results[-1].success_rate_after else first
        return last - first


@dataclass
class MultiDocOptimizationResult:
    """
    Multi-document optimization aggregated result

    Contains optimization results for all documents and overall statistics.
    """

    results: List[DocumentOptimizationResult]  # Per-document results
    total_duration_ms: float  # Total duration
    success_count: int = 0  # Successful document count
    failure_count: int = 0  # Failed document count

    def __post_init__(self):
        """Automatically calculate success/failure counts"""
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
        """Get diagnosis statistics for all documents"""
        stats: Dict[str, int] = {}
        for doc_result in self.results:
            for batch_result in doc_result.batch_results:
                for cause, count in batch_result.diagnosis_stats.items():
                    stats[cause] = stats.get(cause, 0) + count
        return stats

    def get_summary(self) -> Dict[str, Any]:
        """Get summary information"""
        return {
            "total_documents": self.total_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "success_rate": self.success_rate,
            "total_duration_ms": self.total_duration_ms,
            "avg_duration_per_doc_ms": self.total_duration_ms / self.total_count if self.total_count else 0,
            "diagnosis_stats": self.get_overall_diagnosis_stats(),
        }
