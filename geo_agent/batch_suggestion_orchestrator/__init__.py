"""
AgentGEO Batch Suggestion Orchestrator - Intelligent optimization system based on suggestion orchestration

Key Features:
1. Two-phase failure analysis (diagnose + select_tool_strategy)
2. Support for 14 failure categories (FAILURE_TAXONOMY)
3. Async parallel processing of multiple queries
4. HTML DOM chunking support (StructuralHtmlParser)
5. Intelligent suggestion merging (diagnosis-aware)
6. Complete history tracking and policy injection support
7. Document-level parallel processing (MultiDocumentOptimizer)
8. Multiple citation checking methods (LLM, AttrEvaluator, Both)
"""

from .models import (
    SuggestionV2,
    OrchestraGroupV2,
    QueryResultV2,
    OptimizationResultV2,      # 改名
    AgentGEOConfigV2,          # 改名
    # Multi-document types
    MultiDocConfigV2,
    DocumentOptimizationResult,
    MultiDocOptimizationResult,
)
from .agent_geo import AgentGEOV2              # Renamed
from .suggestion_processor import SuggestionProcessorV2  # Renamed
from .multi_doc_optimizer import (
    MultiDocumentOptimizer,
    optimize_multiple_documents,
    ProgressCallback,
)
from .citation_checker import (
    BaseCitationChecker,
    CitationMethod,
    CitationResult,
    LLMCitationChecker,
    AttrEvaluatorCitationChecker,
    CompositeCitationChecker,
    create_citation_checker,
)

__all__ = [
    # Core classes
    "AgentGEOV2",                    # Renamed
    "SuggestionProcessorV2",         # Renamed
    # Single-document models
    "SuggestionV2",
    "OrchestraGroupV2",
    "QueryResultV2",
    "OptimizationResultV2",          # Renamed
    "AgentGEOConfigV2",              # Renamed
    # Multi-document classes
    "MultiDocumentOptimizer",
    "MultiDocConfigV2",
    "DocumentOptimizationResult",
    "MultiDocOptimizationResult",
    # Citation checker classes
    "BaseCitationChecker",
    "CitationMethod",
    "CitationResult",
    "LLMCitationChecker",
    "AttrEvaluatorCitationChecker",
    "CompositeCitationChecker",
    "create_citation_checker",
    # Utilities
    "optimize_multiple_documents",
    "ProgressCallback",
]

__version__ = "2.3.0"  # Version update
