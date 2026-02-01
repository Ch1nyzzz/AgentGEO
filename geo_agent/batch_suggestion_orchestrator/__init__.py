"""
AgentGEO Batch Suggestion Orchestrator - 基于建议编排的智能优化系统

主要特性：
1. 使用两阶段失败分析（diagnose + select_tool_strategy）
2. 支持 14 类故障分类法（FAILURE_TAXONOMY）
3. 异步并行处理多个 query
4. 支持 HTML DOM 分块（StructuralHtmlParser）
5. 智能建议合并（基于诊断结果）
6. 完整的历史记录和策略注入支持
7. 文档级并行处理（MultiDocumentOptimizer）
8. 多引用检查方式支持（LLM, AttrEvaluator, Both）
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
from .agent_geo import AgentGEOV2              # 改名
from .suggestion_processor import SuggestionProcessorV2  # 改名
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
    "AgentGEOV2",                    # 改名
    "SuggestionProcessorV2",         # 改名
    # Single-document models
    "SuggestionV2",
    "OrchestraGroupV2",
    "QueryResultV2",
    "OptimizationResultV2",          # 改名
    "AgentGEOConfigV2",              # 改名
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

__version__ = "2.3.0"  # 版本号更新
