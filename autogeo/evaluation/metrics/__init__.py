"""
Evaluation metrics module for AutoGEO.

Provides metrics for evaluating rewritten documents:
- GEO score: Visibility metrics (position, token count, citation frequency)
- GEU score: Utility metrics (citation quality, keypoint coverage, response quality)

Uses lazy imports to avoid initializing API clients at module load time.
"""

# GEO score functions are lightweight, import directly
from .geo_score import (
    get_num_words,
    extract_citations_new,
    impression_wordpos_count_simple,
    impression_word_count_simple,
    impression_pos_count_simple,
)

__all__ = [
    # GEO score functions
    'get_num_words',
    'extract_citations_new',
    'impression_wordpos_count_simple',
    'impression_word_count_simple',
    'impression_pos_count_simple',
    # GEU score functions (lazy loaded)
    'preprocess_data_for_evaluation',
    'calculate_citation_quality',
    'calculate_quality_dimensions',
    'calculate_keypoint_coverage',
    'process_single_question',
    'evaluate_ge_utility',
    'geu_score',
]


def __getattr__(name):
    """Lazy import GEU score functions to avoid OpenAI client initialization."""
    geu_functions = {
        'preprocess_data_for_evaluation',
        'calculate_citation_quality',
        'calculate_quality_dimensions',
        'calculate_keypoint_coverage',
        'process_single_question',
        'evaluate_ge_utility',
        'geu_score',
    }
    if name in geu_functions:
        from . import geu_score as geu_module
        return getattr(geu_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


