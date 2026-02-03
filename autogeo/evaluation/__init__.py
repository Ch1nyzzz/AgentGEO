"""
Evaluation module for AutoGEO.

Uses lazy imports to avoid initializing API clients at module load time.
"""

__all__ = [
    'autogeo_evaluation',
    'aggregate_json_files',
    'generate_answer_gemini',
    'generate_answer_gpt',
    'generate_answer_claude',
]


def __getattr__(name):
    """Lazy import to avoid API client initialization at import time."""
    if name == 'autogeo_evaluation':
        from .evaluator import autogeo_evaluation
        return autogeo_evaluation
    elif name == 'aggregate_json_files':
        from .aggregate_results import aggregate_json_files
        return aggregate_json_files
    elif name == 'generate_answer_gemini':
        from .generative_engine import generate_answer_gemini
        return generate_answer_gemini
    elif name == 'generate_answer_gpt':
        from .generative_engine import generate_answer_gpt
        return generate_answer_gpt
    elif name == 'generate_answer_claude':
        from .generative_engine import generate_answer_claude
        return generate_answer_claude
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

