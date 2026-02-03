"""
Optimizer wrapper module

Provides a unified interface for three optimization methods:
- AutoGEO: Rule-based rewriting method from the paper
- AgentGEO: Intelligent optimization system based on suggestion orchestration
- Baseline: 9 baseline methods provided by GEO-Bench
"""
from typing import Dict, Any
from pathlib import Path
import sys

# Add path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from .agentgeo_optimizer import AgentGEOOptimizer
from .autogeo_optimizer import AutoGEOOptimizer
from .baseline_optimizer import BaselineOptimizers


def create_optimizer(method: str, config: Dict) -> Dict[str, Any]:
    """Create optimizer based on method

    Args:
        method: Optimization method (autogeo, agentgeo, baseline, all)
        config: Configuration dictionary

    Returns:
        Optimizer dictionary {name: optimizer}
    """
    optimizers = {}

    if method == "autogeo" or method == "all":
        optimizers["autogeo"] = AutoGEOOptimizer(
            **config.get("autogeo", {})
        )

    if method == "agentgeo" or method == "all":
        optimizers["agentgeo"] = AgentGEOOptimizer(
            **config.get("agentgeo", {})
        )

    if method == "baseline" or method == "all":
        baseline_config = config.get("baseline", {})
        baseline_methods = baseline_config.get("methods") or [
            "authoritative", "cite_sources", "statistics_addition",
            "keyword_stuffing", "quotation_addition", "easy_to_understand",
            "fluency_optimization", "unique_words", "technical_terms"
        ]
        for m in baseline_methods:
            optimizers[f"baseline_{m}"] = BaselineOptimizers(
                method_name=m,
                provider=baseline_config["provider"],
                model=baseline_config["model"]
            )

    return optimizers


__all__ = [
    "create_optimizer",
    "AutoGEOOptimizer",
    "AgentGEOOptimizer",
    "BaselineOptimizers",
]
