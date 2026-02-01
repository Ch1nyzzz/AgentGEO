"""
优化器封装模块

提供统一接口来使用三种优化方式：
- AutoGEO: 论文中的基于规则的重写方法
- AgentGEO: 基于建议编排的智能优化系统
- Baseline: GEO-Bench 提供的 9 种 baseline 方法
"""
from typing import Dict, Any
from pathlib import Path
import sys

# 添加路径
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from .agentgeo_optimizer import AgentGEOOptimizer
from .autogeo_optimizer import AutoGEOOptimizer
from .baseline_optimizer import BaselineOptimizers


def create_optimizer(method: str, config: Dict) -> Dict[str, Any]:
    """根据方法创建优化器

    Args:
        method: 优化方法 (autogeo, agentgeo, baseline, all)
        config: 配置字典

    Returns:
        优化器字典 {name: optimizer}
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
