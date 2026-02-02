#!/usr/bin/env python3
"""
AgentGEO 统一优化脚本

支持三种优化方式：
1. AutoGEO - 基于规则的重写（论文baseline）
2. AgentGEO - 我们的方法（建议编排优化）
3. GEO-Bench - 9种baseline优化方法

用法:
    python run_optimization.py --config optimization_config.yaml
    python run_optimization.py --method agentgeo
    python run_optimization.py --method all --data test.parquet
"""
import argparse
import asyncio
import json
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from optimizers import create_optimizer
from utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def process_document(
    doc: Dict[str, Any],
    optimizers: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """处理单个文档"""
    doc_id = doc.get("doc_id", "unknown")
    logger.info(f"Processing document: {doc_id}")

    result = {
        "doc_id": doc_id,
        "url": doc.get("url", ""),
        "original_text_length": len(doc.get("raw_html", "")),
    }

    # 运行每个优化器
    for name, optimizer in optimizers.items():
        try:
            logger.info(f"  Running {name}...")
            optimized = await optimizer.optimize_async(
                raw_html=doc["raw_html"],
                train_queries=doc.get("train_queries", []),
                url=doc.get("url", "")
            )

            if hasattr(optimized, 'optimized_text'):
                result[f"{name}_text"] = optimized.optimized_text
                result[f"{name}_html"] = optimized.optimized_html
            else:
                result[f"{name}_text"] = optimized

            logger.info(f"  ✓ {name} completed")
        except Exception as e:
            logger.error(f"  ✗ {name} failed: {e}")
            result[f"{name}_error"] = str(e)

    return result


async def main():
    parser = argparse.ArgumentParser(description="AgentGEO 统一优化脚本")
    parser.add_argument(
        "--config",
        default="optimization_config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--method",
        choices=["autogeo", "agentgeo", "baseline", "all"],
        help="优化方法（覆盖配置文件）"
    )
    parser.add_argument(
        "--data",
        help="数据文件路径（覆盖配置文件）"
    )
    parser.add_argument(
        "--output-dir",
        help="输出目录（覆盖配置文件）"
    )

    args = parser.parse_args()

    # 1. 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = REPO_ROOT / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖
    if args.method:
        config["optimizer"]["method"] = args.method
    if args.data:
        config["data"]["input_path"] = args.data
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir

    logger.info("=" * 60)
    logger.info("AgentGEO 优化系统")
    logger.info("=" * 60)
    logger.info(f"优化方法: {config['optimizer']['method']}")

    # 2. 初始化数据加载器
    data_loader = DataLoader(config["data"])
    documents = data_loader.load()
    logger.info(f"加载了 {len(documents)} 个文档")

    # 3. 创建优化器
    method = config["optimizer"]["method"]
    optimizers = create_optimizer(method, config)
    logger.info(f"创建了 {len(optimizers)} 个优化器")

    # 4. 处理文档
    results = []
    for i, doc in enumerate(documents):
        logger.info(f"\n[{i+1}/{len(documents)}] 处理文档...")
        result = await process_document(doc, optimizers, config)
        results.append(result)

    # 5. 保存结果
    output_dir = Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"optimization_results_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"优化完成！")
    logger.info(f"处理了 {len(results)} 个文档")
    logger.info(f"结果保存到: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
