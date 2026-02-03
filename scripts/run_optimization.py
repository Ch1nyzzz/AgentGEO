#!/usr/bin/env python3
"""
AgentGEO Unified Optimization Script

Supports three optimization methods:
1. AutoGEO - Rule-based rewriting (paper baseline)
2. AgentGEO - Our method (suggestion-orchestrated optimization)
3. GEO-Bench - 9 baseline optimization methods

Usage:
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

# Add project path
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
    """Process a single document"""
    doc_id = doc.get("doc_id", "unknown")
    logger.info(f"Processing document: {doc_id}")

    result = {
        "doc_id": doc_id,
        "url": doc.get("url", ""),
        "original_text_length": len(doc.get("raw_html", "")),
    }

    # Run each optimizer
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
    parser = argparse.ArgumentParser(description="AgentGEO Unified Optimization Script")
    parser.add_argument(
        "--config",
        default="optimization_config.yaml",
        help="Configuration file path"
    )
    parser.add_argument(
        "--method",
        choices=["autogeo", "agentgeo", "baseline", "all"],
        help="Optimization method (overrides config file)"
    )
    parser.add_argument(
        "--data",
        help="Data file path (overrides config file)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory (overrides config file)"
    )

    args = parser.parse_args()

    # 1. Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = REPO_ROOT / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Command line arguments override config
    if args.method:
        config["optimizer"]["method"] = args.method
    if args.data:
        config["data"]["input_path"] = args.data
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir

    logger.info("=" * 60)
    logger.info("AgentGEO Optimization System")
    logger.info("=" * 60)
    logger.info(f"Optimization method: {config['optimizer']['method']}")

    # 2. Initialize data loader
    data_loader = DataLoader(config["data"])
    documents = data_loader.load()
    logger.info(f"Loaded {len(documents)} documents")

    # 3. Create optimizers
    method = config["optimizer"]["method"]
    optimizers = create_optimizer(method, config)
    logger.info(f"Created {len(optimizers)} optimizers")

    # 4. Process documents
    results = []
    for i, doc in enumerate(documents):
        logger.info(f"\n[{i+1}/{len(documents)}] Processing document...")
        result = await process_document(doc, optimizers, config)
        results.append(result)

    # 5. Save results
    output_dir = Path(config["output"]["base_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"optimization_results_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Optimization complete!")
    logger.info(f"Processed {len(results)} documents")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
