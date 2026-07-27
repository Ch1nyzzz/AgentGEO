#!/usr/bin/env python3
"""Re-evaluate saved optimized HTML without running optimization or baseline."""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from optimizers import AgentGEOOptimizer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_queries(value: Any) -> List[str]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value or [])


def extract_per_query(eval_result: Dict[str, Any], queries: List[str]) -> Dict[str, Dict]:
    detailed = eval_result.get("detailed", {})
    return {
        query: {
            "is_cited": detailed.get(query, {}).get("is_cited", False),
            "answer": detailed.get(query, {}).get("answer", ""),
            "geo_score": detailed.get(query, {}).get("geo_score"),
            "no_competitors": detailed.get(query, {}).get("no_competitors", False),
        }
        for query in queries
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generation + citation checking on saved optimized HTML only"
    )
    parser.add_argument("--sample", default="sample_50.parquet")
    parser.add_argument("--data", default="data/input.parquet")
    parser.add_argument("--agentgeo-config", default="configs_ablation/b10_mem.yaml")
    parser.add_argument("--output-dir", default="outputs_generation_only/sample50_run1")
    parser.add_argument("--doc-concurrency", type=int, default=16)
    parser.add_argument("--doc-limit", type=int)
    args = parser.parse_args()

    import pandas as pd
    import yaml

    sample = pd.read_parquet(args.sample)
    source = pd.read_parquet(args.data)
    source_by_id = {str(row["doc_id"]): row for row in source.to_dict("records")}
    if args.doc_limit:
        sample = sample.head(args.doc_limit)

    with open(args.agentgeo_config) as handle:
        config = yaml.safe_load(handle)
    evaluator = AgentGEOOptimizer(**config.get("agentgeo", {}))

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "documents"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(args.doc_concurrency)

    async def evaluate_one(index: int, saved: Dict[str, Any]) -> Dict[str, Any]:
        doc_id = str(saved["doc_id"])
        checkpoint = checkpoint_dir / f"{doc_id}.json"
        source_row = source_by_id.get(doc_id)
        if source_row is None:
            raise KeyError(f"{doc_id} is missing from {args.data}")

        queries = normalize_queries(source_row.get("test_queries"))
        if checkpoint.exists():
            cached = json.loads(checkpoint.read_text())
            if len(cached.get("per_query", {})) == len(queries):
                logger.info("[%d/%d] %s resumed", index + 1, len(sample), doc_id)
                return cached

        optimized_html = saved.get("agentgeo_html")
        if not isinstance(optimized_html, str) or not optimized_html.strip():
            raise ValueError(f"{doc_id} has no agentgeo_html in {args.sample}")

        async with semaphore:
            logger.info(
                "[%d/%d] %s generating for %d queries",
                index + 1,
                len(sample),
                doc_id,
                len(queries),
            )
            evaluated = await evaluator.evaluate_page_async(
                raw_html=optimized_html,
                test_queries=queries,
                url=str(saved.get("url") or source_row.get("url") or ""),
            )

        per_query = extract_per_query(evaluated, queries)
        cited = sum(1 for item in per_query.values() if item["is_cited"])
        no_competitors = sum(
            1 for item in per_query.values() if item["no_competitors"]
        )
        result = {
            "doc_id": doc_id,
            "url": str(saved.get("url") or source_row.get("url") or ""),
            "timestamp": datetime.now().isoformat(),
            "total_queries": len(queries),
            "cited": cited,
            "citation_rate": cited / len(queries) if queries else 0.0,
            "no_competitors": no_competitors,
            "per_query": per_query,
        }
        checkpoint.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        logger.info(
            "[%d/%d] %s CR=%.2f%% (%d/%d)",
            index + 1,
            len(sample),
            doc_id,
            100 * result["citation_rate"],
            cited,
            len(queries),
        )
        return result

    rows = sample.to_dict("records")
    completed = await asyncio.gather(
        *(evaluate_one(index, row) for index, row in enumerate(rows))
    )
    completed.sort(key=lambda item: item["doc_id"])

    total_queries = sum(item["total_queries"] for item in completed)
    total_cited = sum(item["cited"] for item in completed)
    valid_docs = [item for item in completed if item["total_queries"]]
    macro_rate = (
        sum(item["citation_rate"] for item in valid_docs) / len(valid_docs)
        if valid_docs
        else 0.0
    )
    summary = {
        "generated_at": datetime.now().isoformat(),
        "sample": args.sample,
        "data": args.data,
        "agentgeo_config": args.agentgeo_config,
        "documents": len(completed),
        "total_queries": total_queries,
        "total_cited": total_cited,
        "micro_citation_rate": total_cited / total_queries if total_queries else 0.0,
        "macro_per_doc_citation_rate": macro_rate,
        "total_no_competitors": sum(item["no_competitors"] for item in completed),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(completed, indent=2, ensure_ascii=False)
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
