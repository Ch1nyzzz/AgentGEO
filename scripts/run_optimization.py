#!/usr/bin/env python3
"""
AgentGEO Unified Optimization Script

Supports three optimization methods:
1. AutoGEO - Rule-based rewriting (paper baseline)
2. AgentGEO - Our method (suggestion-orchestrated optimization)
3. GEO-Bench - 9 baseline optimization methods

Features:
- Training citation rate tracking
- Test query citation evaluation (before/after)
- Document-level parallelism
- Checkpoint resume (skip completed documents)
- Summary analysis report

Usage:
    python run_optimization.py --config optimization_config.yaml
    python run_optimization.py --doc-limit 2  # quick test
    python run_optimization.py --force-restart  # ignore checkpoints
    python run_optimization.py --doc-concurrency 3  # 3 docs in parallel
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import yaml
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv

# Avoid SOCKS proxy issues with httpx (use http_proxy/https_proxy instead)
os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

# Add project path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / ".env")

from optimizers import create_optimizer
from utils.data_loader import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_training_citation_info(optimized) -> Dict[str, Any]:
    """Extract training citation data from AgentGEOResult"""
    per_query = []
    uncitable = []
    batch_rates = []

    for batch in optimized.optimization_results:
        batch_rates.append({
            "batch_id": batch.batch_id,
            "queries": batch.queries,
            "success_rate_before": batch.success_rate_before,
            "success_rate_after": batch.success_rate_after,
            "diagnosis_stats": batch.diagnosis_stats,
        })
        for qr in batch.query_results:
            entry = {
                "query": qr.query,
                "is_cited": qr.is_cited,
                "iterations_used": qr.iterations_used,
                "geo_score_overall": qr.geo_score_overall,
            }
            if qr.diagnosis:
                entry["diagnosis"] = qr.diagnosis.to_dict()
            # Save optimization attempt details for all queries
            if hasattr(qr, 'suggestions') and qr.suggestions:
                entry["optimization_attempts"] = [
                    {
                        "iteration": s.iteration,
                        "tool_name": s.tool_name,
                        "reasoning": s.reasoning,
                        "key_changes": s.key_changes,
                        "diagnosis": s.diagnosis.to_dict() if s.diagnosis else None,
                        "content_before": getattr(s, 'original_content', ''),
                        "content_after": s.proposed_content,
                        "target_segment_index": s.target_segment_index,
                    }
                    for s in qr.suggestions
                ]
            per_query.append(entry)
            if not qr.is_cited:
                uncitable.append(entry)

    total = len(per_query)
    cited = sum(1 for q in per_query if q["is_cited"])

    return {
        "train_citation_rate": cited / total if total > 0 else 0.0,
        "cited_count": cited,
        "total_queries": total,
        "uncitable_queries": uncitable,
        "batch_progression": batch_rates,
        "per_query_results": per_query,
        "diagnosis_stats": optimized.diagnosis_stats,
    }


async def process_document(
    doc: Dict[str, Any],
    optimizers: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Process a single document: optimize, extract training citation, evaluate test queries"""
    doc_id = doc.get("doc_id", "unknown")
    logger.info(f"Processing document: {doc_id}")

    result = {
        "doc_id": doc_id,
        "url": doc.get("url", ""),
        "original_text_length": len(doc.get("raw_html", "")),
        "timestamp": datetime.now().isoformat(),
    }

    eval_config = config.get("evaluation", {})
    enable_citation_eval = eval_config.get("enable_citation", False)
    test_queries = doc.get("test_queries", [])

    for name, optimizer in optimizers.items():
        try:
            logger.info(f"  [{doc_id}] Running {name}...")

            # 1. Pre-optimization test evaluation (baseline)
            if enable_citation_eval and test_queries and hasattr(optimizer, 'evaluate_page_async'):
                logger.info(f"    [{doc_id}] Evaluating baseline with {len(test_queries)} test queries...")
                baseline_eval = await optimizer.evaluate_page_async(
                    raw_html=doc["raw_html"],
                    test_queries=test_queries,
                    url=doc.get("url", "")
                )
                baseline_citation_rate = baseline_eval.get("ratio", 0.0)
                result[f"{name}_baseline_test_citation_rate"] = baseline_citation_rate
                # Per-query detail
                baseline_per_query = {q: baseline_eval.get(q, False) for q in test_queries}
                result[f"{name}_baseline_test_per_query"] = baseline_per_query
                logger.info(f"    [{doc_id}] Baseline test citation rate: {baseline_citation_rate:.2%}")
            else:
                baseline_citation_rate = None

            # 2. Run optimization with train_queries
            optimized = await optimizer.optimize_async(
                raw_html=doc["raw_html"],
                train_queries=doc.get("train_queries", []),
                url=doc.get("url", "")
            )

            if hasattr(optimized, 'optimized_text'):
                result[f"{name}_text"] = optimized.optimized_text
                result[f"{name}_html"] = optimized.optimized_html
                optimized_html = optimized.optimized_html

                # Extract training citation info
                train_citation = extract_training_citation_info(optimized)
                result[f"{name}_train_citation"] = train_citation
                logger.info(
                    f"  [{doc_id}] Training citation rate: {train_citation['train_citation_rate']:.2%} "
                    f"({train_citation['cited_count']}/{train_citation['total_queries']})"
                )
                if train_citation['uncitable_queries']:
                    logger.info(f"  [{doc_id}] Uncitable queries: {len(train_citation['uncitable_queries'])}")
            else:
                result[f"{name}_text"] = optimized
                optimized_html = optimized

            # 3. Post-optimization test evaluation
            if enable_citation_eval and test_queries and hasattr(optimizer, 'evaluate_page_async'):
                logger.info(f"    [{doc_id}] Evaluating optimized page with {len(test_queries)} test queries...")
                optimized_eval = await optimizer.evaluate_page_async(
                    raw_html=optimized_html,
                    test_queries=test_queries,
                    url=doc.get("url", "")
                )
                optimized_citation_rate = optimized_eval.get("ratio", 0.0)
                result[f"{name}_optimized_test_citation_rate"] = optimized_citation_rate
                # Per-query detail
                optimized_per_query = {q: optimized_eval.get(q, False) for q in test_queries}
                result[f"{name}_optimized_test_per_query"] = optimized_per_query
                logger.info(f"    [{doc_id}] Optimized test citation rate: {optimized_citation_rate:.2%}")

                if baseline_citation_rate is not None:
                    delta = optimized_citation_rate - baseline_citation_rate
                    result[f"{name}_delta_test_citation_rate"] = delta
                    logger.info(f"    [{doc_id}] Test delta: {delta:+.2%}")

            logger.info(f"  [{doc_id}] {name} completed")
        except Exception as e:
            logger.error(f"  [{doc_id}] {name} failed: {e}", exc_info=True)
            result[f"{name}_error"] = str(e)

    return result


def get_completed_doc_ids(checkpoint_dir: Path) -> set:
    """Get set of completed doc IDs from checkpoint directory"""
    completed = set()
    if checkpoint_dir.exists():
        for f in checkpoint_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if "doc_id" in data:
                    completed.add(data["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def save_checkpoint(result: Dict[str, Any], checkpoint_dir: Path):
    """Save single document result as checkpoint"""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    doc_id = result.get("doc_id", "unknown")
    safe_id = str(doc_id).replace("/", "_").replace("\\", "_")
    filepath = checkpoint_dir / f"{safe_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info(f"  Checkpoint saved: {filepath}")


def generate_analysis_report(results: List[Dict[str, Any]], output_dir: Path):
    """Generate summary analysis report from all document results"""
    total_docs = len(results)
    all_uncitable = []
    diagnosis_distribution = defaultdict(int)
    per_doc_summary = []
    total_train_cited = 0
    total_train_queries = 0
    total_test_baseline_cited = 0
    total_test_optimized_cited = 0
    total_test_queries = 0

    for r in results:
        doc_summary = {"doc_id": r["doc_id"], "url": r.get("url", "")}

        # Training citation
        for key, val in r.items():
            if key.endswith("_train_citation") and isinstance(val, dict):
                doc_summary["train_citation_rate"] = val["train_citation_rate"]
                doc_summary["cited_count"] = val["cited_count"]
                doc_summary["total_train_queries"] = val["total_queries"]
                doc_summary["uncitable_count"] = len(val.get("uncitable_queries", []))
                total_train_cited += val["cited_count"]
                total_train_queries += val["total_queries"]

                for uq in val.get("uncitable_queries", []):
                    all_uncitable.append({
                        "doc_id": r["doc_id"],
                        "query": uq["query"],
                        "diagnosis": uq.get("diagnosis", {}),
                        "iterations_used": uq.get("iterations_used", 0),
                    })

                for cause, count in val.get("diagnosis_stats", {}).items():
                    diagnosis_distribution[cause] += count
                break

        # Test citation
        for key, val in r.items():
            if key.endswith("_baseline_test_citation_rate"):
                prefix = key.replace("_baseline_test_citation_rate", "")
                baseline_rate = val
                optimized_rate = r.get(f"{prefix}_optimized_test_citation_rate", 0.0)
                delta = r.get(f"{prefix}_delta_test_citation_rate", 0.0)
                doc_summary["test_baseline_citation_rate"] = baseline_rate
                doc_summary["test_optimized_citation_rate"] = optimized_rate
                doc_summary["test_delta"] = delta

                # Count per-query for aggregation
                baseline_pq = r.get(f"{prefix}_baseline_test_per_query", {})
                optimized_pq = r.get(f"{prefix}_optimized_test_per_query", {})
                n_test = len(baseline_pq) if baseline_pq else 0
                total_test_queries += n_test
                total_test_baseline_cited += sum(1 for v in baseline_pq.values() if v)
                total_test_optimized_cited += sum(1 for v in optimized_pq.values() if v)
                break

        per_doc_summary.append(doc_summary)

    # Group uncitable queries by diagnosis
    uncitable_by_diagnosis = defaultdict(list)
    never_cited = []
    for uq in all_uncitable:
        diag = uq.get("diagnosis", {})
        cause = diag.get("root_cause", "UNKNOWN") if diag else "UNKNOWN"
        uncitable_by_diagnosis[cause].append(uq["query"])
        if uq.get("iterations_used", 0) >= 3:
            never_cited.append(uq)

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_docs": total_docs,
            "train": {
                "avg_citation_rate": total_train_cited / total_train_queries if total_train_queries > 0 else 0.0,
                "total_cited": total_train_cited,
                "total_queries": total_train_queries,
                "total_uncitable": len(all_uncitable),
            },
            "test": {
                "avg_baseline_citation_rate": total_test_baseline_cited / total_test_queries if total_test_queries > 0 else 0.0,
                "avg_optimized_citation_rate": total_test_optimized_cited / total_test_queries if total_test_queries > 0 else 0.0,
                "avg_delta": (total_test_optimized_cited - total_test_baseline_cited) / total_test_queries if total_test_queries > 0 else 0.0,
                "total_queries": total_test_queries,
            },
            "diagnosis_distribution": dict(diagnosis_distribution),
        },
        "uncitable_query_analysis": {
            "by_diagnosis": {k: v for k, v in uncitable_by_diagnosis.items()},
            "never_cited_across_retries": [
                {"doc_id": u["doc_id"], "query": u["query"], "iterations": u["iterations_used"]}
                for u in never_cited
            ],
        },
        "per_document_summary": per_doc_summary,
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"analysis_report_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Analysis report saved: {report_path}")
    logger.info(f"  Train - avg citation rate: {report['summary']['train']['avg_citation_rate']:.2%}")
    logger.info(f"  Train - uncitable queries: {report['summary']['train']['total_uncitable']}")
    if total_test_queries > 0:
        logger.info(f"  Test  - baseline: {report['summary']['test']['avg_baseline_citation_rate']:.2%}")
        logger.info(f"  Test  - optimized: {report['summary']['test']['avg_optimized_citation_rate']:.2%}")
        logger.info(f"  Test  - delta: {report['summary']['test']['avg_delta']:+.2%}")
    logger.info(f"  Diagnosis distribution: {dict(diagnosis_distribution)}")

    return report


async def main():
    parser = argparse.ArgumentParser(description="AgentGEO Unified Optimization Script")
    parser.add_argument("--config", default="optimization_config.yaml", help="Configuration file path")
    parser.add_argument("--method", choices=["autogeo", "agentgeo", "baseline", "all"],
                        help="Optimization method (overrides config)")
    parser.add_argument("--data", help="Data file path (overrides config)")
    parser.add_argument("--output-dir", help="Output directory (overrides config)")
    parser.add_argument("--doc-limit", type=int, help="Limit number of documents (overrides config)")
    parser.add_argument("--doc-offset", type=int, help="Document offset (overrides config)")
    parser.add_argument("--doc-concurrency", type=int, default=1,
                        help="Number of documents to process in parallel (default: 1)")
    parser.add_argument("--force-restart", action="store_true", help="Ignore checkpoints")

    args = parser.parse_args()

    # 1. Load configuration
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = REPO_ROOT / args.config

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.method:
        config["optimizer"]["method"] = args.method
    if args.data:
        config["data"]["input_path"] = args.data
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir
    if args.doc_limit is not None:
        config["data"]["doc_limit"] = args.doc_limit
    if args.doc_offset is not None:
        config["data"]["doc_offset"] = args.doc_offset

    doc_concurrency = args.doc_concurrency

    logger.info("=" * 60)
    logger.info("AgentGEO Optimization System")
    logger.info("=" * 60)
    logger.info(f"Optimization method: {config['optimizer']['method']}")
    logger.info(f"Document concurrency: {doc_concurrency}")
    logger.info(f"Query concurrency: {config.get('agentgeo', {}).get('max_concurrency', 4)}")
    logger.info(f"Test evaluation: {config.get('evaluation', {}).get('enable_citation', False)}")

    # 2. Load data
    data_loader = DataLoader(config["data"])
    documents = data_loader.load()
    logger.info(f"Loaded {len(documents)} documents")

    # 3. Create optimizers
    method = config["optimizer"]["method"]
    optimizers = create_optimizer(method, config)
    logger.info(f"Created {len(optimizers)} optimizers")

    # 4. Checkpoint setup
    output_dir = Path(config["output"]["base_dir"])
    checkpoint_dir = output_dir / "documents"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.force_restart:
        completed_ids = set()
        logger.info("Force restart: ignoring existing checkpoints")
    else:
        completed_ids = get_completed_doc_ids(checkpoint_dir)
        if completed_ids:
            logger.info(f"Resuming: {len(completed_ids)} documents already completed")

    # 5. Process documents (with concurrency control)
    results = [None] * len(documents)
    skipped = 0
    semaphore = asyncio.Semaphore(doc_concurrency)
    lock = asyncio.Lock()

    async def process_one(idx: int, doc: Dict[str, Any]):
        nonlocal skipped
        doc_id = doc.get("doc_id", "unknown")

        if doc_id in completed_ids:
            safe_id = str(doc_id).replace("/", "_").replace("\\", "_")
            cached_path = checkpoint_dir / f"{safe_id}.json"
            if cached_path.exists():
                result = json.loads(cached_path.read_text(encoding="utf-8"))
                results[idx] = result
                async with lock:
                    skipped += 1
                logger.info(f"[{idx+1}/{len(documents)}] Skipped (cached): {doc_id}")
                return

        async with semaphore:
            logger.info(f"\n[{idx+1}/{len(documents)}] Processing document: {doc_id}")
            try:
                result = await process_document(doc, optimizers, config)
                results[idx] = result
                save_checkpoint(result, checkpoint_dir)
            except Exception as e:
                logger.error(f"Document {doc_id} failed: {e}", exc_info=True)
                results[idx] = {"doc_id": doc_id, "error": str(e), "timestamp": datetime.now().isoformat()}

    tasks = [asyncio.create_task(process_one(i, doc)) for i, doc in enumerate(documents)]
    await asyncio.gather(*tasks)

    # Filter out None (shouldn't happen but be safe)
    results = [r for r in results if r is not None]

    # 6. Save all results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"optimization_results_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 7. Generate analysis report
    successful_results = [r for r in results if "error" not in r or any(k.endswith("_train_citation") for k in r)]
    if successful_results:
        generate_analysis_report(successful_results, output_dir)

    logger.info("=" * 60)
    logger.info(f"Optimization complete!")
    logger.info(f"Processed {len(results)} documents ({skipped} from cache)")
    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
