#!/usr/bin/env python3
"""
Post-hoc evaluation script: read existing optimized text, run test evaluation + similarity.
Skips the optimization step entirely — only evaluates.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
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
    format="%(asctime)s - [%(levelname)s] - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_similarity(original_text: str, optimized_text: str) -> Dict[str, float]:
    if not original_text or not optimized_text:
        return {"sequence_similarity": 0.0, "jaccard_similarity": 0.0, "cosine_similarity": 0.0}

    seq_sim = SequenceMatcher(None, original_text[:50000], optimized_text[:50000]).ratio()

    orig_words = set(original_text.lower().split())
    opt_words = set(optimized_text.lower().split())
    union = orig_words | opt_words
    jaccard = len(orig_words & opt_words) / len(union) if union else 0.0

    orig_counts = Counter(original_text.lower().split())
    opt_counts = Counter(optimized_text.lower().split())
    all_words = set(orig_counts) | set(opt_counts)
    dot = sum(orig_counts.get(w, 0) * opt_counts.get(w, 0) for w in all_words)
    norm_a = sum(v ** 2 for v in orig_counts.values()) ** 0.5
    norm_b = sum(v ** 2 for v in opt_counts.values()) ** 0.5
    cosine_sim = dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    return {
        "sequence_similarity": round(seq_sim, 4),
        "jaccard_similarity": round(jaccard, 4),
        "cosine_similarity": round(cosine_sim, 4),
    }


def text_to_html(text: str) -> str:
    """Wrap plain text in HTML <p> tags for evaluator compatibility."""
    paragraphs = ''.join(f'<p>{line}</p>' for line in text.split('\n') if line.strip())
    return f"<html><body>{paragraphs}</body></html>"


def extract_eval_geo_scores(eval_result: Dict, queries: List[str]) -> Dict[str, Any]:
    detailed = eval_result.get("detailed", {})
    per_query = {}
    geo_scores = []
    for q in queries:
        info = detailed.get(q, {})
        geo = info.get("geo_score")
        per_query[q] = {
            "is_cited": info.get("is_cited", False),
            "answer": info.get("answer", ""),
            "geo_score": geo,
        }
        if geo:
            geo_scores.append(geo["overall"])
    avg_geo = sum(geo_scores) / len(geo_scores) if geo_scores else 0.0
    return {"per_query": per_query, "avg_geo_score": round(avg_geo, 4)}


async def evaluate_document(
    doc_result: Dict[str, Any],
    original_html: str,
    test_queries: List[str],
    evaluator: AgentGEOOptimizer,
    methods: List[str],
) -> Dict[str, Any]:
    """Evaluate one document across all methods."""
    doc_id = doc_result["doc_id"]
    url = doc_result.get("url", "")

    result = {
        "doc_id": doc_id,
        "url": url,
        "timestamp": datetime.now().isoformat(),
    }

    # Extract original text for similarity
    try:
        from trafilatura import extract as tf_extract
        original_text = tf_extract(original_html) or ""
    except Exception:
        original_text = ""

    # Baseline evaluation (original page) — run once
    logger.info(f"  [{doc_id}] Evaluating baseline with {len(test_queries)} test queries...")
    baseline_eval = await evaluator.evaluate_page_async(
        raw_html=original_html,
        test_queries=test_queries,
        url=url,
    )
    baseline_citation_rate = baseline_eval.get("ratio", 0.0)
    baseline_detail = extract_eval_geo_scores(baseline_eval, test_queries)
    logger.info(f"  [{doc_id}] Baseline: citation={baseline_citation_rate:.1%}, geo={baseline_detail['avg_geo_score']:.4f}")

    for method in methods:
        text_key = f"{method}_text"
        optimized_text = doc_result.get(text_key, "")
        if not optimized_text:
            logger.warning(f"  [{doc_id}] {method}: no optimized text, skipping")
            continue

        try:
            # Similarity
            if original_text:
                sim = compute_similarity(original_text, optimized_text)
                result[f"{method}_similarity"] = sim
                result[f"{method}_length_ratio"] = round(
                    len(optimized_text) / len(original_text), 4
                ) if original_text else 0.0

            # Baseline test results (shared)
            result[f"{method}_baseline_test_citation_rate"] = baseline_citation_rate
            result[f"{method}_baseline_test_per_query"] = baseline_detail["per_query"]
            result[f"{method}_baseline_test_avg_geo_score"] = baseline_detail["avg_geo_score"]

            # Optimized test evaluation
            # Use HTML directly if available, otherwise wrap plain text
            eval_html = doc_result.get(f"{method}_html", "")
            if not eval_html:
                eval_html = text_to_html(optimized_text)
            elif not eval_html.strip().startswith('<'):
                eval_html = text_to_html(eval_html)
            logger.info(f"  [{doc_id}] Evaluating {method}...")
            optimized_eval = await evaluator.evaluate_page_async(
                raw_html=eval_html,
                test_queries=test_queries,
                url=url,
            )
            opt_rate = optimized_eval.get("ratio", 0.0)
            opt_detail = extract_eval_geo_scores(optimized_eval, test_queries)

            result[f"{method}_text"] = optimized_text
            result[f"{method}_optimized_test_citation_rate"] = opt_rate
            result[f"{method}_optimized_test_per_query"] = opt_detail["per_query"]
            result[f"{method}_optimized_test_avg_geo_score"] = opt_detail["avg_geo_score"]
            result[f"{method}_delta_test_citation_rate"] = round(opt_rate - baseline_citation_rate, 4)
            result[f"{method}_delta_test_avg_geo_score"] = round(
                opt_detail["avg_geo_score"] - baseline_detail["avg_geo_score"], 4
            )

            logger.info(f"  [{doc_id}] {method}: citation={opt_rate:.1%} (delta={opt_rate - baseline_citation_rate:+.1%}), "
                         f"geo={opt_detail['avg_geo_score']:.4f}")

        except Exception as e:
            logger.error(f"  [{doc_id}] {method} eval failed: {e}", exc_info=True)
            result[f"{method}_error"] = str(e)

    return result


async def main():
    parser = argparse.ArgumentParser(description="Post-hoc evaluation for existing optimization results")
    parser.add_argument("--results", required=True, help="Path to optimization_results JSON file")
    parser.add_argument("--data", default="data/input.parquet", help="Original data file")
    parser.add_argument("--config", default="geo_agent/config_gpt4.1.yaml", help="AgentGEO config for evaluator")
    parser.add_argument("--output-dir", default="outputs_gpt4.1_baselines_eval", help="Output directory")
    parser.add_argument("--doc-limit", type=int, help="Limit number of documents")
    parser.add_argument("--doc-concurrency", type=int, default=1)
    parser.add_argument("--methods", help="Comma-separated methods to evaluate (default: all)")
    parser.add_argument("--agentgeo-config", default="optimization_config_gpt4.1_baselines.yaml",
                        help="Full optimization config (for agentgeo section)")
    args = parser.parse_args()

    # Load optimization results
    with open(args.results) as f:
        all_results = json.load(f)
    logger.info(f"Loaded {len(all_results)} document results from {args.results}")

    if args.doc_limit:
        all_results = all_results[:args.doc_limit]

    # Load original data
    import pandas as pd
    df = pd.read_parquet(args.data)
    doc_map = {}
    for _, row in df.iterrows():
        doc_id = row.get("doc_id", "")
        doc_map[doc_id] = row

    # Detect methods from first result, or use --methods filter
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = []
        if all_results:
            for k in all_results[0].keys():
                if k.endswith("_text") and all_results[0][k]:
                    methods.append(k.replace("_text", ""))
    logger.info(f"Methods to evaluate: {methods}")

    # Create evaluator
    import yaml
    with open(args.agentgeo_config) as f:
        full_config = yaml.safe_load(f)
    agentgeo_config = full_config.get("agentgeo", {})
    evaluator = AgentGEOOptimizer(**agentgeo_config)

    # Output setup
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "documents"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Check existing checkpoints
    completed_ids = set()
    for cp in checkpoint_dir.glob("*.json"):
        try:
            data = json.loads(cp.read_text())
            if "doc_id" in data:
                completed_ids.add(data["doc_id"])
        except Exception:
            pass
    if completed_ids:
        logger.info(f"Resuming: {len(completed_ids)} documents already completed")

    # Process
    results = []
    semaphore = asyncio.Semaphore(args.doc_concurrency)

    async def process_one(idx, doc_result):
        doc_id = doc_result["doc_id"]
        if doc_id in completed_ids:
            # Load cached result
            safe_id = str(doc_id).replace("/", "_").replace("\\", "_")
            cached = json.loads((checkpoint_dir / f"{safe_id}.json").read_text())
            logger.info(f"[{idx+1}/{len(all_results)}] Skipped (cached): {doc_id}")
            return cached

        orig_row = doc_map.get(doc_id)
        if orig_row is None:
            logger.warning(f"[{idx+1}/{len(all_results)}] {doc_id}: not found in data, skipping")
            return None

        async with semaphore:
            logger.info(f"\n[{idx+1}/{len(all_results)}] Evaluating document: {doc_id}")
            test_queries = orig_row.get("test_queries", [])
            if hasattr(test_queries, 'tolist'):
                test_queries = test_queries.tolist()

            result = await evaluate_document(
                doc_result, orig_row["raw_html"], test_queries, evaluator, methods
            )

            # Save checkpoint
            safe_id = str(doc_id).replace("/", "_").replace("\\", "_")
            with open(checkpoint_dir / f"{safe_id}.json", "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"  Checkpoint saved: {doc_id}")
            return result

    tasks = [asyncio.create_task(process_one(i, dr)) for i, dr in enumerate(all_results)]
    for t in asyncio.as_completed(tasks):
        r = await t
        if r:
            results.append(r)

    # Save final results
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"eval_results_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info(f"Evaluation complete! {len(results)} documents")
    logger.info(f"Results saved to: {output_dir / f'eval_results_{ts}.json'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
