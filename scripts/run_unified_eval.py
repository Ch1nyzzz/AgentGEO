#!/usr/bin/env python3
"""
Unified evaluation: one baseline eval per doc, evaluate all methods' optimized text.
Reads from multiple result files, shares baseline eval across methods.
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


def compute_similarity(original_text: str, optimized_text: str) -> Dict[str, float]:
    if not original_text or not optimized_text:
        return {"tfidf_similarity": 0.0, "jaccard_similarity": 0.0, "embedding_similarity": 0.0}

    # 1. TF-IDF cosine similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([original_text, optimized_text])
    tfidf_sim = sk_cosine(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

    # 2. Jaccard similarity (word-level)
    orig_words = set(original_text.lower().split())
    opt_words = set(optimized_text.lower().split())
    union = orig_words | opt_words
    jaccard = len(orig_words & opt_words) / len(union) if union else 0.0

    # 3. Embedding similarity (sentence-transformers)
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        emb_orig = model.encode(original_text[:10000], convert_to_numpy=True)
        emb_opt = model.encode(optimized_text[:10000], convert_to_numpy=True)
        embed_sim = float(np.dot(emb_orig, emb_opt) / (np.linalg.norm(emb_orig) * np.linalg.norm(emb_opt)))
    except Exception:
        embed_sim = 0.0

    return {
        "tfidf_similarity": round(float(tfidf_sim), 4),
        "jaccard_similarity": round(jaccard, 4),
        "embedding_similarity": round(embed_sim, 4),
    }


def text_to_html(text: str) -> str:
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
    doc_id: str,
    url: str,
    original_html: str,
    original_text: str,
    test_queries: List[str],
    method_texts: Dict[str, Dict[str, str]],  # {method: {"text": ..., "html": ...}}
    evaluator: AgentGEOOptimizer,
) -> Dict[str, Any]:
    result = {"doc_id": doc_id, "url": url, "timestamp": datetime.now().isoformat()}

    # 1. Baseline eval — ONE time
    logger.info(f"  [{doc_id}] Evaluating baseline ({len(test_queries)} queries)...")
    baseline_eval = await evaluator.evaluate_page_async(
        raw_html=original_html, test_queries=test_queries, url=url
    )
    baseline_rate = baseline_eval.get("ratio", 0.0)
    baseline_detail = extract_eval_geo_scores(baseline_eval, test_queries)
    result["baseline_test_citation_rate"] = baseline_rate
    result["baseline_test_avg_geo_score"] = baseline_detail["avg_geo_score"]
    result["baseline_test_per_query"] = baseline_detail["per_query"]
    logger.info(f"  [{doc_id}] Baseline: citation={baseline_rate:.1%}, geo={baseline_detail['avg_geo_score']:.4f}")

    # 2. Each method's optimized eval
    for method, content in method_texts.items():
        opt_text = content.get("text", "")
        opt_html = content.get("html", "")
        if not opt_text and not opt_html:
            continue
        try:
            # Similarity
            if original_text and opt_text:
                sim = compute_similarity(original_text, opt_text)
                result[f"{method}_similarity"] = sim
                result[f"{method}_length_ratio"] = round(len(opt_text) / len(original_text), 4)

            # Determine eval HTML
            eval_html = opt_html if opt_html and opt_html.strip().startswith('<') else text_to_html(opt_text or opt_html)

            logger.info(f"  [{doc_id}] Evaluating {method}...")
            opt_eval = await evaluator.evaluate_page_async(
                raw_html=eval_html, test_queries=test_queries, url=url
            )
            opt_rate = opt_eval.get("ratio", 0.0)
            opt_detail = extract_eval_geo_scores(opt_eval, test_queries)

            result[f"{method}_optimized_test_citation_rate"] = opt_rate
            result[f"{method}_optimized_test_avg_geo_score"] = opt_detail["avg_geo_score"]
            result[f"{method}_optimized_test_per_query"] = opt_detail["per_query"]
            result[f"{method}_delta_test_citation_rate"] = round(opt_rate - baseline_rate, 4)
            result[f"{method}_delta_test_avg_geo_score"] = round(
                opt_detail["avg_geo_score"] - baseline_detail["avg_geo_score"], 4
            )
            logger.info(f"  [{doc_id}] {method}: citation={opt_rate:.1%} (Δ={opt_rate - baseline_rate:+.1%}), "
                         f"geo={opt_detail['avg_geo_score']:.4f}")
        except Exception as e:
            logger.error(f"  [{doc_id}] {method} failed: {e}", exc_info=True)
            result[f"{method}_error"] = str(e)

    return result


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentgeo-results", required=True, help="AgentGEO optimization_results JSON")
    parser.add_argument("--baselines-results", required=True, help="Baselines optimization_results JSON")
    parser.add_argument("--methods", required=True, help="Comma-separated methods to evaluate")
    parser.add_argument("--data", default="data/input.parquet")
    parser.add_argument("--agentgeo-config", default="optimization_config_gpt4.1_baselines.yaml")
    parser.add_argument("--output-dir", default="outputs_gpt4.1_unified_eval")
    parser.add_argument("--doc-limit", type=int)
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    logger.info(f"Methods: {methods}")

    # Load results
    ag_results = {r["doc_id"]: r for r in json.load(open(args.agentgeo_results))}
    bl_results = {r["doc_id"]: r for r in json.load(open(args.baselines_results))}
    all_doc_ids = sorted(set(ag_results) | set(bl_results))
    logger.info(f"AgentGEO docs: {len(ag_results)}, Baselines docs: {len(bl_results)}, Union: {len(all_doc_ids)}")

    if args.doc_limit:
        all_doc_ids = all_doc_ids[:args.doc_limit]

    # Load original data
    import pandas as pd
    df = pd.read_parquet(args.data)
    doc_map = {row["doc_id"]: row for _, row in df.iterrows()}

    # Evaluator
    import yaml
    full_config = yaml.safe_load(open(args.agentgeo_config))
    evaluator = AgentGEOOptimizer(**full_config.get("agentgeo", {}))

    # Checkpoints
    output_dir = Path(args.output_dir)
    cp_dir = output_dir / "documents"
    cp_dir.mkdir(parents=True, exist_ok=True)
    completed = set()
    for f in cp_dir.glob("*.json"):
        try:
            completed.add(json.loads(f.read_text())["doc_id"])
        except Exception:
            pass
    if completed:
        logger.info(f"Resuming: {len(completed)} done")

    results = []
    for idx, doc_id in enumerate(all_doc_ids):
        if doc_id in completed:
            cached = json.loads((cp_dir / f"{doc_id}.json").read_text())
            results.append(cached)
            logger.info(f"[{idx+1}/{len(all_doc_ids)}] Skipped (cached): {doc_id}")
            continue

        orig = doc_map.get(doc_id)
        if orig is None:
            logger.warning(f"[{idx+1}/{len(all_doc_ids)}] {doc_id} not in data, skip")
            continue

        test_queries = orig.get("test_queries", [])
        if hasattr(test_queries, "tolist"):
            test_queries = test_queries.tolist()

        # Extract original text
        try:
            from trafilatura import extract as tf_extract
            original_text = tf_extract(orig["raw_html"]) or ""
        except Exception:
            original_text = ""

        # Gather method texts
        method_texts = {}
        for m in methods:
            if m == "agentgeo":
                src = ag_results.get(doc_id, {})
            else:
                src = bl_results.get(doc_id, {})
            t = src.get(f"{m}_text", "")
            h = src.get(f"{m}_html", "")
            if t or h:
                method_texts[m] = {"text": t, "html": h}

        if not method_texts:
            logger.warning(f"[{idx+1}/{len(all_doc_ids)}] {doc_id}: no method texts, skip")
            continue

        logger.info(f"\n[{idx+1}/{len(all_doc_ids)}] Evaluating {doc_id} ({list(method_texts.keys())})")
        try:
            r = await evaluate_document(
                doc_id, orig.get("url", ""), orig["raw_html"], original_text,
                test_queries, method_texts, evaluator
            )
            results.append(r)
            with open(cp_dir / f"{doc_id}.json", "w") as f:
                json.dump(r, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[{idx+1}/{len(all_doc_ids)}] {doc_id} failed: {e}", exc_info=True)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(output_dir / f"unified_eval_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Done! {len(results)} docs saved to {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
