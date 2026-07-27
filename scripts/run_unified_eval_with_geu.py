#!/usr/bin/env python3
"""Unified evaluator for baseline raw / AutoGEO / AgentGEO on GPT-4.1.

Computes per-query metrics:
- Citation rate
- GEO score (word, position, wordpos, overall)
- GEU citation quality (precision, recall) - LLM judge
- GEU quality dimensions (Clarity, Insightfulness) - LLM judge
"""
import argparse
import asyncio
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

os.environ.pop("all_proxy", None)
os.environ.pop("ALL_PROXY", None)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
load_dotenv(REPO_ROOT / ".env")

from optimizers import AgentGEOOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


# ============================================================
# GEU Metric Computation (local, async)
# ============================================================

CLAIM_EXTRACTOR_PROMPT = """You are an information extraction expert.
Given a report, extract all distinct factual claims. For each claim, identify the source indices it cites (e.g., [1], [2], [3]). Indices are **1-based** (the first source is [1], not [0]).

Return a JSON object with a "claims" list, where each entry has:
- "claim_id": A sequential integer starting from 1.
- "claim": A concise, complete sentence of the claim.
- "source_indices": A list of 1-based integer indices cited for this claim (e.g., [1] or [1, 2]). Empty list [] if no source cited.

**IMPORTANT**: Only factual claims; indices must be integers from citations like `[1]` or `[1][2]`.

Report:
\"\"\"
{answer}
\"\"\"

Return only the JSON object."""

CITATION_CHECKER_PROMPT = """You are a meticulous fact-checker. Evaluate if a "Statement" is supported by the "Source Text".
Respond in JSON with "support" ('full_support', 'partial_support', or 'no_support') and "justification".

- full_support: All info in statement is directly supported.
- partial_support: Some parts supported, others not.
- no_support: Source does not support statement.

Statement: "{claim}"

Source Text:
\"\"\"
{document_content}
\"\"\"

JSON response:"""

QUALITY_EVALUATOR_PROMPT = """You are a strict expert evaluator. Assess the quality of an "Answer" to a "Question" based ONLY on the criterion of **{criterion_name}**.

**Criterion: {criterion_name}**
{criterion_description}

**Question:** {question}

**Answer:** {answer}

Provide JSON with:
1. "rating": integer 0 (poor) to 10 (excellent).
2. "justification": brief justification.

Do not be generous. High scores only for outstanding answers.

JSON response:"""

QUALITY_CRITERIA = {
    "Clarity": "Assess how clearly and rigorously the answer is structured. High-quality responses are like in-depth reports with distinct, non-overlapping points and strong logical flow. Penalize redundancy, ambiguity, and filler.",
    "Insightfulness": "Assess originality and value. Excellent reports go beyond common knowledge, offering original synthesis or thought-provoking connections. Recommendations must be concrete and actionable.",
}

SUPPORT_SCORE = {"full_support": 1.0, "partial_support": 0.5, "no_support": 0.0}


class GEUClient:
    def __init__(self, model: str = "gpt-4o-mini", max_concurrency: int = 16):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def call_json(self, prompt: str, retries: int = 3) -> Optional[Dict]:
        async with self.semaphore:
            for attempt in range(retries):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                    )
                    return json.loads(resp.choices[0].message.content)
                except Exception as e:
                    if attempt == retries - 1:
                        logger.warning(f"GEU LLM call failed: {e}")
                        return None
                    await asyncio.sleep(2 ** attempt)
        return None

    async def citation_quality(self, answer: str, documents: List[str]) -> Tuple[Optional[float], Optional[float]]:
        """Return (precision, recall) using 1-based indices."""
        if not answer or not answer.strip():
            return None, None

        # Skip extraction if answer has no citation markers at all
        if not re.search(r"\[\d+\]", answer):
            return 0.0, 0.0

        claims_data = await self.call_json(CLAIM_EXTRACTOR_PROMPT.format(answer=answer))
        if not claims_data or "claims" not in claims_data or not claims_data["claims"]:
            return None, None

        total = len(claims_data["claims"])
        cited = [c for c in claims_data["claims"] if c.get("source_indices")]
        recall = len(cited) / total if total else 0.0
        if not cited:
            return (0.0, recall)

        # For each cited claim, score against each cited source (max across sources)
        async def score_claim(claim):
            scores = []
            for idx in claim.get("source_indices", []):
                # 1-based → 0-based document index
                d_idx = idx - 1
                if 0 <= d_idx < len(documents) and documents[d_idx]:
                    res = await self.call_json(CITATION_CHECKER_PROMPT.format(
                        claim=claim.get("claim", ""),
                        document_content=documents[d_idx][:4000],
                    ))
                    if res and "support" in res:
                        scores.append(SUPPORT_SCORE.get(res["support"], 0.0))
            return max(scores) if scores else 0.0

        claim_scores = await asyncio.gather(*[score_claim(c) for c in cited])
        precision = sum(claim_scores) / len(claim_scores) if claim_scores else 0.0
        return precision, recall

    async def quality_dimension(self, query: str, answer: str, criterion: str) -> Optional[float]:
        if not answer or not answer.strip():
            return None
        result = await self.call_json(QUALITY_EVALUATOR_PROMPT.format(
            criterion_name=criterion,
            criterion_description=QUALITY_CRITERIA[criterion],
            question=query,
            answer=answer,
        ))
        if not result or "rating" not in result:
            return None
        try:
            return float(result["rating"]) / 10.0
        except Exception:
            return None

    async def evaluate(self, query: str, answer: str, documents: List[str]) -> Dict[str, Optional[float]]:
        """Return precision, recall, clarity, insightfulness."""
        cite_task = asyncio.create_task(self.citation_quality(answer, documents))
        clarity_task = asyncio.create_task(self.quality_dimension(query, answer, "Clarity"))
        insight_task = asyncio.create_task(self.quality_dimension(query, answer, "Insightfulness"))
        (precision, recall), clarity, insight = await asyncio.gather(cite_task, clarity_task, insight_task)
        return {
            "precision": precision,
            "recall": recall,
            "clarity": clarity,
            "insightfulness": insight,
        }


# ============================================================
# Aggregation utilities
# ============================================================

def avg(values):
    vs = [v for v in values if v is not None]
    return round(sum(vs) / len(vs), 4) if vs else None


def text_to_html(text: str) -> str:
    paragraphs = "".join(f"<p>{line}</p>" for line in text.split("\n") if line.strip())
    return f"<html><body>{paragraphs}</body></html>"


def aggregate_per_query(per_query: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute aggregate GEO + GEU averages from per-query details."""
    word, pos, wp, overall = [], [], [], []
    prec, rec, clr, ins = [], [], [], []
    cited = 0
    for q, d in per_query.items():
        if d.get("is_cited"):
            cited += 1
        geo = d.get("geo_score") or {}
        if geo:
            word.append(geo.get("word"))
            pos.append(geo.get("position"))
            wp.append(geo.get("wordpos"))
            overall.append(geo.get("overall"))
        geu = d.get("geu_score") or {}
        if geu:
            prec.append(geu.get("precision"))
            rec.append(geu.get("recall"))
            clr.append(geu.get("clarity"))
            ins.append(geu.get("insightfulness"))
    n = len(per_query)
    return {
        "citation_rate": cited / n if n else 0.0,
        "avg_word": avg(word),
        "avg_position": avg(pos),
        "avg_wordpos": avg(wp),
        "avg_overall": avg(overall),
        "avg_precision": avg(prec),
        "avg_recall": avg(rec),
        "avg_clarity": avg(clr),
        "avg_insightfulness": avg(ins),
    }


# ============================================================
# Per-document evaluation
# ============================================================

async def eval_one_method(
    label: str,
    eval_html: str,
    test_queries: List[str],
    url: str,
    evaluator: AgentGEOOptimizer,
    geu: GEUClient,
    skip_geu: bool = False,
) -> Dict[str, Any]:
    """Evaluate one (document, method) pair → per-query GEO+GEU + aggregates."""
    page_eval = await evaluator.evaluate_page_async(
        raw_html=eval_html, test_queries=test_queries, url=url
    )
    detailed = page_eval.get("detailed", {})

    # Compute GEU per query in parallel (only when answer non-empty)
    if not skip_geu:
        geu_tasks = []
        ordered_queries = []
        for q, d in detailed.items():
            ans = d.get("answer", "")
            docs = d.get("documents", [])
            if ans and docs:
                geu_tasks.append(geu.evaluate(q, ans, docs))
                ordered_queries.append(q)
        if geu_tasks:
            geu_results = await asyncio.gather(*geu_tasks)
            for q, gres in zip(ordered_queries, geu_results):
                detailed[q]["geu_score"] = gres

    # Strip large per-query 'documents' to save disk space
    cleaned = {}
    for q, d in detailed.items():
        cleaned[q] = {k: v for k, v in d.items() if k != "documents"}

    agg = aggregate_per_query(cleaned)
    return {"per_query": cleaned, **agg}


async def evaluate_document(
    doc_id: str,
    url: str,
    raw_html: str,
    test_queries: List[str],
    method_html_map: Dict[str, str],
    evaluator: AgentGEOOptimizer,
    geu: GEUClient,
) -> Dict[str, Any]:
    result = {"doc_id": doc_id, "url": url, "timestamp": datetime.now().isoformat()}

    # Baseline (raw) — evaluate once, GEU included
    logger.info(f"  [{doc_id}] eval baseline raw")
    baseline = await eval_one_method("baseline", raw_html, test_queries, url, evaluator, geu)
    for k, v in baseline.items():
        result[f"baseline_{k}"] = v

    # Each optimization method
    for method, opt_html in method_html_map.items():
        if not opt_html:
            logger.warning(f"  [{doc_id}] {method}: missing optimized HTML, skipping")
            continue
        eval_html = opt_html if opt_html.strip().startswith("<") else text_to_html(opt_html)
        logger.info(f"  [{doc_id}] eval {method}")
        try:
            mres = await eval_one_method(method, eval_html, test_queries, url, evaluator, geu)
            for k, v in mres.items():
                result[f"{method}_{k}"] = v
        except Exception as e:
            logger.error(f"  [{doc_id}] {method} failed: {e}", exc_info=True)
            result[f"{method}_error"] = str(e)

    return result


def load_method_results(path: Optional[str], text_key: str) -> Dict[str, str]:
    """Load doc_id → optimized_html (preferred) or text from a results JSON file."""
    if not path or not Path(path).exists():
        return {}
    with open(path) as f:
        results = json.load(f)
    out = {}
    for r in results:
        doc_id = r.get("doc_id")
        if not doc_id:
            continue
        html = r.get(f"{text_key}_html") or r.get(f"{text_key}_text")
        if html:
            out[doc_id] = html
    return out


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/input.parquet")
    parser.add_argument("--agentgeo-results", required=True, help="path to AgentGEO optimization_results JSON")
    parser.add_argument("--autogeo-results", required=True, help="path to AutoGEO optimization_results JSON")
    parser.add_argument("--config", default="geo_agent/config_gpt4.1.yaml")
    parser.add_argument("--agentgeo-config", default="optimization_config_gpt4.1.yaml")
    parser.add_argument("--output-dir", default="outputs_gpt4.1_three_way_eval")
    parser.add_argument("--doc-limit", type=int, help="limit docs (for smoke test)")
    parser.add_argument("--doc-concurrency", type=int, default=2)
    parser.add_argument("--geu-model", default="gpt-4o-mini", help="model for GEU LLM judge")
    parser.add_argument("--geu-concurrency", type=int, default=24, help="max concurrent GEU LLM calls")
    args = parser.parse_args()

    # Load original data
    import pandas as pd
    df = pd.read_parquet(args.data)
    if args.doc_limit:
        df = df.head(args.doc_limit)
    logger.info(f"Loaded {len(df)} documents from {args.data}")

    # Load method outputs
    agentgeo_map = load_method_results(args.agentgeo_results, "agentgeo")
    autogeo_map = load_method_results(args.autogeo_results, "autogeo")
    logger.info(f"AgentGEO: {len(agentgeo_map)} docs | AutoGEO: {len(autogeo_map)} docs")

    # Build evaluator + GEU client
    import yaml
    with open(args.agentgeo_config) as f:
        full_config = yaml.safe_load(f)
    agentgeo_cfg = full_config.get("agentgeo", {})
    evaluator = AgentGEOOptimizer(**agentgeo_cfg)
    geu = GEUClient(model=args.geu_model, max_concurrency=args.geu_concurrency)

    # Output setup
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "documents"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    completed = set()
    for cp in ckpt_dir.glob("*.json"):
        try:
            d = json.loads(cp.read_text())
            if "doc_id" in d:
                completed.add(d["doc_id"])
        except Exception:
            pass
    if completed:
        logger.info(f"Resume: {len(completed)} docs already done")

    sem = asyncio.Semaphore(args.doc_concurrency)
    results: List[Dict] = []
    lock = asyncio.Lock()

    async def process_one(idx: int, row):
        doc_id = row["doc_id"]
        if doc_id in completed:
            cached = json.loads((ckpt_dir / f"{doc_id}.json").read_text())
            async with lock:
                results.append(cached)
            logger.info(f"[{idx+1}/{len(df)}] cached: {doc_id}")
            return

        async with sem:
            tq = row.get("test_queries", [])
            if hasattr(tq, "tolist"):
                tq = tq.tolist()
            tq = list(tq)

            method_map = {}
            if doc_id in autogeo_map:
                method_map["autogeo"] = autogeo_map[doc_id]
            if doc_id in agentgeo_map:
                method_map["agentgeo"] = agentgeo_map[doc_id]

            try:
                logger.info(f"\n[{idx+1}/{len(df)}] {doc_id} (methods: {list(method_map)})")
                res = await evaluate_document(
                    doc_id=doc_id, url=row.get("url", ""), raw_html=row["raw_html"],
                    test_queries=tq, method_html_map=method_map,
                    evaluator=evaluator, geu=geu,
                )
                with open(ckpt_dir / f"{doc_id}.json", "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                async with lock:
                    results.append(res)
                logger.info(f"  [{doc_id}] done | base_cite={res.get('baseline_citation_rate'):.2f} "
                            f"agent_cite={res.get('agentgeo_citation_rate', 'N/A')} "
                            f"auto_cite={res.get('autogeo_citation_rate', 'N/A')}")
            except Exception as e:
                logger.error(f"  [{doc_id}] failed: {e}", exc_info=True)

    tasks = [asyncio.create_task(process_one(i, row)) for i, row in enumerate(df.to_dict("records"))]
    await asyncio.gather(*tasks)

    # Final dump
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = output_dir / f"unified_eval_{ts}.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"\n=== Done | {len(results)} docs → {final_path}")


if __name__ == "__main__":
    asyncio.run(main())
