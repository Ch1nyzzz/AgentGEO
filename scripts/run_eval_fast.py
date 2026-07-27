#!/usr/bin/env python3
"""Fast 3-way unified eval (省时版).

- baseline raw + AgentGEO: reuse existing answer & GEO from old unified_eval,
  only compute Clarity + Insightfulness via LLM judge.
- AutoGEO: run evaluator (get answer + documents + GEO), then compute full GEU
  (Precision, Recall, Clarity, Insightfulness).
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
# GEU prompts
# ============================================================

CLAIM_EXTRACTOR_PROMPT = """You are an information extraction expert.
Given a report, extract all distinct factual claims. For each claim, identify the source indices it cites (e.g., [1], [2], [3]). Indices are **1-based**.

Return a JSON object with a "claims" list, where each entry has:
- "claim_id": sequential int from 1.
- "claim": concise sentence.
- "source_indices": list of 1-based int indices, or [] if none.

Report:
\"\"\"
{answer}
\"\"\"

Return only the JSON."""

CITATION_CHECKER_PROMPT = """You are a fact-checker. Evaluate if a "Statement" is supported by the "Source Text".
Respond in JSON with "support" ('full_support', 'partial_support', or 'no_support') and "justification".

Statement: "{claim}"

Source:
\"\"\"
{document_content}
\"\"\"

JSON:"""

QUALITY_EVALUATOR_PROMPT = """You are a strict expert evaluator. Rate the "Answer" to "Question" on **{criterion_name}** only.

**{criterion_name}**: {criterion_description}

Question: {question}
Answer: {answer}

JSON: {{"rating": int 0-10, "justification": str}}. Be strict — high scores only for outstanding answers.
JSON:"""

QUALITY_CRITERIA = {
    "Clarity": "Assess how clearly and rigorously the answer is structured. High-quality responses are like in-depth reports with distinct, non-overlapping points and strong logical flow. Penalize redundancy, ambiguity, and filler.",
    "Insightfulness": "Assess originality and value. Excellent reports go beyond common knowledge, offering original synthesis or thought-provoking connections.",
}

SUPPORT_SCORE = {"full_support": 1.0, "partial_support": 0.5, "no_support": 0.0}


class GEUClient:
    def __init__(self, model: str = "gpt-4o-mini", max_concurrency: int = 64):
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
                        logger.warning(f"GEU call failed: {e}")
                        return None
                    await asyncio.sleep(2 ** attempt)
        return None

    async def quality_dimension(self, query: str, answer: str, criterion: str) -> Optional[float]:
        if not answer or not answer.strip():
            return None
        result = await self.call_json(QUALITY_EVALUATOR_PROMPT.format(
            criterion_name=criterion,
            criterion_description=QUALITY_CRITERIA[criterion],
            question=query, answer=answer,
        ))
        if not result or "rating" not in result:
            return None
        try:
            return float(result["rating"]) / 10.0
        except Exception:
            return None

    async def citation_quality(self, answer: str, documents: List[str]) -> Tuple[Optional[float], Optional[float]]:
        if not answer or not answer.strip():
            return None, None
        if not re.search(r"\[\d+\]", answer):
            return 0.0, 0.0

        claims_data = await self.call_json(CLAIM_EXTRACTOR_PROMPT.format(answer=answer))
        if not claims_data or not claims_data.get("claims"):
            return None, None

        total = len(claims_data["claims"])
        cited = [c for c in claims_data["claims"] if c.get("source_indices")]
        recall = len(cited) / total if total else 0.0
        if not cited:
            return 0.0, recall

        async def score_claim(claim):
            scores = []
            for idx in claim.get("source_indices", []):
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

    async def evaluate_clarity_insight(self, query: str, answer: str) -> Dict[str, Optional[float]]:
        clarity, insight = await asyncio.gather(
            self.quality_dimension(query, answer, "Clarity"),
            self.quality_dimension(query, answer, "Insightfulness"),
        )
        return {"clarity": clarity, "insightfulness": insight}

    async def evaluate_full(self, query: str, answer: str, documents: List[str]) -> Dict[str, Optional[float]]:
        cite_task = asyncio.create_task(self.citation_quality(answer, documents))
        ci_task = asyncio.create_task(self.evaluate_clarity_insight(query, answer))
        (precision, recall), ci = await asyncio.gather(cite_task, ci_task)
        return {"precision": precision, "recall": recall, **ci}


# ============================================================
# Aggregation
# ============================================================

def avg(values):
    vs = [v for v in values if v is not None]
    return round(sum(vs) / len(vs), 4) if vs else None


def text_to_html(text: str) -> str:
    paragraphs = "".join(f"<p>{line}</p>" for line in text.split("\n") if line.strip())
    return f"<html><body>{paragraphs}</body></html>"


def aggregate(per_query: Dict[str, Dict]) -> Dict[str, Any]:
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
# Per-method handlers
# ============================================================

async def reuse_method(label: str, per_query_old: Dict[str, Dict], geu: GEUClient) -> Dict[str, Any]:
    """Reuse old answer + GEO; only compute Clarity + Insightfulness."""
    if not per_query_old:
        return {}
    queries = list(per_query_old.keys())
    tasks = [
        geu.evaluate_clarity_insight(q, per_query_old[q].get("answer", ""))
        for q in queries
    ]
    ci_results = await asyncio.gather(*tasks)
    pq = {}
    for q, ci in zip(queries, ci_results):
        old = per_query_old[q]
        pq[q] = {
            "is_cited": old.get("is_cited", False),
            "answer": old.get("answer", ""),
            "geo_score": old.get("geo_score"),
            "geu_score": {
                "precision": None,
                "recall": None,
                "clarity": ci.get("clarity"),
                "insightfulness": ci.get("insightfulness"),
            },
        }
    return {"per_query": pq, **aggregate(pq)}


async def full_method(label: str, eval_html: str, test_queries: List[str], url: str,
                       evaluator: AgentGEOOptimizer, geu: GEUClient) -> Dict[str, Any]:
    """Full evaluator pass + complete GEU."""
    page_eval = await evaluator.evaluate_page_async(
        raw_html=eval_html, test_queries=test_queries, url=url
    )
    detailed = page_eval.get("detailed", {})
    queries = list(detailed.keys())
    tasks = []
    for q in queries:
        d = detailed[q]
        tasks.append(geu.evaluate_full(q, d.get("answer", ""), d.get("documents", [])))
    geu_results = await asyncio.gather(*tasks)
    pq = {}
    for q, gres in zip(queries, geu_results):
        d = detailed[q]
        pq[q] = {
            "is_cited": d.get("is_cited"),
            "answer": d.get("answer", ""),
            "geo_score": d.get("geo_score"),
            "geu_score": gres,
        }
    return {"per_query": pq, **aggregate(pq)}


async def evaluate_document(
    doc_id: str, url: str, raw_html: str, test_queries: List[str],
    autogeo_html: Optional[str],
    old_baseline_pq: Dict[str, Dict], old_agentgeo_pq: Dict[str, Dict],
    evaluator: AgentGEOOptimizer, geu: GEUClient,
) -> Dict[str, Any]:
    result = {"doc_id": doc_id, "url": url, "timestamp": datetime.now().isoformat()}

    # Run methods concurrently
    tasks = {}
    if old_baseline_pq:
        tasks["baseline"] = reuse_method("baseline", old_baseline_pq, geu)
    if old_agentgeo_pq:
        tasks["agentgeo"] = reuse_method("agentgeo", old_agentgeo_pq, geu)
    if autogeo_html:
        eval_html = autogeo_html if autogeo_html.strip().startswith("<") else text_to_html(autogeo_html)
        tasks["autogeo"] = full_method("autogeo", eval_html, test_queries, url, evaluator, geu)

    method_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for method, mres in zip(tasks.keys(), method_results):
        if isinstance(mres, Exception):
            logger.error(f"  [{doc_id}] {method} failed: {mres}")
            result[f"{method}_error"] = str(mres)
            continue
        for k, v in mres.items():
            result[f"{method}_{k}"] = v
    return result


# ============================================================
# Main
# ============================================================

def load_old_unified(path: str) -> Dict[str, Dict[str, Any]]:
    """doc_id → {'baseline': per_query, 'agentgeo': per_query}"""
    if not Path(path).exists():
        return {}
    data = json.load(open(path))
    out = {}
    for d in data:
        doc_id = d.get("doc_id")
        if not doc_id:
            continue
        out[doc_id] = {
            "baseline": d.get("baseline_test_per_query", {}) or {},
            "agentgeo": d.get("agentgeo_optimized_test_per_query", {}) or {},
        }
    return out


def load_autogeo_html(path: str) -> Dict[str, str]:
    if not Path(path).exists():
        return {}
    data = json.load(open(path))
    out = {}
    for r in data:
        doc_id = r.get("doc_id")
        if not doc_id:
            continue
        html = r.get("autogeo_html") or r.get("autogeo_text")
        if html:
            out[doc_id] = html
    return out


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/input.parquet")
    parser.add_argument("--old-unified", default="outputs_gpt4.1_unified_eval/unified_eval_20260328_190810.json",
                        help="Old unified_eval to reuse baseline+agentgeo answers")
    parser.add_argument("--autogeo-results", required=True)
    parser.add_argument("--agentgeo-config", default="optimization_config_gpt4.1.yaml")
    parser.add_argument("--output-dir", default="outputs_gpt4.1_three_way_eval_fast")
    parser.add_argument("--doc-limit", type=int, default=50)
    parser.add_argument("--doc-concurrency", type=int, default=32)
    parser.add_argument("--geu-model", default="gpt-4o-mini")
    parser.add_argument("--geu-concurrency", type=int, default=64)
    args = parser.parse_args()

    import pandas as pd
    df = pd.read_parquet(args.data)
    if args.doc_limit:
        df = df.head(args.doc_limit)
    logger.info(f"Loaded {len(df)} documents")

    old_map = load_old_unified(args.old_unified)
    autogeo_map = load_autogeo_html(args.autogeo_results)
    logger.info(f"Old unified: {len(old_map)} docs | AutoGEO: {len(autogeo_map)} docs")

    import yaml
    with open(args.agentgeo_config) as f:
        cfg = yaml.safe_load(f)
    evaluator = AgentGEOOptimizer(**cfg.get("agentgeo", {}))
    geu = GEUClient(model=args.geu_model, max_concurrency=args.geu_concurrency)

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
        logger.info(f"Resume: {len(completed)} done")

    sem = asyncio.Semaphore(args.doc_concurrency)
    results: List[Dict] = []
    lock = asyncio.Lock()

    async def process_one(idx, row):
        doc_id = row["doc_id"]
        if doc_id in completed:
            cached = json.loads((ckpt_dir / f"{doc_id}.json").read_text())
            async with lock:
                results.append(cached)
            return

        async with sem:
            tq = row.get("test_queries", [])
            if hasattr(tq, "tolist"):
                tq = tq.tolist()
            tq = list(tq)
            old = old_map.get(doc_id, {})
            autogeo_html = autogeo_map.get(doc_id)
            try:
                logger.info(f"[{idx+1}/{len(df)}] {doc_id}")
                res = await evaluate_document(
                    doc_id, row.get("url", ""), row["raw_html"], tq,
                    autogeo_html, old.get("baseline", {}), old.get("agentgeo", {}),
                    evaluator, geu,
                )
                with open(ckpt_dir / f"{doc_id}.json", "w") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                async with lock:
                    results.append(res)
                bc = res.get("baseline_citation_rate")
                ac = res.get("agentgeo_citation_rate")
                tc = res.get("autogeo_citation_rate")
                logger.info(f"  [{doc_id}] done | base={bc:.2f} agent={ac if ac is not None else 'NA'} auto={tc if tc is not None else 'NA'}")
            except Exception as e:
                logger.error(f"  [{doc_id}] failed: {e}", exc_info=True)

    tasks = [asyncio.create_task(process_one(i, row)) for i, row in enumerate(df.to_dict("records"))]
    await asyncio.gather(*tasks)

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_path = output_dir / f"unified_eval_fast_{ts}.json"
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"=== Done | {len(results)} docs → {final_path}")


if __name__ == "__main__":
    asyncio.run(main())
