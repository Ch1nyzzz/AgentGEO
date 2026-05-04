"""
Test citation rates using GPT-5 with URL browsing.

For selected docs/queries from sample_50.parquet, run two tests:
  1. Give GPT the original page URL + 5 competitor URLs → check citation
  2. Give GPT the optimized page URL + 5 competitor URLs → check citation

GPT-5 browses the URLs itself via web_search_preview tool (Responses API).
"""

import asyncio
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import nltk
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autogeo.evaluation.metrics import (
    extract_citations_new,
    impression_pos_count_simple,
    impression_word_count_simple,
    impression_wordpos_count_simple,
)

load_dotenv()

# -- Config --
MODEL = "gpt-4.1-mini"
CONCURRENCY = 32
NUM_COMPETITORS = 10
SAMPLE_PATH = Path("sample_50.parquet")
OUTPUT_PATH = Path("outputs/gpt5mini_citation_results.json")
RUN_OUTPUT_DIR = OUTPUT_PATH.with_suffix("")
CHECKPOINT_PATH = RUN_OUTPUT_DIR / "checkpoint.json"
JSONL_PATH = RUN_OUTPUT_DIR / "records.jsonl"
DOC_OUTPUT_DIR = RUN_OUTPUT_DIR / "by_doc"
PROBE_HITS_PATH = RUN_OUTPUT_DIR / "probe_original_hits.json"
BASE_URL = "https://ch1nyzzz.github.io/AgentGEO"


SYSTEM_PROMPT = (
    "You are a research assistant. The user will provide a question and a numbered list of source URLs. "
    "You MUST read all provided sources and answer using only information from those sources.\n\n"

    "STRICT RULES:\n"
    "1. ONLY use information from the provided sources. Do NOT use your own knowledge or any external websites. Do not "
"use source URLs, file names, paths, slugs, or webpage link names to judge relevance; rely only on source content.\n"

    "2. Before writing the answer, identify the source-based information points needed to answer the question. "
    "For each information point, choose the smallest set of sources that directly supports it.\n"

    "3. Do not cite a source only because it is relevant, similar, or also mentions the topic. "
    "Cite a source only when it is necessary to support a specific information point in the sentence.\n"

    "4. If multiple sources support the same information point, cite only the source or sources that provide the clearest and most direct support. "
    "Do not cite all matching sources.\n"

    "5. Sentences that only summarize, organize, compare, or generalize from already cited information do not need citations.\n"

    "6. Place citations immediately after the supported information point using [index] format.\n"

    "7. When citing multiple sources for the same information point, use [1][2][3] format, not [1, 2, 3].\n"

    "8. Do NOT cite URLs directly. Do NOT use markdown links. ONLY use [index] format.\n"

    "9. If a source is not necessary for supporting a specific information point, do not cite it.\n"

    "10. The answer should be accurate, concise, and written in an unbiased journalistic tone.\n"
    
    "11. Citation necessity test: before citing a source, check whether the sentence would still be fully supported without that source. "
"If the answer is yes, do not cite that source."
)
# SYSTEM_PROMPT = (
#     "You are a research assistant. The user will provide a question and a numbered list of source URLs. "
#     "You MUST visit each URL to read its content, then write an answer based ONLY on information found in those sources.\n\n"
#     "STRICT RULES:\n"
#     "1. ONLY use information from the provided sources. Do NOT use your own knowledge or any external websites.\n"
#     "2. Add an inline citation only when the sentence contains an information point that appears in one or more provided sources.\n"

#     "3. Do not add citations to sentences that only summarize, organize, compare, or generalize from already cited information.\n"

#     "4. If a sentence contains both a source-based information point and a summary or interpretation, cite only when the source-based part is essential to the sentence.\n"

#     "5. Place the citation immediately after the supported information point using [index] format, where index is the Source number, e.g. [1], [2], [3].\n"
    
#     "6. When citing multiple sources for the same information point, use [1][2][3] format, NOT [1, 2, 3], (Source 1), or any other format.\n"

#     "7. Do NOT cite URLs directly. Do NOT use markdown links like [text](url). ONLY use [index] format.\n"

#     "8. If a source is not relevant to the question, do not cite it.\n"

#     "9. Do not include claims that cannot be traced to the provided sources.\n"

#     "10. The answer should be accurate, concise, and written in an unbiased journalistic tone."
# )
    # "2. Cite only factual claims that are directly supported by specific, distinctive evidence in the provided sources using [index] format, "
    # "where index is the Source number (e.g. [1], [2], [3]).\n"
    # "3. When citing multiple sources, use [1][2][3] format, NOT [1, 2, 3] or (Source 1) or any other format.\n"
    # "4. Do NOT cite URLs directly. Do NOT use markdown links like [text](url). ONLY use [index] format.\n"
    # "5. If a source is not relevant to the question, do not cite it.\n"
    # "6. The answer should be accurate, concise, and written in an unbiased journalistic tone."
# )


def compute_geo_scores(answer: str, target_idx: int, num_sources: int) -> dict:
    """
    Compute target-source Word/Pos/WordPos scores.

    Follows autogeo/evaluation/evaluator.py:
      citations = extract_citations_new(answer)
      wordpos = impression_wordpos_count_simple(citations)[target_id]
      word = impression_word_count_simple(citations)[target_id]
      pos = impression_pos_count_simple(citations)[target_id]

    `target_idx` is 1-based because citations use [1], [2], ...
    """
    target_id = target_idx - 1
    try:
        citations = extract_citations_new(answer)
    except LookupError:
        # Newer NLTK releases may require punkt_tab in addition to punkt.
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        citations = extract_citations_new(answer)
    return {
        "wordpos": impression_wordpos_count_simple(citations, n=num_sources)[target_id],
        "word": impression_word_count_simple(citations, n=num_sources)[target_id],
        "pos": impression_pos_count_simple(citations, n=num_sources)[target_id],
    }


def check_citation(
    answer: str,
    target_idx: int,
    target_url: str = "",
    num_sources: int = 0,
) -> dict:
    """Check if target is cited in the answer."""
    cited_by_index = f"[{target_idx}]" in answer
    cited_by_url = bool(target_url) and target_url in answer
    is_cited = cited_by_index or cited_by_url
    cited_indices = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    num_sources = num_sources or max(cited_indices + [target_idx])
    geo_scores = compute_geo_scores(answer, target_idx, num_sources)
    return {
        "cr": 1.0 if is_cited else 0.0,
        **geo_scores,
        "is_cited": is_cited,
        "cited_by_index": cited_by_index,
        "cited_by_url": cited_by_url,
        "cited_indices": cited_indices,
        "target_idx": target_idx,
    }


def build_url_sources(
    target_url: str,
    competitor_urls: list[str],
    target_position: int | None = None,
    source_order: list[dict] | None = None,
) -> tuple[str, int]:
    """
    Build source list with target URL inserted into a requested/random source order.
    Returns (sources_text, target_index_1based).
    """
    if source_order is not None:
        target_indices = [
            i + 1 for i, source in enumerate(source_order)
            if source["type"] == "target"
        ]
        if len(target_indices) != 1:
            raise ValueError("source_order must contain exactly one target source")
        sources_text = ""
        for i, source in enumerate(source_order):
            sources_text += f"[Source {i + 1}] URL: {source['url']}\n"
        return sources_text, target_indices[0]

    urls = [u for u in competitor_urls if u]
    num_sources = len(urls) + 1
    target_idx = num_sources if target_position is None else target_position
    if not 1 <= target_idx <= num_sources:
        raise ValueError(f"target_position must be in [1, {num_sources}], got {target_idx}")
    urls.insert(target_idx - 1, target_url)

    sources_text = ""
    for i, url in enumerate(urls):
        sources_text += f"[Source {i + 1}] URL: {url}\n"

    return sources_text, target_idx


def query_page_url(doc_id: str, query_index: int, page_name: str) -> str:
    """Build the published URL for a query-specific page."""
    return f"{BASE_URL}/{doc_id}/query_{query_index}/{page_name}"


def valid_test_queries(row: pd.Series) -> list[str]:
    """Return parquet queries in their stored order."""
    per_query = row.get("agentgeo_baseline_test_per_query")
    if isinstance(per_query, dict):
        return [query for query, value in per_query.items() if value is not None]

    test_queries = row.get("test_queries")
    if isinstance(test_queries, (list, tuple)):
        return [str(query) for query in test_queries if query]

    query = row.get("query")
    return [str(query)] if query else []


def parse_doc_indices(raw: str) -> list[int]:
    """Parse 1-based doc indices like '10,11,15-18'."""
    indices = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid doc index range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    if any(index < 1 for index in indices):
        raise ValueError("Document indices are 1-based and must be >= 1")
    return list(dict.fromkeys(indices))


def configure_output_paths(output_path: Path) -> None:
    """Set output/checkpoint paths from the requested result file."""
    global OUTPUT_PATH, RUN_OUTPUT_DIR, CHECKPOINT_PATH, JSONL_PATH, DOC_OUTPUT_DIR, PROBE_HITS_PATH

    OUTPUT_PATH = output_path
    RUN_OUTPUT_DIR = OUTPUT_PATH.with_suffix("")
    CHECKPOINT_PATH = RUN_OUTPUT_DIR / "checkpoint.json"
    JSONL_PATH = RUN_OUTPUT_DIR / "records.jsonl"
    DOC_OUTPUT_DIR = RUN_OUTPUT_DIR / "by_doc"
    PROBE_HITS_PATH = RUN_OUTPUT_DIR / "probe_original_hits.json"


def load_cases_from_sample(
    sample_path: Path,
    doc_start: int,
    doc_end: int,
    queries_per_doc: int,
    num_competitors: int,
    query_start: int = 1,
    doc_indices: list[int] | None = None,
) -> list[dict]:
    """
    Build citation-test cases from sample_50.

    doc_start/doc_end are 1-based and inclusive: 10-15 means iloc[9:15].
    """
    df = pd.read_parquet(sample_path)
    if doc_indices:
        selected_items = [
            (index, df.iloc[index - 1])
            for index in doc_indices
            if index <= len(df)
        ]
    else:
        selected_items = [
            (index, row)
            for index, (_, row) in enumerate(
                df.iloc[doc_start - 1:doc_end].iterrows(),
                start=doc_start,
            )
        ]

    cases = []
    for doc_position, row in selected_items:
        doc_id = row["doc_id"]
        query_items = [
            (query_index, query)
            for query_index, query in enumerate(valid_test_queries(row), start=1)
            if query_index >= query_start
        ]
        if queries_per_doc > 0:
            query_items = query_items[:queries_per_doc]

        for query_offset, query in query_items:
            cases.append({
                "doc_position": doc_position,
                "doc_id": doc_id,
                "doc_url": row.get("url", ""),
                "query": query,
                "query_index": query_offset,
                "original_url": query_page_url(doc_id, query_offset, "original.html"),
                "optimized_url": query_page_url(doc_id, query_offset, "optimized.html"),
                "competitor_urls": [
                    query_page_url(doc_id, query_offset, f"competitor_{i}.html")
                    for i in range(1, num_competitors + 1)
                ],
                "baseline_citation": (
                    row.get("agentgeo_baseline_test_per_query", {}).get(query)
                    if isinstance(row.get("agentgeo_baseline_test_per_query"), dict)
                    else None
                ),
                "optimized_citation": (
                    row.get("agentgeo_optimized_test_per_query", {}).get(query)
                    if isinstance(row.get("agentgeo_optimized_test_per_query"), dict)
                    else None
                ),
            })
    return cases


async def run_citation_test(
    client: AsyncOpenAI,
    query: str,
    target_url: str,
    competitor_urls: list[str],
    target_position: int | None = None,
    source_order: list[dict] | None = None,
) -> dict:
    """Run a single citation test with URLs using Responses API."""
    if source_order is not None:
        source_order = [
            {
                **source,
                "url": target_url if source["type"] == "target" else source["url"],
            }
            for source in source_order
        ]
    sources_text, target_idx = build_url_sources(
        target_url,
        competitor_urls,
        target_position=target_position,
        source_order=source_order,
    )
    num_sources = (
        len(source_order)
        if source_order is not None
        else len([u for u in competitor_urls if u]) + 1
    )

    user_msg = (
        f"Question: {query}\n\n"
        f"Below are the ONLY sources you may use. Visit each URL, read its content, "
        f"and write your answer citing ONLY these sources using [index] format.\n"
        f"Do NOT search the web for additional information. Do NOT cite external URLs.\n\n"
        f"Sources:\n{sources_text}"
    )

    response = await client.responses.create(
        model=MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_msg,
        tools=[{"type": "web_search_preview"}],
        # reasoning={"effort": "low"},
        max_output_tokens=6000,
    )

    # Extract text from response (Responses API)
    answer = getattr(response, "output_text", "") or ""
    if not answer:
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        answer += content.text

    citation = check_citation(answer, target_idx, target_url, num_sources=num_sources)
    citation["generated_answer"] = answer
    citation["target_url"] = target_url
    citation["num_sources"] = num_sources
    if source_order is not None:
        citation["source_order"] = [
            {
                "source_index": i + 1,
                "type": source["type"],
                "url": source["url"],
                **(
                    {"competitor_index": source["competitor_index"]}
                    if source["type"] == "competitor"
                    else {}
                ),
            }
            for i, source in enumerate(source_order)
        ]
    return citation


def average_citation_runs(runs: list[dict]) -> dict:
    """Average citation metrics across repeated target placements."""
    metrics = ("cr", "word", "pos", "wordpos")
    averaged = {
        metric: sum(run[metric] for run in runs) / len(runs)
        for metric in metrics
    }
    return {
        **averaged,
        "is_cited": averaged["cr"] > 0,
        "runs": runs,
        "generated_answers": [run["generated_answer"] for run in runs],
        "target_positions": [run["target_idx"] for run in runs],
        "num_repeats": len(runs),
    }


def completed_variants(results: list[dict]) -> list[str]:
    """Return variants present in result records, preserving normal display order."""
    ordered = ["original", "optimized"]
    present = {key for record in results for key in record if key in ordered}
    return [variant for variant in ordered if variant in present]


def mean_metric(results: list[dict], variant: str, metric: str) -> float:
    """Mean metric value for one variant."""
    values = [
        r[variant][metric]
        for r in results
        if variant in r and metric in r[variant]
    ]
    return sum(values) / len(values) if values else 0.0


def aggregate_variant_metrics(results: list[dict], variant: str) -> dict:
    """Aggregate CR/Word/Pos/WordPos for one variant."""
    return {
        "cr": mean_metric(results, variant, "cr"),
        "word": mean_metric(results, variant, "word"),
        "pos": mean_metric(results, variant, "pos"),
        "wordpos": mean_metric(results, variant, "wordpos"),
    }


def build_summary(results: list[dict]) -> dict:
    """Build overall and per-doc summaries from completed records."""
    total = len(results)
    variants = completed_variants(results)
    variant_metrics = {
        variant: aggregate_variant_metrics(results, variant)
        for variant in variants
    }
    cited_counts = {
        variant: sum(r[variant]["cr"] for r in results if variant in r)
        for variant in variants
    }
    overall = {
        f"{variant}_citation_rate": cited_counts[variant] / total if total else 0
        for variant in variants
    }
    overall.update({
        f"{variant}_cited_count": cited_counts[variant]
        for variant in variants
    })
    overall.update({
        f"{variant}_metrics": variant_metrics[variant]
        for variant in variants
    })
    if "original" in variants and "optimized" in variants:
        overall["delta"] = (
            cited_counts["optimized"] - cited_counts["original"]
        ) / total if total else 0
        overall["delta_metrics"] = {
            metric: variant_metrics["optimized"][metric] - variant_metrics["original"][metric]
            for metric in variant_metrics["original"]
        }

    doc_stats = {}
    for r in results:
        did = r["doc_id"]
        if did not in doc_stats:
            doc_stats[did] = {
                "total": 0,
                "variants": {
                    variant: {"cited": 0, "metrics": []}
                    for variant in variants
                },
            }
        doc_stats[did]["total"] += 1
        for variant in variants:
            if variant not in r:
                continue
            doc_stats[did]["variants"][variant]["metrics"].append({
                metric: r[variant][metric]
                for metric in ("cr", "word", "pos", "wordpos")
            })
            doc_stats[did]["variants"][variant]["cited"] += r[variant]["cr"]

    per_doc = {}
    for did, stats in doc_stats.items():
        doc_summary = {"queries": stats["total"]}
        for variant in variants:
            variant_stats = stats["variants"][variant]
            count = len(variant_stats["metrics"])
            if count == 0:
                continue
            doc_summary[f"{variant}_citation_rate"] = variant_stats["cited"] / count
            doc_summary[f"{variant}_metrics"] = {
                metric: sum(item[metric] for item in variant_stats["metrics"]) / count
                for metric in ("cr", "word", "pos", "wordpos")
            }
        if "original" in variants and "optimized" in variants:
            original_count = len(stats["variants"]["original"]["metrics"])
            optimized_count = len(stats["variants"]["optimized"]["metrics"])
            if original_count and optimized_count:
                doc_summary["delta"] = (
                    stats["variants"]["optimized"]["cited"] / optimized_count
                    - stats["variants"]["original"]["cited"] / original_count
                )
                doc_summary["delta_metrics"] = {
                    metric: (
                        doc_summary["optimized_metrics"][metric]
                        - doc_summary["original_metrics"][metric]
                    )
                    for metric in ("cr", "word", "pos", "wordpos")
                }
        per_doc[did] = doc_summary

    return {
        "model": MODEL,
        "mode": "url_browsing",
        "base_url": BASE_URL,
        "num_competitors": NUM_COMPETITORS,
        "placement_mode": "random_insert_average",
        "total_queries": total,
        "variants": variants,
        "overall": overall,
        "per_doc": per_doc,
    }


def write_json(path: Path, data: dict) -> None:
    """Atomically write JSON so interrupted runs do not leave half files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def result_key(record: dict) -> tuple[str, int]:
    """Unique key for one evaluated sample query."""
    return (record["doc_id"], record["query_index"])


def merge_results(existing: list[dict], new_results: list[dict]) -> list[dict]:
    """Merge result records, letting newly completed records replace the same doc/query."""
    merged = {result_key(record): record for record in existing}
    for record in new_results:
        merged[result_key(record)] = record
    return sorted(merged.values(), key=lambda r: (r["doc_id"], r["query_index"]))


def load_existing_results(path: Path) -> list[dict]:
    """Load previous completed results if present."""
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    results = payload.get("results", [])
    return results if isinstance(results, list) else []


def write_completed_outputs(
    results: list[dict],
    total_cases: int,
    current_run_completed: int = 0,
    run_metadata: dict | None = None,
) -> dict:
    """Persist all completed records plus per-doc answer/score files."""
    ordered_results = sorted(results, key=lambda r: (r["doc_id"], r["query_index"]))
    summary = build_summary(ordered_results)
    payload = {
        "summary": summary,
        "completed_queries": len(ordered_results),
        "current_run_completed_queries": current_run_completed,
        "current_run_requested_queries": total_cases,
        "run_metadata": run_metadata or {},
        "results": ordered_results,
    }

    write_json(CHECKPOINT_PATH, payload)
    write_json(OUTPUT_PATH, {
        "summary": summary,
        "run_metadata": run_metadata or {},
        "results": ordered_results,
    })

    DOC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    doc_ids = sorted({r["doc_id"] for r in ordered_results})
    for doc_id in doc_ids:
        doc_results = [r for r in ordered_results if r["doc_id"] == doc_id]
        doc_payload = {
            "doc_id": doc_id,
            "doc_url": doc_results[0].get("doc_url", ""),
            "completed_queries": len(doc_results),
            "summary": summary["per_doc"].get(doc_id, {}),
            "results": doc_results,
        }
        write_json(DOC_OUTPUT_DIR / f"{doc_id}.json", doc_payload)

    return summary


def case_key(case: dict) -> tuple[str, int]:
    """Unique key for one sample query case before it has results."""
    return (case["doc_id"], case["query_index"])


def optimized_ge_original(record: dict) -> bool:
    """Return whether optimized CR is at least original CR for a completed record."""
    return (
        "original" in record
        and "optimized" in record
        and record["optimized"].get("cr", 0.0) >= record["original"].get("cr", 0.0)
    )


def optimized_gt_original(record: dict) -> bool:
    """Return whether optimized CR is strictly greater than original CR."""
    return (
        "original" in record
        and "optimized" in record
        and record["optimized"].get("cr", 0.0) > record["original"].get("cr", 0.0)
    )


def optimized_ge_original_count(records: list[dict]) -> int:
    """Count completed records where optimized CR is at least original CR."""
    return sum(1 for record in records if optimized_ge_original(record))


def optimized_gt_original_count(records: list[dict]) -> int:
    """Count completed records where optimized CR is strictly greater than original CR."""
    return sum(1 for record in records if optimized_gt_original(record))


def build_case_source_orders(
    cases: list[dict],
    repeats: int,
    seed: int,
) -> dict[tuple[str, int], list[list[dict]]]:
    """Build independently shuffled source orders for each query repeat."""
    if repeats < 1:
        raise ValueError("--position-repeats must be at least 1")
    rng = random.Random(seed)
    orders = {}
    for case in sorted(cases, key=lambda c: (c["doc_id"], c["query_index"])):
        case_orders = []
        competitor_sources = [
            {"type": "competitor", "url": url, "competitor_index": i}
            for i, url in enumerate(case["competitor_urls"][:NUM_COMPETITORS], start=1)
            if url
        ]
        base_sources = [{"type": "target", "url": None}, *competitor_sources]
        for _ in range(repeats):
            shuffled_sources = [dict(source) for source in base_sources]
            rng.shuffle(shuffled_sources)
            case_orders.append(shuffled_sources)
        orders[case_key(case)] = case_orders
    return orders


def append_jsonl(path: Path, record: dict) -> None:
    """Append one completed case for easy tailing/recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def group_cases_by_doc(cases: list[dict]) -> dict[str, list[dict]]:
    """Group cases by doc while preserving case order."""
    grouped = {}
    for case in cases:
        grouped.setdefault(case["doc_id"], []).append(case)
    return grouped


def load_cases_from_probe_hits(path: Path) -> list[dict]:
    """Load saved probe hits as reusable score-mode cases."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = []
    for doc_result in payload.get("doc_results", []):
        selected = doc_result.get("selected_queries_for_next_run") or []
        for item in selected:
            case = item.get("case")
            if case:
                cases.append(case)
    if cases:
        return cases

    # Backward compatibility with the older probe file that stored one hit per doc.
    for hit in payload.get("hits", []):
        case = hit.get("case")
        if case:
            cases.append(case)
    return cases


async def probe_original_doc(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    doc_id: str,
    doc_cases: list[dict],
    doc_idx: int,
    total_docs: int,
    case_source_orders: dict[tuple[str, int], list[list[dict]]],
    min_hits: int,
    max_hits: int,
    random_fill: int,
    seed: int,
    citation_threshold: float,
) -> dict:
    """Probe one doc until max_hits original CR-below-threshold queries are found."""
    uncited_queries = []
    tested_queries = []
    for case in doc_cases:
        async with sem:
            t0 = time.time()
            runs = []
            for source_order in case_source_orders[case_key(case)]:
                runs.append(await run_citation_test(
                    client,
                    case["query"],
                    case["original_url"],
                    case["competitor_urls"][:NUM_COMPETITORS],
                    source_order=source_order,
                ))
            original_result = average_citation_runs(runs)
            elapsed = time.time() - t0

        is_uncited = original_result["cr"] < citation_threshold
        tested_record = {
            "doc_id": doc_id,
            "doc_position": case.get("doc_position"),
            "query_index": case["query_index"],
            "query": case["query"],
            "case": case,
            "original": original_result,
            "is_uncited": is_uncited,
            "elapsed_seconds": elapsed,
        }
        tested_queries.append(tested_record)
        print(
            f"[probe {doc_idx+1}/{total_docs}] {doc_id} "
            f"Q{case['query_index']} original_cr={original_result['cr']:.2f} "
            f"{'UNCITED' if is_uncited else 'cited'} "
            f"hits={len(uncited_queries) + (1 if is_uncited else 0)}/{max_hits} "
            f"positions={original_result['target_positions']} ({elapsed:.1f}s)"
        )
        if is_uncited:
            uncited_queries.append(tested_record)
            if len(uncited_queries) >= max_hits:
                break

    first_case = doc_cases[0] if doc_cases else {}
    hit_keys = {
        (item["doc_id"], item["query_index"])
        for item in uncited_queries
    }
    supplement_pool = [
        case for case in doc_cases
        if (case["doc_id"], case["query_index"]) not in hit_keys
    ]
    random_supplements = []
    if len(uncited_queries) < min_hits and random_fill > 0 and supplement_pool:
        rng = random.Random(f"{seed}:{doc_id}")
        sampled_cases = rng.sample(supplement_pool, k=min(random_fill, len(supplement_pool)))
        random_supplements = [
            {
                "doc_id": doc_id,
                "doc_position": case.get("doc_position"),
                "query_index": case["query_index"],
                "query": case["query"],
                "case": case,
                "supplement_reason": (
                    f"Only found {len(uncited_queries)} queries with "
                    f"original_cr < {citation_threshold}; random fallback."
                ),
            }
            for case in sampled_cases
        ]

    selected_queries = (
        uncited_queries[:min_hits]
        if len(uncited_queries) >= min_hits
        else [*uncited_queries, *random_supplements]
    )
    return {
        "status": "hit" if uncited_queries else "miss",
        "doc_id": doc_id,
        "doc_position": first_case.get("doc_position"),
        "has_min_uncited_queries": len(uncited_queries) >= min_hits,
        "min_uncited_queries": min_hits,
        "max_uncited_queries": max_hits,
        "citation_threshold": citation_threshold,
        "uncited_count": len(uncited_queries),
        "candidate_query_count": len(doc_cases),
        "tested_queries": len(tested_queries),
        "tested_query_count": len(tested_queries),
        "stopped_because": "max_uncited_reached" if len(uncited_queries) >= max_hits else "exhausted_cases",
        "uncited_queries": uncited_queries,
        "first_five_uncited_queries": uncited_queries[:min_hits],
        "random_supplements": random_supplements,
        "selected_queries_for_next_run": selected_queries,
        "trace": tested_queries,
    }


def write_probe_outputs(path: Path, doc_results: list[dict], total_docs: int) -> None:
    """Persist probe hits and misses."""
    ordered = sorted(doc_results, key=lambda h: (h.get("doc_position") or 0, h["doc_id"]))
    all_uncited = [
        item
        for doc_result in ordered
        for item in doc_result.get("uncited_queries", [])
    ]
    payload = {
        "model": MODEL,
        "mode": "probe-original",
        "base_url": BASE_URL,
        "num_competitors": NUM_COMPETITORS,
        "total_docs": total_docs,
        "docs_with_any_uncited": sum(1 for result in ordered if result.get("uncited_count", 0) > 0),
        "docs_with_min_uncited": sum(1 for result in ordered if result.get("has_min_uncited_queries")),
        "total_uncited_queries": len(all_uncited),
        "has_any_doc_with_min_uncited": any(result.get("has_min_uncited_queries") for result in ordered),
        "doc_results": ordered,
        "all_uncited_queries": all_uncited,
    }
    write_json(path, payload)


async def process_test_case(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    case: dict,
    case_idx: int,
    total: int,
    source_orders: list[list[dict]],
    variants: list[str],
) -> dict:
    """Process one sample query entry for the requested variants."""
    async with sem:
        query = case["query"]
        doc_id = case["doc_id"]
        doc_url = case.get("doc_url", "")

        variant_urls = {
            "original": case["original_url"],
            "optimized": case["optimized_url"],
        }
        competitor_urls = case["competitor_urls"][:NUM_COMPETITORS]

        result_by_variant = {}
        timing_by_variant = {}
        for variant in variants:
            t0 = time.time()
            runs = []
            for source_order in source_orders:
                runs.append(await run_citation_test(
                    client,
                    query,
                    variant_urls[variant],
                    competitor_urls,
                    source_order=source_order,
                ))
            timing_by_variant[variant] = time.time() - t0
            result_by_variant[variant] = average_citation_runs(runs)

        variant_status = " ".join(
            f"{variant}_cr={result_by_variant[variant]['cr']:.2f}"
            for variant in variants
        )
        timing_status = " + ".join(
            f"{timing_by_variant[variant]:.1f}s"
            for variant in variants
        )
        status = (
            f"[{case_idx+1}/{total}] {doc_id} Q{case['query_index']}: "
            f"positions={next(iter(result_by_variant.values()))['target_positions']} "
            f"{variant_status} ({timing_status})"
        )
        print(status)

        record = {
            "doc_id": doc_id,
            "doc_position": case.get("doc_position"),
            "doc_url": doc_url,
            "query": query,
            "query_index": case["query_index"],
            **result_by_variant,
            "baseline_citation_sample": case.get("baseline_citation"),
            "optimized_citation_sample": case.get("optimized_citation"),
        }
        if "original" in result_by_variant and "optimized" in result_by_variant:
            record["optimized_ge_original_cr"] = optimized_ge_original(record)
            record["optimized_gt_original_cr"] = optimized_gt_original(record)
        return record


async def run_score_cases(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    cases: list[dict],
    case_source_orders: dict[tuple[str, int], list[list[dict]]],
    variants: list[str],
    concurrency: int,
    existing_results: list[dict],
    stop_optimized_ge_original: int,
    stop_optimized_gt_original: int,
) -> tuple[list[dict], dict]:
    """Run score-mode cases, optionally stopping when enough target cases finish."""
    run_results = []
    pending: set[asyncio.Task] = set()
    next_case_idx = 0
    stop_triggered = False
    stop_reason = ""

    def launch_next() -> bool:
        nonlocal next_case_idx
        if next_case_idx >= len(cases):
            return False
        case_idx = next_case_idx
        case = cases[case_idx]
        next_case_idx += 1
        pending.add(asyncio.create_task(process_test_case(
            sem,
            client,
            case,
            case_idx,
            len(cases),
            case_source_orders[case_key(case)],
            variants,
        )))
        return True

    for _ in range(min(concurrency, len(cases))):
        launch_next()

    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = await task
            run_results.append(result)
            append_jsonl(JSONL_PATH, result)

            ge_count = optimized_ge_original_count(run_results)
            gt_count = optimized_gt_original_count(run_results)
            run_metadata = {
                "stop_optimized_ge_original": stop_optimized_ge_original,
                "stop_optimized_gt_original": stop_optimized_gt_original,
                "optimized_ge_original_count": ge_count,
                "optimized_gt_original_count": gt_count,
                "early_stopped": False,
                "launched_queries": next_case_idx,
                "cancelled_queries": 0,
            }
            combined_results = merge_results(existing_results, run_results)
            write_completed_outputs(
                combined_results,
                len(cases),
                current_run_completed=len(run_results),
                run_metadata=run_metadata,
            )

            if stop_optimized_ge_original and ge_count >= stop_optimized_ge_original:
                stop_triggered = True
                stop_reason = (
                    f"optimized_ge_original_count reached "
                    f"{ge_count}/{stop_optimized_ge_original}"
                )
            if stop_optimized_gt_original and gt_count >= stop_optimized_gt_original:
                stop_triggered = True
                stop_reason = (
                    f"optimized_gt_original_count reached "
                    f"{gt_count}/{stop_optimized_gt_original}"
                )

        if stop_triggered:
            cancelled = len(pending)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            pending.clear()
            return run_results, {
                "stop_optimized_ge_original": stop_optimized_ge_original,
                "stop_optimized_gt_original": stop_optimized_gt_original,
                "optimized_ge_original_count": optimized_ge_original_count(run_results),
                "optimized_gt_original_count": optimized_gt_original_count(run_results),
                "early_stopped": True,
                "stop_reason": stop_reason,
                "launched_queries": next_case_idx,
                "cancelled_queries": cancelled,
            }

        while len(pending) < concurrency and launch_next():
            pass

    return run_results, {
        "stop_optimized_ge_original": stop_optimized_ge_original,
        "stop_optimized_gt_original": stop_optimized_gt_original,
        "optimized_ge_original_count": optimized_ge_original_count(run_results),
        "optimized_gt_original_count": optimized_gt_original_count(run_results),
        "early_stopped": False,
        "stop_reason": "completed_all_cases",
        "launched_queries": next_case_idx,
        "cancelled_queries": 0,
    }


async def score_doc_until_retained(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    doc_id: str,
    doc_cases: list[dict],
    doc_idx: int,
    total_docs: int,
    case_index_by_key: dict[tuple[str, int], int],
    case_source_orders: dict[tuple[str, int], list[list[dict]]],
    variants: list[str],
    retain_per_doc: int,
) -> dict:
    """Score one doc, retaining only queries where optimized CR >= original CR."""
    retained = []
    rejected = []
    tested = []

    for case in doc_cases:
        result = await process_test_case(
            sem,
            client,
            case,
            case_index_by_key[case_key(case)],
            len(case_index_by_key),
            case_source_orders[case_key(case)],
            variants,
        )
        keep = optimized_ge_original(result)
        result["retained_for_summary"] = keep
        result["retention_reason"] = (
            "optimized_cr_ge_original_cr"
            if keep
            else "optimized_cr_lt_original_cr"
        )
        tested.append(result)
        append_jsonl(JSONL_PATH, result)

        if keep:
            retained.append(result)
        else:
            rejected.append(result)

        print(
            f"[retain {doc_idx+1}/{total_docs}] {doc_id} "
            f"Q{result['query_index']} keep={keep} "
            f"retained={len(retained)}/{retain_per_doc}"
        )
        if len(retained) >= retain_per_doc:
            break

    first_case = doc_cases[0] if doc_cases else {}
    return {
        "doc_id": doc_id,
        "doc_position": first_case.get("doc_position"),
        "target_retained_queries": retain_per_doc,
        "has_target_retained_queries": len(retained) >= retain_per_doc,
        "retained_count": len(retained),
        "rejected_count": len(rejected),
        "tested_query_count": len(tested),
        "candidate_query_count": len(doc_cases),
        "stopped_because": (
            "target_retained_reached"
            if len(retained) >= retain_per_doc
            else "exhausted_cases"
        ),
        "retained_results": retained,
        "rejected_results": rejected,
        "tested_results": tested,
    }


async def run_score_cases_per_doc_retain(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    cases: list[dict],
    case_source_orders: dict[tuple[str, int], list[list[dict]]],
    variants: list[str],
    existing_results: list[dict],
    retain_per_doc: int,
) -> tuple[list[dict], dict]:
    """Run score mode grouped by doc and retain only optimized>=original queries."""
    grouped_cases = group_cases_by_doc(cases)
    case_index_by_key = {
        case_key(case): i
        for i, case in enumerate(cases)
    }
    tasks = [
        asyncio.create_task(score_doc_until_retained(
            sem,
            client,
            doc_id,
            doc_cases,
            i,
            len(grouped_cases),
            case_index_by_key,
            case_source_orders,
            variants,
            retain_per_doc,
        ))
        for i, (doc_id, doc_cases) in enumerate(grouped_cases.items())
    ]

    doc_results = []
    retained_results = []
    for task in asyncio.as_completed(tasks):
        doc_result = await task
        doc_results.append(doc_result)
        retained_results.extend(doc_result["retained_results"])
        run_metadata = {
            "retain_optimized_ge_original_per_doc": retain_per_doc,
            "docs_completed": len(doc_results),
            "docs_requested": len(grouped_cases),
            "docs_with_target_retained": sum(
                1 for item in doc_results if item["has_target_retained_queries"]
            ),
            "retained_query_count": len(retained_results),
            "tested_query_count": sum(item["tested_query_count"] for item in doc_results),
            "doc_results": doc_results,
        }
        combined_results = merge_results(existing_results, retained_results)
        write_completed_outputs(
            combined_results,
            len(cases),
            current_run_completed=len(retained_results),
            run_metadata=run_metadata,
        )

    return retained_results, {
        "retain_optimized_ge_original_per_doc": retain_per_doc,
        "docs_completed": len(doc_results),
        "docs_requested": len(grouped_cases),
        "docs_with_target_retained": sum(
            1 for item in doc_results if item["has_target_retained_queries"]
        ),
        "retained_query_count": len(retained_results),
        "tested_query_count": sum(item["tested_query_count"] for item in doc_results),
        "doc_results": sorted(
            doc_results,
            key=lambda item: (item.get("doc_position") or 0, item["doc_id"]),
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT citation scoring on sample_50 docs/queries."
    )
    parser.add_argument(
        "--mode",
        choices=["score", "probe-original"],
        default="score",
        help="score computes requested variants; probe-original finds original CR-below-threshold queries.",
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--sample-path", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--doc-start", type=int, default=10, help="1-based inclusive doc start")
    parser.add_argument("--doc-end", type=int, default=15, help="1-based inclusive doc end")
    parser.add_argument(
        "--doc-indices",
        default="",
        help="Comma/range 1-based doc indices, e.g. '10,11,15-18'. Overrides doc-start/doc-end.",
    )
    parser.add_argument("--query-start", type=int, default=2, help="1-based query index to start from")
    parser.add_argument("--queries-per-doc", type=int, default=5)
    parser.add_argument("--num-competitors", type=int, default=NUM_COMPETITORS)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    parser.add_argument("--position-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--probe-min-hits", type=int, default=5)
    parser.add_argument("--probe-max-hits", type=int, default=10)
    parser.add_argument("--probe-random-fill", type=int, default=2)
    parser.add_argument(
        "--probe-citation-threshold",
        type=float,
        default=1.0,
        help="Treat original_cr below this threshold as unreferenced/not fully cited.",
    )
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--probe-output-path", type=Path, default=None)
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=None,
        help="Load cases from a probe_original_hits.json file instead of selecting from parquet.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=["original", "optimized"],
        default=["original", "optimized"],
        help="Variants to score in score mode. Use '--variants optimized' after probe mode.",
    )
    parser.add_argument(
        "--stop-optimized-ge-original",
        type=int,
        default=0,
        help="In score mode, stop after this many current-run queries have optimized CR >= original CR.",
    )
    parser.add_argument(
        "--stop-optimized-gt-original",
        type=int,
        default=0,
        help="In score mode, stop after this many current-run queries have optimized CR > original CR.",
    )
    parser.add_argument(
        "--retain-optimized-ge-original-per-doc",
        type=int,
        default=0,
        help=(
            "In score mode, retain only optimized CR >= original CR queries for summary, "
            "and stop each doc after this many retained queries."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print selected cases without calling GPT")
    return parser.parse_args()


async def main():
    global MODEL, NUM_COMPETITORS

    args = parse_args()
    if args.probe_min_hits < 1:
        raise ValueError("--probe-min-hits must be at least 1")
    if args.probe_max_hits < args.probe_min_hits:
        raise ValueError("--probe-max-hits must be >= --probe-min-hits")
    if args.probe_random_fill < 0:
        raise ValueError("--probe-random-fill must be >= 0")
    if args.stop_optimized_ge_original < 0:
        raise ValueError("--stop-optimized-ge-original must be >= 0")
    if args.stop_optimized_gt_original < 0:
        raise ValueError("--stop-optimized-gt-original must be >= 0")
    if args.stop_optimized_ge_original and not {"original", "optimized"}.issubset(set(args.variants)):
        raise ValueError("--stop-optimized-ge-original requires --variants original optimized")
    if args.stop_optimized_gt_original and not {"original", "optimized"}.issubset(set(args.variants)):
        raise ValueError("--stop-optimized-gt-original requires --variants original optimized")
    if args.stop_optimized_ge_original and args.stop_optimized_gt_original:
        raise ValueError(
            "Use either --stop-optimized-ge-original or --stop-optimized-gt-original, not both."
        )
    if args.retain_optimized_ge_original_per_doc < 0:
        raise ValueError("--retain-optimized-ge-original-per-doc must be >= 0")
    if args.retain_optimized_ge_original_per_doc and not {"original", "optimized"}.issubset(set(args.variants)):
        raise ValueError("--retain-optimized-ge-original-per-doc requires --variants original optimized")
    if args.retain_optimized_ge_original_per_doc and (
        args.stop_optimized_ge_original or args.stop_optimized_gt_original
    ):
        raise ValueError(
            "Use either --retain-optimized-ge-original-per-doc or "
            "a stop-optimized-* option, not both."
        )
    MODEL = args.model
    NUM_COMPETITORS = args.num_competitors
    configure_output_paths(args.output_path)
    probe_output_path = args.probe_output_path or PROBE_HITS_PATH

    doc_indices = parse_doc_indices(args.doc_indices) if args.doc_indices else None

    if args.cases_path:
        cases = load_cases_from_probe_hits(args.cases_path)
    else:
        cases = load_cases_from_sample(
            sample_path=args.sample_path,
            doc_start=args.doc_start,
            doc_end=args.doc_end,
            queries_per_doc=args.queries_per_doc,
            num_competitors=args.num_competitors,
            query_start=args.query_start,
            doc_indices=doc_indices,
        )
    case_source_orders = build_case_source_orders(
        cases,
        repeats=args.position_repeats,
        seed=args.seed,
    )
    doc_selection = (
        f"doc indices {args.doc_indices}"
        if args.doc_indices
        else f"docs {args.doc_start}-{args.doc_end}"
    )
    print(
        f"Loaded {len(cases)} test cases "
        f"from {args.cases_path or args.sample_path} "
        f"({doc_selection}, query_start={args.query_start}, "
        f"{args.queries_per_doc} queries/doc, "
        f"{args.num_competitors} competitors/query)"
    )
    print(f"Mode: {args.mode}")
    print(f"Model: {MODEL}, Concurrency: {args.concurrency}")
    print(f"Mode: URL browsing via web_search_preview (Responses API)")
    print(f"Base URL: {BASE_URL}")
    print(f"Competitors per query: {NUM_COMPETITORS}")
    print(f"Random source-order repeats per query: {args.position_repeats} (seed={args.seed})\n")
    if args.mode == "probe-original":
        print(
            f"Probe target: find at least {args.probe_min_hits} and at most "
            f"{args.probe_max_hits} original_cr < {args.probe_citation_threshold} "
            f"queries per doc; random fill={args.probe_random_fill}\n"
        )
    elif args.stop_optimized_ge_original:
        print(
            f"Score early stop: stop after {args.stop_optimized_ge_original} "
            f"current-run queries with optimized_cr >= original_cr\n"
        )
    elif args.stop_optimized_gt_original:
        print(
            f"Score early stop: stop after {args.stop_optimized_gt_original} "
            f"current-run queries with optimized_cr > original_cr\n"
        )
    elif args.retain_optimized_ge_original_per_doc:
        print(
            f"Score retain mode: each doc keeps only optimized_cr >= original_cr "
            f"queries, stopping after {args.retain_optimized_ge_original_per_doc} "
            f"retained queries per doc. Summary uses retained queries only.\n"
        )

    if args.dry_run:
        for case in cases:
            source_orders = case_source_orders[case_key(case)]
            target_positions = [
                next(
                    i + 1 for i, source in enumerate(source_order)
                    if source["type"] == "target"
                )
                for source_order in source_orders
            ]
            print(
                f"{case['doc_id']} Q{case['query_index']}: {case['query']}\n"
                f"  doc position: {case.get('doc_position')}\n"
                f"  original: {case['original_url']}\n"
                f"  optimized: {case['optimized_url']}\n"
                f"  target positions: {target_positions}\n"
                f"  competitors: {case['competitor_urls'][0]} ... {case['competitor_urls'][-1]}"
            )
        return

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)

    if args.mode == "probe-original":
        grouped_cases = group_cases_by_doc(cases)
        tasks = [
            asyncio.create_task(probe_original_doc(
                sem,
                client,
                doc_id,
                doc_cases,
                i,
                len(grouped_cases),
                case_source_orders,
                args.probe_min_hits,
                args.probe_max_hits,
                args.probe_random_fill,
                args.seed,
                args.probe_citation_threshold,
            ))
            for i, (doc_id, doc_cases) in enumerate(grouped_cases.items())
        ]
        doc_results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            doc_results.append(result)
            write_probe_outputs(probe_output_path, doc_results, len(grouped_cases))

        write_probe_outputs(probe_output_path, doc_results, len(grouped_cases))
        print(f"\n{'='*60}")
        print("Original Probe Results")
        print(f"{'='*60}")
        docs_with_min = sum(1 for result in doc_results if result["has_min_uncited_queries"])
        total_uncited = sum(result["uncited_count"] for result in doc_results)
        print(
            f"Docs with >= {args.probe_min_hits} unreferenced queries: "
            f"{docs_with_min}/{len(grouped_cases)}"
        )
        print(f"Total unreferenced queries recorded: {total_uncited}")
        for result in sorted(doc_results, key=lambda h: (h.get("doc_position") or 0, h["doc_id"])):
            print(
                f"doc#{result.get('doc_position')} {result['doc_id']}: "
                f"has_5={result['has_min_uncited_queries']} "
                f"uncited={result['uncited_count']} "
                f"tested={result['tested_query_count']} "
                f"stop={result['stopped_because']}"
            )
            for hit in result["uncited_queries"]:
                metrics = hit["original"]
                print(
                    f"  Q{hit['query_index']} cr={metrics['cr']:.2f} "
                    f"word={metrics['word']:.4f} pos={metrics['pos']:.4f} "
                    f"wordpos={metrics['wordpos']:.4f} | {hit['query']}"
                )
            if result["random_supplements"]:
                print("  Random supplements:")
                for item in result["random_supplements"]:
                    print(f"  Q{item['query_index']} | {item['query']}")
        print(f"\nProbe hits saved to: {probe_output_path}")
        return

    existing_results = load_existing_results(OUTPUT_PATH)
    if existing_results:
        print(f"Loaded {len(existing_results)} existing completed queries from {OUTPUT_PATH}")

    if args.retain_optimized_ge_original_per_doc:
        run_results, run_metadata = await run_score_cases_per_doc_retain(
            sem,
            client,
            cases,
            case_source_orders,
            args.variants,
            existing_results,
            args.retain_optimized_ge_original_per_doc,
        )
    else:
        run_results, run_metadata = await run_score_cases(
            sem,
            client,
            cases,
            case_source_orders,
            args.variants,
            args.concurrency,
            existing_results,
            args.stop_optimized_ge_original,
            args.stop_optimized_gt_original,
        )

    results = merge_results(existing_results, run_results)
    summary = write_completed_outputs(
        results,
        len(cases),
        current_run_completed=len(run_results),
        run_metadata=run_metadata,
    )

    # ── Aggregate stats ──
    total = summary["total_queries"]
    variants = summary.get("variants", [])

    # -- Print report --
    print(f"\n{'='*60}")
    print(f"{MODEL} Citation Test Results (URL Browsing)")
    print(f"{'='*60}")
    print(f"Current run completed queries: {len(run_results)}/{len(cases)}")
    if run_metadata.get("early_stopped"):
        print(f"Early stop: {run_metadata.get('stop_reason')}")
        print(f"Cancelled in-flight queries: {run_metadata.get('cancelled_queries', 0)}")
    if args.stop_optimized_ge_original:
        print(
            "Current-run optimized>=original count: "
            f"{run_metadata.get('optimized_ge_original_count', 0)}/"
            f"{args.stop_optimized_ge_original}"
        )
    if args.stop_optimized_gt_original:
        print(
            "Current-run optimized>original count: "
            f"{run_metadata.get('optimized_gt_original_count', 0)}/"
            f"{args.stop_optimized_gt_original}"
        )
    if args.retain_optimized_ge_original_per_doc:
        print(
            "Per-doc retain mode: "
            f"{run_metadata.get('docs_with_target_retained', 0)}/"
            f"{run_metadata.get('docs_requested', 0)} docs reached "
            f"{args.retain_optimized_ge_original_per_doc} retained queries"
        )
        print(
            f"Tested queries: {run_metadata.get('tested_query_count', 0)}, "
            f"retained for summary: {run_metadata.get('retained_query_count', 0)}"
        )
    print(f"Total merged queries: {total}")
    for variant in variants:
        cited = summary["overall"][f"{variant}_cited_count"]
        print(f"{variant.title()} citation rate: {cited:.2f}/{total} = {cited/total:.1%}")
    if "delta" in summary["overall"]:
        original_cited = summary["overall"]["original_cited_count"]
        optimized_cited = summary["overall"]["optimized_cited_count"]
        print(f"Delta: {(optimized_cited-original_cited)/total:+.1%}")

    print(f"\nOverall metrics:")
    print(f"{'Variant':<12} {'CR':>8} {'Word':>8} {'Pos':>8} {'WordPos':>8}")
    print(f"{'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    metric_rows = [
        (variant.title(), summary["overall"][f"{variant}_metrics"])
        for variant in variants
    ]
    if "delta_metrics" in summary["overall"]:
        metric_rows.append(("Delta", summary["overall"]["delta_metrics"]))
    for label, metrics in metric_rows:
        print(
            f"{label:<12} "
            f"{metrics['cr']:>8.4f} "
            f"{metrics['word']:>8.4f} "
            f"{metrics['pos']:>8.4f} "
            f"{metrics['wordpos']:>8.4f}"
        )

    print(f"\nPer-document breakdown:")
    if "original" in variants and "optimized" in variants:
        print(f"{'Doc ID':<14} {'Orig CR':>8} {'Opt CR':>8} {'dCR':>8} {'dWord':>8} {'dPos':>8} {'dWPos':>8}")
        print(f"{'-'*14} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    else:
        print(f"{'Doc ID':<14} {'Variant':<10} {'CR':>8} {'Word':>8} {'Pos':>8} {'WPos':>8}")
        print(f"{'-'*14} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for did, s in summary["per_doc"].items():
        if "original" in variants and "optimized" in variants:
            o_rate = s["original_citation_rate"]
            p_rate = s["optimized_citation_rate"]
            o_metrics = s["original_metrics"]
            p_metrics = s["optimized_metrics"]
            d_metrics = s["delta_metrics"]
            print(
                f"{did:<14} "
                f"{o_metrics['cr']:>8.2%} "
                f"{p_metrics['cr']:>8.2%} "
                f"{p_rate-o_rate:>+8.2%} "
                f"{d_metrics['word']:>+8.4f} "
                f"{d_metrics['pos']:>+8.4f} "
                f"{d_metrics['wordpos']:>+8.4f}"
            )
        else:
            for variant in variants:
                metrics = s.get(f"{variant}_metrics")
                if not metrics:
                    continue
                print(
                    f"{did:<14} {variant:<10} "
                    f"{metrics['cr']:>8.2%} "
                    f"{metrics['word']:>8.4f} "
                    f"{metrics['pos']:>8.4f} "
                    f"{metrics['wordpos']:>8.4f}"
                )
    print(f"\nResults saved to: {OUTPUT_PATH}")
    print(f"Checkpoint saved to: {CHECKPOINT_PATH}")
    print(f"Per-doc answers and scores saved to: {DOC_OUTPUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
