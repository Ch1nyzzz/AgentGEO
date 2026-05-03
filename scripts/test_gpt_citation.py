"""
Test citation rates using GPT-5 with URL browsing.

For each query in manifest.json, run two tests:
  1. Give GPT the original page URL + 5 competitor URLs → check citation
  2. Give GPT the optimized page URL + 5 competitor URLs → check citation

GPT-5 browses the URLs itself via web_search_preview tool (Responses API).
"""

import asyncio
import json
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ── Config ──
MODEL = "gpt-5"
CONCURRENCY = 32
NUM_COMPETITORS = 10
PAGES_DIR = Path("docs/pages")
OUTPUT_PATH = Path("outputs/gpt5_citation_results.json")
BASE_URL = "https://ch1nyzzz.github.io/AgentGEO"

SYSTEM_PROMPT = (
    "You are a research assistant. The user will provide a question and a numbered list of source URLs. "
    "You MUST visit each URL to read its content, then write an answer based ONLY on information found in those sources.\n\n"
    "STRICT RULES:\n"
    "1. ONLY use information from the provided sources. Do NOT use your own knowledge or any external websites.\n"
    "2. Every sentence in the answer MUST be immediately followed by an inline citation using [index] format, "
    "where index is the Source number (e.g. [1], [2], [3]).\n"
    "3. When citing multiple sources, use [1][2][3] format, NOT [1, 2, 3] or (Source 1) or any other format.\n"
    "4. Do NOT cite URLs directly. Do NOT use markdown links like [text](url). ONLY use [index] format.\n"
    "5. If a source is not relevant to the question, do not cite it.\n"
    "6. The answer should be accurate, concise, and written in an unbiased journalistic tone."
)


def check_citation(answer: str, target_idx: int, target_url: str = "") -> dict:
    """Check if target is cited in the answer."""
    cited_by_index = f"[{target_idx}]" in answer
    cited_by_url = bool(target_url) and target_url in answer
    is_cited = cited_by_index or cited_by_url
    cited_indices = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    return {
        "is_cited": is_cited,
        "cited_by_index": cited_by_index,
        "cited_by_url": cited_by_url,
        "cited_indices": cited_indices,
        "target_idx": target_idx,
    }


def build_url_sources(target_url: str, competitor_urls: list[str]) -> tuple[str, int]:
    """
    Build source list with target URL fixed at the end (same position for original & optimized).
    Returns (sources_text, target_index_1based).
    """
    urls = [u for u in competitor_urls if u]
    urls.append(target_url)
    target_idx = len(urls)

    sources_text = ""
    for i, url in enumerate(urls):
        sources_text += f"[Source {i + 1}] URL: {url}\n"

    return sources_text, target_idx


async def run_citation_test(
    client: AsyncOpenAI,
    query: str,
    target_url: str,
    competitor_urls: list[str],
) -> dict:
    """Run a single citation test with URLs using Responses API."""
    sources_text, target_idx = build_url_sources(target_url, competitor_urls)

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
        max_output_tokens=16384,
    )

    # Extract text from response (Responses API)
    answer = getattr(response, "output_text", "") or ""
    if not answer:
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        answer += content.text

    citation = check_citation(answer, target_idx, target_url)
    citation["generated_answer"] = answer
    citation["target_url"] = target_url
    return citation


async def process_test_case(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    case: dict,
    case_idx: int,
    total: int,
) -> dict:
    """Process one manifest entry: test both original and optimized."""
    async with sem:
        query = case["query"]
        doc_id = case["doc_id"]
        doc_url = case.get("doc_url", "")

        # Build URLs from GitHub Pages
        original_url = f"{BASE_URL}/{case['original_path']}"
        optimized_url = f"{BASE_URL}/{case['optimized_path']}"
        competitor_urls = [
            f"{BASE_URL}/{comp['path']}"
            for comp in case["competitors"][:NUM_COMPETITORS]
        ]

        # Test original
        t0 = time.time()
        original_result = await run_citation_test(
            client, query, original_url, competitor_urls
        )
        t1 = time.time()

        # Test optimized
        optimized_result = await run_citation_test(
            client, query, optimized_url, competitor_urls
        )
        t2 = time.time()

        status = (
            f"[{case_idx+1}/{total}] {doc_id} Q{case['query_index']}: "
            f"original={'✓' if original_result['is_cited'] else '✗'} "
            f"optimized={'✓' if optimized_result['is_cited'] else '✗'} "
            f"({t1-t0:.1f}s + {t2-t1:.1f}s)"
        )
        print(status)

        return {
            "doc_id": doc_id,
            "doc_url": doc_url,
            "query": query,
            "query_index": case["query_index"],
            "original": original_result,
            "optimized": optimized_result,
            "baseline_citation_rate_claude": case.get("baseline_citation_rate"),
            "optimized_citation_rate_claude": case.get("optimized_citation_rate"),
        }


async def main():
    manifest = json.loads((PAGES_DIR / "manifest.json").read_text())
    print(f"Loaded {len(manifest)} test cases from manifest.json")
    print(f"Model: {MODEL}, Concurrency: {CONCURRENCY}")
    print(f"Mode: URL browsing via web_search_preview (Responses API)")
    print(f"Base URL: {BASE_URL}")
    print(f"Competitors per query: {NUM_COMPETITORS}\n")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(CONCURRENCY)

    tasks = [
        process_test_case(sem, client, case, i, len(manifest))
        for i, case in enumerate(manifest)
    ]
    results = await asyncio.gather(*tasks)

    # ── Aggregate stats ──
    original_cited = sum(1 for r in results if r["original"]["is_cited"])
    optimized_cited = sum(1 for r in results if r["optimized"]["is_cited"])
    total = len(results)

    # Per-doc stats
    doc_stats = {}
    for r in results:
        did = r["doc_id"]
        if did not in doc_stats:
            doc_stats[did] = {"original_cited": 0, "optimized_cited": 0, "total": 0}
        doc_stats[did]["total"] += 1
        if r["original"]["is_cited"]:
            doc_stats[did]["original_cited"] += 1
        if r["optimized"]["is_cited"]:
            doc_stats[did]["optimized_cited"] += 1

    summary = {
        "model": MODEL,
        "mode": "url_browsing",
        "base_url": BASE_URL,
        "num_competitors": NUM_COMPETITORS,
        "total_queries": total,
        "overall": {
            "original_citation_rate": original_cited / total if total else 0,
            "optimized_citation_rate": optimized_cited / total if total else 0,
            "delta": (optimized_cited - original_cited) / total if total else 0,
            "original_cited_count": original_cited,
            "optimized_cited_count": optimized_cited,
        },
        "per_doc": {
            did: {
                "original_citation_rate": s["original_cited"] / s["total"],
                "optimized_citation_rate": s["optimized_cited"] / s["total"],
                "delta": (s["optimized_cited"] - s["original_cited"]) / s["total"],
                "queries": s["total"],
            }
            for did, s in doc_stats.items()
        },
    }

    output = {"summary": summary, "results": results}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # ── Print report ──
    print(f"\n{'='*60}")
    print(f"GPT-5 Citation Test Results (URL Browsing)")
    print(f"{'='*60}")
    print(f"Total queries: {total}")
    print(f"Original citation rate:  {original_cited}/{total} = {original_cited/total:.1%}")
    print(f"Optimized citation rate: {optimized_cited}/{total} = {optimized_cited/total:.1%}")
    print(f"Delta: {(optimized_cited-original_cited)/total:+.1%}")
    print(f"\nPer-document breakdown:")
    print(f"{'Doc ID':<14} {'Original':>10} {'Optimized':>10} {'Delta':>8}")
    print(f"{'-'*14} {'-'*10} {'-'*10} {'-'*8}")
    for did, s in doc_stats.items():
        o_rate = s["original_cited"] / s["total"]
        p_rate = s["optimized_cited"] / s["total"]
        print(f"{did:<14} {o_rate:>9.0%} {p_rate:>10.0%} {p_rate-o_rate:>+7.0%}")
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
