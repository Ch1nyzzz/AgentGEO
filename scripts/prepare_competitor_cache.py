"""
Prepare competitor cache for build_html_showcase.

For each (doc, query) in sample_50.parquet:
  1. Extract {sha256(query)}.json from search_records.tar.gz into CACHE_DIR
  2. Copy existing {uuid}.html files from e2e_comparison/cache
  3. Fetch missing URLs concurrently and save as {uuid}.html

Output: cache/competitors/ ready to use as CACHE_DIR.
"""
import asyncio
import hashlib
import json
import shutil
import tarfile
from collections import Counter
from pathlib import Path

import httpx
import pandas as pd

ARCHIVE = Path("/Users/erv1n/autoGEO_reproduce/experiments/search_records.tar.gz")
SOURCE_HTML_CACHE = Path("/Users/erv1n/autoGEO_reproduce/experiments/e2e_comparison/experiments/cache")
TARGET_CACHE = Path("/Users/erv1n/AgentGEO/cache/competitors")

NUM_QUERIES = 5
NUM_COMPETITORS = 10
CONCURRENCY = 32
TIMEOUT = 20.0


def collect_targets():
    opt_df = pd.read_parquet("/Users/erv1n/AgentGEO/sample_50.parquet")
    input_df = pd.read_parquet("/Users/erv1n/AgentGEO/data/input.parquet")
    df = opt_df.merge(input_df[["doc_id", "test_queries"]], on="doc_id", how="left")

    queries = []
    for _, row in df.iterrows():
        for q in list(row["test_queries"])[:NUM_QUERIES]:
            queries.append(q)
    return queries


def extract_jsons(query_hashes: set[str]) -> dict[str, list]:
    """Extract matched JSONs from archive, also write to cache dir."""
    extracted = {}
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            h = member.name.replace(".json", "")
            if h not in query_hashes:
                continue
            data = json.load(tf.extractfile(member))
            extracted[h] = data
            (TARGET_CACHE / f"{h}.json").write_text(
                json.dumps(data, ensure_ascii=False), encoding="utf-8"
            )
    return extracted


def copy_existing_htmls(needed_uuids: set[str]) -> int:
    """Copy {uuid}.html from e2e_comparison cache if exists."""
    copied = 0
    for uuid in needed_uuids:
        src = SOURCE_HTML_CACHE / f"{uuid}.html"
        if src.exists():
            shutil.copy2(src, TARGET_CACHE / f"{uuid}.html")
            copied += 1
    return copied


async def fetch_one(client: httpx.AsyncClient, sem: asyncio.Semaphore, uuid: str, url: str) -> tuple[str, str, int]:
    """Fetch url, return (uuid, status, bytes). status: 'ok' / 'error:<reason>'."""
    async with sem:
        try:
            r = await client.get(url, timeout=TIMEOUT, follow_redirects=True)
            if r.status_code == 200 and r.text:
                (TARGET_CACHE / f"{uuid}.html").write_text(r.text, encoding="utf-8")
                return uuid, "ok", len(r.text)
            return uuid, f"error:http_{r.status_code}", 0
        except Exception as e:
            etype = type(e).__name__
            return uuid, f"error:{etype}", 0


async def fetch_missing(missing_pairs: list[tuple[str, str]]):
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    }
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient(headers=headers, http2=False) as client:
        tasks = [fetch_one(client, sem, uuid, url) for uuid, url in missing_pairs]
        results = []
        done = 0
        for fut in asyncio.as_completed(tasks):
            r = await fut
            results.append(r)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                ok = sum(1 for x in results if x[1] == "ok")
                print(f"  fetched {done}/{len(tasks)}  ok={ok}")
        return results


def main():
    TARGET_CACHE.mkdir(parents=True, exist_ok=True)
    queries = collect_targets()
    query_hashes = {hashlib.sha256(q.encode("utf-8")).hexdigest() for q in queries}
    print(f"Targeting {len(query_hashes)} unique queries")

    # 1. Extract JSONs
    print("\n[1/3] Extracting search records JSONs...")
    extracted = extract_jsons(query_hashes)
    print(f"  extracted {len(extracted)}/{len(query_hashes)} JSONs to {TARGET_CACHE}")

    # Collect needed uuid + url
    needed = {}  # uuid -> url
    for h, data in extracted.items():
        for item in data[:NUM_COMPETITORS]:
            uuid = item.get("uuid")
            url = item.get("url")
            if uuid and url:
                needed[uuid] = url
    print(f"  unique competitor uuids needed: {len(needed)}")

    # 2. Copy existing HTMLs
    print("\n[2/3] Copying existing HTMLs from e2e_comparison/cache...")
    copied = copy_existing_htmls(set(needed.keys()))
    print(f"  copied {copied}/{len(needed)} HTMLs")

    # 3. Fetch missing
    missing = [(uuid, url) for uuid, url in needed.items()
               if not (TARGET_CACHE / f"{uuid}.html").exists()]
    print(f"\n[3/3] Fetching {len(missing)} missing URLs (concurrency={CONCURRENCY})...")
    results = asyncio.run(fetch_missing(missing))

    # Summary
    status_counter = Counter(s for _, s, _ in results)
    ok = status_counter.get("ok", 0)
    print(f"\n=== Fetch summary ===")
    print(f"  ok: {ok}/{len(missing)} ({ok/len(missing)*100:.1f}%)" if missing else "  no missing")
    for s, n in status_counter.most_common():
        if s != "ok":
            print(f"  {s}: {n}")

    # Final coverage
    total_html = len(list(TARGET_CACHE.glob("*.html")))
    print(f"\n=== Cache final state ===")
    print(f"  {TARGET_CACHE}")
    print(f"  JSONs: {len(list(TARGET_CACHE.glob('*.json')))}")
    print(f"  HTMLs: {total_html}/{len(needed)}")

    # Per-query coverage
    full10 = at_least_5 = zero = 0
    for h, data in extracted.items():
        avail = sum(1 for item in data[:NUM_COMPETITORS]
                    if item.get("uuid") and (TARGET_CACHE / f"{item['uuid']}.html").exists())
        if avail >= 10: full10 += 1
        if avail >= 5: at_least_5 += 1
        if avail == 0: zero += 1
    print(f"\nPer-query competitor availability:")
    print(f"  >=10: {full10}/{len(extracted)}")
    print(f"  >=5:  {at_least_5}/{len(extracted)}")
    print(f"  =0:   {zero}/{len(extracted)}")


if __name__ == "__main__":
    main()
