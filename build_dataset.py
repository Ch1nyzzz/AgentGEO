"""Build a dataset of (query -> sources, citations) using gpt-5-mini + web_search,
then aggregate per URL: which queries surfaced it vs which queries cited it.

Resume + dedup: each unique query string is run at most once across the whole
dataset, even if the same string appears under multiple (uuid, split, idx).
The JSONL stores one record per unique query; the aggregate step expands each
query back to all its (uuid, split, idx) provenance refs.

Usage:
  python build_dataset.py                       # full run, resumes from existing jsonl
  python build_dataset.py --limit 5             # smoke test on N pending queries
  python build_dataset.py --aggregate-only      # rebuild aggregate from existing jsonl
"""
import argparse
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai_client import OpenAIClient

DEFAULT_QUERIES_DIR = Path(
    "processed_dedup"
)
DEFAULT_OUT_DIR = Path("/Users/zhihuat/Codebases/AgentGEO_RE/data")


def load_query_index(queries_dir):
    """Return (unique_queries, query_to_refs).

    unique_queries: list of distinct query strings, in stable first-seen order.
    query_to_refs:  query string -> list of {uuid, split, idx} dicts (every
                    place that query appeared in the source files).
    """
    seen_order = []
    refs_by_query = defaultdict(list)
    for f in sorted(queries_dir.glob("*.json")):
        d = json.loads(f.read_text())
        uuid = d["uuid"]
        for split in ("train", "test"):
            for idx, q in enumerate(d.get("dataset", {}).get(split, [])):
                if q not in refs_by_query:
                    seen_order.append(q)
                refs_by_query[q].append({"uuid": uuid, "split": split, "idx": idx})
    return seen_order, dict(refs_by_query)


def load_done_queries(jsonl_path):
    """Set of query strings whose successful result is already on disk."""
    if not jsonl_path.exists():
        return set()
    done = set()
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue  # allow retry on next run
            q = rec.get("query")
            if q is not None:
                done.add(q)
    return done


def run_one(client, query, model, reasoning_effort):
    try:
        res = client.search(query, model=model, reasoning_effort=reasoning_effort)
        return {"query": query, **res}
    except Exception as e:
        return {"query": query, "error": f"{type(e).__name__}: {e}"}


def build_aggregate(jsonl_path, query_to_refs, agg_path, focus_path):
    """For each URL: which queries surfaced it / cited it (and the (uuid, split,
    idx) provenance for each query)."""
    surfaced_queries = defaultdict(set)
    cited_queries = defaultdict(set)
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("error"):
                continue
            q = rec["query"]
            for url in rec.get("sources", []):
                surfaced_queries[url].add(q)
            for url in set(rec.get("citations", [])):
                cited_queries[url].add(q)

    def expand(queries):
        return [
            {"query": q, "refs": query_to_refs.get(q, [])}
            for q in sorted(queries)
        ]

    agg = {}
    for url in set(surfaced_queries) | set(cited_queries):
        agg[url] = {
            "n_surfaced_queries": len(surfaced_queries[url]),
            "n_cited_queries": len(cited_queries[url]),
            "surfaced_by": expand(surfaced_queries[url]),
            "cited_by": expand(cited_queries[url]),
        }
    agg_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2))

    focus = sorted(
        [
            {
                "url": url,
                "n_surfaced_queries": v["n_surfaced_queries"],
                "surfaced_by": v["surfaced_by"],
            }
            for url, v in agg.items()
            if v["n_cited_queries"] == 0 and v["n_surfaced_queries"] > 0
        ],
        key=lambda x: x["n_surfaced_queries"],
        reverse=True,
    )
    focus_path.write_text(json.dumps(focus, ensure_ascii=False, indent=2))
    return agg, focus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--queries-dir", type=Path, default=DEFAULT_QUERIES_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--model", default="gpt-5-mini")
    p.add_argument(
        "--reasoning-effort",
        default="low",
        help="web_search rejects 'minimal'; 'low' is the lowest viable setting",
    )
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N pending queries (use for smoke tests)",
    )
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Skip API calls, just rebuild aggregate from existing jsonl",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.out_dir / "per_query_results.jsonl"
    agg_path = args.out_dir / "aggregate_per_url.json"
    focus_path = args.out_dir / "surfaced_not_cited.json"

    unique_queries, query_to_refs = load_query_index(args.queries_dir)
    n_total_refs = sum(len(v) for v in query_to_refs.values())

    if not args.aggregate_only:
        done = load_done_queries(jsonl_path)
        pending = [q for q in unique_queries if q not in done]
        if args.limit is not None:
            pending = pending[: args.limit]
        print(
            f"unique queries: {len(unique_queries)} (across {n_total_refs} refs), "
            f"done: {len(done)}, pending: {len(pending)}",
            flush=True,
        )

        client = OpenAIClient()
        write_lock = threading.Lock()
        n_ok = 0
        n_err = 0
        with jsonl_path.open("a") as out, ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as ex:
            futures = {
                ex.submit(run_one, client, q, args.model, args.reasoning_effort): q
                for q in pending
            }
            for i, fut in enumerate(as_completed(futures), 1):
                rec = fut.result()
                with write_lock:
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out.flush()
                if rec.get("error"):
                    n_err += 1
                    preview = rec["query"][:60].replace("\n", " ")
                    print(
                        f"  [err] {preview!r}: {rec['error']}",
                        flush=True,
                    )
                else:
                    n_ok += 1
                if i % 25 == 0:
                    print(
                        f"  progress: {i}/{len(pending)} (ok={n_ok}, err={n_err})",
                        flush=True,
                    )
        print(f"finished: ok={n_ok}, err={n_err}", flush=True)

    print("aggregating...", flush=True)
    agg, focus = build_aggregate(jsonl_path, query_to_refs, agg_path, focus_path)
    print(f"  unique URLs total: {len(agg)}", flush=True)
    print(f"  surfaced-not-cited URLs: {len(focus)}", flush=True)
    if focus[:5]:
        print("  top 5 surfaced-not-cited:")
        for x in focus[:5]:
            print(f"    {x['n_surfaced_queries']:>3}q  {x['url']}")


if __name__ == "__main__":
    main()
