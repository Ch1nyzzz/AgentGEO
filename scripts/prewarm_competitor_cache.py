"""
Prewarm the on-disk competitor cache so optimization runs need no live ChatNoir calls.

Two gaps are filled, both into the configured disk cache directory:

  1. Search records: queries whose ``{sha256(query)}.json`` is absent are restored
     from ``search_records.tar.gz``.
  2. Competitor HTML: every ``{uuid}.html`` referenced by those search records but
     missing from the cache is fetched from ChatNoir.

Documents ChatNoir no longer serves are recorded in ``_prewarm_missing.json`` and
skipped on later runs (use --retry-missing to try them again).

Usage:
    python scripts/prewarm_competitor_cache.py
    python scripts/prewarm_competitor_cache.py --doc-limit 5 --dry-run
"""
import argparse
import hashlib
import json
import logging
import sys
import tarfile
import threading
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from geo_agent.search_engine.chatnoir import ChatNoirClient  # noqa: E402

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("geo_agent.search_engine.chatnoir").setLevel(logging.ERROR)

ARCHIVE = Path("/Users/erv1n/autoGEO_reproduce/experiments/search_records.tar.gz")
MISSING_MANIFEST = "_prewarm_missing.json"


def cache_key(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_queries(data_cfg: dict, fields: list) -> list:
    df = pd.read_parquet(REPO_ROOT / data_cfg["input_path"])
    offset = data_cfg.get("doc_offset", 0)
    limit = data_cfg.get("doc_limit")
    df = df.iloc[offset: offset + limit] if limit else df.iloc[offset:]
    queries = []
    for _, row in df.iterrows():
        for field in fields:
            queries.extend(list(row[field]))
    return queries


def restore_search_records(queries: list, cache_dir: Path, dry_run: bool) -> dict:
    """Restore missing search records from the archive. Returns {hash: records}."""
    wanted = {cache_key(q) for q in queries}
    records, absent = {}, set()
    for h in wanted:
        path = cache_dir / f"{h}.json"
        if path.exists():
            try:
                records[h] = json.loads(path.read_text(encoding="utf-8"))
                continue
            except json.JSONDecodeError:
                logger.warning(f"Corrupt search record {h}.json; will restore from archive")
        absent.add(h)

    if not absent:
        logger.info(f"Search records: {len(records)}/{len(wanted)} already cached, nothing to restore")
        return records

    if not ARCHIVE.exists():
        logger.error(f"{len(absent)} search records missing and archive not found at {ARCHIVE}")
        return records

    restored = 0
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            stem = Path(member.name).stem
            if stem not in absent:
                continue
            try:
                payload = json.load(tf.extractfile(member))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(f"Failed to read {stem} from archive: {exc}")
                continue
            records[stem] = payload
            if not dry_run:
                (cache_dir / f"{stem}.json").write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            restored += 1

    logger.info(
        f"Search records: {len(records)}/{len(wanted)} available "
        f"({restored} restored from archive, {len(absent) - restored} unavailable)"
    )
    return records


def fetch_search_records(queries: list, have: set, cache_dir: Path, concurrency: int) -> tuple:
    """Search ChatNoir for queries with no cached record. Returns (records, empty_queries).

    Uses SearchManager so the provider and max_results match what the optimizer itself
    would issue; records are written in the same shape the optimizer caches.
    """
    from geo_agent.search_engine.manager import SearchManager

    manager = SearchManager(str(REPO_ROOT / "geo_agent" / "config.yaml"))
    todo = [q for q in dict.fromkeys(queries) if cache_key(q) not in have]
    records, empty = {}, []
    lock = threading.Lock()
    done = 0

    def search(query: str):
        try:
            return query, manager.search(query), None
        except Exception as exc:
            return query, None, exc

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for future in as_completed([pool.submit(search, q) for q in todo]):
            query, results, exc = future.result()
            with lock:
                done += 1
                if exc is not None:
                    logger.warning(f"[{done}/{len(todo)}] search failed: {exc}")
                    empty.append(query)
                elif results:
                    payload = [r.model_dump() if hasattr(r, "model_dump") else r.dict() for r in results]
                    (cache_dir / f"{cache_key(query)}.json").write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    records[cache_key(query)] = payload
                else:
                    empty.append(query)
                if done % 200 == 0:
                    logger.info(f"Search progress: {done}/{len(todo)}  ok={len(records)}  empty={len(empty)}")

    logger.info(f"Searched {len(todo)} queries: {len(records)} returned results, {len(empty)} returned nothing")
    return records, empty


def collect_missing_uuids(records: dict, cache_dir: Path) -> list:
    """UUIDs referenced by the search records whose HTML is not on disk (order preserved)."""
    seen, missing = set(), []
    for payload in records.values():
        items = payload if isinstance(payload, list) else payload.get("results", [])
        for item in items:
            if not isinstance(item, dict):
                continue
            uuid = item.get("uuid")
            if not uuid or uuid in seen:
                continue
            seen.add(uuid)
            if not (cache_dir / f"{uuid}.html").exists():
                missing.append(uuid)
    return missing


def fetch_all(uuids: list, cache_dir: Path, concurrency: int) -> tuple:
    """Fetch HTML for each uuid into the cache. Returns (fetched, unavailable)."""
    client = ChatNoirClient()
    fetched, unavailable = [], []
    lock = threading.Lock()
    done = 0

    def fetch(uuid: str):
        try:
            return uuid, client.get_html_content(uuid), None
        except Exception as exc:
            return uuid, "", exc

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fetch, u) for u in uuids]
        for future in as_completed(futures):
            uuid, html, exc = future.result()
            with lock:
                done += 1
                if exc is not None:
                    unavailable.append(uuid)
                    logger.warning(f"[{done}/{len(uuids)}] {uuid} failed: {exc}")
                elif html:
                    (cache_dir / f"{uuid}.html").write_text(html, encoding="utf-8")
                    fetched.append(uuid)
                else:
                    unavailable.append(uuid)
                if done % 200 == 0:
                    logger.info(f"Progress: {done}/{len(uuids)}  fetched={len(fetched)}  unavailable={len(unavailable)}")

    return fetched, unavailable


def report_coverage(records: dict, cache_dir: Path, target_count: int):
    """Per-query competitor availability — the number that actually matters."""
    available = {p.stem for p in cache_dir.glob("*.html")}
    counts = []
    for payload in records.values():
        items = payload if isinstance(payload, list) else payload.get("results", [])
        uuids = [i.get("uuid") for i in items if isinstance(i, dict) and i.get("uuid")]
        counts.append(sum(u in available for u in uuids))
    if not counts:
        return
    full = sum(c >= target_count for c in counts)
    empty = sum(c == 0 for c in counts)
    logger.info("=" * 60)
    logger.info(f"Queries:              {len(counts)}")
    logger.info(f"Mean competitors:     {sum(counts) / len(counts):.2f} / {target_count}")
    logger.info(f"Queries at full {target_count}:    {full} ({full / len(counts) * 100:.1f}%)")
    logger.info(f"Queries with zero:    {empty} ({empty / len(counts) * 100:.1f}%)")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prewarm the competitor cache for offline optimization runs")
    parser.add_argument("--config", default="optimization_config.yaml")
    parser.add_argument("--fields", default="train_queries,test_queries",
                        help="Comma-separated query fields to cover")
    parser.add_argument("--data", help="Override data.input_path")
    parser.add_argument("--doc-limit", type=int, help="Override data.doc_limit")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--target-count", type=int, default=10,
                        help="Competitors per query the optimizer asks for (for the coverage report)")
    parser.add_argument("--retry-missing", action="store_true",
                        help="Retry UUIDs previously recorded as unavailable")
    parser.add_argument("--dry-run", action="store_true", help="Report the gap without fetching")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        config_path = REPO_ROOT / args.config
    config = load_config(config_path)

    cache_dir = config.get("agentgeo", {}).get("disk_cache_dir")
    if not cache_dir:
        logger.error("agentgeo.disk_cache_dir is not set; nothing to prewarm")
        return 1
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Cache directory: {cache_dir}")

    data_cfg = dict(config["data"])
    if args.data:
        data_cfg["input_path"] = args.data
        data_cfg["doc_limit"] = None
        data_cfg["doc_offset"] = 0
    if args.doc_limit is not None:
        data_cfg["doc_limit"] = args.doc_limit

    queries = collect_queries(data_cfg, args.fields.split(","))
    logger.info(f"Covering {len(queries)} queries ({len(set(queries))} unique) from {args.fields}")

    records = restore_search_records(queries, cache_dir, args.dry_run)

    # Queries with no cached or archived record must be searched live, otherwise the
    # optimizer would run them with an empty candidate set and silently score them 0.
    if not args.dry_run:
        fresh, empty = fetch_search_records(queries, set(records), cache_dir, args.concurrency)
        records.update(fresh)
        if empty:
            logger.warning(f"{len(empty)} queries returned no search results at all")
    if not records:
        logger.error("No search records available; cannot determine competitors")
        return 1

    missing = collect_missing_uuids(records, cache_dir)

    manifest_path = cache_dir / MISSING_MANIFEST
    known_missing = set()
    if manifest_path.exists() and not args.retry_missing:
        try:
            known_missing = set(json.loads(manifest_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    skipped = [u for u in missing if u in known_missing]
    missing = [u for u in missing if u not in known_missing]

    logger.info(f"Competitor HTML to fetch: {len(missing)}" +
                (f" ({len(skipped)} previously unavailable, skipped)" if skipped else ""))

    if args.dry_run:
        report_coverage(records, cache_dir, args.target_count)
        return 0

    if missing:
        fetched, unavailable = fetch_all(missing, cache_dir, args.concurrency)
        logger.info(f"Fetched {len(fetched)}, unavailable {len(unavailable)}")
        if unavailable:
            manifest_path.write_text(
                json.dumps(sorted(known_missing | set(unavailable)), indent=2), encoding="utf-8"
            )
            logger.info(f"Recorded {len(unavailable)} unavailable UUIDs in {manifest_path.name}")

    report_coverage(records, cache_dir, args.target_count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
