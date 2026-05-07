#!/usr/bin/env python3
"""
Build an optimization input dataset from data/page_optim_split.json.

The generated Parquet file matches the existing scripts/run_optimization.py
schema: one row per target URL with raw_html, train_queries, and test_queries.
"""
import argparse
import hashlib
import html
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent.parent


def stable_doc_id(url: str, index: int) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"page_optim_{index:04d}_{digest}"


def cache_path_for_url(cache_dir: Path, url: str, suffix: str = ".html") -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}{suffix}"


def is_reddit_url(url: str) -> bool:
    return "reddit.com" in urlparse(url).netloc.lower()


def extract_queries(items: Iterable[Dict[str, Any]], dedupe: bool = True) -> List[str]:
    queries: List[str] = []
    seen = set()
    for item in items or []:
        query = (item.get("query") or "").strip()
        if not query:
            continue
        if dedupe and query in seen:
            continue
        seen.add(query)
        queries.append(query)
    return queries


def collect_refs(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    for item in items or []:
        refs.extend(item.get("refs") or [])
    return refs


def load_aggregate_info(aggregate_path: Optional[Path], url: str) -> Dict[str, Any]:
    if not aggregate_path:
        return {}
    with aggregate_path.open("r", encoding="utf-8") as f:
        aggregate = json.load(f)
    info = aggregate.get(url, {})
    return {
        "aggregate_n_surfaced_queries": info.get("n_surfaced_queries"),
        "aggregate_n_cited_queries": info.get("n_cited_queries"),
    }


def pdf_bytes_to_html(content: bytes, source_url: str, max_pages: int = 25) -> str:
    text = ""
    try:
        import fitz  # PyMuPDF, available in the local workspace environment

        with fitz.open(stream=content, filetype="pdf") as doc:
            pages = [doc.load_page(i).get_text("text") for i in range(min(len(doc), max_pages))]
        text = "\n\n".join(page.strip() for page in pages if page.strip())
    except Exception as exc:
        LOGGER.warning("PyMuPDF PDF extraction failed for %s: %s", source_url, exc)
        try:
            import io
            import pdfplumber

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages = [page.extract_text() or "" for page in pdf.pages[:max_pages]]
            text = "\n\n".join(page.strip() for page in pages if page.strip())
        except Exception as fallback_exc:
            LOGGER.warning("pdfplumber PDF extraction failed for %s: %s", source_url, fallback_exc)

    if not text:
        text = f"PDF document downloaded from {source_url}, but text extraction failed."

    body = "\n".join(f"<p>{html.escape(block)}</p>" for block in text.split("\n\n") if block.strip())
    page_note = ""
    if max_pages:
        page_note = f"<p>PDF text extraction was capped at the first {max_pages} pages.</p>"

    return (
        "<!doctype html><html><head>"
        f"<title>{html.escape(source_url)}</title>"
        "</head><body>"
        f"<h1>Source document: {html.escape(source_url)}</h1>"
        f"{page_note}{body}</body></html>"
    )


def wrap_text_as_html(text: str, source_url: str, title: str = "") -> str:
    escaped_title = html.escape(title or source_url)
    paragraphs = "\n".join(
        f"<p>{html.escape(block)}</p>" for block in text.split("\n\n") if block.strip()
    )
    return (
        "<!doctype html><html><head>"
        f"<title>{escaped_title}</title>"
        "</head><body>"
        f"<h1>{escaped_title}</h1>"
        f"{paragraphs}</body></html>"
    )


def fetch_raw_html(
    url: str,
    cache_dir: Path,
    timeout: int,
    refresh: bool = False,
    pdf_max_pages: int = 25,
) -> Tuple[str, Dict[str, Any]]:
    parsed = urlparse(url)
    is_pdf_url = parsed.path.lower().endswith(".pdf")
    cache_path = cache_path_for_url(cache_dir, url, ".pdf" if is_pdf_url else ".html")

    if cache_path.exists() and not refresh:
        if cache_path.suffix == ".pdf":
            content = cache_path.read_bytes()
            return pdf_bytes_to_html(content, url, max_pages=pdf_max_pages), {
                "html_fetch_status": "cache_pdf",
                "html_cache_path": str(cache_path),
            }
        return cache_path.read_text(encoding="utf-8", errors="replace"), {
            "html_fetch_status": "cache_html",
            "html_cache_path": str(cache_path),
        }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; AgentGEO/1.0; "
            "+https://github.com/ch1nyzzz/AgentGEO)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf;q=0.8,*/*;q=0.7",
    }
    response = requests.get(url, headers=headers, timeout=(10, timeout))
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()

    if "application/pdf" in content_type or is_pdf_url:
        cache_path = cache_path_for_url(cache_dir, url, ".pdf")
        cache_path.write_bytes(response.content)
        return pdf_bytes_to_html(response.content, url, max_pages=pdf_max_pages), {
            "html_fetch_status": "fetched_pdf",
            "http_status": response.status_code,
            "content_type": content_type,
            "html_cache_path": str(cache_path),
        }

    text = response.text or response.content.decode(response.encoding or "utf-8", errors="replace")
    if "<html" not in text[:1000].lower():
        text = wrap_text_as_html(text, url, title=url)

    cache_path = cache_path_for_url(cache_dir, url, ".html")
    cache_path.write_text(text, encoding="utf-8")
    return text, {
        "html_fetch_status": "fetched_html",
        "http_status": response.status_code,
        "content_type": content_type,
        "html_cache_path": str(cache_path),
    }


def build_rows(
    split_path: Path,
    limit: int,
    offset: int,
    cache_dir: Path,
    timeout: int,
    aggregate_path: Optional[Path],
    no_fetch: bool,
    refresh: bool,
    dedupe_queries: bool,
    pdf_max_pages: int,
    skip_pdf: bool,
    skip_reddit: bool,
    skip_fetch_error: bool,
) -> List[Dict[str, Any]]:
    with split_path.open("r", encoding="utf-8") as f:
        split = json.load(f)

    rows: List[Dict[str, Any]] = []
    cache_dir.mkdir(parents=True, exist_ok=True)

    aggregate_cache: Optional[Dict[str, Any]] = None
    if aggregate_path:
        with aggregate_path.open("r", encoding="utf-8") as f:
            aggregate_cache = json.load(f)

    for local_index, item in enumerate(split[offset:], start=offset):
        if limit and len(rows) >= limit:
            break

        url = item["url"]
        doc_id = stable_doc_id(url, local_index)
        train_queries = extract_queries(item.get("train", []), dedupe=dedupe_queries)
        test_queries = extract_queries(item.get("test", []), dedupe=dedupe_queries)

        is_pdf_url = urlparse(url).path.lower().endswith(".pdf")
        if skip_pdf and is_pdf_url:
            LOGGER.info("Skipping PDF target URL: %s", url)
            continue
        if skip_reddit and is_reddit_url(url):
            LOGGER.info("Skipping Reddit target URL: %s", url)
            continue

        if no_fetch:
            raw_html = wrap_text_as_html(
                f"Placeholder page for {url}. Rebuild without --no-fetch before running optimization.",
                url,
            )
            fetch_meta = {"html_fetch_status": "placeholder"}
        else:
            try:
                raw_html, fetch_meta = fetch_raw_html(
                    url,
                    cache_dir,
                    timeout,
                    refresh=refresh,
                    pdf_max_pages=pdf_max_pages,
                )
            except Exception as exc:
                LOGGER.warning("Failed to fetch %s: %s", url, exc)
                if skip_fetch_error:
                    LOGGER.info("Skipping fetch-error target URL: %s", url)
                    continue
                raw_html = wrap_text_as_html(
                    f"Fetch failed for {url}. Error: {type(exc).__name__}: {exc}",
                    url,
                )
                fetch_meta = {
                    "html_fetch_status": "fetch_error",
                    "fetch_error": f"{type(exc).__name__}: {exc}",
                }

        aggregate_info = {}
        if aggregate_cache is not None:
            info = aggregate_cache.get(url, {})
            aggregate_info = {
                "aggregate_n_surfaced_queries": info.get("n_surfaced_queries"),
                "aggregate_n_cited_queries": info.get("n_cited_queries"),
            }

        row = {
            "doc_id": doc_id,
            "url": url,
            "raw_html": raw_html,
            "train_queries": train_queries,
            "test_queries": test_queries,
            "n_train": len(train_queries),
            "n_test": len(test_queries),
            "n_surfaced_queries": item.get("n_surfaced_queries"),
            "source_split_index": local_index,
            "train_refs": collect_refs(item.get("train", [])),
            "test_refs": collect_refs(item.get("test", [])),
            **fetch_meta,
            **aggregate_info,
        }
        rows.append(row)
        LOGGER.info(
            "Prepared %s: train=%d test=%d status=%s",
            doc_id,
            len(train_queries),
            len(test_queries),
            fetch_meta.get("html_fetch_status"),
        )

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build page optimization input from page_optim_split.json")
    parser.add_argument("--split", default="data/page_optim_split.json", help="Path to page optimization split JSON")
    parser.add_argument("--aggregate", default="data/aggregate_per_url.json", help="Optional aggregate_per_url JSON")
    parser.add_argument("--output", default="data/page_optim_top10.parquet", help="Output Parquet path")
    parser.add_argument("--manifest", default="data/page_optim_top10_manifest.json", help="Output manifest JSON path")
    parser.add_argument("--limit", type=int, default=10, help="Number of URLs to include; 0 means all")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in the split file")
    parser.add_argument("--cache-dir", default="data/page_optim_html_cache", help="Downloaded HTML/PDF cache directory")
    parser.add_argument("--timeout", type=int, default=25, help="HTTP timeout in seconds")
    parser.add_argument("--pdf-max-pages", type=int, default=25, help="Maximum PDF pages to extract")
    parser.add_argument("--skip-pdf", action="store_true", help="Exclude PDF target URLs from the output")
    parser.add_argument("--skip-reddit", action="store_true", help="Exclude Reddit target URLs from the output")
    parser.add_argument("--skip-fetch-error", action="store_true", help="Exclude target URLs whose HTML fetch fails")
    parser.add_argument("--no-fetch", action="store_true", help="Create schema-valid placeholder rows without HTTP fetch")
    parser.add_argument("--refresh", action="store_true", help="Re-fetch URLs even when cache files exist")
    parser.add_argument("--keep-duplicate-queries", action="store_true", help="Preserve duplicate query strings within a split")
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    split_path = resolve_repo_path(args.split)
    aggregate_path = resolve_repo_path(args.aggregate) if args.aggregate else None
    output_path = resolve_repo_path(args.output)
    manifest_path = resolve_repo_path(args.manifest)
    cache_dir = resolve_repo_path(args.cache_dir)

    rows = build_rows(
        split_path=split_path,
        limit=args.limit,
        offset=args.offset,
        cache_dir=cache_dir,
        timeout=args.timeout,
        aggregate_path=aggregate_path if aggregate_path and aggregate_path.exists() else None,
        no_fetch=args.no_fetch,
        refresh=args.refresh,
        dedupe_queries=not args.keep_duplicate_queries,
        pdf_max_pages=args.pdf_max_pages,
        skip_pdf=args.skip_pdf,
        skip_reddit=args.skip_reddit,
        skip_fetch_error=args.skip_fetch_error,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)

    manifest = {
        "output": str(output_path),
        "rows": len(rows),
        "total_train_queries": sum(len(row["train_queries"]) for row in rows),
        "total_test_queries": sum(len(row["test_queries"]) for row in rows),
        "documents": [
            {
                "doc_id": row["doc_id"],
                "url": row["url"],
                "n_train": row["n_train"],
                "n_test": row["n_test"],
                "html_fetch_status": row["html_fetch_status"],
                "html_cache_path": row.get("html_cache_path"),
            }
            for row in rows
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    LOGGER.info("Wrote %d rows to %s", len(rows), output_path)
    LOGGER.info("Wrote manifest to %s", manifest_path)
    LOGGER.info(
        "Run optimization with: python scripts/run_optimization.py --method agentgeo --data %s --output-dir outputs/page_optim_top10 --force-restart",
        output_path.relative_to(REPO_ROOT) if output_path.is_relative_to(REPO_ROOT) else output_path,
    )


if __name__ == "__main__":
    main()
