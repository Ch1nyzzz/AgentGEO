"""
Build raw HTML pages for GitHub Pages deployment + GPT citation testing.
50 docs x 5 queries x up to 10 competitors.

Output:
  docs/pages/{doc_id}/original.html
  docs/pages/{doc_id}/optimized.html
  docs/pages/{doc_id}/query_{n}/competitor_{m}.html
  docs/pages/manifest.json
  docs/pages/index.html
"""

import json
import hashlib
import os
from pathlib import Path

import pandas as pd

# ── Config ──
NUM_DOCS = 50
NUM_QUERIES = 5
NUM_COMPETITORS = 10
MIN_COMPETITORS = 5  # Skip queries with fewer than this many usable competitors
CACHE_DIR = Path(os.getenv("AGENTGEO_CACHE_DIR", "cache/competitors"))
OUTPUT_DIR = Path("docs/pages")


def read_competitor_html(uuid: str) -> str:
    path = CACHE_DIR / f"{uuid}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return ""


def get_search_results(query: str):
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return json.load(open(cache_path, "r", encoding="utf-8"))
    return []


# ── Load data ──
opt_df = pd.read_parquet("sample_50.parquet")
input_df = pd.read_parquet("data/input.parquet")
df = opt_df.merge(input_df[["doc_id", "raw_html", "test_queries"]], on="doc_id", how="left")
df = df.sort_values("agentgeo_delta_test_citation_rate", ascending=False)
# Top-by-delta: keep all 50 docs from sample_50.parquet
selected_df = df.head(NUM_DOCS)

# ── Build ──
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
manifest = []
index_entries = []

for _, row in selected_df.iterrows():
    doc_id = row["doc_id"]
    original_html = row["raw_html"]
    optimized_html = row["agentgeo_html"]
    test_queries = list(row["test_queries"])[:NUM_QUERIES]
    url = row["url"]
    baseline_rate = row.get("agentgeo_baseline_test_citation_rate", 0)
    optimized_rate = row.get("agentgeo_optimized_test_citation_rate", 0)
    delta = row.get("agentgeo_delta_test_citation_rate", 0)

    doc_dir = OUTPUT_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    (doc_dir / "original.html").write_text(original_html, encoding="utf-8")
    (doc_dir / "optimized.html").write_text(optimized_html, encoding="utf-8")

    query_entries = []
    skipped_queries = 0
    for qi, query in enumerate(test_queries):
        q_dir = doc_dir / f"query_{qi+1}"

        search_results = get_search_results(query)
        # Probe usable competitors first; skip query if too few
        usable = [sr for sr in search_results[:NUM_COMPETITORS]
                  if read_competitor_html(sr.get("uuid", ""))]
        if len(usable) < MIN_COMPETITORS:
            skipped_queries += 1
            continue

        q_dir.mkdir(parents=True, exist_ok=True)
        competitor_paths = []
        for ci, sr in enumerate(usable):
            comp_html = read_competitor_html(sr["uuid"])
            fname = f"competitor_{ci+1}.html"
            (q_dir / fname).write_text(comp_html, encoding="utf-8")
            competitor_paths.append({
                "path": f"{doc_id}/query_{qi+1}/{fname}",
                "title": sr.get("title", ""),
                "url": sr.get("url", ""),
            })

        manifest.append({
            "doc_id": doc_id,
            "doc_url": url,
            "query": query,
            "query_index": qi + 1,
            "original_path": f"{doc_id}/original.html",
            "optimized_path": f"{doc_id}/optimized.html",
            "competitors": competitor_paths,
            "baseline_citation_rate": baseline_rate,
            "optimized_citation_rate": optimized_rate,
            "delta": delta,
        })
        query_entries.append({"query": query, "index": qi + 1, "num_competitors": len(competitor_paths)})

    if not query_entries:
        # Doc has no usable queries — skip from index but keep target HTMLs already written
        print(f"⚠ {doc_id} 0 usable queries (skipped {skipped_queries})")
        continue
    index_entries.append({
        "doc_id": doc_id, "url": url,
        "baseline": baseline_rate, "optimized": optimized_rate, "delta": delta,
        "queries": query_entries,
    })
    print(f"✓ {doc_id} ({len(query_entries)} queries, skipped {skipped_queries})")

(OUTPUT_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ── Index page ──
main_index = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentGEO Pages</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 24px; } .subtitle { color: #6b7280; margin-bottom: 24px; }
  .doc { border: 1px solid #e5e7eb; border-radius: 8px; padding: 20px; margin: 16px 0; }
  .doc h2 { font-size: 16px; margin: 0 0 4px 0; }
  .doc .url { color: #6b7280; font-size: 12px; word-break: break-all; }
  .metrics { display: flex; gap: 12px; margin: 12px 0; font-size: 13px; }
  .metrics span { padding: 4px 10px; border-radius: 4px; }
  .baseline { background: #f3f4f6; } .optimized { background: #dcfce7; color: #16a34a; font-weight: 600; }
  .delta { background: #dcfce7; color: #16a34a; font-weight: 600; }
  .pages { display: flex; gap: 8px; margin: 8px 0; }
  .pages a { padding: 4px 12px; border-radius: 4px; font-size: 13px; text-decoration: none; color: white; }
  .pages .orig { background: #2563eb; } .pages .opt { background: #16a34a; }
  .query-list { margin: 12px 0 0 0; padding: 0; list-style: none; font-size: 13px; }
  .query-list li { padding: 6px 0; border-top: 1px solid #f3f4f6; }
  .query-list li:first-child { border-top: none; }
  .query-text { color: #374151; }
  .comp-links { margin-left: 8px; }
  .comp-links a { color: #dc2626; text-decoration: none; font-size: 12px; margin-right: 4px; }
</style></head><body>
<h1>AgentGEO Test Pages</h1>
<p class="subtitle">50 docs x 5 queries x up to 10 competitors — raw HTML for GPT citation testing</p>
"""

for i, entry in enumerate(index_entries):
    d = entry["delta"]
    main_index += f"""<div class="doc">
<h2>#{i+1} {entry['doc_id']}</h2>
<div class="url">{entry['url']}</div>
<div class="metrics">
  <span class="baseline">Baseline: {entry['baseline']:.0%}</span>
  <span class="optimized">Optimized: {entry['optimized']:.0%}</span>
  <span class="delta">{'+'if d>0 else ''}{d:.0%}</span>
</div>
<div class="pages">
  <a class="orig" href="{entry['doc_id']}/original.html">Original</a>
  <a class="opt" href="{entry['doc_id']}/optimized.html">Optimized</a>
</div>
<ul class="query-list">
"""
    for qe in entry["queries"]:
        comp_links = " ".join(
            f'<a href="{entry["doc_id"]}/query_{qe["index"]}/competitor_{c+1}.html">C{c+1}</a>'
            for c in range(qe["num_competitors"])
        )
        main_index += f'<li><span class="query-text">Q{qe["index"]}: "{qe["query"]}"</span><span class="comp-links">{comp_links}</span></li>\n'
    main_index += "</ul></div>\n"

main_index += "</body></html>"
(OUTPUT_DIR / "index.html").write_text(main_index, encoding="utf-8")

print(f"\n✅ Done! {len(index_entries)} docs, {len(manifest)} test cases")
print(f"   Pages: {OUTPUT_DIR}/")
print(f"   Manifest: {OUTPUT_DIR}/manifest.json")
