"""
Build HTML showcase for AgentGEO optimized documents.
For each of 20 docs, pick 2 test queries.
For each query, save: original HTML, optimized HTML, top 3 competitor HTMLs.
Output: docs/showcase/ directory with index.html and per-doc/query folders.
"""

import json
import hashlib
import os
import re
from pathlib import Path

import pandas as pd

# ── Config ──
NUM_DOCS = 20
NUM_QUERIES = 2
NUM_COMPETITORS = 3
CACHE_DIR = Path("/Users/erv1n/autoGEO_reproduce/experiments/cache")
OUTPUT_DIR = Path("docs/showcase")

# ── Load data ──
opt_results = json.load(open("outputs/optimization_results_20260215_175657.json"))
input_df = pd.read_parquet("data/input.parquet")

# Sort by delta (best improvements first) and pick top 20
opt_results.sort(key=lambda x: x.get("agentgeo_delta_test_citation_rate", 0), reverse=True)
selected = opt_results[:NUM_DOCS]


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name)[:80].strip("_").lower()


def read_competitor_html(uuid: str) -> str:
    """Read raw HTML for a competitor doc from cache."""
    path = CACHE_DIR / f"{uuid}.html"
    if path.exists():
        return path.read_text(encoding="utf-8", errors="replace")
    return f"<html><body><p>HTML not available for uuid: {uuid}</p></body></html>"


def get_search_results(query: str):
    key = hashlib.sha256(query.encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return json.load(open(cache_path, "r", encoding="utf-8"))
    return []


def wrap_html(title: str, content_html: str, label: str, color: str) -> str:
    """Wrap raw HTML content in a styled container page."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; }}
  .banner {{ background: {color}; color: white; padding: 12px 24px; font-size: 14px; font-weight: 600;
             position: sticky; top: 0; z-index: 1000; display: flex; justify-content: space-between; align-items: center; }}
  .banner a {{ color: white; text-decoration: underline; }}
  .content {{ padding: 0; }}
  iframe {{ width: 100%; height: calc(100vh - 48px); border: none; }}
</style>
</head>
<body>
<div class="banner">
  <span>{label}</span>
  <a href="../index.html">← Back</a>
</div>
<div class="content">
  <iframe srcdoc="{content_html}"></iframe>
</div>
</body>
</html>"""


def escape_for_srcdoc(html: str) -> str:
    """Escape HTML for use in iframe srcdoc attribute."""
    return html.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


# ── Build ──
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

index_entries = []

for doc in selected:
    doc_id = doc["doc_id"]
    row = input_df[input_df["doc_id"] == doc_id]
    if row.empty:
        print(f"  ⚠ doc_id {doc_id} not found in input.parquet, skipping")
        continue
    row = row.iloc[0]

    original_html = row["raw_html"]
    optimized_html = doc["agentgeo_html"]
    test_queries = list(row["test_queries"])[:NUM_QUERIES]
    url = doc["url"]
    baseline_rate = doc.get("agentgeo_baseline_test_citation_rate", 0)
    optimized_rate = doc.get("agentgeo_optimized_test_citation_rate", 0)
    delta = doc.get("agentgeo_delta_test_citation_rate", 0)

    doc_dir = OUTPUT_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    query_entries = []

    for qi, query in enumerate(test_queries):
        q_dir = doc_dir / f"query_{qi+1}"
        q_dir.mkdir(parents=True, exist_ok=True)

        # Save original
        orig_escaped = escape_for_srcdoc(original_html)
        (q_dir / "original.html").write_text(
            wrap_html(f"Original - {query}", orig_escaped, f"🔵 ORIGINAL | Query: {query}", "#2563eb"),
            encoding="utf-8",
        )

        # Save optimized
        opt_escaped = escape_for_srcdoc(optimized_html)
        (q_dir / "optimized.html").write_text(
            wrap_html(f"Optimized - {query}", opt_escaped, f"🟢 OPTIMIZED (AgentGEO) | Query: {query}", "#16a34a"),
            encoding="utf-8",
        )

        # Save competitors
        search_results = get_search_results(query)
        comp_entries = []
        for ci, sr in enumerate(search_results[:NUM_COMPETITORS]):
            comp_html = read_competitor_html(sr["uuid"])
            comp_escaped = escape_for_srcdoc(comp_html)
            comp_title = sr.get("title", f"Competitor {ci+1}")
            comp_url = sr.get("url", "")
            fname = f"competitor_{ci+1}.html"
            (q_dir / fname).write_text(
                wrap_html(
                    f"Competitor {ci+1} - {query}",
                    comp_escaped,
                    f"🔴 COMPETITOR {ci+1}: {comp_title[:60]} | {comp_url[:60]}",
                    "#dc2626",
                ),
                encoding="utf-8",
            )
            comp_entries.append({"title": comp_title, "url": comp_url, "file": fname})

        # Query-level index
        query_index_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Query: {query}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 20px; }} h2 {{ font-size: 16px; color: #555; }}
  .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 12px 0; }}
  .card a {{ text-decoration: none; font-weight: 600; font-size: 15px; }}
  .original a {{ color: #2563eb; }} .optimized a {{ color: #16a34a; }} .competitor a {{ color: #dc2626; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; color: white; margin-right: 8px; }}
  .back {{ color: #6b7280; text-decoration: none; font-size: 14px; }}
</style></head><body>
<a class="back" href="../index.html">← Back to document</a>
<h1>Query: "{query}"</h1>
<div class="card original"><span class="tag" style="background:#2563eb;">Original</span><a href="original.html">View Original Page</a></div>
<div class="card optimized"><span class="tag" style="background:#16a34a;">Optimized</span><a href="optimized.html">View AgentGEO Optimized Page</a></div>
"""
        for ce in comp_entries:
            query_index_html += f'<div class="card competitor"><span class="tag" style="background:#dc2626;">Competitor</span><a href="{ce["file"]}">{ce["title"][:80]}</a><br><small style="color:#888;">{ce["url"][:80]}</small></div>\n'
        query_index_html += "</body></html>"
        (q_dir / "index.html").write_text(query_index_html, encoding="utf-8")

        query_entries.append({"query": query, "dir": f"query_{qi+1}"})

    # Doc-level index
    doc_index_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Doc: {doc_id}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 20px; }} .url {{ color: #6b7280; font-size: 13px; word-break: break-all; }}
  .metrics {{ display: flex; gap: 16px; margin: 16px 0; }}
  .metric {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px 20px; text-align: center; }}
  .metric .value {{ font-size: 24px; font-weight: 700; }} .metric .label {{ font-size: 12px; color: #6b7280; }}
  .up {{ color: #16a34a; }} .neutral {{ color: #6b7280; }}
  .query-card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 12px 0; }}
  .query-card a {{ text-decoration: none; color: #2563eb; font-weight: 600; }}
  .back {{ color: #6b7280; text-decoration: none; font-size: 14px; }}
</style></head><body>
<a class="back" href="../index.html">← Back to showcase</a>
<h1>Document: {doc_id}</h1>
<p class="url">{url}</p>
<div class="metrics">
  <div class="metric"><div class="value">{baseline_rate:.0%}</div><div class="label">Baseline Citation</div></div>
  <div class="metric"><div class="value up">{optimized_rate:.0%}</div><div class="label">Optimized Citation</div></div>
  <div class="metric"><div class="value {'up' if delta > 0 else 'neutral'}">{'+' if delta > 0 else ''}{delta:.0%}</div><div class="label">Delta</div></div>
</div>
"""
    for qe in query_entries:
        doc_index_html += f'<div class="query-card"><a href="{qe["dir"]}/index.html">Query: "{qe["query"]}"</a></div>\n'
    doc_index_html += "</body></html>"
    (doc_dir / "index.html").write_text(doc_index_html, encoding="utf-8")

    index_entries.append({
        "doc_id": doc_id,
        "url": url,
        "baseline": baseline_rate,
        "optimized": optimized_rate,
        "delta": delta,
        "queries": [qe["query"] for qe in query_entries],
    })
    print(f"✓ {doc_id} ({len(query_entries)} queries)")

# ── Top-level index ──
main_index = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentGEO Showcase</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }
  h1 { font-size: 28px; } .subtitle { color: #6b7280; margin-bottom: 32px; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th { text-align: left; padding: 10px 12px; border-bottom: 2px solid #e5e7eb; font-size: 13px; color: #6b7280; }
  td { padding: 10px 12px; border-bottom: 1px solid #f3f4f6; }
  tr:hover { background: #f9fafb; }
  a { color: #2563eb; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .up { color: #16a34a; font-weight: 600; } .down { color: #dc2626; } .neutral { color: #6b7280; }
  .queries { font-size: 12px; color: #6b7280; }
</style></head><body>
<h1>AgentGEO Showcase</h1>
<p class="subtitle">Comparing original, AgentGEO-optimized, and competitor web pages across 20 documents and 40 queries.</p>
<table>
<thead><tr><th>#</th><th>Document</th><th>Baseline</th><th>Optimized</th><th>Delta</th><th>Queries</th></tr></thead>
<tbody>
"""

for i, entry in enumerate(index_entries):
    delta_class = "up" if entry["delta"] > 0 else ("down" if entry["delta"] < 0 else "neutral")
    delta_sign = "+" if entry["delta"] > 0 else ""
    queries_str = "<br>".join(f'"{q}"' for q in entry["queries"])
    main_index += f"""<tr>
<td>{i+1}</td>
<td><a href="{entry['doc_id']}/index.html">{entry['doc_id']}</a><br><small style="color:#999;">{entry['url'][:60]}...</small></td>
<td>{entry['baseline']:.0%}</td>
<td>{entry['optimized']:.0%}</td>
<td class="{delta_class}">{delta_sign}{entry['delta']:.0%}</td>
<td class="queries">{queries_str}</td>
</tr>\n"""

main_index += "</tbody></table></body></html>"
(OUTPUT_DIR / "index.html").write_text(main_index, encoding="utf-8")

print(f"\n✅ Done! {len(index_entries)} docs, output at {OUTPUT_DIR}/")
print(f"   Open {OUTPUT_DIR}/index.html to browse")
