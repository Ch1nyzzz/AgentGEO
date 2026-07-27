#!/usr/bin/env python3
"""
Export every recorded tool invocation from AgentGEO run outputs into one flat file.

Each record is a single tool call: which diagnosis triggered it, which tool was
selected, why (reasoning), and what it changed. Query-level outcome is attached
so the diagnosis -> tool -> citation chain can be analysed without reopening the
per-document files.

Usage:
    python scripts/export_tool_traces.py                     # all default dirs
    python scripts/export_tool_traces.py -o traces.json      # custom output
    python scripts/export_tool_traces.py --with-content      # include full rewrites
"""
import argparse
import glob
import json
import os
from collections import Counter

DEFAULT_DIRS = [
    "outputs_gpt4.1",
    "outputs_gpt4.1_newdocs",
    "outputs_gpt4.1_newdocs2",
]


def collect(dirs, with_content=False):
    records = []
    docs_seen = 0
    docs_with_trace = 0

    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "documents", "*.json"))):
            try:
                doc = json.load(open(path, encoding="utf-8"))
            except Exception as e:
                print(f"  skip {path}: {e}")
                continue

            docs_seen += 1
            train = doc.get("agentgeo_train_citation") or {}
            per_query = train.get("per_query_results") or []
            has_trace = False

            for qr in per_query:
                attempts = qr.get("optimization_attempts") or []
                if not attempts:
                    continue
                has_trace = True

                iters_used = qr.get("iterations_used")
                # A trace is self-consistent only when the iteration numbers run
                # 0..N-1 with no gap AND the count matches iterations_used (the
                # final round is either a passing verification or the exhausted
                # last attempt). A gap means a round was dropped by policy block,
                # duplicate detection, or arg-regen failure.
                seq = [a.get("iteration") for a in attempts]
                contiguous = seq == list(range(len(attempts)))
                complete = (
                    contiguous
                    and iters_used is not None
                    and len(attempts) in (iters_used - 1, iters_used)
                )

                for att in attempts:
                    diag = att.get("diagnosis") or {}
                    before = att.get("content_before") or ""
                    after = att.get("content_after") or ""

                    rec = {
                        # provenance
                        "source_dir": d,
                        "doc_id": doc.get("doc_id"),
                        "url": doc.get("url"),
                        "run_timestamp": doc.get("timestamp"),
                        # the loop step
                        "query": qr.get("query"),
                        "iteration": att.get("iteration"),
                        "tool_name": att.get("tool_name"),
                        "reasoning": att.get("reasoning"),
                        "key_changes": att.get("key_changes") or [],
                        "target_segment_index": att.get("target_segment_index"),
                        # why this tool: the diagnosis that drove selection
                        "root_cause": diag.get("root_cause"),
                        "severity": diag.get("severity"),
                        "key_deficiency": diag.get("key_deficiency"),
                        "diagnosis_explanation": diag.get("explanation"),
                        # query-level outcome (no per-attempt citation was logged)
                        "query_is_cited": qr.get("is_cited"),
                        "query_iterations_used": iters_used,
                        "query_geo_score_overall": qr.get("geo_score_overall"),
                        # trace integrity
                        "attempts_in_query": len(attempts),
                        "trace_complete": complete,
                        "content_before_len": len(before),
                        "content_after_len": len(after),
                    }
                    if with_content:
                        rec["content_before"] = before
                        rec["content_after"] = after
                    records.append(rec)

            if has_trace:
                docs_with_trace += 1

    return records, docs_seen, docs_with_trace


def summarize(records):
    tools = Counter(r["tool_name"] for r in records)
    causes = Counter(r["root_cause"] for r in records)
    pairs = Counter((r["root_cause"], r["tool_name"]) for r in records)
    days = Counter((r["run_timestamp"] or "?")[:10] for r in records)
    incomplete = sum(1 for r in records if not r["trace_complete"])

    return {
        "total_tool_calls": len(records),
        "unique_queries": len({(r["doc_id"], r["query"]) for r in records}),
        "unique_docs": len({r["doc_id"] for r in records}),
        "run_dates": dict(sorted(days.items())),
        "tool_distribution": dict(tools.most_common()),
        "root_cause_distribution": dict(causes.most_common()),
        "diagnosis_tool_pairs": {f"{c} -> {t}": n for (c, t), n in pairs.most_common()},
        "calls_in_incomplete_traces": incomplete,
        "note": (
            "Only iterations that produced a suggestion were logged. Rounds lost to "
            "policy block / duplicate detection / arg-regen failure leave no record, "
            "so calls_in_incomplete_traces marks records whose query has missing rounds."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dirs", nargs="*", default=DEFAULT_DIRS)
    ap.add_argument("-o", "--output", default="tool_traces.json")
    ap.add_argument("--with-content", action="store_true",
                    help="embed full content_before/content_after (large)")
    args = ap.parse_args()

    records, docs_seen, docs_with_trace = collect(args.dirs, args.with_content)
    summary = summarize(records)

    out = {
        "meta": {
            "source_dirs": args.dirs,
            "documents_scanned": docs_seen,
            "documents_with_trace": docs_with_trace,
            "includes_full_content": args.with_content,
            "schema": (
                "records[] = one tool invocation; diagnosis fields explain why the "
                "tool was picked, reasoning is the model's own justification"
            ),
        },
        "summary": summary,
        "records": records,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(args.output) / 1e6
    print(f"Wrote {args.output} ({size_mb:.1f} MB)")
    print(f"  {summary['total_tool_calls']} tool calls "
          f"from {summary['unique_queries']} queries "
          f"across {summary['unique_docs']} docs")
    print(f"  run dates: {summary['run_dates']}")
    print(f"  incomplete traces: {summary['calls_in_incomplete_traces']} calls")


if __name__ == "__main__":
    main()
