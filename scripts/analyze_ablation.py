"""
Turn the repeated ablation runs into the statistics the reviewer asked for:
error bars across runs, and paired tests between configurations.

Reads outputs_rerun/{config}_run{n}/documents/*.json and reports

  * per-configuration citation rate as mean +/- SD over runs (the error bar)
  * the noise floor: SD of the *baseline* CR across runs, where no optimization
    is involved, so it isolates run-to-run variability of the harness itself
  * paired tests between configurations, on documents averaged over runs
    (Wilcoxon signed-rank) and on individual (document, query) outcomes
    within each run (McNemar), with Holm correction across comparisons
  * paired bootstrap CI for each contrast

Usage:
    python scripts/analyze_ablation.py
    python scripts/analyze_ablation.py --root outputs_rerun --baseline b10_mem
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

BASE_PREFIX = "agentgeo_baseline_test"
OPT_PREFIX = "agentgeo_optimized_test"
N_BOOTSTRAP = 10000
RNG_SEED = 0


def load_runs(root: Path, include_partial=False) -> dict:
    """{config: {run: {doc_id: record}}} for every completed run directory.

    Runs without a .done marker are still in flight and cover only a prefix of the
    document set, so including them would compare different document subsets and
    inflate the across-run SD. They are skipped unless explicitly requested.
    """
    runs = defaultdict(dict)
    skipped = []
    for run_dir in sorted(root.glob("*_run*")):
        docs_dir = run_dir / "documents"
        if not docs_dir.is_dir():
            continue
        if not (run_dir / ".done").exists() and not include_partial:
            n = len(list(docs_dir.glob("*.json")))
            skipped.append(f"{run_dir.name} ({n} docs)")
            continue
        config, _, run = run_dir.name.rpartition("_run")
        docs = {}
        for path in docs_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if any(k.endswith("_error") for k in record):
                continue
            if f"{OPT_PREFIX}_per_query" not in record:
                continue
            docs[record["doc_id"]] = record
        if docs:
            runs[config][int(run)] = docs
    if skipped:
        print(f"Skipped {len(skipped)} run(s) still in flight: {', '.join(skipped)}\n")
    return dict(runs)


def pooled_cr(docs: dict, prefix: str) -> float:
    """Citation rate pooled over all (document, query) pairs — the paper's definition."""
    cited = total = 0
    for record in docs.values():
        for result in record[f"{prefix}_per_query"].values():
            cited += bool(result["is_cited"])
            total += 1
    return cited / total if total else float("nan")


def doc_cr(record: dict, prefix: str) -> float:
    results = record[f"{prefix}_per_query"]
    if not results:
        return float("nan")
    return sum(bool(r["is_cited"]) for r in results.values()) / len(results)


def mean_metric(docs: dict, key: str, sub: str = None) -> float:
    values = []
    for record in docs.values():
        value = record.get(key)
        if isinstance(value, dict) and sub:
            value = value.get(sub)
        if isinstance(value, (int, float)):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def per_doc_mean_cr(config_runs: dict, prefix: str) -> dict:
    """{doc_id: CR averaged over runs} — averaging shrinks run noise before pairing."""
    acc = defaultdict(list)
    for docs in config_runs.values():
        for doc_id, record in docs.items():
            acc[doc_id].append(doc_cr(record, prefix))
    return {d: float(np.mean(v)) for d, v in acc.items() if v}


def bootstrap_ci(values: np.ndarray, alpha=0.05) -> tuple:
    """Percentile bootstrap CI for the mean, resampling documents."""
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, alpha=0.05) -> tuple:
    """Percentile bootstrap CI for the paired mean difference, resampling documents.

    Documents are resampled as units, so the interval reflects document-sampling
    uncertainty; pairing keeps each document's two measurements together.
    """
    return bootstrap_ci(a - b, alpha)


def mcnemar(config_a: dict, config_b: dict) -> tuple:
    """Exact McNemar over (document, query) outcomes paired within each run."""
    b = c = 0
    for run in sorted(set(config_a) & set(config_b)):
        docs_a, docs_b = config_a[run], config_b[run]
        for doc_id in set(docs_a) & set(docs_b):
            qa = docs_a[doc_id][f"{OPT_PREFIX}_per_query"]
            qb = docs_b[doc_id][f"{OPT_PREFIX}_per_query"]
            for query in set(qa) & set(qb):
                x, y = bool(qa[query]["is_cited"]), bool(qb[query]["is_cited"])
                b += x and not y
                c += y and not x
    if b + c == 0:
        return b, c, 1.0
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue)


def holm(pvalues: list) -> list:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="outputs_rerun")
    parser.add_argument("--include-partial", action="store_true",
                        help="Also include runs that have not finished (unequal document sets)")
    parser.add_argument("--baseline", default="b10_mem",
                        help="Configuration every contrast is measured against")
    args = parser.parse_args()

    runs = load_runs(Path(args.root), args.include_partial)
    if not runs:
        print(f"No completed runs under {args.root}")
        return

    print("=" * 100)
    print("PER-CONFIGURATION RESULTS (mean +/- SD over runs)")
    print("=" * 100)
    print(f"{'Config':12s} {'runs':>4s} {'docs':>5s} {'CR % (mean+/-SD)':>18s} "
          f"{'95% CI (doc boot)':>20s} {'Word':>7s} {'TF-IDF':>7s} {'Embed':>7s} {'Jaccard':>8s}")

    for config in sorted(runs):
        crs, words, tfidf, embed, jacc = [], [], [], [], []
        for docs in runs[config].values():
            crs.append(pooled_cr(docs, OPT_PREFIX) * 100)
            words.append(mean_metric(docs, f"{OPT_PREFIX}_avg_geo_score"))
            tfidf.append(mean_metric(docs, "agentgeo_similarity", "tfidf_similarity"))
            embed.append(mean_metric(docs, "agentgeo_similarity", "embedding_similarity"))
            jacc.append(mean_metric(docs, "agentgeo_similarity", "jaccard_similarity"))
        n_docs = len(next(iter(runs[config].values())))
        sd = np.std(crs, ddof=1) if len(crs) > 1 else float("nan")
        per_doc = np.array(list(per_doc_mean_cr(runs[config], OPT_PREFIX).values())) * 100
        lo, hi = bootstrap_ci(per_doc)
        print(f"{config:12s} {len(crs):4d} {n_docs:5d} "
              f"{np.mean(crs):10.2f}+/-{sd:5.2f} [{lo:8.2f},{hi:8.2f}] {np.mean(words):7.4f} "
              f"{np.mean(tfidf):7.4f} {np.mean(embed):7.4f} {np.mean(jacc):8.4f}")

    print("\nSD is across runs (LLM/harness non-determinism, temperature=0).")
    print("95% CI resamples documents (document-sampling uncertainty). They answer")
    print("different questions and are not interchangeable.")

    print()
    print("=" * 100)
    print("NOISE FLOOR — baseline CR across runs (no optimization involved)")
    print("=" * 100)
    for config in sorted(runs):
        base = [pooled_cr(docs, BASE_PREFIX) * 100 for docs in runs[config].values()]
        if len(base) > 1:
            print(f"{config:12s} runs={len(base)}  baseline CR: "
                  f"{', '.join(f'{b:.2f}' for b in base)}   "
                  f"SD={np.std(base, ddof=1):.2f}  range={max(base) - min(base):.2f} pp")
        else:
            print(f"{config:12s} runs=1  baseline CR: {base[0]:.2f}  (need >=2 runs for SD)")

    others = [c for c in sorted(runs) if c != args.baseline]
    if args.baseline not in runs or not others:
        return

    print()
    print("=" * 100)
    print(f"PAIRED CONTRASTS vs {args.baseline}")
    print("=" * 100)

    rows, raw_w, raw_m = [], [], []
    ref = per_doc_mean_cr(runs[args.baseline], OPT_PREFIX)
    for config in others:
        cur = per_doc_mean_cr(runs[config], OPT_PREFIX)
        shared = sorted(set(ref) & set(cur))
        a = np.array([cur[d] for d in shared]) * 100
        b = np.array([ref[d] for d in shared]) * 100
        delta = float(np.mean(a - b))
        try:
            w_p = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:      # all differences zero
            w_p = 1.0
        lo, hi = paired_bootstrap_ci(a, b)
        b_cnt, c_cnt, m_p = mcnemar(runs[config], runs[args.baseline])
        rows.append((config, len(shared), delta, lo, hi, b_cnt, c_cnt))
        raw_w.append(w_p)
        raw_m.append(m_p)

    adj_w, adj_m = holm(raw_w), holm(raw_m)
    print(f"{'Config':12s} {'docs':>5s} {'dCR pp':>8s} {'95% CI':>18s} "
          f"{'Wilcoxon':>10s} {'McNemar':>10s} {'b/c':>12s}")
    for (config, n, delta, lo, hi, b_cnt, c_cnt), pw, pm in zip(rows, adj_w, adj_m):
        mark = "*" if min(pw, pm) < 0.05 else " "
        print(f"{config:12s} {n:5d} {delta:+8.2f} [{lo:+7.2f},{hi:+7.2f}] "
              f"{pw:10.4f} {pm:10.4f} {b_cnt:5d}/{c_cnt:<6d}{mark}")
    print("\np-values are Holm-corrected within each test family; * marks p < 0.05.")
    print("Wilcoxon pairs documents (CR averaged over runs); McNemar pairs individual")
    print("(document, query) outcomes within runs. b/c are the discordant-pair counts.")


if __name__ == "__main__":
    main()
