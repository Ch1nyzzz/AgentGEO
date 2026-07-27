#!/usr/bin/env bash
# Run the Table-10 ablation configurations, all repeats of one configuration back to back.
#
# Repeats of the same configuration MUST share a doc-concurrency: the SD we report is
# computed across those repeats, so an inconsistent setting there would contaminate the
# error bar itself. Concurrency is therefore pinned per configuration below, and the
# value used is recorded in each run directory.
#
# Each (config, run) pair gets its own output directory, so the checkpoint logic in
# run_optimization.py never mistakes one repeat for another. Interrupted runs resume:
# completed documents are skipped, errored ones retried.
#
# Usage: scripts/run_ablation_repeats.sh [runs] [doc_limit]

set -uo pipefail
cd "$(dirname "$0")/.."

RUNS="${1:-3}"
DOC_LIMIT="${2:-50}"

# Configurations in execution order, with the concurrency pinned to each.
# The first three already have a run1 at 16, so their remaining repeats stay at 16.
# (case instead of an associative array: macOS ships bash 3.2, which lacks declare -A)
CONFIGS=(b5_mem b10_mem b10_nomem b1_mem b20_mem)
concurrency_for() {
  case "$1" in
    b1_mem|b20_mem) echo 32 ;;   # not started yet, free to run faster
    *)              echo 16 ;;   # run1 already recorded at 16; repeats must match
  esac
}

OUT_ROOT="outputs_rerun"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

echo "=== Ablation: ${#CONFIGS[@]} configs x $RUNS runs, $DOC_LIMIT docs ==="
for cfg in "${CONFIGS[@]}"; do echo "    $cfg -> concurrency $(concurrency_for "$cfg")"; done
echo "Started: $(date '+%F %T')"

failed=()
for cfg in "${CONFIGS[@]}"; do
  conc="$(concurrency_for "$cfg")"
  for run in $(seq 1 "$RUNS"); do
    tag="${cfg}_run${run}"
    out="$OUT_ROOT/$tag"
    log="$LOG_DIR/$tag.log"

    if [[ -f "$out/.done" ]]; then
      echo "[skip] $tag already complete"
      continue
    fi

    echo "[start] $tag  concurrency=$conc  $(date '+%T')"
    python3 scripts/run_optimization.py \
      --config "configs_ablation/${cfg}.yaml" \
      --method agentgeo \
      --output-dir "$out" \
      --doc-limit "$DOC_LIMIT" \
      --doc-concurrency "$conc" \
      >> "$log" 2>&1

    if [[ $? -eq 0 ]]; then
      # Record the run's actual concurrency alongside its results.
      echo "{\"config\":\"$cfg\",\"run\":$run,\"doc_concurrency\":$conc,\"doc_limit\":$DOC_LIMIT}" > "$out/run_meta.json"
      touch "$out/.done"
      echo "[done]  $tag  $(date '+%T')"
    else
      failed+=("$tag")
      echo "[FAIL]  $tag  $(date '+%T')  see $log"
    fi
  done
done

echo "=== Finished: $(date '+%F %T') ==="
if (( ${#failed[@]} )); then
  echo "FAILED RUNS: ${failed[*]}"
  exit 1
fi
echo "All $(( RUNS * ${#CONFIGS[@]} )) runs completed"
