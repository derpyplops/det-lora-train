#!/usr/bin/env bash
# Long-run determinism: 20-min, 1-hr, 5-hr trainings (N=2 each, fp32, 3090).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
export PY

# Step counts calibrated to ~2.7 it/s on a 3090.
"$ROOT/experiments/run_repeat.sh" long_20min 3000  fp32 2
"$ROOT/experiments/run_repeat.sh" long_1hr   10000 fp32 2
"$ROOT/experiments/run_repeat.sh" long_5hr   50000 fp32 2

echo "##### Long-run summary #####"
for name in long_20min long_1hr long_5hr; do
    echo "--- $name ---"
    cat "$ROOT/runs/$name/diff_report.txt" 2>/dev/null || echo "  (no report)"
done
