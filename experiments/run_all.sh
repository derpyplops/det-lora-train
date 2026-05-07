#!/usr/bin/env bash
# Run all same-hardware determinism experiments.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-python}"
export PY

echo "##### Experiment 1: baseline fp32 50 steps #####"
"$ROOT/experiments/run_repeat.sh" baseline_fp32_50 50 fp32 3

echo "##### Experiment 2: longer fp32 500 steps #####"
"$ROOT/experiments/run_repeat.sh" longer_fp32_500 500 fp32 3

echo "##### Experiment 3: bf16 50 steps #####"
"$ROOT/experiments/run_repeat.sh" bf16_50 50 bf16 3

echo "##### Experiment 4: checkpoint-resume parity (fp32, 50 steps) #####"
"$ROOT/experiments/run_resume.sh" 50 fp32

echo "##### Summary #####"
for name in baseline_fp32_50 longer_fp32_500 bf16_50 resume_fp32_50; do
    echo "--- $name ---"
    cat "$ROOT/runs/$name/diff_report.txt" 2>/dev/null || echo "  (no report)"
done
