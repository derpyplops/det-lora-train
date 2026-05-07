#!/usr/bin/env bash
# Train the same LoRA N times and diff the resulting adapters.
set -euo pipefail

N="${N:-3}"
STEPS="${STEPS:-50}"
SEED="${SEED:-42}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
RUNS="$ROOT/runs"

rm -rf "$RUNS"
mkdir -p "$RUNS"

for i in $(seq 1 "$N"); do
    echo "=== Run $i/$N ==="
    uv run python train.py \
        --output-dir "$RUNS/run-$i" \
        --seed "$SEED" \
        --steps "$STEPS" 2>&1 | tee "$RUNS/run-$i.log"
done

echo "=== Diffing adapters ==="
ADAPTERS=()
for i in $(seq 1 "$N"); do
    ADAPTERS+=("$RUNS/run-$i/adapter")
done
uv run python diff_adapters.py "${ADAPTERS[@]}" --report "$RUNS/diff_report.txt"
