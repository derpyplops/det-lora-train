#!/usr/bin/env bash
# Run the same training N times and diff adapters.
# Usage: run_repeat.sh <name> <steps> <precision> [N]
set -euo pipefail
NAME="$1"; STEPS="$2"; PREC="$3"; N="${4:-3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/runs/$NAME"
PY="${PY:-python}"

rm -rf "$OUT"; mkdir -p "$OUT"

for i in $(seq 1 "$N"); do
    echo "=== [$NAME] Run $i/$N (steps=$STEPS prec=$PREC) ==="
    $PY "$ROOT/train.py" \
        --output-dir "$OUT/run-$i" \
        --steps "$STEPS" \
        --precision "$PREC" 2>&1 | tee "$OUT/run-$i.log"
done

echo "=== [$NAME] Diffing adapters ==="
ADAPTERS=()
for i in $(seq 1 "$N"); do ADAPTERS+=("$OUT/run-$i/adapter"); done
$PY "$ROOT/diff_adapters.py" "${ADAPTERS[@]}" --report "$OUT/diff_report.txt"
