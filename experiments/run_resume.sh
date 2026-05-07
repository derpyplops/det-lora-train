#!/usr/bin/env bash
# Checkpoint-resume parity test:
#   A) train STEPS steps in one shot
#   B) train STEPS/2, save checkpoint, then resume and continue to STEPS
# Compare A's final adapter vs B's final adapter.
set -euo pipefail
STEPS="${1:-50}"
PREC="${2:-fp32}"
HALF=$((STEPS / 2))

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/runs/resume_${PREC}_${STEPS}"
PY="${PY:-python}"

rm -rf "$OUT"; mkdir -p "$OUT"

echo "=== [resume] A: direct $STEPS-step run ==="
$PY "$ROOT/train.py" \
    --output-dir "$OUT/A-direct" \
    --steps "$STEPS" \
    --precision "$PREC" 2>&1 | tee "$OUT/A-direct.log"

echo "=== [resume] B1: first $HALF steps, save checkpoint ==="
$PY "$ROOT/train.py" \
    --output-dir "$OUT/B-part1" \
    --steps "$HALF" \
    --save-steps "$HALF" \
    --precision "$PREC" 2>&1 | tee "$OUT/B-part1.log"

CKPT="$OUT/B-part1/checkpoint-$HALF"
if [ ! -d "$CKPT" ]; then
    echo "ERROR: expected checkpoint at $CKPT not found"
    ls "$OUT/B-part1" || true
    exit 1
fi

echo "=== [resume] B2: resume from $CKPT, continue to $STEPS ==="
$PY "$ROOT/train.py" \
    --output-dir "$OUT/B-part2" \
    --steps "$STEPS" \
    --precision "$PREC" \
    --resume-from "$CKPT" 2>&1 | tee "$OUT/B-part2.log"

echo "=== [resume] Diffing A (direct) vs B (resumed) ==="
$PY "$ROOT/diff_adapters.py" \
    "$OUT/A-direct/adapter" \
    "$OUT/B-part2/adapter" \
    --report "$OUT/diff_report.txt"
