"""Compare PEFT adapter checkpoints across runs.

Usage:
    python diff_adapters.py runs/run-1/adapter runs/run-2/adapter runs/run-3/adapter

For each pair (run-i, run-j) it reports:
  * whether every tensor is bitwise identical
  * the maximum absolute difference across all tensors
  * the first tensor that differs (if any)
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import torch
from safetensors.torch import load_file


def load_adapter(adapter_dir: Path) -> dict[str, torch.Tensor]:
    candidates = list(adapter_dir.glob("adapter_model.safetensors")) + list(
        adapter_dir.glob("adapter_model.bin")
    )
    if not candidates:
        raise FileNotFoundError(f"No adapter weights in {adapter_dir}")
    p = candidates[0]
    if p.suffix == ".safetensors":
        return load_file(str(p))
    return torch.load(str(p), map_location="cpu", weights_only=True)


def compare(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict:
    if a.keys() != b.keys():
        return {
            "bitwise_equal": False,
            "max_abs_diff": float("inf"),
            "first_diff_key": "<<key set differs>>",
            "n_keys": len(a),
        }

    max_abs = 0.0
    first_diff_key = None
    bitwise_ok = True

    for k in a:
        ta, tb = a[k], b[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype:
            bitwise_ok = False
            first_diff_key = first_diff_key or k
            continue
        if not torch.equal(ta, tb):
            bitwise_ok = False
            first_diff_key = first_diff_key or k
            d = (ta.float() - tb.float()).abs().max().item()
            max_abs = max(max_abs, d)

    return {
        "bitwise_equal": bitwise_ok,
        "max_abs_diff": max_abs,
        "first_diff_key": first_diff_key,
        "n_keys": len(a),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("adapters", nargs="+", type=Path)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if len(args.adapters) < 2:
        print("Need at least two adapter dirs to compare.", file=sys.stderr)
        return 2

    loaded = [(p, load_adapter(p)) for p in args.adapters]
    lines: list[str] = []

    def out(s: str) -> None:
        print(s)
        lines.append(s)

    out(f"Loaded {len(loaded)} adapters, {len(loaded[0][1])} tensors each.")
    all_equal = True
    for (pa, sa), (pb, sb) in combinations(loaded, 2):
        r = compare(sa, sb)
        tag = "BITWISE EQUAL" if r["bitwise_equal"] else "DIFFER"
        all_equal &= r["bitwise_equal"]
        out(f"  {pa.parent.name} vs {pb.parent.name}: {tag}  max|Δ|={r['max_abs_diff']:.3e}  first_diff={r['first_diff_key']}")

    out("")
    out("RESULT: " + ("DETERMINISTIC ✓" if all_equal else "NONDETERMINISTIC ✗"))

    if args.report:
        args.report.write_text("\n".join(lines) + "\n")
        print(f"Wrote report to {args.report}")

    return 0 if all_equal else 1


if __name__ == "__main__":
    sys.exit(main())
