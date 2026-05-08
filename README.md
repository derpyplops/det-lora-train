# det-lora-train

Experiment: can LoRA fine-tuning be made bit-for-bit reproducible across runs on the same machine?

## Approach

- Base: `Qwen/Qwen2.5-0.5B`
- Adapter: PEFT LoRA on `q_proj` / `v_proj`
- Data: a tiny fixed slice of `tatsu-lab/alpaca` (32 examples, 50 steps)
- Determinism: all the standard flags (see `train.py`)
- Verification: `diff_adapters.py` compares adapter weights across runs and reports bitwise equality + max abs diff

## Run

```bash
uv sync
./run_experiment.sh   # trains 3x and diffs the adapters
```

Outputs go in `runs/run-{1,2,3}/adapter/` and the diff report in `runs/diff_report.txt`.

## Results

Tested on vast.ai with `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, torch 2.5.1+cu124, NVIDIA driver 580.76.

| Experiment                    | Hardware       | Steps | Precision | Result          | max\|Δ\|    |
|-------------------------------|----------------|-------|-----------|------------------|-------------|
| Same-machine repeat (3 runs)  | RTX 3090       | 50    | fp32      | **bit-equal ✓**  | 0           |
| Same-machine repeat (3 runs)  | RTX 3090       | 500   | fp32      | **bit-equal ✓**  | 0           |
| Same-machine repeat (3 runs)  | RTX 3090       | 50    | bf16      | **bit-equal ✓**  | 0           |
| Same-machine repeat (3 runs)  | A100 SXM4      | 50    | fp32      | **bit-equal ✓**  | 0           |
| Same-machine repeat (2 runs)  | RTX 3090       | 3,000 (≈ 23 min)   | fp32 | **bit-equal ✓** | 0     |
| Same-machine repeat (2 runs)  | RTX 3090       | 10,000 (≈ 76 min)  | fp32 | **bit-equal ✓** | 0     |
| Same-machine repeat (2 runs)  | RTX 3090       | 50,000 (≈ 6.4 hr)  | fp32 | **bit-equal ✓** | 0     |
| Cross-hardware (3090 vs A100) | 3090 ↔ A100    | 50    | fp32      | differs          | 5.53e-05    |
| Resume-from-checkpoint vs direct | RTX 3090   | 50 (25+25) | fp32 | **differs**      | **9.65e-04** |

Per-step training losses on each individual machine matched to all printed digits across the 3 repeat runs (e.g. `train_loss=6.282602729797364` identical to 13 decimals).

### Findings

1. **Same-machine bit-equality is achievable**, including under bf16 mixed precision and at every duration tested — from 50 steps up to **50,000 steps (≈ 6.4 hr of continuous training)**. Floating-point error does not accumulate into nondeterminism over time when each op is individually deterministic. The standard determinism flags (below) are sufficient.
2. **Cross-hardware drift is small** but real: 5.5e-5 max delta in adapter weights between 3090 and A100 after 50 fp32 steps. Caused by different cuBLAS GEMM kernels selected for the two SM architectures.
3. **Checkpoint-resume does NOT bit-match direct training** — and the drift (9.7e-4) is roughly **17× larger than the cross-hardware drift**. Splitting a 50-step run as 25+save+resume+25 produced a noticeably different adapter than running 50 in one shot, even with full determinism flags and identical seeds. Likely culprit: HF's `resume_from_checkpoint` skips forward through the dataloader to recover step count, but that fast-forward draws RNG in a different sequence than a fresh run. (Or the optimizer state save loses precision.) Worth investigating if you depend on resume parity for long runs.

## What made this work

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before `import torch`
- `torch.use_deterministic_algorithms(True)`
- `cudnn.deterministic=True`, `cudnn.benchmark=False`
- TF32 disabled on both matmul and cudnn paths
- LoRA dropout=0
- HF `TrainingArguments(full_determinism=True, seed=42, data_seed=42)`
- `dataloader_num_workers=0`

bf16 still gave bit-equality on the same machine — interesting, since one would expect AMP cast paths to introduce nondeterminism. Suggests `bf16=True` in HF's TrainingArguments leaves the determinism guarantees intact under the flags above.

The throughput cost vs. a tuned config is small at this scale (50 steps in ~18s on a 3090, ~9s on an A100).

## Reproducing

```bash
uv sync
./experiments/run_all.sh        # same-machine matrix
# or pass PY=python ./experiments/run_all.sh on a host with system torch already installed
```

Diffs land in `runs/<experiment>/diff_report.txt`.
