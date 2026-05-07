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

## Result

On a single RTX 3090 (driver 580.76, CUDA 12.4, torch 2.5.1+cu124), three back-to-back runs produced **bitwise-identical** adapter weights across all 96 tensors:

```
run-1 vs run-2: BITWISE EQUAL  max|Δ|=0.000e+00
run-1 vs run-3: BITWISE EQUAL  max|Δ|=0.000e+00
run-2 vs run-3: BITWISE EQUAL  max|Δ|=0.000e+00
RESULT: DETERMINISTIC ✓
```

Per-step losses also matched to all printed digits (e.g. `loss=9.4465` at step 5 in every run), and `train_loss=6.282602729797364` was identical to all 13 printed decimals.

## What made this work

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` set before `import torch`
- `torch.use_deterministic_algorithms(True)`
- `cudnn.deterministic=True`, `cudnn.benchmark=False`
- TF32 disabled on both matmul and cudnn paths
- fp32 throughout — no AMP/bf16/fp16
- LoRA dropout=0
- HF `TrainingArguments(full_determinism=True, seed=42, data_seed=42)`
- `dataloader_num_workers=0`

The expected cost is ~10–30% throughput vs. a tuned-for-speed config; in this run it didn't matter (50 steps in ~18s).
