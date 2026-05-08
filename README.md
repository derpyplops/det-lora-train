# det-lora-train

Experiment: can LoRA fine-tuning be made bit-for-bit reproducible across runs on the same machine?

## Approach

- Base: `Qwen/Qwen2.5-0.5B`
- Adapter: PEFT LoRA on `q_proj` / `v_proj`, r=8, dropout=0
- Data: a tiny fixed slice of `tatsu-lab/alpaca` (32 examples), batch size 2
- Determinism: all the standard flags (see `train.py`)
- Verification: `diff_adapters.py` compares saved adapter weights across runs and reports bitwise equality + max absolute difference

## Results

Tested on vast.ai with `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, torch 2.5.1+cu124, NVIDIA driver 580.76.

### Same-machine repeats — every one is bit-equal ✓

| Hardware  | Steps    | Wall time | Precision | N runs | Result          | max\|Δ\| |
|-----------|---------:|----------:|-----------|-------:|------------------|---------:|
| RTX 3090  |       50 |     18 s  | fp32      | 3      | **bit-equal ✓**  | 0        |
| RTX 3090  |      500 |    3 min  | fp32      | 3      | **bit-equal ✓**  | 0        |
| RTX 3090  |       50 |     18 s  | **bf16**  | 3      | **bit-equal ✓**  | 0        |
| A100 SXM4 |       50 |      9 s  | fp32      | 3      | **bit-equal ✓**  | 0        |
| RTX 3090  |    3,000 |    23 min | fp32      | 2      | **bit-equal ✓**  | 0        |
| RTX 3090  |   10,000 |    76 min | fp32      | 2      | **bit-equal ✓**  | 0        |
| RTX 3090  | **50,000** | **6 hr 23 min** | fp32 | 2 | **bit-equal ✓** | 0       |

The 50,000-step run is the headline: **6+ hours of continuous fp32 training and the two adapter checkpoints are byte-for-byte identical** — every one of the 96 saved tensors matches exactly.

### Things that DO break determinism

| What                              | Steps        | max\|Δ\|     | Notes |
|-----------------------------------|-------------:|-------------:|-------|
| Different GPU (3090 vs A100 SXM4) | 50 fp32      | **5.53e-05** | cuBLAS picks different GEMM kernels per SM |
| Checkpoint-resume vs direct       | 50 (25+25)   | **9.65e-04** | resuming a run does NOT match running it in one shot |

The checkpoint-resume drift (~1e-3) is roughly **17× larger** than the cross-hardware drift. Splitting a 50-step training as `25 → save → resume → 25` produced a measurably different adapter than running 50 in one shot, despite identical seeds and full determinism flags. Likely culprit: HF's `resume_from_checkpoint` fast-forwards through the dataloader to recover step count, and that draws RNG in a different sequence than a fresh run.

### Other observations

- Per-step losses match to all printed digits across same-machine repeat runs (e.g. `train_loss=6.282602729797364` identical to 13 decimals).
- Floating-point error does **not** accumulate into nondeterminism over time. 50,000 fp32 steps stayed bit-equal — if individual ops are deterministic, the composition is too.
- bf16 mixed precision stays bit-equal on the same machine. AMP doesn't re-introduce nondeterminism under the flags below.

## What made this work

```python
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"   # before `import torch`
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

Plus, on the HF side: `TrainingArguments(full_determinism=True, seed=42, data_seed=42, dataloader_num_workers=0, ...)` and `lora_dropout=0` in the LoRA config.

Throughput cost vs. a tuned config is modest at this scale (≈ 2.2 it/s on a 3090 in fp32 with all the flags on).

## Reproducing

```bash
uv sync
./experiments/run_all.sh         # 50/500-step matrix + bf16 + resume parity
./experiments/run_long.sh        # 20-min, 1-hr, 5-hr fp32 runs (~13 hr on a 3090)
```

On a remote box that already has torch installed (e.g. `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`), skip `uv sync` — just `pip install transformers peft datasets accelerate safetensors` and run with `PY=python ./experiments/run_all.sh`.

Diffs land in `runs/<experiment>/diff_report.txt`. Pre-computed reports from the runs above are checked in under `runs/*_diff.txt`.
