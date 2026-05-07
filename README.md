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
