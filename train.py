"""Deterministic LoRA fine-tune of a small base model.

Knobs are intentionally minimal — the point is to lock down every source of
nondeterminism, not to produce a useful model.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Must be set BEFORE importing torch for `use_deterministic_algorithms` to be
# happy with cuBLAS GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

MODEL_ID = "Qwen/Qwen2.5-0.5B"
DATASET_ID = "tatsu-lab/alpaca"
N_EXAMPLES = 32
MAX_LEN = 256
SEED = 42


def lock_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # TF32 changes math depending on tensor cores — kill it.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_dataset(tokenizer):
    raw = load_dataset(DATASET_ID, split=f"train[:{N_EXAMPLES}]")

    def fmt(ex):
        if ex["input"]:
            prompt = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n"
        text = prompt + ex["output"] + tokenizer.eos_token
        out = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors=None,
        )
        out["labels"] = list(out["input_ids"])
        return out

    return raw.map(fmt, remove_columns=raw.column_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    lock_determinism(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,  # fp32 to avoid AMP nondeterminism
    )

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,  # dropout adds RNG draws, kill it
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(base, lora_cfg)
    model.print_trainable_parameters()

    ds = build_dataset(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        warmup_steps=0,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        dataloader_num_workers=0,           # multi-worker shuffling = nondeterministic
        dataloader_drop_last=True,
        bf16=False,
        fp16=False,                         # fp32 only
        full_determinism=True,              # HF's umbrella flag
        optim="adamw_torch",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        tokenizer=tokenizer,
    )
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"Saved adapter to {adapter_dir}")


if __name__ == "__main__":
    main()
