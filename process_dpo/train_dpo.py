#!/usr/bin/env python3
"""Continue an Evidence-SFT LoRA with DPO while keeping an internal frozen copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-adapter", default="outputs/evidence-sft")
    parser.add_argument("--train-file", default="data/dpo/v1/train/train.jsonl")
    parser.add_argument("--validation-file", default="data/dpo/v1/validation/validation.jsonl")
    parser.add_argument("--output-dir", default="outputs/evidence-dpo-v1")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--loss-type", default="sigmoid")
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--torch-dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device-map", default="none", help="Use 'none' for Accelerate/DDP or 'cuda:0' for one GPU.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def _source_ids(path: str) -> set[str]:
    ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                ids.add(str(row["source_id"]))
    return ids


def main() -> None:
    args = parse_args()
    train_ids = _source_ids(args.train_file)
    eval_ids = _source_ids(args.validation_file)
    overlap = train_ids & eval_ids
    if overlap:
        raise ValueError(f"DPO train/validation source leakage: {sorted(overlap)[:5]}")

    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dtype = getattr(torch, args.torch_dtype)
    device_map = None if args.device_map.lower() in {"none", ""} else args.device_map
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    model.config.use_cache = False

    datasets = load_dataset(
        "json",
        data_files={"train": args.train_file, "validation": args.validation_file},
    )
    keep = {"prompt", "chosen", "rejected"}
    for split in ("train", "validation"):
        remove = [name for name in datasets[split].column_names if name not in keep]
        if remove:
            datasets[split] = datasets[split].remove_columns(remove)
    if args.max_train_samples > 0:
        datasets["train"] = datasets["train"].select(
            range(min(args.max_train_samples, len(datasets["train"])))
        )
    if args.max_eval_samples > 0:
        datasets["validation"] = datasets["validation"].select(
            range(min(args.max_eval_samples, len(datasets["validation"])))
        )

    config = DPOConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        beta=args.beta,
        loss_type=args.loss_type,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.torch_dtype == "bfloat16",
        fp16=args.torch_dtype == "float16",
        report_to=args.report_to,
        remove_unused_columns=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=config,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
    )
    if "ref" not in getattr(trainer.model, "peft_config", {}):
        raise RuntimeError(
            "This TRL version did not create a frozen 'ref' copy of the initial SFT adapter. "
            "Upgrade to a compatible TRL release before training; otherwise the reference policy "
            "would incorrectly fall back to the base model."
        )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
