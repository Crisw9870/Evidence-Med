#!/usr/bin/env python3
"""Generate Evidence Mask variants and preserve pair metadata for paired analysis."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=str(ROOT / "Qwen/Qwen2.5-7B-Instruct"))
    parser.add_argument("--adapter", required=True)
    parser.add_argument(
        "--test-file",
        default=str(ROOT / "data/evidence_mask/v1/validated/test_pairs.jsonl"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1152)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def score_mask_prediction(
    row: dict[str, Any],
    prompt: str,
    gold: dict[str, Any],
    generated_text: str,
    generated_tokens: int,
    ended_with_eos: bool,
) -> dict[str, Any]:
    from evaluate_evidence import score_prediction

    variant_id = str(row["variant_id"])
    scored = score_prediction(
        variant_id,
        prompt,
        gold,
        generated_text,
        generated_tokens,
        ended_with_eos,
    )
    scored["variant_id"] = scored.pop("source_id")
    scored.update(
        {
            "source_id": row["source_id"],
            "pair_id": row["pair_id"],
            "parent_source_id": row["parent_source_id"],
            "mask_type": row["mask_type"],
            "masked": bool(row.get("masked")),
            "masked_evidence_ids": row.get("masked_evidence_ids", []),
            "removed_spans": row.get("removed_spans", []),
            "mask_assessment": row.get("mask_assessment"),
        }
    )
    removed = [
        str(item.get("span", ""))
        for item in row.get("removed_spans", [])
        if isinstance(item, dict) and item.get("span")
    ]
    evidence = scored.get("parsed_output", {}).get("evidence", []) if isinstance(scored.get("parsed_output"), dict) else []
    predicted_spans = [
        str(item.get("span", ""))
        for item in evidence
        if isinstance(item, dict) and item.get("span")
    ]
    scored["metrics"]["removed_span_evidence_leakage"] = any(
        span in predicted_spans for span in removed
    )
    scored["metrics"]["removed_span_literal_mention"] = any(
        span in generated_text for span in removed
    )
    return scored


def aggregate_by_variant(
    rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float
) -> dict[str, Any]:
    from evaluate_evidence import aggregate

    overall = aggregate(rows, args, elapsed)
    by_type: dict[str, Any] = {}
    for mask_type in sorted({str(row["mask_type"]) for row in rows}):
        subset = [row for row in rows if row["mask_type"] == mask_type]
        summary = aggregate(subset, args, elapsed)
        summary["removed_fact"] = {
            "evidence_leakage_count": sum(
                bool(row["metrics"].get("removed_span_evidence_leakage"))
                for row in subset
            ),
            "evidence_leakage_rate": round(
                sum(
                    bool(row["metrics"].get("removed_span_evidence_leakage"))
                    for row in subset
                )
                / len(subset),
                6,
            )
            if subset
            else 0.0,
            "literal_mention_count": sum(
                bool(row["metrics"].get("removed_span_literal_mention"))
                for row in subset
            ),
        }
        by_type[mask_type] = summary
    return {
        "evaluation": overall["evaluation"],
        "variant_counts": dict(Counter(str(row["mask_type"]) for row in rows)),
        "overall": overall,
        "by_mask_type": by_type,
    }


def main() -> None:
    args = parse_args()
    from evaluate_evidence import load_jsonl

    test_rows = load_jsonl(Path(args.test_file))
    if args.limit > 0:
        test_rows = test_rows[: args.limit]
    variant_ids = [row.get("variant_id") for row in test_rows]
    if any(not isinstance(value, str) or not value for value in variant_ids):
        raise ValueError("every Mask test row must have a non-empty variant_id")
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("Mask test variant_id values must be unique")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "metrics.json"

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and predictions_path.exists():
        completed = {row["variant_id"]: row for row in load_jsonl(predictions_path)}
    pending = [row for row in test_rows if row["variant_id"] not in completed]
    print(
        f"Mask variants={len(test_rows)}, completed={len(completed)}, pending={len(pending)}",
        flush=True,
    )
    started = time.monotonic()
    if pending:
        import torch
        from peft import PeftModel
        from tqdm.auto import tqdm
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, use_fast=True, trust_remote_code=True
        )
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()

        mode = "a" if args.resume and predictions_path.exists() else "w"
        with predictions_path.open(mode, encoding="utf-8") as output, tqdm(
            total=len(test_rows),
            initial=sum(row["variant_id"] in completed for row in test_rows),
            desc="Evidence Mask eval",
            unit="variant",
            dynamic_ncols=True,
        ) as progress:
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start : start + args.batch_size]
                prompts = [row["conversations"][0]["value"] for row in batch]
                gold = [json.loads(row["conversations"][1]["value"]) for row in batch]
                messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
                rendered = [
                    tokenizer.apply_chat_template(
                        item, tokenize=False, add_generation_prompt=True
                    )
                    for item in messages
                ]
                inputs = tokenizer(
                    rendered, return_tensors="pt", padding=True, truncation=False
                ).to(model.device)
                input_width = inputs["input_ids"].shape[1]
                with torch.inference_mode():
                    sequences = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                        use_cache=True,
                    )
                generated = sequences[:, input_width:]
                for index, row in enumerate(batch):
                    token_ids = generated[index].tolist()
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    scored = score_mask_prediction(
                        row,
                        prompts[index],
                        gold[index],
                        text,
                        len(token_ids),
                        tokenizer.eos_token_id in token_ids,
                    )
                    output.write(json.dumps(scored, ensure_ascii=False) + "\n")
                    completed[row["variant_id"]] = scored
                output.flush()
                progress.update(len(batch))
        del model, base_model
        torch.cuda.empty_cache()

    ordered = [completed[row["variant_id"]] for row in test_rows]
    summary = aggregate_by_variant(ordered, args, time.monotonic() - started)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Predictions: {predictions_path}", flush=True)
    print(f"Metrics: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
