#!/usr/bin/env python3
"""Generate and score the held-out Evidence-SFT test set."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FIELDS = {
    "task_type",
    "query_intent",
    "evidence_sufficiency",
    "evidence",
    "critical_evidence_ids",
    "missing_information",
    "clinical_reasoning",
    "final_answer",
}
TASK_TYPES = {"diagnostic_reasoning", "confirmed_management"}
SUFFICIENCY_LEVELS = {"sufficient", "partial", "insufficient", "conflicting"}
IMPORTANCE_LEVELS = {"critical", "supporting"}
CASE_MARKER = "病例描述：\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model", default=str(ROOT / "Qwen/Qwen2.5-7B-Instruct")
    )
    parser.add_argument(
        "--adapter", default=str(ROOT / "outputs/evidence-sft-v2-2")
    )
    parser.add_argument(
        "--test-file",
        default=str(ROOT / "data/evidence_sft/validated_v2_2/test.jsonl"),
    )
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results/evidence_eval")
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1152)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def strict_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def valid_string_list(value: Any, *, allow_empty: bool) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item.strip() for item in value)
    )


def validate_schema(value: Any) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(value, dict):
        return False, ["not_object"]
    missing = sorted(REQUIRED_FIELDS - set(value))
    errors.extend(f"missing:{field}" for field in missing)
    if value.get("task_type") not in TASK_TYPES:
        errors.append("invalid_task_type")
    if value.get("evidence_sufficiency") not in SUFFICIENCY_LEVELS:
        errors.append("invalid_sufficiency")
    if not valid_string_list(value.get("query_intent"), allow_empty=False):
        errors.append("invalid_query_intent")
    if not valid_string_list(value.get("critical_evidence_ids"), allow_empty=True):
        errors.append("invalid_critical_ids")
    if not valid_string_list(value.get("missing_information"), allow_empty=True):
        errors.append("invalid_missing_information")
    if not isinstance(value.get("clinical_reasoning"), str) or not value.get(
        "clinical_reasoning", ""
    ).strip():
        errors.append("invalid_clinical_reasoning")
    if not isinstance(value.get("final_answer"), str) or not value.get(
        "final_answer", ""
    ).strip():
        errors.append("invalid_final_answer")

    evidence = value.get("evidence")
    if not isinstance(evidence, list):
        errors.append("invalid_evidence")
    else:
        for item in evidence:
            if not isinstance(item, dict):
                errors.append("evidence_not_object")
                continue
            if not isinstance(item.get("id"), str) or not item.get("id", "").strip():
                errors.append("invalid_evidence_id")
            if not isinstance(item.get("span"), str) or not item.get("span", "").strip():
                errors.append("invalid_evidence_span")
            if item.get("importance") not in IMPORTANCE_LEVELS:
                errors.append("invalid_evidence_importance")
            if not isinstance(item.get("role"), str) or not item.get("role", "").strip():
                errors.append("invalid_evidence_role")
    return not errors, sorted(set(errors))


def evidence_spans(value: Any) -> list[str]:
    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        return []
    return [
        item["span"]
        for item in value["evidence"]
        if isinstance(item, dict) and isinstance(item.get("span"), str)
    ]


def critical_ids(value: Any) -> list[str]:
    if not isinstance(value, dict) or not isinstance(value.get("critical_evidence_ids"), list):
        return []
    return [item for item in value["critical_evidence_ids"] if isinstance(item, str)]


def critical_spans(value: Any) -> list[str]:
    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        return []
    critical = set(critical_ids(value))
    return [
        item["span"]
        for item in value["evidence"]
        if isinstance(item, dict)
        and item.get("id") in critical
        and isinstance(item.get("span"), str)
    ]


def critical_consistent(value: Any) -> bool:
    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        return False
    ids: list[str] = []
    marked: list[str] = []
    for item in value["evidence"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return False
        ids.append(item["id"])
        if item.get("importance") == "critical":
            marked.append(item["id"])
    requested = critical_ids(value)
    return (
        len(ids) == len(set(ids))
        and len(requested) == len(set(requested))
        and set(requested).issubset(ids)
        and set(requested) == set(marked)
    )


def multiset_overlap(predicted: list[str], gold: list[str]) -> int:
    return sum((Counter(predicted) & Counter(gold)).values())


def score_prediction(
    source_id: str,
    prompt: str,
    gold: dict[str, Any],
    generated_text: str,
    generated_tokens: int,
    ended_with_eos: bool,
) -> dict[str, Any]:
    case_text = prompt.split(CASE_MARKER, 1)[-1] if CASE_MARKER in prompt else prompt
    strict = strict_json_object(generated_text)
    parsed = strict if strict is not None else extract_json_object(generated_text)
    schema_valid, schema_errors = validate_schema(parsed)
    predicted_spans = evidence_spans(parsed)
    gold_spans = evidence_spans(gold)
    predicted_critical = critical_spans(parsed)
    gold_critical = critical_spans(gold)
    grounded = [span in case_text for span in predicted_spans]

    return {
        "source_id": source_id,
        "case_text": case_text,
        "prompt": prompt,
        "gold": gold,
        "generated_text": generated_text,
        "parsed_output": parsed,
        "generation": {
            "generated_tokens": generated_tokens,
            "ended_with_eos": ended_with_eos,
        },
        "metrics": {
            "strict_json_valid": strict is not None,
            "recoverable_json_valid": parsed is not None,
            "schema_valid": schema_valid,
            "schema_errors": schema_errors,
            "all_evidence_grounded": bool(predicted_spans) and all(grounded),
            "grounded_evidence_count": sum(grounded),
            "predicted_evidence_count": len(predicted_spans),
            "gold_evidence_count": len(gold_spans),
            "evidence_exact_overlap": multiset_overlap(predicted_spans, gold_spans),
            "critical_consistent": critical_consistent(parsed),
            "predicted_critical_count": len(predicted_critical),
            "gold_critical_count": len(gold_critical),
            "critical_span_exact_overlap": multiset_overlap(
                predicted_critical, gold_critical
            ),
            "task_type_correct": isinstance(parsed, dict)
            and parsed.get("task_type") == gold.get("task_type"),
            "sufficiency_correct": isinstance(parsed, dict)
            and parsed.get("evidence_sufficiency")
            == gold.get("evidence_sufficiency"),
            "final_answer_nonempty": isinstance(parsed, dict)
            and isinstance(parsed.get("final_answer"), str)
            and bool(parsed["final_answer"].strip()),
        },
    }


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def aggregate(rows: list[dict[str, Any]], args: argparse.Namespace, elapsed: float) -> dict[str, Any]:
    total = len(rows)
    metric_rows = [row["metrics"] for row in rows]
    booleans = [
        "strict_json_valid",
        "recoverable_json_valid",
        "schema_valid",
        "all_evidence_grounded",
        "critical_consistent",
        "task_type_correct",
        "sufficiency_correct",
        "final_answer_nonempty",
    ]
    counts = {name: sum(bool(row[name]) for row in metric_rows) for name in booleans}
    pred_evidence = sum(row["predicted_evidence_count"] for row in metric_rows)
    gold_evidence = sum(row["gold_evidence_count"] for row in metric_rows)
    evidence_overlap = sum(row["evidence_exact_overlap"] for row in metric_rows)
    grounded_evidence = sum(row["grounded_evidence_count"] for row in metric_rows)
    pred_critical = sum(row["predicted_critical_count"] for row in metric_rows)
    gold_critical = sum(row["gold_critical_count"] for row in metric_rows)
    critical_overlap = sum(row["critical_span_exact_overlap"] for row in metric_rows)

    evidence_precision = safe_rate(evidence_overlap, pred_evidence)
    evidence_recall = safe_rate(evidence_overlap, gold_evidence)
    critical_precision = safe_rate(critical_overlap, pred_critical)
    critical_recall = safe_rate(critical_overlap, gold_critical)

    schema_errors = Counter(
        error for row in metric_rows for error in row.get("schema_errors", [])
    )
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    sufficiency_confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        gold = row["gold"]
        parsed = row["parsed_output"] or {}
        confusion[str(gold.get("task_type"))][str(parsed.get("task_type"))] += 1
        sufficiency_confusion[str(gold.get("evidence_sufficiency"))][
            str(parsed.get("evidence_sufficiency"))
        ] += 1

    return {
        "evaluation": {
            "test_file": args.test_file,
            "base_model": args.base_model,
            "adapter": args.adapter,
            "samples": total,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "elapsed_seconds": round(elapsed, 2),
        },
        "sample_level": {
            **{f"{name}_count": counts[name] for name in booleans},
            **{f"{name}_rate": safe_rate(counts[name], total) for name in booleans},
            "ended_with_eos_count": sum(
                row["generation"]["ended_with_eos"] for row in rows
            ),
            "ended_with_eos_rate": safe_rate(
                sum(row["generation"]["ended_with_eos"] for row in rows), total
            ),
        },
        "micro_evidence": {
            "predicted_spans": pred_evidence,
            "gold_spans": gold_evidence,
            "grounded_predicted_spans": grounded_evidence,
            "grounding_rate": safe_rate(grounded_evidence, pred_evidence),
            "teacher_exact_overlap": evidence_overlap,
            "teacher_exact_precision": evidence_precision,
            "teacher_exact_recall": evidence_recall,
            "teacher_exact_f1": safe_rate(
                2 * evidence_precision * evidence_recall,
                evidence_precision + evidence_recall,
            ),
        },
        "micro_critical_span": {
            "predicted_spans": pred_critical,
            "gold_spans": gold_critical,
            "teacher_exact_overlap": critical_overlap,
            "teacher_exact_precision": critical_precision,
            "teacher_exact_recall": critical_recall,
            "teacher_exact_f1": safe_rate(
                2 * critical_precision * critical_recall,
                critical_precision + critical_recall,
            ),
        },
        "schema_error_counts": dict(schema_errors.most_common()),
        "task_type_confusion": {
            gold: dict(values) for gold, values in confusion.items()
        },
        "sufficiency_confusion": {
            gold: dict(values) for gold, values in sufficiency_confusion.items()
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    test_rows = load_jsonl(Path(args.test_file))
    if args.limit > 0:
        test_rows = test_rows[: args.limit]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "metrics.json"

    completed: dict[str, dict[str, Any]] = {}
    if args.resume and predictions_path.exists():
        completed = {row["source_id"]: row for row in load_jsonl(predictions_path)}
    pending = [row for row in test_rows if row["source_id"] not in completed]

    print(
        f"Test samples={len(test_rows)}, completed={len(completed)}, pending={len(pending)}",
        flush=True,
    )
    started = time.monotonic()
    if pending:
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
        completed_in_test = sum(
            row["source_id"] in completed for row in test_rows
        )
        with predictions_path.open(mode, encoding="utf-8") as output, tqdm(
            total=len(test_rows),
            initial=completed_in_test,
            desc="Evidence eval",
            unit="sample",
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
                    scored = score_prediction(
                        row["source_id"],
                        prompts[index],
                        gold[index],
                        text,
                        len(token_ids),
                        tokenizer.eos_token_id in token_ids,
                    )
                    output.write(json.dumps(scored, ensure_ascii=False) + "\n")
                    completed[row["source_id"]] = scored
                output.flush()
                progress.update(len(batch))

        del model, base_model
        torch.cuda.empty_cache()

    ordered = [completed[row["source_id"]] for row in test_rows]
    elapsed = time.monotonic() - started
    summary = aggregate(ordered, args, elapsed)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Predictions: {predictions_path}", flush=True)
    print(f"Metrics: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
