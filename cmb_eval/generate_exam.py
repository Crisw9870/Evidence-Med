#!/usr/bin/env python3
"""Generate deterministic CMB-Exam predictions for one checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tqdm.auto import tqdm

from cmb_utils import (
    DEFAULT_EXAM_TEST,
    exam_item_id,
    extract_choice,
    format_exam_prompt,
    load_json_list,
    read_jsonl,
)
from model_runner import ModelRunner, add_model_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=str(DEFAULT_EXAM_TEST))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cot", action="store_true", help="Secondary CoT evaluation.")
    add_model_arguments(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_json_list(args.data_file)
    if args.limit > 0:
        rows = rows[: args.limit]
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume or choose another path: {output}")

    completed: dict[str, dict] = {}
    if args.resume and output.exists():
        completed = {str(row["item_id"]): row for row in read_jsonl(output)}
    indexed = [(index, row) for index, row in enumerate(rows)]
    pending = [item for item in indexed if exam_item_id(item[1], item[0]) not in completed]
    print(f"items={len(rows)} completed={len(completed)} pending={len(pending)}")
    if not pending:
        return

    runner = ModelRunner(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output.exists() else "w"
    started = time.monotonic()
    with output.open(mode, encoding="utf-8") as handle, tqdm(
        total=len(pending), desc=f"CMB-Exam {args.model_label}", unit="item"
    ) as progress:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            prompts = [format_exam_prompt(row, cot=args.cot) for _, row in batch]
            rendered = [
                runner.render(runner.messages(prompt, args.system_prompt))
                for prompt in prompts
            ]
            generated = runner.generate_rendered(
                rendered, max_new_tokens=args.max_new_tokens
            )
            for (index, row), prompt, result in zip(batch, prompts, generated):
                predicted, method = extract_choice(
                    result["text"], row["option"].keys(), str(row["question_type"])
                )
                record = {
                    "item_id": exam_item_id(row, index),
                    "row_index": index,
                    "model_label": args.model_label,
                    "exam_type": row.get("exam_type"),
                    "exam_class": row.get("exam_class"),
                    "exam_subject": row.get("exam_subject"),
                    "question_type": row.get("question_type"),
                    "prompt": prompt,
                    "raw_output": result["text"],
                    "predicted_answer": predicted,
                    "extraction_method": method,
                    "generated_tokens": result["generated_tokens"],
                    "ended_with_eos": result["ended_with_eos"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            progress.update(len(batch))
    print(f"saved={output} elapsed_seconds={time.monotonic() - started:.1f}")


if __name__ == "__main__":
    main()
