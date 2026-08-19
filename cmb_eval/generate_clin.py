#!/usr/bin/env python3
"""Generate multi-turn CMB-Clin answers for one checkpoint."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from cmb_utils import DEFAULT_CLIN, load_json_list, read_jsonl
from model_runner import ModelRunner, add_model_arguments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", default=str(DEFAULT_CLIN))
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    add_model_arguments(parser)
    parser.set_defaults(batch_size=1, max_new_tokens=640)
    return parser.parse_args()


def first_user_message(description: str, question: str) -> str:
    return f"以下是一位病人的病例：\n{description}\n\n{question}"


def generation_settings(args: argparse.Namespace) -> dict[str, object]:
    """Parameters that must remain identical within one resumable output file."""
    return {
        "model": args.model,
        "adapter": args.adapter,
        "additional_adapter": args.additional_adapter,
        "tokenizer": args.tokenizer or args.model,
        "torch_dtype": args.torch_dtype,
        "system_prompt": args.system_prompt,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "num_beams": 1,
    }


def main() -> None:
    args = parse_args()
    cases = load_json_list(args.data_file)
    if args.limit_cases > 0:
        cases = cases[: args.limit_cases]
    output = Path(args.output)
    if output.exists() and not args.resume:
        raise FileExistsError(f"output exists; use --resume or choose another path: {output}")

    existing = read_jsonl(output) if args.resume and output.exists() else []
    settings = generation_settings(args)
    mismatched = [
        row
        for row in existing
        if row.get("generation_settings") != settings
    ]
    if mismatched:
        raise ValueError(
            f"{output} contains {len(mismatched)} rows generated with different or "
            "unrecorded settings. Use a new --output directory/file instead of "
            "mixing evaluation configurations."
        )
    existing_counts = Counter(str(row.get("case_id")) for row in existing)
    expected_counts = {
        str(case.get("id")): len(case.get("QA_pairs", [])) for case in cases
    }
    complete_case_ids = {
        case_id
        for case_id, expected in expected_counts.items()
        if expected > 0 and existing_counts[case_id] == expected
    }
    kept = [row for row in existing if str(row.get("case_id")) in complete_case_ids]
    pending = [case for case in cases if str(case.get("id")) not in complete_case_ids]
    print(
        f"cases={len(cases)} complete={len(complete_case_ids)} pending={len(pending)}"
    )
    if not pending:
        return

    runner = ModelRunner(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with tqdm(total=len(pending), desc=f"CMB-Clin {args.model_label}", unit="case") as progress:
            for case in pending:
                case_id = str(case.get("id"))
                description = str(case.get("description", ""))
                messages: list[dict[str, str]] = []
                if args.system_prompt:
                    messages.append({"role": "system", "content": args.system_prompt})
                for turn_index, pair in enumerate(case.get("QA_pairs", [])):
                    question = str(pair.get("question", ""))
                    user_content = (
                        first_user_message(description, question)
                        if turn_index == 0
                        else question
                    )
                    messages.append({"role": "user", "content": user_content})
                    rendered = runner.render(messages)
                    result = runner.generate_rendered(
                        [rendered], max_new_tokens=args.max_new_tokens
                    )[0]
                    answer = result["text"]
                    history_before = [dict(message) for message in messages[:-1]]
                    record = {
                        "item_id": f"{case_id}:{turn_index}",
                        "case_id": case_id,
                        "case_title": case.get("title"),
                        "turn_index": turn_index,
                        "model_label": args.model_label,
                        "case_description": description,
                        "history_before": history_before,
                        "question": question,
                        "reference_answer": pair.get("answer"),
                        "model_answer": answer,
                        "generated_tokens": result["generated_tokens"],
                        "ended_with_eos": result["ended_with_eos"],
                        "generation_settings": settings,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    messages.append({"role": "assistant", "content": answer})
                progress.update(1)
    print(f"saved={output} elapsed_seconds={time.monotonic() - started:.1f}")


if __name__ == "__main__":
    main()
