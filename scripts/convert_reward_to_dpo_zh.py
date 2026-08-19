"""
Convert reward-model JSONL data into DPO-style Chinese JSONL.

Input format supported by default:
  {
    "question": "...",
    "response_chosen": "...",
    "response_rejected": "..."
  }

Output format:
  {
    "conversations": [{"from": "human", "value": "..."}],
    "chosen": "...",
    "rejected": "..."
  }

If a record already contains "conversations", it is preserved. Any extra
"tools" field is also kept when present.

Usage:
  python scripts/convert_reward_to_dpo_zh.py \
      --input data/reward/train.jsonl \
      --output data/reward/train_dpo_zh.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert reward JSONL to DPO zh format.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/reward/train.jsonl"),
        help="Input JSONL file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reward/train_dpo_zh.jsonl"),
        help="Output JSONL file.",
    )
    parser.add_argument(
        "--question-key",
        type=str,
        default="question",
        help="Field name for the prompt/question in the source file.",
    )
    parser.add_argument(
        "--chosen-key",
        type=str,
        default="response_chosen",
        help="Field name for the preferred response in the source file.",
    )
    parser.add_argument(
        "--rejected-key",
        type=str,
        default="response_rejected",
        help="Field name for the rejected response in the source file.",
    )
    return parser


def convert_record(record: dict, question_key: str, chosen_key: str, rejected_key: str) -> dict:
    if "conversations" in record and "chosen" in record and "rejected" in record:
        output = {
            "conversations": record["conversations"],
            "chosen": record["chosen"],
            "rejected": record["rejected"],
        }
        if "tools" in record:
            output["tools"] = record["tools"]
        return output

    question = record.get(question_key, "")
    chosen = record.get(chosen_key, "")
    rejected = record.get(rejected_key, "")

    if not question or chosen == "" or rejected == "":
        missing = [
            name
            for name, value in (
                (question_key, question),
                (chosen_key, chosen),
                (rejected_key, rejected),
            )
            if not value
        ]
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")

    output = {
        "conversations": [{"from": "human", "value": question}],
        "chosen": chosen,
        "rejected": rejected,
    }

    if "tools" in record:
        output["tools"] = record["tools"]

    return output


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0

    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line_number, line in enumerate(fin, start=1):
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                converted = convert_record(record, args.question_key, args.chosen_key, args.rejected_key)
            except Exception as exc:
                raise RuntimeError(f"Failed to convert line {line_number}: {exc}") from exc

            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(f"Converted {written} / {total} lines: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()