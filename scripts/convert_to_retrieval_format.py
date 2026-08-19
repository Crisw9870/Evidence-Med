"""
Convert common JSONL dataset formats into the retrieval-style format used by
retrieval_sft.jsonl, i.e.:

{
  "conversations": [
    {"from": "human", "value": "..."},
    {"from": "gpt", "value": "..."}
  ]
}

Supported input styles:
- already has "conversations"
- {"instruction": ..., "output": ...}
- {"input": ..., "output": ...}
- {"prompt": ..., "response": ...}
- {"question": ..., "answer": ...}

Usage:
  python scripts/convert_to_retrieval_format.py \
      --input data/sft/your_file.jsonl \
      --output data/sft/your_file_retrieval.jsonl \
      --nums 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert dataset JSONL to retrieval-style conversations")
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    parser.add_argument("--nums", type=int, default=None, help="Optional number of rows to convert")
    return parser


def _prepend_label(text: str, record: dict[str, Any]) -> str:
    label = record.get("label") or record.get("category") or record.get("domain")
    if not label:
        return str(text)
    if not str(text).strip():
        return f"{str(label)}："
    return f"{str(label)}：{str(text)}"


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, str):
        raise ValueError("Each line must be a JSON object")

    if "conversations" in record:
        return record

    question = None
    answer = None

    for human_key, answer_key in [
        ("instruction", "output"),
        ("input", "output"),
        ("prompt", "response"),
        ("question", "answer"),
    ]:
        if human_key in record and answer_key in record:
            question = record[human_key]
            answer = record[answer_key]
            break

    if question is None or answer is None:
        raise ValueError(
            "Unsupported format. Need one of: instruction/output, input/output, prompt/response, question/answer, or already-conversations"
        )

    output = {
        "conversations": [
            {"from": "human", "value": _prepend_label(question, record)},
            {"from": "gpt", "value": str(answer)},
        ]
    }

    for key, value in record.items():
        if key not in {"instruction", "input", "prompt", "question", "output", "response", "answer", "id", "score", "label", "related_diseases"}:
            output[key] = value

    return output


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with args.input.open("r", encoding="utf-8") as fin, args.output.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue

            if args.nums is not None and written >= args.nums:
                break

            record = json.loads(line)
            converted = normalize_record(record)
            fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(f"Converted {written} lines: {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
