#!/usr/bin/env python3
"""Merge DPO JSONL files while rejecting duplicate or conflicting keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dpo_common import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="pair_id")
    return parser.parse_args()


def merge_rows(
    input_rows: list[tuple[str, list[dict[str, Any]]]], key: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: dict[str, tuple[str, dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    for name, rows in input_rows:
        counts[name] = len(rows)
        for row in rows:
            value = str(row.get(key, ""))
            if not value:
                raise ValueError(f"{name} row is missing {key}")
            if value in seen:
                previous_name, previous = seen[value]
                if row != previous:
                    raise ValueError(
                        f"conflicting {key}={value} in {previous_name} and {name}"
                    )
                raise ValueError(f"duplicate {key}={value} in {previous_name} and {name}")
            seen[value] = (name, row)
            merged.append(row)
    return merged, {
        "inputs": counts,
        "output_rows": len(merged),
        "key": key,
        "unique_keys": len(seen),
    }


def main() -> None:
    args = parse_args()
    inputs = [(path, read_jsonl(path)) for path in args.inputs]
    merged, stats = merge_rows(inputs, args.key)
    write_jsonl(args.output, merged)
    stats_path = Path(args.output).with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
