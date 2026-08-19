#!/usr/bin/env python3
"""Select a stable recovery batch from sources without accepted natural pairs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dpo_common import NATURAL_PAIR_TYPES, read_jsonl, stable_int, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/dpo/answer_v1/00_sources.jsonl")
    parser.add_argument("--pairs", default="data/dpo/answer_v1/02_pair_candidates.jsonl")
    parser.add_argument(
        "--reconciled",
        default="data/dpo/answer_v1/03_reconciled_judgments.jsonl",
    )
    parser.add_argument(
        "--exclude-candidates",
        action="append",
        default=[],
        help="Candidate JSONL from earlier recovery rounds; may be repeated.",
    )
    parser.add_argument(
        "--output", default="data/dpo/answer_v1/recovery/round_01_sources.jsonl"
    )
    parser.add_argument("--stats-output", default="")
    parser.add_argument("--train-limit", type=int, default=225)
    parser.add_argument("--validation-limit", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_recovery_sources(
    sources: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    reconciled: list[dict[str, Any]],
    excluded_source_ids: set[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_by_id = {row["pair_id"]: row for row in pairs}
    accepted_sources = {
        pair_by_id[row["pair_id"]]["source_id"]
        for row in reconciled
        if row.get("consistency_tier") in {"A", "B"}
        and row.get("pair_id") in pair_by_id
        and pair_by_id[row["pair_id"]].get("pair_type") in NATURAL_PAIR_TYPES
    }
    eligible = [
        row
        for row in sources
        if row["source_id"] not in accepted_sources
        and row["source_id"] not in excluded_source_ids
    ]
    limits = {"train": args.train_limit, "validation": args.validation_limit}
    selected: list[dict[str, Any]] = []
    for split, limit in limits.items():
        values = [row for row in eligible if row.get("split") == split]
        values.sort(
            key=lambda row: stable_int(
                "dpo_recovery_source", row["source_id"], seed=args.seed
            )
        )
        selected.extend(values[: max(0, limit)])
    stats = {
        "sources": len(sources),
        "accepted_natural_sources": len(accepted_sources),
        "excluded_previous_recovery_sources": len(excluded_source_ids),
        "eligible": len(eligible),
        "eligible_by_split": dict(Counter(row["split"] for row in eligible)),
        "selected": len(selected),
        "selected_by_split": dict(Counter(row["split"] for row in selected)),
        "requested_by_split": limits,
        "source_exhausted": len(selected) < sum(max(0, value) for value in limits.values()),
        "seed": args.seed,
    }
    return selected, stats


def main() -> None:
    args = parse_args()
    excluded: set[str] = set()
    for path in args.exclude_candidates:
        excluded.update(
            row["source_id"]
            for row in read_jsonl(path)
            if row.get("source_id")
        )
    selected, stats = select_recovery_sources(
        read_jsonl(args.sources),
        read_jsonl(args.pairs),
        read_jsonl(args.reconciled),
        excluded,
        args,
    )
    write_jsonl(args.output, selected)
    stats_path = (
        Path(args.stats_output)
        if args.stats_output
        else Path(args.output).with_suffix(".stats.json")
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
