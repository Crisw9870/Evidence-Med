#!/usr/bin/env python3
"""Build recovery pairs from one new t0.6 answer and existing candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_dpo_pairs import (
    _candidate_order_key,
    _make_pair,
    _normalized_model_candidate,
    _target_candidate,
)
from dpo_common import answer_level_signature, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources", default="data/dpo/answer_v1/recovery/round_01_sources.jsonl"
    )
    parser.add_argument(
        "--existing-candidates",
        default="data/dpo/answer_v1/01_sft_candidates.jsonl",
    )
    parser.add_argument(
        "--new-candidates",
        default="data/dpo/answer_v1/recovery/round_01_candidates.jsonl",
    )
    parser.add_argument(
        "--output", default="data/dpo/answer_v1/recovery/round_01_pairs.jsonl"
    )
    parser.add_argument("--stats-output", default="")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_recovery_pairs(
    sources: list[dict[str, Any]],
    existing_candidate_rows: list[dict[str, Any]],
    new_candidate_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_by_source = {row["source_id"]: row for row in existing_candidate_rows}
    new_by_source = {row["source_id"]: row for row in new_candidate_rows}
    pairs: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for source in sources:
        source_id = source["source_id"]
        new_row = new_by_source.get(source_id)
        if new_row is None:
            reasons["missing_new_candidate_record"] += 1
            continue
        normalized_new = [
            _normalized_model_candidate(
                source,
                {
                    **candidate,
                    "candidate_id": (
                        f"recovery_r{args.round:02d}_"
                        f"{candidate.get('candidate_id', 'sample')}"
                    ),
                },
            )
            for candidate in new_row.get("candidates", [])
            if isinstance(candidate, dict) and str(candidate.get("text", "")).strip()
        ]
        normalized_new = [item for item in normalized_new if item is not None]
        if not normalized_new:
            reasons["no_projectable_new_candidate"] += 1
            continue
        new_candidate = normalized_new[0]
        target = _target_candidate(source)
        target_signature = answer_level_signature(target["answer_view"])
        new_signature = answer_level_signature(new_candidate["answer_view"])
        if new_signature == target_signature:
            reasons["new_duplicates_target"] += 1
        else:
            pair = _make_pair(
                source,
                "target_vs_new",
                target,
                new_candidate,
                args.seed + args.round,
            )
            pair["recovery"] = {
                "round": args.round,
                "kind": "target_vs_new",
                "new_candidate_id": new_candidate["candidate_id"],
            }
            pairs.append(pair)
            reasons["target_vs_new"] += 1

        existing_row = existing_by_source.get(source_id, {})
        normalized_existing = [
            _normalized_model_candidate(source, candidate)
            for candidate in existing_row.get("candidates", [])
            if isinstance(candidate, dict) and str(candidate.get("text", "")).strip()
        ]
        unique_existing: dict[str, dict[str, Any]] = {}
        for item in normalized_existing:
            if item is None:
                continue
            signature = answer_level_signature(item["answer_view"])
            if signature not in {target_signature, new_signature}:
                unique_existing.setdefault(signature, item)
        if not unique_existing:
            reasons["no_distinct_existing_candidate"] += 1
            continue
        ranked = sorted(
            unique_existing.values(),
            key=lambda item: _candidate_order_key(
                source, item, source["target"], args.seed
            ),
            reverse=True,
        )
        best_existing = ranked[0]
        pair = _make_pair(
            source,
            "best_existing_vs_new",
            best_existing,
            new_candidate,
            args.seed + args.round,
        )
        pair["recovery"] = {
            "round": args.round,
            "kind": "best_existing_vs_new",
            "new_candidate_id": new_candidate["candidate_id"],
            "existing_candidate_id": best_existing["candidate_id"],
        }
        pairs.append(pair)
        reasons["best_existing_vs_new"] += 1

    stats = {
        "sources": len(sources),
        "new_candidate_records": len(new_candidate_rows),
        "pairs": len(pairs),
        "pairs_by_type": dict(Counter(row["pair_type"] for row in pairs)),
        "pairs_by_split": dict(Counter(row["split"] for row in pairs)),
        "reasons": dict(reasons),
        "round": args.round,
        "seed": args.seed,
    }
    return pairs, stats


def main() -> None:
    args = parse_args()
    pairs, stats = build_recovery_pairs(
        read_jsonl(args.sources),
        read_jsonl(args.existing_candidates),
        read_jsonl(args.new_candidates),
        args,
    )
    write_jsonl(args.output, pairs)
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
