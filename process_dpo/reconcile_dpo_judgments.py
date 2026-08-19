#!/usr/bin/env python3
"""Reconcile original and A/B-swapped DPO judgments into auditable tiers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dpo_common import (
    NATURAL_PAIR_TYPES,
    judge_score_total,
    read_jsonl,
    valid_judgment,
    write_jsonl,
)
from export_dpo_dataset import _qualify


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/dpo/answer_v1/02_pair_candidates.jsonl")
    parser.add_argument("--judgments", default="data/dpo/answer_v1/03_judgments.jsonl")
    parser.add_argument(
        "--swap-judgments", default="data/dpo/answer_v1/03_swap_judgments.jsonl"
    )
    parser.add_argument(
        "--output", default="data/dpo/answer_v1/03_reconciled_judgments.jsonl"
    )
    parser.add_argument(
        "--stats-output",
        default="data/dpo/answer_v1/03_reconciled_judgments.stats.json",
    )
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--min-score-margin", type=int, default=2)
    parser.add_argument("--tier-b-confidence", type=float, default=0.90)
    parser.add_argument("--tier-b-min-direction-margin", type=int, default=1)
    parser.add_argument("--tier-b-min-average-margin", type=float, default=2.0)
    return parser.parse_args()


def _index_unique(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    ids = [str(row.get("pair_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains missing or duplicate pair_id")
    return {row["pair_id"]: row for row in rows}


def _winner_label(judgment: dict[str, Any]) -> str | None:
    decision = judgment.get("decision")
    if decision == "A_better":
        return "A"
    if decision == "B_better":
        return "B"
    return None


def _other(label: str) -> str:
    return "B" if label == "A" else "A"


def _mapped_margin(
    judgment: dict[str, Any], original_winner: str, *, swapped: bool
) -> int:
    presented_winner = _other(original_winner) if swapped else original_winner
    presented_loser = _other(presented_winner)
    return judge_score_total(judgment, presented_winner) - judge_score_total(
        judgment, presented_loser
    )


def _metadata_matches(pair: dict[str, Any], row: dict[str, Any]) -> bool:
    return (
        row.get("pair_id") == pair.get("pair_id")
        and row.get("source_id") == pair.get("source_id")
        and row.get("split") == pair.get("split")
    )


def reconcile_natural_pair(
    pair: dict[str, Any],
    original_row: dict[str, Any],
    swap_row: dict[str, Any] | None,
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, str]:
    qualify_args = SimpleNamespace(
        min_confidence=args.min_confidence,
        min_score_margin=args.min_score_margin,
    )
    original_item, original_reason = _qualify(pair, original_row, qualify_args)
    if original_item is None:
        return None, f"original:{original_reason}"
    if swap_row is None:
        return None, "swap:missing"
    if not _metadata_matches(pair, swap_row):
        return None, "swap:metadata_mismatch"
    if swap_row.get("status") != "ok":
        return None, "swap:not_ok"
    swap = swap_row.get("parsed_judgment")
    if not valid_judgment(swap):
        return None, "swap:invalid_judgment"
    if (
        original_row.get("judge_model") is not None
        and swap_row.get("judge_model") != original_row.get("judge_model")
    ):
        return None, "swap:judge_model_mismatch"

    original = original_item["judgment"]
    winner = original_item["chosen_label"]
    expected_swap_winner = _other(winner)
    original_margin = _mapped_margin(original, winner, swapped=False)
    swap_margin = _mapped_margin(swap, winner, swapped=True)
    swap_confidence = float(swap["confidence"])
    swap_winner = _winner_label(swap)
    swap_winner_hard = swap["hard_failures"][expected_swap_winner]

    tier = None
    reason = ""
    if (
        swap_winner == expected_swap_winner
        and swap_confidence >= args.min_confidence
        and swap_margin >= args.min_score_margin
        and not swap_winner_hard
    ):
        tier = "A"
        reason = "strict_bidirectional_decision"
    else:
        confidence_ok = (
            float(original["confidence"]) >= args.tier_b_confidence
            and swap_confidence >= args.tier_b_confidence
        )
        direction_ok = (
            original_margin >= args.tier_b_min_direction_margin
            and swap_margin >= args.tier_b_min_direction_margin
        )
        average_margin = (original_margin + swap_margin) / 2.0
        winner_hard_free = (
            not original["hard_failures"][winner]
            and not swap["hard_failures"][expected_swap_winner]
        )
        if (
            confidence_ok
            and direction_ok
            and average_margin >= args.tier_b_min_average_margin
            and winner_hard_free
        ):
            tier = "B"
            reason = "score_consistent_bidirectional"
        elif not confidence_ok:
            return None, "tier_b:low_confidence"
        elif not direction_ok:
            return None, "tier_b:score_direction_inconsistent"
        elif average_margin < args.tier_b_min_average_margin:
            return None, "tier_b:low_average_margin"
        else:
            return None, "tier_b:winner_hard_failure"

    reconciled = dict(original_row)
    reconciled["consistency_tier"] = tier
    reconciled["swap_consistency"] = {
        "required": True,
        "accepted": True,
        "tier": tier,
        "reason": reason,
        "original_winner": winner,
        "swap_expected_winner": expected_swap_winner,
        "swap_decision": swap["decision"],
        "original_confidence": float(original["confidence"]),
        "swap_confidence": swap_confidence,
        "original_mapped_margin": original_margin,
        "swap_mapped_margin": swap_margin,
        "average_mapped_margin": (original_margin + swap_margin) / 2.0,
        "original_judge_model": original_row.get("judge_model"),
        "swap_judge_model": swap_row.get("judge_model"),
        "swap_judge_version": swap_row.get("judge_version"),
        "swap_parsed_judgment": swap,
    }
    return reconciled, f"tier_{tier.lower()}"


def reconcile_judgments(
    pairs: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    swap_judgments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair_by_id = _index_unique(pairs, "pairs")
    original_by_id = _index_unique(judgments, "judgments")
    swap_by_id = _index_unique(swap_judgments, "swap judgments")
    qualify_args = SimpleNamespace(
        min_confidence=args.min_confidence,
        min_score_margin=args.min_score_margin,
    )
    output: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    original_eligible: Counter[str] = Counter()
    swap_covered: Counter[str] = Counter()

    for pair in pairs:
        pair_type = pair.get("pair_type")
        original_row = original_by_id.get(pair["pair_id"])
        if original_row is None or not _metadata_matches(pair, original_row):
            reasons["original:missing_or_metadata_mismatch"] += 1
            continue
        if pair_type in NATURAL_PAIR_TYPES:
            item, _ = _qualify(pair, original_row, qualify_args)
            if item is not None:
                original_eligible[pair_type] += 1
                if pair["pair_id"] in swap_by_id:
                    swap_covered[pair_type] += 1
            reconciled, reason = reconcile_natural_pair(
                pair, original_row, swap_by_id.get(pair["pair_id"]), args
            )
            reasons[reason] += 1
            if reconciled is not None:
                output.append(reconciled)
            continue
        if pair_type != "controlled_negative":
            reasons["unsupported_pair_type"] += 1
            continue
        item, reason = _qualify(pair, original_row, qualify_args)
        if item is None:
            reasons[f"controlled:{reason}"] += 1
            continue
        controlled = dict(original_row)
        controlled["consistency_tier"] = "controlled"
        controlled["swap_consistency"] = {
            "required": False,
            "accepted": True,
            "tier": "controlled",
            "reason": "controlled_single_pass",
        }
        output.append(controlled)
        reasons["controlled_qualified"] += 1

    pair_types = Counter(pair_by_id[row["pair_id"]]["pair_type"] for row in output)
    tiers = Counter(row["consistency_tier"] for row in output)
    reported_types = sorted(
        {
            pair_by_id[row["pair_id"]]["pair_type"]
            for row in output
            if row["pair_id"] in pair_by_id
        }
        | NATURAL_PAIR_TYPES
        | {"controlled_negative"}
    )
    tiers_by_type = {
        pair_type: dict(
            Counter(
                row["consistency_tier"]
                for row in output
                if pair_by_id[row["pair_id"]]["pair_type"] == pair_type
            )
        )
        for pair_type in reported_types
    }
    positions_by_type = {}
    for pair_type in NATURAL_PAIR_TYPES:
        rows = [
            row
            for row in output
            if pair_by_id[row["pair_id"]]["pair_type"] == pair_type
        ]
        positions_by_type[pair_type] = dict(
            Counter(
                "original_A_win"
                if row["parsed_judgment"]["decision"] == "A_better"
                else "original_B_win"
                for row in rows
            )
        )
    consistency_rates = {}
    for pair_type in sorted(NATURAL_PAIR_TYPES):
        eligible_count = original_eligible[pair_type]
        type_tiers = tiers_by_type.get(pair_type, {})
        tier_a_count = type_tiers.get("A", 0)
        accepted_count = tier_a_count + type_tiers.get("B", 0)
        consistency_rates[pair_type] = {
            "swap_coverage_rate": (
                swap_covered[pair_type] / eligible_count if eligible_count else 0.0
            ),
            "tier_a_rate_of_covered": (
                tier_a_count / swap_covered[pair_type]
                if swap_covered[pair_type]
                else 0.0
            ),
            "tier_a_plus_b_rate_of_covered": (
                accepted_count / swap_covered[pair_type]
                if swap_covered[pair_type]
                else 0.0
            ),
        }
    stats = {
        "input_pairs": len(pairs),
        "input_judgments": len(judgments),
        "input_swap_judgments": len(swap_judgments),
        "original_eligible_natural": dict(original_eligible),
        "swap_coverage_natural": dict(swap_covered),
        "accepted": len(output),
        "accepted_by_type": dict(pair_types),
        "tiers": dict(tiers),
        "tiers_by_type": tiers_by_type,
        "consistency_rates": consistency_rates,
        "original_position_by_type": positions_by_type,
        "reasons": dict(reasons),
        "thresholds": {
            "tier_a_min_confidence": args.min_confidence,
            "tier_a_min_margin": args.min_score_margin,
            "tier_b_min_confidence": args.tier_b_confidence,
            "tier_b_min_direction_margin": args.tier_b_min_direction_margin,
            "tier_b_min_average_margin": args.tier_b_min_average_margin,
        },
    }
    return output, stats


def main() -> None:
    args = parse_args()
    reconciled, stats = reconcile_judgments(
        read_jsonl(args.pairs),
        read_jsonl(args.judgments),
        read_jsonl(args.swap_judgments),
        args,
    )
    write_jsonl(args.output, reconciled)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
