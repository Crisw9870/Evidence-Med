#!/usr/bin/env python3
"""Filter answer-level judgments and export isolated DPO train/validation files."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dpo_common import (
    DPO_SCHEMA_VERSION,
    NATURAL_PAIR_TYPES,
    answer_level_differences,
    is_isolated_answer_pair,
    judge_score_total,
    pair_type_group,
    read_jsonl,
    valid_judgment,
    write_jsonl,
)


TYPE_WEIGHTS = {
    "target_vs_model": 0.55,
    "model_vs_model": 0.35,
    "controlled_negative": 0.10,
}
NATURAL_PAIR_GROUPS = {"target_vs_model", "model_vs_model"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs", default="data/dpo/answer_v1/02_pair_candidates.jsonl"
    )
    parser.add_argument(
        "--judgments",
        default="data/dpo/answer_v1/03_reconciled_judgments.jsonl",
    )
    parser.add_argument("--output-root", default="data/dpo/answer_v1")
    parser.add_argument("--train-limit", type=int, default=3000)
    parser.add_argument("--validation-limit", type=int, default=200)
    parser.add_argument("--min-train", type=int, default=1500)
    parser.add_argument("--min-validation", type=int, default=100)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--min-score-margin", type=int, default=2)
    parser.add_argument("--max-pairs-per-source", type=int, default=2)
    parser.add_argument("--target-vs-model-weight", type=float, default=0.55)
    parser.add_argument("--model-vs-model-weight", type=float, default=0.35)
    parser.add_argument("--controlled-weight", type=float, default=0.10)
    parser.add_argument("--max-tier-b-natural-fraction", type=float, default=0.20)
    parser.add_argument("--min-original-a-win-fraction", type=float, default=0.45)
    parser.add_argument("--max-original-a-win-fraction", type=float, default=0.55)
    parser.add_argument(
        "--allow-type-backfill",
        action="store_true",
        help="Legacy mode: fill unused type quotas from other pair types.",
    )
    parser.add_argument(
        "--no-require-swap-consistency",
        action="store_true",
        help="Diagnostic compatibility mode only; do not use for final export.",
    )
    parser.add_argument(
        "--recovery-status",
        default="",
        help="Optional recovery/exhaustion stats JSON included in final export stats.",
    )
    return parser.parse_args()


def _without_parsed(audit: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if key != "parsed_output"}


def _qualify(
    pair: dict[str, Any], judgment_row: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any] | None, str]:
    if judgment_row.get("status") != "ok":
        return None, "judgment_not_ok"
    judgment = judgment_row.get("parsed_judgment")
    if not valid_judgment(judgment):
        return None, "invalid_judgment"
    if judgment["decision"] not in {"A_better", "B_better"}:
        return None, f"decision:{judgment['decision']}"
    if float(judgment["confidence"]) < args.min_confidence:
        return None, "low_confidence"

    chosen_label = "A" if judgment["decision"] == "A_better" else "B"
    rejected_label = "B" if chosen_label == "A" else "A"
    if judgment["hard_failures"][chosen_label]:
        return None, "chosen_judge_hard_failure"
    margin = judge_score_total(judgment, chosen_label) - judge_score_total(
        judgment, rejected_label
    )
    if margin < args.min_score_margin:
        return None, "low_score_margin"

    chosen = pair[f"candidate_{chosen_label}"]
    rejected = pair[f"candidate_{rejected_label}"]
    if chosen["text"].strip() == rejected["text"].strip():
        return None, "identical_completions"

    chosen_audit = chosen.get("audit")
    rejected_audit = rejected.get("audit")
    if not isinstance(chosen_audit, dict) or not isinstance(rejected_audit, dict):
        return None, "missing_candidate_audit"
    if chosen_audit.get("hard_failures") or chosen_audit.get("schema_errors"):
        return None, "chosen_automatic_failure"
    if rejected_audit.get("hard_failures") or rejected_audit.get("schema_errors"):
        return None, "rejected_automatic_failure"

    chosen_parsed = chosen_audit.get("parsed_output")
    rejected_parsed = rejected_audit.get("parsed_output")
    if (
        not isinstance(chosen_parsed, dict)
        or not isinstance(rejected_parsed, dict)
        or not is_isolated_answer_pair(chosen_parsed, rejected_parsed)
    ):
        return None, "answer_level_isolation_failure"

    return {
        "pair": pair,
        "judgment": judgment,
        "chosen": chosen,
        "rejected": rejected,
        "chosen_label": chosen_label,
        "rejected_label": rejected_label,
        "margin": margin,
        "difference_fields": answer_level_differences(chosen_parsed, rejected_parsed),
    }, "qualified"


def _type_quotas(
    limit: int, weights: dict[str, float] | None = None
) -> dict[str, int]:
    weights = weights or TYPE_WEIGHTS
    raw = {pair_type: limit * weight for pair_type, weight in weights.items()}
    quotas = {pair_type: math.floor(value) for pair_type, value in raw.items()}
    remainder = limit - sum(quotas.values())
    order = sorted(
        weights,
        key=lambda pair_type: (
            raw[pair_type] - quotas[pair_type],
            weights[pair_type],
            pair_type,
        ),
        reverse=True,
    )
    for pair_type in order[:remainder]:
        quotas[pair_type] += 1
    return quotas


def _configured_weights(args: argparse.Namespace) -> dict[str, float]:
    weights = {
        "target_vs_model": float(
            getattr(args, "target_vs_model_weight", TYPE_WEIGHTS["target_vs_model"])
        ),
        "model_vs_model": float(
            getattr(args, "model_vs_model_weight", TYPE_WEIGHTS["model_vs_model"])
        ),
        "controlled_negative": float(
            getattr(args, "controlled_weight", TYPE_WEIGHTS["controlled_negative"])
        ),
    }
    if any(value < 0 for value in weights.values()):
        raise ValueError("pair type weights must be non-negative")
    total = sum(weights.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9):
        raise ValueError(f"pair type weights must sum to 1.0, got {total}")
    if weights["controlled_negative"] > 0.10 + 1e-9:
        raise ValueError("controlled_negative weight must not exceed 0.10")
    return weights


def _quality_key(item: dict[str, Any]) -> tuple[Any, ...]:
    tier = item["judgment_row"].get("consistency_tier")
    tier_rank = {"A": 2, "B": 1, "controlled": 2}.get(tier, 0)
    swap = item["judgment_row"].get("swap_consistency") or {}
    return (
        tier_rank,
        min(
            float(item["judgment"]["confidence"]),
            float(swap.get("swap_confidence", item["judgment"]["confidence"])),
        ),
        min(item["margin"], int(swap.get("swap_mapped_margin", item["margin"]))),
        item["pair"]["pair_id"],
    )


def _select_position_bucket(
    candidates: list[dict[str, Any]],
    count: int,
    selected: list[dict[str, Any]],
    selected_ids: set[str],
    source_counts: Counter[str],
    max_per_source: int,
    tier_b_count: int,
    max_tier_b: int,
) -> tuple[bool, int]:
    for item in sorted(candidates, key=_quality_key, reverse=True):
        if count <= 0:
            break
        pair = item["pair"]
        pair_id = pair["pair_id"]
        source_id = pair["source_id"]
        tier = item["judgment_row"].get("consistency_tier")
        if pair_id in selected_ids or source_counts[source_id] >= max_per_source:
            continue
        if tier == "B" and tier_b_count >= max_tier_b:
            continue
        selected.append(item)
        selected_ids.add(pair_id)
        source_counts[source_id] += 1
        if tier == "B":
            tier_b_count += 1
        count -= 1
    return count == 0, tier_b_count


def _attempt_strict_selection(
    by_type: dict[str, list[dict[str, Any]]],
    quotas: dict[str, int],
    max_per_source: int,
    max_tier_b_fraction: float,
    min_a_fraction: float,
    max_a_fraction: float,
    type_order: tuple[str, ...],
) -> list[dict[str, Any]] | None:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    natural_quota = sum(quotas[pair_type] for pair_type in NATURAL_PAIR_GROUPS)
    max_tier_b = math.floor(natural_quota * max_tier_b_fraction)
    tier_b_count = 0

    for pair_type in type_order:
        quota = quotas[pair_type]
        values = by_type.get(pair_type, [])
        if pair_type == "controlled_negative":
            ok, tier_b_count = _select_position_bucket(
                values,
                quota,
                selected,
                selected_ids,
                source_counts,
                max_per_source,
                tier_b_count,
                max_tier_b,
            )
            if not ok:
                return None
            continue

        by_position = {
            "A": [
                item
                for item in values
                if item["judgment"]["decision"] == "A_better"
            ],
            "B": [
                item
                for item in values
                if item["judgment"]["decision"] == "B_better"
            ],
        }
        min_a = math.ceil(quota * min_a_fraction)
        max_a = math.floor(quota * max_a_fraction)
        if min_a > max_a:
            return None
        preferred_a = min(max(quota // 2, min_a), max_a)
        possible_a = [
            count
            for count in range(min_a, max_a + 1)
            if count <= len(by_position["A"])
            and quota - count <= len(by_position["B"])
        ]
        possible_a.sort(key=lambda count: (abs(count - preferred_a), count))
        if not possible_a:
            return None

        snapshot = (
            list(selected),
            set(selected_ids),
            Counter(source_counts),
            tier_b_count,
        )
        selected_this_type = False
        for a_count in possible_a:
            selected[:], selected_ids, source_counts, tier_b_count = (
                list(snapshot[0]),
                set(snapshot[1]),
                Counter(snapshot[2]),
                snapshot[3],
            )
            bucket_counts = {"A": a_count, "B": quota - a_count}
            position_order = sorted(
                ("A", "B"),
                key=lambda label: (
                    len(by_position[label]) - bucket_counts[label],
                    label,
                ),
            )
            ok = True
            for label in position_order:
                bucket_ok, tier_b_count = _select_position_bucket(
                    by_position[label],
                    bucket_counts[label],
                    selected,
                    selected_ids,
                    source_counts,
                    max_per_source,
                    tier_b_count,
                    max_tier_b,
                )
                if not bucket_ok:
                    ok = False
                    break
            if ok:
                selected_this_type = True
                break
        if not selected_this_type:
            return None
    return selected


def _select_split_strict(
    items: list[dict[str, Any]],
    limit: int,
    max_per_source: int,
    weights: dict[str, float],
    max_tier_b_fraction: float,
    min_a_fraction: float,
    max_a_fraction: float,
) -> list[dict[str, Any]] | None:
    quotas = _type_quotas(limit, weights)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        group = item.get("quota_pair_type") or pair_type_group(
            item["pair"]["pair_type"]
        )
        if group in TYPE_WEIGHTS:
            by_type[group].append(item)
    natural_orders = list(itertools.permutations(sorted(NATURAL_PAIR_GROUPS)))
    for natural_order in natural_orders:
        selected = _attempt_strict_selection(
            by_type,
            quotas,
            max_per_source,
            max_tier_b_fraction,
            min_a_fraction,
            max_a_fraction,
            (*natural_order, "controlled_negative"),
        )
        if selected is not None and len(selected) == limit:
            return selected
    return None


def _find_maximum_feasible_split(
    items: list[dict[str, Any]],
    maximum: int,
    minimum: int,
    args: argparse.Namespace,
    weights: dict[str, float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def availability() -> dict[str, Any]:
        quotas = _type_quotas(minimum, weights)
        by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            group = item.get("quota_pair_type") or pair_type_group(
                item["pair"]["pair_type"]
            )
            if group in TYPE_WEIGHTS:
                by_group[group].append(item)
        groups: dict[str, Any] = {}
        for group in TYPE_WEIGHTS:
            values = by_group[group]
            details: dict[str, Any] = {
                "available": len(values),
                "minimum_quota": quotas[group],
                "raw_shortfall": max(0, quotas[group] - len(values)),
                "tiers": dict(
                    Counter(
                        item["judgment_row"].get("consistency_tier")
                        for item in values
                    )
                ),
            }
            if group in NATURAL_PAIR_GROUPS:
                positions = Counter(
                    item["judgment"]["decision"][0] for item in values
                )
                min_a = math.ceil(quotas[group] * args.min_original_a_win_fraction)
                max_a = math.floor(quotas[group] * args.max_original_a_win_fraction)
                min_b = quotas[group] - max_a
                details["positions"] = dict(positions)
                details["minimum_position_counts"] = {"A": min_a, "B": min_b}
                details["position_shortfall"] = {
                    "A": max(0, min_a - positions["A"]),
                    "B": max(0, min_b - positions["B"]),
                }
            groups[group] = details
        return {
            "at_minimum_size": minimum,
            "groups": groups,
            "unique_sources": len({item["pair"]["source_id"] for item in items}),
            "source_cap_upper_bound": len(
                {item["pair"]["source_id"] for item in items}
            )
            * args.max_pairs_per_source,
        }

    if maximum < minimum:
        return [], {
            "ready": False,
            "reason": "maximum_below_minimum",
            "availability": availability(),
        }
    max_available = min(maximum, len(items))
    for limit in range(max_available, minimum - 1, -1):
        quotas = _type_quotas(limit, weights)
        # Largest-remainder rounding can assign the remainder to the
        # controlled bucket. When its configured weight is exactly 10%, that
        # can make the actual share exceed the hard safety cap. Treat that
        # size as infeasible and continue searching for the next valid size.
        if quotas["controlled_negative"] > math.floor(limit * 0.10 + 1e-9):
            continue
        selected = _select_split_strict(
            items,
            limit,
            args.max_pairs_per_source,
            weights,
            float(args.max_tier_b_natural_fraction),
            float(args.min_original_a_win_fraction),
            float(args.max_original_a_win_fraction),
        )
        if selected is not None:
            return selected, {
                "ready": True,
                "requested_maximum": maximum,
                "selected_limit": limit,
                "target_quotas": quotas,
                "reduced_by": maximum - limit,
            }
    return [], {
        "ready": False,
        "reason": "no_feasible_size_at_or_above_minimum",
        "requested_maximum": maximum,
        "minimum": minimum,
        "availability": availability(),
    }


def _select_split(
    items: list[dict[str, Any]], limit: int, max_per_source: int
) -> list[dict[str, Any]]:
    if limit <= 0:
        limit = len(items)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_type[item["pair"]["pair_type"]].append(item)
    for values in by_type.values():
        values.sort(
            key=lambda item: (
                float(item["judgment"]["confidence"]),
                item["margin"],
                item["pair"]["pair_id"],
            ),
            reverse=True,
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    source_counts: Counter[str] = Counter()
    quotas = _type_quotas(limit)
    for pair_type in TYPE_WEIGHTS:
        type_count = 0
        for item in by_type[pair_type]:
            if len(selected) >= limit or type_count >= quotas[pair_type]:
                break
            pair_id = item["pair"]["pair_id"]
            source_id = item["pair"]["source_id"]
            if pair_id in selected_ids or source_counts[source_id] >= max_per_source:
                continue
            selected.append(item)
            selected_ids.add(pair_id)
            source_counts[source_id] += 1
            type_count += 1

    remaining = [
        item for item in items if item["pair"]["pair_id"] not in selected_ids
    ]
    remaining.sort(
        key=lambda item: (
            float(item["judgment"]["confidence"]),
            item["margin"],
            item["pair"]["pair_id"],
        ),
        reverse=True,
    )
    for item in remaining:
        if len(selected) >= min(limit, len(items)):
            break
        pair_id = item["pair"]["pair_id"]
        source_id = item["pair"]["source_id"]
        if pair_id in selected_ids or source_counts[source_id] >= max_per_source:
            continue
        selected.append(item)
        selected_ids.add(pair_id)
        source_counts[source_id] += 1
    return selected


def export_dataset(
    pairs: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    pair_ids = [str(row.get("pair_id", "")) for row in pairs]
    if not all(pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("pair candidates contain missing or duplicate pair_id")
    judgment_ids = [str(row.get("pair_id", "")) for row in judgments]
    if not all(judgment_ids) or len(judgment_ids) != len(set(judgment_ids)):
        raise ValueError("judgments contain missing or duplicate pair_id")

    pair_by_id = {row["pair_id"]: row for row in pairs}
    allow_type_backfill = bool(getattr(args, "allow_type_backfill", True))
    strict_type_quotas = not allow_type_backfill
    require_swap_consistency = (
        strict_type_quotas
        and not bool(getattr(args, "no_require_swap_consistency", True))
    )
    weights = _configured_weights(args) if strict_type_quotas else TYPE_WEIGHTS
    qualified: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    rejected_counts = Counter()
    for judgment_row in judgments:
        pair = pair_by_id.get(judgment_row["pair_id"])
        if not pair or pair.get("split") not in qualified:
            rejected_counts["missing_pair_or_locked_split"] += 1
            continue
        if (
            judgment_row.get("source_id") != pair.get("source_id")
            or judgment_row.get("split") != pair.get("split")
        ):
            rejected_counts["judgment_pair_metadata_mismatch"] += 1
            continue
        pair_type = pair.get("pair_type")
        if require_swap_consistency:
            tier = judgment_row.get("consistency_tier")
            swap = judgment_row.get("swap_consistency")
            if pair_type in NATURAL_PAIR_TYPES and (
                tier not in {"A", "B"}
                or not isinstance(swap, dict)
                or swap.get("required") is not True
                or swap.get("accepted") is not True
            ):
                rejected_counts["missing_natural_swap_consistency"] += 1
                continue
            if pair_type == "controlled_negative" and tier != "controlled":
                rejected_counts["missing_controlled_reconciliation"] += 1
                continue
        item, reason = _qualify(pair, judgment_row, args)
        if item is None:
            rejected_counts[reason] += 1
            continue
        item["judgment_row"] = judgment_row
        item["quota_pair_type"] = pair_type_group(pair_type)
        if item["quota_pair_type"] not in TYPE_WEIGHTS:
            rejected_counts["unsupported_pair_type"] += 1
            continue
        qualified[pair["split"]].append(item)

    capacity: dict[str, Any]
    if strict_type_quotas:
        train_selected, train_capacity = _find_maximum_feasible_split(
            qualified["train"],
            args.train_limit,
            args.min_train,
            args,
            weights,
        )
        validation_selected, validation_capacity = _find_maximum_feasible_split(
            qualified["validation"],
            args.validation_limit,
            args.min_validation,
            args,
            weights,
        )
        selected = {
            "train": train_selected,
            "validation": validation_selected,
        }
        capacity = {
            "train": train_capacity,
            "validation": validation_capacity,
            "ready": bool(
                train_capacity.get("ready") and validation_capacity.get("ready")
            ),
        }
        if not capacity["ready"]:
            selected = {"train": [], "validation": []}
    else:
        selected = {
            "train": _select_split(
                qualified["train"], args.train_limit, args.max_pairs_per_source
            ),
            "validation": _select_split(
                qualified["validation"],
                args.validation_limit,
                args.max_pairs_per_source,
            ),
        }
        capacity = {
            "train": {"ready": True, "selected_limit": len(selected["train"])},
            "validation": {
                "ready": True,
                "selected_limit": len(selected["validation"]),
            },
            "ready": True,
            "mode": "legacy_type_backfill",
        }
    datasets: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    audits: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    for split, items in selected.items():
        for item in items:
            pair, judgment = item["pair"], item["judgment"]
            datasets[split].append(
                {
                    "schema_version": DPO_SCHEMA_VERSION,
                    "preference_scope": "answer_level",
                    "source_id": pair["source_id"],
                    "pair_id": pair["pair_id"],
                    "prompt": pair["prompt"],
                    "conversations": [
                        {"from": "human", "value": pair["prompt"]}
                    ],
                    "chosen": item["chosen"]["text"],
                    "rejected": item["rejected"]["text"],
                }
            )
            audits[split].append(
                {
                    "schema_version": DPO_SCHEMA_VERSION,
                    "preference_scope": "answer_level",
                    "source_id": pair["source_id"],
                    "pair_id": pair["pair_id"],
                    "split": split,
                    "pair_type": pair["pair_type"],
                    "pair_type_group": item["quota_pair_type"],
                    "difference_fields": item["difference_fields"],
                    "chosen_origin": item["chosen"].get("origin"),
                    "rejected_origin": item["rejected"].get("origin"),
                    "chosen_candidate_id": item["chosen"].get("candidate_id"),
                    "rejected_candidate_id": item["rejected"].get("candidate_id"),
                    "chosen_audit": _without_parsed(
                        item["chosen"].get("audit", {})
                    ),
                    "rejected_audit": _without_parsed(
                        item["rejected"].get("audit", {})
                    ),
                    "judge": judgment,
                    "score_margin": item["margin"],
                    "consistency_tier": item["judgment_row"].get(
                        "consistency_tier"
                    ),
                    "swap_consistency": item["judgment_row"].get(
                        "swap_consistency"
                    ),
                    "original_winner_position": item["chosen_label"],
                }
            )

    train_ids = {row["source_id"] for row in datasets["train"]}
    validation_ids = {row["source_id"] for row in datasets["validation"]}
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            f"source_id leakage between train and validation: {sorted(overlap)[:5]}"
        )
    if strict_type_quotas and capacity["ready"]:
        for split, rows in audits.items():
            expected = _type_quotas(len(rows), weights)
            actual = Counter(row["pair_type_group"] for row in rows)
            if any(actual[pair_type] != count for pair_type, count in expected.items()):
                raise ValueError(f"{split} pair type quotas were not met")
            if actual["controlled_negative"] > math.floor(len(rows) * 0.10 + 1e-9):
                raise ValueError(f"{split} controlled_negative exceeds 10%")
            natural = [
                row
                for row in rows
                if row["pair_type_group"] in {"target_vs_model", "model_vs_model"}
            ]
            tier_b = sum(row["consistency_tier"] == "B" for row in natural)
            if natural and tier_b / len(natural) > args.max_tier_b_natural_fraction:
                raise ValueError(f"{split} Tier B exceeds configured cap")
            for pair_type in NATURAL_PAIR_GROUPS:
                type_rows = [
                    row
                    for row in rows
                    if row["pair_type_group"] == pair_type
                ]
                if not type_rows:
                    raise ValueError(f"{split} missing required type {pair_type}")
                a_fraction = sum(
                    row["original_winner_position"] == "A" for row in type_rows
                ) / len(type_rows)
                if not (
                    args.min_original_a_win_fraction
                    <= a_fraction
                    <= args.max_original_a_win_fraction
                ):
                    raise ValueError(
                        f"{split} {pair_type} original A-win fraction "
                        f"{a_fraction:.4f} outside configured range"
                    )
    pair_types = {
        split: dict(Counter(row["pair_type_group"] for row in audits[split]))
        for split in audits
    }
    raw_pair_types = {
        split: dict(Counter(row["pair_type"] for row in audits[split]))
        for split in audits
    }
    tiers = {
        split: dict(
            Counter(row.get("consistency_tier") for row in audits[split])
        )
        for split in audits
    }
    original_positions = {
        split: {
            pair_type: dict(
                Counter(
                    row["original_winner_position"]
                    for row in audits[split]
                    if row["pair_type_group"] == pair_type
                )
            )
            for pair_type in sorted(NATURAL_PAIR_GROUPS)
        }
        for split in audits
    }
    position_fractions = {
        split: {
            pair_type: (
                counts.get("A", 0) / sum(counts.values())
                if sum(counts.values())
                else 0.0
            )
            for pair_type, counts in original_positions[split].items()
        }
        for split in audits
    }
    unexported_by_type = {
        split: {
            pair_type: sum(
                item["quota_pair_type"] == pair_type for item in qualified[split]
            )
            - pair_types[split].get(pair_type, 0)
            for pair_type in TYPE_WEIGHTS
        }
        for split in qualified
    }
    recovery_status = None
    recovery_status_path = getattr(args, "recovery_status", "")
    if recovery_status_path:
        recovery_status = json.loads(
            Path(recovery_status_path).read_text(encoding="utf-8")
        )
    final_status = "ready" if capacity["ready"] else "recovery_required"
    if (
        not capacity["ready"]
        and isinstance(recovery_status, dict)
        and recovery_status.get("source_exhausted")
    ):
        final_status = "preference_data_insufficient"
    stats = {
        "schema_version": DPO_SCHEMA_VERSION,
        "preference_scope": "answer_level",
        "qualified": {
            split: len(values) for split, values in qualified.items()
        },
        "exported": {
            split: len(values) for split, values in datasets.items()
        },
        "pair_types": pair_types,
        "raw_pair_types": raw_pair_types,
        "target_type_weights": weights,
        "actual_type_ratios": {
            split: {
                pair_type: (
                    count / len(audits[split]) if audits[split] else 0.0
                )
                for pair_type, count in pair_types[split].items()
            }
            for split in audits
        },
        "consistency_tiers": tiers,
        "original_winner_positions": original_positions,
        "original_a_win_fractions": position_fractions,
        "difference_fields": {
            split: dict(
                Counter(
                    field
                    for row in audits[split]
                    for field in row["difference_fields"]
                )
            )
            for split in audits
        },
        "unique_sources": {
            split: len({row["source_id"] for row in datasets[split]})
            for split in datasets
        },
        "rejected": dict(rejected_counts),
        "qualified_not_exported_by_type": unexported_by_type,
        "capacity": capacity,
        "export_ready": capacity["ready"],
        "final_status": final_status,
        "recovery_status": recovery_status,
        "min_confidence": args.min_confidence,
        "min_score_margin": args.min_score_margin,
        "max_pairs_per_source": args.max_pairs_per_source,
        "swap_consistency_check": require_swap_consistency,
        "type_backfill_allowed": allow_type_backfill,
        "max_tier_b_natural_fraction": getattr(
            args, "max_tier_b_natural_fraction", None
        ),
        "original_a_win_fraction_range": [
            getattr(args, "min_original_a_win_fraction", None),
            getattr(args, "max_original_a_win_fraction", None),
        ],
    }
    return datasets, audits, stats


def main() -> None:
    args = parse_args()
    datasets, audits, stats = export_dataset(
        read_jsonl(args.pairs), read_jsonl(args.judgments), args
    )
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    stats_path = root / "04_export.stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = {
        "export_ready": stats["export_ready"],
        "final_status": stats["final_status"],
        "formal_files_written_this_run": bool(stats["export_ready"]),
        "stats_path": str(stats_path),
        "capacity": stats["capacity"],
    }
    (root / "04_export.status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not stats["export_ready"]:
        raise SystemExit(
            "Preference data insufficient: strict export conditions were not met; "
            "formal train/validation files were not modified."
        )
    write_jsonl(root / "train/train.jsonl", datasets["train"])
    write_jsonl(root / "validation/validation.jsonl", datasets["validation"])
    write_jsonl(root / "audit/train_audit.jsonl", audits["train"])
    write_jsonl(root / "audit/validation_audit.jsonl", audits["validation"])
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"DPO dataset root: {root}")


if __name__ == "__main__":
    main()
