#!/usr/bin/env python3
"""Select deterministic, stratified Evidence-SFT sources for DPO construction."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

from dpo_common import (
    DPO_SCHEMA_VERSION,
    assess_target_warning_risk,
    audit_response,
    build_training_prompt,
    clean_target,
    has_high_risk_warning,
    read_jsonl,
    stable_int,
    write_jsonl,
)


SUFFICIENCY_WEIGHTS = {
    "partial": 0.55,
    "insufficient": 0.25,
    "sufficient": 0.19,
    "conflicting": 0.01,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/evidence_sft/validated_v2_2/03_validated_full.jsonl",
    )
    parser.add_argument("--output", default="data/dpo/answer_v1/00_sources.jsonl")
    parser.add_argument("--stats-output", default="data/dpo/answer_v1/00_sources.stats.json")
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--validation-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--warning-policy",
        choices=("strict", "tiered", "all"),
        default="tiered",
        help=(
            "strict reproduces the legacy one-vote veto; tiered keeps task-type "
            "corrections, rechecks strong claims against evidence, and caps medium "
            "numeric risk; all ignores warning risk after hard target audit."
        ),
    )
    parser.add_argument(
        "--max-medium-risk-fraction",
        type=float,
        default=0.15,
        help="Maximum medium-risk share selected per split under tiered policy.",
    )
    parser.add_argument(
        "--include-high-risk-warnings",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if not 0 <= args.max_medium_risk_fraction <= 1:
        parser.error("--max-medium-risk-fraction must be between 0 and 1")
    if args.include_high_risk_warnings:
        args.warning_policy = "all"
    return args


def _alternate_task_types(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target = row["target"]
        groups[str(target.get("task_type"))].append(row)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for key, values in groups.items():
        values.sort(key=lambda item: stable_int(item["source_id"], seed=seed))
        queues[key] = deque(values)
    ordered_keys = sorted(queues, key=lambda key: stable_int(*key, seed=seed))
    ordered: list[dict[str, Any]] = []
    while any(queues.values()):
        progressed = False
        for key in ordered_keys:
            if queues[key]:
                ordered.append(queues[key].popleft())
                progressed = True
        if not progressed:
            break
    return ordered


def _stratified_take(
    rows: list[dict[str, Any]],
    limit: int,
    seed: int,
    *,
    max_medium_risk_fraction: float | None = None,
) -> list[dict[str, Any]]:
    target_count = len(rows) if limit <= 0 else min(limit, len(rows))
    if max_medium_risk_fraction is not None:
        medium_budget = int(target_count * max_medium_risk_fraction)
        medium = [
            row
            for row in rows
            if row.get("_warning_risk", {}).get("level") == "medium"
        ]
        medium.sort(
            key=lambda item: stable_int(item["source_id"], "medium-risk", seed=seed)
        )
        rows = [
            row
            for row in rows
            if row.get("_warning_risk", {}).get("level") != "medium"
        ] + medium[:medium_budget]
        target_count = min(target_count, len(rows))

    by_sufficiency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sufficiency[str(row["target"].get("evidence_sufficiency"))].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for sufficiency, weight in SUFFICIENCY_WEIGHTS.items():
        quota = min(len(by_sufficiency[sufficiency]), round(target_count * weight))
        ordered = _alternate_task_types(by_sufficiency[sufficiency], seed)
        for row in ordered[:quota]:
            selected.append(row)
            selected_ids.add(row["source_id"])

    if len(selected) < target_count:
        remaining = [row for row in rows if row["source_id"] not in selected_ids]
        remaining.sort(key=lambda item: stable_int(item["source_id"], "fill", seed=seed))
        selected.extend(remaining[: target_count - len(selected)])
    selected.sort(key=lambda item: stable_int(item["source_id"], "final", seed=seed))
    return selected


def select_sources(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    warning_policy = getattr(args, "warning_policy", "tiered")
    if getattr(args, "include_high_risk_warnings", False):
        warning_policy = "all"
    if warning_policy not in {"strict", "tiered", "all"}:
        raise ValueError(f"unsupported warning policy: {warning_policy}")
    medium_fraction = float(getattr(args, "max_medium_risk_fraction", 0.15))
    if not 0 <= medium_fraction <= 1:
        raise ValueError("max_medium_risk_fraction must be between 0 and 1")

    eligible: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    rejected = Counter()
    assessed_risk = Counter()
    assessed_reasons = Counter()
    seen_ids: set[str] = set()
    for row in rows:
        source_id = str(row.get("source_id", ""))
        split = row.get("split")
        target = row.get("target")
        case_text = row.get("case_text")
        if split not in eligible:
            rejected[f"locked_split:{split}"] += 1
            continue
        if not source_id or source_id in seen_ids:
            rejected["missing_or_duplicate_source_id"] += 1
            continue
        if not isinstance(case_text, str) or not isinstance(target, dict):
            rejected["missing_case_or_target"] += 1
            continue
        warnings = row.get("validation_warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        risk = assess_target_warning_risk(case_text, target, warnings)
        assessed_risk[risk["level"]] += 1
        for reason in risk["reasons"]:
            assessed_reasons[reason.split(":", 1)[0]] += 1
        if warning_policy == "strict" and has_high_risk_warning(warnings):
            rejected["warning_policy:strict"] += 1
            continue
        if warning_policy == "tiered" and risk["level"] == "high":
            rejected["warning_risk:high"] += 1
            continue
        target_audit = audit_response(case_text, target, target)
        if target_audit["hard_failures"] or target_audit["schema_errors"]:
            rejected["invalid_target"] += 1
            continue
        seen_ids.add(source_id)
        eligible_row = dict(row)
        eligible_row["_warning_risk"] = risk
        eligible[split].append(eligible_row)

    selected: list[dict[str, Any]] = []
    for split, limit in (("train", args.train_limit), ("validation", args.validation_limit)):
        selected_rows = _stratified_take(
            eligible[split],
            limit,
            args.seed,
            max_medium_risk_fraction=(
                medium_fraction if warning_policy == "tiered" else None
            ),
        )
        for row in selected_rows:
            target = clean_target(row["target"])
            selected.append(
                {
                    "schema_version": DPO_SCHEMA_VERSION,
                    "source_id": row["source_id"],
                    "split": split,
                    "category": row.get("category", "未标注医学主题"),
                    "case_text": row["case_text"],
                    "prompt": build_training_prompt(row["case_text"]),
                    "target": target,
                    "target_warnings": row.get("validation_warnings", []),
                    "warning_risk": row["_warning_risk"],
                    "teacher_model": row.get("teacher_model"),
                }
            )

    stats = {
        "schema_version": DPO_SCHEMA_VERSION,
        "input": str(args.input),
        "input_rows": len(rows),
        "eligible": {split: len(values) for split, values in eligible.items()},
        "selected": dict(Counter(row["split"] for row in selected)),
        "task_type": dict(Counter(row["target"]["task_type"] for row in selected)),
        "sufficiency": dict(
            Counter(row["target"]["evidence_sufficiency"] for row in selected)
        ),
        "warning_policy": warning_policy,
        "max_medium_risk_fraction": (
            medium_fraction if warning_policy == "tiered" else None
        ),
        "assessed_warning_risk": dict(assessed_risk),
        "assessed_warning_reasons": dict(assessed_reasons),
        "eligible_warning_risk": {
            split: dict(
                Counter(row["_warning_risk"]["level"] for row in values)
            )
            for split, values in eligible.items()
        },
        "selected_warning_risk": dict(
            Counter(row["warning_risk"]["level"] for row in selected)
        ),
        "rejected": dict(rejected),
        "seed": args.seed,
    }
    return selected, stats


def main() -> None:
    args = parse_args()
    selected, stats = select_sources(read_jsonl(args.input), args)
    write_jsonl(args.output, selected)
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Sources: {args.output}")


if __name__ == "__main__":
    main()
