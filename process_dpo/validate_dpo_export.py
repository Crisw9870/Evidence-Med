#!/usr/bin/env python3
"""Validate a final DPO export and create a stable human-review sample."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dpo_common import (
    extract_first_json_object,
    is_isolated_answer_pair,
    read_jsonl,
    stable_int,
    write_jsonl,
)


NATURAL_GROUPS = {"target_vs_model", "model_vs_model"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/dpo/answer_v1")
    parser.add_argument("--train-sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-train", type=int, default=1500)
    parser.add_argument("--min-validation", type=int, default=100)
    parser.add_argument("--max-per-source", type=int, default=2)
    return parser.parse_args()


def validate_export(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    train = read_jsonl(root / "train/train.jsonl")
    validation = read_jsonl(root / "validation/validation.jsonl")
    train_audit = read_jsonl(root / "audit/train_audit.jsonl")
    validation_audit = read_jsonl(root / "audit/validation_audit.jsonl")
    stats = json.loads((root / "04_export.stats.json").read_text(encoding="utf-8"))
    datasets = {"train": train, "validation": validation}
    audits = {"train": train_audit, "validation": validation_audit}
    errors: list[str] = []

    if not stats.get("export_ready"):
        errors.append("stats.export_ready is not true")
    if len(train) < args.min_train:
        errors.append(f"train below minimum: {len(train)}")
    if len(validation) < args.min_validation:
        errors.append(f"validation below minimum: {len(validation)}")

    train_sources = {row["source_id"] for row in train}
    validation_sources = {row["source_id"] for row in validation}
    if train_sources & validation_sources:
        errors.append("train/validation source overlap")

    isolation_failures = 0
    source_counts: Counter[str] = Counter()
    for split in ("train", "validation"):
        rows = datasets[split]
        audit_rows = audits[split]
        pair_ids = [row.get("pair_id") for row in rows]
        audit_ids = [row.get("pair_id") for row in audit_rows]
        if len(pair_ids) != len(set(pair_ids)):
            errors.append(f"{split} duplicate pair_id")
        if pair_ids != audit_ids:
            errors.append(f"{split} data/audit order or ids mismatch")
        for row in rows:
            source_counts[row["source_id"]] += 1
            if row.get("chosen", "").strip() == row.get("rejected", "").strip():
                isolation_failures += 1
                continue
            chosen = extract_first_json_object(row.get("chosen", ""))
            rejected = extract_first_json_object(row.get("rejected", ""))
            if (
                not isinstance(chosen, dict)
                or not isinstance(rejected, dict)
                or not is_isolated_answer_pair(chosen, rejected)
            ):
                isolation_failures += 1
    if isolation_failures:
        errors.append(f"answer-level isolation failures: {isolation_failures}")
    if source_counts and max(source_counts.values()) > args.max_per_source:
        errors.append("max pairs per source exceeded")

    distribution: dict[str, Any] = {}
    for split, rows in audits.items():
        group_counts = Counter(row["pair_type_group"] for row in rows)
        natural = [row for row in rows if row["pair_type_group"] in NATURAL_GROUPS]
        tier_b = sum(row.get("consistency_tier") == "B" for row in natural)
        controlled_fraction = (
            group_counts["controlled_negative"] / len(rows) if rows else 0.0
        )
        tier_b_fraction = tier_b / len(natural) if natural else 0.0
        positions = {}
        for group in sorted(NATURAL_GROUPS):
            values = [row for row in rows if row["pair_type_group"] == group]
            a_fraction = (
                sum(row["original_winner_position"] == "A" for row in values)
                / len(values)
                if values
                else 0.0
            )
            positions[group] = a_fraction
            if not 0.45 <= a_fraction <= 0.55:
                errors.append(f"{split} {group} A-win fraction={a_fraction:.4f}")
        if controlled_fraction > 0.10 + 1e-9:
            errors.append(f"{split} controlled fraction exceeds 10%")
        if tier_b_fraction > 0.20 + 1e-9:
            errors.append(f"{split} Tier B fraction exceeds 20%")
        distribution[split] = {
            "rows": len(rows),
            "pair_type_groups": dict(group_counts),
            "controlled_fraction": controlled_fraction,
            "tier_b_natural_fraction": tier_b_fraction,
            "original_a_win_fraction": positions,
        }

    train_review = sorted(
        train_audit,
        key=lambda row: stable_int(
            "final_export_review", row["pair_id"], seed=args.seed
        ),
    )[: args.train_sample]
    review = [
        {**row, "review_scope": "train_sample"} for row in train_review
    ] + [
        {**row, "review_scope": "all_validation"} for row in validation_audit
    ]
    write_jsonl(root / "audit/final_review_sample.jsonl", review)

    chosen_longer = sum(
        len(data["chosen"]) > len(data["rejected"])
        for split in datasets
        for data in datasets[split]
    )
    total = len(train) + len(validation)
    result = {
        "ready": not errors,
        "errors": errors,
        "counts": {"train": len(train), "validation": len(validation)},
        "distribution": distribution,
        "unique_sources": {
            "train": len(train_sources),
            "validation": len(validation_sources),
        },
        "maximum_pairs_per_source": max(source_counts.values()) if source_counts else 0,
        "answer_level_isolation_failures": isolation_failures,
        "review_sample": {
            "train": len(train_review),
            "validation": len(validation_audit),
            "path": str(root / "audit/final_review_sample.jsonl"),
        },
        "length_diagnostic": {
            "chosen_longer_count": chosen_longer,
            "total": total,
            "chosen_longer_rate": chosen_longer / total if total else 0.0,
        },
    }
    return result


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    result = validate_export(root, args)
    output = root / "04_export.validation.json"
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready"]:
        raise SystemExit("DPO export validation failed")


if __name__ == "__main__":
    main()
