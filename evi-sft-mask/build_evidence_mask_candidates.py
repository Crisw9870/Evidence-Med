#!/usr/bin/env python3
"""Select deterministic Evidence Mask candidates without exposing hidden facts to a teacher."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evidence_mask_common import (
    choose_random_span,
    choose_supporting,
    delete_spans,
    evidence_by_importance,
    mask_eligible,
    read_jsonl,
    stable_rank,
    stratified_sample,
    text_sha256,
    write_jsonl,
)


MASK_PROMPT_VERSION = "evidence-mask-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="data/evidence_sft/03_validated_full.jsonl",
    )
    parser.add_argument("--output-dir", default="data/evidence_mask/v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-candidates", type=int, default=2300)
    parser.add_argument("--validation-candidates", type=int, default=240)
    parser.add_argument(
        "--test-candidates",
        "--test-pairs",
        dest="test_candidates",
        type=int,
        default=250,
        help="Oversampled test parents; validation later locks the final 200 pairs.",
    )
    parser.add_argument("--no-random-control", action="store_true")
    return parser.parse_args()


def parent_view(record: dict[str, Any]) -> dict[str, Any]:
    target = record["target"]
    return {
        "source_id": record["source_id"],
        "split": record["split"],
        "category": record.get("category", "未标注医学主题"),
        "task_type": target["task_type"],
        "evidence_sufficiency": target["evidence_sufficiency"],
        "case_text": record["case_text"],
        "target": target,
    }


def build_variant(
    parent: dict[str, Any],
    mask_type: str,
    removed: list[dict[str, Any]],
) -> dict[str, Any]:
    variant_code = {
        "critical": "C",
        "supporting": "S",
        "random": "R",
    }[mask_type]
    removed_ids = "+".join(str(item["id"]) for item in removed)
    variant_id = f"{parent['source_id']}::{variant_code}:{removed_ids}"
    masked_case = delete_spans(parent["case_text"], removed)
    if not masked_case:
        raise ValueError(f"{variant_id} produced an empty case")
    for item in removed:
        if str(item["span"]) in masked_case:
            raise ValueError(f"{variant_id} still contains removed span {item['span']!r}")
    return {
        "source_id": variant_id,
        "variant_id": variant_id,
        "pair_id": parent["source_id"],
        "parent_source_id": parent["source_id"],
        "split": parent["split"],
        "category": parent["category"],
        "task_type_hint": parent["task_type"],
        "original_evidence_sufficiency": parent["evidence_sufficiency"],
        "mask_type": mask_type,
        "masked_evidence_ids": [str(item["id"]) for item in removed],
        "removed_spans": [
            {
                "id": str(item["id"]),
                "span": str(item["span"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
            }
            for item in removed
        ],
        "case_text": masked_case,
        "original_case_text": parent["case_text"],
        "original_case_text_sha256": text_sha256(parent["case_text"]),
        "original_target": parent["target"],
        "prompt_version": MASK_PROMPT_VERSION,
    }


def build_candidate_rows(
    accepted_rows: list[dict[str, Any]],
    *,
    seed: int,
    train_candidates: int,
    validation_candidates: int,
    test_pairs: int,
    include_random: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [parent_view(row) for row in accepted_rows if mask_eligible(row)]
    by_split = {
        split: [row for row in eligible if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    requested = {
        "train": train_candidates,
        "validation": validation_candidates,
        "test": test_pairs,
    }
    for split, count in requested.items():
        if count > len(by_split[split]):
            raise ValueError(
                f"requested {count} {split} parents, only {len(by_split[split])} eligible"
            )

    selected = {
        split: stratified_sample(rows, requested[split], seed)
        for split, rows in by_split.items()
    }
    variants: list[dict[str, Any]] = []
    random_missing = 0
    for split, parents in selected.items():
        for parent in parents:
            target = parent["target"]
            critical = evidence_by_importance(target, "critical")
            supporting = evidence_by_importance(target, "supporting")
            variants.append(build_variant(parent, "critical", critical))
            if split != "test":
                continue
            chosen_supporting = choose_supporting(critical[0], supporting, seed, parent["source_id"])
            variants.append(build_variant(parent, "supporting", [chosen_supporting]))
            if include_random:
                random_span = choose_random_span(
                    parent["case_text"],
                    target["evidence"],
                    len(str(critical[0]["span"])),
                    seed,
                    parent["source_id"],
                )
                if random_span is None:
                    random_missing += 1
                else:
                    variants.append(build_variant(parent, "random", [random_span]))

    variants.sort(key=lambda row: (row["split"], stable_rank(row["variant_id"], seed)))
    counts = Counter((row["split"], row["mask_type"]) for row in variants)
    stats = {
        "prompt_version": MASK_PROMPT_VERSION,
        "seed": seed,
        "input_rows": len(accepted_rows),
        "eligible_single_critical_with_supporting": {
            split: len(rows) for split, rows in by_split.items()
        },
        "selected_parents": {split: len(rows) for split, rows in selected.items()},
        "variant_counts": {
            f"{split}:{mask_type}": count
            for (split, mask_type), count in sorted(counts.items())
        },
        "random_control_missing": random_missing,
    }
    return variants, stats


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    variants, stats = build_candidate_rows(
        rows,
        seed=args.seed,
        train_candidates=args.train_candidates,
        validation_candidates=args.validation_candidates,
        test_pairs=args.test_candidates,
        include_random=not args.no_random_control,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "00_candidates.jsonl"
    stats_path = output_dir / "00_candidates.stats.json"
    write_jsonl(output_path, variants)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Candidates: {output_path}")


if __name__ == "__main__":
    main()
