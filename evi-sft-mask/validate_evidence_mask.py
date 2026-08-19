#!/usr/bin/env python3
"""Validate Evidence Mask targets and export equal-budget M0/M1 manifests."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from evidence_mask_common import read_jsonl, stable_rank, without_offsets, write_jsonl
from evidence_sft_common import EVIDENCE_SCHEMA_VERSION, extract_first_json_object
from judge_evidence_mask_targets import valid_judgment
from validate_evidence_sft import audit_output, build_training_prompt


MASK_PROMPT_VERSION = "evidence-mask-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="data/evidence_mask/v1/00_candidates.jsonl")
    parser.add_argument("--teacher", default="data/evidence_mask/v1/01_teacher_raw.jsonl")
    parser.add_argument("--judgments", default="data/evidence_mask/v1/02_judgments.jsonl")
    parser.add_argument("--output-dir", default="data/evidence_mask/v1/validated")
    parser.add_argument(
        "--original-train",
        default="data/evidence_sft/validated_v2_2/train.jsonl",
    )
    parser.add_argument(
        "--original-validation",
        default="data/evidence_sft/validated_v2_2/validation.jsonl",
    )
    parser.add_argument(
        "--original-test",
        default="data/evidence_sft/validated_v2_2/test.jsonl",
    )
    parser.add_argument("--train-replacements", type=int, default=1918)
    parser.add_argument("--validation-replacements", type=int, default=200)
    parser.add_argument("--test-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-judgments", action="store_true")
    return parser.parse_args()


def index_unique(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row has no valid {field}")
        if value in index:
            raise ValueError(f"{label} duplicates {field}={value}")
        index[value] = row
    return index


def issue_code(value: str) -> str:
    return value.split(":", 1)[0]


def target_evidence_contains_removed_span(
    target: dict[str, Any], removed_spans: list[dict[str, Any]]
) -> bool:
    removed = {str(item["span"]) for item in removed_spans}
    evidence = target.get("evidence")
    return isinstance(evidence, list) and any(
        isinstance(item, dict) and str(item.get("span", "")) in removed
        for item in evidence
    )


def target_text_mentions_removed_span(
    target: dict[str, Any], removed_spans: list[dict[str, Any]]
) -> bool:
    narrative = "\n".join(
        str(target.get(field, ""))
        for field in ("clinical_reasoning", "final_answer")
    )
    return any(str(item["span"]) in narrative for item in removed_spans)


def to_mask_sft_row(
    candidate: dict[str, Any],
    normalized_target: dict[str, Any],
    judgment: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "source_id": candidate["parent_source_id"],
        "variant_id": candidate["variant_id"],
        "pair_id": candidate["pair_id"],
        "parent_source_id": candidate["parent_source_id"],
        "split": candidate["split"],
        "task_type": normalized_target["task_type"],
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "masked": True,
        "mask_type": candidate["mask_type"],
        "masked_evidence_ids": candidate["masked_evidence_ids"],
        "removed_spans": candidate["removed_spans"],
        "mask_assessment": judgment,
        "conversations": [
            {"from": "human", "value": build_training_prompt(candidate["case_text"])},
            {
                "from": "gpt",
                "value": json.dumps(without_offsets(normalized_target), ensure_ascii=False),
            },
        ],
    }


def replace_manifest_rows(
    originals: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    replacement_by_parent = index_unique(replacements, "parent_source_id", "replacement")
    original_ids = {row.get("source_id") for row in originals}
    missing = set(replacement_by_parent) - original_ids
    if missing:
        raise ValueError(f"replacement parents missing from original manifest: {sorted(missing)[:5]}")
    result = [replacement_by_parent.get(str(row.get("source_id")), row) for row in originals]
    if len(result) != len(originals):
        raise AssertionError("replacement changed manifest length")
    if {row.get("source_id") for row in result} != original_ids:
        raise AssertionError("replacement changed manifest parent source IDs")
    return result


def build_test_pairs(
    accepted: list[dict[str, Any]],
    original_test: list[dict[str, Any]],
    requested_pairs: int,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    original_by_id = index_unique(original_test, "source_id", "original test")
    by_parent: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in accepted:
        if row["split"] == "test":
            by_parent[row["parent_source_id"]][row["mask_type"]] = row
    complete = [
        parent
        for parent, variants in by_parent.items()
        if {"critical", "supporting"}.issubset(variants) and parent in original_by_id
    ]
    complete.sort(key=lambda parent: stable_rank(parent, seed))
    if len(complete) < requested_pairs:
        raise ValueError(
            f"need {requested_pairs} complete test pairs, only {len(complete)} passed validation"
        )

    output: list[dict[str, Any]] = []
    random_count = 0
    for parent in complete[:requested_pairs]:
        original = copy.deepcopy(original_by_id[parent])
        original.update(
            {
                "variant_id": f"{parent}::U",
                "pair_id": parent,
                "parent_source_id": parent,
                "masked": False,
                "mask_type": "unmasked",
                "masked_evidence_ids": [],
                "removed_spans": [],
                "mask_assessment": None,
            }
        )
        output.append(original)
        variants = by_parent[parent]
        output.append(variants["critical"])
        output.append(variants["supporting"])
        if "random" in variants:
            output.append(variants["random"])
            random_count += 1
    return output, random_count


def validate_mask_dataset(
    candidates_path: str | Path,
    teacher_path: str | Path,
    judgments_path: str | Path | None,
    output_dir: str | Path,
    original_train_path: str | Path,
    original_validation_path: str | Path,
    original_test_path: str | Path,
    *,
    train_replacements: int,
    validation_replacements: int,
    test_pairs: int,
    seed: int,
    require_judgments: bool = True,
) -> dict[str, Any]:
    candidates = index_unique(read_jsonl(candidates_path), "source_id", "candidates")
    teachers = index_unique(read_jsonl(teacher_path), "source_id", "teacher")
    judgments: dict[str, dict[str, Any]] = {}
    if judgments_path is not None and Path(judgments_path).exists():
        judgments = index_unique(read_jsonl(judgments_path), "source_id", "judgments")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    accepted: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()

    for source_id, candidate in candidates.items():
        errors: list[str] = []
        review_reasons: list[str] = []
        warnings: list[str] = []
        teacher = teachers.get(source_id)
        judgment_record = judgments.get(source_id)
        parsed_judgment: dict[str, Any] | None = None

        if teacher is None:
            errors.append("teacher_record_missing")
            parsed_output: Any = None
        else:
            for field in (
                "variant_id",
                "pair_id",
                "parent_source_id",
                "split",
                "mask_type",
                "case_text",
            ):
                if teacher.get(field) != candidate.get(field):
                    errors.append(f"teacher_{field}_mismatch")
            if teacher.get("status") != "ok":
                errors.append("teacher_status_not_ok")
            if teacher.get("prompt_version") != MASK_PROMPT_VERSION:
                errors.append("unexpected_prompt_version")
            parsed_output = teacher.get("parsed_output")
            parsed_from_text = extract_first_json_object(str(teacher.get("teacher_text", "")))
            if not isinstance(parsed_output, dict):
                if parsed_from_text is None:
                    errors.append("teacher_output_not_parseable")
                else:
                    parsed_output = parsed_from_text
                    warnings.append("parsed_output_recovered_from_teacher_text")
            elif parsed_from_text is not None and parsed_from_text != parsed_output:
                errors.append("parsed_output_differs_from_teacher_text")

        if candidate.get("split") not in {"train", "validation", "test"}:
            errors.append("invalid_parent_split")
        if candidate.get("source_id") != candidate.get("variant_id"):
            errors.append("variant_source_id_mismatch")
        if candidate.get("pair_id") != candidate.get("parent_source_id"):
            errors.append("pair_parent_mismatch")
        case_text = candidate.get("case_text")
        removed_spans = candidate.get("removed_spans")
        if not isinstance(case_text, str) or not case_text:
            errors.append("masked_case_missing")
            case_text = ""
        if not isinstance(removed_spans, list) or not removed_spans:
            errors.append("removed_spans_missing")
            removed_spans = []
        for item in removed_spans:
            if isinstance(item, dict) and str(item.get("span", "")) in case_text:
                errors.append("removed_span_still_in_masked_case")

        audit = audit_output(case_text, parsed_output)
        errors.extend(audit.errors)
        review_reasons.extend(audit.review_reasons)
        warnings.extend(audit.warnings)
        if isinstance(parsed_output, dict):
            if target_evidence_contains_removed_span(parsed_output, removed_spans):
                errors.append("removed_span_reintroduced_as_evidence")
            if target_text_mentions_removed_span(parsed_output, removed_spans):
                warnings.append("removed_span_mentioned_in_narrative_requires_judge")

        if judgment_record is None:
            if require_judgments:
                errors.append("judgment_missing")
            else:
                warnings.append("judgment_not_run")
        else:
            value = judgment_record.get("parsed_judgment")
            if judgment_record.get("status") != "ok" or not valid_judgment(value):
                errors.append("invalid_judgment")
            else:
                parsed_judgment = value
                if value["decision"] == "reject":
                    errors.append("judge_rejected")
                elif value["decision"] == "review":
                    review_reasons.append("judge_requires_review")

        errors = sorted(set(errors))
        review_reasons = sorted(set(review_reasons))
        warnings = sorted(set(warnings))
        base_record = {
            **candidate,
            "teacher_model": teacher.get("teacher_model", "") if teacher else "",
            "judge_model": judgment_record.get("judge_model", "") if judgment_record else "",
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "target": audit.normalized,
            "mask_assessment": parsed_judgment,
            "validation_errors": errors,
            "validation_review_reasons": review_reasons,
            "validation_warnings": warnings,
        }
        for warning in warnings:
            warning_counts[issue_code(warning)] += 1
        if errors or audit.normalized is None:
            base_record["validation_status"] = "rejected"
            rejected.append(base_record)
            for error in errors:
                reason_counts[issue_code(error)] += 1
            continue
        if review_reasons:
            base_record["validation_status"] = "review"
            review.append(base_record)
            for reason in review_reasons:
                reason_counts[issue_code(reason)] += 1
            continue
        base_record["validation_status"] = "accepted"
        sft_row = to_mask_sft_row(candidate, audit.normalized, parsed_judgment)
        base_record["sft_record"] = sft_row
        accepted.append(base_record)

    write_jsonl(output_path / "03_validated_full.jsonl", accepted)
    write_jsonl(output_path / "03_review.jsonl", review)
    write_jsonl(output_path / "03_rejected.jsonl", rejected)

    accepted_sft = [row["sft_record"] for row in accepted]
    for split in ("train", "validation", "test"):
        write_jsonl(
            output_path / f"masked_{split}.jsonl",
            [row for row in accepted_sft if row["split"] == split],
        )

    original_train = read_jsonl(original_train_path)
    original_validation = read_jsonl(original_validation_path)
    original_test = read_jsonl(original_test_path)
    critical_train = [
        row for row in accepted_sft if row["split"] == "train" and row["mask_type"] == "critical"
    ]
    critical_validation = [
        row
        for row in accepted_sft
        if row["split"] == "validation" and row["mask_type"] == "critical"
    ]
    critical_train.sort(key=lambda row: stable_rank(row["variant_id"], seed))
    critical_validation.sort(key=lambda row: stable_rank(row["variant_id"], seed))
    if len(critical_train) < train_replacements:
        raise ValueError(
            f"need {train_replacements} accepted train masks, only {len(critical_train)} available"
        )
    if len(critical_validation) < validation_replacements:
        raise ValueError(
            f"need {validation_replacements} accepted validation masks, only {len(critical_validation)} available"
        )
    selected_train = critical_train[:train_replacements]
    selected_validation = critical_validation[:validation_replacements]

    m0_dir = output_path / "m0"
    m1_dir = output_path / "m1"
    write_jsonl(m0_dir / "train.jsonl", original_train)
    write_jsonl(m0_dir / "validation.jsonl", original_validation)
    m1_train = replace_manifest_rows(original_train, selected_train)
    m1_validation = replace_manifest_rows(original_validation, selected_validation)
    write_jsonl(m1_dir / "train.jsonl", m1_train)
    write_jsonl(m1_dir / "validation.jsonl", m1_validation)

    test_pair_rows, random_test_pairs = build_test_pairs(
        accepted_sft, original_test, test_pairs, seed
    )
    write_jsonl(output_path / "test_pairs.jsonl", test_pair_rows)

    original_train_ids = {row["source_id"] for row in original_train}
    original_validation_ids = {row["source_id"] for row in original_validation}
    original_test_ids = {row["source_id"] for row in original_test}
    if original_train_ids & original_validation_ids or original_train_ids & original_test_ids or original_validation_ids & original_test_ids:
        raise ValueError("original parent splits overlap")

    stats = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "prompt_version": MASK_PROMPT_VERSION,
        "seed": seed,
        "counts": {
            "candidates": len(candidates),
            "teacher_records": len(teachers),
            "judgments": len(judgments),
            "accepted": len(accepted),
            "review": len(review),
            "rejected": len(rejected),
            "m0_train": len(original_train),
            "m1_train": len(m1_train),
            "m1_train_replacements": len(selected_train),
            "m0_validation": len(original_validation),
            "m1_validation": len(m1_validation),
            "m1_validation_replacements": len(selected_validation),
            "test_pairs": test_pairs,
            "test_pair_rows": len(test_pair_rows),
            "test_random_controls": random_test_pairs,
        },
        "reason_counts": dict(reason_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "integrity": {
            "m0_m1_train_parent_ids_equal": {row["source_id"] for row in original_train}
            == {row["source_id"] for row in m1_train},
            "m0_m1_validation_parent_ids_equal": {
                row["source_id"] for row in original_validation
            }
            == {row["source_id"] for row in m1_validation},
            "parent_split_intersections": 0,
        },
    }
    (output_path / "03_validation.stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def main() -> None:
    args = parse_args()
    stats = validate_mask_dataset(
        args.candidates,
        args.teacher,
        args.judgments,
        args.output_dir,
        args.original_train,
        args.original_validation,
        args.original_test,
        train_replacements=args.train_replacements,
        validation_replacements=args.validation_replacements,
        test_pairs=args.test_pairs,
        seed=args.seed,
        require_judgments=not args.allow_missing_judgments,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
