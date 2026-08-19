#!/usr/bin/env python3
"""Validate V2.2 teacher outputs and export audited Evidence-SFT datasets.

The validator deliberately separates deterministic failures from heuristic
quality signals:

* rejected: structurally unsafe and must not be used for training;
* review: structurally recoverable, but needs human review before training;
* accepted: passes hard and review gates; warnings remain available for audits.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from evidence_sft_common import (
    EVIDENCE_IMPORTANCE_LEVELS,
    EVIDENCE_SCHEMA_VERSION,
    SUFFICIENCY_LEVELS,
    TASK_TYPES,
    canonicalize_text,
    extract_first_json_object,
)


NUMBER_RE = re.compile(
    r"(?<![A-Za-z])\d+(?:\.\d+)?(?:%|℃|岁|天|周|月|年|次|mg|g|ml|mmHg)?",
    re.IGNORECASE,
)
EVIDENCE_ID_RE = re.compile(r"E[1-9]\d*")
STRONG_ROLE_RE = re.compile(r"排除|证实|证明|确定|明确不是|因此一定是")
QUERY_LIKE_SPAN_RE = re.compile(
    r"怎么办|怎么治疗|如何治疗|什么病|怎么回事|是否|是不是|请问|需要什么检查|能活多久|[？?]"
)

REQUIRED_OUTPUT_FIELDS = {
    "task_type",
    "query_intent",
    "evidence_sufficiency",
    "evidence",
    "critical_evidence_ids",
    "missing_information",
    "clinical_reasoning",
    "final_answer",
}
REQUIRED_EVIDENCE_FIELDS = {"id", "span", "importance", "role"}
VALID_SPLITS = {"train", "validation", "test"}
LONG_EVIDENCE_SPAN_CHARS = 40
MAX_RECOMMENDED_EVIDENCE_ITEMS = 8
MAX_RECOMMENDED_MISSING_ITEMS = 5


@dataclass(frozen=True)
class AuditResult:
    normalized: dict[str, Any] | None
    errors: list[str]
    review_reasons: list[str]
    warnings: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/evidence_sft/01_teacher_raw.jsonl")
    parser.add_argument("--candidates", default="data/evidence_sft/00_candidates.jsonl")
    parser.add_argument("--output-dir", default="data/evidence_sft/validated_v2_2")
    parser.add_argument("--expected-prompt-version", default=EVIDENCE_SCHEMA_VERSION)
    return parser.parse_args()


def _deduplicate(values: list[str]) -> list[str]:
    return sorted(set(values))


def _issue_code(value: str) -> str:
    """Collapse dynamic issue details into a stable stats key."""
    return value.split(":", 1)[0]


def _string_list(
    value: Any,
    field: str,
    errors: list[str],
    *,
    require_nonempty: bool,
) -> list[str]:
    if not isinstance(value, list):
        errors.append(f"{field}_not_list")
        return []
    if not all(isinstance(item, str) and item.strip() for item in value):
        errors.append(f"{field}_contains_invalid_item")
        return []
    normalized = [item.strip() for item in value]
    if require_nonempty and not normalized:
        errors.append(f"{field}_empty")
    return normalized


def _has_overlapping_spans(evidence: list[dict[str, Any]]) -> bool:
    intervals = sorted((item["start"], item["end"]) for item in evidence)
    return any(
        current_start < previous_end
        for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:])
    )


def audit_output(case_text: str, value: Any, original_answer: str = "") -> AuditResult:
    """Audit one parsed V2.2 output without applying dataset-level rules."""
    errors: list[str] = []
    review_reasons: list[str] = []
    warnings: list[str] = []
    if not isinstance(value, dict):
        return AuditResult(None, ["output_not_object"], [], [])

    missing_fields = sorted(REQUIRED_OUTPUT_FIELDS - set(value))
    unexpected_fields = sorted(set(value) - REQUIRED_OUTPUT_FIELDS)
    errors.extend(f"missing_required_field:{field}" for field in missing_fields)
    if unexpected_fields:
        review_reasons.append("unexpected_output_fields:" + ",".join(unexpected_fields))

    task_type = value.get("task_type")
    sufficiency = value.get("evidence_sufficiency")
    if task_type not in TASK_TYPES:
        errors.append("invalid_task_type")
    if sufficiency not in SUFFICIENCY_LEVELS:
        errors.append("invalid_evidence_sufficiency")

    query_intent = _string_list(
        value.get("query_intent"), "query_intent", errors, require_nonempty=True
    )
    missing_information = _string_list(
        value.get("missing_information"),
        "missing_information",
        errors,
        require_nonempty=False,
    )
    critical_ids = _string_list(
        value.get("critical_evidence_ids"),
        "critical_evidence_ids",
        errors,
        require_nonempty=False,
    )
    if len(query_intent) != len(set(query_intent)):
        review_reasons.append("duplicate_query_intent")
    if len(missing_information) != len(set(missing_information)):
        review_reasons.append("duplicate_missing_information")
    if len(critical_ids) != len(set(critical_ids)):
        errors.append("duplicate_critical_evidence_id")
    if len(missing_information) > MAX_RECOMMENDED_MISSING_ITEMS:
        warnings.append("too_many_missing_information_items")

    clinical_reasoning = value.get("clinical_reasoning")
    final_answer = value.get("final_answer")
    if not isinstance(clinical_reasoning, str) or len(clinical_reasoning.strip()) < 20:
        errors.append("clinical_reasoning_too_short")
    if not isinstance(final_answer, str) or len(final_answer.strip()) < 30:
        errors.append("final_answer_too_short")
    if (
        isinstance(final_answer, str)
        and original_answer.strip()
        and canonicalize_text(final_answer) == canonicalize_text(original_answer)
    ):
        errors.append("final_answer_copies_original")

    evidence_value = value.get("evidence")
    if not isinstance(evidence_value, list):
        errors.append("evidence_not_list")
        evidence_value = []
    if len(evidence_value) > MAX_RECOMMENDED_EVIDENCE_ITEMS:
        warnings.append("too_many_evidence_items")

    normalized_evidence: list[dict[str, Any]] = []
    evidence_ids: list[str] = []
    evidence_spans: list[str] = []
    critical_by_importance: list[str] = []
    for item in evidence_value:
        if not isinstance(item, dict):
            errors.append("evidence_item_not_object")
            continue

        missing_evidence_fields = sorted(REQUIRED_EVIDENCE_FIELDS - set(item))
        unexpected_evidence_fields = sorted(set(item) - REQUIRED_EVIDENCE_FIELDS)
        errors.extend(
            f"missing_evidence_field:{field}" for field in missing_evidence_fields
        )
        if unexpected_evidence_fields:
            review_reasons.append(
                "unexpected_evidence_fields:" + ",".join(unexpected_evidence_fields)
            )

        evidence_id = item.get("id")
        span = item.get("span")
        importance = item.get("importance")
        role = item.get("role")
        item_valid = True

        if not isinstance(evidence_id, str) or not evidence_id.strip():
            errors.append("evidence_id_missing")
            item_valid = False
        else:
            evidence_ids.append(evidence_id)
            if evidence_id != evidence_id.strip() or not EVIDENCE_ID_RE.fullmatch(evidence_id):
                review_reasons.append("noncanonical_evidence_id")

        if not isinstance(span, str) or not span:
            errors.append("evidence_span_missing")
            item_valid = False
        elif span not in case_text:
            errors.append("evidence_span_not_in_case")
            item_valid = False
        else:
            evidence_spans.append(span)
            if case_text.count(span) > 1:
                review_reasons.append("ambiguous_repeated_evidence_span")
            if len(span) == 1:
                review_reasons.append("evidence_span_too_short")
            if len(span) > LONG_EVIDENCE_SPAN_CHARS:
                review_reasons.append("long_evidence_span_requires_atomicity_review")
            if QUERY_LIKE_SPAN_RE.search(span):
                review_reasons.append("query_like_evidence_span")

        if importance not in EVIDENCE_IMPORTANCE_LEVELS:
            errors.append("invalid_evidence_importance")
            item_valid = False
        elif importance == "critical" and isinstance(evidence_id, str):
            critical_by_importance.append(evidence_id)

        if not isinstance(role, str) or len(role.strip()) < 4:
            errors.append("evidence_role_too_short")
            item_valid = False
        elif STRONG_ROLE_RE.search(role):
            warnings.append("strong_role_claim_requires_review")

        if item_valid:
            start = case_text.find(span)
            normalized_evidence.append(
                {
                    "id": evidence_id,
                    "span": span,
                    "importance": importance,
                    "role": role.strip(),
                    "start": start,
                    "end": start + len(span),
                }
            )

    if len(evidence_ids) != len(set(evidence_ids)):
        errors.append("duplicate_evidence_id")
    if len(evidence_spans) != len(set(evidence_spans)):
        errors.append("duplicate_evidence_span")

    expected_ids = [f"E{index}" for index in range(1, len(evidence_ids) + 1)]
    if evidence_ids and evidence_ids != expected_ids:
        review_reasons.append("nonsequential_evidence_ids")

    if normalized_evidence and _has_overlapping_spans(normalized_evidence):
        review_reasons.append("overlapping_evidence_spans")

    if sufficiency == "sufficient" and not evidence_value:
        errors.append("sufficient_without_evidence")
    if sufficiency == "partial" and not evidence_value:
        errors.append("partial_without_evidence")
    if sufficiency == "conflicting" and len(evidence_value) < 2:
        errors.append("conflicting_without_multiple_evidence")
    if sufficiency in {"partial", "insufficient"} and not missing_information:
        review_reasons.append(f"{sufficiency}_without_missing_information")

    unknown_critical = sorted(set(critical_ids) - set(evidence_ids))
    if unknown_critical:
        errors.append("critical_evidence_id_not_found")
    if set(critical_ids) != set(critical_by_importance):
        errors.append("critical_importance_mismatch")

    if isinstance(clinical_reasoning, str) and isinstance(final_answer, str):
        case_numbers = set(NUMBER_RE.findall(case_text))
        generated_numbers = set(NUMBER_RE.findall(clinical_reasoning + "\n" + final_answer))
        unsupported_numbers = sorted(generated_numbers - case_numbers)
        if unsupported_numbers:
            warnings.append(
                "generated_numbers_require_review:" + ",".join(unsupported_numbers[:8])
            )

    errors = _deduplicate(errors)
    review_reasons = _deduplicate(review_reasons)
    warnings = _deduplicate(warnings)
    if errors:
        return AuditResult(None, errors, review_reasons, warnings)

    normalized = {
        "task_type": task_type,
        "query_intent": query_intent,
        "evidence_sufficiency": sufficiency,
        "evidence": normalized_evidence,
        "critical_evidence_ids": critical_ids,
        "missing_information": missing_information,
        "clinical_reasoning": clinical_reasoning.strip(),
        "final_answer": final_answer.strip(),
    }
    return AuditResult(normalized, [], review_reasons, warnings)


def validate_output(
    case_text: str, value: Any
) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    """Backward-compatible validation API used by earlier callers.

    Review reasons are surfaced as warnings here. New code should call
    :func:`audit_output` to keep all three severity levels separate.
    """
    result = audit_output(case_text, value)
    return result.normalized, result.errors, result.review_reasons + result.warnings


def build_training_prompt(case_text: str) -> str:
    return (
        "以下是患者或用户提供的病例描述。请仅依据病例原文输出一个合法 JSON 对象，依次给出"
        "query_intent、evidence_sufficiency、原子化 evidence（含 importance 和 role）、"
        "critical_evidence_ids、missing_information、clinical_reasoning 和 final_answer。"
        "不得虚构病例信息。\n\n病例描述：\n" + case_text
    )


def _iter_jsonl_tolerant(
    path: Path,
) -> Iterator[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                value = json.loads(raw_line)
            except json.JSONDecodeError:
                yield line_number, None, raw_line
                continue
            if not isinstance(value, dict):
                yield line_number, None, raw_line
                continue
            yield line_number, value, None


def _load_candidates(path: Path) -> tuple[dict[str, dict[str, Any]], int]:
    candidates: dict[str, dict[str, Any]] = {}
    row_count = 0
    for line_number, record, raw_line in _iter_jsonl_tolerant(path):
        row_count += 1
        if record is None:
            raise ValueError(f"{path}:{line_number} is not a valid JSON object: {raw_line!r}")
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"{path}:{line_number} has no valid source_id")
        if source_id in candidates:
            raise ValueError(f"{path}:{line_number} duplicates source_id {source_id}")
        candidates[source_id] = record
    return candidates, row_count


def _write_jsonl(handle: Any, value: dict[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _base_audited_record(
    record: dict[str, Any],
    normalized: dict[str, Any],
    status: str,
    review_reasons: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "source_id": record.get("source_id", ""),
        "split": record.get("split", ""),
        "category": record.get("category", "未标注医学主题"),
        "case_text": record.get("case_text", ""),
        "original_answer": record.get("original_answer", ""),
        "teacher_model": record.get("teacher_model", ""),
        "prompt_version": record.get("prompt_version", ""),
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "validation_status": status,
        "target": normalized,
        "validation_review_reasons": review_reasons,
        "validation_warnings": warnings,
    }


def validate_dataset(
    input_path: str | Path,
    candidates_path: str | Path,
    output_dir: str | Path,
    expected_prompt_version: str = EVIDENCE_SCHEMA_VERSION,
) -> dict[str, Any]:
    input_path = Path(input_path)
    candidates_path = Path(candidates_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates, candidate_rows = _load_candidates(candidates_path)
    input_rows = list(_iter_jsonl_tolerant(input_path))
    source_id_counts = Counter(
        record.get("source_id")
        for _, record, _ in input_rows
        if record is not None and isinstance(record.get("source_id"), str)
    )

    accepted_path = output_dir / "03_validated_full.jsonl"
    rejected_path = output_dir / "03_rejected.jsonl"
    review_path = output_dir / "03_review.jsonl"
    split_paths = {
        split: output_dir / f"{split}.jsonl" for split in sorted(VALID_SPLITS)
    }

    counts: Counter[str] = Counter()
    reject_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    split_status_counts: Counter[str] = Counter()
    sufficiency_counts: Counter[str] = Counter()
    task_type_counts: Counter[str] = Counter()
    importance_counts: Counter[str] = Counter()
    model_counts: Counter[str] = Counter()
    prompt_version_counts: Counter[str] = Counter()
    seen_input_ids: set[str] = set()

    with ExitStack() as stack:
        accepted_handle = stack.enter_context(accepted_path.open("w", encoding="utf-8"))
        rejected_handle = stack.enter_context(rejected_path.open("w", encoding="utf-8"))
        review_handle = stack.enter_context(review_path.open("w", encoding="utf-8"))
        split_handles = {
            split: stack.enter_context(path.open("w", encoding="utf-8"))
            for split, path in split_paths.items()
        }

        for line_number, record, malformed_line in input_rows:
            counts["total"] += 1
            if record is None:
                counts["rejected"] += 1
                reject_counts["input_not_json_object"] += 1
                _write_jsonl(
                    rejected_handle,
                    {
                        "input_line": line_number,
                        "raw_line": malformed_line,
                        "validation_errors": ["input_not_json_object"],
                        "validation_review_reasons": [],
                        "validation_warnings": [],
                    },
                )
                continue

            errors: list[str] = []
            review_reasons: list[str] = []
            warnings: list[str] = []
            source_id = record.get("source_id")
            case_text = record.get("case_text")
            split = record.get("split")
            candidate = candidates.get(source_id) if isinstance(source_id, str) else None

            if not isinstance(source_id, str) or not source_id:
                errors.append("source_id_missing")
            else:
                seen_input_ids.add(source_id)
                if source_id_counts[source_id] > 1:
                    errors.append("duplicate_source_id")
            if not isinstance(case_text, str) or not case_text.strip():
                errors.append("case_text_missing")
                case_text = ""
            if split not in VALID_SPLITS:
                errors.append("invalid_split")
            if record.get("status") != "ok":
                errors.append("teacher_record_status_not_ok")
            if not isinstance(record.get("teacher_model"), str) or not record.get("teacher_model"):
                errors.append("teacher_model_missing")
            if record.get("prompt_version") != expected_prompt_version:
                errors.append("unexpected_prompt_version")

            if candidate is None:
                errors.append("source_id_not_in_candidates")
            else:
                for field in ("case_text", "original_answer", "split", "task_type_hint"):
                    if record.get(field) != candidate.get(field):
                        errors.append(f"candidate_{field}_mismatch")
                if record.get("category") != candidate.get("category"):
                    warnings.append("candidate_category_mismatch")

            parsed_output = record.get("parsed_output")
            teacher_text = record.get("teacher_text")
            parsed_from_text = (
                extract_first_json_object(teacher_text) if isinstance(teacher_text, str) else None
            )
            if not isinstance(parsed_output, dict):
                if parsed_from_text is None:
                    errors.append("teacher_output_not_parseable")
                else:
                    parsed_output = parsed_from_text
                    warnings.append("parsed_output_recovered_from_teacher_text")
            elif parsed_from_text is None:
                review_reasons.append("teacher_text_not_parseable")
            elif parsed_from_text != parsed_output:
                errors.append("parsed_output_differs_from_teacher_text")

            audit = audit_output(
                case_text,
                parsed_output,
                str(record.get("original_answer", "")),
            )
            errors.extend(audit.errors)
            review_reasons.extend(audit.review_reasons)
            warnings.extend(audit.warnings)
            if (
                isinstance(parsed_output, dict)
                and record.get("task_type_hint")
                and parsed_output.get("task_type") != record.get("task_type_hint")
            ):
                warnings.append("task_type_changed_by_teacher")

            errors = _deduplicate(errors)
            review_reasons = _deduplicate(review_reasons)
            warnings = _deduplicate(warnings)
            for warning in warnings:
                warning_counts[_issue_code(warning)] += 1
            model_counts[str(record.get("teacher_model", ""))] += 1
            prompt_version_counts[str(record.get("prompt_version", ""))] += 1

            if isinstance(parsed_output, dict):
                task_type_counts[str(parsed_output.get("task_type"))] += 1
                sufficiency_counts[str(parsed_output.get("evidence_sufficiency"))] += 1
                evidence = parsed_output.get("evidence")
                if isinstance(evidence, list):
                    for item in evidence:
                        if isinstance(item, dict):
                            importance_counts[str(item.get("importance"))] += 1

            if errors or audit.normalized is None:
                counts["rejected"] += 1
                if split in VALID_SPLITS:
                    split_status_counts[f"{split}:rejected"] += 1
                for error in errors:
                    reject_counts[_issue_code(error)] += 1
                rejected = copy.deepcopy(record)
                rejected["input_line"] = line_number
                rejected["validation_errors"] = errors
                rejected["validation_review_reasons"] = review_reasons
                rejected["validation_warnings"] = warnings
                _write_jsonl(rejected_handle, rejected)
                continue

            audited_record = _base_audited_record(
                record,
                audit.normalized,
                "review" if review_reasons else "accepted",
                review_reasons,
                warnings,
            )
            if review_reasons:
                counts["review"] += 1
                split_status_counts[f"{split}:review"] += 1
                for reason in review_reasons:
                    review_counts[_issue_code(reason)] += 1
                _write_jsonl(review_handle, audited_record)
                continue

            counts["accepted"] += 1
            split_status_counts[f"{split}:accepted"] += 1
            _write_jsonl(accepted_handle, audited_record)

            assistant_target = copy.deepcopy(audit.normalized)
            for item in assistant_target["evidence"]:
                item.pop("start", None)
                item.pop("end", None)
            sft_record = {
                "source_id": source_id,
                "split": split,
                "task_type": audit.normalized["task_type"],
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "masked": False,
                "conversations": [
                    {"from": "human", "value": build_training_prompt(case_text)},
                    {
                        "from": "gpt",
                        "value": json.dumps(assistant_target, ensure_ascii=False),
                    },
                ],
            }
            _write_jsonl(split_handles[split], sft_record)

    total = counts["total"]
    report = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "input": str(input_path),
        "candidates": str(candidates_path),
        "output_dir": str(output_dir),
        "outputs": {
            "accepted": str(accepted_path),
            "review": str(review_path),
            "rejected": str(rejected_path),
            **{split: str(path) for split, path in split_paths.items()},
        },
        "counts": {
            "total": total,
            "accepted": counts["accepted"],
            "review": counts["review"],
            "rejected": counts["rejected"],
            "accepted_rate": round(counts["accepted"] / total, 6) if total else 0.0,
            "review_rate": round(counts["review"] / total, 6) if total else 0.0,
            "rejected_rate": round(counts["rejected"] / total, 6) if total else 0.0,
        },
        "integrity": {
            "candidate_rows": candidate_rows,
            "candidate_unique_ids": len(candidates),
            "input_unique_source_ids": len(seen_input_ids),
            "candidate_ids_missing_from_input": len(set(candidates) - seen_input_ids),
            "input_ids_missing_from_candidates": len(seen_input_ids - set(candidates)),
            "duplicate_source_ids": sum(count > 1 for count in source_id_counts.values()),
        },
        "split_status_counts": dict(sorted(split_status_counts.items())),
        "reject_reason_counts": dict(reject_counts.most_common()),
        "review_reason_counts": dict(review_counts.most_common()),
        "warning_counts": dict(warning_counts.most_common()),
        "distributions": {
            "task_type": dict(task_type_counts.most_common()),
            "evidence_sufficiency": dict(sufficiency_counts.most_common()),
            "evidence_importance": dict(importance_counts.most_common()),
            "teacher_model": dict(model_counts.most_common()),
            "prompt_version": dict(prompt_version_counts.most_common()),
        },
        "thresholds": {
            "long_evidence_span_chars": LONG_EVIDENCE_SPAN_CHARS,
            "max_recommended_evidence_items": MAX_RECOMMENDED_EVIDENCE_ITEMS,
            "max_recommended_missing_information_items": MAX_RECOMMENDED_MISSING_ITEMS,
        },
    }
    report_path = output_dir / "03_validation.stats.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    args = parse_args()
    report = validate_dataset(
        args.input,
        args.candidates,
        args.output_dir,
        args.expected_prompt_version,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
