#!/usr/bin/env python3
"""Shared schemas, deterministic utilities, and audits for the DPO pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
PROCESS_DIR = ROOT / "process_sft"
if str(PROCESS_DIR) not in sys.path:
    sys.path.insert(0, str(PROCESS_DIR))

from evidence_sft_common import extract_first_json_object  # noqa: E402
from validate_evidence_sft import (  # noqa: E402
    audit_output,
    build_training_prompt,
    classify_generated_number_risk,
    classify_strong_role_claim_risk,
)


DPO_SCHEMA_VERSION = "evidence-dpo-answer-v1"
PAIR_TYPE_GROUPS = {
    "target_vs_model": "target_vs_model",
    "model_vs_model": "model_vs_model",
    "controlled_negative": "controlled_negative",
    "target_vs_new": "target_vs_model",
    "best_existing_vs_new": "model_vs_model",
}
PAIR_TYPES = set(PAIR_TYPE_GROUPS)
NATURAL_PAIR_TYPES = {
    pair_type
    for pair_type, group in PAIR_TYPE_GROUPS.items()
    if group != "controlled_negative"
}
JUDGE_DECISIONS = {"A_better", "B_better", "tie", "both_bad", "unjudgeable"}
FROZEN_RESPONSE_FIELDS = (
    "task_type",
    "query_intent",
    "evidence_sufficiency",
    "evidence",
    "critical_evidence_ids",
)
ANSWER_LEVEL_FIELDS = ("missing_information", "clinical_reasoning", "final_answer")
HIGH_RISK_WARNING_PREFIXES = (
    "task_type_changed_by_teacher",
    "strong_role_claim_requires_review",
    "generated_numbers_require_review",
)
WARNING_RISK_ORDER = {"clean": 0, "low": 1, "medium": 2, "high": 3}
SCORE_LIMITS = {
    "medical_correctness": 3,
    "evidence_faithfulness": 3,
    "answer_completeness": 3,
    "calibration": 3,
    "missing_information": 2,
    "actionability_safety": 2,
    "expression": 1,
}
HARD_FAILURE_CODES = {"H1", "H2", "H3", "H4", "H5", "H6"}


def pair_type_group(pair_type: str) -> str | None:
    return PAIR_TYPE_GROUPS.get(pair_type)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_environment_file(path: str | Path = ".teacher_env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def stable_int(*parts: Any, seed: int = 42) -> int:
    payload = ":".join(str(part) for part in (seed, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def stable_id(prefix: str, *parts: Any, seed: int = 42) -> str:
    value = stable_int(*parts, seed=seed)
    return f"{prefix}_{value:016x}"


def has_high_risk_warning(warnings: list[Any]) -> bool:
    return any(
        isinstance(warning, str)
        and any(warning.startswith(prefix) for prefix in HIGH_RISK_WARNING_PREFIXES)
        for warning in warnings
    )


def assess_target_warning_risk(
    case_text: str, target: dict[str, Any], warnings: list[Any]
) -> dict[str, Any]:
    """Reassess legacy warning flags using target content and evidence context."""
    warning_codes = {
        warning.split(":", 1)[0]
        for warning in warnings
        if isinstance(warning, str)
    }
    levels: list[str] = []
    reasons: list[str] = []
    strong_evidence_ids: list[str] = []
    medium_strong_evidence_ids: list[str] = []

    if "task_type_changed_by_teacher" in warning_codes:
        levels.append("low")
        reasons.append("teacher_task_type_correction_audit_only")

    evidence = target.get("evidence", [])
    if isinstance(evidence, list):
        for item in evidence:
            if not isinstance(item, dict):
                continue
            span, role = item.get("span"), item.get("role")
            if isinstance(span, str) and isinstance(role, str):
                role_risk = classify_strong_role_claim_risk(span, role)
                if role_risk == "high":
                    strong_evidence_ids.append(str(item.get("id", "unknown")))
                elif role_risk == "medium":
                    medium_strong_evidence_ids.append(
                        str(item.get("id", "unknown"))
                    )
    if strong_evidence_ids:
        levels.append("high")
        reasons.append(
            "unsupported_unhedged_strong_role:" + ",".join(strong_evidence_ids)
        )
    if medium_strong_evidence_ids:
        levels.append("medium")
        reasons.append(
            "context_supported_strong_role_requires_review:"
            + ",".join(medium_strong_evidence_ids)
        )
    elif (
        not strong_evidence_ids
        and "strong_role_claim_requires_review" in warning_codes
    ):
        levels.append("low")
        reasons.append("strong_role_warning_resolved_by_context")

    clinical_reasoning = target.get("clinical_reasoning", "")
    final_answer = target.get("final_answer", "")
    number_risk = {"high": [], "medium": []}
    if isinstance(clinical_reasoning, str) and isinstance(final_answer, str):
        number_risk = classify_generated_number_risk(
            case_text, clinical_reasoning, final_answer
        )
    if number_risk["high"]:
        levels.append("high")
        reasons.append(
            "unsupported_medical_parameter:" + ",".join(number_risk["high"][:8])
        )
    if number_risk["medium"]:
        levels.append("medium")
        reasons.append(
            "unsupported_time_or_unitless_number:"
            + ",".join(number_risk["medium"][:8])
        )
    if (
        "generated_numbers_require_review" in warning_codes
        and not number_risk["high"]
        and not number_risk["medium"]
    ):
        levels.append("low")
        reasons.append("generated_number_warning_resolved_as_enumeration")

    level = max(levels, key=WARNING_RISK_ORDER.__getitem__) if levels else "clean"
    return {
        "level": level,
        "reasons": reasons,
        "source_warning_codes": sorted(warning_codes),
        "strong_evidence_ids": strong_evidence_ids,
        "medium_strong_evidence_ids": medium_strong_evidence_ids,
        "high_risk_numbers": number_risk["high"],
        "medium_risk_numbers": number_risk["medium"],
    }


def clean_target(target: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(target)
    evidence = cleaned.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                item.pop("start", None)
                item.pop("end", None)
    return cleaned


def compact_json(value: dict[str, Any]) -> str:
    return json.dumps(clean_target(value), ensure_ascii=False, separators=(",", ":"))


def shared_structure(target: dict[str, Any]) -> dict[str, Any]:
    """Return the response fields that must be identical inside one DPO pair."""
    cleaned = clean_target(target)
    return {field: copy.deepcopy(cleaned.get(field)) for field in FROZEN_RESPONSE_FIELDS}


def _evidence_alignment_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    evidence = value.get("evidence", [])
    if not isinstance(evidence, list):
        return ()
    id_to_span: dict[str, str] = {}
    evidence_items: list[tuple[str, str]] = []
    for item in evidence:
        if not isinstance(item, dict):
            return ()
        evidence_id = item.get("id")
        span = item.get("span")
        importance = item.get("importance")
        if not all(isinstance(part, str) for part in (evidence_id, span, importance)):
            return ()
        id_to_span[evidence_id] = span
        evidence_items.append((span, importance))
    critical_spans = sorted(
        id_to_span[evidence_id]
        for evidence_id in value.get("critical_evidence_ids", [])
        if evidence_id in id_to_span
    )
    return tuple(sorted(evidence_items)), tuple(critical_spans)


def answer_structure_compatible(
    response: dict[str, Any], target: dict[str, Any]
) -> bool:
    """Check semantic structure alignment before projecting answer-level fields.

    Query-intent wording and evidence roles may be paraphrased by the model. They
    are canonicalized from the target after task type, sufficiency, evidence
    spans/importance, and critical evidence have been aligned.
    """
    response_clean = clean_target(response)
    target_clean = clean_target(target)
    return (
        response_clean.get("task_type") == target_clean.get("task_type")
        and response_clean.get("evidence_sufficiency")
        == target_clean.get("evidence_sufficiency")
        and _evidence_alignment_signature(response_clean)
        == _evidence_alignment_signature(target_clean)
    )


def project_answer_level_response(
    target: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    """Canonicalize a compatible response onto the target's shared structure."""
    if not answer_structure_compatible(response, target):
        raise ValueError("response structure is not compatible with the target")
    projected = clean_target(target)
    response_clean = clean_target(response)
    for field in ANSWER_LEVEL_FIELDS:
        projected[field] = copy.deepcopy(response_clean.get(field))
    return projected


def project_answer_fields_onto_target(
    target: dict[str, Any], response: dict[str, Any]
) -> dict[str, Any]:
    """Project a candidate's answer fields onto the validated target structure.

    Unlike :func:`project_answer_level_response`, this function deliberately does
    not require the candidate to reproduce the target's task/evidence structure.
    The resulting full response must still be audited before it is admitted to a
    DPO pair.
    """
    if not isinstance(response, dict):
        raise ValueError("candidate response must be a JSON object")

    missing_information = response.get("missing_information")
    if not isinstance(missing_information, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in missing_information
    ):
        raise ValueError("missing_information must be a list of non-empty strings")
    for field in ("clinical_reasoning", "final_answer"):
        value = response.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    projected = clean_target(target)
    for field in ANSWER_LEVEL_FIELDS:
        projected[field] = copy.deepcopy(response[field])
    return projected


def answer_level_signature(response: dict[str, Any]) -> str:
    """Return a canonical key for deduplicating answer-level candidate content."""
    answer_view = {
        field: copy.deepcopy(response.get(field)) for field in ANSWER_LEVEL_FIELDS
    }
    return json.dumps(
        answer_view, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def answer_level_differences(
    first: dict[str, Any], second: dict[str, Any]
) -> list[str]:
    """Return changed answer-level fields, or an empty list if structure drifts."""
    first_clean, second_clean = clean_target(first), clean_target(second)
    if any(
        first_clean.get(field) != second_clean.get(field)
        for field in FROZEN_RESPONSE_FIELDS
    ):
        return []
    return [
        field
        for field in ANSWER_LEVEL_FIELDS
        if first_clean.get(field) != second_clean.get(field)
    ]


def is_isolated_answer_pair(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """True when a pair differs only in at least one answer-level field."""
    return bool(answer_level_differences(first, second))



def _spans(value: Any, critical_only: bool = False) -> list[str]:
    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        return []
    critical_ids = set(value.get("critical_evidence_ids", []))
    spans: list[str] = []
    for item in value["evidence"]:
        if not isinstance(item, dict) or not isinstance(item.get("span"), str):
            continue
        if critical_only and item.get("id") not in critical_ids:
            continue
        spans.append(item["span"])
    return spans


def _overlap_f1(predicted: list[str], reference: list[str]) -> tuple[float, float, float]:
    pred, ref = set(predicted), set(reference)
    overlap = len(pred & ref)
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(ref) if ref else (1.0 if not pred else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 6), round(recall, 6), round(f1, 6)


def audit_response(case_text: str, response: str | dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    parsed = response if isinstance(response, dict) else extract_first_json_object(response)
    if parsed is None:
        return {
            "parsed_output": None,
            "schema_errors": ["invalid_or_incomplete_json"],
            "review_reasons": [],
            "warnings": [],
            "hard_failures": ["H6"],
            "error_tags": ["invalid_json_or_schema"],
            "metrics": {},
        }

    audited = audit_output(case_text, parsed)
    errors = list(audited.errors)
    hard_failures: set[str] = set()
    if errors:
        hard_failures.add("H6")
    if any(error.startswith("evidence_span_not_in_case") for error in errors):
        hard_failures.add("H3")

    normalized = audited.normalized or clean_target(parsed)
    target_clean = clean_target(target)
    predicted_spans = _spans(normalized)
    target_spans = _spans(target_clean)
    predicted_critical = _spans(normalized, critical_only=True)
    target_critical = _spans(target_clean, critical_only=True)
    evidence_precision, evidence_recall, evidence_f1 = _overlap_f1(
        predicted_spans, target_spans
    )
    critical_precision, critical_recall, critical_f1 = _overlap_f1(
        predicted_critical, target_critical
    )

    tags: list[str] = []
    if errors:
        tags.append("invalid_json_or_schema")
    if isinstance(normalized, dict):
        if normalized.get("task_type") != target_clean.get("task_type"):
            tags.append("wrong_task_type")
        if normalized.get("evidence_sufficiency") != target_clean.get("evidence_sufficiency"):
            tags.append("wrong_sufficiency")
        if critical_recall < 1.0:
            tags.append("missing_critical_evidence")
        if evidence_precision < 1.0:
            tags.append("nonreference_evidence")
        if not str(normalized.get("final_answer", "")).strip():
            tags.append("empty_final_answer")

    return {
        "parsed_output": normalized,
        "schema_errors": errors,
        "review_reasons": list(audited.review_reasons),
        "warnings": list(audited.warnings),
        "hard_failures": sorted(hard_failures),
        "error_tags": list(dict.fromkeys(tags)),
        "metrics": {
            "task_type_match": isinstance(normalized, dict)
            and normalized.get("task_type") == target_clean.get("task_type"),
            "sufficiency_match": isinstance(normalized, dict)
            and normalized.get("evidence_sufficiency")
            == target_clean.get("evidence_sufficiency"),
            "evidence_precision": evidence_precision,
            "evidence_recall": evidence_recall,
            "evidence_f1": evidence_f1,
            "critical_precision": critical_precision,
            "critical_recall": critical_recall,
            "critical_f1": critical_f1,
        },
    }


def audit_rank(audit: dict[str, Any]) -> tuple[float, ...]:
    metrics = audit.get("metrics", {})
    return (
        1.0 if not audit.get("hard_failures") else 0.0,
        1.0 if not audit.get("schema_errors") else 0.0,
        float(metrics.get("task_type_match", False)),
        float(metrics.get("sufficiency_match", False)),
        float(metrics.get("critical_f1", 0.0)),
        float(metrics.get("evidence_f1", 0.0)),
    )


def make_controlled_negative(target: dict[str, Any], source_id: str, seed: int = 42) -> dict[str, Any]:
    """Create a deterministic answer-level defect without changing evidence structure."""
    negative = clean_target(target)
    sufficiency = negative.get("evidence_sufficiency")
    if sufficiency == "sufficient":
        options = ["unnecessary_uncertainty", "generic_non_answer"]
    else:
        options = ["overconfident_answer", "generic_non_answer"]
    error_type = options[stable_int(source_id, "controlled", seed=seed) % len(options)]

    if error_type == "overconfident_answer":
        negative["missing_information"] = []
        negative["clinical_reasoning"] = (
            "现有资料已经足以形成明确判断，不需要再核对关键病史、体征或检查结果。"
        )
        negative["final_answer"] = (
            "根据当前描述已经可以得出确定结论，无需补充关键检查或病史，可直接按照该判断处理。"
        )
    elif error_type == "unnecessary_uncertainty":
        negative["missing_information"] = [
            "仍需补充所有相关病史和检查后才能进行任何判断"
        ]
        negative["clinical_reasoning"] = (
            "虽然现有证据已经较充分，但仍拒绝基于这些信息形成有针对性的判断或建议。"
        )
        negative["final_answer"] = (
            "目前完全无法提供任何有针对性的结论或下一步建议，只能等待更多信息后再作判断。"
        )
    else:
        negative["clinical_reasoning"] = (
            "未结合病例中的关键证据进行分析，仅给出缺乏针对性的笼统建议。"
        )
        negative["final_answer"] = (
            "建议结合自身情况继续观察，并在需要时咨询专业人员以获得进一步帮助和处理建议。"
        )

    return {
        "candidate_id": f"controlled_{error_type}",
        "origin": "controlled_negative",
        "intended_error": error_type,
        "text": compact_json(negative),
        "parsed_output": negative,
    }


def judgment_validation_errors(value: Any) -> list[str]:
    """Return stable, field-level errors for a Judge response.

    Keep this validator strict: diagnostics are intended to explain why a response
    was rejected, not to silently coerce an incomplete or ambiguous judgment.
    """

    if not isinstance(value, dict):
        return ["judgment:expected_object"]

    errors: list[str] = []
    decision = value.get("decision")
    if not isinstance(decision, str) or decision not in JUDGE_DECISIONS:
        errors.append(f"decision:invalid:{decision!r}")

    confidence = value.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not 0 <= confidence <= 1
    ):
        errors.append(f"confidence:expected_number_0_to_1:{confidence!r}")

    if not isinstance(value.get("reason"), str) or not value["reason"].strip():
        errors.append("reason:expected_non_empty_string")

    dimensions = value.get("decisive_dimensions")
    if not isinstance(dimensions, list):
        errors.append("decisive_dimensions:expected_non_empty_list")
    elif not dimensions and decision not in {"tie", "unjudgeable"}:
        errors.append("decisive_dimensions:must_not_be_empty")
    else:
        for index, item in enumerate(dimensions):
            if not isinstance(item, str) or item not in SCORE_LIMITS:
                errors.append(
                    f"decisive_dimensions[{index}]:unknown_dimension:{item!r}"
                )

    hard = value.get("hard_failures")
    scores = value.get("scores")
    if not isinstance(hard, dict):
        errors.append("hard_failures:expected_object")
    if not isinstance(scores, dict):
        errors.append("scores:expected_object")

    for label in ("A", "B"):
        if isinstance(hard, dict):
            failures = hard.get(label)
            if not isinstance(failures, list):
                errors.append(f"hard_failures.{label}:expected_list")
            else:
                for index, item in enumerate(failures):
                    if not isinstance(item, str) or item not in HARD_FAILURE_CODES:
                        errors.append(
                            f"hard_failures.{label}[{index}]:unknown_code:{item!r}"
                        )

        if not isinstance(scores, dict):
            continue
        label_scores = scores.get(label)
        if not isinstance(label_scores, dict):
            errors.append(f"scores.{label}:expected_object")
            continue
        expected_names = set(SCORE_LIMITS)
        actual_names = set(label_scores)
        for name in sorted(expected_names - actual_names):
            errors.append(f"scores.{label}.{name}:missing")
        for name in sorted(actual_names - expected_names):
            errors.append(f"scores.{label}.{name}:unexpected")
        for name, maximum in SCORE_LIMITS.items():
            if name not in label_scores:
                continue
            score = label_scores.get(name)
            if (
                isinstance(score, bool)
                or not isinstance(score, int)
                or not 0 <= score <= maximum
            ):
                errors.append(
                    f"scores.{label}.{name}:"
                    f"expected_integer_0_to_{maximum}:{score!r}"
                )
    return errors


def valid_judgment(value: Any) -> bool:
    return not judgment_validation_errors(value)


def judge_score_total(judgment: dict[str, Any], label: str) -> int:
    return sum(int(judgment["scores"][label][name]) for name in SCORE_LIMITS)


__all__ = [
    "ANSWER_LEVEL_FIELDS",
    "DPO_SCHEMA_VERSION",
    "FROZEN_RESPONSE_FIELDS",
    "NATURAL_PAIR_TYPES",
    "PAIR_TYPE_GROUPS",
    "PAIR_TYPES",
    "SCORE_LIMITS",
    "answer_level_differences",
    "answer_level_signature",
    "answer_structure_compatible",
    "assess_target_warning_risk",
    "audit_rank",
    "audit_response",
    "build_training_prompt",
    "clean_target",
    "compact_json",
    "extract_first_json_object",
    "has_high_risk_warning",
    "is_isolated_answer_pair",
    "judgment_validation_errors",
    "judge_score_total",
    "load_environment_file",
    "make_controlled_negative",
    "pair_type_group",
    "project_answer_fields_onto_target",
    "project_answer_level_response",
    "read_jsonl",
    "shared_structure",
    "stable_id",
    "stable_int",
    "valid_judgment",
    "write_jsonl",
]
