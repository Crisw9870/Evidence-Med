#!/usr/bin/env python3
"""Aggregate paired Evidence Mask metrics and optionally compare M1 against M0."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SUFFICIENCY_LABELS = ("insufficient", "partial", "sufficient", "conflicting")
SUFFICIENCY_RANK = {"insufficient": 0, "partial": 1, "sufficient": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="M1 or single-model predictions.jsonl")
    parser.add_argument("--judgments", default="", help="Response judgments for --predictions")
    parser.add_argument("--baseline-predictions", default="", help="Optional M0 predictions.jsonl")
    parser.add_argument("--baseline-judgments", default="", help="Optional M0 response judgments")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def index_unique(rows: list[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{label} row has no valid {field}")
        if value in result:
            raise ValueError(f"{label} duplicates {field}={value}")
        result[value] = row
    return result


def parsed_judgments(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    rows = read_jsonl(path)
    return {
        row["variant_id"]: row["parsed_judgment"]
        for row in rows
        if row.get("status") == "ok"
        and isinstance(row.get("variant_id"), str)
        and isinstance(row.get("parsed_judgment"), dict)
    }


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_bootstrap_ci(
    differences: list[float], iters: int, seed: int
) -> tuple[float, float]:
    if not differences:
        return 0.0, 0.0
    rng = random.Random(seed)
    sample_count = len(differences)
    estimates = [
        sum(differences[rng.randrange(sample_count)] for _ in range(sample_count))
        / sample_count
        for _ in range(iters)
    ]
    return percentile(estimates, 0.025), percentile(estimates, 0.975)


def exact_mcnemar_p(baseline_only_wrong: int, treatment_only_wrong: int) -> float:
    discordant = baseline_only_wrong + treatment_only_wrong
    if discordant == 0:
        return 1.0
    tail = min(baseline_only_wrong, treatment_only_wrong)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
    return min(1.0, 2 * probability)


def classification_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        gold = str((row.get("gold") or {}).get("evidence_sufficiency"))
        pred = str((row.get("parsed_output") or {}).get("evidence_sufficiency"))
        confusion[gold][pred] += 1
    gold_labels = [label for label in SUFFICIENCY_LABELS if sum(confusion[label].values())]
    per_class: dict[str, Any] = {}
    for label in SUFFICIENCY_LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in confusion if other != label)
        fn = sum(value for pred, value in confusion[label].items() if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": tp + fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    macro_f1 = (
        sum(per_class[label]["f1"] for label in gold_labels) / len(gold_labels)
        if gold_labels
        else 0.0
    )
    balanced_accuracy = (
        sum(per_class[label]["recall"] for label in gold_labels) / len(gold_labels)
        if gold_labels
        else 0.0
    )
    total = len(rows)
    correct = sum(confusion[label][label] for label in confusion)
    majority = max((sum(values.values()) for values in confusion.values()), default=0)
    return {
        "samples": total,
        "accuracy": safe_rate(correct, total),
        "majority_class_baseline": safe_rate(majority, total),
        "macro_f1_over_gold_classes": round(macro_f1, 6),
        "balanced_accuracy_over_gold_classes": round(balanced_accuracy, 6),
        "per_class": per_class,
        "confusion": {gold: dict(values) for gold, values in confusion.items()},
    }


def rate_metric(rows: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    count = sum(bool(row.get("metrics", {}).get(metric)) for row in rows)
    return {"count": count, "rate": safe_rate(count, len(rows))}


def judge_summary(
    rows: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    judged = [(row, judgments[row["variant_id"]]) for row in rows if row["variant_id"] in judgments]
    fields = (
        "appropriate_response",
        "removed_fact_leakage",
        "conclusion_scope_appropriate",
        "missing_information_appropriate",
        "safe_answer",
    )
    return {
        "judged_samples": len(judged),
        **{
            field: {
                "count": sum(bool(value.get(field)) for _, value in judged),
                "rate": safe_rate(sum(bool(value.get(field)) for _, value in judged), len(judged)),
            }
            for field in fields
        },
        "main_conclusion_behavior": dict(
            Counter(str(value.get("main_conclusion_behavior")) for _, value in judged)
        ),
    }


def expected_change_appropriate(
    unmasked: dict[str, Any], masked: dict[str, Any]
) -> bool | None:
    assessment = masked.get("mask_assessment")
    if not isinstance(assessment, dict):
        return None
    expected = assessment.get("expected_certainty_change")
    unmasked_suff = (unmasked.get("parsed_output") or {}).get("evidence_sufficiency")
    masked_suff = (masked.get("parsed_output") or {}).get("evidence_sufficiency")
    if unmasked_suff not in SUFFICIENCY_RANK or masked_suff not in SUFFICIENCY_RANK:
        return None
    if expected == "downgrade":
        return SUFFICIENCY_RANK[masked_suff] < SUFFICIENCY_RANK[unmasked_suff]
    if expected == "stay":
        return SUFFICIENCY_RANK[masked_suff] == SUFFICIENCY_RANK[unmasked_suff]
    return None


def model_summary(
    prediction_rows: list[dict[str, Any]], judgments: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in prediction_rows:
        by_type[str(row["mask_type"])].append(row)
        by_pair[str(row["pair_id"])][str(row["mask_type"])] = row

    variant_summaries: dict[str, Any] = {}
    for mask_type, rows in sorted(by_type.items()):
        evidence_predicted = sum(row["metrics"].get("predicted_evidence_count", 0) for row in rows)
        evidence_grounded = sum(row["metrics"].get("grounded_evidence_count", 0) for row in rows)
        variant_summaries[mask_type] = {
            "samples": len(rows),
            "schema": rate_metric(rows, "schema_valid"),
            "all_evidence_grounded": rate_metric(rows, "all_evidence_grounded"),
            "span_grounding_rate": safe_rate(evidence_grounded, evidence_predicted),
            "removed_span_evidence_leakage": rate_metric(rows, "removed_span_evidence_leakage"),
            "removed_span_literal_mention": rate_metric(rows, "removed_span_literal_mention"),
            "sufficiency": classification_metrics(rows),
            "judge": judge_summary(rows, judgments),
        }

    change_results: dict[str, list[bool]] = defaultdict(list)
    for variants in by_pair.values():
        unmasked = variants.get("unmasked")
        if unmasked is None:
            continue
        for mask_type, masked in variants.items():
            if mask_type == "unmasked":
                continue
            result = expected_change_appropriate(unmasked, masked)
            if result is not None:
                change_results[mask_type].append(result)
    expected_change = {
        mask_type: {
            "eligible": len(values),
            "appropriate_count": sum(values),
            "appropriate_rate": safe_rate(sum(values), len(values)),
        }
        for mask_type, values in sorted(change_results.items())
    }

    return {
        "samples": len(prediction_rows),
        "variant_counts": dict(Counter(str(row["mask_type"]) for row in prediction_rows)),
        "by_mask_type": variant_summaries,
        "sufficiency_expected_change": expected_change,
    }


def compare_binary_judgment(
    treatment_predictions: list[dict[str, Any]],
    treatment_judgments: dict[str, dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    baseline_judgments: dict[str, dict[str, Any]],
    *,
    field: str,
    mask_type: str,
    bootstrap_iters: int,
    seed: int,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    treatment_index = index_unique(treatment_predictions, "variant_id", "treatment predictions")
    baseline_index = index_unique(baseline_predictions, "variant_id", "baseline predictions")
    common = sorted(
        variant_id
        for variant_id in set(treatment_index) & set(baseline_index)
        if treatment_index[variant_id].get("mask_type") == mask_type
        and variant_id in treatment_judgments
        and variant_id in baseline_judgments
    )
    pairs = [
        (
            bool(baseline_judgments[variant_id].get(field)),
            bool(treatment_judgments[variant_id].get(field)),
        )
        for variant_id in common
    ]
    treatment_wins = sum(not baseline and treatment for baseline, treatment in pairs)
    baseline_wins = sum(baseline and not treatment for baseline, treatment in pairs)
    differences = [float(treatment) - float(baseline) for baseline, treatment in pairs]
    lower, upper = paired_bootstrap_ci(differences, bootstrap_iters, seed)
    return {
        "mask_type": mask_type,
        "field": field,
        "paired_samples": len(pairs),
        "baseline_rate": safe_rate(sum(baseline for baseline, _ in pairs), len(pairs)),
        "treatment_rate": safe_rate(sum(treatment for _, treatment in pairs), len(pairs)),
        "delta_pp": round((sum(differences) / len(differences)) * 100, 4) if differences else 0.0,
        "paired_bootstrap_95ci_pp": [round(lower * 100, 4), round(upper * 100, 4)],
        "treatment_only_true": treatment_wins,
        "baseline_only_true": baseline_wins,
        "higher_is_better": higher_is_better,
        "net_treatment_favorable": treatment_wins - baseline_wins
        if higher_is_better
        else baseline_wins - treatment_wins,
        "exact_mcnemar_p": round(exact_mcnemar_p(treatment_wins, baseline_wins), 8),
    }


def compare_specificity_increment(
    treatment_predictions: list[dict[str, Any]],
    treatment_judgments: dict[str, dict[str, Any]],
    baseline_predictions: list[dict[str, Any]],
    baseline_judgments: dict[str, dict[str, Any]],
    *,
    control_type: str,
    bootstrap_iters: int,
    seed: int,
) -> dict[str, Any]:
    def by_pair_and_type(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
        result: dict[tuple[str, str], str] = {}
        for row in rows:
            key = (str(row["pair_id"]), str(row["mask_type"]))
            if key in result:
                raise ValueError(f"duplicate prediction for pair/mask_type={key}")
            result[key] = str(row["variant_id"])
        return result

    treatment_index = by_pair_and_type(treatment_predictions)
    baseline_index = by_pair_and_type(baseline_predictions)
    pair_ids = sorted(
        {
            pair_id
            for pair_id, mask_type in treatment_index
            if mask_type == "critical"
        }
        & {
            pair_id
            for pair_id, mask_type in baseline_index
            if mask_type == "critical"
        }
    )
    differences: list[float] = []
    for pair_id in pair_ids:
        keys = ((pair_id, "critical"), (pair_id, control_type))
        if any(key not in treatment_index or key not in baseline_index for key in keys):
            continue
        treatment_critical = treatment_index[keys[0]]
        treatment_control = treatment_index[keys[1]]
        baseline_critical = baseline_index[keys[0]]
        baseline_control = baseline_index[keys[1]]
        judgment_ids = (
            (treatment_critical, treatment_judgments),
            (treatment_control, treatment_judgments),
            (baseline_critical, baseline_judgments),
            (baseline_control, baseline_judgments),
        )
        if any(variant_id not in judgments for variant_id, judgments in judgment_ids):
            continue
        critical_increment = float(
            treatment_judgments[treatment_critical].get("appropriate_response")
        ) - float(baseline_judgments[baseline_critical].get("appropriate_response"))
        control_increment = float(
            treatment_judgments[treatment_control].get("appropriate_response")
        ) - float(baseline_judgments[baseline_control].get("appropriate_response"))
        differences.append(critical_increment - control_increment)

    lower, upper = paired_bootstrap_ci(differences, bootstrap_iters, seed)
    return {
        "control_type": control_type,
        "paired_cases": len(differences),
        "difference_in_differences_pp": round(
            (sum(differences) / len(differences)) * 100, 4
        )
        if differences
        else 0.0,
        "paired_bootstrap_95ci_pp": [round(lower * 100, 4), round(upper * 100, 4)],
        "positive_cases": sum(value > 0 for value in differences),
        "negative_cases": sum(value < 0 for value in differences),
        "zero_cases": sum(value == 0 for value in differences),
    }


def main() -> None:
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    judgments = parsed_judgments(args.judgments)
    result: dict[str, Any] = {
        "model": model_summary(predictions, judgments),
        "inputs": {
            "predictions": args.predictions,
            "judgments": args.judgments or None,
        },
    }
    if args.baseline_predictions:
        baseline_predictions = read_jsonl(args.baseline_predictions)
        baseline_judgments = parsed_judgments(args.baseline_judgments)
        result["baseline"] = model_summary(baseline_predictions, baseline_judgments)
        comparison = {
            "primary_critical_appropriate_response": compare_binary_judgment(
                predictions,
                judgments,
                baseline_predictions,
                baseline_judgments,
                field="appropriate_response",
                mask_type="critical",
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed,
            ),
            "critical_removed_fact_leakage": compare_binary_judgment(
                predictions,
                judgments,
                baseline_predictions,
                baseline_judgments,
                field="removed_fact_leakage",
                mask_type="critical",
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + 1,
                higher_is_better=False,
            ),
            "supporting_appropriate_response": compare_binary_judgment(
                predictions,
                judgments,
                baseline_predictions,
                baseline_judgments,
                field="appropriate_response",
                mask_type="supporting",
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + 2,
            ),
            "random_appropriate_response": compare_binary_judgment(
                predictions,
                judgments,
                baseline_predictions,
                baseline_judgments,
                field="appropriate_response",
                mask_type="random",
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + 3,
            ),
        }
        comparison["critical_specificity_increment"] = {
            "vs_supporting": compare_specificity_increment(
                predictions,
                judgments,
                baseline_predictions,
                baseline_judgments,
                control_type="supporting",
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + 4,
            ),
            "vs_random": compare_specificity_increment(
                predictions,
                judgments,
                baseline_predictions,
                baseline_judgments,
                control_type="random",
                bootstrap_iters=args.bootstrap_iters,
                seed=args.seed + 5,
            ),
            "interpretation": (
                "Positive values mean the M1-vs-M0 gain is larger on critical deletion "
                "than on the text-deletion control."
            ),
        }
        result["comparison"] = comparison
        result["inputs"].update(
            {
                "baseline_predictions": args.baseline_predictions,
                "baseline_judgments": args.baseline_judgments or None,
            }
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Metrics: {output_path}")


if __name__ == "__main__":
    main()
