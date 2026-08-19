#!/usr/bin/env python3
"""Aggregate the primary Evidence-SFT vs Evidence-DPO CMB-Clin comparison.

Confidence intervals resample whole cases, not individual QA turns.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from cmb_utils import read_jsonl, write_json
from judge_clin import DIMENSIONS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--baseline-label", default="Evidence-SFT")
    parser.add_argument("--candidate-label", default="Evidence-DPO")
    parser.add_argument("--bootstrap-iters", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def ci95(values: list[float]) -> list[float | None]:
    return [percentile(values, 0.025), percentile(values, 0.975)]


def case_id(row: dict[str, Any]) -> str:
    value = row.get("case_id")
    return str(value) if value is not None else str(row.get("item_id", "")).split(":", 1)[0]


def per_item_scores(
    row: dict[str, Any], labels: tuple[str, str]
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    score_values = {
        label: {dimension: [] for dimension in DIMENSIONS} for label in labels
    }
    hard_values = {label: [] for label in labels}
    for call_name, order_name in (("first", "first_order"), ("swapped", "swapped_order")):
        call = row.get(call_name, {})
        judgment = call.get("judgment") if isinstance(call, dict) else None
        order = row.get(order_name, {})
        if not isinstance(judgment, dict) or not isinstance(order, dict):
            continue
        for side in ("A", "B"):
            label = order.get(side)
            if label not in score_values:
                continue
            for dimension in DIMENSIONS:
                score_values[label][dimension].append(
                    float(judgment["scores"][side][dimension])
                )
            hard_values[label].append(
                1.0 if judgment["hard_medical_error"][side] else 0.0
            )
    scores = {
        label: {
            dimension: float(mean(values) or 0.0)
            for dimension, values in dimensions.items()
        }
        for label, dimensions in score_values.items()
    }
    hard = {label: float(mean(values) or 0.0) for label, values in hard_values.items()}
    return scores, hard


def cluster_bootstrap(
    records: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], dict[str, float | None]],
    iterations: int,
    seed: int,
) -> dict[str, list[float | None]]:
    if not records or iterations <= 0:
        return {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["case_id"])].append(row)
    cases = sorted(grouped)
    rng = random.Random(seed)
    sampled_values: dict[str, list[float]] = defaultdict(list)
    for _ in range(iterations):
        sampled: list[dict[str, Any]] = []
        for sampled_case in (rng.choice(cases) for _ in cases):
            sampled.extend(grouped[sampled_case])
        for key, value in statistic(sampled).items():
            if value is not None:
                sampled_values[key].append(float(value))
    return {key: ci95(values) for key, values in sampled_values.items()}


def outcome_statistics(
    rows: list[dict[str, Any]], baseline: str, candidate: str
) -> dict[str, float | None]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    comparable = outcomes[baseline] + outcomes[candidate] + outcomes["tie"]
    if comparable == 0:
        return {"candidate_win_rate": None, "net_win_rate": None}
    return {
        "candidate_win_rate": outcomes[candidate] / comparable,
        "net_win_rate": (outcomes[candidate] - outcomes[baseline]) / comparable,
    }


def delta_statistics(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    result = {
        dimension: mean([float(row["score_delta"][dimension]) for row in rows])
        for dimension in DIMENSIONS
    }
    result["hard_medical_error_rate"] = mean(
        [float(row["hard_error_delta"]) for row in rows]
    )
    return result


def prediction_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_values = [float(row["generated_tokens"]) for row in rows if row.get("generated_tokens") is not None]
    char_values = [float(len(str(row.get("model_answer", "")))) for row in rows]
    truncations = [not bool(row.get("ended_with_eos")) for row in rows]
    return {
        "items": len(rows),
        "mean_generated_tokens": mean(token_values),
        "mean_answer_characters": mean(char_values),
        "truncation_rate": sum(truncations) / len(truncations) if truncations else None,
    }


def main() -> None:
    args = parse_args()
    judgments = read_jsonl(args.judgments)
    baseline_predictions = read_jsonl(args.baseline_predictions)
    candidate_predictions = read_jsonl(args.candidate_predictions)
    baseline_by_id = {str(row["item_id"]): row for row in baseline_predictions}
    candidate_by_id = {str(row["item_id"]): row for row in candidate_predictions}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate prediction item sets differ")

    labels = (args.baseline_label, args.candidate_label)
    ok = [row for row in judgments if row.get("status") == "ok"]
    consistent = [row for row in ok if row.get("swap_consistent")]
    outcomes = Counter(str(row.get("reconciled_outcome")) for row in consistent)

    score_records: list[dict[str, Any]] = []
    score_values = {
        label: {dimension: [] for dimension in DIMENSIONS} for label in labels
    }
    hard_values = {label: [] for label in labels}
    for row in ok:
        scores, hard = per_item_scores(row, labels)
        for label in labels:
            for dimension in DIMENSIONS:
                score_values[label][dimension].append(scores[label][dimension])
            hard_values[label].append(hard[label])
        score_records.append(
            {
                "case_id": case_id(row),
                "score_delta": {
                    dimension: scores[args.candidate_label][dimension]
                    - scores[args.baseline_label][dimension]
                    for dimension in DIMENSIONS
                },
                "hard_error_delta": hard[args.candidate_label]
                - hard[args.baseline_label],
            }
        )

    outcome_records = [
        {
            "case_id": case_id(row),
            "outcome": str(row.get("reconciled_outcome")),
        }
        for row in consistent
        if row.get("reconciled_outcome")
        in {args.baseline_label, args.candidate_label, "tie"}
    ]
    outcome_point = outcome_statistics(
        outcome_records, args.baseline_label, args.candidate_label
    )
    delta_point = delta_statistics(score_records)

    report: dict[str, Any] = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "total_prediction_items": len(baseline_predictions),
        "total_judgment_records": len(judgments),
        "judge_ok": len(ok),
        "judge_errors": len(judgments) - len(ok),
        "swap_consistent": len(consistent),
        "swap_consistency_rate": len(consistent) / len(ok) if ok else 0.0,
        "outcomes": dict(outcomes),
        "candidate_win_rate_among_comparable": outcome_point["candidate_win_rate"],
        "net_win_rate": outcome_point["net_win_rate"],
        "outcome_cluster_bootstrap_ci95": cluster_bootstrap(
            outcome_records,
            lambda rows: outcome_statistics(
                rows, args.baseline_label, args.candidate_label
            ),
            args.bootstrap_iters,
            args.seed,
        ),
        "average_scores_over_both_positions": {
            label: {
                dimension: mean(values)
                for dimension, values in dimensions.items()
            }
            for label, dimensions in score_values.items()
        },
        "score_delta_candidate_minus_baseline": {
            dimension: delta_point[dimension] for dimension in DIMENSIONS
        },
        "hard_medical_error_rate": {
            label: mean(values) for label, values in hard_values.items()
        },
        "hard_medical_error_rate_delta": delta_point["hard_medical_error_rate"],
        "delta_cluster_bootstrap_ci95": cluster_bootstrap(
            score_records,
            delta_statistics,
            args.bootstrap_iters,
            args.seed + 1,
        ),
        "response_length_and_truncation": {
            args.baseline_label: prediction_stats(baseline_predictions),
            args.candidate_label: prediction_stats(candidate_predictions),
        },
        "bootstrap": {
            "unit": "case",
            "cases": len({case_id(row) for row in judgments}),
            "iterations": args.bootstrap_iters,
            "seed": args.seed,
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
