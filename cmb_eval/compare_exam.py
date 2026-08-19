#!/usr/bin/env python3
"""Paired comparison of two scored CMB-Exam runs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any

from cmb_utils import read_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="Baseline scored_items.jsonl")
    parser.add_argument("--candidate", required=True, help="Candidate scored_items.jsonl")
    parser.add_argument("--baseline-label", default="SFT")
    parser.add_argument("--candidate-label", default="DPO")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--noninferiority-margin",
        type=float,
        default=0.01,
        help="Allowed absolute accuracy loss; pre-specify before reading results.",
    )
    return parser.parse_args()


def two_sided_mcnemar_p(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    if n <= 1000:
        tail = sum(math.comb(n, k) for k in range(0, min(better, worse) + 1)) / (2**n)
        return min(1.0, 2 * tail)
    statistic = (abs(better - worse) - 1) / math.sqrt(n)
    return math.erfc(statistic / math.sqrt(2))


def paired_delta_ci95(better: int, worse: int, total: int) -> list[float]:
    if total <= 1:
        delta = (better - worse) / total if total else 0.0
        return [delta, delta]
    delta = (better - worse) / total
    sum_squares = better + worse
    variance = max(0.0, (sum_squares - total * delta * delta) / (total - 1))
    margin = 1.959963984540054 * math.sqrt(variance / total)
    return [delta - margin, delta + margin]


def main() -> None:
    args = parse_args()
    baseline_rows = {str(row["item_id"]): row for row in read_jsonl(args.baseline)}
    candidate_rows = {str(row["item_id"]): row for row in read_jsonl(args.candidate)}
    if baseline_rows.keys() != candidate_rows.keys():
        missing_baseline = sorted(candidate_rows.keys() - baseline_rows.keys())[:10]
        missing_candidate = sorted(baseline_rows.keys() - candidate_rows.keys())[:10]
        raise ValueError(
            f"item sets differ; missing_baseline={missing_baseline}, "
            f"missing_candidate={missing_candidate}"
        )

    better = worse = both_correct = both_wrong = 0
    by_class: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for item_id in baseline_rows:
        baseline = bool(baseline_rows[item_id]["correct"])
        candidate = bool(candidate_rows[item_id]["correct"])
        if baseline and candidate:
            both_correct += 1
        elif not baseline and not candidate:
            both_wrong += 1
        elif candidate:
            better += 1
        else:
            worse += 1
        by_class[str(baseline_rows[item_id].get("exam_class"))].append(
            (baseline, candidate)
        )

    total = len(baseline_rows)
    baseline_correct = both_correct + worse
    candidate_correct = both_correct + better
    delta_ci95 = paired_delta_ci95(better, worse, total)
    report: dict[str, Any] = {
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "total": total,
        "baseline_accuracy": baseline_correct / total,
        "candidate_accuracy": candidate_correct / total,
        "accuracy_delta": (candidate_correct - baseline_correct) / total,
        "accuracy_delta_ci95_paired_normal": delta_ci95,
        "noninferiority": {
            "margin": args.noninferiority_margin,
            "passes": delta_ci95[0] >= -args.noninferiority_margin,
        },
        "paired_outcomes": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "candidate_only_correct": better,
            "baseline_only_correct": worse,
        },
        "mcnemar_two_sided_p": two_sided_mcnemar_p(better, worse),
        "by_exam_class": {},
    }
    for key, values in sorted(by_class.items()):
        base_accuracy = sum(base for base, _ in values) / len(values)
        candidate_accuracy = sum(candidate for _, candidate in values) / len(values)
        report["by_exam_class"][key] = {
            "total": len(values),
            "baseline_accuracy": base_accuracy,
            "candidate_accuracy": candidate_accuracy,
            "delta": candidate_accuracy - base_accuracy,
        }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
