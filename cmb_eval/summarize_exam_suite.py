#!/usr/bin/env python3
"""Build a compact JSON/CSV scorecard from multiple CMB-Exam runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from cmb_utils import load_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=METRICS_JSON",
        help="Repeat once per model, in desired table order.",
    )
    parser.add_argument("--baseline-label", default="Base")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"invalid --run value: {value}")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise ValueError(f"invalid --run value: {value}")
    return label.strip(), Path(path)


def accuracy_for(metrics: dict[str, Any], question_type: str) -> float | None:
    value = metrics.get("by_question_type", {}).get(question_type)
    return float(value["accuracy"]) if isinstance(value, dict) else None


def main() -> None:
    args = parse_args()
    loaded: list[tuple[str, dict[str, Any]]] = [
        (label, load_json(path)) for label, path in map(parse_run, args.run)
    ]
    if len({label for label, _ in loaded}) != len(loaded):
        raise ValueError("duplicate model labels")
    baseline = next(
        (metrics for label, metrics in loaded if label == args.baseline_label), None
    )
    if baseline is None:
        raise ValueError(f"baseline label not found: {args.baseline_label}")
    rows: list[dict[str, Any]] = []
    for label, metrics in loaded:
        accuracy = float(metrics["accuracy"])
        rows.append(
            {
                "model": label,
                "correct": int(metrics["correct"]),
                "total": int(metrics["total"]),
                "accuracy": accuracy,
                "delta_vs_base": accuracy - float(baseline["accuracy"]),
                "macro_subcategory_accuracy": float(
                    metrics["macro_subcategory_accuracy"]
                ),
                "single_choice_accuracy": accuracy_for(metrics, "单项选择题"),
                "multiple_choice_accuracy": accuracy_for(metrics, "多项选择题"),
                "c_type_accuracy": accuracy_for(metrics, "C型选择题"),
                "invalid_predictions": int(metrics["invalid_predictions"]),
                "missing_predictions": int(metrics["missing_predictions"]),
            }
        )
    report = {
        "baseline_label": args.baseline_label,
        "models": rows,
    }
    write_json(args.output_json, report)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
