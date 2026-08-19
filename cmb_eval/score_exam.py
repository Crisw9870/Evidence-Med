#!/usr/bin/env python3
"""Score CMB-Exam predictions with exact-match single/multiple-choice accuracy."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from cmb_utils import (
    DEFAULT_EXAM_TEST,
    DEFAULT_TEST_ANSWERS,
    exam_item_id,
    extract_choice,
    load_json_list,
    normalized_choice,
    read_jsonl,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--questions", default=str(DEFAULT_EXAM_TEST))
    parser.add_argument(
        "--answers",
        default=str(DEFAULT_TEST_ANSWERS),
        help="Use an empty string when answers are embedded in --questions (e.g. val).",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def wilson(correct: int, total: int) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def aggregate(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    buckets: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field, "unknown"))].append(bool(row["correct"]))
    return {
        key: {
            "correct": sum(values),
            "total": len(values),
            "accuracy": sum(values) / len(values),
        }
        for key, values in sorted(buckets.items())
    }


def main() -> None:
    args = parse_args()
    questions = load_json_list(args.questions)
    predictions = read_jsonl(args.predictions)
    prediction_by_id = {str(row.get("item_id")): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("predictions contain duplicate item_id values")

    external_answers: dict[str, Any] = {}
    if args.answers:
        external_answers = {
            str(row.get("id")): row.get("answer")
            for row in load_json_list(args.answers)
        }

    scored: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        item_id = exam_item_id(question, index)
        prediction = prediction_by_id.get(item_id, {})
        options = question.get("option", {})
        gold_raw = external_answers.get(item_id, question.get("answer"))
        gold = normalized_choice(gold_raw, options)
        if gold is None:
            raise ValueError(f"missing or invalid gold answer for item_id={item_id}")
        predicted = normalized_choice(prediction.get("predicted_answer"), options)
        extraction_method = prediction.get("extraction_method", "missing")
        if predicted is None and isinstance(prediction.get("raw_output"), str):
            reextracted, extraction_method = extract_choice(
                prediction["raw_output"],
                options.keys(),
                str(question.get("question_type", "")),
            )
            predicted = normalized_choice(reextracted, options)
        scored.append(
            {
                "item_id": item_id,
                "model_label": prediction.get("model_label", "unknown"),
                "exam_type": question.get("exam_type"),
                "exam_class": question.get("exam_class"),
                "exam_subject": question.get("exam_subject"),
                "question_type": question.get("question_type"),
                "gold_answer": gold,
                "predicted_answer": predicted,
                "correct": predicted == gold,
                "missing_prediction": not bool(prediction),
                "invalid_prediction": bool(prediction) and predicted is None,
                "extraction_method": extraction_method,
            }
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = output_dir / "scored_items.jsonl"
    with scored_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    correct = sum(row["correct"] for row in scored)
    by_subcategory = aggregate(scored, "exam_class")
    metrics = {
        "model_label": next(
            (row["model_label"] for row in scored if row["model_label"] != "unknown"),
            "unknown",
        ),
        "correct": correct,
        "total": len(scored),
        "accuracy": correct / len(scored),
        "accuracy_ci95_wilson": wilson(correct, len(scored)),
        "macro_subcategory_accuracy": sum(
            value["accuracy"] for value in by_subcategory.values()
        )
        / len(by_subcategory),
        "missing_predictions": sum(row["missing_prediction"] for row in scored),
        "invalid_predictions": sum(row["invalid_prediction"] for row in scored),
        "by_question_type": aggregate(scored, "question_type"),
        "by_exam_type": aggregate(scored, "exam_type"),
        "by_exam_class": by_subcategory,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"scored_items={scored_path}")


if __name__ == "__main__":
    main()
