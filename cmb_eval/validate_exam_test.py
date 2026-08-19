#!/usr/bin/env python3
"""Validate only CMB-Exam test questions and their separate answer file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

from cmb_utils import (
    DEFAULT_EXAM_TEST,
    DEFAULT_TEST_ANSWERS,
    load_json_list,
    normalized_choice,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default=str(DEFAULT_EXAM_TEST))
    parser.add_argument("--answers", default=str(DEFAULT_TEST_ANSWERS))
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_json_list(args.test)
    answers = load_json_list(args.answers)
    answer_by_id = {str(row.get("id")): row.get("answer") for row in answers}
    ids: list[str] = []
    missing_fields = 0
    missing_answers = 0
    invalid_answers = 0
    question_types: Counter[str] = Counter()
    exam_types: Counter[str] = Counter()
    exam_classes: Counter[str] = Counter()
    required = {
        "id",
        "exam_type",
        "exam_class",
        "exam_subject",
        "question",
        "question_type",
        "option",
    }
    for row in questions:
        if not required.issubset(row) or not isinstance(row.get("option"), dict):
            missing_fields += 1
            continue
        item_id = str(row["id"])
        ids.append(item_id)
        if item_id not in answer_by_id:
            missing_answers += 1
        elif normalized_choice(answer_by_id[item_id], row["option"].keys()) is None:
            invalid_answers += 1
        question_types[str(row["question_type"])] += 1
        exam_types[str(row["exam_type"])] += 1
        exam_classes[str(row["exam_class"])] += 1

    report: dict[str, Any] = {
        "test_file": args.test,
        "answer_file": args.answers,
        "questions": len(questions),
        "answers": len(answers),
        "missing_fields": missing_fields,
        "duplicate_question_ids": len(ids) - len(set(ids)),
        "missing_answers": missing_answers,
        "invalid_answers": invalid_answers,
        "major_category_count": len(exam_types),
        "subcategory_count": len(exam_classes),
        "by_question_type": dict(question_types),
        "by_exam_type": dict(exam_types),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output, report)
    if (
        len(questions) != len(answers)
        or missing_fields
        or report["duplicate_question_ids"]
        or missing_answers
        or invalid_answers
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
