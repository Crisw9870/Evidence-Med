#!/usr/bin/env python3
"""Select deterministic case-style candidates for teacher distillation.

python scripts/prepare_evidence_candidates.py \
  --input data/sft/medical_100k.jsonl \
  --output data/evidence_sft_5k/00_candidates.jsonl \
  --stats-output data/evidence_sft_5k/00_candidates.stats.json \
  --target-size 5000 \
  --min-score 6 \
  --diagnostic-ratio 0.6 \
  --max-category-fraction 0.25 \
  --seed 42

"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from evidence_sft_common import (
    assign_split,
    canonicalize_text,
    classify_task,
    extract_question_answer,
    iter_jsonl,
    score_case_candidate,
    split_department,
    stable_source_id,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/sft/medical_100k.jsonl")
    parser.add_argument("--output", default="data/evidence_sft/00_candidates.jsonl")
    parser.add_argument("--stats-output", default="data/evidence_sft/00_candidates.stats.json")
    parser.add_argument("--target-size", type=int, default=5000)
    parser.add_argument("--min-score", type=int, default=6)
    parser.add_argument("--diagnostic-ratio", type=float, default=0.6)
    parser.add_argument("--max-category-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def deterministic_order(rows: list[dict], seed: int) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['source_id']}".encode("utf-8")).hexdigest(),
    )


def select_with_category_cap(
    rows: list[dict], target: int, category_cap: int, current_counts: Counter[str]
) -> tuple[list[dict], list[dict]]:
    selected: list[dict] = []
    deferred: list[dict] = []
    for row in rows:
        if len(selected) >= target:
            deferred.append(row)
            continue
        category = row["category"]
        if current_counts[category] >= category_cap:
            deferred.append(row)
            continue
        selected.append(row)
        current_counts[category] += 1
    return selected, deferred


def main() -> None:
    args = parse_args()
    if args.target_size <= 0:
        raise ValueError("--target-size must be positive")
    if not 0 <= args.diagnostic_ratio <= 1:
        raise ValueError("--diagnostic-ratio must be within [0, 1]")

    input_path = Path(args.input)
    candidates: list[dict] = []
    seen: set[str] = set()
    rejection_reasons: Counter[str] = Counter()
    input_rows = 0

    for line_number, record in iter_jsonl(input_path):
        input_rows += 1
        question, answer = extract_question_answer(record)
        if not question or not answer:
            rejection_reasons["missing_question_or_answer"] += 1
            continue
        fingerprint = canonicalize_text(question)
        if not fingerprint or fingerprint in seen:
            rejection_reasons["normalized_duplicate"] += 1
            continue
        seen.add(fingerprint)

        score, reasons = score_case_candidate(question, answer)
        if score < args.min_score:
            if score == 0 and reasons:
                rejection_reasons[reasons[0]] += 1
            else:
                rejection_reasons["score_below_threshold"] += 1
            continue

        category, case_text = split_department(question)
        source_id = stable_source_id(question)
        candidates.append(
            {
                "source_id": source_id,
                "source_file": str(input_path),
                "source_line": line_number,
                "split": assign_split(source_id, args.seed),
                "category": category,
                "task_type_hint": classify_task(case_text),
                "case_text": case_text,
                "original_question": question,
                "original_answer": answer,
                "candidate_score": score,
                "candidate_reasons": reasons,
            }
        )

    by_task = {
        task: deterministic_order([row for row in candidates if row["task_type_hint"] == task], args.seed)
        for task in ("diagnostic_reasoning", "confirmed_management")
    }
    diagnostic_target = round(args.target_size * args.diagnostic_ratio)
    task_targets = {
        "diagnostic_reasoning": diagnostic_target,
        "confirmed_management": args.target_size - diagnostic_target,
    }
    category_cap = max(1, round(args.target_size * args.max_category_fraction))
    category_counts: Counter[str] = Counter()
    selected: list[dict] = []
    deferred: list[dict] = []

    for task in ("diagnostic_reasoning", "confirmed_management"):
        chosen, task_deferred = select_with_category_cap(
            by_task[task], task_targets[task], category_cap, category_counts
        )
        selected.extend(chosen)
        deferred.extend(task_deferred)

    if len(selected) < args.target_size:
        selected_ids = {row["source_id"] for row in selected}
        fill_pool = deterministic_order(
            [row for row in candidates if row["source_id"] not in selected_ids], args.seed + 1
        )
        selected.extend(fill_pool[: args.target_size - len(selected)])

    random.Random(args.seed).shuffle(selected)
    selected = selected[: args.target_size]
    written = write_jsonl(args.output, selected)

    stats = {
        "input": str(input_path),
        "output": str(args.output),
        "seed": args.seed,
        "input_rows": input_rows,
        "unique_case_candidates": len(candidates),
        "selected": written,
        "min_score": args.min_score,
        "task_counts": dict(Counter(row["task_type_hint"] for row in selected)),
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "category_counts": dict(Counter(row["category"] for row in selected).most_common()),
        "rejection_reasons": dict(rejection_reasons.most_common()),
    }
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

