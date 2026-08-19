#!/usr/bin/env python3
"""Judge eligible natural DPO pairs once more with Candidate A/B swapped."""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dpo_common import NATURAL_PAIR_TYPES, read_jsonl, stable_int, write_jsonl
from export_dpo_dataset import _qualify
from judge_dpo_pairs import (
    MAX_CONSECUTIVE_FAILURES,
    SYSTEM_PROMPT,
    _append_rows,
    _compact_failed_output,
    _completed_ids,
    _failed_ids,
    _failed_rows_by_id,
    _recover_failed_rows,
    call_judge,
    build_user_prompt,
    load_environment_file,
)


SWAP_JUDGE_VERSION = "evidence-dpo-answer-judge-v1-single-pass-swapped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/dpo/answer_v1/02_pair_candidates.jsonl")
    parser.add_argument("--judgments", default="data/dpo/answer_v1/03_judgments.jsonl")
    parser.add_argument("--output", default="data/dpo/answer_v1/03_swap_judgments.jsonl")
    parser.add_argument(
        "--failed-output", default="data/dpo/answer_v1/03_swap_judgments_failed.jsonl"
    )
    parser.add_argument(
        "--stats-output", default="data/dpo/answer_v1/03_swap_judgments.stats.json"
    )
    parser.add_argument(
        "--preview-output", default="data/dpo/answer_v1/03_swap_requests.preview.jsonl"
    )
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "mimo-v2.5"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("TEACHER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", ""),
    )
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-per-type", type=int, default=0)
    parser.add_argument("--min-confidence", type=float, default=0.85)
    parser.add_argument("--min-score-margin", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-consecutive-failures", type=int, default=MAX_CONSECUTIVE_FAILURES
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry-failed-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _validate_unique(rows: list[dict[str, Any]], name: str) -> None:
    ids = [str(row.get("pair_id", "")) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError(f"{name} contains missing or duplicate pair_id")


def _swapped_pair(pair: dict[str, Any]) -> dict[str, Any]:
    return {
        **pair,
        "candidate_A": pair["candidate_B"],
        "candidate_B": pair["candidate_A"],
    }


def eligible_natural_pairs(
    pairs: list[dict[str, Any]],
    judgments: list[dict[str, Any]],
    min_confidence: float,
    min_score_margin: int,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    _validate_unique(pairs, "pairs")
    _validate_unique(judgments, "judgments")
    judgment_by_id = {row["pair_id"]: row for row in judgments}
    qualify_args = SimpleNamespace(
        min_confidence=min_confidence,
        min_score_margin=min_score_margin,
    )
    eligible: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    for pair in pairs:
        if pair.get("pair_type") not in NATURAL_PAIR_TYPES:
            continue
        judgment = judgment_by_id.get(pair["pair_id"])
        if judgment is None:
            reasons["missing_original_judgment"] += 1
            continue
        if (
            judgment.get("source_id") != pair.get("source_id")
            or judgment.get("split") != pair.get("split")
        ):
            reasons["original_metadata_mismatch"] += 1
            continue
        item, reason = _qualify(pair, judgment, qualify_args)
        if item is None:
            reasons[reason] += 1
            continue
        eligible.append(pair)
        reasons["eligible"] += 1
    return eligible, reasons


def select_swap_pairs(
    eligible: list[dict[str, Any]], limit_per_type: int, seed: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for pair_type in sorted(NATURAL_PAIR_TYPES):
        values = [pair for pair in eligible if pair["pair_type"] == pair_type]
        values.sort(
            key=lambda pair: stable_int(
                "swap_smoke", pair["pair_id"], seed=seed
            )
        )
        if limit_per_type > 0:
            values = values[:limit_per_type]
        selected.extend(values)
    return selected


def _tag_swapped(result: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(result)
    tagged["presentation"] = "swapped"
    tagged["judge_version"] = SWAP_JUDGE_VERSION
    tagged["presentation_map"] = {"A": "original_B", "B": "original_A"}
    return tagged


def call_swapped_judge(
    client: Any, pair: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    return _tag_swapped(call_judge(client, _swapped_pair(pair), args))


def swap_stats(
    eligible: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    original_reasons: Counter[str],
    limit_per_type: int,
) -> dict[str, Any]:
    selected_ids = {pair["pair_id"] for pair in selected}
    completed = [row for row in completed_rows if row.get("pair_id") in selected_ids]
    failed = [row for row in failed_rows if row.get("pair_id") in selected_ids]
    pair_by_id = {pair["pair_id"]: pair for pair in selected}
    return {
        "eligible": len(eligible),
        "eligible_by_type": dict(Counter(pair["pair_type"] for pair in eligible)),
        "eligible_by_split": dict(Counter(pair["split"] for pair in eligible)),
        "original_filter_reasons": dict(original_reasons),
        "selected": len(selected),
        "selected_by_type": dict(Counter(pair["pair_type"] for pair in selected)),
        "selected_by_split": dict(Counter(pair["split"] for pair in selected)),
        "completed": len(completed),
        "failed": len(failed),
        "coverage_rate": len(completed) / len(selected) if selected else 0.0,
        "decisions": dict(
            Counter(row["parsed_judgment"]["decision"] for row in completed)
        ),
        "decisions_by_type": {
            pair_type: dict(
                Counter(
                    row["parsed_judgment"]["decision"]
                    for row in completed
                    if pair_by_id[row["pair_id"]]["pair_type"] == pair_type
                )
            )
            for pair_type in sorted(NATURAL_PAIR_TYPES)
        },
        "limit_per_type": limit_per_type,
        "acceptance": {
            "minimum_schema_coverage": 0.98,
            "schema_coverage_pass": (
                len(completed) / len(selected) >= 0.98 if selected else False
            ),
            "failed_must_be_zero": len(failed) == 0,
        },
    }


def _write_stats(
    path: Path,
    eligible: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    original_reasons: Counter[str],
    output: Path,
    failed: Path,
    limit_per_type: int,
) -> dict[str, Any]:
    completed_rows = read_jsonl(output) if output.exists() else []
    failed_rows = read_jsonl(failed) if failed.exists() else []
    stats = swap_stats(
        eligible,
        selected,
        completed_rows,
        failed_rows,
        original_reasons,
        limit_per_type,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    load_environment_file()
    args = parse_args()
    if args.workers < 1 or args.max_retries < 1:
        raise SystemExit("--workers and --max-retries must be at least 1")
    if args.max_consecutive_failures < 0:
        raise SystemExit("--max-consecutive-failures must be at least 0")

    pairs = read_jsonl(args.pairs)
    judgments = read_jsonl(args.judgments)
    eligible, original_reasons = eligible_natural_pairs(
        pairs, judgments, args.min_confidence, args.min_score_margin
    )
    selected = select_swap_pairs(eligible, args.limit_per_type, args.seed)
    selected_by_id = {pair["pair_id"]: pair for pair in selected}
    output = Path(args.output)
    failed = Path(args.failed_output)
    completed = _completed_ids(output)

    failed_rows = _failed_rows_by_id(failed) if args.retry_failed_only else {}
    recoverable = _recover_failed_rows(failed_rows, completed, set(selected_by_id))
    recoverable = [_tag_swapped(row) for row in recoverable]
    recoverable_ids = {row["pair_id"] for row in recoverable}
    if args.retry_failed_only and not args.dry_run and recoverable:
        _append_rows(output, recoverable)
        completed.update(recoverable_ids)
        _compact_failed_output(failed, completed)

    pending = [pair for pair in selected if pair["pair_id"] not in completed]
    if args.retry_failed_only:
        retry_ids = set(failed_rows) if args.dry_run else _failed_ids(failed)
        pending = [pair for pair in pending if pair["pair_id"] in retry_ids]

    if args.dry_run:
        previews = [
            {
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "split": pair["split"],
                "presentation": "swapped",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(_swapped_pair(pair)),
                    },
                ],
            }
            for pair in pending
        ]
        write_jsonl(args.preview_output, previews)
        stats = _write_stats(
            Path(args.stats_output), eligible, selected, original_reasons,
            output, failed, args.limit_per_type
        )
        print(
            f"Dry run wrote {len(previews)} swapped requests; "
            f"eligible={len(eligible)}, selected={len(selected)}, "
            f"completed={stats['completed']}"
        )
        return

    if not pending:
        stats = _write_stats(
            Path(args.stats_output), eligible, selected, original_reasons,
            output, failed, args.limit_per_type
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return

    api_key = (
        os.environ.get(args.api_key_env)
        or os.environ.get("TEACHER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(f"Missing API key: set {args.api_key_env}")
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)
    run_success_ids: set[str] = set()
    ok_count = failed_count = consecutive_failed = 0
    lock = threading.Lock()
    try:
        with output.open("a", encoding="utf-8") as success_handle, failed.open(
            "a", encoding="utf-8"
        ) as failed_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(call_swapped_judge, client, pair, args): pair["pair_id"]
                for pair in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                with lock:
                    if result["status"] == "ok":
                        success_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        success_handle.flush()
                        run_success_ids.add(result["pair_id"])
                        ok_count += 1
                        consecutive_failed = 0
                    else:
                        failed_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        failed_handle.flush()
                        failed_count += 1
                        consecutive_failed += 1
                if (
                    args.max_consecutive_failures > 0
                    and consecutive_failed > args.max_consecutive_failures
                ):
                    for item in futures:
                        item.cancel()
                    raise SystemExit(
                        f"Stopped after {consecutive_failed} consecutive failures"
                    )
                if index % 20 == 0 or index == len(pending):
                    print(
                        f"Swapped Judge {index}/{len(pending)}: "
                        f"ok={ok_count}, failed={failed_count}"
                    )
    finally:
        _compact_failed_output(failed, completed | run_success_ids)
        stats = _write_stats(
            Path(args.stats_output), eligible, selected, original_reasons,
            output, failed, args.limit_per_type
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
