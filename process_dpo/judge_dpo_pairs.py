#!/usr/bin/env python3
"""Judge DPO pairs once with a blind A/B rubric using an OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dpo_common import (
    DPO_SCHEMA_VERSION,
    extract_first_json_object,
    judgment_validation_errors,
    load_environment_file,
    read_jsonl,
    write_jsonl,
)


JUDGE_VERSION = "evidence-dpo-answer-judge-v1-single-pass"
MAX_CONSECUTIVE_FAILURES = 10
SYSTEM_PROMPT = """你是中文医疗 Answer-level DPO 偏好裁判。回答 A 与 B 已共享同一 task type、证据充分性和证据结构；你只评价最终回答层面的质量，不得猜测候选来源。

裁决优先级：医学安全与正确性 > 病例证据忠实性 > 回答完整性 > 结论校准与缺失信息 > 建议的安全可执行性 > 表达。

硬错误：H1 医学事实或安全性错误；H2 捏造病例事实；H3 引用 shared_structure 之外的病例事实；H4 结论与当前证据矛盾；H5 严重偏离用户问题；H6 回答不可用或明显截断。

评分范围：medical_correctness 0-3，evidence_faithfulness 0-3，answer_completeness 0-3，calibration 0-3，missing_information 0-2，actionability_safety 0-2，expression 0-1。

病例原文始终优先。shared_structure 对 A/B 完全相同，不代表任一候选来源。不要因为回答更长、更保守或更像固定模板就偏好它。若胜负不清，返回 tie；两者均有严重错误返回 both_bad；信息不足以判断返回 unjudgeable。

只返回下列结构的合法 JSON，不要 Markdown：
{
  "decision": "A_better|B_better|tie|both_bad|unjudgeable",
  "hard_failures": {"A": ["H1"], "B": []},
  "scores": {
    "A": {"medical_correctness": 0, "evidence_faithfulness": 0, "answer_completeness": 0, "calibration": 0, "missing_information": 0, "actionability_safety": 0, "expression": 0},
    "B": {"medical_correctness": 0, "evidence_faithfulness": 0, "answer_completeness": 0, "calibration": 0, "missing_information": 0, "actionability_safety": 0, "expression": 0}
  },
  "decisive_dimensions": ["evidence_faithfulness"],
  "reason": "具体、简短、可核验的裁决理由",
  "confidence": 0.0
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", default="data/dpo/answer_v1/02_pair_candidates.jsonl")
    parser.add_argument("--output", default="data/dpo/answer_v1/03_judgments.jsonl")
    parser.add_argument("--failed-output", default="data/dpo/answer_v1/03_judgments_failed.jsonl")
    parser.add_argument("--preview-output", default="data/dpo/answer_v1/03_judge_requests.preview.jsonl")
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "mimo-v2.5"))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("TEACHER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", ""),
    )
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=MAX_CONSECUTIVE_FAILURES,
        help=(
            "Stop after more than this many consecutive failures; "
            "0 disables the guard."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--retry-failed-only",
        action="store_true",
        help=(
            "Only process unresolved pair_ids currently present in --failed-output; "
            "--limit is applied after this filtering."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_user_prompt(pair: dict[str, Any]) -> str:
    payload = {
        "case_text": pair["case_text"],
        "shared_structure": pair["shared_structure"],
        "answer_A": pair["candidate_A"]["answer_view"],
        "answer_B": pair["candidate_B"]["answer_view"],
    }
    return "请按 rubric 比较两个回答：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["pair_id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok" and row.get("pair_id")
    }


def _failed_rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        pair_id = row.get("pair_id")
        if pair_id and row.get("status") == "judge_error":
            latest[str(pair_id)] = row
    return latest


def _failed_ids(path: Path) -> set[str]:
    return set(_failed_rows_by_id(path))


def _select_pending_pairs(
    pairs: list[dict[str, Any]],
    completed_ids: set[str],
    *,
    retry_failed_only: bool,
    failed_ids: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    pending = [pair for pair in pairs if pair["pair_id"] not in completed_ids]
    if retry_failed_only:
        pending = [pair for pair in pending if pair["pair_id"] in failed_ids]
    if limit > 0:
        pending = pending[:limit]
    return pending


def _compact_failed_output(path: Path, resolved_ids: set[str]) -> int:
    """Keep only the latest row for each currently unresolved failed pair."""

    latest = _failed_rows_by_id(path)
    active = [row for pair_id, row in latest.items() if pair_id not in resolved_ids]
    return write_jsonl(path, active)


def _recover_failed_rows(
    failed_rows: dict[str, dict[str, Any]],
    completed_ids: set[str],
    pair_ids: set[str],
) -> list[dict[str, Any]]:
    """Revalidate saved parsed responses without calling the Judge again."""

    recovered: list[dict[str, Any]] = []
    for pair_id, row in failed_rows.items():
        if pair_id in completed_ids or pair_id not in pair_ids:
            continue
        attempts = row.get("attempt_failures")
        if not isinstance(attempts, list):
            continue
        for attempt in reversed(attempts):
            if not isinstance(attempt, dict):
                continue
            parsed = attempt.get("parsed_judgment")
            if judgment_validation_errors(parsed):
                continue
            recovered.append(
                {
                    "schema_version": row.get("schema_version", DPO_SCHEMA_VERSION),
                    "pair_id": pair_id,
                    "source_id": row.get("source_id"),
                    "split": row.get("split"),
                    "judge_model": row.get("judge_model"),
                    "judge_version": row.get("judge_version", JUDGE_VERSION),
                    "status": "ok",
                    "attempt": attempt.get("attempt"),
                    "finish_reason": attempt.get("finish_reason"),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "parsed_judgment": parsed,
                    "recovered_from_failed": True,
                    "recovery": {
                        "method": "schema_revalidation",
                        "failure_created_at": row.get("created_at"),
                        "previous_validation_errors": attempt.get(
                            "validation_errors", []
                        ),
                        "raw_response": attempt.get("raw_response"),
                    },
                }
            )
            break
    return recovered


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
    return len(rows)


def call_judge(client: Any, pair: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    last_error = "unknown_error"
    attempt_failures: list[dict[str, Any]] = []
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(pair)},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                timeout=args.timeout,
            )
            choice = response.choices[0]
            raw_response = choice.message.content or ""
            parsed = extract_first_json_object(raw_response)
            validation_errors = judgment_validation_errors(parsed)
            if not validation_errors:
                result = {
                    "schema_version": DPO_SCHEMA_VERSION,
                    "pair_id": pair["pair_id"],
                    "source_id": pair["source_id"],
                    "split": pair["split"],
                    "judge_model": args.model,
                    "judge_version": JUDGE_VERSION,
                    "status": "ok",
                    "attempt": attempt,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "parsed_judgment": parsed,
                }
                if attempt_failures:
                    result["prior_attempt_failures"] = attempt_failures
                return result

            last_error = "invalid_judgment_schema"
            attempt_failures.append(
                {
                    "attempt": attempt,
                    "error": last_error,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "raw_response": raw_response,
                    "parsed_judgment": parsed,
                    "validation_errors": validation_errors,
                }
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            attempt_failures.append(
                {
                    "attempt": attempt,
                    "error": last_error,
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                }
            )
        if attempt < args.max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    return {
        "schema_version": DPO_SCHEMA_VERSION,
        "pair_id": pair["pair_id"],
        "source_id": pair["source_id"],
        "split": pair["split"],
        "judge_model": args.model,
        "judge_version": JUDGE_VERSION,
        "status": "judge_error",
        "error": last_error,
        "attempt_count": len(attempt_failures),
        "attempt_failures": attempt_failures,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    load_environment_file()
    args = parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.max_retries < 1:
        raise SystemExit("--max-retries must be at least 1")
    if args.max_consecutive_failures < 0:
        raise SystemExit("--max-consecutive-failures must be at least 0")

    pairs = read_jsonl(args.pairs)
    pair_ids = {pair["pair_id"] for pair in pairs}
    output = Path(args.output)
    failed = Path(args.failed_output)
    completed = _completed_ids(output)
    failed_rows = _failed_rows_by_id(failed) if args.retry_failed_only else {}
    recoverable_rows = _recover_failed_rows(failed_rows, completed, pair_ids)
    recoverable_ids = {row["pair_id"] for row in recoverable_rows}
    recovered_count = 0

    if args.retry_failed_only and not args.dry_run and recoverable_rows:
        recovered_count = _append_rows(output, recoverable_rows)
        completed.update(recoverable_ids)
        _compact_failed_output(failed, completed)

    selection_completed = completed | recoverable_ids
    retry_ids = (
        set(failed_rows) if args.dry_run else _failed_ids(failed)
    ) if args.retry_failed_only else set()
    pending = _select_pending_pairs(
        pairs,
        selection_completed,
        retry_failed_only=args.retry_failed_only,
        failed_ids=retry_ids,
        limit=args.limit,
    )

    if args.retry_failed_only:
        missing_ids = set(failed_rows) - pair_ids
        if missing_ids:
            print(
                f"Warning: {len(missing_ids)} failed pair_ids are absent from "
                f"{args.pairs}"
            )

    if args.dry_run:
        previews = [
            {
                "pair_id": pair["pair_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(pair)},
                ],
            }
            for pair in pending
        ]
        write_jsonl(args.preview_output, previews)
        mode = "failed-only" if args.retry_failed_only else "all-pending"
        print(
            f"Dry run wrote {len(previews)} {mode} single-pass requests to "
            f"{args.preview_output}; locally_recoverable={len(recoverable_rows)}"
        )
        return

    if not pending:
        active_failed = (
            _compact_failed_output(failed, completed) if failed.exists() else 0
        )
        mode = "failed-only" if args.retry_failed_only else "all-pending"
        print(
            f"No {mode} pairs to judge: skipped={len(completed)}, "
            f"recovered={recovered_count}, active_failed={active_failed}"
        )
        return

    api_key = (
        os.environ.get(args.api_key_env)
        or os.environ.get("TEACHER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            f"Missing API key: set {args.api_key_env}, "
            "TEACHER_API_KEY, or OPENAI_API_KEY"
        )
    if not args.model:
        raise SystemExit("Missing judge model: set JUDGE_MODEL or pass --model")

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    failed.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    ok_count = failed_count = consecutive_failed = 0
    run_success_ids: set[str] = set()
    try:
        with output.open("a", encoding="utf-8") as success_handle, failed.open(
            "a", encoding="utf-8"
        ) as failed_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(call_judge, client, pair, args): pair["pair_id"]
                for pair in pending
            }
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                with lock:
                    if result["status"] == "ok":
                        success_handle.write(
                            json.dumps(result, ensure_ascii=False) + "\n"
                        )
                        success_handle.flush()
                        run_success_ids.add(str(result["pair_id"]))
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
                        f"Judged {index}/{len(pending)}: "
                        f"ok={ok_count}, failed={failed_count}"
                    )
    finally:
        active_failed = _compact_failed_output(
            failed, completed | run_success_ids
        )

    mode = "failed-only" if args.retry_failed_only else "all-pending"
    print(
        f"Finished ({mode}): ok={ok_count}, failed={failed_count}, "
        f"recovered={recovered_count}, skipped={len(completed)}, "
        f"active_failed={active_failed}"
    )


if __name__ == "__main__":
    main()
