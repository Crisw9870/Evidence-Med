#!/usr/bin/env python3
"""Run swap-consistent, source-blind D0-vs-D1 answer-quality evaluation."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dpo_common import (
    ANSWER_LEVEL_FIELDS,
    DPO_SCHEMA_VERSION,
    extract_first_json_object,
    load_environment_file,
    read_jsonl,
    stable_int,
    valid_judgment,
    write_jsonl,
)


EVAL_VERSION = "evidence-dpo-answer-eval-v1-swap"
MAX_CONSECUTIVE_FAILURES = 10
SYSTEM_PROMPT = """你是中文医疗回答质量裁判。请只根据病例原文，比较回答 A 与回答 B 的用户可见质量，不得猜测模型来源。

优先级：医学安全与正确性 > 病例证据忠实性 > 回答完整性 > 结论校准与缺失信息 > 建议的安全可执行性 > 表达。不要因为回答更长、更保守或使用固定模板就偏好它。

硬错误：H1 医学事实或安全性错误；H2 捏造病例事实；H3 引用病例原文不存在的事实；H4 结论与病例证据矛盾；H5 严重偏离用户问题；H6 回答不可用或明显截断。

评分范围：medical_correctness 0-3，evidence_faithfulness 0-3，answer_completeness 0-3，calibration 0-3，missing_information 0-2，actionability_safety 0-2，expression 0-1。

若胜负不清返回 tie；两者均有严重错误返回 both_bad；信息不足以判断返回 unjudgeable。只返回合法 JSON：
{
  "decision": "A_better|B_better|tie|both_bad|unjudgeable",
  "hard_failures": {"A": [], "B": []},
  "scores": {
    "A": {"medical_correctness": 0, "evidence_faithfulness": 0, "answer_completeness": 0, "calibration": 0, "missing_information": 0, "actionability_safety": 0, "expression": 0},
    "B": {"medical_correctness": 0, "evidence_faithfulness": 0, "answer_completeness": 0, "calibration": 0, "missing_information": 0, "actionability_safety": 0, "expression": 0}
  },
  "decisive_dimensions": ["answer_completeness"],
  "reason": "具体、简短、可核验的理由",
  "confidence": 0.0
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-predictions",
        default="results/evidence-frombase_eval/predictions.jsonl",
    )
    parser.add_argument(
        "--dpo-predictions",
        default="results/evidence_dpo_answer_eval/predictions.jsonl",
    )
    parser.add_argument(
        "--output",
        default="results/evidence_dpo_answer_eval/answer_ab_judgments.jsonl",
    )
    parser.add_argument(
        "--failed-output",
        default="results/evidence_dpo_answer_eval/answer_ab_failed.jsonl",
    )
    parser.add_argument(
        "--summary-output",
        default="results/evidence_dpo_answer_eval/answer_ab_summary.json",
    )
    parser.add_argument(
        "--preview-output",
        default="results/evidence_dpo_answer_eval/answer_ab_preview.jsonl",
    )
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", ""))
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
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _parsed_prediction(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed = row.get("parsed_output")
    if isinstance(parsed, dict):
        return parsed
    text = row.get("generated_text")
    if isinstance(text, str):
        return extract_first_json_object(text)
    return None


def _answer_view(row: dict[str, Any]) -> dict[str, Any] | None:
    parsed = _parsed_prediction(row)
    if not isinstance(parsed, dict):
        return None
    answer = {field: parsed.get(field) for field in ANSWER_LEVEL_FIELDS}
    if not all(answer.get(field) is not None for field in ANSWER_LEVEL_FIELDS):
        return None
    return answer


def build_eval_items(
    baseline_rows: list[dict[str, Any]],
    dpo_rows: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    baseline_by_id = {row.get("source_id"): row for row in baseline_rows}
    dpo_by_id = {row.get("source_id"): row for row in dpo_rows}
    common = sorted(
        source_id
        for source_id in baseline_by_id.keys() & dpo_by_id.keys()
        if isinstance(source_id, str) and source_id
    )
    items: list[dict[str, Any]] = []
    for source_id in common:
        baseline = baseline_by_id[source_id]
        dpo = dpo_by_id[source_id]
        if baseline.get("case_text") != dpo.get("case_text"):
            raise ValueError(f"case_text mismatch for {source_id}")
        answers = {
            "baseline": _answer_view(baseline),
            "dpo": _answer_view(dpo),
        }
        if not all(isinstance(value, dict) for value in answers.values()):
            continue
        if stable_int(source_id, "eval_position", seed=seed) % 2:
            first_order = {"A": "dpo", "B": "baseline"}
        else:
            first_order = {"A": "baseline", "B": "dpo"}
        items.append(
            {
                "schema_version": DPO_SCHEMA_VERSION,
                "source_id": source_id,
                "case_text": baseline["case_text"],
                "answers": answers,
                "first_order": first_order,
            }
        )
    return items


def build_user_prompt(item: dict[str, Any], swapped: bool) -> str:
    order = item["first_order"]
    if swapped:
        order = {"A": order["B"], "B": order["A"]}
    payload = {
        "case_text": item["case_text"],
        "answer_A": item["answers"][order["A"]],
        "answer_B": item["answers"][order["B"]],
    }
    return "请比较两个回答：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _winner(judgment: dict[str, Any], order: dict[str, str]) -> str:
    decision = judgment["decision"]
    if decision == "A_better":
        return order["A"]
    if decision == "B_better":
        return order["B"]
    return decision


def reconcile_swapped(
    first: dict[str, Any],
    second: dict[str, Any],
    first_order: dict[str, str],
) -> tuple[bool, str]:
    first_winner = _winner(first, first_order)
    swapped_order = {"A": first_order["B"], "B": first_order["A"]}
    second_winner = _winner(second, swapped_order)
    return first_winner == second_winner, (
        first_winner if first_winner == second_winner else "inconsistent"
    )


def _call_once(
    client: Any,
    item: dict[str, Any],
    args: argparse.Namespace,
    swapped: bool,
) -> dict[str, Any]:
    last_error = "unknown_error"
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(item, swapped),
                    },
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                timeout=args.timeout,
            )
            text = response.choices[0].message.content or ""
            parsed = extract_first_json_object(text)
            if valid_judgment(parsed):
                return {"status": "ok", "attempt": attempt, "judgment": parsed}
            last_error = "invalid_judgment_schema"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    return {"status": "judge_error", "error": last_error}


def judge_item(client: Any, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    first = _call_once(client, item, args, swapped=False)
    second = _call_once(client, item, args, swapped=True)
    if first["status"] != "ok" or second["status"] != "ok":
        return {
            "schema_version": DPO_SCHEMA_VERSION,
            "source_id": item["source_id"],
            "status": "judge_error",
            "first": first,
            "swapped": second,
        }
    consistent, winner = reconcile_swapped(
        first["judgment"], second["judgment"], item["first_order"]
    )
    return {
        "schema_version": DPO_SCHEMA_VERSION,
        "source_id": item["source_id"],
        "judge_model": args.model,
        "eval_version": EVAL_VERSION,
        "status": "ok",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "first_order": item["first_order"],
        "first_judgment": first["judgment"],
        "swapped_judgment": second["judgment"],
        "swap_consistent": consistent,
        "winner": winner,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "ok"]
    consistent = [row for row in completed if row.get("swap_consistent")]
    outcomes = Counter(row.get("winner") for row in consistent)
    decisive = outcomes["dpo"] + outcomes["baseline"]
    return {
        "schema_version": DPO_SCHEMA_VERSION,
        "eval_version": EVAL_VERSION,
        "completed": len(completed),
        "swap_consistent": len(consistent),
        "swap_consistency_rate": (
            len(consistent) / len(completed) if completed else 0.0
        ),
        "outcomes": dict(outcomes),
        "dpo_win_rate_excluding_ties": (
            outcomes["dpo"] / decisive if decisive else 0.0
        ),
        "note": "This is model-judge preference, not clinical correctness.",
    }


def main() -> None:
    load_environment_file()
    args = parse_args()
    items = build_eval_items(
        read_jsonl(args.baseline_predictions),
        read_jsonl(args.dpo_predictions),
        args.seed,
    )
    completed_ids: set[str] = set()
    output = Path(args.output)
    if output.exists():
        completed_ids = {
            row["source_id"]
            for row in read_jsonl(output)
            if row.get("status") == "ok" and row.get("source_id")
        }
    pending = [item for item in items if item["source_id"] not in completed_ids]
    if args.limit > 0:
        pending = pending[: args.limit]

    if args.dry_run:
        previews = []
        for item in pending:
            previews.extend(
                {
                    "source_id": item["source_id"],
                    "swapped": swapped,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_user_prompt(item, swapped),
                        },
                    ],
                }
                for swapped in (False, True)
            )
        write_jsonl(args.preview_output, previews)
        print(f"Dry run wrote {len(previews)} requests to {args.preview_output}")
        return

    api_key = (
        os.environ.get(args.api_key_env)
        or os.environ.get("TEACHER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(f"Missing API key: set {args.api_key_env}")
    if not args.model:
        raise SystemExit("Missing judge model: set JUDGE_MODEL or pass --model")

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    failed = Path(args.failed_output)
    failed.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    consecutive_failed = 0
    with output.open("a", encoding="utf-8") as success_handle, failed.open(
        "a", encoding="utf-8"
    ) as failed_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(judge_item, client, item, args): item["source_id"]
            for item in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            with lock:
                if result["status"] == "ok":
                    success_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success_handle.flush()
                    consecutive_failed = 0
                else:
                    failed_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    failed_handle.flush()
                    consecutive_failed += 1
            if consecutive_failed > MAX_CONSECUTIVE_FAILURES:
                for task in futures:
                    task.cancel()
                raise SystemExit("Stopped after repeated evaluation failures")
            if index % 20 == 0 or index == len(pending):
                print(f"Evaluated {index}/{len(pending)}")

    all_rows = read_jsonl(output)
    summary = summarize(all_rows)
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
