#!/usr/bin/env python3
"""Blind, swap-consistent pairwise LLM judging for SFT-vs-DPO CMB-Clin answers."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from cmb_utils import extract_first_json_object, read_jsonl, stable_int


DIMENSIONS = ("fluency", "relevance", "completeness", "medical_proficiency")
DECISIONS = {"A_better", "B_better", "tie", "both_bad", "unjudgeable"}
SYSTEM_PROMPT = """你是中文医疗模型评测专家。请依据病例、当前问题和参考答案，盲评回答 A 与回答 B。不要猜测模型来源。

评分维度均为 1 至 5 分：
- fluency：语言流畅、清晰程度；
- relevance：是否直接回答当前问题；
- completeness：是否覆盖参考答案中的关键点；
- medical_proficiency：医学知识、诊断和处置是否正确。

医学正确性与安全性优先于文风和篇幅。不要因为回答更长、更保守或术语更多而偏好它。若存在可能导致错误诊疗的明确事实、禁忌或处置错误，将对应 hard_medical_error 标为 true。

只输出合法 JSON：
{
  "decision": "A_better|B_better|tie|both_bad|unjudgeable",
  "scores": {
    "A": {"fluency": 1, "relevance": 1, "completeness": 1, "medical_proficiency": 1},
    "B": {"fluency": 1, "relevance": 1, "completeness": 1, "medical_proficiency": 1}
  },
  "hard_medical_error": {"A": false, "B": false},
  "reason": "简短、具体、可核验的理由"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--candidate-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-label", default="SFT")
    parser.add_argument("--candidate-label", default="DPO")
    parser.add_argument("--model", default=os.environ.get("JUDGE_MODEL", ""))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", ""),
    )
    parser.add_argument("--api-key-env", default="JUDGE_API_KEY")
    parser.add_argument("--env-file", default=str(Path(__file__).resolve().parent.parent / ".env"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_env(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def valid_judgment(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("decision") not in DECISIONS:
        return False
    scores = value.get("scores")
    if not isinstance(scores, dict) or set(scores) != {"A", "B"}:
        return False
    for side in ("A", "B"):
        if not isinstance(scores[side], dict) or set(scores[side]) != set(DIMENSIONS):
            return False
        if not all(
            isinstance(scores[side][dimension], int)
            and 1 <= scores[side][dimension] <= 5
            for dimension in DIMENSIONS
        ):
            return False
    errors = value.get("hard_medical_error")
    return (
        isinstance(errors, dict)
        and set(errors) == {"A", "B"}
        and all(isinstance(errors[side], bool) for side in ("A", "B"))
        and isinstance(value.get("reason"), str)
        and bool(value["reason"].strip())
    )


def build_items(args: argparse.Namespace) -> list[dict[str, Any]]:
    baseline = {str(row["item_id"]): row for row in read_jsonl(args.baseline_predictions)}
    candidate = {str(row["item_id"]): row for row in read_jsonl(args.candidate_predictions)}
    if baseline.keys() != candidate.keys():
        raise ValueError("baseline and candidate item_id sets differ")
    items: list[dict[str, Any]] = []
    for item_id in sorted(baseline, key=lambda value: tuple(int(x) for x in value.split(":"))):
        base_row = baseline[item_id]
        candidate_row = candidate[item_id]
        for field in ("case_description", "question", "reference_answer"):
            if base_row.get(field) != candidate_row.get(field):
                raise ValueError(f"{field} mismatch for {item_id}")
        answers = {
            args.baseline_label: base_row.get("model_answer", ""),
            args.candidate_label: candidate_row.get("model_answer", ""),
        }
        histories = {
            args.baseline_label: base_row.get("history_before", []),
            args.candidate_label: candidate_row.get("history_before", []),
        }
        labels = [args.baseline_label, args.candidate_label]
        if stable_int(item_id, "position", seed=args.seed) % 2:
            labels.reverse()
        items.append(
            {
                "item_id": item_id,
                "case_id": base_row.get("case_id"),
                "turn_index": base_row.get("turn_index"),
                "case_description": base_row.get("case_description"),
                "question": base_row.get("question"),
                "reference_answer": base_row.get("reference_answer"),
                "answers": answers,
                "histories": histories,
                "first_order": {"A": labels[0], "B": labels[1]},
            }
        )
    return items


def user_prompt(item: dict[str, Any], order: dict[str, str]) -> str:
    payload = {
        "病例": item["case_description"],
        "当前问题": item["question"],
        "参考答案": item["reference_answer"],
        "回答A的前序对话": item["histories"][order["A"]],
        "回答A": item["answers"][order["A"]],
        "回答B的前序对话": item["histories"][order["B"]],
        "回答B": item["answers"][order["B"]],
    }
    return "请盲评以下两个回答：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def winner(judgment: dict[str, Any], order: dict[str, str]) -> str:
    decision = judgment["decision"]
    if decision == "A_better":
        return order["A"]
    if decision == "B_better":
        return order["B"]
    return decision


def call_once(client: Any, item: dict[str, Any], order: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    last_error = "unknown_error"
    for attempt in range(1, args.max_retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(item, order)},
                ],
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "timeout": args.timeout,
            }
            if not args.no_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if args.disable_thinking:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            response = client.chat.completions.create(**kwargs)
            text = response.choices[0].message.content or ""
            parsed = extract_first_json_object(text)
            if valid_judgment(parsed):
                return {"status": "ok", "attempt": attempt, "text": text, "judgment": parsed}
            last_error = "invalid_judgment_schema"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    return {"status": "judge_error", "error": last_error}


def judge_item(client: Any, item: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    first_order = item["first_order"]
    swapped_order = {"A": first_order["B"], "B": first_order["A"]}
    first = call_once(client, item, first_order, args)
    swapped = call_once(client, item, swapped_order, args)
    record: dict[str, Any] = {
        "item_id": item["item_id"],
        "case_id": item["case_id"],
        "turn_index": item["turn_index"],
        "judge_model": args.model,
        "first_order": first_order,
        "swapped_order": swapped_order,
        "first": first,
        "swapped": swapped,
    }
    if first["status"] != "ok" or swapped["status"] != "ok":
        record["status"] = "judge_error"
        return record
    first_winner = winner(first["judgment"], first_order)
    swapped_winner = winner(swapped["judgment"], swapped_order)
    record.update(
        status="ok",
        swap_consistent=first_winner == swapped_winner,
        reconciled_outcome=(
            first_winner if first_winner == swapped_winner else "inconsistent"
        ),
    )
    return record


def main() -> None:
    args = parse_args()
    load_env(args.env_file)
    args.model = args.model or os.environ.get("JUDGE_MODEL", "")
    args.base_url = (
        args.base_url
        or os.environ.get("JUDGE_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", "")
    )
    if not args.model:
        raise ValueError("--model or JUDGE_MODEL is required")
    items = build_items(args)
    if args.limit > 0:
        items = items[: args.limit]
    output = Path(args.output)
    if output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"output exists; use --resume or choose another path: {output}")
    existing = read_jsonl(output) if args.resume and output.exists() else []
    completed = {str(row["item_id"]) for row in existing if row.get("status") == "ok"}
    pending = [item for item in items if item["item_id"] not in completed]

    if args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        preview = output.with_suffix(output.suffix + ".preview.jsonl")
        with preview.open("w", encoding="utf-8") as handle:
            for item in pending:
                handle.write(
                    json.dumps(
                        {
                            "item_id": item["item_id"],
                            "first_order": item["first_order"],
                            "prompt": user_prompt(item, item["first_order"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"preview={preview} items={len(pending)}")
        return

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("judge_clin.py requires the openai package") from exc
    api_key = os.environ.get(args.api_key_env) or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    thread_local = threading.local()

    def get_client() -> Any:
        if not hasattr(thread_local, "client"):
            kwargs: dict[str, Any] = {"api_key": api_key}
            if args.base_url:
                kwargs["base_url"] = args.base_url
            thread_local.client = OpenAI(**kwargs)
        return thread_local.client

    def judge_with_thread_client(item: dict[str, Any]) -> dict[str, Any]:
        return judge_item(get_client(), item, args)

    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and output.exists() else "w"
    with output.open(mode, encoding="utf-8") as handle, ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor, tqdm(total=len(pending), desc="CMB-Clin judge", unit="item") as progress:
        futures = {
            executor.submit(judge_with_thread_client, item): item["item_id"]
            for item in pending
        }
        for future in as_completed(futures):
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "item_id": futures[future],
                    "status": "judge_error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            progress.update(1)
    print(f"saved={output}")


if __name__ == "__main__":
    main()
