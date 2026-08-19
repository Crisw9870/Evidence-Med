#!/usr/bin/env python3
"""Judge whether a masked target is a valid counterfactual of its parent case."""

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

from build_evidence_mask_targets import load_environment_file
from evidence_mask_common import read_jsonl
from evidence_sft_common import extract_first_json_object


JUDGE_VERSION = "evidence-mask-judge-v1"
MAX_CONSECUTIVE_FAILURES = 10

SYSTEM_PROMPT = """你是 Evidence Mask 反事实数据审核员。你会看到完整病例、删除后的病例、被删除事实、完整目标和删除后目标。

请审核的不是完整目标是否和删除后目标逐字相同，而是删除后目标是否只依据仍然可见的信息，并对证据缺失作出合理反应。

判定标准：
1. 删除后病例语义仍可理解，用户问题没有被破坏；
2. 被删除事实确实不再出现在删除后病例中，也不能由剩余文本直接恢复；
3. 删除后 evidence 全部来自删除后病例原文；
4. 删除后回答没有把被删除事实当作患者已知事实；
5. 主要结论、回答范围或确定性应与剩余证据匹配；存在证据冗余时允许保持，不得机械要求每次都降级；
6. missing_information 应指出真正影响判断的缺失概念，但不得声称患者已经具有该事实；
7. 对 supporting/random 删除，若主要结论不应改变，应标记 expected_certainty_change=stay。

只输出：
{
  "decision": "accept|review|reject",
  "expected_certainty_change": "downgrade|stay",
  "removed_fact_concepts": ["被删除事实的简短概念"],
  "required_missing_concepts": ["回答中应体现的关键缺失概念"],
  "allowed_conclusion_scope": "剩余证据允许达到的结论范围",
  "forbidden_specific_claims": ["当前不再有依据的具体断言"],
  "reasons": ["简短审核理由"]
}
"""

REQUIRED_FIELDS = {
    "decision",
    "expected_certainty_change",
    "removed_fact_concepts",
    "required_missing_concepts",
    "allowed_conclusion_scope",
    "forbidden_specific_claims",
    "reasons",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="data/evidence_mask/v1/00_candidates.jsonl")
    parser.add_argument("--teacher", default="data/evidence_mask/v1/01_teacher_raw.jsonl")
    parser.add_argument("--output", default="data/evidence_mask/v1/02_judgments.jsonl")
    parser.add_argument("--failed-output", default="data/evidence_mask/v1/02_judgments_failed.jsonl")
    parser.add_argument(
        "--preview-output",
        default="data/evidence_mask/v1/02_judge_requests.preview.jsonl",
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def valid_judgment(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != REQUIRED_FIELDS:
        return False
    if value.get("decision") not in {"accept", "review", "reject"}:
        return False
    if value.get("expected_certainty_change") not in {"downgrade", "stay"}:
        return False
    list_fields = (
        "removed_fact_concepts",
        "required_missing_concepts",
        "forbidden_specific_claims",
        "reasons",
    )
    if not all(
        isinstance(value.get(field), list)
        and all(isinstance(item, str) and item.strip() for item in value[field])
        for field in list_fields
    ):
        return False
    return isinstance(value.get("allowed_conclusion_scope"), str) and bool(
        value["allowed_conclusion_scope"].strip()
    )


def build_user_prompt(candidate: dict[str, Any], teacher: dict[str, Any]) -> str:
    payload = {
        "mask_type": candidate["mask_type"],
        "full_case": candidate["original_case_text"],
        "masked_case": candidate["case_text"],
        "removed_facts": [item["span"] for item in candidate["removed_spans"]],
        "full_target": candidate["original_target"],
        "masked_target": teacher["parsed_output"],
    }
    return "请审核以下反事实样本：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["source_id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok" and row.get("source_id")
    }


def call_judge(
    client: Any,
    candidate: dict[str, Any],
    teacher: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    last_error = "unknown_error"
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(candidate, teacher)},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                timeout=args.timeout,
            )
            choice = response.choices[0]
            text = choice.message.content or ""
            parsed = extract_first_json_object(text)
            if valid_judgment(parsed):
                return {
                    "source_id": candidate["source_id"],
                    "variant_id": candidate["variant_id"],
                    "pair_id": candidate["pair_id"],
                    "judge_model": args.model,
                    "judge_version": JUDGE_VERSION,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "status": "ok",
                    "attempt": attempt,
                    "finish_reason": getattr(choice, "finish_reason", None),
                    "judge_text": text,
                    "parsed_judgment": parsed,
                }
            last_error = "invalid_judgment_schema"
        except Exception as exc:  # provider-specific exceptions vary
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    return {
        "source_id": candidate["source_id"],
        "variant_id": candidate["variant_id"],
        "pair_id": candidate["pair_id"],
        "judge_model": args.model,
        "judge_version": JUDGE_VERSION,
        "status": "judge_error",
        "error": last_error,
    }


def main() -> None:
    load_environment_file()
    args = parse_args()
    candidates = {row["source_id"]: row for row in read_jsonl(args.candidates)}
    teachers = {
        row["source_id"]: row
        for row in read_jsonl(args.teacher)
        if row.get("status") == "ok" and isinstance(row.get("parsed_output"), dict)
    }
    completed = load_completed_ids(Path(args.output))
    pending_ids = [source_id for source_id in candidates if source_id in teachers and source_id not in completed]
    if args.limit > 0:
        pending_ids = pending_ids[: args.limit]

    if args.dry_run:
        previews = [
            {
                "source_id": source_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(candidates[source_id], teachers[source_id]),
                    },
                ],
            }
            for source_id in pending_ids
        ]
        Path(args.preview_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.preview_output).open("w", encoding="utf-8") as handle:
            for row in previews:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Dry run wrote {len(previews)} requests to {args.preview_output}")
        return

    api_key = os.environ.get(args.api_key_env) or os.environ.get("TEACHER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            f"Missing judge API key: set {args.api_key_env}, TEACHER_API_KEY, or OPENAI_API_KEY"
        )
    if not args.model:
        raise SystemExit("Missing judge model: set JUDGE_MODEL or pass --model")

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key, "timeout": args.timeout}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = OpenAI(**client_kwargs)

    output_path = Path(args.output)
    failed_path = Path(args.failed_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    ok_count = failed_count = consecutive_failed = 0
    started = time.monotonic()
    with output_path.open("a", encoding="utf-8") as success_handle, failed_path.open(
        "a", encoding="utf-8"
    ) as failed_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(call_judge, client, candidates[source_id], teachers[source_id], args): source_id
            for source_id in pending_ids
        }
        for index, future in enumerate(as_completed(future_to_id), start=1):
            result = future.result()
            with lock:
                if result.get("status") == "ok":
                    success_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    success_handle.flush()
                    ok_count += 1
                    consecutive_failed = 0
                else:
                    failed_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    failed_handle.flush()
                    failed_count += 1
                    consecutive_failed += 1
            if consecutive_failed > MAX_CONSECUTIVE_FAILURES:
                for pending_future in future_to_id:
                    pending_future.cancel()
                raise SystemExit(f"Stopped after {consecutive_failed} consecutive judge failures")
            if index % 20 == 0 or index == len(pending_ids):
                print(
                    f"Judged {index}/{len(pending_ids)}: ok={ok_count}, failed={failed_count}, "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
    print(f"Finished: ok={ok_count}, failed={failed_count}, skipped={len(completed)}")


if __name__ == "__main__":
    main()
