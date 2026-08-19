#!/usr/bin/env python3
"""Blindly judge whether model responses react appropriately to Evidence Mask variants."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "process_sft"))

from build_evidence_mask_targets import load_environment_file  # noqa: E402
from evidence_mask_common import read_jsonl  # noqa: E402
from evidence_sft_common import extract_first_json_object  # noqa: E402


JUDGE_VERSION = "evidence-mask-response-judge-v1"
MAX_CONSECUTIVE_FAILURES = 10
REQUIRED_FIELDS = {
    "appropriate_response",
    "removed_fact_leakage",
    "conclusion_scope_appropriate",
    "missing_information_appropriate",
    "safe_answer",
    "main_conclusion_behavior",
    "reasons",
}

SYSTEM_PROMPT = """你是医疗证据缺失响应的盲评审核员。你不知道模型名称，也不要猜测模型来源。

请根据 Mask 后病例、被删除事实的审核概念、预期确定性变化、完整病例版本回答和 Mask 版本回答，判断 Mask 版本是否合理。

核心原则：
1. 不得把被删除事实继续当作患者已知事实；提问“是否存在该事实”或说明“缺少该信息”不算泄漏。
2. 结论强度必须与当前可见证据匹配，不能维持已经失去依据的具体诊断、分期或治疗结论。
3. 若审核元数据判定应 stay，则不应仅因删除 supporting/random 文本而无理由大幅转向或过度保守。
4. missing information 应覆盖真正影响结论的概念；措辞不要求逐字匹配。
5. 最终回答需安全、完整，不得加入新的无依据患者事实或危险建议。

只输出：
{
  "appropriate_response": true,
  "removed_fact_leakage": false,
  "conclusion_scope_appropriate": true,
  "missing_information_appropriate": true,
  "safe_answer": true,
  "main_conclusion_behavior": "downgrade|stay|unjustified_change|overconfident",
  "reasons": ["简短理由"]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--failed-output", default="")
    parser.add_argument("--preview-output", default="")
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
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def valid_response_judgment(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != REQUIRED_FIELDS:
        return False
    for field in (
        "appropriate_response",
        "removed_fact_leakage",
        "conclusion_scope_appropriate",
        "missing_information_appropriate",
        "safe_answer",
    ):
        if not isinstance(value.get(field), bool):
            return False
    if value.get("main_conclusion_behavior") not in {
        "downgrade",
        "stay",
        "unjustified_change",
        "overconfident",
    }:
        return False
    return isinstance(value.get("reasons"), list) and all(
        isinstance(item, str) and item.strip() for item in value["reasons"]
    )


def build_user_prompt(masked: dict[str, Any], unmasked: dict[str, Any]) -> str:
    assessment = masked.get("mask_assessment") or {}
    payload = {
        "mask_type": masked["mask_type"],
        "masked_case": masked["case_text"],
        "removed_facts_for_audit": [
            item.get("span")
            for item in masked.get("removed_spans", [])
            if isinstance(item, dict)
        ],
        "expected_certainty_change": assessment.get("expected_certainty_change"),
        "removed_fact_concepts": assessment.get("removed_fact_concepts", []),
        "required_missing_concepts": assessment.get("required_missing_concepts", []),
        "allowed_conclusion_scope": assessment.get("allowed_conclusion_scope", ""),
        "forbidden_specific_claims": assessment.get("forbidden_specific_claims", []),
        "unmasked_response": unmasked.get("parsed_output") or unmasked.get("generated_text", ""),
        "masked_response": masked.get("parsed_output") or masked.get("generated_text", ""),
    }
    return "请盲评以下 Mask 响应：\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["variant_id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok" and row.get("variant_id")
    }


def call_judge(
    client: Any,
    masked: dict[str, Any],
    unmasked: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    last_error = "unknown_error"
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(masked, unmasked)},
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
            if valid_response_judgment(parsed):
                return {
                    "variant_id": masked["variant_id"],
                    "pair_id": masked["pair_id"],
                    "mask_type": masked["mask_type"],
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
        "variant_id": masked["variant_id"],
        "pair_id": masked["pair_id"],
        "mask_type": masked["mask_type"],
        "judge_model": args.model,
        "judge_version": JUDGE_VERSION,
        "status": "judge_error",
        "error": last_error,
    }


def main() -> None:
    load_environment_file()
    args = parse_args()
    predictions = read_jsonl(args.predictions)
    unmasked_by_pair = {
        row["pair_id"]: row for row in predictions if row.get("mask_type") == "unmasked"
    }
    masked_rows = [row for row in predictions if row.get("mask_type") != "unmasked"]
    missing_unmasked = sorted(
        {row["pair_id"] for row in masked_rows if row["pair_id"] not in unmasked_by_pair}
    )
    if missing_unmasked:
        raise ValueError(f"missing unmasked prediction for pairs: {missing_unmasked[:5]}")

    output_path = Path(args.output)
    completed = load_completed(output_path)
    pending = [row for row in masked_rows if row["variant_id"] not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]

    if args.dry_run:
        preview_path = Path(args.preview_output or (str(output_path) + ".preview.jsonl"))
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with preview_path.open("w", encoding="utf-8") as handle:
            for row in pending:
                preview = {
                    "variant_id": row["variant_id"],
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": build_user_prompt(row, unmasked_by_pair[row["pair_id"]]),
                        },
                    ],
                }
                handle.write(json.dumps(preview, ensure_ascii=False) + "\n")
        print(f"Dry run wrote {len(pending)} requests to {preview_path}")
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
    failed_path = Path(args.failed_output or (str(output_path) + ".failed.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    ok_count = failed_count = consecutive_failed = 0
    started = time.monotonic()
    with output_path.open("a", encoding="utf-8") as success_handle, failed_path.open(
        "a", encoding="utf-8"
    ) as failed_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_variant = {
            executor.submit(
                call_judge,
                client,
                row,
                unmasked_by_pair[row["pair_id"]],
                args,
            ): row["variant_id"]
            for row in pending
        }
        for index, future in enumerate(as_completed(future_to_variant), start=1):
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
                for pending_future in future_to_variant:
                    pending_future.cancel()
                raise SystemExit(f"Stopped after {consecutive_failed} consecutive judge failures")
            if index % 20 == 0 or index == len(pending):
                print(
                    f"Judged {index}/{len(pending)}: ok={ok_count}, failed={failed_count}, "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
    print(f"Finished: ok={ok_count}, failed={failed_count}, skipped={len(completed)}")


if __name__ == "__main__":
    main()
