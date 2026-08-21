#!/usr/bin/env python3
"""Generate counterfactual Evidence-SFT targets from masked cases only."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROCESS_SFT = ROOT / "process_sft"
if str(PROCESS_SFT) not in sys.path:
    sys.path.insert(0, str(PROCESS_SFT))

from evidence_mask_common import read_jsonl  # noqa: E402
from evidence_sft_common import extract_first_json_object  # noqa: E402


PROMPT_VERSION = "evidence-mask-v1"
MAX_CONSECUTIVE_FAILURES = 10

SYSTEM_PROMPT = """你是中文医疗证据约束数据构造专家。请只依据当前可见的病例描述生成一个合法 JSON 对象。

重要约束：
1. 当前病例描述是唯一事实来源。你看不到、也不得猜测任何被删除或未提供的信息。
2. 不要提及“遮蔽”“删除”“Mask”“原病例”或数据构造过程。
3. evidence 中每个 span 必须是当前病例描述中的连续原文，逐字复制，不得改写或拼接。
4. 如果当前可见证据不足，必须降低结论强度，并在 missing_information 中说明真正影响判断的缺失信息。
5. 不得把常见医学知识、可能的检查结果、诊断、剂量或阈值当成患者已经存在的事实。
6. critical 仅标记：单独删除后会明显改变主要判断、回答范围或 evidence_sufficiency 的最小原子证据。
7. final_answer 必须直接、安全地回答用户，但不能维持当前证据已经不支持的确定诊断、分期或治疗结论。

只输出以下 8 个字段，不要输出额外字段：
{
  "task_type": "diagnostic_reasoning|confirmed_management",
  "query_intent": ["用户实际询问的目标"],
  "evidence_sufficiency": "sufficient|partial|insufficient|conflicting",
  "evidence": [
    {"id": "E1", "span": "当前病例中的连续原文", "importance": "critical|supporting", "role": "该事实支持或限制什么判断"}
  ],
  "critical_evidence_ids": ["E1"],
  "missing_information": ["补充后会实质改变判断的信息"],
  "clinical_reasoning": "说明当前可见证据能支持什么、不能支持什么",
  "final_answer": "面向用户的完整、安全回答"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/evidence_mask/v1/00_candidates.jsonl")
    parser.add_argument("--output", default="data/evidence_mask/v1/01_teacher_raw.jsonl")
    parser.add_argument("--failed-output", default="data/evidence_mask/v1/01_teacher_failed.jsonl")
    parser.add_argument(
        "--preview-output",
        default="data/evidence_mask/v1/01_teacher_requests.preview.jsonl",
    )
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", ""))
    parser.add_argument(
        "--base-url",
        default=os.environ.get("TEACHER_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL", ""),
    )
    parser.add_argument("--api-key-env", default="TEACHER_API_KEY")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_environment_file(path: Path | str | None = None) -> None:
    env_path = Path(path or ".teacher_env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def build_user_prompt(candidate: dict[str, Any]) -> str:
    payload = {
        "task_type_hint": candidate["task_type_hint"],
        "case_text": candidate["case_text"],
    }
    return (
        "请根据当前可见病例生成证据约束 JSON。task_type_hint 仅供参考，可以纠正。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        str(row["source_id"])
        for row in read_jsonl(path)
        if row.get("status") == "ok" and row.get("source_id")
    }


def build_raw_record(
    candidate: dict[str, Any],
    model: str,
    teacher_text: str,
    parsed: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "source_id": candidate["source_id"],
        "variant_id": candidate["variant_id"],
        "pair_id": candidate["pair_id"],
        "parent_source_id": candidate["parent_source_id"],
        "split": candidate["split"],
        "category": candidate.get("category", "未标注医学主题"),
        "task_type_hint": candidate["task_type_hint"],
        "mask_type": candidate["mask_type"],
        "case_text": candidate["case_text"],
        "teacher_model": model,
        "prompt_version": PROMPT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if parsed is not None else "parse_error",
        "teacher_text": teacher_text,
        "parsed_output": parsed,
    }


def call_teacher(client: Any, candidate: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    last_error = "unknown_error"
    last_record: dict[str, Any] | None = None
    for attempt in range(1, args.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(candidate)},
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                timeout=args.timeout,
            )
            choice = response.choices[0]
            teacher_text = choice.message.content or ""
            parsed = extract_first_json_object(teacher_text)
            record = build_raw_record(candidate, args.model, teacher_text, parsed)
            record["attempt"] = attempt
            record["finish_reason"] = getattr(choice, "finish_reason", None)
            last_record = record
            if parsed is not None:
                return record
            last_error = "empty_teacher_content" if not teacher_text else "incomplete_or_invalid_json"
        except Exception as exc:  # provider-specific exceptions vary
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < args.max_retries:
            time.sleep(min(2 ** (attempt - 1), 8))
    if last_record is not None:
        last_record["error"] = last_error
        return last_record
    return {
        "source_id": candidate["source_id"],
        "variant_id": candidate["variant_id"],
        "status": "api_error",
        "teacher_model": args.model,
        "prompt_version": PROMPT_VERSION,
        "error": last_error,
    }


def main() -> None:
    load_environment_file()
    args = parse_args()
    candidates = read_jsonl(args.input)
    completed_ids = load_completed_ids(Path(args.output))
    pending = [row for row in candidates if row["source_id"] not in completed_ids]
    if args.limit > 0:
        pending = pending[: args.limit]

    if args.dry_run:
        preview_rows = [
            {
                "source_id": row["source_id"],
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                ],
            }
            for row in pending
        ]
        Path(args.preview_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.preview_output).open("w", encoding="utf-8") as handle:
            for row in preview_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Dry run wrote {len(preview_rows)} requests to {args.preview_output}")
        return

    api_key = os.environ.get(args.api_key_env) or (
        os.environ.get("OPENAI_API_KEY") if args.api_key_env != "OPENAI_API_KEY" else None
    )
    if not api_key:
        raise SystemExit(f"Missing teacher API key: set {args.api_key_env} or OPENAI_API_KEY")
    if not args.model:
        raise SystemExit("Missing teacher model: set TEACHER_MODEL or pass --model")

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
    ok_count = 0
    failed_count = 0
    consecutive_failed = 0
    started = time.monotonic()

    with output_path.open("a", encoding="utf-8") as success_handle, failed_path.open(
        "a", encoding="utf-8"
    ) as failed_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {
            executor.submit(call_teacher, client, row, args): row["source_id"]
            for row in pending
        }
        for index, future in enumerate(as_completed(future_to_id), start=1):
            record = future.result()
            with lock:
                if record.get("status") == "ok":
                    success_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success_handle.flush()
                    ok_count += 1
                    consecutive_failed = 0
                else:
                    failed_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    failed_handle.flush()
                    failed_count += 1
                    consecutive_failed += 1
            if consecutive_failed > MAX_CONSECUTIVE_FAILURES:
                for pending_future in future_to_id:
                    pending_future.cancel()
                raise SystemExit(
                    f"Stopped after {consecutive_failed} consecutive teacher failures"
                )
            if index % 20 == 0 or index == len(pending):
                print(
                    f"Processed {index}/{len(pending)}: ok={ok_count}, failed={failed_count}, "
                    f"elapsed={time.monotonic() - started:.1f}s"
                )
    print(f"Finished: ok={ok_count}, failed={failed_count}, skipped={len(completed_ids)}")


if __name__ == "__main__":
    main()
