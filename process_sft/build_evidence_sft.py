#!/usr/bin/env python3
"""Distill structured, source-grounded medical answers from a strong teacher."""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evidence_sft_common import extract_first_json_object, iter_jsonl


PROMPT_VERSION = "evidence-sft-v2.2"
MAX_CONSECUTIVE_FAILURES = 10


SYSTEM_PROMPT = """你是中文医疗问答数据构造专家。你的任务是把低质量医疗病例问答重构为可验证、证据约束、适合后续反事实 Evidence Mask 的训练样本。

你的目标不是尽可能完整地摘录病例，而是构造高精度、低噪声、最小粒度、可独立验证和可独立删除的临床证据。

必须严格遵守以下规则：

1. case_text 是唯一的病例事实来源
- 所有患者事实判断都只能依据 case_text。
- evidence 中每个 span 必须是 case_text 中的连续原文片段，并逐字复制。
- 不得改写 evidence。
- 不得拼接多个不连续片段。
- 不得从 original_answer 中抽取 evidence。
- 不得补充 case_text 中没有出现的患者症状、诊断、检查结果、既往史、分期、并发症或用药史。
- 即使某个医学事实很常见，只要 case_text 没有明确给出，就不能作为患者事实使用。

original_answer 只是可能包含错误的低质量参考答案，仅用于帮助识别原回答的问题。
不得默认 original_answer 正确，也不得把其中新增的信息当作患者事实。

2. query_intent 表示“用户想问什么”
query_intent 应简洁概括用户真正要求回答的目标。

例如：
- “判断持续干咳的可能原因”
- “咨询下一步检查和治疗建议”
- “咨询已确诊肝硬化的一般治疗原则”
- “询问晚期肝硬化的一般预后”

query_intent 不是临床证据。
用户问题中出现的疾病、分期、治疗方式或预后范围，不代表患者本人一定具有该状态。

例如：
“肝硬化晚期一般能活几年”
只表示用户询问晚期肝硬化的预后，
不能据此判断患者本人已经处于晚期。

3. evidence 只能包含患者事实
Evidence 可以包括：
- 症状及其性质；
- 症状持续时间；
- 起病或诱发背景；
- 明确诊断；
- 检查结果；
- 既往史；
- 用药或治疗情况；
- 治疗反应；
- 与当前判断直接相关的暴露因素或人口学信息。

不要把“可能得什么病”“怎么治疗”“能活几年”等问题本身作为 evidence。

4. Evidence 必须遵循“最小原子证据”原则
每条 evidence 应尽量只表达一个可以独立删除、独立解释的临床事实。

如果一句原文包含多个作用不同的事实，应拆成多个连续 span。

例如原文：
“感冒好了以后，一直干咳，有两个月了，但是无其他症状。”

不要优先抽成：
E1 = “感冒好了以后，一直干咳，有两个月了，但是无其他症状。”

应优先根据临床作用拆分，例如：
E1 = “感冒好了以后”
E2 = “一直干咳，有两个月了”
E3 = “但是无其他症状”

其中每个 span 仍然必须逐字存在于 case_text 中。

允许从同一句话抽取多个互不重叠的连续片段。
除非语义无法独立成立，否则尽量避免 evidence 之间重叠。

5. critical evidence 必须使用“最小充分 span”
如果一段较长文本中只有其中一小部分真正决定主要判断，
critical evidence 应只保留能够产生该作用的最小连续原文片段。

不要为了保持句子完整而扩大 critical span。

例如：
如果真正决定判断的是“持续干咳两个月”，
则不要因为它和“感冒后”“无其他症状”出现在同一句话中，
就把整句话一起标为 critical。

这条规则非常重要，因为 critical evidence 将用于后续 Evidence Mask。

6. Evidence 不是病例摘要
只保留对当前用户问题具有直接作用的信息。

该信息必须至少影响以下一项：
- 诊断或鉴别诊断方向；
- 治疗或管理建议；
- 风险或预后判断；
- 是否需要进一步检查；
- 回答的确定性；
- 回答能够达到的范围。

如果删除某条信息后，对当前回答几乎没有影响，则不要输出该 evidence。

年龄、性别等人口学信息只有在当前问题中确实会实质影响判断时才保留。
不要为了完整性机械抽取所有病例信息。

宁可少抽，不要加入弱相关证据。

7. evidence importance 只允许 critical 或 supporting

critical：
假设只删除这一条最小 evidence span，
而 case_text 中其他信息全部保持不变，
主要诊断方向、主要回答范围或 evidence_sufficiency 会明显改变。

supporting：
该信息会帮助回答、缩小方向或补充边界，
但删除后主要诊断方向、主要回答范围和 evidence_sufficiency 基本不变。

不要输出 irrelevant evidence。

8. critical 使用保守标准
critical 是后续反事实 Evidence Mask 的高精度标签，因此宁缺毋滥。

判断某条 evidence 是否 critical 时，必须做如下反事实检查：

“如果只删除这一条 evidence，其他病例事实完全保留，
当前回答最重要的结论、回答范围或不确定性等级是否会明显变化？”

只有明确“会明显变化”时才能标记 critical。

以下情况通常只属于 supporting：
- 让已有判断稍微更有把握；
- 提供背景信息；
- 只影响一个次要鉴别诊断；
- 只影响一个次要管理建议；
- 仅增强已有结论但不改变主要回答。

如果无法明确判断为 critical，优先标记 supporting。

不要为了保证每条样本都有 critical evidence 而强行标注。
允许：
"critical_evidence_ids": []

9. critical_evidence_ids 必须与 importance 完全一致
- 只能包含 importance="critical" 的 evidence ID。
- supporting evidence 不能进入 critical_evidence_ids。
- query_intent 永远不能进入 critical_evidence_ids。

10. role 只能描述 evidence 的直接作用
role 必须严格限制推断强度。

不得仅凭弱证据使用：
- “排除”
- “证实”
- “证明”
- “确定”
- “明确不是”
- “因此一定是”
等强结论。

除非 case_text 中存在能够直接支持该结论的明确诊断或检查结果。

例如：
“使用多种抗生素后无效”

可以写：
“说明既往自行使用多种抗生素后未见明显改善，提示需要重新评估病因和处理策略。”

不要写：
“排除了感染性疾病。”

对于“无其他症状”等描述，只能说明当前病例未报告相关伴随症状，
不能因此声称已经排除严重疾病。

role 应尽量短、具体、可从该 span 本身核验。

11. evidence_sufficiency 衡量患者事实对“用户请求粒度”的覆盖程度

sufficient：
已有患者事实足以支持用户所问的主要结论。
即使仍有可补充信息，这些信息通常不会显著改变主要判断或回答范围。

partial：
已有患者事实能够支持部分结论、方向性判断、一般管理原则或下一步建议，
但不能回答用户请求中的一个或多个关键部分，
或者不足以进行个体化诊断、治疗方案选择、风险分层或预后估计。

insufficient：
缺少最基本的患者事实，
连有针对性的方向性判断或针对性建议都无法由 case_text 支持。

conflicting：
case_text 内部存在会实质改变判断的相互冲突信息。

注意：
- 不要仅因为在线回答不能替代面诊就机械标记 insufficient。
- 不要因为还能介绍一些一般医学知识就机械标记 sufficient。
- 如果只能回答一般方向，但不能回答用户要求的关键个体化部分，通常应标记 partial。

12. missing_information 只保留高价值缺失信息
只填写如果补充后会明显改变以下至少一项的信息：
- 诊断或鉴别方向；
- 治疗建议；
- 风险分层；
- 预后判断；
- evidence_sufficiency。

不要机械罗列完整的“年龄、性别、病史、体检、化验、影像学”等检查清单。

优先保留最有信息价值的少量项目。
通常控制在 0～5 项。

13. task_type

diagnostic_reasoning：
用户主要根据症状、病史或检查结果询问：
- 可能是什么疾病；
- 症状可能由什么原因引起；
- 鉴别诊断；
- 下一步应该检查什么。

confirmed_management：
疾病、术后状态或治疗状态已经明确，
用户主要咨询：
- 治疗；
- 用药；
- 随访；
- 并发症；
- 风险；
- 预后；
- 日常管理。

task_type_hint 仅供参考。
如果与 case_text 实际任务不符，可以纠正。

14. clinical_reasoning 必须是简洁的“证据边界摘要”
clinical_reasoning 不是完整诊疗思维链，也不是医学知识综述。

只需要说明：
- 已有患者事实能够支持什么；
- 目前还不能支持什么；
- 为什么回答需要保持当前程度的不确定性。

推荐形式：
“已有证据能够支持……；但由于缺少……，目前不能进一步确定……。”

不得输出冗长隐藏思维过程。

15. final_answer 必须独立重构
original_answer 只是低质量参考，可能包含错误。

final_answer 必须：
- 直接回答 query_intent；
- 结论强度与 evidence_sufficiency 一致；
- 不把一般医学知识伪装成患者个体化结论；
- 不虚构 case_text 中不存在的患者情况；
- 对证据不足的部分明确说明边界；
- 必要时给出合理的下一步就医或检查建议；
- 纠正 original_answer 中未经证实的治疗、危险建议、事实错误和过度诊断；
- 不照抄 original_answer。

16. 输出前执行一致性自检
返回 JSON 前必须检查：

A. 每个 evidence.span 是否都能在 case_text 中逐字找到？
B. evidence 是否全部是患者事实，而不是用户的问题？
C. 每条 evidence 是否尽量只包含一个独立临床事实？
D. 是否存在一个 evidence 可以进一步拆成作用不同的更小 span？如果可以，应优先拆分。
E. critical evidence 是否已经缩小到能够产生关键作用的最小连续 span？
F. 是否存在对当前回答几乎没有作用的弱 evidence？如果有则删除。
G. role 是否超出了该 span 本身能够支持的结论？如果有则降低措辞。
H. 每条 critical evidence 单独删除后是否真的会明显改变主要结论、回答范围或 sufficiency？
I. critical_evidence_ids 是否只包含 critical evidence？
J. evidence_sufficiency 是否与 clinical_reasoning 和 final_answer 的确定性一致？
K. final_answer 是否错误地把 missing_information 当成患者已有事实？

只返回一个合法 JSON 对象。
不要使用 Markdown。
不要输出 JSON 之外的任何内容。

输出结构必须严格为：

{
  "task_type": "diagnostic_reasoning|confirmed_management",

  "query_intent": [
    "用户实际询问的目标1",
    "用户实际询问的目标2"
  ],

  "evidence_sufficiency": "sufficient|partial|insufficient|conflicting",

  "evidence": [
    {
      "id": "E1",
      "span": "case_text中的最小连续患者事实原文片段",
      "importance": "critical|supporting",
      "role": "该患者事实直接支持或限制什么判断"
    }
  ],

  "critical_evidence_ids": ["E1"],

  "missing_information": [
    "补充后会实质改变当前判断或建议的信息"
  ],

  "clinical_reasoning": "简洁说明已有证据能支持什么、当前不能支持什么",

  "final_answer": "直接面向用户的完整、安全回答"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/new-evidence/new_candidate.jsonl")
    parser.add_argument("--output", default="data/new-evidence/01_teacher_raw.jsonl")
    parser.add_argument("--failed-output", default="data/new-evidence/01_teacher_failed.jsonl")
    parser.add_argument("--preview-output", default="data/new-evidence/01_teacher_requests.preview.jsonl")
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", ""))
    parser.add_argument(
        "--base-url", default=os.environ.get("TEACHER_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
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


def build_user_prompt(candidate: dict[str, Any]) -> str:
    payload = {
        "task_type_hint": candidate["task_type_hint"],
        "case_text": candidate["case_text"],
        "original_answer": candidate["original_answer"],
    }
    return (
        "请根据下列输入生成证据约束样本。task_type_hint 只是启发，如判断有误可以纠正。\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


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
        key = match.group(1)
        value = match.group(2).strip()
        if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for _, record in iter_jsonl(path):
        if record.get("status") == "ok" and record.get("source_id"):
            completed.add(record["source_id"])
    return completed


def build_raw_record(
    candidate: dict[str, Any], model: str, teacher_text: str, parsed: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "source_id": candidate["source_id"],
        "source_file": candidate.get("source_file"),
        "source_line": candidate.get("source_line"),
        "split": candidate["split"],
        "category": candidate.get("category", "未标注医学主题"),
        "task_type_hint": candidate["task_type_hint"],
        "case_text": candidate["case_text"],
        "original_answer": candidate["original_answer"],
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
            if attempt < args.max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
        except Exception as exc:  # provider-specific exceptions vary
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < args.max_retries:
                time.sleep(min(2 ** (attempt - 1), 8))
    if last_record is not None:
        last_record["error"] = last_error
        return last_record
    return {
        "source_id": candidate["source_id"],
        "status": "api_error",
        "teacher_model": args.model,
        "prompt_version": PROMPT_VERSION,
        "error": last_error,
    }


def main() -> None:
    load_environment_file()
    args = parse_args()
    candidates = [record for _, record in iter_jsonl(args.input)]
    completed_ids = load_completed_ids(Path(args.output))
    pending = [row for row in candidates if row["source_id"] not in completed_ids]
    if args.limit > 0:
        pending = pending[: args.limit]

    if args.dry_run:
        preview_rows = [
            {
                "source_id": row["source_id"],
                "model": args.model or "<TEACHER_MODEL>",
                "prompt_version": PROMPT_VERSION,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(row)},
                ],
            }
            for row in pending
        ]
        output_path = Path(args.preview_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in preview_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Dry run wrote {len(preview_rows)} requests to {output_path}")
        return

    api_key = os.environ.get(args.api_key_env) or (
        os.environ.get("OPENAI_API_KEY") if args.api_key_env != "OPENAI_API_KEY" else None
    )
    if not api_key:
        raise SystemExit(
            f"Missing teacher API key. Set {args.api_key_env} (recommended) or OPENAI_API_KEY."
        )
    if not args.model:
        raise SystemExit("Missing teacher model. Set TEACHER_MODEL or pass --model.")

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
    consecutive_failed_count = 0
    stop_message: str | None = None
    processing_started_at = time.monotonic()

    with output_path.open("a", encoding="utf-8") as success_handle, failed_path.open(
        "a", encoding="utf-8"
    ) as failed_handle, ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_id = {executor.submit(call_teacher, client, row, args): row["source_id"] for row in pending}
        for index, future in enumerate(as_completed(future_to_id), start=1):
            record = future.result()
            with lock:
                if record.get("status") == "ok":
                    success_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    success_handle.flush()
                    ok_count += 1
                    consecutive_failed_count = 0
                else:
                    failed_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    failed_handle.flush()
                    failed_count += 1
                    consecutive_failed_count += 1
            if consecutive_failed_count > MAX_CONSECUTIVE_FAILURES:
                cancelled_count = sum(
                    pending_future.cancel()
                    for pending_future in future_to_id
                    if pending_future is not future
                )
                stop_message = (
                    f"Stopped after {consecutive_failed_count} consecutive failures; "
                    f"cancelled {cancelled_count} pending requests."
                )
                print(stop_message)
                break
            if index % 20 == 0 or index == len(pending):
                elapsed_seconds = time.monotonic() - processing_started_at
                print(
                    f"Processed {index}/{len(pending)}: ok={ok_count}, failed={failed_count}, "
                    f"elapsed={elapsed_seconds:.1f}s"
                )

    if stop_message is not None:
        raise SystemExit(1)
    print(f"Finished: ok={ok_count}, failed={failed_count}, skipped={len(completed_ids)}")


if __name__ == "__main__":
    main()
