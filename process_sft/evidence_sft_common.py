"""Shared utilities for the evidence-grounded SFT data pipeline."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Iterator


TASK_TYPES = {"diagnostic_reasoning", "confirmed_management"}
SUFFICIENCY_LEVELS = {"sufficient", "partial", "insufficient", "conflicting"}
EVIDENCE_IMPORTANCE_LEVELS = {"critical", "supporting"}
EVIDENCE_SCHEMA_VERSION = "evidence-sft-v2.2"

DEPARTMENT_RE = re.compile(r"^\s*([\u4e00-\u9fffA-Za-z0-9/·-]{1,20}(?:科|科学|医学))\s*[：:]\s*")
SPACE_RE = re.compile(r"\s+")
NON_WORD_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")

PATIENT_SIGNALS = (
    "患者", "患儿", "病人", "本人", "我", "孩子", "宝宝", "父亲", "母亲", "爸爸", "妈妈",
    "老人", "男性", "女性", "孕妇", "术后",
)
TIME_SIGNALS = (
    "天", "周", "月", "年", "小时", "近日", "最近", "长期", "反复", "突然", "逐渐", "持续",
)
SYMPTOM_SIGNALS = (
    "疼", "痛", "发热", "低热", "高热", "咳", "痒", "肿", "出血", "头晕", "乏力", "恶心",
    "呕吐", "腹泻", "便秘", "不适", "麻木", "视力", "气短", "胸闷", "心悸", "皮疹", "消瘦",
    "尿", "月经", "分泌物", "呼吸", "食欲", "失眠", "斜视", "复视",
)
EXAM_SIGNALS = (
    "检查", "化验", "检验", "结果", "提示", "显示", "诊断", "确诊", "影像", "CT", "MRI", "B超",
    "彩超", "X线", "血压", "血糖", "指标", "阳性", "阴性", "病理", "超声", "心电图",
)
QUESTION_SIGNALS = (
    "怎么办", "怎么治疗", "如何治疗", "怎么回事", "什么病", "可能是", "是否", "需要", "应该",
    "请问", "治疗", "手术", "用药", "什么检查", "如何检查", "怎么检查", "如何诊断", "注意什么",
    "会不会", "严重吗", "想得到怎样的帮助", "？", "?",
)
MANAGEMENT_SIGNALS = (
    "确诊", "诊断为", "查出", "患有", "得了", "术后", "手术后", "治疗后", "服用", "用药",
    "复查", "如何治疗", "怎么治疗", "治疗方法", "注意什么", "康复", "预后",
)
CONFIRMED_SIGNALS = (
    "已经确诊", "已确诊", "被确诊", "医生诊断", "诊断为", "检查结果是", "检查结果为", "结果显示",
    "结果提示", "查出", "患有", "得了", "术后", "手术后", "治疗后", "医生说是", "检查说是", "医院检查说",
)
DIAGNOSTIC_SIGNALS = (
    "什么病", "可能是", "怎么回事", "是否患", "是不是", "诊断", "病因", "是什么原因", "可能患",
)
KNOWLEDGE_ONLY_RE = re.compile(
    r"^(?:请问|咨询一下|想知道)?(?:什么是|介绍一下|解释一下|简述|何谓|定义)"
)
PROVIDER_RECOMMENDATION_RE = re.compile(r"(?:哪家|哪个|最好的|最好)医院|哪里治疗|去哪(?:里)?治疗")


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield non-empty JSON objects with their one-based line numbers."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            yield line_number, value


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    return count


def extract_question_answer(record: dict[str, Any]) -> tuple[str, str]:
    """Extract the first user question and first following assistant answer."""
    question = ""
    answer = ""
    for turn in record.get("conversations", []):
        if not isinstance(turn, dict):
            continue
        role = turn.get("from") or turn.get("role")
        value = str(turn.get("value") if "value" in turn else turn.get("content", "")).strip()
        if not question and role in {"human", "user"}:
            question = value
        elif question and not answer and role in {"gpt", "assistant"}:
            answer = value
            break
    return question, answer


def split_department(question: str) -> tuple[str, str]:
    match = DEPARTMENT_RE.match(question)
    if not match:
        return "未标注医学主题", question.strip()
    return match.group(1), question[match.end():].strip()


def canonicalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = NON_WORD_RE.sub("", text)
    return text


def stable_source_id(question: str) -> str:
    digest = hashlib.sha256(canonicalize_text(question).encode("utf-8")).hexdigest()[:16]
    return f"medical_{digest}"


def deterministic_bucket(source_id: str, seed: int = 42) -> int:
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 100


def assign_split(source_id: str, seed: int = 42) -> str:
    bucket = deterministic_bucket(source_id, seed)
    if bucket < 85:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def classify_task(question: str) -> str:
    if contains_any(question, CONFIRMED_SIGNALS):
        return "confirmed_management"
    management = sum(term in question for term in MANAGEMENT_SIGNALS)
    diagnostic = sum(term in question for term in DIAGNOSTIC_SIGNALS)
    if management > diagnostic:
        return "confirmed_management"
    return "diagnostic_reasoning"


def score_case_candidate(question: str, answer: str) -> tuple[int, list[str]]:
    """Return a conservative heuristic score and human-readable reasons."""
    _, case_text = split_department(question)
    reasons: list[str] = []

    if len(case_text) < 35 or len(case_text) > 1800:
        return 0, ["question_length_out_of_range"]
    if len(answer.strip()) < 20:
        return 0, ["answer_too_short"]

    patient = contains_any(case_text, PATIENT_SIGNALS)
    symptom = contains_any(case_text, SYMPTOM_SIGNALS)
    examination = contains_any(case_text, EXAM_SIGNALS)
    temporal = contains_any(case_text, TIME_SIGNALS)
    question_intent = contains_any(case_text, QUESTION_SIGNALS)

    if KNOWLEDGE_ONLY_RE.search(case_text) and not (patient or examination):
        return 0, ["knowledge_only_question"]
    if PROVIDER_RECOMMENDATION_RE.search(case_text):
        return 0, ["provider_recommendation_question"]
    if not question_intent:
        return 0, ["missing_clinical_question_intent"]
    if not (patient or symptom or examination):
        return 0, ["missing_case_evidence"]

    score = 0
    for active, points, reason in (
        (patient, 2, "patient_context"),
        (symptom, 2, "symptom_context"),
        (examination, 2, "exam_context"),
        (temporal, 1, "temporal_context"),
        (question_intent, 1, "clinical_question"),
        (len(case_text) >= 80, 1, "detailed_case"),
        (len(answer) >= 80, 1, "usable_reference_answer"),
    ):
        if active:
            score += points
            reasons.append(reason)
    return score, reasons


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from plain text or a fenced response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start:index + 1])
                    return value if isinstance(value, dict) else None
                except json.JSONDecodeError:
                    return None
    return None

