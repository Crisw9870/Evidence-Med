#!/usr/bin/env python3
"""Shared, dependency-free helpers for the local CMB evaluation suite."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
DEFAULT_CMB_ROOT = WORKSPACE / "CMB"
DEFAULT_EXAM_TEST = (
    DEFAULT_CMB_ROOT / "CMB-Exam/CMB-test/CMB-test-choice-question-merge.json"
)
DEFAULT_EXAM_VAL = DEFAULT_CMB_ROOT / "CMB-Exam/CMB-val/CMB-val-merge.json"
DEFAULT_EXAM_TRAIN = DEFAULT_CMB_ROOT / "CMB-Exam/CMB-train/CMB-train-merge.json"
DEFAULT_TEST_ANSWERS = DEFAULT_CMB_ROOT / "CMB-Exam/CMB-test/CMB-test-choice-answer.json"
DEFAULT_CLIN = DEFAULT_CMB_ROOT / "CMB-Clin/CMB-Clin-qa.json"

CHOICE_PATTERN = re.compile(r"[A-F]", re.IGNORECASE)
ANSWER_PATTERNS = (
    re.compile(
        r"(?:最终答案|正确答案|答案|选择|选项)\s*(?:是|为|应为|：|:)?\s*"
        r"([A-F](?:(?:\s*[、,，/和及]\s*|\s+)?[A-F])*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|\n)\s*(?:答\s*[：:]\s*)?"
        r"[\[（(]?([A-F](?:(?:\s*[、,，/和及]\s*|\s+)?[A-F])*)[\]）)]?"
        r"(?:\s*[。.]?)\s*(?:$|\n)",
        re.IGNORECASE,
    ),
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    value = load_json(path)
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"expected a JSON list of objects: {path}")
    return value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def normalized_choice(value: Any, valid_keys: Iterable[str] = tuple("ABCDEF")) -> str | None:
    valid = {str(key).upper() for key in valid_keys}
    choices = sorted({item.upper() for item in CHOICE_PATTERN.findall(str(value or ""))})
    if not choices or not set(choices).issubset(valid):
        return None
    return "".join(choices)


def extract_choice(
    text: str, valid_keys: Iterable[str], question_type: str
) -> tuple[str | None, str]:
    """Extract a final option set without mining arbitrary letters from rationale text."""

    valid = {str(key).upper() for key in valid_keys}
    source = str(text or "").strip().upper()
    if not source:
        return None, "empty"

    compact = re.sub(r"[\s、,，/和及\[\]（）()]", "", source)
    compact = compact.rstrip("。.")
    if compact and set(compact).issubset(valid):
        answer = "".join(sorted(set(compact)))
        if "单项" in question_type and len(answer) != 1:
            return None, "multiple_for_single"
        return answer, "exact"

    leading = re.match(r"^\s*([A-F]+)\s*[.．。:：)）]", source)
    if leading:
        answer = normalized_choice(leading.group(1), valid)
        if answer and "单项" in question_type and len(answer) != 1:
            return None, "multiple_for_single"
        if answer:
            return answer, "leading_option"

    candidates: list[str] = []
    for pattern in ANSWER_PATTERNS:
        for match in pattern.finditer(source):
            answer = normalized_choice(match.group(1), valid)
            if answer:
                candidates.append(answer)
    if not candidates:
        return None, "no_answer_pattern"
    answer = candidates[-1]
    if "单项" in question_type and len(answer) != 1:
        return None, "multiple_for_single"
    return answer, "answer_pattern"


def exam_item_id(row: dict[str, Any], index: int) -> str:
    return str(row.get("id", index + 1))


def format_exam_prompt(row: dict[str, Any], *, cot: bool = False) -> str:
    options = row.get("option")
    if not isinstance(options, dict) or not options:
        raise ValueError("exam item has no options")
    option_lines = "\n".join(f"{key}. {value}" for key, value in options.items())
    if cot:
        instruction = "请分析各选项，并在回答最后单独一行写“最终答案：选项字母”。"
    else:
        instruction = "不要分析或解释，只输出答案选项字母；多选题按字母顺序连续输出。"
    return (
        f"以下是中国{row.get('exam_type', '')}中{row.get('exam_class', '')}考试的"
        f"一道{row.get('question_type', '选择题')}。{instruction}\n\n"
        f"{row.get('question', '')}\n{option_lines}\n\n答案："
    )


def stable_int(*parts: Any, seed: int = 0) -> int:
    payload = "\x1f".join(str(part) for part in (*parts, seed))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def extract_first_json_object(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    try:
        value = json.loads(source)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = source.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(source)):
            char = source[index]
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
                        value = json.loads(source[start : index + 1])
                        return value if isinstance(value, dict) else None
                    except json.JSONDecodeError:
                        break
        start = source.find("{", start + 1)
    return None
