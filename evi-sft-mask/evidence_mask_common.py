"""Shared helpers for counterfactual Evidence Mask data construction."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PUNCTUATION = "，,。；;！？!?：:\n"
QUERY_MARKERS = (
    "请问",
    "想得到的帮助",
    "想问",
    "怎么办",
    "怎么治疗",
    "如何治疗",
    "是否",
    "是不是",
    "？",
    "?",
)
CLAUSE_RE = re.compile(rf"[^{re.escape(PUNCTUATION)}]+")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def stable_rank(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def evidence_by_importance(target: dict[str, Any], importance: str) -> list[dict[str, Any]]:
    evidence = target.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [
        item
        for item in evidence
        if isinstance(item, dict) and item.get("importance") == importance
    ]


def valid_evidence_offsets(case_text: str, evidence: Iterable[dict[str, Any]]) -> bool:
    intervals: list[tuple[int, int]] = []
    for item in evidence:
        start = item.get("start")
        end = item.get("end")
        span = item.get("span")
        if not isinstance(start, int) or not isinstance(end, int) or not isinstance(span, str):
            return False
        if start < 0 or end <= start or case_text[start:end] != span:
            return False
        if case_text.count(span) != 1:
            return False
        intervals.append((start, end))
    intervals.sort()
    return all(previous_end <= current_start for (_, previous_end), (current_start, _) in zip(intervals, intervals[1:]))


def mask_eligible(record: dict[str, Any], *, require_single_critical: bool = True) -> bool:
    if record.get("validation_status") != "accepted":
        return False
    case_text = record.get("case_text")
    target = record.get("target")
    if not isinstance(case_text, str) or not isinstance(target, dict):
        return False
    critical = evidence_by_importance(target, "critical")
    supporting = evidence_by_importance(target, "supporting")
    if not critical or not supporting:
        return False
    if require_single_critical and len(critical) != 1:
        return False
    return valid_evidence_offsets(case_text, [*critical, *supporting])


def choose_supporting(
    critical: dict[str, Any], supporting: list[dict[str, Any]], seed: int, parent_id: str
) -> dict[str, Any]:
    target_length = len(str(critical["span"]))
    return min(
        supporting,
        key=lambda item: (
            abs(len(str(item["span"])) - target_length),
            stable_rank(f"{parent_id}:{item['id']}", seed),
        ),
    )


def choose_random_span(
    case_text: str,
    evidence: list[dict[str, Any]],
    target_length: int,
    seed: int,
    parent_id: str,
) -> dict[str, Any] | None:
    excluded = [(int(item["start"]), int(item["end"])) for item in evidence]
    candidates: list[dict[str, Any]] = []
    for match in CLAUSE_RE.finditer(case_text):
        start, end = match.span()
        span = match.group(0).strip()
        leading = len(match.group(0)) - len(match.group(0).lstrip())
        trailing = len(match.group(0)) - len(match.group(0).rstrip())
        start += leading
        end -= trailing
        if not span or len(span) < 2:
            continue
        if case_text.count(span) != 1:
            continue
        if any(marker in span for marker in QUERY_MARKERS):
            continue
        if any(start < excluded_end and excluded_start < end for excluded_start, excluded_end in excluded):
            continue
        if len(span) < max(2, math.floor(target_length * 0.5)):
            continue
        if len(span) > max(target_length * 2, target_length + 8):
            continue
        candidates.append({"id": "R1", "span": span, "start": start, "end": end})
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            abs(len(item["span"]) - target_length),
            stable_rank(f"{parent_id}:{item['start']}:{item['span']}", seed),
        ),
    )


def delete_spans(case_text: str, spans: Iterable[dict[str, Any]]) -> str:
    masked = case_text
    ordered = sorted(spans, key=lambda item: int(item["start"]), reverse=True)
    for item in ordered:
        start = int(item["start"])
        end = int(item["end"])
        span = str(item["span"])
        if masked[start:end] != span:
            raise ValueError(f"span offset mismatch for {span!r}")
        masked = masked[:start] + masked[end:]

    masked = re.sub(r"[ \t]+", " ", masked)
    masked = re.sub(r"\s+([，,。；;！？!?：:])", r"\1", masked)
    masked = re.sub(r"([，,。；;！？!?：:])\1+", r"\1", masked)
    masked = re.sub(r"[，,]\s*([。；;！？!?])", r"\1", masked)
    masked = re.sub(r"(^|\n)[，,；;：:]", r"\1", masked)
    return masked.strip()


def stratified_sample(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    fields: tuple[str, ...] = ("task_type", "evidence_sufficiency", "category"),
) -> list[dict[str, Any]]:
    if count < 0:
        raise ValueError("count must be non-negative")
    if count >= len(rows):
        return sorted(rows, key=lambda row: stable_rank(str(row["source_id"]), seed))

    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in fields)
        groups[key].append(row)

    allocation: dict[tuple[str, ...], int] = {}
    fractions: list[tuple[float, tuple[str, ...]]] = []
    for key, group in groups.items():
        exact = count * len(group) / len(rows)
        base = min(len(group), math.floor(exact))
        allocation[key] = base
        fractions.append((exact - base, key))

    remaining = count - sum(allocation.values())
    for _, key in sorted(fractions, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if allocation[key] < len(groups[key]):
            allocation[key] += 1
            remaining -= 1
    if remaining:
        for key in sorted(groups):
            while remaining and allocation[key] < len(groups[key]):
                allocation[key] += 1
                remaining -= 1

    selected: list[dict[str, Any]] = []
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: stable_rank(str(row["source_id"]), seed))
        selected.extend(ordered[: allocation[key]])
    return sorted(selected, key=lambda row: stable_rank(str(row["source_id"]), seed))


def without_offsets(target: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(target)
    for item in cleaned.get("evidence", []):
        if isinstance(item, dict):
            item.pop("start", None)
            item.pop("end", None)
    return cleaned
