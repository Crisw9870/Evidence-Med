from __future__ import annotations

import hashlib
import json
import sys
import unittest
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESS_DPO = ROOT / "process_dpo"
if str(PROCESS_DPO) not in sys.path:
    sys.path.insert(0, str(PROCESS_DPO))

from build_dpo_pairs import build_pairs
from dpo_common import (
    SCORE_LIMITS,
    assess_target_warning_risk,
    audit_response,
    build_training_prompt,
    compact_json,
    make_controlled_negative,
    shared_structure,
    valid_judgment,
)
from export_dpo_dataset import export_dataset
from judge_dpo_pairs import build_user_prompt
from select_dpo_sources import select_sources


CASE_TEXT = "患者咳嗽三天，伴发热，胸片提示右下肺浸润，询问可能原因和下一步怎么办？"
TARGET = {
    "task_type": "diagnostic_reasoning",
    "query_intent": ["判断咳嗽发热和肺部浸润的可能原因及下一步处理"],
    "evidence_sufficiency": "partial",
    "evidence": [
        {
            "id": "E1",
            "span": "咳嗽三天",
            "importance": "supporting",
            "role": "提供呼吸道症状及病程信息",
        },
        {
            "id": "E2",
            "span": "伴发热",
            "importance": "supporting",
            "role": "支持存在感染或炎症过程",
        },
        {
            "id": "E3",
            "span": "胸片提示右下肺浸润",
            "importance": "critical",
            "role": "提示肺部存在影像学异常并决定主要判断方向",
        },
    ],
    "critical_evidence_ids": ["E3"],
    "missing_information": ["血常规及炎症指标", "病原学检查结果"],
    "clinical_reasoning": "已有症状和胸片异常支持肺部感染方向，但缺少实验室及病原学结果，暂不能确定具体病因。",
    "final_answer": "目前资料提示肺部感染的可能性较高，但尚不能确定具体病原体，建议尽快就医结合血液检查和病原学结果进一步判断。",
}


def source(source_id: str, split: str = "train", warnings: list[str] | None = None) -> dict:
    return {
        "source_id": source_id,
        "split": split,
        "category": "呼吸科",
        "case_text": CASE_TEXT,
        "target": json.loads(json.dumps(TARGET, ensure_ascii=False)),
        "validation_warnings": warnings or [],
    }


def model_candidate(source_row: dict, candidate_id: str, response: dict) -> dict:
    text = compact_json(response)
    return {
        "candidate_id": candidate_id,
        "origin": "sft_model",
        "text": text,
        "audit": audit_response(source_row["case_text"], text, source_row["target"]),
    }


def judgment(decision: str = "A_better", confidence: float = 0.95) -> dict:
    high = {name: maximum for name, maximum in SCORE_LIMITS.items()}
    low = {name: max(0, maximum - 1) for name, maximum in SCORE_LIMITS.items()}
    return {
        "decision": decision,
        "hard_failures": {"A": [], "B": []},
        "scores": {"A": high, "B": low},
        "decisive_dimensions": ["evidence_faithfulness"],
        "reason": "A 的证据与病例一致，B 存在明确的标签或证据缺陷。",
        "confidence": confidence,
    }


def test_controlled_negative_is_valid_and_detectably_worse() -> None:
    intended_errors = {"overconfident_answer", "generic_non_answer"}
    for index in range(12):
        negative = make_controlled_negative(TARGET, f"source_{index}", seed=42)
        audit = audit_response(CASE_TEXT, negative["text"], TARGET)
        assert audit["parsed_output"] is not None
        assert not audit["schema_errors"]
        assert not audit["hard_failures"]
        assert negative["text"] != compact_json(TARGET)
        assert negative["intended_error"] in intended_errors
        assert negative["parsed_output"]["task_type"] == TARGET["task_type"]
        assert (
            negative["parsed_output"]["evidence_sufficiency"]
            == TARGET["evidence_sufficiency"]
        )
        assert negative["parsed_output"]["evidence"] == TARGET["evidence"]


def test_source_selection_locks_test_and_excludes_high_risk_warning() -> None:
    rows = [
        source("train_clean", "train"),
        source("validation_clean", "validation"),
        source("test_locked", "test"),
        source("train_risky", "train", ["task_type_changed_by_teacher"]),
    ]
    args = Namespace(
        input="fixture.jsonl",
        train_limit=10,
        validation_limit=10,
        seed=42,
        warning_policy="strict",
        max_medium_risk_fraction=0.15,
        include_high_risk_warnings=False,
    )
    selected, stats = select_sources(rows, args)
    assert {row["source_id"] for row in selected} == {"train_clean", "validation_clean"}
    assert all(row["split"] in {"train", "validation"} for row in selected)
    assert stats["rejected"]["locked_split:test"] == 1
    assert stats["rejected"]["warning_policy:strict"] == 1


def test_frozen_dpo_v1_source_manifest() -> None:
    """Lock the exact source order used by the completed DPO v1 experiment."""
    path = ROOT / "data" / "dpo" / "answer_v1" / "00_sources.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3200
    assert sum(row["split"] == "train" for row in rows) == 3000
    assert sum(row["split"] == "validation" for row in rows) == 200
    assert len({row["source_id"] for row in rows}) == 3200
    payload = "\n".join(
        f"{row['split']}\t{row['source_id']}" for row in rows
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == (
        "3913d351282fbfc51ddf1089ef1a13ce95464e149a5c92a5a14ed69a1f058563"
    )


def test_warning_risk_classification() -> None:
    target = json.loads(json.dumps(TARGET, ensure_ascii=False))
    target["evidence"][0]["role"] = "证明患者已经发生心肌梗死"
    target["clinical_reasoning"] += " 建议继续观察3天。"
    target["final_answer"] += " 不要自行服用5mg药物。"
    risk = assess_target_warning_risk(
        CASE_TEXT,
        target,
        ["strong_role_claim_requires_review", "generated_numbers_require_review"],
    )
    assert risk["level"] == "high"
    assert "E1" in risk["strong_evidence_ids"]
    assert "5mg" in risk["high_risk_numbers"]
    assert "3天" in risk["medium_risk_numbers"]


def test_pair_builder_produces_three_types_and_blind_prompt() -> None:
    raw = source("pair_source")
    selected_source = {
        **raw,
        "prompt": build_training_prompt(CASE_TEXT),
        "target_warnings": [],
    }
    wrong_task = json.loads(json.dumps(TARGET, ensure_ascii=False))
    wrong_task["task_type"] = "confirmed_management"
    wrong_task["final_answer"] += " 请同时记录体温和症状变化。"
    wrong_sufficiency = json.loads(json.dumps(TARGET, ensure_ascii=False))
    wrong_sufficiency["evidence_sufficiency"] = "sufficient"
    wrong_sufficiency["clinical_reasoning"] = (
        "现有表现支持肺部感染方向，但仍需结合血常规和病原学检查判断具体病因。"
    )
    wrong_sufficiency["final_answer"] += " 如出现呼吸困难，应及时急诊就医。"
    candidates = [
        {
            "source_id": "pair_source",
            "candidates": [
                model_candidate(selected_source, "m_good", TARGET),
                model_candidate(selected_source, "m_task", wrong_task),
                model_candidate(selected_source, "m_suff", wrong_sufficiency),
            ],
        }
    ]
    args = Namespace(
        allow_invalid_json_rejected=False,
        target_model_per_source=1,
        model_model_per_source=1,
        controlled_per_source=1,
        seed=42,
    )
    pairs = build_pairs([selected_source], candidates, args)
    assert {row["pair_type"] for row in pairs} == {
        "target_vs_model",
        "model_vs_model",
        "controlled_negative",
    }
    prompt = build_user_prompt(pairs[0])
    assert "validated_target" not in prompt
    assert "sft_model" not in prompt
    assert "answer_A" in prompt and "answer_B" in prompt


def test_judgment_schema_and_export_quality_gates() -> None:
    assert valid_judgment(judgment())
    selected_source = {
        **source("train_source"),
        "prompt": build_training_prompt(CASE_TEXT),
    }
    target_candidate = model_candidate(selected_source, "target", TARGET)
    target_candidate["origin"] = "validated_target"
    alternative = json.loads(json.dumps(TARGET, ensure_ascii=False))
    alternative["clinical_reasoning"] = (
        "病例提示肺部感染可能，但缺少实验室和病原学结果，暂不能确定病原体。"
    )
    alternative["final_answer"] = (
        "目前更考虑肺部感染，但尚不能确定具体病原体，建议结合血常规、炎症指标和病原学检查进一步确认。"
    )
    negative = model_candidate(selected_source, "alternative", alternative)
    pair = {
        "pair_id": "pair_train",
        "pair_type": "target_vs_model",
        "source_id": "train_source",
        "split": "train",
        "case_text": CASE_TEXT,
        "prompt": selected_source["prompt"],
        "shared_structure": shared_structure(TARGET),
        "candidate_A": target_candidate,
        "candidate_B": negative,
    }
    validation_pair = {
        **pair,
        "pair_id": "pair_validation",
        "source_id": "validation_source",
        "split": "validation",
    }
    judgments = [
        {
            "pair_id": "pair_train",
            "source_id": "train_source",
            "split": "train",
            "status": "ok",
            "parsed_judgment": judgment(),
        },
        {
            "pair_id": "pair_validation",
            "source_id": "validation_source",
            "split": "validation",
            "status": "ok",
            "parsed_judgment": judgment(),
        },
    ]
    args = Namespace(
        min_confidence=0.85,
        min_score_margin=2,
        train_limit=10,
        validation_limit=10,
        max_pairs_per_source=2,
    )
    datasets, audits, stats = export_dataset([pair, validation_pair], judgments, args)
    assert len(datasets["train"]) == 1
    assert len(datasets["validation"]) == 1
    assert datasets["train"][0]["chosen"] == target_candidate["text"]
    assert datasets["train"][0]["rejected"] == negative["text"]
    assert datasets["train"][0]["prompt"] == selected_source["prompt"]
    assert audits["train"][0]["chosen_origin"] == "validated_target"
    assert stats["swap_consistency_check"] is False


def test_low_confidence_and_chosen_hard_failure_are_rejected() -> None:
    selected_source = {**source("gate_source"), "prompt": build_training_prompt(CASE_TEXT)}
    good = model_candidate(selected_source, "good", TARGET)
    bad_json = {
        "candidate_id": "bad",
        "origin": "sft_model",
        "text": "not json",
        "audit": audit_response(CASE_TEXT, "not json", TARGET),
    }
    pair = {
        "pair_id": "pair_gate",
        "pair_type": "target_vs_model",
        "source_id": "gate_source",
        "split": "train",
        "case_text": CASE_TEXT,
        "prompt": selected_source["prompt"],
        "shared_structure": shared_structure(TARGET),
        "candidate_A": good,
        "candidate_B": bad_json,
    }
    low_confidence = {
        "pair_id": "pair_gate",
        "source_id": "gate_source",
        "split": "train",
        "status": "ok",
        "parsed_judgment": judgment(confidence=0.5),
    }
    args = Namespace(
        min_confidence=0.85,
        min_score_margin=2,
        train_limit=10,
        validation_limit=10,
        max_pairs_per_source=2,
    )
    datasets, _, _ = export_dataset([pair], [low_confidence], args)
    assert datasets["train"] == []


class DPOPipelineTests(unittest.TestCase):
    def test_controlled_negative(self) -> None:
        test_controlled_negative_is_valid_and_detectably_worse()

    def test_source_selection(self) -> None:
        test_source_selection_locks_test_and_excludes_high_risk_warning()

    def test_frozen_source_manifest(self) -> None:
        test_frozen_dpo_v1_source_manifest()

    def test_warning_risk(self) -> None:
        test_warning_risk_classification()

    def test_pair_builder_and_blinding(self) -> None:
        test_pair_builder_produces_three_types_and_blind_prompt()

    def test_judgment_and_export(self) -> None:
        test_judgment_schema_and_export_quality_gates()

    def test_quality_gates(self) -> None:
        test_low_confidence_and_chosen_hard_failure_are_rejected()


if __name__ == "__main__":
    unittest.main()
