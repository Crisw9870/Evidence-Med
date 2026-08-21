from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "process_sft"))

from evidence_sft_common import (  # noqa: E402
    EVIDENCE_SCHEMA_VERSION,
    assign_split,
    classify_task,
    extract_first_json_object,
    score_case_candidate,
    stable_source_id,
)
from validate_evidence_sft import (  # noqa: E402
    audit_output,
    validate_dataset,
    validate_output,
)


class EvidenceSFTPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.case = "患者男性，45岁，反复胸痛2周，活动后加重，心电图提示ST段异常。请问可能是什么问题？"

    def make_target(self, **overrides: object) -> dict[str, object]:
        value: dict[str, object] = {
            "task_type": "diagnostic_reasoning",
            "query_intent": ["判断反复胸痛的可能原因并咨询下一步检查"],
            "evidence_sufficiency": "partial",
            "evidence": [
                {
                    "id": "E1",
                    "span": "反复胸痛2周",
                    "importance": "supporting",
                    "role": "提示存在持续两周的胸痛症状",
                },
                {
                    "id": "E2",
                    "span": "心电图提示ST段异常",
                    "importance": "critical",
                    "role": "支持优先评估心肌缺血等心血管问题",
                },
            ],
            "critical_evidence_ids": ["E2"],
            "missing_information": ["胸痛发作持续时间及有无大汗、呼吸困难等伴随症状"],
            "clinical_reasoning": "已有胸痛和心电图异常证据，需要优先排查心肌缺血，但目前仍不能仅凭描述确定诊断。",
            "final_answer": "这些表现需要尽快由心内科进一步评估；若胸痛持续或伴大汗、呼吸困难，应立即急诊就医。",
        }
        value.update(overrides)
        return value

    def test_candidate_scoring_and_task_classification(self) -> None:
        score, reasons = score_case_candidate(
            self.case, "建议尽快到心内科进一步检查并排除冠心病。"
        )
        self.assertGreaterEqual(score, 5)
        self.assertIn("exam_context", reasons)
        self.assertEqual(classify_task(self.case), "diagnostic_reasoning")
        self.assertEqual(
            classify_task("患者已经确诊高血压，请问如何治疗？"),
            "confirmed_management",
        )

    def test_stable_id_and_split(self) -> None:
        first = stable_source_id(self.case)
        second = stable_source_id("  " + self.case + "  ")
        self.assertEqual(first, second)
        self.assertEqual(assign_split(first, 42), assign_split(first, 42))

    def test_json_extraction(self) -> None:
        parsed = extract_first_json_object(
            '说明文字\n```json\n{"task_type":"diagnostic_reasoning"}\n```'
        )
        self.assertEqual(parsed, {"task_type": "diagnostic_reasoning"})

    def test_validator_accepts_v22_and_preserves_fields(self) -> None:
        result = audit_output(self.case, self.make_target())
        self.assertEqual(result.errors, [])
        self.assertEqual(result.review_reasons, [])
        self.assertIsNotNone(result.normalized)
        assert result.normalized is not None
        self.assertEqual(result.normalized["query_intent"][0], "判断反复胸痛的可能原因并咨询下一步检查")
        self.assertEqual(result.normalized["evidence"][1]["importance"], "critical")
        self.assertEqual(
            result.normalized["evidence"][0]["start"], self.case.index("反复胸痛2周")
        )

    def test_partial_is_a_valid_sufficiency_level(self) -> None:
        normalized, errors, _ = validate_output(self.case, self.make_target())
        self.assertEqual(errors, [])
        self.assertIsNotNone(normalized)

    def test_zero_critical_evidence_is_allowed(self) -> None:
        target = self.make_target(
            evidence_sufficiency="sufficient",
            evidence=[
                {
                    "id": "E1",
                    "span": "反复胸痛2周",
                    "importance": "supporting",
                    "role": "提示存在持续两周的胸痛症状",
                }
            ],
            critical_evidence_ids=[],
            missing_information=[],
        )
        result = audit_output(self.case, target)
        self.assertEqual(result.errors, [])
        self.assertNotIn("missing_critical_evidence_ids", result.warnings)

    def test_validator_rejects_hallucinated_evidence(self) -> None:
        target = self.make_target(
            evidence=[
                {
                    "id": "E1",
                    "span": "肌钙蛋白升高",
                    "importance": "critical",
                    "role": "支持存在心肌损伤",
                }
            ],
            critical_evidence_ids=["E1"],
        )
        result = audit_output(self.case, target)
        self.assertIsNone(result.normalized)
        self.assertIn("evidence_span_not_in_case", result.errors)

    def test_validator_rejects_critical_importance_mismatch(self) -> None:
        target = self.make_target(critical_evidence_ids=["E1"])
        result = audit_output(self.case, target)
        self.assertIn("critical_importance_mismatch", result.errors)

    def test_validator_rejects_sufficient_without_evidence(self) -> None:
        target = self.make_target(
            evidence_sufficiency="sufficient",
            evidence=[],
            critical_evidence_ids=[],
            missing_information=[],
        )
        result = audit_output(self.case, target)
        self.assertIn("sufficient_without_evidence", result.errors)

    def test_validator_reviews_schema_drift(self) -> None:
        target = self.make_target(self_check_notes="已完成检查")
        result = audit_output(self.case, target)
        self.assertEqual(result.errors, [])
        self.assertTrue(
            any(reason.startswith("unexpected_output_fields") for reason in result.review_reasons)
        )

    def test_validator_reviews_atomicity_and_ambiguous_location(self) -> None:
        case = "胸痛伴随活动加重，而且每次持续十分钟以上，需要进一步评估具体病因和可能存在的危险情况。之后胸痛再次出现。"
        target = self.make_target(
            evidence=[
                {
                    "id": "E1",
                    "span": "胸痛",
                    "importance": "critical",
                    "role": "提示存在胸痛症状",
                },
                {
                    "id": "E2",
                    "span": "胸痛伴随活动加重，而且每次持续十分钟以上，需要进一步评估具体病因和可能存在的危险情况",
                    "importance": "supporting",
                    "role": "补充疼痛诱发因素和持续时间",
                },
            ],
            critical_evidence_ids=["E1"],
        )
        result = audit_output(case, target)
        self.assertEqual(result.errors, [])
        self.assertIn("ambiguous_repeated_evidence_span", result.review_reasons)
        self.assertIn(
            "long_evidence_span_requires_atomicity_review", result.review_reasons
        )
        self.assertIn("overlapping_evidence_spans", result.review_reasons)

    def test_partial_without_missing_information_requires_review(self) -> None:
        result = audit_output(
            self.case,
            self.make_target(missing_information=[]),
        )
        self.assertEqual(result.errors, [])
        self.assertIn("partial_without_missing_information", result.review_reasons)

    def test_validator_rejects_original_answer_copy(self) -> None:
        target = self.make_target()
        result = audit_output(self.case, target, str(target["final_answer"]))
        self.assertIn("final_answer_copies_original", result.errors)

    def test_dataset_validation_exports_only_clean_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidates.jsonl"
            raw_path = root / "raw.jsonl"
            output_dir = root / "validated"
            source_id = stable_source_id(self.case)
            target = self.make_target()
            candidate = {
                "source_id": source_id,
                "split": "train",
                "category": "内科",
                "task_type_hint": "diagnostic_reasoning",
                "case_text": self.case,
                "original_answer": "原始回答可能存在错误，因此需要由教师重新构造安全、完整且有证据边界的回答。",
            }
            raw = {
                **candidate,
                "teacher_model": "deepseek-v4-flash-free",
                "prompt_version": EVIDENCE_SCHEMA_VERSION,
                "status": "ok",
                "teacher_text": json.dumps(target, ensure_ascii=False),
                "parsed_output": target,
            }
            candidate_path.write_text(
                json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            raw_path.write_text(
                json.dumps(raw, ensure_ascii=False) + "\n", encoding="utf-8"
            )

            report = validate_dataset(raw_path, candidate_path, output_dir)
            self.assertEqual(report["counts"]["accepted"], 1)
            self.assertEqual(report["counts"]["review"], 0)
            self.assertEqual(report["counts"]["rejected"], 0)
            self.assertEqual(report["integrity"]["candidate_ids_missing_from_input"], 0)

            training_row = json.loads(
                (output_dir / "train.jsonl").read_text(encoding="utf-8").strip()
            )
            assistant_target = json.loads(training_row["conversations"][1]["value"])
            self.assertIn("query_intent", assistant_target)
            self.assertEqual(assistant_target["evidence"][1]["importance"], "critical")
            self.assertNotIn("start", assistant_target["evidence"][0])

            audited_row = json.loads(
                (output_dir / "03_validated_full.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertIn("start", audited_row["target"]["evidence"][0])
            self.assertEqual(audited_row["validation_status"], "accepted")


if __name__ == "__main__":
    unittest.main()
