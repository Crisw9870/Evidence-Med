from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "process"))
sys.path.insert(0, str(ROOT / "evi-sft-traing"))

from aggregate_evidence_mask import (  # noqa: E402
    compare_binary_judgment,
    compare_specificity_increment,
    model_summary,
)
from build_evidence_mask_candidates import build_candidate_rows  # noqa: E402
from build_evidence_mask_targets import build_user_prompt  # noqa: E402
from evidence_mask_common import delete_spans, mask_eligible  # noqa: E402
from validate_evidence_mask import validate_mask_dataset  # noqa: E402
from validate_evidence_sft import build_training_prompt  # noqa: E402


class EvidenceMaskPipelineTests(unittest.TestCase):
    def make_parent(self, source_id: str, split: str) -> dict[str, object]:
        case = "患者男性，45岁，反复胸痛2周，活动后加重，心电图提示ST段异常。请问可能是什么问题？"
        supporting = "反复胸痛2周"
        critical = "心电图提示ST段异常"
        target = {
            "task_type": "diagnostic_reasoning",
            "query_intent": ["判断反复胸痛的可能原因"],
            "evidence_sufficiency": "partial",
            "evidence": [
                {
                    "id": "E1",
                    "span": supporting,
                    "importance": "supporting",
                    "role": "说明患者存在持续两周的胸痛症状",
                    "start": case.index(supporting),
                    "end": case.index(supporting) + len(supporting),
                },
                {
                    "id": "E2",
                    "span": critical,
                    "importance": "critical",
                    "role": "提示需要优先排查心肌缺血等心血管问题",
                    "start": case.index(critical),
                    "end": case.index(critical) + len(critical),
                },
            ],
            "critical_evidence_ids": ["E2"],
            "missing_information": ["胸痛发作特点和心肌损伤标志物检查结果"],
            "clinical_reasoning": "已有持续胸痛和心电图异常证据，需要优先评估心血管疾病，但目前还不能确定具体诊断。",
            "final_answer": "建议尽快到心内科进一步检查；如果胸痛持续或伴有大汗、呼吸困难，应立即前往急诊。",
        }
        return {
            "source_id": source_id,
            "split": split,
            "category": "内科",
            "case_text": case,
            "validation_status": "accepted",
            "target": target,
        }

    @staticmethod
    def make_mask_target(candidate: dict[str, object]) -> dict[str, object]:
        case = str(candidate["case_text"])
        if candidate["mask_type"] == "critical":
            span = "反复胸痛2周"
            importance = "supporting"
            critical_ids: list[str] = []
            role = "说明患者存在持续两周的胸痛症状"
        else:
            span = "心电图提示ST段异常"
            importance = "critical"
            critical_ids = ["E1"]
            role = "提示需要优先排查心肌缺血等心血管问题"
        assert span in case
        return {
            "task_type": "diagnostic_reasoning",
            "query_intent": ["判断胸痛的可能原因"],
            "evidence_sufficiency": "partial",
            "evidence": [
                {
                    "id": "E1",
                    "span": span,
                    "importance": importance,
                    "role": role,
                }
            ],
            "critical_evidence_ids": critical_ids,
            "missing_information": ["完整心电图和心肌损伤标志物检查结果"],
            "clinical_reasoning": "当前可见信息只能提示需要继续评估胸痛原因，仍缺少足够证据确定具体疾病。",
            "final_answer": "目前信息不足以确定具体诊断，建议尽快到心内科完善检查；如果胸痛持续或加重，应立即急诊就医。",
        }

    @staticmethod
    def make_original_sft(parent: dict[str, object]) -> dict[str, object]:
        target = json.loads(json.dumps(parent["target"], ensure_ascii=False))
        for item in target["evidence"]:
            item.pop("start", None)
            item.pop("end", None)
        return {
            "source_id": parent["source_id"],
            "split": parent["split"],
            "task_type": target["task_type"],
            "schema_version": "evidence-sft-v2.2",
            "masked": False,
            "conversations": [
                {"from": "human", "value": build_training_prompt(str(parent["case_text"]))},
                {"from": "gpt", "value": json.dumps(target, ensure_ascii=False)},
            ],
        }

    @staticmethod
    def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_candidate_builder_preserves_parent_split_and_hides_critical(self) -> None:
        parents = [
            self.make_parent("train_case", "train"),
            self.make_parent("validation_case", "validation"),
            self.make_parent("test_case", "test"),
        ]
        self.assertTrue(all(mask_eligible(parent) for parent in parents))
        variants, stats = build_candidate_rows(
            parents,
            seed=42,
            train_candidates=1,
            validation_candidates=1,
            test_pairs=1,
            include_random=False,
        )
        self.assertEqual(stats["selected_parents"], {"train": 1, "validation": 1, "test": 1})
        self.assertEqual(len(variants), 4)
        for variant in variants:
            expected_split = next(
                parent["split"]
                for parent in parents
                if parent["source_id"] == variant["parent_source_id"]
            )
            self.assertEqual(variant["split"], expected_split)
            for removed in variant["removed_spans"]:
                self.assertNotIn(removed["span"], variant["case_text"])
            self.assertNotIn("[MASK]", variant["case_text"])

    def test_delete_spans_fails_on_wrong_offset(self) -> None:
        with self.assertRaises(ValueError):
            delete_spans("患者反复胸痛。", [{"span": "胸痛", "start": 0, "end": 2}])

    def test_teacher_prompt_contains_masked_case_only(self) -> None:
        parent = self.make_parent("train_case", "train")
        variants, _ = build_candidate_rows(
            [parent],
            seed=42,
            train_candidates=1,
            validation_candidates=0,
            test_pairs=0,
            include_random=False,
        )
        variant = variants[0]
        prompt = build_user_prompt(variant)
        self.assertIn(variant["case_text"], prompt)
        self.assertNotIn(variant["original_case_text"], prompt)
        self.assertNotIn("original_target", prompt)
        self.assertNotIn(variant["removed_spans"][0]["span"], prompt)

    def test_validation_exports_equal_budget_m0_m1_and_test_pairs(self) -> None:
        parents = [
            self.make_parent("train_case", "train"),
            self.make_parent("validation_case", "validation"),
            self.make_parent("test_case", "test"),
        ]
        candidates, _ = build_candidate_rows(
            parents,
            seed=42,
            train_candidates=1,
            validation_candidates=1,
            test_pairs=1,
            include_random=False,
        )
        teachers: list[dict[str, object]] = []
        judgments: list[dict[str, object]] = []
        for candidate in candidates:
            target = self.make_mask_target(candidate)
            teachers.append(
                {
                    "source_id": candidate["source_id"],
                    "variant_id": candidate["variant_id"],
                    "pair_id": candidate["pair_id"],
                    "parent_source_id": candidate["parent_source_id"],
                    "split": candidate["split"],
                    "mask_type": candidate["mask_type"],
                    "case_text": candidate["case_text"],
                    "teacher_model": "test-teacher",
                    "prompt_version": "evidence-mask-v1",
                    "status": "ok",
                    "teacher_text": json.dumps(target, ensure_ascii=False),
                    "parsed_output": target,
                }
            )
            expected = "downgrade" if candidate["mask_type"] == "critical" else "stay"
            judgment = {
                "decision": "accept",
                "expected_certainty_change": expected,
                "removed_fact_concepts": ["被删除的病例事实"],
                "required_missing_concepts": ["需要补充的关键检查"],
                "allowed_conclusion_scope": "只能建议进一步评估，不能确定诊断",
                "forbidden_specific_claims": ["已经确诊具体疾病"],
                "reasons": ["反事实目标与剩余证据一致"],
            }
            judgments.append(
                {
                    "source_id": candidate["source_id"],
                    "variant_id": candidate["variant_id"],
                    "pair_id": candidate["pair_id"],
                    "judge_model": "test-judge",
                    "status": "ok",
                    "parsed_judgment": judgment,
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates_path = root / "candidates.jsonl"
            teachers_path = root / "teachers.jsonl"
            judgments_path = root / "judgments.jsonl"
            train_path = root / "train.jsonl"
            validation_path = root / "validation.jsonl"
            test_path = root / "test.jsonl"
            output_dir = root / "validated"
            self.write_rows(candidates_path, candidates)
            self.write_rows(teachers_path, teachers)
            self.write_rows(judgments_path, judgments)
            self.write_rows(train_path, [self.make_original_sft(parents[0])])
            self.write_rows(validation_path, [self.make_original_sft(parents[1])])
            self.write_rows(test_path, [self.make_original_sft(parents[2])])

            stats = validate_mask_dataset(
                candidates_path,
                teachers_path,
                judgments_path,
                output_dir,
                train_path,
                validation_path,
                test_path,
                train_replacements=1,
                validation_replacements=1,
                test_pairs=1,
                seed=42,
            )
            self.assertEqual(stats["counts"]["accepted"], 4)
            self.assertEqual(stats["counts"]["m0_train"], 1)
            self.assertEqual(stats["counts"]["m1_train"], 1)
            self.assertTrue(stats["integrity"]["m0_m1_train_parent_ids_equal"])

            m1_row = json.loads((output_dir / "m1/train.jsonl").read_text(encoding="utf-8"))
            self.assertTrue(m1_row["masked"])
            self.assertEqual(m1_row["source_id"], "train_case")
            self.assertEqual(m1_row["parent_source_id"], "train_case")
            test_rows = [
                json.loads(line)
                for line in (output_dir / "test_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual({row["mask_type"] for row in test_rows}, {"unmasked", "critical", "supporting"})
            self.assertEqual(len({row["variant_id"] for row in test_rows}), 3)

    @staticmethod
    def make_prediction(pair: str, mask_type: str, appropriate: bool) -> tuple[dict[str, object], dict[str, object]]:
        variant_id = f"{pair}::{mask_type}"
        prediction = {
            "variant_id": variant_id,
            "source_id": pair,
            "pair_id": pair,
            "parent_source_id": pair,
            "mask_type": mask_type,
            "gold": {"evidence_sufficiency": "partial"},
            "parsed_output": {"evidence_sufficiency": "partial"},
            "mask_assessment": {"expected_certainty_change": "stay"},
            "metrics": {
                "schema_valid": True,
                "all_evidence_grounded": True,
                "predicted_evidence_count": 1,
                "grounded_evidence_count": 1,
                "removed_span_evidence_leakage": False,
                "removed_span_literal_mention": False,
            },
        }
        judgment = {
            "appropriate_response": appropriate,
            "removed_fact_leakage": False,
            "conclusion_scope_appropriate": appropriate,
            "missing_information_appropriate": appropriate,
            "safe_answer": True,
            "main_conclusion_behavior": "stay",
            "reasons": ["test"],
        }
        return prediction, judgment

    def test_paired_aggregation_reports_treatment_gain(self) -> None:
        baseline_predictions: list[dict[str, object]] = []
        treatment_predictions: list[dict[str, object]] = []
        baseline_judgments: dict[str, dict[str, object]] = {}
        treatment_judgments: dict[str, dict[str, object]] = {}
        for pair in ("p1", "p2"):
            baseline, baseline_judge = self.make_prediction(pair, "critical", False)
            treatment, treatment_judge = self.make_prediction(pair, "critical", True)
            baseline_predictions.append(baseline)
            treatment_predictions.append(treatment)
            baseline_judgments[str(baseline["variant_id"])] = baseline_judge
            treatment_judgments[str(treatment["variant_id"])] = treatment_judge

        comparison = compare_binary_judgment(
            treatment_predictions,
            treatment_judgments,
            baseline_predictions,
            baseline_judgments,
            field="appropriate_response",
            mask_type="critical",
            bootstrap_iters=200,
            seed=42,
        )
        self.assertEqual(comparison["paired_samples"], 2)
        self.assertEqual(comparison["delta_pp"], 100.0)
        self.assertEqual(comparison["treatment_only_true"], 2)
        self.assertEqual(comparison["net_treatment_favorable"], 2)
        summary = model_summary(treatment_predictions, treatment_judgments)
        self.assertEqual(summary["by_mask_type"]["critical"]["judge"]["appropriate_response"]["rate"], 1.0)

        for pair in ("p1", "p2"):
            baseline, baseline_judge = self.make_prediction(pair, "supporting", True)
            treatment, treatment_judge = self.make_prediction(pair, "supporting", True)
            baseline_predictions.append(baseline)
            treatment_predictions.append(treatment)
            baseline_judgments[str(baseline["variant_id"])] = baseline_judge
            treatment_judgments[str(treatment["variant_id"])] = treatment_judge
        specificity = compare_specificity_increment(
            treatment_predictions,
            treatment_judgments,
            baseline_predictions,
            baseline_judgments,
            control_type="supporting",
            bootstrap_iters=200,
            seed=42,
        )
        self.assertEqual(specificity["paired_cases"], 2)
        self.assertEqual(specificity["difference_in_differences_pp"], 100.0)


if __name__ == "__main__":
    unittest.main()
