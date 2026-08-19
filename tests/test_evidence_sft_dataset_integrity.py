from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "process"))

from evidence_sft_common import EVIDENCE_SCHEMA_VERSION, stable_source_id  # noqa: E402
from validate_evidence_sft import validate_dataset  # noqa: E402


class EvidenceSFTDatasetIntegrityTests(unittest.TestCase):
    def test_duplicate_source_ids_and_candidate_mismatch_are_rejected(self) -> None:
        case_text = "患者反复胸痛2周，活动后加重，心电图提示ST段异常。请问可能是什么问题？"
        target = {
            "task_type": "diagnostic_reasoning",
            "query_intent": ["判断胸痛的可能原因"],
            "evidence_sufficiency": "partial",
            "evidence": [
                {
                    "id": "E1",
                    "span": "反复胸痛2周",
                    "importance": "critical",
                    "role": "提示存在持续性胸痛，需要进一步评估病因",
                }
            ],
            "critical_evidence_ids": ["E1"],
            "missing_information": ["胸痛发作持续时间和伴随症状"],
            "clinical_reasoning": "已有持续性胸痛证据，但缺少完整伴随症状和检查结果，目前不能进一步确定具体诊断。",
            "final_answer": "建议尽快到心内科进一步检查；如果胸痛持续或伴有大汗、呼吸困难，应立即前往急诊。",
        }
        source_id = stable_source_id(case_text)
        candidate = {
            "source_id": source_id,
            "split": "train",
            "category": "内科",
            "task_type_hint": "diagnostic_reasoning",
            "case_text": case_text,
            "original_answer": "原始答案仅供教师识别问题，不能作为患者证据，也不能直接复制到训练目标中。",
        }
        raw = {
            **candidate,
            "teacher_model": "deepseek-v4-flash",
            "prompt_version": EVIDENCE_SCHEMA_VERSION,
            "status": "ok",
            "teacher_text": json.dumps(target, ensure_ascii=False),
            "parsed_output": target,
        }
        duplicate_with_wrong_split = {**raw, "split": "validation"}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate_path = root / "candidates.jsonl"
            raw_path = root / "raw.jsonl"
            output_dir = root / "validated"
            candidate_path.write_text(
                json.dumps(candidate, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            raw_path.write_text(
                "\n".join(
                    [
                        json.dumps(raw, ensure_ascii=False),
                        json.dumps(duplicate_with_wrong_split, ensure_ascii=False),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_dataset(raw_path, candidate_path, output_dir)
            self.assertEqual(report["counts"]["accepted"], 0)
            self.assertEqual(report["counts"]["rejected"], 2)
            self.assertEqual(report["integrity"]["duplicate_source_ids"], 1)
            self.assertEqual(report["reject_reason_counts"]["duplicate_source_id"], 2)
            self.assertEqual(
                report["reject_reason_counts"]["candidate_split_mismatch"], 1
            )


if __name__ == "__main__":
    unittest.main()
