#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cmb_utils import extract_choice, format_exam_prompt, normalized_choice  # noqa: E402
from judge_clin import valid_judgment  # noqa: E402


class ChoiceExtractionTests(unittest.TestCase):
    def test_exact_single(self) -> None:
        self.assertEqual(extract_choice("D", "ABCDE", "单项选择题"), ("D", "exact"))

    def test_leading_option_with_text(self) -> None:
        self.assertEqual(
            extract_choice("D. 共5对鳃弓", "ABCDE", "单项选择题"),
            ("D", "leading_option"),
        )

    def test_answer_phrase_multiple(self) -> None:
        answer, method = extract_choice("分析略。最终答案：A、C、D", "ABCDE", "多项选择题")
        self.assertEqual(answer, "ACD")
        self.assertEqual(method, "answer_pattern")

    def test_reject_arbitrary_rationale_letters(self) -> None:
        answer, method = extract_choice("A项似乎合理，但B项也可能。", "ABCDE", "单项选择题")
        self.assertIsNone(answer)
        self.assertEqual(method, "no_answer_pattern")

    def test_reject_multiple_for_single(self) -> None:
        answer, method = extract_choice("答案：AC", "ABCDE", "单项选择题")
        self.assertIsNone(answer)
        self.assertEqual(method, "multiple_for_single")

    def test_normalize_set(self) -> None:
        self.assertEqual(normalized_choice("C、A、C", "ABCDE"), "AC")

    def test_prompt_contains_all_options(self) -> None:
        prompt = format_exam_prompt(
            {
                "exam_type": "医师考试",
                "exam_class": "执业医师",
                "question_type": "单项选择题",
                "question": "示例题",
                "option": {"A": "甲", "B": "乙"},
            }
        )
        self.assertIn("A. 甲", prompt)
        self.assertIn("B. 乙", prompt)


class ClinJudgmentTests(unittest.TestCase):
    def test_valid_judgment(self) -> None:
        scores = {
            side: {
                "fluency": 4,
                "relevance": 4,
                "completeness": 4,
                "medical_proficiency": 4,
            }
            for side in ("A", "B")
        }
        self.assertTrue(
            valid_judgment(
                {
                    "decision": "tie",
                    "scores": scores,
                    "hard_medical_error": {"A": False, "B": False},
                    "reason": "质量相当",
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
