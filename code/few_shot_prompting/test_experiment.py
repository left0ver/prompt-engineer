"""Unit tests that do not make real model calls."""

from __future__ import annotations

import unittest

from code.few_shot_prompting.experiment import (
    EVALUATION_SET,
    FEW_SHOT_EXAMPLES,
    Sample,
    build_prompt,
    parse_label,
    run_ablation,
)


class FewShotExperimentTests(unittest.TestCase):
    def test_ablation_only_adds_demonstrations(self) -> None:
        sample = Sample("test", "这次体验很不错。", "正面")
        baseline = build_prompt(sample)
        treatment = build_prompt(sample, FEW_SHOT_EXAMPLES)

        self.assertIn(sample.text, baseline)
        self.assertNotIn("示例：", baseline)
        self.assertIn("示例：", treatment)
        self.assertTrue(treatment.startswith(baseline.split("\n\n文本：")[0]))

    def test_parse_label_rejects_ambiguous_or_missing_output(self) -> None:
        self.assertEqual(parse_label("  正面\n"), "正面")
        self.assertEqual(parse_label("负面"), "负面")
        self.assertIsNone(parse_label("标签：正面"))
        self.assertIsNone(parse_label("这段文本是正面情感"))

    def test_evaluation_set_has_ten_balanced_held_out_examples(self) -> None:
        self.assertEqual(len(EVALUATION_SET), 10)
        self.assertEqual({sample.label for sample in EVALUATION_SET}, {"正面", "负面"})
        self.assertEqual(
            sum(sample.label == "正面" for sample in EVALUATION_SET),
            sum(sample.label == "负面" for sample in EVALUATION_SET),
        )
        self.assertTrue(
            {sample.text for sample in EVALUATION_SET}.isdisjoint(
                sample.text for sample in FEW_SHOT_EXAMPLES
            )
        )

    def test_run_ablation_reports_paired_improvement(self) -> None:
        samples = (
            Sample("one", "第一条文本", "正面"),
            Sample("two", "第二条文本", "负面"),
        )

        def fake_model(prompt: str) -> str:
            if "示例：" not in prompt:
                return "正面"
            return "正面" if "第一条文本" in prompt else "负面"

        report = run_ablation(samples, model_caller=fake_model)

        self.assertEqual(report["conditions"]["baseline"]["accuracy"], 0.5)
        self.assertEqual(report["conditions"]["treatment"]["accuracy"], 1.0)
        self.assertEqual(report["comparison"]["improved"], 1)
        self.assertEqual(report["comparison"]["regressed"], 0)


if __name__ == "__main__":
    unittest.main()
