"""Unit tests that do not make real model calls."""

from __future__ import annotations

import unittest

from code.self_consistency_prompting.experiment import (
    Sample,
    majority_vote,
    parse_answer,
    run_ablation,
)


class SelfConsistencyExperimentTests(unittest.TestCase):
    def test_parse_answer_uses_last_boxed_integer(self) -> None:
        self.assertEqual(parse_answer("\\boxed{3}，最终为\\boxed{1,240}"), 1240)
        self.assertIsNone(parse_answer("没有使用 boxed 格式"))

    def test_majority_vote_reports_counts_and_stable_tie(self) -> None:
        vote = majority_vote([20, 19, 20, None, 19])
        self.assertEqual(vote["prediction"], 20)
        self.assertEqual(vote["vote_counts"], {"19": 2, "20": 2})
        self.assertTrue(vote["is_tie"])
        self.assertEqual(vote["agreement"], 0.5)

    def test_self_consistency_can_correct_a_single_path_error(self) -> None:
        samples = (
            Sample("easy", "简单题", 10),
            Sample("hard", "难题", 20),
        )
        sampled_calls = 0

        def fake_model(prompt: str, *, temperature: float | None = None) -> str:
            nonlocal sampled_calls
            if "简单题" in prompt:
                return "\\boxed{10}"
            if temperature == 0.0:
                return "\\boxed{19}"
            answers = (20, 20, 19)
            answer = answers[sampled_calls % len(answers)]
            sampled_calls += 1
            return f"推理路径。\\boxed{{{answer}}}"

        report = run_ablation(
            samples,
            paths=3,
            baseline_temperature=0.0,
            sampling_temperature=0.7,
            model_caller=fake_model,
        )

        self.assertEqual(report["summary"]["baseline"]["accuracy"], 0.5)
        self.assertEqual(report["summary"]["self_consistency"]["accuracy"], 1.0)
        self.assertEqual(report["summary"]["comparison"]["improved"], 1)
        self.assertEqual(report["model_call_count"], 8)


if __name__ == "__main__":
    unittest.main()
