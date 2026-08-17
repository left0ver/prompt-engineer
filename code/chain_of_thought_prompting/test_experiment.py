"""Unit tests that do not make real model calls."""

from __future__ import annotations

import unittest

from code.chain_of_thought_prompting.experiment import (
    AUTO_COT_GENERATION_INSTRUCTION,
    MANUAL_DEMONSTRATIONS,
    GeneratedDemonstration,
    Sample,
    build_prompt,
    parse_answer,
    run_ablation,
    select_auto_cot_representatives,
)


class ChainOfThoughtExperimentTests(unittest.TestCase):
    def test_manual_cot_only_adds_reasoning_to_answer_examples(self) -> None:
        sample = Sample("test", "测试题目", 42)
        baseline = build_prompt(sample, "answer_only_few_shot")
        treatment = build_prompt(sample, "manual_cot")

        for demonstration in MANUAL_DEMONSTRATIONS:
            self.assertIn(demonstration.problem, baseline)
            self.assertIn(demonstration.problem, treatment)
            self.assertIn(f"\\boxed{{{demonstration.answer}}}", baseline)
            self.assertIn(f"\\boxed{{{demonstration.answer}}}", treatment)
            self.assertNotIn(demonstration.reasoning, baseline)
            self.assertIn(demonstration.reasoning, treatment)

    def test_zero_shot_cot_only_adds_trigger(self) -> None:
        sample = Sample("test", "测试题目", 42)
        direct = build_prompt(sample, "direct")
        zero_shot_cot = build_prompt(sample, "zero_shot_cot")

        self.assertTrue(zero_shot_cot.startswith(direct))
        self.assertEqual(zero_shot_cot.removeprefix(direct), "\n让我们逐步思考。")

    def test_auto_cot_requires_generated_demonstrations(self) -> None:
        sample = Sample("test", "测试题目", 42)
        with self.assertRaises(ValueError):
            build_prompt(sample, "auto_cot")

        demonstration = GeneratedDemonstration(
            "auto", "示例题", 1, "步骤。\n\\boxed{1}", 1, True
        )
        prompt = build_prompt(
            sample, "auto_cot", auto_demonstrations=(demonstration,)
        )
        self.assertIn("自动生成示例", prompt)
        self.assertIn(demonstration.generated_solution, prompt)

    def test_representative_selection_is_deterministic_and_unique(self) -> None:
        candidates = (
            Sample("a", "购买3件商品，每件5元，总价多少？", 15),
            Sample("b", "长方形长8米宽4米，面积多少？", 32),
            Sample("c", "鸡兔共有20只，共60条腿，兔有多少？", 10),
            Sample("d", "100元打八折是多少钱？", 80),
        )
        first = select_auto_cot_representatives(candidates, 3)
        second = select_auto_cot_representatives(candidates, 3)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len({sample.sample_id for sample in first}), 3)

    def test_parse_answer_uses_last_boxed_answer(self) -> None:
        self.assertEqual(parse_answer("计算得到\\boxed{20}。\n\\boxed{1,240}"), 1240)
        self.assertEqual(parse_answer("\\boxed{ -8 }"), -8)
        self.assertIsNone(parse_answer("最终答案是42"))

    def test_full_comparison_includes_all_conditions(self) -> None:
        samples = (
            Sample("simple", "简单题", 10),
            Sample("hard", "较难题", 20),
        )
        candidates = (
            Sample("candidate-a", "候选加法题", 10),
            Sample("candidate-b", "候选面积题", 10),
        )

        def fake_model(prompt: str) -> str:
            if prompt.startswith(AUTO_COT_GENERATION_INSTRUCTION):
                return "自动生成的计算步骤。\n\\boxed{10}"
            if "简单题" in prompt:
                return "\\boxed{10}"
            uses_reasoning = any(
                demonstration.reasoning in prompt
                for demonstration in MANUAL_DEMONSTRATIONS
            )
            uses_reasoning = uses_reasoning or "让我们逐步思考。" in prompt
            uses_reasoning = uses_reasoning or "自动生成示例" in prompt
            return "推理。\n\\boxed{20}" if uses_reasoning else "\\boxed{19}"

        report = run_ablation(
            samples,
            cluster_count=2,
            auto_cot_candidates=candidates,
            model_caller=fake_model,
        )

        self.assertEqual(
            set(report["conditions"]),
            {
                "direct",
                "answer_only_few_shot",
                "manual_cot",
                "zero_shot_cot",
                "auto_cot",
            },
        )
        self.assertEqual(report["conditions"]["direct"]["accuracy"], 0.5)
        self.assertEqual(report["conditions"]["manual_cot"]["accuracy"], 1.0)
        self.assertEqual(report["conditions"]["zero_shot_cot"]["accuracy"], 1.0)
        self.assertEqual(report["conditions"]["auto_cot"]["accuracy"], 1.0)
        self.assertEqual(report["model_call_count"], 12)
        self.assertEqual(
            report["comparisons"]["zero_shot_cot_vs_direct"]["improved"], 1
        )


if __name__ == "__main__":
    unittest.main()
