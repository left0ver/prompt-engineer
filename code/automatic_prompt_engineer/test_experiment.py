"""Offline tests for the APE experiment."""

from __future__ import annotations

import unittest

from code.automatic_prompt_engineer.experiment import (
    BASELINE_INSTRUCTION,
    Sample,
    build_instruction_generation_prompt,
    parse_answer,
    parse_instruction_candidates,
    run_ape,
)


class AutomaticPromptEngineerTests(unittest.TestCase):
    def test_generation_prompt_contains_only_demonstration_io(self) -> None:
        prompt = build_instruction_generation_prompt((Sample("one", "题目一", 12),), 2)
        self.assertIn("输入：题目一", prompt)
        self.assertIn("输出：12", prompt)
        self.assertIn("2 条不同", prompt)

    def test_candidate_parser_deduplicates_and_strips_numbering(self) -> None:
        candidates = parse_instruction_candidates(
            "1. 解题并用 \\boxed{整数} 作答。\n2. 解题并用 \\boxed{整数} 作答。\n3、仔细计算，最后写 \\boxed{整数}。",
            3,
        )
        self.assertEqual(len(candidates), 2)
        self.assertTrue(candidates[0].startswith("解题"))

    def test_parse_answer_uses_the_last_box(self) -> None:
        self.assertEqual(parse_answer("\\boxed{3}\n\\boxed{1,024}"), 1024)
        self.assertIsNone(parse_answer("答案是 1024"))

    def test_run_ape_selects_best_candidate_and_keeps_test_held_out(self) -> None:
        samples = tuple(Sample(str(index), f"题目{index}", index) for index in range(1, 8))

        def inference(_: str) -> str:
            return "1. BAD 指令，最后用 \\boxed{整数}。\n2. GOOD 指令，最后用 \\boxed{整数}。"

        def target(prompt: str) -> str:
            number = int(prompt.split("题目")[-1].split("\n")[0])
            if "GOOD" in prompt or prompt.startswith(BASELINE_INSTRUCTION):
                return f"\\boxed{{{number}}}"
            return "\\boxed{0}"

        report = run_ape(samples, demonstration_count=2, selection_count=2, candidate_count=2, inference_caller=inference, target_caller=target)
        self.assertIn("GOOD", report["selected_instruction"])
        self.assertEqual(report["final_evaluation"]["ape"]["accuracy"], 1.0)
        self.assertEqual(report["splits"]["test_sample_ids"], ["5", "6", "7"])
        self.assertEqual(report["model_call_count"], 11)


if __name__ == "__main__":
    unittest.main()
