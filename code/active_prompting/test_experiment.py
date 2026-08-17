"""Unit tests for the Active-Prompt workflow; no live model calls."""

from __future__ import annotations

import unittest

from code.active_prompting.experiment import (
    Annotation, Sample, build_inference_prompt, collect_candidates, disagreement,
    infer, select_uncertain,
)


class ActivePromptTests(unittest.TestCase):
    def test_disagreement_counts_invalid_answers(self) -> None:
        score = disagreement([12, 12, None, 10])
        self.assertEqual(score["score"], 0.5)
        self.assertEqual(score["answer_counts"], {"10": 1, "12": 2, "<invalid>": 1})

    def test_selection_prefers_highest_uncertainty_then_id(self) -> None:
        records = [
            {"sample_id": "b", "uncertainty": {"score": 0.5}},
            {"sample_id": "a", "uncertainty": {"score": 0.5}},
            {"sample_id": "c", "uncertainty": {"score": 0.2}},
        ]
        self.assertEqual([item["sample_id"] for item in select_uncertain(records, 2)], ["a", "b"])

    def test_collect_and_infer_form_a_two_stage_workflow(self) -> None:
        samples = (Sample("s1", "一加一是多少？", 2), Sample("s2", "二加二是多少？", 4))
        calls: dict[str, int] = {"s1": 0, "s2": 0}

        def fake_model(prompt: str, *, temperature: float | None = None) -> str:
            key = "s1" if "一加一" in prompt else "s2"
            calls[key] += 1
            if "模仿示例" in prompt:
                return "推理。\\boxed{2}" if key == "s1" else "推理。\\boxed{4}"
            return "\\boxed{2}" if key == "s1" else ("\\boxed{3}" if calls[key] % 2 else "\\boxed{4}")

        candidates = collect_candidates(samples, paths=2, temperature=0.7, model_caller=fake_model)
        selected = select_uncertain(candidates, 1)
        self.assertEqual(selected[0]["sample_id"], "s2")
        annotations = (Annotation("s2", "二加二是多少？", "2 加 2 等于 4。", 4),)
        self.assertIn("2 加 2 等于 4", build_inference_prompt(samples[0], annotations))
        result = infer(samples, annotations, temperature=0.0, model_caller=fake_model)
        self.assertEqual(result["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
