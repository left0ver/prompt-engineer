"""Offline tests for the Reflexion Actor/Evaluator/Reflection loop."""

from __future__ import annotations

import unittest

from code.reflexion_prompting.experiment import (
    ExactAnswerEvaluator,
    ReflexionAgent,
    build_actor_prompt,
    extract_final_answer,
)


class ReflexionExperimentTests(unittest.TestCase):
    def test_failed_attempt_becomes_memory_for_the_next_actor_trial(self) -> None:
        actor_prompts: list[str] = []
        outputs = iter(("推理错误\nFINAL: 68", "检查年龄差\nFINAL: 67"))

        def actor(prompt: str) -> str:
            actor_prompts.append(prompt)
            return next(outputs)

        reflection_prompts: list[str] = []

        def reflector(prompt: str) -> str:
            reflection_prompts.append(prompt)
            return "先固定年龄差为 3 岁，再代入当前年龄。"

        result = ReflexionAgent(ExactAnswerEvaluator("67"), actor=actor, reflector=reflector).run("年龄题")

        self.assertTrue(result.solved)
        self.assertEqual(result.answer, "67")
        self.assertEqual(len(result.attempts), 2)
        self.assertEqual(result.attempts[0].evaluation.reward, 0.0)
        self.assertIsNone(result.attempts[1].reflection)
        self.assertIn("年龄差为 3 岁", actor_prompts[1])
        self.assertIn("Evaluator reward：0", reflection_prompts[0])

    def test_memory_is_a_sliding_window_and_exhaustion_is_reported(self) -> None:
        reflections = iter(("记忆一", "记忆二", "记忆三"))
        agent = ReflexionAgent(
            ExactAnswerEvaluator("right"),
            actor=lambda _: "FINAL: wrong",
            reflector=lambda _: next(reflections),
            max_trials=3,
            memory_capacity=2,
        )
        result = agent.run("task")

        self.assertFalse(result.solved)
        self.assertEqual(result.memory, ("记忆二", "记忆三"))
        self.assertEqual([attempt.reflection for attempt in result.attempts], ["记忆一", "记忆二", "记忆三"])

    def test_prompt_and_final_parser_handle_empty_memory_and_last_final_line(self) -> None:
        self.assertIn("暂无", build_actor_prompt("task", ()))
        self.assertEqual(extract_final_answer("FINAL: first\nFINAL: second"), "second")
        with self.assertRaisesRegex(ValueError, "task 不能为空"):
            build_actor_prompt("  ", ())


if __name__ == "__main__":
    unittest.main()
