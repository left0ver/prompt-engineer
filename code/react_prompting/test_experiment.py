"""Offline tests for the ReAct agent loop."""

from __future__ import annotations

import unittest

from code.react_prompting.experiment import ReActAgent, calculator, parse_action


class ReActExperimentTests(unittest.TestCase):
    def test_agent_interleaves_action_and_observation_before_finishing(self) -> None:
        responses = iter(
            (
                "Thought 1: I need the stored value.\nAction 1: Search[rate]",
                "Thought 2: Now calculate it.\nAction 2: Calculator[29^0.23]",
                "Thought 3: I have the result.\nAction 3: Finish[2.16946]",
            )
        )
        prompts: list[str] = []

        def fake_model(prompt: str) -> str:
            prompts.append(prompt)
            return next(responses)

        agent = ReActAgent(fake_model, tools={"search": lambda _: "29"})
        result = agent.run("What is the result?")

        self.assertEqual(result.answer, "2.16946")
        self.assertEqual([step.action.name for step in result.steps], ["search", "calculator", "finish"])
        self.assertEqual(result.steps[0].observation, "29")
        self.assertIn("Observation 1: 29", prompts[1])
        self.assertIn("Observation 2: 答案: 2.16946", prompts[2])

    def test_parser_accepts_chinese_action_label_and_uses_last_action(self) -> None:
        action = parse_action("操作 1: Search[first]\n操作 2: Finish[done]")
        self.assertIsNotNone(action)
        self.assertEqual(action.name, "finish")
        self.assertEqual(action.input, "done")

    def test_calculator_rejects_code_execution_syntax(self) -> None:
        self.assertEqual(calculator("2^3"), "答案: 8")
        self.assertTrue(calculator("__import__('os').system('whoami')").startswith("计算错误:"))

    def test_agent_stops_when_model_never_finishes(self) -> None:
        agent = ReActAgent(
            lambda _: "Thought 1: keep looking\nAction 1: Search[x]",
            tools={"search": lambda _: "nothing"},
            max_steps=2,
        )
        with self.assertRaisesRegex(RuntimeError, "没有执行 Finish"):
            agent.run("question")


if __name__ == "__main__":
    unittest.main()
