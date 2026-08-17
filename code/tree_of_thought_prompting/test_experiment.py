"""Unit tests for the ToT experiment; no real model calls."""

import unittest
from fractions import Fraction

from code.tree_of_thought_prompting.experiment import Puzzle, State, parse_actions, solve_tot, verify_expression


class TreeOfThoughtTests(unittest.TestCase):
    def test_parser_rejects_invalid_or_unavailable_actions(self) -> None:
        state = State((Fraction(1), Fraction(3), Fraction(4), Fraction(6)))
        children = parse_actions("6 / 3 = 2\n9 + 1 = 10\n4 * 1 = 5", state)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].values, (Fraction(1), Fraction(2), Fraction(4)))

    def test_expression_verification_requires_all_input_numbers(self) -> None:
        puzzle = Puzzle("x", (1, 3, 4, 6))
        self.assertEqual(verify_expression("FINAL: 6 / (1 - 3 / 4)", puzzle), (True, "6 / (1 - 3 / 4)"))
        self.assertEqual(verify_expression("FINAL: 24", puzzle)[0], False)

    def test_bfs_can_follow_valid_model_thoughts(self) -> None:
        puzzle = Puzzle("x", (1, 3, 4, 6))
        def fake_model(prompt: str) -> str:
            if "只回复一个词" in prompt: return "sure"
            if "当前数字：1, 3, 4, 6" in prompt: return "3 / 4 = 3/4"
            if "当前数字：3/4, 1, 6" in prompt: return "1 - 3/4 = 1/4"
            if "当前数字：1/4, 6" in prompt: return "6 / 1/4 = 24"
            return ""
        result = solve_tot(puzzle, branching_factor=3, beam_width=5, model_caller=fake_model)
        self.assertTrue(result["correct"])
        self.assertEqual(len(result["steps"]), 3)


if __name__ == "__main__":
    unittest.main()
