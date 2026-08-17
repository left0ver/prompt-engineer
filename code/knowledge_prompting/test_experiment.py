import unittest

from code.knowledge_prompting.experiment import (
    Sample,
    is_correct,
    parse_answer,
    run_ablation,
)


class GeneratedKnowledgeExperimentTests(unittest.TestCase):
    def test_parse_answer_uses_last_boxed_answer(self):
        self.assertEqual(parse_answer("\\boxed{first} \\boxed{Jane Austen}"), "Jane Austen")

    def test_answer_matching_accepts_aliases_and_ignores_articles(self):
        sample = Sample("a", "question", "Vincent van Gogh", ("van Gogh",))
        self.assertTrue(is_correct("The van Gogh", sample))

    def test_ablation_records_knowledge(self):
        samples = (Sample("a", "first", "first answer"), Sample("b", "second", "second answer"))

        def fake(prompt):
            if "list 2 relevant" in prompt:
                return "relation"
            if "Generated knowledge" in prompt:
                return "\\boxed{first answer}" if "first" in prompt else "\\boxed{second answer}"
            return "\\boxed{first answer}"

        report = run_ablation(samples, model_caller=fake)
        self.assertEqual(report["comparison"]["improved"], 1)
        self.assertEqual(report["conditions"]["generated_knowledge"]["records"][0]["generated_knowledge"], "relation")


if __name__ == "__main__":
    unittest.main()
