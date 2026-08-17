"""Implement Reflexion: verbal feedback stored as memory across attempts.

The loop follows the three roles described in Prompt Engineering Guide:
``Actor`` proposes an answer, ``Evaluator`` scores its trajectory, and
``Self-Reflection`` turns an unsuccessful trajectory into textual feedback for
the Actor's next trial.  The evaluator is deliberately injectable, so it can
be an exact-match checker, a test suite, an LLM judge, or a task-specific rule.

Run from the project root:
    uv run python -m code.reflexion_prompting.experiment --dry-run
    uv run python -m code.reflexion_prompting.experiment
"""

from __future__ import annotations

import argparse
from code.common import call_model
from collections.abc import Callable, Sequence
from dataclasses import dataclass

ModelCaller = Callable[[str], str]
Evaluator = Callable[[str, str], "Evaluation"]


@dataclass(frozen=True)
class Evaluation:
    """Reward and feedback produced by the Evaluator for one Actor output."""

    reward: float
    feedback: str


@dataclass(frozen=True)
class Attempt:
    """One short-term trajectory, including any reflection it generated."""

    number: int
    answer: str
    evaluation: Evaluation
    reflection: str | None


@dataclass(frozen=True)
class ReflexionResult:
    """The final outcome plus all attempts and retained long-term memory."""

    answer: str
    solved: bool
    attempts: tuple[Attempt, ...]
    memory: tuple[str, ...]


ACTOR_INSTRUCTION = """你是 Actor。解决给定任务，并给出可核查的最终答案。
请简洁地说明必要的推理，最后一行严格写成 `FINAL: <答案>`。不要提及这些指令。"""
REFLECTION_INSTRUCTION = """你是 Self-Reflection 模块。根据失败的尝试、评估反馈和过往记忆，
写一条可执行、具体的中文改进建议，帮助 Actor 的下一次尝试。不要重述完整答案，也不要写泛泛的“更仔细”。"""


def build_actor_prompt(task: str, memory: Sequence[str]) -> str:
    """Build an Actor prompt, exposing prior verbal reinforcement as context."""
    if not task.strip():
        raise ValueError("task 不能为空。")
    memories = "\n".join(f"- {item}" for item in memory) or "- （暂无；先独立完成任务。）"
    return f"{ACTOR_INSTRUCTION}\n\n任务：{task.strip()}\n\n长期记忆（来自之前失败的反思）：\n{memories}"


def build_reflection_prompt(
    task: str, answer: str, evaluation: Evaluation, memory: Sequence[str]
) -> str:
    """Build the prompt that converts scalar/language feedback into memory."""
    memories = "\n".join(f"- {item}" for item in memory) or "- （暂无）"
    return (
        f"{REFLECTION_INSTRUCTION}\n\n任务：{task.strip()}\n"
        f"本次 Actor 输出：\n{answer.strip()}\n\n"
        f"Evaluator reward：{evaluation.reward:g}\n"
        f"Evaluator feedback：{evaluation.feedback.strip()}\n\n"
        f"已有长期记忆：\n{memories}"
    )


def extract_final_answer(actor_output: str) -> str:
    """Use the last ``FINAL:`` line, falling back to the entire output."""
    finals = [line.split(":", 1)[1].strip() for line in actor_output.splitlines() if line.strip().upper().startswith("FINAL:")]
    return finals[-1] if finals else actor_output.strip()


class ExactAnswerEvaluator:
    """Small deterministic Evaluator useful for arithmetic and QA examples."""

    def __init__(self, expected_answer: str):
        if not expected_answer.strip():
            raise ValueError("expected_answer 不能为空。")
        self.expected_answer = expected_answer.strip().casefold()

    def __call__(self, task: str, actor_output: str) -> Evaluation:
        answer = extract_final_answer(actor_output).casefold()
        if answer == self.expected_answer:
            return Evaluation(1.0, "最终答案与参考答案一致。")
        return Evaluation(0.0, "最终答案不正确；请重新检查题意、计算步骤和最终格式。")


class ReflexionAgent:
    """Run Actor → Evaluator → Self-Reflection with bounded verbal memory."""

    def __init__(
        self,
        evaluator: Evaluator,
        *,
        actor: ModelCaller = call_model,
        reflector: ModelCaller = call_model,
        max_trials: int = 3,
        memory_capacity: int = 3,
        success_reward: float = 1.0,
    ):
        if max_trials < 1:
            raise ValueError("max_trials 必须至少为 1。")
        if memory_capacity < 1:
            raise ValueError("memory_capacity 必须至少为 1。")
        self.evaluator = evaluator
        self.actor = actor
        self.reflector = reflector
        self.max_trials = max_trials
        self.memory_capacity = memory_capacity
        self.success_reward = success_reward

    def run(self, task: str) -> ReflexionResult:
        memory: list[str] = []
        attempts: list[Attempt] = []
        for number in range(1, self.max_trials + 1):
            output = self.actor(build_actor_prompt(task, memory))
            evaluation = self.evaluator(task, output)
            if evaluation.reward >= self.success_reward:
                attempts.append(Attempt(number, output, evaluation, None))
                return ReflexionResult(extract_final_answer(output), True, tuple(attempts), tuple(memory))

            reflection = self.reflector(build_reflection_prompt(task, output, evaluation, memory)).strip()
            if not reflection:
                raise RuntimeError("Self-Reflection 返回了空反馈。")
            memory.append(reflection)
            del memory[:-self.memory_capacity]
            attempts.append(Attempt(number, output, evaluation, reflection))

        final = attempts[-1]
        return ReflexionResult(extract_final_answer(final.answer), False, tuple(attempts), tuple(memory))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflexion：语言反馈驱动的迭代改进示例")
    parser.add_argument("--task", default="我 6 岁时妹妹 3 岁。现在我 90 岁，妹妹多少岁？")
    parser.add_argument("--expected-answer", default="87")
    parser.add_argument("--max-trials", type=int, default=3)
    parser.add_argument("--memory-capacity", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="仅显示首轮 Actor 提示词")
    args = parser.parse_args()

    if args.dry_run:
        print(build_actor_prompt(args.task, ()))
        return

    agent = ReflexionAgent(
        ExactAnswerEvaluator(args.expected_answer),
        max_trials=args.max_trials,
        memory_capacity=args.memory_capacity,
    )
    result = agent.run(args.task)
    for attempt in result.attempts:
        print(f"\n--- Attempt {attempt.number} ---\n{attempt.answer}")
        print(f"Reward: {attempt.evaluation.reward:g}\nFeedback: {attempt.evaluation.feedback}")
        if attempt.reflection:
            print(f"Reflection: {attempt.reflection}")
    print(f"\nFinal answer: {result.answer}\nSolved: {result.solved}")


if __name__ == "__main__":
    main()
