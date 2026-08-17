"""A small, inspectable implementation of the ReAct prompting pattern.

The agent alternates between a model-produced ``Thought``/``Action`` and a
locally executed tool ``Observation``.  This mirrors the ReAct trajectory from
the Prompt Engineering Guide while keeping the example deterministic and safe
to run: search is over a supplied local knowledge base and calculator input is
parsed with ``ast`` rather than evaluated with ``eval``.

Run from the project root:
    uv run python -m code.react_prompting.experiment --dry-run
    uv run python -m code.react_prompting.experiment --question "29 的 0.23 次方是多少？"
"""

from __future__ import annotations

import argparse
import ast
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

from code.common import call_model


SYSTEM_PROMPT = """You are a ReAct agent. Solve the user's question by alternating
between concise reasoning and tool use. You may use Search, Calculator, and
Finish. Never invent an Observation: observations are supplied by the system."""

ACTION_PATTERN = re.compile(
    r"(?:Action|操作)\s*\d*\s*:\s*"
    r"(?P<name>Search|Calculator|Finish)\s*\[\s*(?P<input>.*?)\s*\]",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class Action:
    """One action requested by the model."""

    name: str
    input: str


@dataclass(frozen=True)
class Step:
    """One completed ReAct turn, including the tool observation."""

    number: int
    model_output: str
    action: Action
    observation: str | None


@dataclass(frozen=True)
class ReActResult:
    """The final answer and the complete trajectory."""

    answer: str
    steps: tuple[Step, ...]


class Tool(Protocol):
    """A tool receives the action argument and returns an observation."""

    def __call__(self, tool_input: str) -> str: ...


ModelCaller = Callable[[str], str]


def parse_action(model_output: str) -> Action | None:
    """Return the last well-formed ReAct action in a model response.

    Taking the last match lets a response contain its thought and one or more
    illustrative lines without accidentally executing an earlier example.
    """
    matches = list(ACTION_PATTERN.finditer(model_output))
    if not matches:
        return None
    match = matches[-1]
    return Action(match["name"].lower(), match["input"].strip())


def _calculate(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    ):
        left, right = _calculate(node.left), _calculate(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        return left % right
    raise ValueError("只允许数字和 +、-、*、/、**、% 运算。")


def calculator(expression: str) -> str:
    """Safely calculate a basic arithmetic expression for a ReAct observation."""
    try:
        value = _calculate(ast.parse(expression.replace("^", "**"), mode="eval"))
        if not math.isfinite(value):
            raise ValueError("结果不是有限数字。")
        return f"答案: {value:g}"
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
        return f"计算错误: {error}"


class LocalSearch:
    """A tiny injectable knowledge base used as an external information source."""

    def __init__(self, documents: Mapping[str, str]):
        self.documents = dict(documents)

    def __call__(self, query: str) -> str:
        normalized_query = query.casefold().strip()
        matches = [
            f"{title}: {body}"
            for title, body in self.documents.items()
            if normalized_query in title.casefold() or normalized_query in body.casefold()
        ]
        if matches:
            return "\n".join(matches[:3])
        return "未找到相关资料；请尝试更具体的检索词。"


DEFAULT_SEARCH = LocalSearch(
    {
        "哈里·斯泰尔斯年龄": "哈里·斯泰尔斯出生于 1994 年 2 月 1 日。",
        "高平原": "高平原的海拔从 1800 到 7000 英尺（550 到 2130 米）不等。",
    }
)


def build_prompt(question: str, steps: tuple[Step, ...] = ()) -> str:
    """Render the current trajectory as the next model prompt."""
    if not question.strip():
        raise ValueError("question 不能为空。")
    lines = [
        SYSTEM_PROMPT,
        "Question: " + question.strip(),
        "Use this exact format for the next turn:",
        "Thought N: brief reason for the next action",
        "Action N: Search[query] | Calculator[expression] | Finish[final answer]",
    ]
    for step in steps:
        lines.extend(
            [
                f"\nThought {step.number} and Action {step.number}:",
                step.model_output.strip(),
            ]
        )
        if step.observation is not None:
            lines.append(f"Observation {step.number}: {step.observation}")
    return "\n".join(lines)


class ReActAgent:
    """Run the model/tool feedback loop until it emits ``Finish``."""

    def __init__(
        self,
        model_caller: ModelCaller = call_model,
        *,
        tools: Mapping[str, Tool] | None = None,
        max_steps: int = 6,
    ):
        if max_steps < 1:
            raise ValueError("max_steps 必须至少为 1。")
        self.model_caller = model_caller
        # Custom tools override defaults but do not accidentally remove the
        # companion tool (for example, tests commonly replace only Search).
        self.tools: dict[str, Tool] = {"search": DEFAULT_SEARCH, "calculator": calculator}
        if tools:
            self.tools.update(tools)
        self.max_steps = max_steps

    def run(self, question: str) -> ReActResult:
        steps: list[Step] = []
        for number in range(1, self.max_steps + 1):
            model_output = self.model_caller(build_prompt(question, tuple(steps)))
            action = parse_action(model_output)
            if action is None:
                raise RuntimeError("模型没有输出格式正确的 Action N: Tool[input]。")
            if action.name == "finish":
                steps.append(Step(number, model_output, action, None))
                return ReActResult(action.input, tuple(steps))

            tool = self.tools.get(action.name)
            observation = tool(action.input) if tool else f"未知工具: {action.name}"
            steps.append(Step(number, model_output, action, observation))
        raise RuntimeError(f"ReAct 在 {self.max_steps} 步内没有执行 Finish。")


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct 推理-行动-观察循环示例")
    parser.add_argument("--question", default="高平原的海拔范围是多少？")
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--dry-run", action="store_true", help="显示首轮提示词而不调用模型")
    args = parser.parse_args()

    if args.dry_run:
        print(build_prompt(args.question))
        return

    result = ReActAgent(max_steps=args.max_steps).run(args.question)
    for step in result.steps:
        print(f"\n--- Step {step.number} ---")
        print(step.model_output)
        if step.observation is not None:
            print(f"Observation {step.number}: {step.observation}")
    print(f"\nFinal answer: {result.answer}")


if __name__ == "__main__":
    main()
