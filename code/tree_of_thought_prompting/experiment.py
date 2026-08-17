"""Test whether Tree of Thoughts improves 24-game solving over CoT.

The implementation follows the ToT workflow described by Prompt Engineering
Guide: generate several intermediate thoughts, evaluate each partial state,
then use breadth-first search to retain the best states.  Final answers are
verified by Python rather than trusting the model's own judgement.

Run from the repository root:
    uv run python -m code.tree_of_thought_prompting.experiment --limit 3
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from code.common import call_model
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal

ModelCaller = Callable[[str], str]
Condition = Literal["cot", "tot_bfs"]
CONDITIONS: tuple[Condition, ...] = ("cot", "tot_bfs")
ACTION_PATTERN = re.compile(
    r"^\s*(-?\d+(?:/\d+)?)\s*([+\-*/])\s*(-?\d+(?:/\d+)?)\s*=\s*(-?\d+(?:/\d+)?)\s*$"
)
RATING_PATTERN = re.compile(r"\b(sure|maybe|impossible)\b", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"-?\d+(?:/\d+)?")


@dataclass(frozen=True)
class Puzzle:
    sample_id: str
    numbers: tuple[int, int, int, int]


@dataclass(frozen=True)
class State:
    values: tuple[Fraction, ...]
    steps: tuple[str, ...] = ()


# All puzzles have at least one solution.  They span easy and less-obvious
# combinations while staying independent of the prompts below.
PUZZLES: tuple[Puzzle, ...] = (
    Puzzle("24-01", (4, 5, 6, 10)),
    Puzzle("24-02", (1, 3, 4, 6)),
    Puzzle("24-03", (1, 5, 5, 5)),
    Puzzle("24-04", (2, 3, 7, 7)),
    Puzzle("24-05", (2, 4, 6, 8)),
    Puzzle("24-06", (3, 3, 8, 8)),
    Puzzle("24-07", (3, 4, 6, 7)),
    Puzzle("24-08", (4, 4, 7, 7)),
    Puzzle("24-09", (2, 2, 9, 9)),
    Puzzle("24-10", (1, 2, 7, 7)),
)

COT_PROMPT = """玩 24 点游戏：使用给定的四个数字各一次，并只使用 +、-、*、/ 和括号，构造值为 24 的表达式。
请逐步推理并检查每个算式。最后一行必须写 `FINAL: <表达式>`，不要使用其他数字。"""
PROPOSE_PROMPT = """你正在用思维树（ToT）解 24 点。一个“思维”是把当前两个数合并成一个中间结果。
对给定状态提出至多 {branching_factor} 个不同、合理的下一步。每行严格使用 `a op b = c`，其中 a、b 必须来自当前数字，c 必须正确；只用 + - * /。不要解释。"""
EVALUATE_PROMPT = """评估一个 24 点中间状态能否在剩余步骤中得到 24。只回复一个词：sure、maybe 或 impossible。
sure：很容易验证可完成；maybe：看起来仍可能完成；impossible：明显不能完成。"""


def _format_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _format_values(values: Sequence[Fraction]) -> str:
    return ", ".join(_format_fraction(value) for value in values)


def build_cot_prompt(puzzle: Puzzle) -> str:
    return f"{COT_PROMPT}\n数字：{', '.join(map(str, puzzle.numbers))}"


def build_propose_prompt(state: State, branching_factor: int) -> str:
    history = "\n".join(state.steps) if state.steps else "（尚未开始）"
    return (
        f"{PROPOSE_PROMPT.format(branching_factor=branching_factor)}\n"
        f"当前数字：{_format_values(state.values)}\n已有步骤：\n{history}"
    )


def build_evaluate_prompt(state: State) -> str:
    return f"{EVALUATE_PROMPT}\n当前剩余数字：{_format_values(state.values)}"


def _parse_fraction(text: str) -> Fraction:
    return Fraction(text)


def parse_actions(raw_output: str, state: State) -> tuple[State, ...]:
    """Parse and validate model actions; invalid or duplicate thoughts vanish."""
    children: list[State] = []
    seen: set[tuple[Fraction, ...]] = set()
    available = list(state.values)
    for line in raw_output.splitlines():
        match = ACTION_PATTERN.match(line)
        if not match:
            continue
        left, operator, right, stated_result = match.groups()
        try:
            left_value, right_value, result = map(_parse_fraction, (left, right, stated_result))
        except (ValueError, ZeroDivisionError):
            continue
        remaining = available.copy()
        try:
            remaining.remove(left_value)
            remaining.remove(right_value)
        except ValueError:
            continue
        if operator == "+":
            calculated = left_value + right_value
        elif operator == "-":
            calculated = left_value - right_value
        elif operator == "*":
            calculated = left_value * right_value
        elif right_value:
            calculated = left_value / right_value
        else:
            continue
        if calculated != result:
            continue
        remaining.append(result)
        values = tuple(sorted(remaining))
        if values in seen:
            continue
        seen.add(values)
        children.append(State(values, (*state.steps, f"{left} {operator} {right} = {stated_result}")))
    return tuple(children)


def parse_rating(raw_output: str) -> str:
    match = RATING_PATTERN.search(raw_output)
    return match.group(1).lower() if match else "impossible"


def score_rating(rating: str) -> int:
    return {"sure": 2, "maybe": 1, "impossible": 0}[rating]


def _safe_expression_value(expression: str) -> Fraction | None:
    """Evaluate a candidate expression using only numeric binary operations."""
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except SyntaxError:
        return None

    def visit(item: ast.expr) -> Fraction:
        if isinstance(item, ast.Constant) and isinstance(item.value, int):
            return Fraction(item.value)
        if isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub):
            return -visit(item.operand)
        if isinstance(item, ast.BinOp):
            left, right = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.Add): return left + right
            if isinstance(item.op, ast.Sub): return left - right
            if isinstance(item.op, ast.Mult): return left * right
            if isinstance(item.op, ast.Div) and right: return left / right
        raise ValueError("不允许的表达式")

    try:
        return visit(node)
    except (ValueError, ZeroDivisionError):
        return None


def verify_expression(raw_output: str, puzzle: Puzzle) -> tuple[bool, str | None]:
    """Find a valid 24 expression in output and ensure it uses input digits exactly."""
    candidates = [line.removeprefix("FINAL:").strip() for line in raw_output.splitlines()]
    candidates.append(raw_output.strip())
    required = sorted(puzzle.numbers)
    for expression in candidates:
        numbers = [int(token) for token in NUMBER_PATTERN.findall(expression) if "/" not in token]
        if sorted(numbers) != required:
            continue
        if _safe_expression_value(expression) == 24:
            return True, expression
    return False, None


def solve_tot(
    puzzle: Puzzle, *, branching_factor: int, beam_width: int, model_caller: ModelCaller
) -> dict[str, Any]:
    """Perform three ToT/BFS layers, retaining evaluator-ranked thoughts."""
    frontier = [State(tuple(Fraction(number) for number in puzzle.numbers))]
    trace: list[dict[str, Any]] = []
    calls = 0
    for depth in range(3):
        ranked: list[tuple[int, State, str]] = []
        for parent in frontier:
            proposed = model_caller(build_propose_prompt(parent, branching_factor))
            calls += 1
            children = parse_actions(proposed, parent)
            for child in children:
                rating = parse_rating(model_caller(build_evaluate_prompt(child)))
                calls += 1
                ranked.append((score_rating(rating), child, rating))
        ranked.sort(key=lambda row: (-row[0], row[1].values, row[1].steps))
        kept = ranked[:beam_width]
        trace.append({
            "depth": depth + 1,
            "generated": len(ranked),
            "kept": [{"rating": rating, "state": _format_values(state.values), "steps": list(state.steps)} for _, state, rating in kept],
        })
        frontier = [state for _, state, _ in kept]
        if not frontier:
            break
    solved = next((state for state in frontier if state.values == (Fraction(24),)), None)
    return {"correct": solved is not None, "steps": list(solved.steps) if solved else [], "trace": trace, "model_calls": calls}


def evaluate_condition(condition: Condition, puzzles: Sequence[Puzzle], *, branching_factor: int, beam_width: int, model_caller: ModelCaller) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for puzzle in puzzles:
        if condition == "tot_bfs":
            result = solve_tot(puzzle, branching_factor=branching_factor, beam_width=beam_width, model_caller=model_caller)
            records.append({**asdict(puzzle), "correct": result["correct"], "solution_steps": result["steps"], "search_trace": result["trace"], "model_calls": result["model_calls"]})
        else:
            output = model_caller(build_cot_prompt(puzzle))
            correct, expression = verify_expression(output, puzzle)
            records.append({**asdict(puzzle), "correct": correct, "raw_output": output, "expression": expression, "model_calls": 1})
    return {"name": condition, "total": len(records), "correct": sum(row["correct"] for row in records), "accuracy": sum(row["correct"] for row in records) / len(records), "model_calls": sum(row["model_calls"] for row in records), "records": records}


def paired_comparison(baseline: dict[str, Any], treatment: dict[str, Any]) -> dict[str, Any]:
    pairs = zip(baseline["records"], treatment["records"], strict=True)
    improved = regressed = both_correct = both_wrong = 0
    for before, after in pairs:
        if not before["correct"] and after["correct"]: improved += 1
        elif before["correct"] and not after["correct"]: regressed += 1
        elif before["correct"]: both_correct += 1
        else: both_wrong += 1
    return {"accuracy_delta": treatment["accuracy"] - baseline["accuracy"], "improved": improved, "regressed": regressed, "both_correct": both_correct, "both_wrong": both_wrong, "net_improved": improved - regressed, "extra_model_calls": treatment["model_calls"] - baseline["model_calls"]}


def run_experiment(puzzles: Sequence[Puzzle] = PUZZLES, *, branching_factor: int = 3, beam_width: int = 5, model_caller: ModelCaller = call_model) -> dict[str, Any]:
    if not puzzles: raise ValueError("评测题不能为空。")
    if branching_factor < 1 or beam_width < 1: raise ValueError("branching_factor 和 beam_width 必须为正整数。")
    conditions = {condition: evaluate_condition(condition, puzzles, branching_factor=branching_factor, beam_width=beam_width, model_caller=model_caller) for condition in CONDITIONS}
    return {"experiment": "tree_of_thoughts_24_game", "created_at": datetime.now().astimezone().isoformat(), "model": os.getenv("LLM_MODEL", "unknown"), "task": "24点；最终表达式由程序严格验算", "parameters": {"branching_factor": branching_factor, "beam_width": beam_width, "max_depth": 3}, "conditions": conditions, "comparisons": {"tot_bfs_vs_cot": paired_comparison(conditions["cot"], conditions["tot_bfs"])}}


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1: raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="24点上的 ToT BFS 对比实验")
    parser.add_argument("--limit", type=_positive_int, help="只运行前 N 道题")
    parser.add_argument("--branching-factor", type=_positive_int, default=5, help="每个状态要求的候选思维数")
    parser.add_argument("--beam-width", type=_positive_int, default=10, help="每层保留的最高评分状态数")
    parser.add_argument("--output", type=Path, help="结果 JSON 路径")
    parser.add_argument("--dry-run", action="store_true", help="仅预览提示，不调用模型")
    args = parser.parse_args()
    puzzles = PUZZLES[: args.limit] if args.limit else PUZZLES
    if args.dry_run:
        puzzle = puzzles[0]
        print("=== cot ===\n" + build_cot_prompt(puzzle))
        initial = State(tuple(Fraction(number) for number in puzzle.numbers))
        print("\n=== ToT proposal ===\n" + build_propose_prompt(initial, args.branching_factor))
        print("\n=== ToT evaluation ===\n" + build_evaluate_prompt(initial))
        print("\nToT 调用次数取决于每层有效候选数；可先使用 --limit 1 试运行。")
        return
    report = run_experiment(puzzles, branching_factor=args.branching_factor, beam_width=args.beam_width)
    output = args.output or Path(__file__).parent / "results" / f"tot_24_{datetime.now().astimezone():%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, result in report["conditions"].items(): print(f"{name}: {result['correct']}/{result['total']} = {result['accuracy']:.1%}, calls={result['model_calls']}")
    print(f"结果已保存：{output}")


if __name__ == "__main__":
    main()
