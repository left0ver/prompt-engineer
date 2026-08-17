"""A small, reproducible implementation of Automatic Prompt Engineer (APE).

APE treats instruction design as black-box optimization: an inference model
proposes instructions from input/output demonstrations, then a target model is
scored on held-out selection examples and the best instruction is evaluated on
a final test split.

Run from the project root:
    uv run python -m code.automatic_prompt_engineer.experiment --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from code.common import call_model
from code.gsm8k import load_test_100

ModelCaller = Callable[[str], str]
BOXED_ANSWER = re.compile(r"\\boxed\s*\{\s*([-+]?\d[\d,，]*)\s*\}")
NUMBERED_INSTRUCTION = re.compile(r"^\s*(?:\d+[.)、:]|[-*])\s*(.+?)\s*$")


@dataclass(frozen=True)
class Sample:
    """One integer-answer GSM8K question."""

    sample_id: str
    question: str
    answer: int


EVALUATION_SET = load_test_100(Sample)
BASELINE_INSTRUCTION = "解决下面的数学应用题，并在最后一行用 \\boxed{整数} 给出答案。"


def parse_answer(raw_output: str) -> int | None:
    """Return the final boxed integer emitted by a target model."""
    matches = BOXED_ANSWER.findall(raw_output)
    if not matches:
        return None
    return int(matches[-1].replace(",", "").replace("，", ""))


def build_instruction_generation_prompt(
    demonstrations: Sequence[Sample], candidate_count: int
) -> str:
    """Ask the inference model to synthesize candidate task instructions.

    Answers are exposed only in this APE proposal stage, as input/output
    demonstrations. They are never embedded in target-model evaluation prompts.
    """
    if not demonstrations:
        raise ValueError("APE 指令生成至少需要一个输入/输出示例。")
    if candidate_count < 1:
        raise ValueError("candidate_count 必须至少为 1。")
    examples = "\n\n".join(
        f"输入：{sample.question}\n输出：{sample.answer}" for sample in demonstrations
    )
    return f"""你是提示词工程师。根据下面的输入/输出示例，归纳该任务，并写出 {candidate_count} 条不同、可直接给语言模型使用的中文任务指令。

每条指令必须说明如何解题，并要求最终答案单独写为 \\boxed{{整数}}。不要包含示例题的具体答案、不要给出解答，也不要添加解释。每行只输出一条指令，并以编号开头。

输入/输出示例：
{examples}
"""


def parse_instruction_candidates(raw_output: str, candidate_count: int) -> tuple[str, ...]:
    """Extract, normalize, and deduplicate candidate instructions from text."""
    if candidate_count < 1:
        raise ValueError("candidate_count 必须至少为 1。")
    candidates: list[str] = []
    seen: set[str] = set()
    for line in raw_output.splitlines():
        match = NUMBERED_INSTRUCTION.match(line)
        candidate = (match.group(1) if match else line).strip(" \t\"'“”")
        normalized = " ".join(candidate.split())
        key = normalized.casefold()
        if len(normalized) >= 8 and key not in seen:
            candidates.append(normalized)
            seen.add(key)
        if len(candidates) == candidate_count:
            break
    if not candidates:
        raise ValueError("推理模型没有返回可用的 APE 指令候选项。")
    return tuple(candidates)


def build_target_prompt(instruction: str, sample: Sample) -> str:
    """Render a candidate instruction for target-model execution."""
    if not instruction.strip():
        raise ValueError("instruction 不能为空。")
    return f"{instruction.strip()}\n\n问题：{sample.question}\n解答："


def evaluate_instruction(
    instruction: str,
    samples: Sequence[Sample],
    *,
    model_caller: ModelCaller,
) -> dict[str, Any]:
    """Score one instruction with exact-match accuracy on a fixed split."""
    if not samples:
        raise ValueError("评测样本不能为空。")
    records: list[dict[str, Any]] = []
    for sample in samples:
        raw_output = model_caller(build_target_prompt(instruction, sample))
        prediction = parse_answer(raw_output)
        records.append(
            {
                **asdict(sample),
                "raw_output": raw_output,
                "prediction": prediction,
                "correct": prediction == sample.answer,
            }
        )
    correct = sum(record["correct"] for record in records)
    return {
        "instruction": instruction,
        "total": len(records),
        "correct": correct,
        "invalid": sum(record["prediction"] is None for record in records),
        "accuracy": correct / len(records),
        "records": records,
    }


def select_best_instruction(candidate_reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Choose highest accuracy; preserve proposal order to resolve ties."""
    if not candidate_reports:
        raise ValueError("没有候选指令可供选择。")
    return max(candidate_reports, key=lambda report: report["accuracy"])


def run_ape(
    samples: Sequence[Sample] = EVALUATION_SET,
    *,
    demonstration_count: int = 4,
    selection_count: int = 16,
    candidate_count: int = 8,
    inference_caller: ModelCaller = call_model,
    target_caller: ModelCaller = call_model,
) -> dict[str, Any]:
    """Run proposal, selection, and final held-out APE evaluation.

    The dataset is deterministically split into demonstrations, a search split,
    and a non-overlapping final evaluation split.
    """
    if demonstration_count < 1 or selection_count < 1 or candidate_count < 1:
        raise ValueError("demonstration_count、selection_count 和 candidate_count 必须为正数。")
    if len(samples) <= demonstration_count + selection_count:
        raise ValueError("样本数必须大于演示数与选择集大小之和。")

    demonstrations = samples[:demonstration_count]
    selection_samples = samples[demonstration_count : demonstration_count + selection_count]
    test_samples = samples[demonstration_count + selection_count :]
    proposal_prompt = build_instruction_generation_prompt(demonstrations, candidate_count)
    proposal_output = inference_caller(proposal_prompt)
    candidates = parse_instruction_candidates(proposal_output, candidate_count)
    candidate_reports = [
        evaluate_instruction(candidate, selection_samples, model_caller=target_caller)
        for candidate in candidates
    ]
    selected = select_best_instruction(candidate_reports)
    ape_test = evaluate_instruction(selected["instruction"], test_samples, model_caller=target_caller)
    baseline_test = evaluate_instruction(BASELINE_INSTRUCTION, test_samples, model_caller=target_caller)

    return {
        "experiment": "automatic_prompt_engineer_gsm8k",
        "source": "https://www.promptingguide.ai/zh/techniques/ape",
        "created_at": datetime.now().astimezone().isoformat(),
        "inference_model": os.getenv("LLM_MODEL", "unknown"),
        "target_model": os.getenv("LLM_MODEL", "unknown"),
        "splits": {
            "demonstrations": [asdict(sample) for sample in demonstrations],
            "selection_sample_ids": [sample.sample_id for sample in selection_samples],
            "test_sample_ids": [sample.sample_id for sample in test_samples],
        },
        "proposal": {"prompt": proposal_prompt, "raw_output": proposal_output, "candidates": list(candidates)},
        "candidate_selection": candidate_reports,
        "selected_instruction": selected["instruction"],
        "final_evaluation": {"baseline": baseline_test, "ape": ape_test, "accuracy_delta": ape_test["accuracy"] - baseline_test["accuracy"]},
        "model_call_count": 1 + len(candidates) * len(selection_samples) + 2 * len(test_samples),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="自动提示工程师（APE）GSM8K 实验")
    parser.add_argument("--limit", type=_positive_int, default=100, help="使用前 N 条 GSM8K 样本")
    parser.add_argument("--demonstrations", type=_positive_int, default=4, help="指令生成示例数")
    parser.add_argument("--selection", type=_positive_int, default=16, help="候选指令选择集大小")
    parser.add_argument("--candidates", type=_positive_int, default=8, help="生成的候选指令数")
    parser.add_argument("--dry-run", action="store_true", help="仅打印生成阶段提示词")
    parser.add_argument("--output", type=Path, help="结果 JSON 路径")
    args = parser.parse_args()
    samples = EVALUATION_SET[: args.limit]
    if len(samples) <= args.demonstrations + args.selection:
        parser.error("--limit 必须大于 --demonstrations + --selection。")
    if args.dry_run:
        print(build_instruction_generation_prompt(samples[: args.demonstrations], args.candidates))
        calls = 1 + args.candidates * args.selection + 2 * (len(samples) - args.demonstrations - args.selection)
        print(f"正式运行将发起最多 {calls} 次模型调用。")
        return
    report = run_ape(samples, demonstration_count=args.demonstrations, selection_count=args.selection, candidate_count=args.candidates)
    output = args.output or Path(__file__).parent / "results" / f"ape_{datetime.now():%Y%m%d-%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    final = report["final_evaluation"]
    print(f"基线准确率：{final['baseline']['accuracy']:.1%}；APE：{final['ape']['accuracy']:.1%}；结果：{output.resolve()}")


if __name__ == "__main__":
    main()
