"""Generated-knowledge prompting on a small factual knowledge-QA set."""

from __future__ import annotations

import argparse
import json
import os
from code.common import call_model
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import re
import unicodedata

ModelCaller = Callable[[str], str]
BOXED = re.compile(r"\\boxed\s*\{\s*(.*?)\s*\}", re.DOTALL)
DATA_PATH = Path(__file__).parents[1] / "data" / "knowledge_qa_10.json"


@dataclass(frozen=True)
class Sample:
    sample_id: str
    question: str
    answer: str
    aliases: tuple[str, ...] = ()


def load_evaluation_set(path: Path = DATA_PATH) -> tuple[Sample, ...]:
    """Load the fixed factual QA set used by this experiment."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    samples = tuple(
        Sample(row["id"], row["question"], row["answer"], tuple(row.get("aliases", [])))
        for row in payload["samples"]
    )
    if len(samples) != 10:
        raise RuntimeError(f"期望知识问答集包含 10 条样本，实际为 {len(samples)} 条。")
    return samples


EVALUATION_SET = load_evaluation_set()


def parse_answer(text: str) -> str | None:
    matches = BOXED.findall(text)
    return matches[-1].strip() if matches and matches[-1].strip() else None


def normalize_answer(text: str) -> str:
    """Use TriviaQA-style case/punctuation-insensitive exact matching."""
    normalized = unicodedata.normalize("NFKD", text).casefold()
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    return " ".join(re.sub(r"[^\w\s]", " ", normalized).split())


def is_correct(prediction: str | None, sample: Sample) -> bool:
    if prediction is None:
        return False
    accepted = (sample.answer, *sample.aliases)
    return normalize_answer(prediction) in {normalize_answer(answer) for answer in accepted}


def build_direct_prompt(sample: Sample) -> str:
    return f"Answer this factual knowledge question. Give only the short answer, enclosed in \\boxed{{}}.\n\nQuestion: {sample.question}\nAnswer:"


def build_knowledge_prompt(sample: Sample, knowledge_count: int) -> str:
    if knowledge_count < 1:
        raise ValueError("knowledge_count 必须至少为 1。")
    return f"For this factual knowledge question, list {knowledge_count} relevant facts, entities, or relations needed to answer it. Do not state the final answer.\n\nQuestion: {sample.question}\nKnowledge:"


def build_answer_with_knowledge_prompt(sample: Sample, knowledge: str) -> str:
    if not knowledge.strip():
        raise ValueError("生成的知识不能为空。")
    return f"Answer the factual knowledge question using the generated knowledge. Give only the short answer, enclosed in \\boxed{{}}.\n\nQuestion: {sample.question}\n\nGenerated knowledge:\n{knowledge}\n\nAnswer:"


def _evaluate(
    samples: Sequence[Sample], trials: int, caller: ModelCaller, count: int | None
) -> dict[str, Any]:
    records = []
    for trial in range(1, trials + 1):
        for sample in samples:
            knowledge = None
            if count is None:
                output = caller(build_direct_prompt(sample))
            else:
                knowledge = caller(build_knowledge_prompt(sample, count))
                output = caller(build_answer_with_knowledge_prompt(sample, knowledge))
            prediction = parse_answer(output)
            records.append(
                {
                    "trial": trial,
                    **asdict(sample),
                    "generated_knowledge": knowledge,
                    "raw_output": output,
                    "prediction": prediction,
                    "correct": is_correct(prediction, sample),
                }
            )
    correct = sum(r["correct"] for r in records)
    return {
        "total": len(records),
        "correct": correct,
        "invalid": sum(r["prediction"] is None for r in records),
        "accuracy": correct / len(records),
        "records": records,
    }


def compare_paired(
    base: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, int | float]:
    left = {(r["trial"], r["sample_id"]): r["correct"] for r in base["records"]}
    right = {(r["trial"], r["sample_id"]): r["correct"] for r in treatment["records"]}
    if left.keys() != right.keys():
        raise ValueError("两个条件的样本不一致。")
    improved = sum(not left[key] and right[key] for key in left)
    regressed = sum(left[key] and not right[key] for key in left)
    return {
        "accuracy_delta": treatment["accuracy"] - base["accuracy"],
        "improved": improved,
        "regressed": regressed,
        "net_improved": improved - regressed,
    }


def run_ablation(
    samples: Sequence[Sample] = EVALUATION_SET,
    *,
    trials: int = 1,
    knowledge_count: int = 2,
    model_caller: ModelCaller = call_model,
) -> dict[str, Any]:
    if not samples or trials < 1 or knowledge_count < 1:
        raise ValueError("样本、trials 和 knowledge_count 必须有效。")
    direct = _evaluate(samples, trials, model_caller, None)
    generated = _evaluate(samples, trials, model_caller, knowledge_count)
    return {
        "experiment": "generated_knowledge_prompting_factual_qa",
        "dataset": "knowledge_qa_10",
        "source": "local fixed factual knowledge-QA set (code/data/knowledge_qa_10.json)",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": os.getenv("LLM_MODEL", "unknown"),
        "sample_count": len(samples),
        "trials": trials,
        "knowledge_count": knowledge_count,
        "model_call_count": len(samples) * trials * 3,
        "conditions": {"direct_baseline": direct, "generated_knowledge": generated},
        "comparison": compare_paired(direct, generated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="事实知识问答的生成知识提示实验")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--knowledge-count", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    samples = EVALUATION_SET[: args.limit]
    if args.dry_run:
        print(build_direct_prompt(samples[0]))
        print(
            "\n--- knowledge stage ---\n"
            + build_knowledge_prompt(samples[0], args.knowledge_count)
        )
        return
    report = run_ablation(
        samples, trials=args.trials, knowledge_count=args.knowledge_count
    )
    output = (
        args.output
        or Path(__file__).parent
        / "results"
        / f"generated_knowledge_{datetime.now():%Y%m%d-%H%M%S}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"直接回答：{report['conditions']['direct_baseline']['accuracy']:.1%}；生成知识：{report['conditions']['generated_knowledge']['accuracy']:.1%}\n结果：{output.resolve()}"
    )


if __name__ == "__main__":
    main()
