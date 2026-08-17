"""Active-Prompt: select uncertain problems, annotate them, then infer.

The workflow deliberately has two commands.  ``collect`` produces a JSON file
for a human to write the selected chain-of-thought demonstrations; ``infer``
uses the completed file as few-shot demonstrations.  This keeps the human
annotation step in Active-Prompt real instead of replacing it with an oracle.

Run from the project root:
    uv run python -m code.active_prompting.experiment collect --limit 20
    # Fill the generated JSON's reasoning field, then:
    uv run python -m code.active_prompting.experiment infer --annotations PATH
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from code.common import call_model
from code.gsm8k import load_test_100

ModelCaller = Callable[..., str]
BOXED_ANSWER = re.compile(r"\\boxed\s*\{\s*([-+]?\d[\d,，]*)\s*\}")
INITIAL_INSTRUCTION = """解决这道算术应用题。写出简短推理过程。
最后一行必须严格写成：\\boxed{整数}。"""
INFERENCE_INSTRUCTION = """模仿示例解决算术应用题，写出简短推理过程。
最后一行必须严格写成：\\boxed{整数}。"""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    question: str
    answer: int


@dataclass(frozen=True)
class Annotation:
    sample_id: str
    question: str
    reasoning: str
    answer: int


EVALUATION_SET = load_test_100(Sample)


def parse_answer(raw_output: str) -> int | None:
    """Extract the final boxed integer emitted by a model."""
    matches = BOXED_ANSWER.findall(raw_output)
    return int(matches[-1].replace(",", "").replace("，", "")) if matches else None


def build_initial_prompt(sample: Sample) -> str:
    return f"{INITIAL_INSTRUCTION}\n\nQ：{sample.question}\nA："


def disagreement(predictions: Sequence[int | None]) -> dict[str, Any]:
    """Measure answer inconsistency as 1 minus the modal-answer proportion.

    Invalid outputs count as an additional distinct answer.  Thus a question
    with unusable generations is surfaced for annotation rather than silently
    appearing certain.
    """
    if not predictions:
        raise ValueError("至少需要一个采样答案。")
    labels = [str(answer) if answer is not None else "<invalid>" for answer in predictions]
    counts = Counter(labels)
    modal_count = max(counts.values())
    return {
        "score": 1 - modal_count / len(predictions),
        "answer_counts": dict(sorted(counts.items())),
        "modal_agreement": modal_count / len(predictions),
        "invalid_count": counts.get("<invalid>", 0),
    }


def select_uncertain(records: Sequence[dict[str, Any]], budget: int) -> list[dict[str, Any]]:
    """Pick the highest-disagreement questions with deterministic tie breaks."""
    if budget < 1:
        raise ValueError("annotation budget 必须至少为 1。")
    if budget > len(records):
        raise ValueError("annotation budget 不能大于候选问题数。")
    return sorted(
        records,
        key=lambda record: (-record["uncertainty"]["score"], record["sample_id"]),
    )[:budget]


def collect_candidates(
    samples: Sequence[Sample], *, paths: int, temperature: float | None,
    model_caller: ModelCaller = call_model,
) -> list[dict[str, Any]]:
    """Sample each candidate ``paths`` times and retain its uncertainty data."""
    if paths < 2:
        raise ValueError("paths 必须至少为 2，才能衡量不一致性。")
    records: list[dict[str, Any]] = []
    for sample in samples:
        prompt = build_initial_prompt(sample)
        outputs = [model_caller(prompt, temperature=temperature) for _ in range(paths)]
        predictions = [parse_answer(output) for output in outputs]
        records.append({
            **asdict(sample),
            "paths": [{"raw_output": output, "prediction": prediction} for output, prediction in zip(outputs, predictions, strict=True)],
            "uncertainty": disagreement(predictions),
        })
    return records


def annotation_template(selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Create the editable hand-off file for human CoT annotations."""
    return {
        "format": "active-prompt-human-annotations-v1",
        "instructions": "请为每题填写 reasoning（简短、正确的逐步推理），并核对 answer 后保存。",
        "annotations": [
            {
                "sample_id": record["sample_id"], "question": record["question"],
                "reasoning": "", "answer": record["answer"],
                "uncertainty": record["uncertainty"],
            }
            for record in selected
        ],
    }


def load_annotations(path: Path) -> tuple[Annotation, ...]:
    """Load a completed annotation hand-off file and reject incomplete entries."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("annotations")
    if not isinstance(rows, list) or not rows:
        raise ValueError("标注文件必须包含非空 annotations 数组。")
    annotations: list[Annotation] = []
    for index, row in enumerate(rows, start=1):
        try:
            annotation = Annotation(
                sample_id=str(row["sample_id"]), question=str(row["question"]),
                reasoning=str(row["reasoning"]).strip(), answer=int(row["answer"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"第 {index} 条标注格式无效。") from error
        if not annotation.reasoning:
            raise ValueError(f"第 {index} 条标注尚未填写 reasoning。")
        annotations.append(annotation)
    return tuple(annotations)


def build_inference_prompt(sample: Sample, annotations: Sequence[Annotation]) -> str:
    if not annotations:
        raise ValueError("至少需要一条人工标注作为示例。")
    demonstrations = "\n\n".join(
        f"Q：{item.question}\nA：{item.reasoning}\n\\boxed{{{item.answer}}}"
        for item in annotations
    )
    return f"{INFERENCE_INSTRUCTION}\n\n{demonstrations}\n\nQ：{sample.question}\nA："


def infer(samples: Sequence[Sample], annotations: Sequence[Annotation], *, temperature: float | None,
          model_caller: ModelCaller = call_model) -> dict[str, Any]:
    """Use human-written uncertain examples as demonstrations for a target set."""
    records = []
    for sample in samples:
        output = model_caller(build_inference_prompt(sample, annotations), temperature=temperature)
        prediction = parse_answer(output)
        records.append({**asdict(sample), "raw_output": output, "prediction": prediction, "correct": prediction == sample.answer})
    correct = sum(record["correct"] for record in records)
    return {"total": len(records), "correct": correct, "accuracy": correct / len(records),
            "invalid": sum(record["prediction"] is None for record in records), "records": records}


def _default_path(kind: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(__file__).parent / "results" / f"active_prompt_{kind}_{timestamp}.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Active-Prompt 两阶段实验")
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect = subparsers.add_parser("collect", help="采样候选题并导出人工标注模板")
    collect.add_argument("--limit", type=_positive_int, default=20, help="候选题数量")
    collect.add_argument("--paths", type=_positive_int, default=5, help="每题采样答案数")
    collect.add_argument("--annotation-budget", type=_positive_int, default=4, help="选择人工标注的题数")
    collect.add_argument("--temperature", type=float, default=0.7)
    collect.add_argument("--output", type=Path)
    collect.add_argument("--dry-run", action="store_true")
    infer_parser = subparsers.add_parser("infer", help="使用完成的人工标注进行推理")
    infer_parser.add_argument("--annotations", type=Path, required=True)
    infer_parser.add_argument("--limit", type=_positive_int, default=100)
    infer_parser.add_argument("--temperature", type=float, default=0.0)
    infer_parser.add_argument("--output", type=Path)
    infer_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.command == "collect":
        samples = EVALUATION_SET[:args.limit]
        if args.paths < 2:
            parser.error("collect 的 --paths 必须至少为 2")
        if args.annotation_budget > len(samples):
            parser.error("--annotation-budget 不能大于 --limit")
        if args.dry_run:
            print(build_initial_prompt(samples[0]))
            print(f"\n正式运行将发起 {len(samples) * args.paths} 次调用，并导出 {args.annotation_budget} 条人工标注任务。")
            return
        candidates = collect_candidates(samples, paths=args.paths, temperature=args.temperature)
        selected = select_uncertain(candidates, args.annotation_budget)
        output = args.output or _default_path("annotations")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(annotation_template(selected), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已选择 {len(selected)} 道不确定题；请填写标注文件：{output.resolve()}")
        return

    annotations = load_annotations(args.annotations)
    annotated_ids = {annotation.sample_id for annotation in annotations}
    samples = tuple(sample for sample in EVALUATION_SET[:args.limit] if sample.sample_id not in annotated_ids)
    if not samples:
        parser.error("移除人工标注题后没有剩余评测样本")
    if args.dry_run:
        print(build_inference_prompt(samples[0], annotations))
        print(f"\n正式运行将发起 {len(samples)} 次模型调用。")
        return
    result = infer(samples, annotations, temperature=args.temperature)
    report = {"experiment": "active_prompt", "created_at": datetime.now().astimezone().isoformat(),
              "sample_count": len(samples), "annotation_count": len(annotations),
              "excluded_annotated_sample_count": args.limit - len(samples), "result": result}
    output = args.output or _default_path("inference")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Active-Prompt：{result['accuracy']:.1%} ({result['correct']}/{result['total']})")
    print(f"完整结果：{output.resolve()}")


if __name__ == "__main__":
    main()
