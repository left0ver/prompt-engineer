"""Compare one CoT path with self-consistent voting over multiple CoT paths.

Run from the project root:
    uv run python -m code.self_consistency_prompting.experiment
"""

from __future__ import annotations

import argparse
import json
import os
import re
from code.common import call_model
from code.gsm8k import load_test_100
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ModelCaller = Callable[..., str]
BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{\s*([-+]?\d[\d,，]*)\s*\}")
TASK_INSTRUCTION = """请模仿示例解决算术推理题，写出简短的推理过程。
回答的最后一行必须严格写成：\\boxed{}
例如:\\boxed{100}
"""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    question: str
    answer: int


@dataclass(frozen=True)
class Demonstration:
    question: str
    reasoning: str
    answer: int


# Adapted from the arithmetic CoT demonstrations presented in the guide's
# self-consistency section. They are separate from the evaluation set.
DEMONSTRATIONS = (
    Demonstration(
        "树林原有15棵树，种完后有21棵。今天种了多少棵？",
        "新增数量等于最后数量减去原有数量，即21 - 15 = 6。",
        6,
    ),
    Demonstration(
        "停车场原有3辆车，又来了2辆。现在有多少辆？",
        "把后来车辆加到原有车辆上：3 + 2 = 5。",
        5,
    ),
    Demonstration(
        "两人分别有32块和42块巧克力，一共吃掉35块，还剩多少块？",
        "原来共有32 + 42 = 74块，吃掉35块后剩74 - 35 = 39块。",
        39,
    ),
    Demonstration(
        "Jason原有20个棒棒糖，送人后只剩12个。他送出了多少个？",
        "送出的数量等于原有数量减去剩余数量：20 - 12 = 8。",
        8,
    ),
)


# The first problem adapts the guide's sister-age example. The rest are held
# out from all few-shot demonstrations.
LEGACY_EVALUATION_SET = (
    Sample(
        "eval-guide-sister-age",
        "我6岁时妹妹3岁。现在我70岁，妹妹现在多少岁？",
        67,
    ),
    Sample("eval-02", "小红有12支笔，又买8支，送出5支。现在有多少支？", 15),
    Sample("eval-03", "图书馆有40本书，收到24本后借出17本。现在有多少本？", 47),
    Sample("eval-04", "公交车上有18人，7人下车后又有11人上车。现在有多少人？", 22),
    Sample("eval-05", "6个书架每个9本书，拿走14本后还剩多少本？", 40),
    Sample("eval-06", "96颗糖平均装进8个袋子，每袋有多少颗？", 12),
    Sample("eval-07", "有5包贴纸，每包12张，送出19张后又买7张。现在有多少张？", 48),
    Sample("eval-08", "账户原有120元，花45元、赚30元后又花10元。余额多少元？", 95),
    Sample("eval-09", "7个班每班25人，其中18人缺席。到场多少人？", 157),
    Sample("eval-10", "4支球队每队11人，3人离队后有2人加入。现在共多少人？", 43),
)

EVALUATION_SET = load_test_100(Sample)


def parse_answer(raw_output: str) -> int | None:
    """Extract the final integer from the last LaTex boxed answer."""
    matches = BOXED_ANSWER_PATTERN.findall(raw_output)
    if not matches:
        return None
    return int(matches[-1].replace(",", "").replace("，", ""))


def _render_demonstration(demonstration: Demonstration) -> str:
    return (
        f"Q：{demonstration.question}\n"
        f"A：{demonstration.reasoning}\n"
        f"\\boxed{{{demonstration.answer}}}"
    )


def build_prompt(sample: Sample) -> str:
    """Build the same few-shot CoT prompt for every sampled reasoning path."""
    demonstrations = "\n\n".join(
        _render_demonstration(demonstration) for demonstration in DEMONSTRATIONS
    )
    return f"{TASK_INSTRUCTION}\n\n{demonstrations}\n\nQ：{sample.question}\nA："


def majority_vote(predictions: Sequence[int | None]) -> dict[str, Any]:
    """Return a stable majority vote while exposing disagreement information."""
    valid = [prediction for prediction in predictions if prediction is not None]
    if not valid:
        return {
            "prediction": None,
            "vote_counts": {},
            "valid_paths": 0,
            "agreement": 0.0,
            "is_tie": False,
        }

    counts = Counter(valid)
    first_position: dict[int, int] = {}
    for position, prediction in enumerate(valid):
        first_position.setdefault(prediction, position)
    highest_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == highest_count]
    winner = min(winners, key=lambda answer: first_position[answer])
    return {
        "prediction": winner,
        "vote_counts": {str(answer): count for answer, count in sorted(counts.items())},
        "valid_paths": len(valid),
        "agreement": highest_count / len(valid),
        "is_tie": len(winners) > 1,
    }


def _call(model_caller: ModelCaller, prompt: str, temperature: float | None) -> str:
    return model_caller(prompt, temperature=temperature)


def evaluate_sample(
    sample: Sample,
    *,
    paths: int,
    baseline_temperature: float | None,
    sampling_temperature: float | None,
    model_caller: ModelCaller,
) -> dict[str, Any]:
    """Run a single CoT baseline and a sampled self-consistency treatment."""
    prompt = build_prompt(sample)
    baseline_output = _call(model_caller, prompt, baseline_temperature)
    baseline_prediction = parse_answer(baseline_output)

    sampled_outputs = [
        _call(model_caller, prompt, sampling_temperature) for _ in range(paths)
    ]
    sampled_predictions = [parse_answer(output) for output in sampled_outputs]
    vote = majority_vote(sampled_predictions)
    return {
        **asdict(sample),
        "baseline": {
            "raw_output": baseline_output,
            "prediction": baseline_prediction,
            "correct": baseline_prediction == sample.answer,
        },
        "self_consistency": {
            "paths": [
                {"raw_output": output, "prediction": prediction}
                for output, prediction in zip(
                    sampled_outputs, sampled_predictions, strict=True
                )
            ],
            **vote,
            "correct": vote["prediction"] == sample.answer,
        },
    }


def _summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    baseline_correct = sum(record["baseline"]["correct"] for record in records)
    consistency_correct = sum(record["self_consistency"]["correct"] for record in records)
    sampled_paths = [
        path
        for record in records
        for path in record["self_consistency"]["paths"]
    ]
    invalid_paths = sum(path["prediction"] is None for path in sampled_paths)
    ties = sum(record["self_consistency"]["is_tie"] for record in records)
    agreements = [record["self_consistency"]["agreement"] for record in records]

    improved = regressed = both_correct = both_wrong = 0
    for record in records:
        baseline = record["baseline"]["correct"]
        consistency = record["self_consistency"]["correct"]
        if not baseline and consistency:
            improved += 1
        elif baseline and not consistency:
            regressed += 1
        elif baseline:
            both_correct += 1
        else:
            both_wrong += 1

    return {
        "baseline": {
            "correct": baseline_correct,
            "total": len(records),
            "accuracy": baseline_correct / len(records),
        },
        "self_consistency": {
            "correct": consistency_correct,
            "total": len(records),
            "accuracy": consistency_correct / len(records),
            "invalid_paths": invalid_paths,
            "total_paths": len(sampled_paths),
            "average_agreement": sum(agreements) / len(agreements),
            "tie_count": ties,
        },
        "comparison": {
            "accuracy_delta": (consistency_correct - baseline_correct) / len(records),
            "improved": improved,
            "regressed": regressed,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "net_improved": improved - regressed,
        },
    }


def run_ablation(
    samples: Sequence[Sample] = EVALUATION_SET,
    *,
    paths: int = 5,
    trials: int = 1,
    baseline_temperature: float | None = 0.0,
    sampling_temperature: float | None = 0.7,
    model_caller: ModelCaller = call_model,
) -> dict[str, Any]:
    """Run a paired single-CoT vs. self-consistency ablation."""
    if not samples:
        raise ValueError("评测集不能为空。")
    if paths < 2:
        raise ValueError("paths 必须至少为 2，才能进行自我一致性投票。")
    if trials < 1:
        raise ValueError("trials 必须至少为 1。")

    records = [
        {"trial": trial, **evaluate_sample(
            sample,
            paths=paths,
            baseline_temperature=baseline_temperature,
            sampling_temperature=sampling_temperature,
            model_caller=model_caller,
        )}
        for trial in range(1, trials + 1)
        for sample in samples
    ]
    summary = _summarize(records)
    return {
        "experiment": "self_consistency_ablation",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": os.getenv("LLM_MODEL", "unknown"),
        "sample_count": len(samples),
        "trials": trials,
        "paths_per_sample": paths,
        "baseline_temperature": baseline_temperature,
        "sampling_temperature": sampling_temperature,
        "model_call_count": len(records) * (paths + 1),
        "summary": summary,
        "records": records,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _default_output_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(__file__).parent / "results" / f"self_consistency_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="自我一致性提示配对消融实验")
    parser.add_argument("--paths", type=_positive_int, default=5, help="每题采样推理路径数")
    parser.add_argument("--trials", type=_positive_int, default=1, help="每题重复次数")
    parser.add_argument("--limit", type=_positive_int, help="只评测前 N 条样本")
    parser.add_argument(
        "--baseline-temperature", type=float, default=0.0, help="单次 CoT 温度"
    )
    parser.add_argument(
        "--sampling-temperature", type=float, default=0.7, help="采样路径温度"
    )
    parser.add_argument("--output", type=Path, help="结果 JSON 路径")
    parser.add_argument(
        "--dry-run", action="store_true", help="显示提示词和调用次数但不调用模型"
    )
    args = parser.parse_args()
    if args.paths < 2:
        parser.error("--paths 必须至少为 2")

    samples = EVALUATION_SET[: args.limit] if args.limit else EVALUATION_SET
    call_count = len(samples) * args.trials * (args.paths + 1)
    if args.dry_run:
        print(build_prompt(samples[0]))
        print(
            f"\n正式运行将发起 {call_count} 次模型调用：每题 1 次单次 CoT "
            f"+ {args.paths} 次采样 CoT。"
        )
        return

    print(f"开始实验：{len(samples)} 条样本，每题 {args.paths} 条采样路径")
    print(f"预计模型调用：{call_count} 次")
    report = run_ablation(
        samples,
        paths=args.paths,
        trials=args.trials,
        baseline_temperature=args.baseline_temperature,
        sampling_temperature=args.sampling_temperature,
    )
    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    baseline = summary["baseline"]
    consistency = summary["self_consistency"]
    comparison = summary["comparison"]
    print(f"单次 CoT：{baseline['accuracy']:.1%} ({baseline['correct']}/{baseline['total']})")
    print(
        f"自我一致性：{consistency['accuracy']:.1%} "
        f"({consistency['correct']}/{consistency['total']})"
    )
    print(f"准确率变化：{comparison['accuracy_delta']:+.1%}")
    print(
        f"投票：平均一致率 {consistency['average_agreement']:.1%}，"
        f"平局 {consistency['tie_count']}，无效路径 "
        f"{consistency['invalid_paths']}/{consistency['total_paths']}"
    )
    print(f"完整结果：{output_path.resolve()}")


if __name__ == "__main__":
    main()
