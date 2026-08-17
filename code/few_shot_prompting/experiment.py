"""A paired zero-shot vs. few-shot sentiment-classification experiment.

Run from the project root:
    uv run python -m code.few_shot_prompting.experiment
"""

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

LABELS = ("正面", "负面")
TASK_INSTRUCTION = """判断文本表达的整体情感是“正面”还是“负面”。
只输出标签“正面”或“负面”，不要解释。"""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    text: str
    label: str


# Demonstrations are deliberately clear, balanced, and separate from the test set.
FEW_SHOT_EXAMPLES = (
    Sample("demo-01", "这家餐厅的服务很周到，菜品也比预想中更美味。", "正面"),
    Sample("demo-02", "耳机用了两天就没有声音，售后还一直推诿。", "负面"),
    Sample("demo-03", "新版本启动速度明显提升，常用功能也更容易找到了。", "正面"),
    Sample("demo-04", "电影情节拖沓，结尾也让人非常失望。", "负面"),
)

# Ten held-out examples inspired by the few-shot classification example in the
# Prompt Engineering Guide. Labels alternate so small --limit runs stay balanced.
EVALUATION_SET = (
    Sample("eval-01", "等了很久终于收到，实物质感比照片还好。", "正面"),
    Sample("eval-02", "结账页面连续崩溃三次，我最后还是没能付款。", "负面"),
    Sample("eval-03", "客服很耐心，几分钟就帮我解决了登录问题。", "正面"),
    Sample("eval-04", "包装已经破损，里面的杯子也碎了一个。", "负面"),
    Sample("eval-05", "更新后的搜索又快又准，确实省了不少时间。", "正面"),
    Sample("eval-06", "宣传里说有离线模式，购买后才发现根本不能用。", "负面"),
    Sample("eval-07", "续航表现令人惊喜，出差一整天都不用充电。", "正面"),
    Sample("eval-08", "活动流程混乱，两个小时几乎都在排队等待。", "负面"),
    Sample("eval-09", "退款过程很顺利，提交申请当天就到账了。", "正面"),
    Sample("eval-10", "问题反馈了一周，至今没有收到任何回复。", "负面"),
)

ModelCaller = Callable[[str], str]


def build_prompt(sample: Sample, examples: Sequence[Sample] = ()) -> str:
    """Build a prompt; the ablation changes only the demonstration block."""
    parts = [TASK_INSTRUCTION]
    if examples:
        rendered_examples = "\n\n".join(
            f"文本：{example.text}\n标签：{example.label}" for example in examples
        )
        parts.append(f"示例：\n{rendered_examples}")
    parts.append(f"文本：{sample.text}\n标签：")
    return "\n\n".join(parts)


def parse_label(raw_output: str) -> str | None:
    """Accept only one of the two labels requested by the prompt."""
    output = raw_output.strip()
    return output if output in LABELS else None


def evaluate_condition(
    name: str,
    samples: Sequence[Sample],
    examples: Sequence[Sample],
    trials: int,
    model_caller: ModelCaller,
) -> dict[str, Any]:
    """Evaluate one prompt condition on all samples."""
    records: list[dict[str, Any]] = []
    for trial in range(1, trials + 1):
        for sample in samples:
            prompt = build_prompt(sample, examples)
            raw_output = model_caller(prompt)
            prediction = parse_label(raw_output)
            records.append(
                {
                    "trial": trial,
                    **asdict(sample),
                    "raw_output": raw_output,
                    "prediction": prediction,
                    "correct": prediction == sample.label,
                }
            )

    correct = sum(record["correct"] for record in records)
    invalid = sum(record["prediction"] is None for record in records)
    return {
        "name": name,
        "shots": len(examples),
        "total": len(records),
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / len(records),
        "records": records,
    }


def compare_paired(
    baseline: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, int | float]:
    """Compare correctness for the same sample and trial in both conditions."""
    baseline_by_key = {
        (record["trial"], record["sample_id"]): record["correct"]
        for record in baseline["records"]
    }
    treatment_by_key = {
        (record["trial"], record["sample_id"]): record["correct"]
        for record in treatment["records"]
    }
    if baseline_by_key.keys() != treatment_by_key.keys():
        raise ValueError("两个实验条件的样本或试验次数不一致，无法配对比较。")

    improved = regressed = both_correct = both_wrong = 0
    for key, baseline_correct in baseline_by_key.items():
        treatment_correct = treatment_by_key[key]
        if not baseline_correct and treatment_correct:
            improved += 1
        elif baseline_correct and not treatment_correct:
            regressed += 1
        elif baseline_correct and treatment_correct:
            both_correct += 1
        else:
            both_wrong += 1

    return {
        "accuracy_delta": treatment["accuracy"] - baseline["accuracy"],
        "improved": improved,
        "regressed": regressed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "net_improved": improved - regressed,
    }


def run_ablation(
    samples: Sequence[Sample] = EVALUATION_SET,
    *,
    trials: int = 1,
    model_caller: ModelCaller = call_model,
) -> dict[str, Any]:
    """Run the paired ablation and return a JSON-serializable report."""
    if not samples:
        raise ValueError("评测集不能为空。")
    if trials < 1:
        raise ValueError("trials 必须至少为 1。")

    baseline = evaluate_condition("zero_shot", samples, (), trials, model_caller)
    treatment = evaluate_condition(
        "few_shot", samples, FEW_SHOT_EXAMPLES, trials, model_caller
    )
    return {
        "experiment": "few_shot_prompting_ablation",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": os.getenv("LLM_MODEL", "unknown"),
        "temperature": 0.0,
        "trials": trials,
        "sample_count": len(samples),
        "conditions": {"baseline": baseline, "treatment": treatment},
        "comparison": compare_paired(baseline, treatment),
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _default_output_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(__file__).parent / "results" / f"few_shot_{timestamp}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="少样本提示配对消融实验")
    parser.add_argument("--trials", type=_positive_int, default=1, help="每条样本重复次数")
    parser.add_argument("--limit", type=_positive_int, help="只评测前 N 条样本")
    parser.add_argument("--output", type=Path, help="结果 JSON 路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="显示两个条件的提示词但不调用模型",
    )
    args = parser.parse_args()

    samples = EVALUATION_SET[: args.limit] if args.limit else EVALUATION_SET
    if args.dry_run:
        print("=== 零样本基线 ===")
        print(build_prompt(samples[0]))
        print("\n=== 少样本处理组 ===")
        print(build_prompt(samples[0], FEW_SHOT_EXAMPLES))
        print(f"\n正式运行将发起 {len(samples) * args.trials * 2} 次模型调用。")
        return

    print(f"开始实验：{len(samples)} 条样本 × {args.trials} 次 × 2 个条件")
    report = run_ablation(samples, trials=args.trials)
    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    baseline = report["conditions"]["baseline"]
    treatment = report["conditions"]["treatment"]
    comparison = report["comparison"]
    print(
        f"零样本准确率：{baseline['accuracy']:.1%} "
        f"({baseline['correct']}/{baseline['total']})"
    )
    print(
        f"少样本准确率：{treatment['accuracy']:.1%} "
        f"({treatment['correct']}/{treatment['total']})"
    )
    print(f"准确率变化：{comparison['accuracy_delta']:+.1%}")
    print(
        f"配对变化：改善 {comparison['improved']}，退步 {comparison['regressed']}，"
        f"净改善 {comparison['net_improved']}"
    )
    print(f"完整结果：{output_path.resolve()}")


if __name__ == "__main__":
    main()
