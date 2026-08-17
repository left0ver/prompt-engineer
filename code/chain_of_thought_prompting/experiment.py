"""Compare direct, CoT, zero-shot CoT, and Auto-CoT prompting.

Run from the project root:
    uv run python -m code.chain_of_thought_prompting.experiment
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from code.common import call_model
from code.gsm8k import load_test_100
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

Condition = Literal[
    "direct",
    "answer_only_few_shot",
    "manual_cot",
    "zero_shot_cot",
    "auto_cot",
]
ModelCaller = Callable[[str], str]

CONDITIONS: tuple[Condition, ...] = (
    "direct",
    "answer_only_few_shot",
    "manual_cot",
    "zero_shot_cot",
    "auto_cot",
)

BOXED_ANSWER_PATTERN = re.compile(r"\\boxed\s*\{\s*([-+]?\d[\d,，]*)\s*\}")
TASK_INSTRUCTION = """请解决下面的数学推理题。
回答的最后一行必须为答案，格式：\\boxed{}
例如：\\boxed{100}
"""
ZERO_SHOT_COT_TRIGGER = "让我们逐步思考。"
AUTO_COT_GENERATION_INSTRUCTION = """请解决下面的数学推理题并展示简短、清晰的中间步骤。
让我们逐步思考。
回答的最后一行必须为答案，格式：\\boxed{}
例如：\\boxed{100}
"""


@dataclass(frozen=True)
class Sample:
    sample_id: str
    problem: str
    answer: int


@dataclass(frozen=True)
class Demonstration:
    sample_id: str
    problem: str
    reasoning: str
    answer: int


@dataclass(frozen=True)
class GeneratedDemonstration:
    sample_id: str
    problem: str
    expected_answer: int
    generated_solution: str
    parsed_answer: int | None
    correct: bool


# The first demonstration adapts the odd-number example from Prompt Engineering
# Guide. Both manual conditions use the same questions and final answers; only
# the chain-of-thought condition exposes the reasoning field.
MANUAL_DEMONSTRATIONS = (
    Demonstration(
        "manual-guide-odd-sum",
        "求数列4、8、9、15、12、2、1中所有奇数的和。",
        "奇数是9、15和1，把它们相加：9 + 15 + 1 = 25。",
        25,
    ),
    Demonstration(
        "manual-02",
        "文具店有3盒铅笔，每盒8支，卖出5支后还剩多少支？",
        "3盒共有3 × 8 = 24支，卖出5支后剩24 - 5 = 19支。",
        19,
    ),
    Demonstration(
        "manual-03",
        "48名学生每6人分成一组，每组需要2个球，一共需要多少个球？",
        "48 ÷ 6 = 8组，每组2个球，所以需要8 × 2 = 16个球。",
        16,
    ),
    Demonstration(
        "manual-04",
        "一本书有120页，第一天读35页，第二天读的页数是第一天的2倍，还剩多少页？",
        "第二天读35 × 2 = 70页，两天共读105页，还剩120 - 105 = 15页。",
        15,
    ),
)


# These unlabeled-to-the-model questions form the Auto-CoT candidate pool.
# Their answers are retained only to audit whether generated demonstrations are
# correct. They never appear in the generation prompt.
AUTO_COT_CANDIDATES = (
    Sample("auto-01", "篮子里有24个苹果，卖出7个后又放入5个，现在有多少个？", 22),
    Sample("auto-02", "72块饼干平均装进8个袋子，每袋有多少块？", 9),
    Sample("auto-03", "求数列3、10、11、8、21中所有奇数的和。", 35),
    Sample("auto-04", "一个长方形长9米、宽6米，面积是多少平方米？", 54),
    Sample("auto-05", "200元的15%是多少元？", 30),
    Sample(
        "auto-06",
        "汽车以每小时50千米行驶150千米，中途休息20分钟，共用多少分钟？",
        200,
    ),
    Sample(
        "auto-07",
        "父亲年龄是孩子的4倍，6年后两人年龄之和为62岁。父亲现在多少岁？",
        40,
    ),
    Sample("auto-08", "鸡和兔共25只，共有70条腿。兔有多少只？", 10),
    Sample(
        "auto-09",
        "4次测验平均80分，前三次是75、82、79分，第四次是多少分？",
        84,
    ),
    Sample(
        "auto-10",
        "5台机器工作4小时，每台每小时生产12件，95%合格。合格品有多少件？",
        228,
    ),
    Sample("auto-11", "500元商品先打八折，再减40元，最终价格是多少元？", 360),
    Sample(
        "auto-12",
        "24米长、18米宽的花园沿内侧铺1米宽小路，未铺路面积是多少平方米？",
        352,
    ),
)


# The first two samples adapt the guide's standard CoT and zero-shot CoT
# examples. The remaining samples broaden the task distribution.
LEGACY_EVALUATION_SET = (
    Sample(
        "eval-guide-odd-sum",
        "求数列15、32、5、13、82、7、1中所有奇数的和。",
        41,
    ),
    Sample(
        "eval-guide-apples",
        "原有10个苹果，送出2个给邻居、2个给修理工，又买5个并吃掉1个，最后有多少个？",
        10,
    ),
    Sample(
        "eval-03",
        "仓库有6箱橙子，每箱14个。去掉9个坏橙子后，把其余橙子平均装进5个篮子，每篮多少个？",
        15,
    ),
    Sample(
        "eval-04",
        "一列火车以每小时60千米行驶240千米，途中停靠35分钟。全程共用多少分钟？",
        275,
    ),
    Sample("eval-05", "农场里鸡和兔共38只，共有112条腿。兔有多少只？", 18),
    Sample(
        "eval-06",
        "一件商品原价360元，先打七五折，再使用30元优惠券，最终价格是多少元？",
        240,
    ),
    Sample(
        "eval-07",
        "五次测验平均84分，前四次为78、85、91、80分。第五次是多少分？",
        86,
    ),
    Sample(
        "eval-08",
        "4台机器工作6小时，每台每小时生产15个零件，10%不合格。合格零件有多少个？",
        324,
    ),
    Sample(
        "eval-09",
        "妈妈年龄是孩子的3倍，8年后两人年龄之和为64岁。妈妈现在多少岁？",
        36,
    ),
    Sample(
        "eval-10",
        "礼堂有23排座位，第一排18个，之后每排多2个。共有多少个座位？",
        920,
    ),
    Sample(
        "eval-11",
        "书店以每本24元购入50本，42本按35元卖出，其余按15元卖出。利润多少元？",
        390,
    ),
    Sample(
        "eval-12",
        "30米长、20米宽的花园沿内侧铺2米宽小路，未铺路面积是多少平方米？",
        416,
    ),
)

# Every technique evaluates the same downloaded GSM8K main/test prefix.
EVALUATION_SET = load_test_100(Sample)


def parse_answer(raw_output: str) -> int | None:
    """Parse the last LaTex-boxed integer, ignoring intermediate values."""
    matches = BOXED_ANSWER_PATTERN.findall(raw_output)
    if not matches:
        return None
    normalized = matches[-1].replace(",", "").replace("，", "")
    return int(normalized)


def _render_manual_demonstration(
    demonstration: Demonstration, *, include_reasoning: bool
) -> str:
    if include_reasoning:
        solution = f"{demonstration.reasoning}\n\\boxed{{{demonstration.answer}}}"
    else:
        solution = f"\\boxed{{{demonstration.answer}}}"
    return f"问题：{demonstration.problem}\n解答：\n{solution}"


def _render_generated_demonstration(demonstration: GeneratedDemonstration) -> str:
    return f"问题：{demonstration.problem}\n解答：\n{demonstration.generated_solution}"


def build_prompt(
    sample: Sample,
    condition: Condition,
    *,
    auto_demonstrations: Sequence[GeneratedDemonstration] = (),
) -> str:
    """Build one of the five controlled experimental prompts."""
    target = f"问题：{sample.problem}\n解答："
    if condition == "direct":
        return f"{TASK_INSTRUCTION}\n\n{target}"
    if condition == "zero_shot_cot":
        return f"{TASK_INSTRUCTION}\n\n{target}\n{ZERO_SHOT_COT_TRIGGER}"
    if condition in ("answer_only_few_shot", "manual_cot"):
        include_reasoning = condition == "manual_cot"
        examples = "\n\n".join(
            _render_manual_demonstration(
                demonstration, include_reasoning=include_reasoning
            )
            for demonstration in MANUAL_DEMONSTRATIONS
        )
        return f"{TASK_INSTRUCTION}\n\n人工示例：\n{examples}\n\n{target}"
    if condition == "auto_cot":
        if not auto_demonstrations:
            raise ValueError("Auto-CoT 条件需要自动生成的演示。")
        examples = "\n\n".join(
            _render_generated_demonstration(demonstration)
            for demonstration in auto_demonstrations
        )
        return f"{TASK_INSTRUCTION}\n\n自动生成示例：\n{examples}\n\n{target}"
    raise ValueError(f"未知实验条件：{condition}")


def build_auto_cot_generation_prompt(sample: Sample) -> str:
    """Build the zero-shot CoT prompt used to create one demonstration."""
    return f"{AUTO_COT_GENERATION_INSTRUCTION}\n\n问题：{sample.problem}\n解答："


def _char_bigrams(text: str) -> Counter[str]:
    """Tokenize Chinese-friendly text without an external NLP dependency."""
    compact = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9%]+", text.lower()))
    if len(compact) < 2:
        return Counter({compact: 1}) if compact else Counter()
    return Counter(compact[index : index + 2] for index in range(len(compact) - 1))


def _normalize(vector: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    return {key: value / norm for key, value in vector.items()} if norm else {}


def _dot(left: dict[str, float], right: dict[str, float]) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def _tfidf_vectors(samples: Sequence[Sample]) -> list[dict[str, float]]:
    documents = [_char_bigrams(sample.problem) for sample in samples]
    document_frequency: Counter[str] = Counter()
    for document in documents:
        document_frequency.update(document.keys())

    count = len(documents)
    vectors: list[dict[str, float]] = []
    for document in documents:
        total = sum(document.values()) or 1
        vector = {
            token: (frequency / total)
            * (math.log((count + 1) / (document_frequency[token] + 1)) + 1)
            for token, frequency in document.items()
        }
        vectors.append(_normalize(vector))
    return vectors


def _centroid(vectors: Sequence[dict[str, float]]) -> dict[str, float]:
    if not vectors:
        return {}
    summed: Counter[str] = Counter()
    for vector in vectors:
        summed.update(vector)
    return _normalize({key: value / len(vectors) for key, value in summed.items()})


def select_auto_cot_representatives(
    candidates: Sequence[Sample], cluster_count: int
) -> tuple[Sample, ...]:
    """Cluster questions with TF-IDF k-means and select centroid-nearest items."""
    if not candidates:
        raise ValueError("Auto-CoT 候选问题不能为空。")
    if cluster_count < 1 or cluster_count > len(candidates):
        raise ValueError("cluster_count 必须在 1 和候选问题数量之间。")

    vectors = _tfidf_vectors(candidates)

    # Deterministic farthest-first initialization gives diverse initial centers.
    center_indices = [0]
    while len(center_indices) < cluster_count:
        unselected = [
            index for index in range(len(vectors)) if index not in center_indices
        ]
        next_index = min(
            unselected,
            key=lambda index: (
                max(_dot(vectors[index], vectors[center]) for center in center_indices),
                index,
            ),
        )
        center_indices.append(next_index)

    centroids = [vectors[index] for index in center_indices]
    assignments: list[int] | None = None
    for _ in range(30):
        new_assignments = [
            max(
                range(cluster_count),
                key=lambda cluster: (_dot(vector, centroids[cluster]), -cluster),
            )
            for vector in vectors
        ]
        if new_assignments == assignments:
            break
        assignments = new_assignments
        centroids = [
            _centroid(
                [
                    vector
                    for vector, assignment in zip(vectors, assignments, strict=True)
                    if assignment == cluster
                ]
            )
            for cluster in range(cluster_count)
        ]

    assert assignments is not None
    representatives: list[Sample] = []
    for cluster, centroid in enumerate(centroids):
        members = [
            index
            for index, assignment in enumerate(assignments)
            if assignment == cluster
        ]
        if not members:
            # Empty clusters can occur with duplicate questions; keep the unique
            # farthest-first seed so the requested number remains stable.
            representative_index = center_indices[cluster]
        else:
            representative_index = max(
                members,
                key=lambda index: (_dot(vectors[index], centroid), -index),
            )
        representatives.append(candidates[representative_index])
    return tuple(representatives)


def generate_auto_cot_demonstrations(
    candidates: Sequence[Sample],
    *,
    cluster_count: int,
    model_caller: ModelCaller,
) -> tuple[GeneratedDemonstration, ...]:
    """Run both Auto-CoT phases: cluster/select, then generate reasoning."""
    representatives = select_auto_cot_representatives(candidates, cluster_count)
    generated: list[GeneratedDemonstration] = []
    for sample in representatives:
        solution = model_caller(build_auto_cot_generation_prompt(sample))
        parsed_answer = parse_answer(solution)
        generated.append(
            GeneratedDemonstration(
                sample_id=sample.sample_id,
                problem=sample.problem,
                expected_answer=sample.answer,
                generated_solution=solution,
                parsed_answer=parsed_answer,
                correct=parsed_answer == sample.answer,
            )
        )
    return tuple(generated)


def evaluate_condition(
    condition: Condition,
    samples: Sequence[Sample],
    *,
    auto_demonstrations: Sequence[GeneratedDemonstration],
    trials: int,
    model_caller: ModelCaller,
) -> dict[str, Any]:
    """Evaluate one prompt condition on all samples."""
    records: list[dict[str, Any]] = []
    for trial in range(1, trials + 1):
        for sample in samples:
            prompt = build_prompt(
                sample,
                condition,
                auto_demonstrations=auto_demonstrations,
            )
            raw_output = model_caller(prompt)
            prediction = parse_answer(raw_output)
            records.append(
                {
                    "trial": trial,
                    **asdict(sample),
                    "raw_output": raw_output,
                    "output_chars": len(raw_output),
                    "prediction": prediction,
                    "absolute_error": (
                        abs(prediction - sample.answer)
                        if prediction is not None
                        else None
                    ),
                    "correct": prediction == sample.answer,
                }
            )

    correct = sum(record["correct"] for record in records)
    invalid = sum(record["prediction"] is None for record in records)
    absolute_errors = [
        record["absolute_error"]
        for record in records
        if record["absolute_error"] is not None
    ]
    return {
        "name": condition,
        "total": len(records),
        "correct": correct,
        "invalid": invalid,
        "accuracy": correct / len(records),
        "mean_absolute_error_valid_outputs": (
            sum(absolute_errors) / len(absolute_errors) if absolute_errors else None
        ),
        "average_output_chars": sum(record["output_chars"] for record in records)
        / len(records),
        "records": records,
    }


def compare_paired(
    baseline: dict[str, Any], treatment: dict[str, Any]
) -> dict[str, int | float]:
    """Compare correctness for the same sample and trial in two conditions."""
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
        "average_output_chars_delta": (
            treatment["average_output_chars"] - baseline["average_output_chars"]
        ),
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
    cluster_count: int = 4,
    auto_cot_candidates: Sequence[Sample] = AUTO_COT_CANDIDATES,
    model_caller: ModelCaller = call_model,
) -> dict[str, Any]:
    """Generate Auto-CoT examples, evaluate five conditions, and compare them."""
    if not samples:
        raise ValueError("评测集不能为空。")
    if trials < 1:
        raise ValueError("trials 必须至少为 1。")

    auto_demonstrations = generate_auto_cot_demonstrations(
        auto_cot_candidates,
        cluster_count=cluster_count,
        model_caller=model_caller,
    )
    conditions = {
        condition: evaluate_condition(
            condition,
            samples,
            auto_demonstrations=auto_demonstrations,
            trials=trials,
            model_caller=model_caller,
        )
        for condition in CONDITIONS
    }

    comparison_pairs = {
        "zero_shot_cot_vs_direct": ("direct", "zero_shot_cot"),
        "manual_cot_vs_answer_only_few_shot": (
            "answer_only_few_shot",
            "manual_cot",
        ),
        "auto_cot_vs_direct": ("direct", "auto_cot"),
        "auto_cot_vs_zero_shot_cot": ("zero_shot_cot", "auto_cot"),
        "auto_cot_vs_manual_cot": ("manual_cot", "auto_cot"),
    }
    comparisons = {
        name: compare_paired(conditions[baseline], conditions[treatment])
        for name, (baseline, treatment) in comparison_pairs.items()
    }
    auto_correct = sum(demonstration.correct for demonstration in auto_demonstrations)
    return {
        "experiment": "cot_zero_shot_cot_auto_cot_comparison",
        "created_at": datetime.now().astimezone().isoformat(),
        "model": os.getenv("LLM_MODEL", "unknown"),
        "sampling_parameters": "provider_default",
        "trials": trials,
        "sample_count": len(samples),
        "model_call_count": cluster_count + len(CONDITIONS) * len(samples) * trials,
        "auto_cot_preparation": {
            "candidate_count": len(auto_cot_candidates),
            "cluster_count": cluster_count,
            "generation_accuracy": auto_correct / len(auto_demonstrations),
            "demonstrations": [
                asdict(demonstration) for demonstration in auto_demonstrations
            ],
        },
        "conditions": conditions,
        "comparisons": comparisons,
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _default_output_path() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(__file__).parent / "results" / f"cot_comparison_{timestamp}.json"


def _print_dry_run(samples: Sequence[Sample], cluster_count: int, trials: int) -> None:
    sample = samples[0]
    for condition in CONDITIONS[:-1]:
        print(f"=== {condition} ===")
        print(build_prompt(sample, condition))
        print()

    representatives = select_auto_cot_representatives(
        AUTO_COT_CANDIDATES, cluster_count
    )
    print("=== Auto-CoT 阶段1：聚类后选出的代表问题 ===")
    for representative in representatives:
        print(f"- {representative.sample_id}: {representative.problem}")
    print("\n=== Auto-CoT 阶段2：第一个推理链生成提示 ===")
    print(build_auto_cot_generation_prompt(representatives[0]))
    call_count = cluster_count + len(CONDITIONS) * len(samples) * trials
    print(f"\n正式运行将发起 {call_count} 次模型调用。")


def main() -> None:
    parser = argparse.ArgumentParser(description="CoT、零样本 CoT 与 Auto-CoT 对比实验")
    parser.add_argument("--trials", type=_positive_int, default=1, help="每条样本重复次数")
    parser.add_argument("--limit", type=_positive_int, help="只评测前 N 条样本")
    parser.add_argument(
        "--auto-clusters",
        type=_positive_int,
        default=4,
        help="Auto-CoT 聚类数及自动演示数",
    )
    parser.add_argument("--output", type=Path, help="结果 JSON 路径")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="显示实验提示和 Auto-CoT 选样但不调用模型",
    )
    args = parser.parse_args()

    if args.auto_clusters > len(AUTO_COT_CANDIDATES):
        parser.error(f"--auto-clusters 不能超过 {len(AUTO_COT_CANDIDATES)}")
    samples = EVALUATION_SET[: args.limit] if args.limit else EVALUATION_SET
    if args.dry_run:
        _print_dry_run(samples, args.auto_clusters, args.trials)
        return

    call_count = args.auto_clusters + len(CONDITIONS) * len(samples) * args.trials
    print(
        f"开始实验：{len(samples)} 条样本 × {args.trials} 次 × "
        f"{len(CONDITIONS)} 个条件 + {args.auto_clusters} 次 Auto-CoT 生成"
    )
    print(f"预计模型调用：{call_count} 次")
    report = run_ablation(
        samples,
        trials=args.trials,
        cluster_count=args.auto_clusters,
    )
    output_path = args.output or _default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n条件结果：")
    for name, result in report["conditions"].items():
        print(
            f"- {name}: {result['accuracy']:.1%} "
            f"({result['correct']}/{result['total']}), "
            f"平均输出 {result['average_output_chars']:.1f} 字符"
        )
    preparation = report["auto_cot_preparation"]
    print(f"Auto-CoT 自动演示正确率：{preparation['generation_accuracy']:.1%}")
    print("\n配对准确率变化：")
    for name, comparison in report["comparisons"].items():
        print(
            f"- {name}: {comparison['accuracy_delta']:+.1%} "
            f"(改善 {comparison['improved']}，退步 {comparison['regressed']})"
        )
    print(f"\n完整结果：{output_path.resolve()}")


if __name__ == "__main__":
    main()
