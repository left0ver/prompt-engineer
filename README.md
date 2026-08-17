# prompt-engineer

提示词技术的可复现实验项目。多数数值推理实验统一使用已下载的 `openai/gsm8k` `main/test` 前 100 条：`code/data/gsm8k_test_100.raw.json`；共享加载器位于 `code/gsm8k.py`。生成知识提示实验单独使用 10 条事实知识问答样本。

## 配置

项目根目录的 `.env` 需要包含：

```dotenv
LLM_MODEL=你的模型名
LLM_BASE_URL=OpenAI兼容接口地址
LLM_API_KEY=你的密钥
```

`LLM_BASE_URL` 可省略，此时使用 OpenAI 默认地址。密钥不要提交到版本控制。

## ReAct 框架

该实现参考 Prompt Engineering Guide 的 [ReAct 框架](https://www.promptingguide.ai/zh/techniques/react)：模型每轮先输出 `Thought N` 和 `Action N`，程序执行 `Search` 或安全的 `Calculator`，再将返回值作为 `Observation N` 拼接进下一轮提示；模型输出 `Finish[答案]` 时结束。这样推理决定下一次工具调用，工具观察又约束后续推理。

```bash
# 只预览首轮 ReAct 提示词（不调用模型）
uv run python -m code.react_prompting.experiment --dry-run

# 使用配置的模型运行；会打印完整 Thought / Action / Observation 轨迹
uv run python -m code.react_prompting.experiment --question "高平原的海拔范围是多少？"
```

搜索工具默认使用可替换的本地知识库，便于复现和离线测试；实际接入搜索 API 时，只需将 `ReActAgent(..., tools={"search": your_search})` 传入返回字符串的搜索函数。计算器使用 AST 白名单解析，只允许基本算术表达式。

## 自我反思（Reflexion）

该实现参考 Prompt Engineering Guide 的 [Reflexion](https://www.promptingguide.ai/zh/techniques/reflexion)：`Actor` 生成一次答案，`Evaluator` 为该次轨迹给出奖励和反馈；失败后，`Self-Reflection` 将轨迹、奖励及既有记忆转为一条具体的语言建议。该建议保存在固定容量的滑动窗口中，作为下一次 Actor 的上下文。因此它不需要微调模型，也能从试错中迭代改进。

```bash
# 只预览首轮 Actor 提示词
uv run python -m code.reflexion_prompting.experiment --dry-run

# 用默认的年龄题运行（精确答案 Evaluator）
uv run python -m code.reflexion_prompting.experiment

# 替换任务、标准答案和试错次数
uv run python -m code.reflexion_prompting.experiment --task "7 * 8 等于多少？" --expected-answer 56 --max-trials 4
```

`ReflexionAgent` 的 `evaluator` 可替换为单元测试、规则检查或 LLM 评审函数；Actor 和 Self-Reflection 调用器同样可注入，因而完整循环可离线测试。

## 少样本提示消融实验

实验参考 Prompt Engineering Guide 的[少样本提示说明](https://www.promptingguide.ai/zh/techniques/fewshot)：在同一组中文情感分类文本上，对比不含演示的零样本基线与包含 4 个正负面演示的少样本处理组。

先预览零样本与少样本提示（不调用模型）：

```bash
uv run python -m code.few_shot_prompting.experiment --dry-run
```

运行完整实验：

```bash
uv run python -m code.few_shot_prompting.experiment
```

默认对 10 条独立测试文本分别运行零样本基线和 4-shot 处理组，共调用模型 20 次。测试集含 5 条正面和 5 条负面文本，演示与测试样本不重叠。结果会保存到 `code/few_shot_prompting/results/`。可用 `--limit 2` 做低成本冒烟测试，或用 `--trials 3` 重复实验。

## CoT、零样本 CoT 与 Auto-CoT 对比实验

实验参考 Prompt Engineering Guide 的[思维链提示说明](https://www.promptingguide.ai/zh/techniques/cot)，并纳入指南中的奇数求和与苹果数量案例。实验包含五个条件：

1. `direct`：直接回答基线。
2. `answer_only_few_shot`：只提供问题和答案的少样本基线。
3. `manual_cot`：在相同少样本示例中加入人工推理链。
4. `zero_shot_cot`：在直接回答提示后加入“让我们逐步思考”。
5. `auto_cot`：自动聚类候选问题、选择代表问题，再由模型生成推理链作为演示。

预览提示词而不调用模型：

```bash
uv run python -m code.chain_of_thought_prompting.experiment --dry-run
```

运行完整实验：

```bash
uv run python -m code.chain_of_thought_prompting.experiment
```

所有条件都要求将最终整数写在 `\boxed{}` 中，评测器会提取最后一个 boxed 值进行验证。默认设置会先生成4个 Auto-CoT 演示，再在 GSM8K 前100题上评测5个条件，共调用模型504次。可以先用 `--limit 2 --auto-clusters 2` 进行12次调用的低成本测试。结果包含各条件准确率、平均绝对误差、回答长度、Auto-CoT 演示正确率以及多组配对改善/退步统计，并保存在 `code/chain_of_thought_prompting/results/`。

## 思维树（ToT）与 CoT 对比实验

实验参考 Prompt Engineering Guide 的[思维树（ToT）](https://www.promptingguide.ai/zh/techniques/tot)：模型先生成多个中间思维、评估中间状态，再使用 BFS/beam search 保留有希望的分支。本实验使用页面中的典型 24 点任务，只比较两个条件：

1. `cot`：单次逐步推理并构造表达式。
2. `tot_bfs`：每层生成多个二元运算步骤、对每个子状态作 `sure/maybe/impossible` 评估，并保留评分最高的状态继续搜索。

最终答案由 Python 严格验算：必须恰好使用四个输入数字各一次、只使用允许的四则运算且结果为 24。结果 JSON 包含搜索轨迹、两组准确率、同题配对的 `improved/regressed`、准确率差异及 ToT 的额外调用次数，便于将效果与成本一起判断。

```bash
# 仅预览提示词
uv run python -m code.tree_of_thought_prompting.experiment --dry-run

# 低成本试运行
uv run python -m code.tree_of_thought_prompting.experiment --limit 1

# 完整的 10 题实验
uv run python -m code.tree_of_thought_prompting.experiment
```

可使用 `--branching-factor`、`--beam-width` 消融 ToT 搜索宽度。结果保存到 `code/tree_of_thought_prompting/results/`。

## 自我一致性实验

实验参考 Prompt Engineering Guide 的[自我一致性说明](https://www.promptingguide.ai/zh/techniques/consistency)。它对同一少样本 CoT 提示采样多条推理路径，再对每条路径最后的 `\boxed{}` 整数答案进行多数投票。实验比较单次 CoT（默认 temperature=0）与5条路径的自我一致性（默认 temperature=0.7），并记录投票分歧、平局和无效答案。

```bash
uv run python -m code.self_consistency_prompting.experiment
```

默认会进行600次调用（100题 × 1次单次 CoT + 5次采样 CoT）。先预览或做低成本测试：

```bash
uv run python -m code.self_consistency_prompting.experiment --dry-run
uv run python -m code.self_consistency_prompting.experiment --limit 2 --paths 3
```

运行不调用真实模型的单元测试：

```bash
uv run python -m unittest discover -s code -p 'test_*.py'
```

## 生成知识提示实验

实验参考 Prompt Engineering Guide 的[生成知识提示](https://www.promptingguide.ai/zh/techniques/knowledge)：评测集为本地固定的 10 条事实知识问答样本（`code/data/knowledge_qa_10.json`），覆盖地理、历史、科学、文学、艺术与人体知识。处理组先生成两条相关事实或实体关系，再把它们与原题一起交给模型作答；直接回答组使用相同问题但不提供知识。答案以大小写、标点和冠词无关的短文本精确匹配评测，并支持数据中声明的同义答案。

```bash
uv run python -m code.knowledge_prompting.experiment --dry-run
uv run python -m code.knowledge_prompting.experiment
```

默认运行 30 次模型调用（10 条样本 × 直接回答 1 次 + 生成知识与整合回答各 1 次），并保存原始回答、生成知识、标签、准确率变化以及逐样本配对改善/退步到 `code/knowledge_prompting/results/`。可用 `--limit 2` 做低成本检查，如需降低偶然性可增加 `--trials 3`。

## Active-Prompt 实验

实验参考 Prompt Engineering Guide 的 [Active-Prompt](https://www.promptingguide.ai/zh/techniques/activeprompt)。它先对候选题采样多个答案，以 `1 - 众数答案占比` 作为不一致度，选择最不确定的问题交给人类编写 CoT 推理；完成的人工标注随后成为少样本示例。

```bash
# 采样前 20 道候选题（每题 5 条路径），导出 4 条人工标注任务
uv run python -m code.active_prompting.experiment collect --limit 20 --paths 5 --annotation-budget 4

# 在导出的 JSON 中填写每条 reasoning 后，用它们评测剩余 GSM8K 样本
uv run python -m code.active_prompting.experiment infer --annotations code/active_prompting/results/active_prompt_annotations_*.json
```

使用 `collect --dry-run` 或 `infer --annotations ... --dry-run` 可分别预览两个阶段的提示词。推理阶段会自动从 GSM8K 前 100 条中排除被人工标注过的题，避免把示例直接泄漏到评测结果中。

## 自动提示工程师（APE）实验

实验参考 Prompt Engineering Guide 的[自动提示工程师（APE）](https://www.promptingguide.ai/zh/techniques/ape)：推理模型从少量输入/输出示例合成多条任务指令；目标模型在独立的选择集上执行每条候选指令并以精确匹配准确率打分；最高分指令最后在未参与生成或选择的留出集上，与固定直接解题基线比较。

```bash
# 只预览用于生成候选指令的元提示词
uv run python -m code.automatic_prompt_engineer.experiment --dry-run

# 低成本试运行：4 个演示、2 个选择题、2 条候选、2 个留出测试题
uv run python -m code.automatic_prompt_engineer.experiment --limit 8 --selection 2 --candidates 2

# 默认完整实验
uv run python -m code.automatic_prompt_engineer.experiment
```

默认按前 4 条 / 后续 16 条 / 剩余 80 条将 GSM8K 前 100 条分为指令生成示例、候选选择集和最终留出集。最多调用 `1 + 8 × 16 + 2 × 80 = 289` 次模型；结果含生成的候选指令、每个候选的选择集记录、选中指令、两种最终提示的原始回答与准确率差，保存到 `code/automatic_prompt_engineer/results/`。
