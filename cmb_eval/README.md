# CMB 模型评测

本目录用于评估医学模型经过 SFT、Evidence-SFT 和 DPO 后的能力变化。CMB 包含两类互补任务：CMB-Exam 测量医学知识选择题能力，CMB-Clin 测量多轮临床问答质量。两类结果分别报告，不合并成一个总分。

##

### 评测目的

本评测主要回答三个问题：

1. 从 Base 到 Full-SFT、Evidence-SFT、Evidence-DPO，模型的医学知识能力如何变化；
2. Evidence-DPO 相比其直接父模型 Evidence-SFT，临床回答是否真正改善；
3. DPO 提升临床回答偏好的同时，是否引入医学知识退化、严重医学错误或只让回答变长。

因此，两部分实验承担不同职责：

| 实验 | 对比对象 | 回答的问题 | 结果定位 |
| --- | --- | --- | --- |
| CMB-Exam | Base、Full-SFT、Evidence-SFT、Evidence-DPO | 各训练阶段的医学知识能力如何变化 | 四模型能力全景，同时作为 DPO 的知识非退化检查 |
| CMB-Clin | Evidence-SFT、Evidence-DPO | DPO 是否提升临床回答质量 | 判断 DPO 效果的主要实验 |

### 数据与评测范围

- CMB-Exam：使用 test 集 11,200 道题，覆盖 6 个大类和 28 个子类；只评 test，不使用 train 和 val；
- CMB-Clin：包含 74 个病例、208 个多轮问题；同一病例内的问题不是相互独立样本；
- 原始数据位于 `/home/medgpt/CMB`，脚本只读取数据，不修改原文件；
- CMB-Exam 本地 test 原文件不含答案，评分使用 `cmb_eval/data/CMB-test-choice-answer.json` 中与 11,200 个 test ID 对齐的官方答案。

### 评测原则与结果解释

- 公平比较要求使用一致的提示词、chat template 和确定性解码参数，并为各 checkpoint 使用与其匹配的 tokenizer；
- CMB-Exam 单选题按唯一选项判分，多选题要求选项集合完全一致；无法抽取的答案计错并单独统计；
- CMB-Exam 除总体准确率外，还报告单选、多选、类别和 macro accuracy；模型差异使用同题配对置信区间与 McNemar 检验；
- CMB-Clin 使用匿名成对 Judge，并交换 A/B 位置各评一次，以降低位置偏差；置信区间按病例聚类 bootstrap，而不是把 208 个问题当作完全独立样本；
- CMB-Clin 重点观察胜/平/负、净胜率、医学专业性、相关性、完整性、清晰度和严重医学错误率，同时检查回答长度与截断率；
- 不应仅凭一个分数宣布 DPO 有效。较有说服力的结论是：Evidence-DPO 在 CMB-Clin 上稳定优于 Evidence-SFT，严重医学错误率不升高，同时 CMB-Exam 的知识能力没有明显退化；
- 如果训练数据包含 CMB-test、CMB-Clin 或相应答案，结果必须标注数据污染风险。

### 脚本说明

日常正式评测只需要直接运行两个 Bash 入口。其余 Python 文件是入口脚本调用的功能组件，也可以在排查单个阶段时独立运行。

#### 正式入口

| 脚本 | 作用 |
| --- | --- |
| `run_exam_four_models.sh` | CMB-Exam 四模型完整流水线：校验 test 数据，依次生成和评分 Base、Full-SFT、Evidence-SFT、Evidence-DPO，完成四组配对比较并生成总表。 |
| `run_clin_dpo_effect.sh` | CMB-Clin 两模型完整流水线：生成 Evidence-SFT 与 Evidence-DPO 多轮回答；可选执行匿名双位置 Judge，并汇总 DPO 效果。 |

#### 公共组件

| 脚本 | 作用 |
| --- | --- |
| `cmb_utils.py` | 集中定义本地 CMB 数据路径，并提供 JSON/JSONL 读写、选择题提示构造、答案抽取与规范化、稳定随机种子等公共函数。 |
| `model_runner.py` | 统一加载 Hugging Face 基座模型与 PEFT adapter；支持在 Evidence-SFT 上叠加 DPO adapter，并以 greedy decoding 执行确定性生成。 |

#### CMB-Exam 组件

| 脚本 | 作用 | 主要产物 |
| --- | --- | --- |
| `validate_exam_test.py` | 校验 test 题目和独立答案文件的数量、ID 对齐、类别分布及答案合法性。 | `test_data_validation.json` |
| `generate_exam.py` | 对单个 checkpoint 生成 test 选择题回答，抽取 A–F 选项并支持断点续跑。 | `predictions.jsonl` |
| `score_exam.py` | 对单选和多选执行严格 exact match，计算总体、题型、类别与 macro accuracy。 | `metrics.json`、`scored_items.jsonl` |
| `compare_exam.py` | 对两个模型做同题配对比较，计算准确率差、置信区间、改对/改错数量、McNemar 检验及非劣效结论。 | `*_vs_*.json` |
| `summarize_exam_suite.py` | 将四个模型的 `metrics.json` 汇总成便于阅读和后续分析的总表。 | `summary.json`、`summary.csv` |

#### CMB-Clin 组件

| 脚本 | 作用 | 主要产物 |
| --- | --- | --- |
| `generate_clin.py` | 按病例逐轮生成回答，把模型上一轮回答带入后续上下文；记录 token、EOS 和生成配置，支持同配置断点续跑并拒绝混合不同配置。 | `predictions.jsonl` |
| `judge_clin.py` | 调用固定的 OpenAI-compatible Judge，对 SFT/DPO 回答匿名评审并交换 A/B 位置；记录胜负、四维评分、严重医学错误和位置一致性。 | `judgments.jsonl` |
| `aggregate_clin_dpo.py` | 汇总胜/平/负、净胜率、四维差值、严重医学错误率、长度和截断率，并按病例进行 cluster bootstrap。 | `evidence_sft_vs_dpo_summary.json` |

测试文件 `tests/test_cmb_eval.py` 覆盖选择题答案抽取和 Clin Judge 返回结构校验，可通过 `python -m unittest discover -s cmb_eval/tests -v` 运行。

下面按两部分说明正式实验。评测目录只保留两个 Bash 入口，不额外维护重复的组合脚本。

## 第一部分：CMB-Exam（四模型对比）

CMB-Exam 只使用 **test 集 11,200 道选择题**，不使用 train 和 val。入口脚本为 `cmb_eval/run_exam_four_models.sh`。

四个模型定义如下：

| 名称 | 加载方式 |
| --- | --- |
| Base | `/home/medgpt/Qwen/Qwen2.5-7B-Instruct` |
| Full-SFT | Base + `/home/medgpt/outputs/sft-base` |
| Evidence-SFT | Base + `/home/medgpt/outputs/evidence-sft-frombase` |
| Evidence-DPO | Base + Evidence-SFT + `/home/medgpt/outputs/evidence-dpo-answer-v1` |

运行命令：

```bash
cd /home/medgpt

CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/root/miniconda3/envs/medgpt/bin/python \
BATCH_SIZE=64 \
bash cmb_eval/run_exam_four_models.sh \
  /home/medgpt/cmb_eval/results/four_models_test
```

这部分已经完成，现有结果保存在 `/home/medgpt/cmb_eval/results/four_models_test`。重新执行时脚本支持断点续跑。

主要输出：

- `summary.json`、`summary.csv`：四模型总表；
- `<model>/metrics.json`：总体、单选、多选、类别及 macro accuracy；
- `<model>/predictions.jsonl`：逐题原始输出和抽取答案；
- `base_vs_full_sft.json`、`base_vs_evidence_sft.json`、`evidence_sft_vs_dpo.json`、`base_vs_evidence_dpo.json`：同题配对比较，包含 accuracy 差异、置信区间和 McNemar 检验。

这部分用于展示四个训练阶段的医学知识选择题能力；其中 `Evidence-SFT vs Evidence-DPO` 也可作为 DPO 是否损害知识能力的辅助检查。

## 第二部分：CMB-Clin（Evidence-SFT vs Evidence-DPO）

CMB-Clin 的目标是直接测量 DPO 相对于其父模型 Evidence-SFT 的效果，因此这里只比较两个模型，不加入 Base 和 Full-SFT。入口脚本为 `cmb_eval/run_clin_dpo_effect.sh`。

| 角色 | 模型 |
| --- | --- |
| Baseline | Base + `/home/medgpt/outputs/evidence-sft-frombase` |
| Candidate | Base + Evidence-SFT + `/home/medgpt/outputs/evidence-dpo-answer-v1` |

先生成两个模型在 74 个病例、208 个多轮问题上的回答，不调用 Judge：

```bash
cd /home/medgpt

CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/root/miniconda3/envs/medgpt/bin/python \
RUN_JUDGE=0 \
CLIN_MAX_NEW_TOKENS=640 \
bash cmb_eval/run_clin_dpo_effect.sh \
  /home/medgpt/cmb_eval/results/clin_dpo_effect
```

默认使用 greedy decoding，并通过统一 system prompt 要求模型结论优先、简洁完整，只回答当前问题，不虚构病例事实。`CLIN_MAX_NEW_TOKENS=640` 为最长参考答案留出结束余量；该值是防截断上限，不是要求模型必须生成到 640 tokens。可以用 `CLIN_SYSTEM_PROMPT` 覆盖默认提示，显式设为空字符串可关闭。

不要把不同生成参数的结果续写到同一个 `predictions.jsonl`。生成记录会保存关键参数，续跑时如发现参数不一致会直接停止。此前 `results/dpo_effect` 下按 512-token 上限生成的文件应保留作检查，不要与新结果混用；调参后的正式结果写入上面示例中的 `results/clin_dpo_effect`。

配置一个固定版本的 OpenAI-compatible Judge 后，运行完整匿名成对评审：

```bash
export JUDGE_MODEL="your-fixed-judge-model"
export JUDGE_BASE_URL="https://your-endpoint/v1"
export JUDGE_API_KEY="your-key"

CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=/root/miniconda3/envs/medgpt/bin/python \
RUN_JUDGE=1 \
JUDGE_WORKERS=8 \
BOOTSTRAP_ITERS=5000 \
CLIN_MAX_NEW_TOKENS=640 \
bash cmb_eval/run_clin_dpo_effect.sh \
  /home/medgpt/cmb_eval/results/clin_dpo_effect
```

脚本默认 `RUN_JUDGE=0`，避免误触发外部请求。完整评审会对每个问题交换 A/B 位置各评一次，共 416 次 Judge 请求；回答生成和 Judge 均支持断点续跑。

主要输出：

- `evidence_sft/predictions.jsonl`：Evidence-SFT 回答；
- `evidence_dpo/predictions.jsonl`：Evidence-DPO 回答；
- `judgments.jsonl`：匿名双位置逐题评审记录；
- `evidence_sft_vs_dpo_summary.json`：最终两模型汇总。

判断 DPO 是否有效时，主要看 CMB-Clin 的胜/平/负、净胜率、医学专业性、相关性、完整性、清晰度、严重医学错误率，以及按病例聚类 bootstrap 得到的置信区间；同时检查回答长度和截断率，避免把单纯变长误判为质量提升。CMB-Exam 四模型结果作为医学知识能力的背景与非退化护栏，不替代这组 Clin 直接对照。
