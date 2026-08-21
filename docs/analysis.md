# Evidence-Med 项目训练与评测总分析报告

> 最后更新：2026-08-21
> 基座模型：Qwen2.5-7B-Instruct  
> 完成链路：Base → Direct Evidence-SFT → Strict Answer-level DPO → Evidence-DPO
> 当前模型决策：**Evidence-SFT 继续作为保守默认模型；Evidence-DPO 已完成训练且医学知识基本不退化，但 CMB-Clin 尚未证明其临床回答稳定优于 Evidence-SFT，因此保留为待复核候选。**

本文档是项目结论的统一入口。第 0 节汇总截至 2026-08-21 的最终 DPO 与 CMB 结果；后续章节保留 Evidence-SFT 路线消融、数据构建和指标定义等详细分析。机器可读结果及专项报告索引见第 12 节。

## 0. 2026-08-21 最新实验总览

### 0.1 已完成的实验链路

```text
Qwen2.5-7B-Instruct
  ├── Full-SFT（对照）
  └── Direct Evidence-SFT（D0，主路线）
        └── Strict Answer-level DPO（D1 / Evidence-DPO）
              ├── Evidence held-out / C-Eval 非退化检查
              ├── CMB-Exam 四模型知识评测
              └── CMB-Clin Evidence-SFT vs Evidence-DPO 临床回答评测
```

Evidence-Mask 不在本轮 DPO 主实验链路中，不能把 D1 表述为 Evidence + Mask + DPO。CMB-Exam 与 CMB-Clin 承担不同职责：前者检查知识能力，后者才直接评估 DPO 对临床回答偏好的影响，两者不合并为一个总分。

### 0.2 最终结论表

| 问题 | 结果 | 当前结论 |
|---|---|---|
| Direct Evidence-SFT 能否学会结构化 Evidence 输出？ | Schema 99.55%，span grounding 98.68% | 能；前置 Full-SFT 不是形成结构能力的必要条件 |
| DPO 训练是否正常完成？ | 1,243/101 对，1 epoch、78 steps；验证偏好准确率 88.12% | 是；优化目标已被模型学到 |
| DPO 是否损害医学选择题知识？ | CMB-Exam 77.330% vs SFT 77.384%，差 -0.054 pp，p=0.586 | 未见明显退化，也没有知识提升证据 |
| DPO 是否稳定改善临床回答？ | CMB-Clin 净胜率 +6.25 pp，病例聚类 95% CI [-12.05, +24.42] pp | 尚未证明；区间跨 0 |
| DPO 是否安全非退化？ | hard error 9.86% vs 8.17%，差 +1.68 pp，CI 跨 0 | 尚未建立；点估计对 DPO 不利 |
| 当前应部署哪个模型？ | DPO 只有小幅完整性信号，Judge 换位一致率仅 43.27% | 默认保留 Evidence-SFT，DPO 作为待复核候选 |

### 0.3 DPO 数据与训练

正式 DPO 使用 Strict Answer-level pair：`task_type`、`query_intent`、`evidence_sufficiency`、`evidence`、`critical_evidence_ids` 在两端冻结，只允许 `missing_information`、`clinical_reasoning`、`final_answer` 不同。

| 项目 | Train | Validation |
|---|---:|---:|
| 导出 pair | 1,243 | 101 |
| 唯一 source | 1,072 | 89 |
| target vs model | 771 | 63 |
| model vs model | 348 | 28 |
| controlled negative | 124 | 10 |
| train/validation source 交叉 | \- | 0 |

训练设置为 D0 合并权重上的新 LoRA、`beta=0.1`、sigmoid loss、学习率 `5e-6`、1 epoch。最终 `train_loss=0.6248`，验证 `eval_loss=0.5670`，偏好准确率为 **88.12%**，reward margin 为 **0.2863**。训练过程正常收敛，但数据存在明确长度倾向：全部导出 pair 中 **81.32%** 的 chosen 比 rejected 更长，因此下游完整性提升必须同时做长度敏感性解释。

### 0.4 CMB-Exam：知识保持

CMB-Exam 使用同一 test 集 11,200 道题，四模型均无缺失预测。

| 模型 | 正确数 | Accuracy | 相对 Base | 无效输出 |
|---|---:|---:|---:|---:|
| Base | 8,504 | 75.929% | 0 | 296 |
| Full-SFT | 8,441 | 75.366% | -0.563 pp | 13 |
| Evidence-SFT | 8,667 | **77.384%** | +1.455 pp | 132 |
| Evidence-DPO | 8,661 | 77.330% | +1.402 pp | 136 |

Evidence-SFT 相对 Base 的提升明确（McNemar `p=3.96e-13`）。Evidence-DPO 相对 Evidence-SFT 只少答对 6 题：两者共同答对 8,622 题、共同答错 2,494 题、DPO 单独答对 39 题、SFT 单独答对 45 题；配对差值 95% CI 为 **[-0.214, +0.107] pp**，`p=0.586`。因此 DPO 通过知识非退化检查，但不能据此宣称 DPO 有效。

### 0.5 CMB-Clin：临床回答质量

CMB-Clin 覆盖 74 个病例、208 个多轮问题。Evidence-SFT 与 Evidence-DPO 的病例、问题、system prompt、tokenizer、greedy decoding 和生成上限一致，两组均无截断。每题由 `mimo-v2.5-pro` 匿名评审两次并交换 A/B 位置，置信区间按病例聚类 bootstrap。

| 指标 | Evidence-SFT | Evidence-DPO | DPO−SFT |
|---|---:|---:|---:|
| 双顺序稳定胜出 | 23 | 28 | +5 题 |
| 平局 / both bad | \- | \- | 29 / 10 |
| 净胜率 | \- | \- | +6.25 pp，95% CI [-12.05, +24.42] pp |
| 流畅度 | 4.625 | 4.630 | +0.005 |
| 相关性 | 4.190 | 4.180 | -0.010 |
| 完整性 | 3.452 | 3.558 | +0.106 |
| 医学专业性 | 3.663 | 3.683 | +0.019 |
| hard medical error | 8.17% | 9.86% | +1.68 pp |
| 平均生成 tokens | 228.4 | 255.8 | +27.4（约 +12%） |

最大问题是 Judge 换位一致率只有 **90/208=43.27%**，118 条结论无法复现。主胜负分析只剩 80 个可比较问题，且所有关键差值的病例聚类区间都跨 0。DPO 的主要正向信号是完整性，但回答平均也变长约 12%；医学专业性没有同步可靠提高，hard-error 点估计反而上升。最准确的表述是：**DPO 改变了回答行为，可能轻微提升完整性，但现有实验无法区分小幅真实增益与 Judge 噪声，也没有建立安全非退化。**

### 0.6 当前模型选择与停止条件

当前不建议继续增加 DPO epoch，也不建议仅凭本批自动 Judge 将默认模型从 Evidence-SFT 切换到 Evidence-DPO。

1. **默认模型：Evidence-SFT。**它是 CMB-Exam 最佳 checkpoint，临床安全证据也更保守。
2. **候选模型：Evidence-DPO。**保留 adapter、训练日志和全部预测，用于后续复核，不宣布失败也不宣布成功。
3. **上线阻断项：**先人工复核 CMB-Clin 中任一模型被双位置确认的 hard-error，优先处理 DPO 独有的 6 个问题。
4. **最小补评：**对 118 条换位不一致记录做第三裁决，或引入第二个固定 Judge；在下一次评测前预先定义净胜率与安全非劣阈值。
5. **长度控制：**增加长度匹配抽样或显式控制冗余的人工评审，避免把回答变长等同于质量提升。

### 0.7 可复跑状态

测试路径已统一到当前目录 `process_sft/`、`process_dpo/`、`evi-sft-mask/`；DPO 依赖的数值风险与强断言风险分类函数已纳入版本库，并有回归测试。当前工作区的验证命令为：

```powershell
& 'D:\miniconda3\envs\medgpt\python.exe' -m unittest discover -s tests -v
& 'D:\miniconda3\envs\medgpt\python.exe' -m unittest discover -s cmb_eval/tests -v
```

DPO v1 的精确复跑从冻结来源清单 `data/dpo/answer_v1/00_sources.jsonl` 开始；该文件固定为 3,000 train + 200 validation，测试会校验全部 3,200 个 `source_id` 的顺序摘要（SHA-256：`3913d351282fbfc51ddf1089ef1a13ce95464e149a5c92a5a14ed69a1f058563`）。历史来源选择使用的 warning-risk 纯函数当时没有进入 Git，因此当前 selector 只用于新数据版本，不能覆盖 v1 清单后声称字节级复现。这个边界写入流程文档后，正式实验链路可稳定从冻结清单继续生成候选、构造 pair、导出和训练，同时避免把近似恢复的启发式规则冒充历史实现。

测试通过只证明冻结数据入口、数据处理、门禁和评测工具的代码行为可复现，不替代模型权重的端到端训练复跑，也不提高现有 Clin Judge 结论的证据等级。

## 1. 技术总结

本项目的目标不是单纯提高医学选择题分数，而是让医疗问答模型在生成结论的同时，输出可检查、可追踪的证据结构，包括任务类型、问题意图、证据充分性、原文证据、关键证据编号、缺失信息、推理过程和最终回答。

本报告将两条 Evidence-SFT 路线作为初始化路径消融：

- **E2 / Direct：** Qwen2.5-7B-Instruct → Evidence-SFT，产物为 `outputs/evidence-sft-frombase`；
- **E3 / Two-stage：** Qwen2.5-7B-Instruct → Full-SFT → Evidence-SFT，本地评测产物为 `outputs/evidence-sft`，历史评测记录中命名为 `evidence-sft-v2-2`。

两路使用相同的 7,671 条 Evidence 训练集、880 条验证集、444 条测试集，以及相同的 Evidence 阶段超参数。逐样本核验显示，两份测试预测的 444 个 `source_id`、病例、prompt 和 gold 完全对齐，因此可以进行配对比较。

主要结果如下：

- **Direct 已独立学会结构化输出。** Direct 的严格 JSON 合法率为 **99.77%**，Two-stage 为 **99.55%**；两者完整 Schema 合法率均为 **99.55%**。Direct 仅多 1 条合法 JSON，不构成实质格式优势。
- **两路证据都能稳定回查原文。** Direct 与 Two-stage 的片段级 grounding 分别为 **98.68%** 和 **99.05%**；sample-level 全证据 grounded 分别为 **94.82%** 和 **96.17%**。Two-stage 方向上略好，但配对差异未达到显著水平。
- **Two-stage 的证据选择与任务路由略优。** 全部证据 exact F1 为 **58.10% vs 59.49%**，边界兼容 F1 为 **83.36% vs 84.51%**，task type agreement 为 **80.86% vs 82.43%**（顺序均为 Direct vs Two-stage）。差距约 0.2～1.6 个百分点，大多数配对置信区间跨越 0。
- **两路都没有学好证据充分性。** Direct 与 Two-stage 的 sufficiency accuracy 分别为 **87.39%** 和 **86.94%**，均低于固定预测 `partial` 的 **87.61%** 多数类基线；两者都从未预测 `sufficient`，macro-F1 仅 **39.10%/39.60%**。
- **医学选择题能力基本保持。** 四组 C-Eval 总分为 Base **89.49%**、Full-SFT **89.12%**、Direct **89.00%**、Two-stage **89.24%**，最大差异仅 4/818 题（0.49 个百分点）。现有结果不能支持显著提升或退化的结论。
- **Direct 的从零训练成本明显更低。** Direct 只需约 1.97 小时 Evidence 训练；若把 Full-SFT 前置阶段计入，Two-stage 总微调约 18.79 小时，墙钟时间约为 Direct 的 9.5 倍，但核心指标差距很小。若 Full-SFT 适配器已存在，则两路 Evidence 阶段的增量成本基本相同。

最稳妥的项目结论是：

> **Evidence 数据本身已足以让 Base 模型获得稳定的 JSON、原文证据引用和字段约束能力。前置 Full-SFT 更像提供了较好的收敛起点，并可能带来很小的证据选择与任务路由收益；当前结果不支持“前置 Full-SFT 是形成 Evidence 能力的必要条件”。自动评测证明的是结构可靠性、原文可追溯性和 teacher 一致性，而不是临床正确性。**

---

## 2. 项目目标与模型训练链路

### 2.1 模型链路与对照定义

实际训练关系如下：

```text
Qwen2.5-7B-Instruct
        ├── E2 / Direct：新建 LoRA → Evidence-SFT
        │                         └── outputs/evidence-sft-frombase
        │
        └── E1：Full-SFT LoRA（outputs/sft-base）
                    └── E3 / Two-stage：继续 Evidence-SFT
                                               └── outputs/evidence-sft
                                                   （历史名 evidence-sft-v2-2）
```

两条路线的区别只在进入 Evidence 阶段前的初始化：

1. **Direct** 从 Base 新建 LoRA，直接学习结构化证据任务；
2. **Two-stage** 从已有 Full-SFT LoRA 继续训练同一组 LoRA 参数，累计保留前置问答训练和 Evidence 训练的更新。

因此，本实验更准确的名称是**初始化路径消融**，而不是严格的“医学 SFT 必要性实验”。前置阶段实际包含 100k 医疗与 20k 通用数据，还额外经历 7,500 个训练 steps；不能把两路差异纯粹归因于医学知识训练。更重要的是，Evidence 候选来自 `medical_100k`，而该数据也是前置 Full-SFT 的主要组成部分，因此 Two-stage 很可能在 Evidence 测试前见过同一病例的原始问答。其较低损失和小幅指标优势可能受到病例预暴露影响，不能直接解释为独立泛化收益。

### 2.2 为什么输出 JSON

JSON 不是最终给普通用户阅读的界面，而是模型与后续程序之间的结构化协议。它的价值主要有四点：

- **可自动验证：** 可以检查字段是否完整、证据是否来自病例原文、关键证据编号是否一致。
- **可追溯：** 最终回答可以和引用证据建立明确关系，而不是只得到一段无法拆解的自由文本。
- **可扩展：** 前端可以按需展示“结论、依据、缺失信息”，不必把原始 JSON 直接呈现给用户。
- **可用于风险控制：** 当证据不足时，系统可以根据 `evidence_sufficiency` 和 `missing_information` 决定是否提示补充信息或转人工。

因此面试时可解释为：**训练目标是让模型生成机器可检查的中间结果；产品层会解析 JSON，再以适合人的方式展示。**

---

## 3. 数据构建与验证结果

### 3.1 原始数据和自动分流

项目使用 deepseek-v4-flash 蒸馏生成 10,000 条 teacher 原始数据。统计文件中出现的 `deepseek-v4-flash` 与 `deepseek-v4-flash-free` 本质上是同一个 teacher，只是历史命名不同，不应解释为多 teacher 数据混合。

验证脚本将数据分为 accepted、review 和 rejected 三类：

| 数据状态 | 样本数 | 占比 | 当前用途 |
|---|---:|---:|---|
| Accepted | 8,995 | 89.95% | 用于训练、验证和测试 |
| Review | 957 | 9.57% | 当前冻结，不进入训练 |
| Rejected | 48 | 0.48% | 当前冻结，不进入训练 |
| 合计 | 10,000 | 100.00% | — |

Accepted 数据进一步拆分为：

| 数据集 | 样本数 | Accepted 内占比 | 用途 |
|---|---:|---:|---|
| Train | 7,671 | 85.28% | 参数训练 |
| Validation | 880 | 9.78% | 训练过程中观察泛化损失 |
| Test | 444 | 4.94% | 训练结束后的独立自动评测 |
| 合计 | 8,995 | 100.00% | — |

在当前实习项目范围内，暂不处理 review/reject 是合理选择。8,995 条 accepted 数据已经足以完成 LoRA SFT 和独立测试；继续人工清洗 1,005 条边缘数据的成本较高，预期收益不确定。需要保留的限制是：当前结论只适用于自动规则接受的数据分布，不代表被搁置样本已经解决。

### 3.2 验证器解决的主要问题

V2.2 验证流程主要检查：

- 样本 ID 完整性、唯一性和候选集对齐；
- JSON 字段和枚举值是否合法；
- evidence span 是否能在病例原文中定位；
- 证据 ID 是否规范、连续，是否存在重复或重叠；
- `critical_evidence_ids` 与 evidence 的 `importance` 是否一致；
- sufficient/partial/insufficient 与证据、缺失信息之间是否存在明显矛盾；
- 过长、过短、重复歧义或疑似 query 文本的证据是否需要 review。

主要 reject 原因是 evidence span 不在病例原文中，共 38 条；其余包括“充分但无证据”8 条、“最终回答直接复制原答案”2 条。主要 review 原因是证据过长需要原子化复核、重复文本导致定位歧义，以及 evidence span 疑似包含 query 文本。

### 3.3 数据分布风险

原始 10,000 条数据的 `evidence_sufficiency` 分布高度不均衡：

| 类别 | 样本数 | 占比 |
|---|---:|---:|
| partial | 8,841 | 88.41% |
| insufficient | 864 | 8.64% |
| sufficient | 294 | 2.94% |
| conflicting | 1 | 0.01% |

这一分布会自然诱导模型偏向输出 `partial`，也是测试集中充分性分类发生类别坍缩的主要背景。由于当前项目重点是证据结构和可追溯性，不建议为了修复一个次要分类字段重新进行大规模蒸馏；应在报告中如实说明，而不是把 86.94% agreement 当成亮点。

---

## 4. Evidence-SFT 训练设置与过程

### 4.1 Evidence 阶段的可比性

除初始化 LoRA 外，两条路线的 Evidence 阶段设置一致：

| 配置项 | Direct | Two-stage |
|---|---:|---:|
| Base model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| LoRA 初始化 | 新建 LoRA | 恢复 `outputs/sft-base` LoRA |
| Train / Validation | 7,671 / 880 | 7,671 / 880 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 | 16 / 32 / 0.05 |
| 单卡 batch / accumulation | 8 / 4 | 8 / 4 |
| 有效 batch size | 32 | 32 |
| Epoch / optimizer steps | 2 / 480 | 2 / 480 |
| Learning rate | 1e-5 | 1e-5 |
| Max sequence length | 1,536 | 1,536 |
| Precision | BF16 | BF16 |
| Gradient checkpointing | 开启 | 开启 |
| Seed | 42 | 42 |

两个 checkpoint 的 TrainingArguments 除输出和日志路径外一致，tokenizer 与 chat template 哈希也一致。训练实现采用 assistant-only cross-entropy：system、user、assistant header 和 padding 的 label 均为 `-100`，只监督 assistant JSON 内容及 EOS。数据中的 `masked=false` 与 loss mask 无关，训练器不读取该字段，因此当前两路实验都**没有实施“关键证据 Mask 增强”**。

按训练使用的 chat template 统计，训练样本最大长度约为 **1,301 tokens**，没有样本超过 1,536；因此两路都不存在因上下文上限造成的训练截断。实际训练为 BF16 LoRA，不是截图设想中的 4-bit QLoRA。

### 4.2 收敛结果对比

| 指标 | Direct | Two-stage | 差异（Direct−Two-stage） |
|---|---:|---:|---:|
| 最终 train loss | 0.7102 | 0.6769 | +0.0333 |
| 最终 validation loss | 0.6116 | 0.6047 | +0.0069 |
| Validation perplexity | 1.8434 | 1.8307 | +0.0127 |
| Optimizer steps | 480 | 480 | 0 |
| Evidence 训练时间 | 7,079.66 秒 | 7,091.83 秒 | -12.17 秒 |
| 训练吞吐 | 2.167 samples/s | 2.163 samples/s | +0.004 |

验证损失曲线如下：

| Step | Direct | Two-stage | 差异 |
| ---: | ---: | ---: | ---: |
| 100 | 0.7239 | 0.6807 | +0.0432 |
| 200 | 0.6502 | 0.6334 | +0.0168 |
| 300 | 0.6253 | 0.6150 | +0.0103 |
| 400 | 0.6142 | 0.6067 | +0.0075 |
| 训练结束 | 0.6116 | 0.6047 | +0.0069 |

两路验证损失都持续下降且没有反弹。Two-stage 全程更低，但差距从 step 100 的 0.0432 快速收窄到最终 0.0069；这更像前置 Full-SFT 提供了较好的收敛起点，而不是 Direct 无法学习 Evidence 能力。最终 PPL 相差约 0.69%，需要结合独立任务指标判断，不能把较低 token loss 直接解释为更好的临床回答。

### 4.3 从零构建时的成本差异

| 路线 | 前置训练 | Evidence 训练 | 总墙钟时间 | 总 FLOPs |
|---|---:|---:|---:|---:|
| Direct | 无 | 7,079.66 秒 | 约 1.97 小时 | 5.81×10^17 |
| Two-stage | 60,555.97 秒 / 7,500 steps | 7,091.83 秒 | 约 18.79 小时 | 6.06×10^18 |

若从 Base 重新搭建完整链路，Two-stage 的总墙钟时间约为 Direct 的 **9.5 倍**，FLOPs 约为 **10.4 倍**。如果 `sft-base` 已经存在，两条路线新增的 Evidence 训练成本则基本相同。因此路线选择应区分两种场景：已有 Full-SFT 资产时可利用其小幅起点优势；从零构建时，Direct 的性价比明显更高。

---

## 5. 444 条 Evidence 自动评测方法

### 5.1 配对评测口径

测试使用 444 条 held-out accepted 样本。两路模型均采用 greedy decoding 自由生成，而不是 teacher forcing；最大新生成长度为 1,152 tokens。两份预测文件各包含 444 个唯一 `source_id`，ID 集合、顺序、`case_text`、prompt 和 gold 均逐条一致，因此指标差异不是测试样本不同造成的。

Direct 与 Two-stage 的评测耗时分别为 478.90 秒和 383.07 秒，且两者都是 444/444 条生成 EOS。由于生成文本长度、运行时负载和环境吞吐都可能影响耗时，本报告不把这一次时延差异解释为模型路线的稳定性能差异。

### 5.2 指标边界

| 指标 | 定义 | 能证明什么 | 不能证明什么 |
|---|---|---|---|
| Strict JSON valid | 完整输出可直接被 JSON parser 解析为对象 | 输出格式稳定 | 医疗内容正确 |
| Schema valid | 必需字段、字段类型和枚举值全部合法 | 协议遵循能力 | 字段语义正确 |
| Evidence grounding | 预测 span 是否逐字出现在病例原文中 | 引用文本可回查 | 该证据医学上重要或结论正确 |
| Critical consistency | `critical_evidence_ids` 是否与 `importance=critical` 对应 | JSON 内部关系一致 | critical 判断正确 |
| Teacher exact F1 | 预测 span 与 teacher span 严格逐字重合 | teacher 边界复现程度 | 独立临床准确率 |
| Boundary-compatible F1 | 预测与 gold 互为子串时按一对一最大匹配 | 降低边界切分差异影响 | 两段证据语义必然等价 |
| Task / Sufficiency agreement | 预测标签与 teacher 相同 | 标签一致性 | teacher 绝对正确或类别均衡泛化 |
| Final answer non-empty | `final_answer` 存在且非空 | 模型完成了字段输出 | 回答完整、安全或正确 |

二分类样本级指标使用配对 McNemar 检验；span P/R/F1 使用以样本为簇的配对 bootstrap。报告中的 95% 置信区间均针对 **Direct−Two-stage**，用于判断差异范围，而不是给某条路线背书。由于同时观察了多项相关指标且只有单个 seed，个别未经多重比较校正的区间不能作为总体显著优势。

---

## 6. Evidence 自动评测结果

### 6.1 总体配对结果

| 评测项 | Direct | Two-stage | 差异（pp） | 配对结果 |
|---|---:|---:|---:|---|
| Strict JSON valid | 443/444，99.77% | 442/444，99.55% | +0.23 | McNemar p=1.000 |
| Schema valid | 442/444，99.55% | 442/444，99.55% | 0.00 | p=1.000 |
| Final answer non-empty | 443/444，99.77% | 442/444，99.55% | +0.23 | p=1.000 |
| Sample-level all evidence grounded | 421/444，94.82% | 427/444，96.17% | -1.35 | 95% CI [-3.38, +0.68]；p=0.286 |
| Evidence span grounding | 1,571/1,592，98.68% | 1,560/1,575，99.05% | -0.37 | 95% CI [-1.02, +0.39] |
| Critical internal consistency | 442/444，99.55% | 441/444，99.32% | +0.23 | p=1.000 |
| Task type agreement | 359/444，80.86% | 366/444，82.43% | -1.58 | 95% CI [-4.50, +1.35]；p=0.349 |
| Sufficiency agreement | 388/444，87.39% | 386/444，86.94% | +0.45 | 95% CI [-1.58, +2.48]；p=0.824 |
| EOS completion | 444/444，100% | 444/444，100% | 0.00 | — |

两条路线都达到了高格式成功率和高原文 grounding。Direct 已证明不依赖前置 Full-SFT 也能形成 Evidence 结构能力；Two-stage 在 sample grounding、span grounding 和 task type 上方向性更好，但差异的置信区间均跨越 0。Direct 在 JSON、critical consistency 和 sufficiency raw accuracy 上的微小领先也都只有 0～1 个样本量级，不能解释为实质优势。

### 6.2 证据抽取与 teacher 一致性

#### 严格逐字匹配

| 范围 | 路线 | Predicted | Gold | Overlap | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| 全部证据 | Direct | 1,592 | 1,582 | 922 | 57.91% | 58.28% | **58.10%** |
| 全部证据 | Two-stage | 1,575 | 1,582 | 939 | 59.62% | 59.36% | **59.49%** |
| Critical | Direct | 478 | 540 | 244 | 51.05% | 45.19% | **47.94%** |
| Critical | Two-stage | 462 | 540 | 241 | 52.16% | 44.63% | **48.10%** |

全部证据 exact F1 的 Direct−Two-stage 差异为 **-1.39 pp**，配对 bootstrap 95% CI 为 **[-3.46, +0.69]**；critical exact F1 差异仅 **-0.17 pp**，95% CI 为 **[-3.65, +3.32]**。两者都不足以确认路线差异。

#### 边界兼容匹配

严格 exact match 对 span 边界非常敏感。按“预测包含 gold 或 gold 包含预测”进行一对一最大匹配后：

| 路线 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Direct | 83.10% | 83.63% | **83.36%** |
| Two-stage | 84.70% | 84.32% | **84.51%** |
| Direct−Two-stage | -1.60 pp | -0.70 pp | **-1.15 pp** |
| 配对 95% CI | [-3.04, -0.16] | [-2.41, +1.09] | **[-2.37, +0.12]** |

Two-stage 的 precision 区间在未经多重比较校正时略高，但 recall 和 F1 的区间仍跨越 0。结合多个相关指标和单 seed 设置，更稳妥的解释是：Two-stage 在证据边界与选择上有小幅方向性优势，但尚不能宣称总体显著优于 Direct。

两路 boundary-compatible F1 都比 strict exact F1 高约 25 个百分点，说明 strict F1 偏低的重要来源是边界切分差异，而不全是证据方向错误。与此同时，critical exact recall 仍只有约 45%，表明“哪些证据必须标为关键”是两路共同的薄弱项。

### 6.3 Task type 分类

| 指标 | Direct | Two-stage |
|---|---:|---:|
| Accuracy / teacher agreement | 80.86% | 82.43% |
| Macro-F1 | 80.16% | 82.00% |
| Balanced accuracy | 80.21% | 82.21% |
| diagnostic_reasoning recall | 135/175，77.14% | 142/175，81.14% |
| confirmed_management recall | 224/269，83.27% | 224/269，83.27% |

两路在 management 类上的命中完全相同，差异主要来自 Two-stage 多正确识别 7 条 diagnostic 样本。其 task macro-F1 高 1.84 pp，但配对 accuracy 检验 p=0.349，现阶段只能视为小幅方向性收益。

### 6.4 Evidence sufficiency 分类

测试集 gold 分布为 `partial=389`、`insufficient=44`、`sufficient=11`，多数类基线为 389/444 = **87.61%**。

| 指标 | Direct | Two-stage |
|---|---:|---:|
| Accuracy | 87.39% | 86.94% |
| Macro-F1 | 39.10% | 39.60% |
| Balanced accuracy | 37.95% | 38.45% |
| partial recall | 381/389，97.94% | 378/389，97.17% |
| insufficient recall | 7/44，15.91% | 8/44，18.18% |
| sufficient recall | 0/11，0% | 0/11，0% |
| 预测为 partial | 429/444，96.62% | 424/444，95.50% |

Direct 的 raw accuracy 虽高 0.45 pp，但仍低于恒猜 `partial` 的基线，且类别坍缩略重；因此不能把这 0.45 pp 当成优势。两条路线都没有学会 `sufficient`，对 `insufficient` 的 recall 也不足 20%。后续应报告 macro-F1、balanced accuracy 和逐类 recall，而不是只看被多数类掩盖的 accuracy。

### 6.5 本节结论

Direct 与 Two-stage 的共同能力远大于路线差异：两者 Schema 都为 99.55%、span grounding 都超过 98.6%、boundary-compatible F1 都超过 83%，而 Evidence 结构能力在 Direct 路线上已经完整形成。Two-stage 的优势主要集中在较低 validation loss、稍好的 grounding、证据匹配和 task routing；Direct 的优势主要是训练链路短、从零成本低。当前证据不支持“前置 Full-SFT 必须存在”，也不支持“Direct 已经全面优于 Two-stage”。

---

## 7. 失败样本与实验限制

### 7.1 格式与 Grounding 失败对比

| 失败类型 | Direct | Two-stage |
|---|---:|---:|
| Strict JSON 失败 | 1 | 2 |
| Schema 失败 | 2 | 2 |
| Sample-level grounding 失败 | 23 | 17 |
| 不落原文的 evidence spans | 21/1,592 | 15/1,575 |
| Critical internal consistency 失败 | 2 | 3 |

Direct 的 23 条 sample-level grounding 失败由 1 条不可解析输出、1 条可解析但 `evidence=[]` 的输出，以及 21 条包含不落原文 span 的输出组成。其 2 条 Schema 失败分别是根对象类型错误和非法 `importance` 值。

Two-stage 的 17 条 sample-level grounding 失败由 2 条不可解析输出、3 条可解析但 `evidence=[]` 的输出，以及 12 条包含 15 个不落原文 span 的输出组成。旧版报告将后 15 个片段误写成 15 条样本，本版已按逐样本预测重新核对并修正。

常见 grounding 失败并非完全捏造病例外事实，而是模型对原文做了轻微删词、改写或错别字纠正。由于项目要求 span 能逐字回查，这些改写仍必须判错。两路失败率都很低，但 Two-stage 少 6 条 sample-level 失败，与总体 grounding 指标的方向一致。

### 7.2 Critical 字段失败

Direct 的 2 条 critical consistency 失败包含 1 条不可解析输出和 1 条内部映射不一致；Two-stage 的 3 条则包含 2 条不可解析输出和 1 条内部不一致。这里衡量的是 `critical_evidence_ids` 与 `importance=critical` 的 JSON 内部关系，应准确命名为 **critical internal consistency**，不能称作 critical clinical accuracy。

### 7.3 最终回答已有初步偏好评测，但尚无临床金标

Evidence held-out evaluator 对 `final_answer` 仍只检查是否非空；新增 CMB-Clin 已使用匿名双位置 Judge 评价流畅度、相关性、完整性、医学专业性和 hard medical error，但换位一致率只有 43.27%，且没有医生金标。因此以下问题仍未被可靠解决：

- 最终诊断或处理建议是否完整、正确和安全；
- 回答是否忠实于 evidence，而不是在证据字段之外加入病例外数字或事实；
- clinical reasoning 是否存在跳步、过度推断或错误因果；
- teacher 的证据和结论是否医学上可靠；
- 模型在长病例、域外病例和真实问诊中的表现。

数据验证阶段的 `accepted` 也只代表通过结构和启发式规则，不等于医学金标。accepted 数据中仍有大量 `generated_numbers` 等 warning；已有抽样可见病例外剂量或阈值进入最终回答的风险。因此 grounding 只能说明“引用片段来自原文”，不能保证“最终回答只使用了这些片段”或“建议临床正确”。CMB-Clin 的自动 Judge 结果同样只能称为“模型裁判偏好与风险标记”，不能称为临床正确率。

### 7.4 路径消融的混杂因素

本实验可以支持“Direct 是否能学会 Evidence 任务”，但不能严格隔离前置 Full-SFT 的因果贡献：

1. **训练量不对等：** Two-stage 额外使用 120k 混合数据和 7,500 steps；
2. **病例预暴露：** Evidence 数据来自 `medical_100k`，该语料也进入了前置 Full-SFT，Two-stage 可能在 held-out Evidence 测试前看过同病例的原始问答；
3. **单次运行：** 两路都只有 seed 42 的一个 checkpoint，没有多 seed 方差；
4. **复现链不完整：** Direct 有当前启动脚本，Two-stage 的历史原始命令和数据快照未完整保留，且本地路径名与历史评测名存在漂移；
5. **评价目标有限：** 当前配对指标主要反映格式、grounding 和 teacher agreement，而非临床效用。

因此最合适的表述是：**Direct 已证明前置 Full-SFT 不是形成结构能力的必要前提；Two-stage 的小幅优势可能来自初始化、额外训练或病例预暴露，当前实验无法区分这三种来源。**

---

## 8. C-Eval 医学知识保持分析

四次 C-Eval 使用相同的三个医学子任务、相同 5-shot 设置和相同 818 道测试题，聚合结果可直接比较。

### 8.1 四条模型路径结果

| 模型 | 基础医学（175） | 临床医学（200） | 医师资格（443） | 总计（818） |
|---|---:|---:|---:|---:|
| E0 Base | 161 / 92.00% | 175 / 87.50% | 396 / 89.39% | **732 / 89.49%** |
| E1 Full-SFT | 160 / 91.43% | 171 / 85.50% | 398 / 89.84% | **729 / 89.12%** |
| E2 Direct | 157 / 89.71% | 176 / 88.00% | 395 / 89.16% | **728 / 89.00%** |
| E3 Two-stage | 158 / 90.29% | 172 / 86.00% | 400 / 90.29% | **730 / 89.24%** |

四组总分位于 728～732 题之间，最大差异只有 4/818 题，即 0.49 个百分点。Direct 总分最低，但仅比 Two-stage 少 2 题、比 Full-SFT 少 1 题、比 Base 少 4 题；同时它在临床医学子任务上反而是四组最高。该模式更符合跨子任务的小幅波动，而不是稳定的知识提升或退化。

### 8.2 Direct 的相对变化

| 对比 | 基础医学 | 临床医学 | 医师资格 | 总分 |
|---|---:|---:|---:|---:|
| Direct vs Base | -4 / -2.29 pp | +1 / +0.50 pp | -1 / -0.23 pp | **-4 / -0.49 pp** |
| Direct vs Full-SFT | -3 / -1.71 pp | +5 / +2.50 pp | -3 / -0.68 pp | **-1 / -0.12 pp** |
| Direct vs Two-stage | -1 / -0.57 pp | +4 / +2.00 pp | -5 / -1.13 pp | **-2 / -0.24 pp** |

Direct 相对 Two-stage 在基础医学少 1 题、临床医学多 4 题、医师资格少 5 题，方向并不一致。Two-stage 的总分优势不能被解释成在所有医学任务上的稳定收益。

需要注意，四个 C-Eval 结果文件只保留任务级聚合分数，没有逐题预测或正误序列，无法做配对 McNemar 检验，也无法给出可信的配对差值置信区间。把同一批题上的两个模型错误地当成独立二项样本进行检验也不合适。因此本节只支持以下结论：

- 两条 Evidence 路线都把 C-Eval 保持在约 89%，未观察到灾难性遗忘；
- Direct 与 Two-stage 的 2 题差距不足以证明前置 Full-SFT 带来稳定知识增益；
- Direct 的临床医学最高、Two-stage 的医师资格最高，说明子任务波动大于统一方向的路线优势；
- C-Eval 是医学知识保护指标，不是生成式医疗问答的完整性、安全性或临床正确率。

### 8.3 D0 vs D1 的 C-Eval 配对检查

DPO 完成后另保存了逐题预测，可对合并后的 D0 与 D1 做配对比较：

| 子任务 | D0 | D1 / Evidence-DPO | 差值 | McNemar p |
|---|---:|---:|---:|---:|
| basic_medicine（175） | 90.286% | 90.286% | 0 | 1.000 |
| clinical_medicine（200） | 87.500% | 89.000% | +1.500 pp | 0.250 |
| physician（443） | 89.391% | 89.391% | 0 | 1.000 |
| 总计（818） | 89.120% | 89.487% | +0.367 pp | 0.375 |

D1 净多答对 3 题，整体差异不显著。该结果与 CMB-Exam 一致：DPO 没有表现出系统性知识退化，也没有足够证据支持知识提升。

---

## 9. 综合能力判断

| 项目目标 | Direct | Two-stage | 判断 |
|---|---:|---:|---|
| Strict JSON | 99.77% | 99.55% | 两路均达成，差 1 条样本 |
| 完整 Schema | 99.55% | 99.55% | 两路均达成 |
| 原文证据引用 | 98.68% span grounding | 99.05% | 两路均达成，Two-stage 略高但差异不显著 |
| Critical 内部一致 | 99.55% | 99.32% | 两路均稳定，不代表临床判断正确 |
| Teacher 严格证据边界 | F1 58.10% | F1 59.49% | 部分达成，Two-stage 方向性略优 |
| 边界兼容证据匹配 | F1 83.36% | F1 84.51% | 两路均较高，F1 差异区间跨 0 |
| Task type | Macro-F1 80.16% | 82.00% | 可用，Two-stage 略优 |
| Evidence sufficiency | Macro-F1 39.10% | 39.60% | 两路均未达成，存在类别坍缩 |
| 医学知识保持 | C-Eval 89.00% | 89.24% | 与 Base 89.49% 基本持平 |
| Final answer 临床质量 | CMB-Clin 自动 Judge 基线 | DPO 净胜率 +6.25 pp，CI 跨 0 | 已初评但证据不足，不能声称稳定提升 |
| DPO 知识保持 | CMB-Exam 77.384% | 77.330% | -0.054 pp，p=0.586，基本不退化 |
| DPO 安全性 | hard error 8.17% | 9.86% | 点估计对 DPO 不利，安全非退化未建立 |

整体评级：**SFT 主实验可 Share with caveats；DPO 模型选择为 Needs revision。**

项目现已形成更完整的实验闭环：prompt 设计 → teacher 蒸馏 → 自动验证与数据分流 → 两条 Evidence-SFT 初始化路径 → Strict Answer-level DPO → Evidence/C-Eval 非退化检查 → CMB-Exam/CMB-Clin 外部评测。Direct 实验的重要价值是证明更短链路足以形成 Evidence 结构能力；DPO 实验则说明“训练目标学会了”不等于“外部临床回答已经稳定改善”。

---

## 10. 当前后续建议

### P0：冻结现有模型与结果，不追加 DPO 训练

保留 Direct Evidence-SFT、Evidence-DPO、训练日志、DPO pair audit、CMB 逐题预测和 Judge 原始记录。当前停止条件已经触发：Clin 净胜率没有稳定超过 0、swap consistency 很低、回答明显变长且 hard-error 点估计上升。此时继续增加 epoch 会放大不确定偏好，不能解决评审证据不足。

### P1：优先做安全人工复核

至少复核 CMB-Clin 中任一模型被两个位置共同确认 hard error 的 13 个问题，优先处理 DPO 独有的 6 个问题：`4:1`、`39:0`、`39:1`、`46:2`、`58:1`、`70:1`。记录专家结论、错误严重度、是否为多轮传播、是否可由提示词修复。自动 Judge 的风险描述只能作为审阅入口，不能直接当作临床终审。

### P2：补强不一致 Judge，而不是重新生成回答

两模型的 208 条回答已经完整、对齐且无截断，无需重新推理。对 118 条换位不一致记录增加第三裁决，或使用第二个固定强 Judge；hard-error 仍应由人工终审。下一次评测前预先定义：

1. 病例聚类净胜率 CI 下界不低于 0；
2. 医学专业性与相关性不退化；
3. hard-error 差值满足预设非劣界；
4. CMB-Exam 知识能力下降不超过预设界值；
5. 长度匹配或去冗余敏感性分析仍保持收益。

### P3：修复偏好数据的长度混杂

正式 DPO pair 中 chosen 更长率为 81.32%，Clin 中 DPO 也平均多 27.4 tokens。下一版数据应在 Judge rubric 中明确区分“覆盖关键内容”与“单纯扩写”，报告 chosen/rejected 长度分层，并优先加入短而正确对长而冗余的反例。若没有新的高质量偏好数据，不启动 DPO v2。

### P4：保留 Evidence-Mask 为独立消融

Evidence-Mask 仍应作为独立实验回答“删除关键证据后，模型能否降低确定性并调整缺失信息”，不能混入本轮 DPO 结论。其训练、评测和报告应使用独立模型名与结果目录。

面向用户时仍不直接展示原始 JSON，而由服务层渲染为结论、支持证据、缺失信息和风险提醒；该展示转换不需要再次训练模型。

---

## 11. 简历与面试表述建议

### 11.1 推荐项目表述

> 基于 Qwen2.5-7B-Instruct 构建 Evidence-SFT 与 Strict Answer-level DPO 流水线：使用 10k 条 teacher 数据完成 accepted/review/rejected 分流，在 8,995 条 accepted 数据上验证 Direct Evidence-SFT 可达到 99.55% Schema 合法率和 98.68% evidence grounding；进一步构造 1,243/101 个 DPO train/validation pair，并用 CMB-Exam 11,200 题和 CMB-Clin 74 个病例进行外部评测。DPO 的 CMB-Exam 为 77.33%，相对父模型仅 -0.054 pp；Clin 净胜率点估计 +6.25 pp，但换位一致率只有 43.27% 且安全非退化未建立，因此保留 DPO 为候选而不夸大结论。

### 11.2 面试时应主动说明

- 98.68%/99.05% 是“预测证据能否在病例原文逐字找到”，不是医疗正确率；
- 58.10%/59.49% 是与 teacher 的严格 span 边界一致性，不是临床准确率；
- 两路 sufficiency accuracy 都低于固定预测 `partial` 的 87.61% 基线；
- Two-stage 的小幅优势没有通过多数配对显著性判断，且受到额外训练量与病例预暴露混杂；
- 四组 C-Eval 最大只差 4/818 题，不主张显著提升或退化；
- 当前为单 seed 实验；final answer 已有自动 Judge，但换位稳定性不足，且尚无医生金标；
- JSON 是模型输出协议，实际产品会转换成人类可读页面。

### 11.3 面试问题的简洁回答

**为什么要比较 Direct 和 Two-stage？**  
为了判断 Evidence 结构能力究竟来自专门的 Evidence 数据，还是必须依赖前置 Full-SFT。结果显示 Direct 已能达到 99.55% Schema 和 98.68% grounding，说明前置阶段不是必要条件；Two-stage 只在部分证据和路由指标上小幅领先。

**为什么不能直接说 Direct 和 Two-stage 一样好？**  
Two-stage 的 validation loss、grounding、证据匹配和 task type 指标方向上更好；只是差距较小，多数置信区间跨 0，而且存在病例预暴露等混杂。所以结论是“前置 Full-SFT 非必要”，不是“两者严格等价”。

**为什么训练 2 个 epoch？**  
两路验证损失都持续下降且后期趋缓，没有反弹。Direct 与 Two-stage 的最终 validation loss 为 0.6116 和 0.6047，两轮已完成主要收敛，同时控制训练成本和过拟合 teacher 表达的风险。

**如果从零搭建，为什么更倾向 Direct？**  
Direct 约 1.97 小时即可完成；Two-stage 若计入前置训练约 18.79 小时，成本高约一个数量级，而当前核心指标差距只有约 0～2 个百分点。如果 Full-SFT 已经存在，则可以利用 Two-stage 的小幅起点优势。

**DPO 应该从哪个模型开始？**  
本轮已经选择 Direct Evidence-SFT 作为 D0，并在其合并权重上训练新 DPO LoRA；reference 是禁用新 DPO adapter 后的同一个 D0。这个起点关系是正确的，不能再把 Two-stage 或原始 Base 混作 reference。

**DPO 训练成功了吗？**
优化过程成功：验证偏好准确率 88.12%，reward margin 0.286。外部效果尚未成功验证：CMB-Exam 证明知识基本不退化，但 CMB-Clin 的净胜率区间跨 0、换位一致率仅 43.27%，所以应表述为“训练完成，临床收益待验证”。

**JSON 对人不友好怎么办？**  
JSON 服务于后端校验和字段解析，不是最终 UI。前端只展示结论、证据和缺失信息即可。

**项目最大的不足是什么？**  
Evidence sufficiency 类别严重不均衡，两路都坍缩到 `partial`；DPO 偏好数据存在明显长度倾向；CMB-Clin 自动 Judge 稳定性不足且没有医生终审，因此最终回答的临床正确性和安全性仍未被可靠验证。

---

## 12. 数据来源与复核文件

本报告基于以下本地文件。JSON/JSONL 是数值来源，Markdown 报告用于解释；若两者冲突，以机器可读结果为准。

### 12.1 Evidence-SFT

- 数据验证统计：`data/evidence_sft/03_validation.stats.json`
- 训练集：`data/evidence_sft/train/train.jsonl`
- 验证集：`data/evidence_sft/validation/validation.jsonl`
- 测试集：`data/evidence_sft/test.jsonl`
- Direct 训练入口：`process_sft/run_evidence_sft.sh`
- Evidence 自动评测实现：`process_sft/evaluate_evidence.py`
- Direct 训练汇总：`outputs/evidence-sft-frombase/train_results.json`
- Direct 验证汇总：`outputs/evidence-sft-frombase/eval_results.json`
- Direct 训练过程：`outputs/evidence-sft-frombase/trainer_state.json`
- Direct Evidence 评测：`results/evidence-frombase_eval/metrics.json`
- Direct 逐样本预测：`results/evidence-frombase_eval/predictions.jsonl`
- Two-stage 训练汇总：`outputs/evidence-sft/train_results.json`
- Two-stage 验证汇总：`outputs/evidence-sft/eval_results.json`
- Two-stage 训练过程：`outputs/evidence-sft/trainer_state.json`
- Two-stage Evidence 评测：`results/evidence_eval/metrics.json`
- Two-stage 逐样本预测：`results/evidence_eval/predictions.jsonl`
- Base C-Eval：`results/base_ceval.json`
- Full-SFT C-Eval：`results/sft_ceval.json`
- Direct C-Eval：`results/evidence-frombase.json`
- Two-stage C-Eval：`results/evidence-sft.json`

### 12.2 Strict Answer-level DPO

- DPO 流程与运行命令：`process_dpo/README.md`
- 数据导出统计：`data/dpo/answer_v1/04_export.stats.json`
- 数据导出验证：`data/dpo/answer_v1/04_export.validation.json`
- 正式训练集：`data/dpo/answer_v1/train/train.jsonl`
- 正式验证集：`data/dpo/answer_v1/validation/validation.jsonl`
- DPO 训练 manifest：`outputs/evidence-dpo-answer-v1/dpo_training_manifest.json`
- DPO 训练与验证汇总：`outputs/evidence-dpo-answer-v1/all_results.json`
- DPO adapter 配置：`outputs/evidence-dpo-answer-v1/adapter_config.json`
- D0/D1 C-Eval 配对比较：`results/dpo_ceval_comparison.json`

### 12.3 CMB-Exam

- 四模型汇总：`cmb_eval/results/four_models_test/summary.json`
- 四模型表格：`cmb_eval/results/four_models_test/summary.csv`
- SFT vs DPO 配对比较：`cmb_eval/results/four_models_test/evidence_sft_vs_dpo.json`
- test 数据完整性：`cmb_eval/results/four_models_test/test_data_validation.json`
- 详细分析报告：`cmb_eval/results/four_models_test/CMB-Exam四模型评测分析报告.md`
- 评测入口：`cmb_eval/run_exam_four_models.sh`

### 12.4 CMB-Clin

- 两模型汇总：`cmb_eval/results/clin_dpo/evidence_sft_vs_dpo_summary.json`
- 原始双位置 Judge：`cmb_eval/results/clin_dpo/judgments.jsonl`
- Evidence-SFT 回答：`cmb_eval/results/clin_dpo/evidence_sft/predictions.jsonl`
- Evidence-DPO 回答：`cmb_eval/results/clin_dpo/evidence_dpo/predictions.jsonl`
- 详细分析报告：`cmb_eval/results/clin_dpo/CMB-Clin_Evidence-SFT_vs_Evidence-DPO_结果分析报告.md`
- 评测入口：`cmb_eval/run_clin_dpo_effect.sh`

命名说明：Two-stage 评测元数据保留历史 Linux 路径 `/home/medgpt/outputs/evidence-sft-v2-2`，当前工作区对应 `outputs/evidence-sft`。DPO adapter 的 base 是合并后的 Direct D0；CMB 推理通过同一个 Base + Evidence-SFT primary adapter，并仅在候选侧叠加 DPO adapter，避免把不同父模型混入 CMB 主比较。

## 13. 最终结论

本项目已经完成从 Evidence-SFT 数据构建、质量验证、初始化路径消融，到 Strict Answer-level DPO 训练及 Evidence/C-Eval/CMB-Exam/CMB-Clin 评测的闭环。

SFT 阶段的结论稳定：**Direct Evidence-SFT 足以获得可检查、可追溯的结构化证据能力，前置 Full-SFT 不是必要条件。**DPO 阶段的结论必须分开表述：训练过程正常收敛，内部偏好准确率达到 88.12%；CMB-Exam 显示其医学知识与父模型基本等价；但 CMB-Clin 只有小幅完整性与净胜率正向点估计，关键区间全部跨 0，Judge 换位一致率仅 43.27%，hard-error 点估计对 DPO 不利。

因此当前决策是：**Evidence-SFT 保持为默认模型，Evidence-DPO 保留为待复核候选；停止追加 DPO 训练，优先完成人工安全复核、对 118 条不一致结果补裁决，并修正偏好数据的长度混杂。**对外可以展示结构可靠性、知识保持和完整实验链路，但不能把 teacher agreement、原文 grounding 或自动 Judge 偏好包装成临床正确率，也不能宣称 DPO 已稳定提升临床能力。
