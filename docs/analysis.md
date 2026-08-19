# Evidence-SFT 项目训练与评测分析报告

> 报告日期：2026-08-12  
> 基座模型：Qwen2.5-7B-Instruct  
> 对照路线：Base→Evidence-SFT（Direct）与 Base→Full-SFT→Evidence-SFT（Two-stage）  
> 评估结论：**直接从 Base 训练已足以形成稳定的结构化证据输出；前置 Full-SFT 在损失、证据选择和任务路由上仅呈小幅方向性优势，当前单次实验不足以证明该前置阶段是必要条件。两条路线均未验证最终回答的临床正确性。**

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

### 7.3 最终回答仍缺乏质量评测

当前 evaluator 对 `final_answer` 只检查是否非空，没有检查：

- 最终诊断或处理建议是否完整、正确和安全；
- 回答是否忠实于 evidence，而不是在证据字段之外加入病例外数字或事实；
- clinical reasoning 是否存在跳步、过度推断或错误因果；
- teacher 的证据和结论是否医学上可靠；
- 模型在长病例、域外病例和真实问诊中的表现。

数据验证阶段的 `accepted` 也只代表通过结构和启发式规则，不等于医学金标。accepted 数据中仍有大量 `generated_numbers` 等 warning；已有抽样可见病例外剂量或阈值进入最终回答的风险。因此 grounding 只能说明“引用片段来自原文”，不能保证“最终回答只使用了这些片段”或“建议临床正确”。

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
| Final answer 临床质量 | 未评测 | 未评测 | 不能对外声称正确、安全或完整 |

整体评级：**Share with caveats（可展示，但必须附带指标边界）**。

项目现已形成更完整的实验闭环：prompt 设计 → teacher 蒸馏 → 自动验证与数据分流 → 两条 Evidence-SFT 初始化路径 → 444 条配对 Evidence 评测 → 四组 C-Eval 知识保持对照。新增 Direct 实验的重要价值不在于“击败”Two-stage，而在于回答了一个更有工程意义的问题：**不经过前置 Full-SFT，Base 也能用更短链路获得几乎同等的 Evidence 结构能力。**

---

## 10. 后续建议

### P0：冻结两条路线的可复核资产

当前应同时保留 Direct 与 Two-stage，而不是只保留一个“最终模型”：

- 两个 LoRA 适配器及训练日志；
- 两套 444 条预测与聚合指标；
- Base、Full-SFT、Direct、Two-stage 四组 C-Eval 汇总；
- 数据版本、chat template、seed 和评测参数；
- Two-stage 的本地目录名与历史 `evidence-sft-v2-2` 名称映射。

还应修复复现入口的叙述漂移：当前 `run_evidence_sft.sh` 实际复现的是 Direct，不是 Two-stage；Two-stage 的原始启动命令需要根据历史记录补档，不能用当前脚本替代。

### P1：先补最终回答质量对照，再决定 DPO 起点

现有 evaluator 只检查 `final_answer` 非空，因此无法根据当前指标判断哪条路线的用户可见答案更完整、更安全。建议在同一批 prompt 上进行 Direct vs Two-stage 的交换顺序双盲 A/B，至少评价：

1. 回答完整性；
2. 对病例与证据的忠实性；
3. 安全保守性和不确定性表达；
4. 病例外数字、剂量与阈值的引入风险。

如果必须在现有指标下先选一个 DPO 起点，可暂以 Two-stage 为主策略候选，因为它的 validation loss、grounding、证据匹配、task routing 和 C-Eval 多数方向略好；但这只是保守工程选择，不是显著优胜结论。若盲评没有确认用户可见答案优势，则 Direct 因链路更短、从零成本更低、因果解释更清晰，应成为优先方案。

### P2：DPO 必须使用同一起点构造 policy/reference

DPO 时应复制同一个选定的 Evidence checkpoint：一份冻结为 reference，一份作为 policy 继续训练。不要把 Direct 当作 Two-stage policy 的 reference，或反过来，否则偏好优化效果会与初始化路径差异混在一起。

两路模型可以共同生成候选答案以增加偏好对多样性，但“答案来自哪条路线”不能直接充当 chosen/rejected 标签。偏好对需要在同一 prompt 下比较，优先固定证据 JSON 结构，仅围绕用户可见 `final_answer` 的完整性、忠实性和安全性构造差异。

### P3：补齐低成本评测项

在下一轮训练前，优先做以下低成本改进：

1. evaluator 固定输出多数类 baseline、macro-F1、balanced accuracy 和逐类 recall；
2. 同时保留 exact 与 boundary-compatible 证据指标；
3. 保存 C-Eval 逐题预测，支持配对检验和错误迁移分析；
4. 增加 masked/unmasked 对照集，验证“关键证据 Mask”是否真正改善证据缺失时的不确定性表达；
5. 若资源允许，对核心路线补 2～3 个 seed，再判断 1 个百分点左右的差异是否稳定。

面向用户时仍不应直接展示原始 JSON，而应在服务层渲染为结论、支持证据、缺失信息和风险提醒。该展示转换不需要再次训练模型。

---

## 11. 简历与面试表述建议

### 11.1 推荐项目表述

> 基于 Qwen2.5-7B-Instruct 设计 Evidence-SFT 数据协议与自动质量验证流程，使用 10k 条 teacher 蒸馏数据完成 accepted/review/rejected 三路分流，并在 8,995 条 accepted 数据上比较 Base 直训与 Full-SFT 后续训两条 LoRA 路径。两路在同一 444 条测试集上的 Schema 合法率均为 99.55%，证据片段原文 grounding 分别为 98.68% 和 99.05%；医学 C-Eval 分别为 89.00% 和 89.24%，与 Base 89.49% 基本持平。实验表明，前置 Full-SFT 不是形成 Evidence 结构能力的必要条件。

### 11.2 面试时应主动说明

- 98.68%/99.05% 是“预测证据能否在病例原文逐字找到”，不是医疗正确率；
- 58.10%/59.49% 是与 teacher 的严格 span 边界一致性，不是临床准确率；
- 两路 sufficiency accuracy 都低于固定预测 `partial` 的 87.61% 基线；
- Two-stage 的小幅优势没有通过多数配对显著性判断，且受到额外训练量与病例预暴露混杂；
- 四组 C-Eval 最大只差 4/818 题，不主张显著提升或退化；
- 当前为单 seed 实验，final answer 尚无医学 Judge 或人工盲评；
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
先用同 prompt 双盲评价最终回答。若必须基于现有指标选择，可暂用 Two-stage；若盲评无明确优势，则用 Direct 简化链路。无论选哪条，都应复制同一 checkpoint 分别作为 policy 和 reference，不能拿另一条路线当 reference。

**JSON 对人不友好怎么办？**  
JSON 服务于后端校验和字段解析，不是最终 UI。前端只展示结论、证据和缺失信息即可。

**项目最大的不足是什么？**  
Evidence sufficiency 类别严重不均衡，两路都坍缩到 `partial`；此外自动评测只验证结构、原文引用和 teacher 一致性，没有验证最终回答的临床正确性和安全性。

---

## 12. 数据来源与复核文件

本报告基于以下本地文件，未重新运行训练或评测：

- 数据验证统计：`data/evidence_sft/validated_v2_2/03_validation.stats.json`
- 训练集：`data/evidence_sft/validated_v2_2/train.jsonl`
- 验证集：`data/evidence_sft/validated_v2_2/validation.jsonl`
- 测试集：`data/evidence_sft/validated_v2_2/test.jsonl`
- 当前 Direct 训练入口：`evi-sft-traing/run_evidence_sft.sh`
- Evidence 自动评测实现：`evi-sft-traing/evaluate_evidence.py`
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

命名说明：Two-stage 评测元数据保留了历史 Linux 路径 `/home/medgpt/outputs/evidence-sft-v2-2`，当前工作区对应的本地适配器目录为 `outputs/evidence-sft`。现有权重与过程文档共同支持该映射，但原始启动命令日志未完整保留；因此当前 `run_evidence_sft.sh` 只能视为 Direct 的复现入口，不能作为 Two-stage 的完整复现证据。

## 13. 最终结论

本项目已经完成从数据构建、质量验证、两条 Evidence-SFT 路径到独立 Evidence/C-Eval 评测的闭环。最重要的新结论是：**直接从 Qwen2.5-7B-Instruct 进行 Evidence-SFT，已经足以获得稳定、可检查、可追溯的结构化证据能力，并把医学 C-Eval 保持在约 89%；前置 Full-SFT 不是形成这项能力的必要前提。**

Two-stage 在 validation loss、grounding、teacher 证据匹配和 task routing 上有约 0.4～1.8 个百分点的方向性优势，但大多数配对区间跨越 0，而且受额外训练量、单 seed 和病例预暴露混杂。它可以作为当前偏保守的候选路线，却不能被表述为已显著优于 Direct。若从零建设，Direct 的成本与解释性更有优势。

最终建议是：**冻结两条 checkpoint 与全部配对结果，先补 final answer 的忠实性、完整性和安全性盲评，再确定 DPO 起点。对外以结构可靠性、证据 grounding、知识保持和诚实的实验限制为主线，不把 teacher agreement 或原文可回查率包装成临床正确率。**
