可以。你面试时最适合讲的不是“我改了三次 Prompt”，而是：

> **我通过对蒸馏结果做 error analysis，发现 Evidence-SFT 数据在证据充分性、证据边界、关键证据判定和证据粒度上存在系统性噪声，因此进行了三轮针对性 Prompt 迭代：V1.1→V2.0→V2.1→V2.2。**

这样听起来是**问题驱动的数据工程迭代**，而不是凭感觉调 Prompt。

下面这份可以直接作为项目总结。

---

# Evidence-SFT Prompt 迭代总结

## 1. 背景与目标

项目目标是将原始低质量医疗 QA 重构成结构化的 Evidence-grounded SFT 数据：

```text
原始病例 QA
    ↓
Teacher LLM
    ↓
Query Intent
    +
Evidence
    +
Evidence Sufficiency
    +
Critical Evidence
    +
Missing Information
    +
Clinical Reasoning
    +
Final Answer
```

最终希望模型学到的不只是：

```text
Question → Answer
```

而是：

```text
Patient Evidence
       ↓
能支持什么结论
       ↓
证据还缺什么
       ↓
回答应该有多确定
       ↓
Final Answer
```

同时 `critical evidence` 会进一步用于后续的 **Evidence Mask / 反事实训练**，因此对 Evidence 的精度要求远高于普通信息抽取。

---

# 2. 第一轮改进：V1.1 → V2.0

## V1.1 的初始设计

V1.1 已经具备基本的 Evidence-grounded 数据结构：

```json
{
  "task_type": "...",
  "evidence_sufficiency": "...",
  "evidence": [...],
  "critical_evidence_ids": [...],
  "missing_information": [...],
  "clinical_reasoning": "...",
  "final_answer": "..."
}
```

核心思想是：

> Evidence 必须逐字来自 `case_text`，Teacher 不能自己虚构患者事实。

这个阶段首先解决了一个非常基础的问题：

### 原始医疗 SFT 数据质量不可靠

例如原答案可能包含：

- 错误诊断；
- 不可靠治疗方案；
- 过度确定；
- 缺乏证据的医学推断。

因此 Teacher 不直接模仿 `original_answer`，而是根据 `case_text` 重新构造答案。

---

## 发现的问题 1：`sufficient` 定义过宽

以肝硬化病例为例。

病例只告诉：

```text
患者已确诊肝硬化
```

但缺少：

```text
病因
分期
肝功能
并发症
```

Teacher 却输出：

```json
"evidence_sufficiency": "sufficient"
```

原因是 V1.1 中的逻辑相当于：

```text
只要还能给一般管理原则
→ sufficient
```

但实际上：

```text
可以给一般知识
≠
证据足以回答用户所要求的个体化问题
```

---

## 改进 1：加入 `partial`

V2.0 将：

```text
sufficient
insufficient
conflicting
```

扩展为：

```text
sufficient
partial
insufficient
conflicting
```

其中：

### sufficient

病例证据已经足够支持用户要求的主要结论。

### partial

可以进行方向性判断或给一般原则，但：

- 不能确定具体诊断；
- 不能制定个体化治疗；
- 不能进行准确风险或预后判断。

### insufficient

连有针对性的方向性判断都无法支持。

---

## 动机

将：

```text
“能不能回答”
```

和：

```text
“能回答到什么粒度”
```

区分开。

本质上是在做：

> **Evidence Coverage Calibration**

---

## 效果

例如干咳病例：

### V1.1

```json
"evidence_sufficiency": "sufficient"
```

### V2.0

```json
"evidence_sufficiency": "partial"
```

因为病例能够支持：

```text
持续性咳嗽需要进一步寻找原因
```

但不能支持：

```text
确定具体疾病 + 给个体化治疗方案
```

这个变化明显改善了模型的**不确定性边界**。

---

# 3. V2.0 的第二个重要改进：Evidence Importance

V1.1 只有：

```json
"critical_evidence_ids": [...]
```

但没有显式区分普通证据和关键证据。

V2.0 给 Evidence 加入：

```json
{
  "importance": "critical|supporting"
}
```

定义：

### critical

只删除这条证据、其他信息不变时：

```text
主要结论
或
回答范围
或
不确定性
```

会明显变化。

### supporting

对回答有帮助，但删除之后主体回答基本不变。

---

## 动机

这个字段不是单纯为了让 JSON 更丰富。

它直接服务于后面的：

```text
Critical Evidence
       ↓
Evidence Mask
       ↓
Counterfactual Case
       ↓
观察模型输出变化
```

即：

\[
x
\rightarrow f(x)
\]

删除 Evidence \(e_i\)：

\[
x-e_i
\rightarrow f(x-e_i)
\]

观察：

\[
\Delta_i
=
D(f(x),f(x-e_i))
\]

如果是关键证据，希望：

\[
\Delta_{critical}
>
\Delta_{supporting}
\]

所以 `critical evidence` 的 Precision 非常重要。

---

# 4. V2.0 带来的效果

V2.0 相比 V1.1：

```text
✔ Sufficiency 更细
✔ 能显式区分 Critical / Supporting
✔ Clinical reasoning 开始体现证据边界
✔ Missing information 更聚焦
```

但是实际跑样本后又发现三个新问题。

---

# 5. V2.0 暴露的问题

以“两个月持续干咳”的病例为例。

V2.0 输出：

```json
E1:
感冒后一直干咳两个月
critical

E2:
无其他症状
supporting

E3:
多种抗生素和止咳药无效
critical

E4:
可能得什么病，怎么治疗
question_scope
```

主要发现三个问题。

---

## 问题 A：用户问题被混进 Evidence

例如：

```text
“可能得的什么病，怎么治疗”
```

它描述的是：

> 用户想知道什么。

不是：

> 模型判断疾病所依据的患者事实。

因此存在概念混淆：

```text
Question Scope
≠
Clinical Evidence
```

而且以后进行 Evidence Mask 时，也绝对不应该去 Mask：

```text
“怎么治疗？”
```

---

## 问题 B：Critical Evidence 偏多

例如：

```text
“多种抗生素和止咳药无效”
```

V2.0 标成：

```text
critical
```

但如果把它删掉，只保留：

```text
感冒后持续干咳两个月
无其他症状
```

主体回答仍然会是：

```text
持续性咳嗽
→ 需要进一步寻找原因
→ 当前不能确定具体病因
```

所以：

```text
Answer(x)
≈
Answer(x - E3)
```

它更应该是：

```text
supporting
```

而不是 critical。

---

## 问题 C：`role` 出现过度推断

例如 V2.0 曾输出：

```text
抗生素无效
→ 排除了常见感染性病因
```

但：

```text
治疗没效果
```

不能直接推出：

```text
病因已经被排除
```

这是一种典型的：

> **Unsupported Post-hoc Interpretation**

也就是说：

```text
Evidence span 本身真实
```

但：

```text
Teacher 对 Evidence 的解释超出了 Evidence 能支持的范围
```

---

# 6. 第二轮改进：V2.0 → V2.1

V2.1 就是针对这三个问题进行定向修复。

---

## 改进 1：拆分 `query_intent` 和 Evidence

从：

```text
Evidence
├── patient_fact
└── question_scope
```

改成：

```text
query_intent
=
用户想回答什么

evidence
=
患者实际提供了什么事实
```

新的 Schema：

```json
{
  "query_intent": [
    "判断持续干咳的可能原因",
    "咨询下一步检查和治疗建议"
  ],

  "evidence": [
    ...
  ]
}
```

这样概念变得非常清晰：

```text
Query Intent
→ 我要回答什么

Evidence
→ 我凭什么回答

Missing Information
→ 还缺什么

Clinical Reasoning
→ 当前证据最多能支持到哪里
```

---

## 改进 2：Critical 判定变得更保守

V2.1 明确要求 Teacher 做反事实判断：

> 如果只删除该 Evidence，其他病例事实完全不变，主要结论、回答范围或者 sufficiency 是否明显改变？

如果无法明确判断：

```text
优先 supporting
```

而不是：

```text
优先 critical
```

并允许：

```json
"critical_evidence_ids": []
```

避免 Teacher 为了满足格式而强行制造 Critical Evidence。

---

## 动机

因为后面 Mask 阶段：

> **Critical Recall 并不是最重要的，Critical Precision 才更重要。**

宁愿少 Mask 一部分真正的重要 Evidence，也不能大量 Mask 实际并不关键的内容，然后强行让模型学习“不确定”。

---

## 改进 3：严格控制 `role` 推断强度

增加规则：

弱证据不能写：

```text
排除
证明
证实
确定
```

例如：

### V2.0

```text
多种抗生素无效
→ 排除了感染性病因
```

### V2.1

变成：

```text
多种抗生素和止咳药使用后无明显改善
→ 提示需要重新评估病因
```

后者明显更加 Evidence-grounded。

---

# 7. V2.1 实际效果

同一个干咳病例：

## V2.0

```text
E1 critical
E3 critical
```

## V2.1

变成：

```text
E1 critical
E2 supporting
```

并且：

```json
"query_intent": [
  "评估持续干咳的可能病因",
  "咨询下一步检查和治疗建议"
]
```

成功把问题范围从 Evidence 中移了出去。

所以 V2.1 主要解决了：

```text
✔ Evidence / Question 边界
✔ Critical Precision
✔ Unsupported Role
```

这轮改进是比较成功的。

---

# 8. V2.1 又暴露出新的问题：Evidence 粒度太粗

V2.1 对病例：

```text
感冒好了以后，一直干咳，有两个月了，但是无其他症状。
```

抽成：

```json
{
  "span":
  "感冒好了以后，一直干咳，有两个月了，但是无其他症状。",
  "importance": "critical"
}
```

问题是：

一个 Evidence 里面实际包含四种信息：

```text
① 感冒后
② 干咳
③ 两个月
④ 无其他症状
```

如果后续 Mask 整个 Evidence：

```text
[MASK]
```

一次性删除了四个变量。

那么即使模型输出发生变化，也无法解释：

> 到底是哪条临床信息造成了变化？

---

# 9. 第三轮改进：V2.1 → V2.2

V2.2 的核心思想是：

# Evidence Atomicity

即：

> **每条 Evidence 尽可能表示一个可以独立删除、独立解释的最小临床事实。**

---

## V2.1

```text
E1 =
感冒好了以后
+
一直干咳
+
两个月
+
无其他症状
```

全部作为：

```text
critical
```

---

## V2.2

拆分：

```text
E1：
感冒好了以后
→ supporting

E2：
一直干咳，有两个月了
→ critical

E3：
但是无其他症状
→ supporting

E4：
多种药物治疗无效
→ supporting
```

然后：

```json
"critical_evidence_ids": ["E2"]
```

---

# 10. 为什么 Atomic Evidence 对项目非常重要

因为后面要做：

\[
Importance(e_i)
=
D[f(x),f(x-e_i)]
\]

这里的 \(e_i\) 如果是：

```text
症状
+
持续时间
+
诱因
+
阴性症状
```

那这个 Importance 根本不可解释。

而如果：

```text
e_i = “持续干咳两个月”
```

就可以比较：

```text
删除关键证据
vs
删除 Supporting Evidence
```

例如：

```text
Mask “持续干咳两个月”
→ 模型诊断方向明显变化

Mask “感冒以后”
→ 主体回答基本保持

Mask “无其他症状”
→ 主体回答基本保持
```

那么才能得到：

\[
\Delta_{critical}
>
\Delta_{supporting}
\]

这会为后面：

```text
Evidence Mask
Evidence Sensitivity
Counterfactual Training
```

提供更强的解释性。

---

# 11. V2.2 还顺便优化了字段设计

你的原数据同时保留：

```text
case_text
original_question
```

而实际上：

```text
original_question:
内科：病例……

case_text:
病例……
```

高度重复。

而 `category` 已经单独保存：

```json
"category": "内科"
```

同时 Teacher 实际请求中本来也只需要：

- `task_type_hint`
- `case_text`
- `original_answer`

现有构造脚本也是围绕这几个字段生成请求。

因此 V2.2 将：

```text
case_text
```

作为唯一 canonical patient input。

删除冗余：

```text
original_question
```

最终：

```text
category
+
case_text
+
original_answer
```

即可。

---

# 12. 三轮改进的整体逻辑

可以压缩成下面这张表。

| 阶段 | 发现的问题 | 改进 | 目的 |
|---|---|---|---|
| **V1.1 → V2.0** | sufficient 太宽；Evidence 没有重要性分级 | 加 `partial`；加 `critical/supporting` | 学习证据覆盖程度与重要性 |
| **V2.0 → V2.1** | question 混进 evidence；critical 偏多；role 过度推断 | 独立 `query_intent`；critical 保守化；限制 role 推断 | 提高 Evidence / Critical Precision |
| **V2.1 → V2.2** | 一条 Evidence 包含多个临床事实 | Atomic Evidence + Minimal Critical Span | 为反事实 Mask 和 Importance 分析提供可解释粒度 |

---

# 13. 三个阶段背后的研究逻辑

其实每一版关注的东西是不一样的。

### V2.0 解决：

> **Evidence Sufficiency**

即：

```text
当前信息够不够？
```

---

### V2.1 解决：

> **Evidence Validity / Importance**

即：

```text
哪些才是真证据？
哪些是真正关键的证据？
```

---

### V2.2 解决：

> **Evidence Granularity**

即：

```text
关键证据到底应该切到什么粒度，
才能进行可靠的反事实干预？
```

所以整个 Prompt 演进可以概括成：

```text
V1.1
Evidence Grounding
      ↓
V2.0
Evidence Sufficiency
      ↓
V2.1
Evidence Precision
      ↓
V2.2
Evidence Atomicity
```

这个逻辑在面试里很好讲。

---

# 14. 面试时可以这样介绍

你可以用这一段：

> 项目早期我并没有直接批量蒸馏全部数据，而是先抽取约百条样本进行 Teacher 输出分析。我发现单纯要求模型抽取病例 Evidence 会存在几个系统性问题：首先模型会把“还能提供一般医学建议”误认为证据充分；其次会把用户的问题和真正的患者事实混在一起；另外 Teacher 容易把大量辅助证据都标成 Critical，并对证据作用进行过度推断。
>
> 因此我进行了三轮 Prompt 迭代。第一轮增加 `partial` sufficiency 和 Critical/Supporting Evidence，使模型能够描述证据覆盖程度和重要性；第二轮把 Query Intent 与 Patient Evidence 完全解耦，并通过反事实删除原则收紧 Critical Evidence 的判定，同时限制 Evidence role 的推断强度；第三轮进一步引入 Atomic Evidence，将一条包含多个临床事实的 Evidence 拆成可独立删除的最小 span，为后续 Evidence Mask 和 Evidence Sensitivity 实验提供可解释的干预单位。

这段非常适合面试。

---

# 15. 如果面试官继续问：“为什么要这么折腾 Prompt？”

可以回答：

> 因为我的最终目标不是生成看起来更漂亮的医疗答案，而是构造可以用于后训练和反事实训练的数据。对于普通 SFT，Evidence 稍微粗一点影响可能不大，但我的后续训练会删除 Critical Evidence 来构造信息不足病例。如果 Critical 标注错误或者一个 span 同时包含多个事实，那么反事实样本本身就是噪声。所以我的优化目标逐渐从“回答质量”转变成了“Evidence Precision 和 Intervention Validity”。

这一句非常重要：

> **不是优化 Prompt 文案，而是在优化后训练数据的 causal validity。**

---

# 16. 面试官如果问：“你怎么证明新版更好？”

不要回答：

> “我肉眼觉得新版更好。”

你可以说计划/实际使用这些指标：

### Evidence Precision

\[
\frac{\text{人工判断真正相关的 Evidence}}
{\text{Teacher 输出的 Evidence}}
\]

---

### Critical Evidence Precision

\[
\frac{\text{真正满足删除反事实标准的 Critical}}
{\text{Teacher 标记的 Critical}}
\]

这是最重要的。

---

### Sufficiency Accuracy

人工判断：

```text
sufficient
partial
insufficient
conflicting
```

是否合理。

---

### Unsupported Role Rate

\[
\frac{\text{存在超出 span 支持范围推断的 Evidence}}
{\text{总 Evidence}}
\]

例如：

```text
抗生素无效
→ 排除感染
```

就算一次错误。

---

### Evidence Atomicity

检查一条 Evidence：

> 是否同时包含多个作用不同、可继续拆分的临床事实。

---

# 17. 最终可以把这个模块总结成一句简历语言

比较普通的写法：

> 使用大模型重构医疗 SFT 数据。

不够好。

可以写成：

> **设计 Evidence-grounded 医疗数据蒸馏流水线，通过多轮 Error-driven Prompt Optimization，引入 Evidence Sufficiency、Critical/Supporting Evidence、Query–Evidence 解耦和 Atomic Evidence Extraction，提升 Teacher 蒸馏数据的证据可追溯性，并为后续反事实 Evidence Mask 与偏好优化构造高质量训练样本。**

这句话基本把你的 Prompt 迭代价值全部讲出来了。

最核心的故事线就是：

> **V2.0 解决“证据够不够”，V2.1 解决“哪些证据真正重要”，V2.2 解决“关键证据应该切多细”。**

这三个问题逐层递进，而且直接服务于后面的 Evidence Mask，而不是为了单纯把 Prompt 越写越长。


在 Teacher 选型阶段，我没有只比较模型最终回答的主观质量，而是针对后续反事实训练需求，把评价拆成 Evidence Precision、Critical Evidence Precision、Atomicity 和 Unsupported Role Rate。实验中发现不同 Teacher 存在明显角色差异：部分模型医学回答更加丰富，但会倾向于扩大 Critical Evidence 范围；另一些模型在结构化证据抽取和反事实重要性判断上更加保守。因此 Teacher 的选择不是简单比较“哪个模型更强”，而是看其输出特性是否匹配后训练数据构造目标。