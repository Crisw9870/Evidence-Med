关键证据 Mask 消融实验的核心目的，是验证：

> Teacher 标注为 `critical` 的证据，是否真的对模型的主要判断具有因果作用；而不是仅仅“看起来重要”。

它不是单纯测试模型准确率，而是在验证 `critical_evidence_ids` 这个标签是否有真实意义。

## 1. 为什么需要这个实验

当前每条样本大致是：

```text
case_text
    ↓
evidence
    ↓
critical_evidence_ids
    ↓
clinical_reasoning / final_answer
```

例如：

```text
患者男性，45岁，反复胸痛2周，活动后加重，心电图提示ST段异常。
```

Teacher 可能标注：

```json
{
  "evidence": [
    {
      "id": "E1",
      "span": "反复胸痛2周",
      "importance": "critical"
    },
    {
      "id": "E2",
      "span": "活动后加重",
      "importance": "supporting"
    },
    {
      "id": "E3",
      "span": "心电图提示ST段异常",
      "importance": "critical"
    }
  ],
  "critical_evidence_ids": ["E1", "E3"]
}
```

但仅凭 Teacher 的文字说明，不能证明：

- `E1` 和 `E3` 真的决定了主要判断；
- `E2` 真的只是辅助信息；
- critical 标签不是 Teacher 随意标的；
- evidence 粒度是否足够细。

Mask 消融就是做一个反事实测试：

> 如果把某一条证据从病例中删掉，模型的判断是否发生了预期变化？

## 2. 实验具体怎么做

对同一个病例构造多个版本。

### 完整病例

```text
患者男性，45岁，反复胸痛2周，活动后加重，心电图提示ST段异常。
```

### Mask critical evidence

例如删除：

```text
心电图提示ST段异常
```

得到：

```text
患者男性，45岁，反复胸痛2周，活动后加重，[MASK]。
```

或者直接删除该 span：

```text
患者男性，45岁，反复胸痛2周，活动后加重。
```

### Mask supporting evidence

删除：

```text
活动后加重
```

得到：

```text
患者男性，45岁，反复胸痛2周，心电图提示ST段异常。
```

### Random mask 对照

随机删除一段长度接近 evidence 的文本，例如删除：

```text
男性，45岁
```

这个对照用于排除“只要删文本，模型就会变化”的可能性。

最终至少比较：

```text
完整病例
vs
Mask critical
vs
Mask supporting
vs
Mask random
```

## 3. 预期现象是什么

如果 Teacher 的标注正确，应该满足：

\[
\Delta_{critical}
>
\Delta_{supporting}
\approx
\Delta_{random}
\]

其中：

\[
\Delta(e)
=
D\big(f(x), f(x^{-e})\big)
\]

含义是：

- \(x\)：完整病例；
- \(x^{-e}\)：删除证据 \(e\) 后的病例；
- \(f(\cdot)\)：模型输出；
- \(D\)：两个输出之间的变化程度。

理想情况是：

### 完整病例

```text
根据胸痛活动后加重以及心电图异常，需要优先排查心肌缺血或冠心病，但尚不能仅凭当前信息确诊。
```

### 删除 critical 证据后

```text
目前仅有反复胸痛的描述，尚不能判断是否为心肌缺血，需要进一步了解疼痛特点并完善心电图、心肌损伤标志物等检查。
```

模型应该：

- 降低诊断确定性；
- 扩大鉴别范围；
- 提到缺失信息；
- 可能将 `sufficient` 降为 `partial` 或 `insufficient`；
- 不再维持原来过于具体的结论。

### 删除 supporting 证据后

```text
仍然可以判断存在需要进一步评估的胸痛，主要诊断方向基本不变。
```

模型输出可以略有变化，但主要结论不应改变。

## 4. 这个实验真正验证什么

它主要验证四件事。

### 4.1 验证 critical 标签是否有因果有效性

如果一条证据被标为 critical，删除后模型应该出现明显变化。

例如：

```text
删除“心电图提示ST段异常”
→ 从“优先排查心肌缺血”变成“仅能进行一般性胸痛鉴别”
```

这说明该证据确实影响了主要判断。

如果删除后模型几乎不变：

```text
完整：考虑心肌缺血，需要心内科评估
Mask 后：考虑心肌缺血，需要心内科评估
```

说明可能存在：

- critical 标签标错；
- 其他证据已经完全冗余；
- 模型没有真正使用这条证据；
- 模型只是套用了固定回答模板。

### 4.2 验证 supporting 标签是否足够保守

supporting 的设计目标不是“完全没用”，而是：

> 有帮助，但不应该独立决定主要结论。

例如删除“活动后加重”后，模型可能从：

```text
需要优先排查心肌缺血
```

变成：

```text
需要进一步评估胸痛原因
```

这是合理的轻微变化。

但如果删除一条 supporting 后，模型从：

```text
考虑冠心病
```

变成：

```text
完全无法进行任何判断
```

就说明这条证据可能其实是 critical，或者模型过度依赖该信息。

### 4.3 验证 atomic evidence 粒度是否合理

这是 V2.2 特别重要的目标。

假设一条 evidence 是：

```text
感冒好了以后，一直干咳，有两个月了，但是无其他症状
```

它实际上包含：

- 感冒后出现；
- 持续干咳；
- 持续两个月；
- 无其他症状。

如果整个 span 一起 Mask，模型变化了，但我们不知道到底是哪一个事实起作用。

因此 V2.2 希望拆成：

```text
E1：感冒好了以后
E2：一直干咳，有两个月了
E3：但是无其他症状
```

分别 Mask：

```text
Mask E1
Mask E2
Mask E3
```

这样才能知道：

- 是“咳嗽”重要；
- 还是“持续两个月”重要；
- 还是“无其他症状”改变了判断；
- 哪些信息只是背景。

这直接关系到后续反事实训练的解释性。

### 4.4 验证模型是否真的学会证据依赖

模型可能在完整病例上表现很好，但依赖的不是证据，而是：

- 疾病关键词；
- 固定模板；
- 原始问题中的疾病名称；
- 训练集记忆；
- 常见医学回答模式。

Mask 消融可以判断模型是否具备真正的证据敏感性。

如果模型只要看到“胸痛”就始终回答冠心病，而删掉 ECG、活动相关性等关键证据后仍然保持同样结论，那么它并没有学会根据证据调整判断。

## 5. 应该比较哪些模型

建议至少比较三个模型：

| 模型 | 用途 |
|---|---|
| Base/Instruct | 判断基础模型本身的证据敏感性 |
| Full-SFT 模型 | 判断普通医疗 SFT 带来的影响 |
| Full-SFT + Evidence-SFT | 判断 Evidence-SFT 是否增强了证据依赖 |

对每个模型都输入相同的：

- 完整病例；
- critical mask；
- supporting mask；
- random mask。

这样可以区分：

```text
模型本来就会根据证据变化
```

和：

```text
Evidence-SFT 让模型真正学会了证据依赖
```

最重要的比较是：

\[
\Delta_{critical}^{EvidenceSFT}
-
\Delta_{critical}^{FullSFT}
\]

如果 Evidence-SFT 有效，应该看到 Evidence-SFT 模型在 Mask critical 后：

- 结论变化更明显；
- 确定性下降更合理；
- missing information 增加；
- 不再强行维持原结论。

同时 Mask supporting 后不应出现过度变化。

## 6. 评估指标怎么设计

不能只用字符串完全匹配，因为 Mask 后合理答案本来就不应和完整答案完全一样。

### 6.1 输出变化率

判断 Mask 后是否发生实质变化：

```text
complete_output ≠ masked_output
```

但这只是粗指标，不能单独使用。

### 6.2 语义变化距离

比较完整输出和 Mask 输出之间的语义差异：

\[
D_{semantic}
=
1-\cos(h(x),h(x^{-e}))
\]

可以使用最终答案或结构化字段的 embedding。

### 6.3 关键结论变化率

人工或规则判断以下内容是否发生变化：

- 主要诊断方向；
- 治疗建议；
- 风险等级；
- sufficiency；
- 回答确定性。

### 6.4 Sufficiency 降级率

对 critical evidence 进行 Mask 后，模型是否合理降低证据充分性：

```text
sufficient → partial
partial → insufficient
```

例如：

```text
完整病例：sufficient
Mask critical：partial
```

这是很有价值的指标。

但不能要求所有样本都必须降级，因为有些病例存在证据冗余，删除一条 critical 后仍然可能足够判断。

### 6.5 Critical Sensitivity

定义：

\[
S_{critical}
=
P\big(
D(f(x), f(x^{-e_{critical}}))>\tau
\big)
\]

表示被标成 critical 的证据中，有多少条确实引起了明显输出变化。

### 6.6 Supporting Stability

定义：

\[
S_{supporting}
=
P\big(
D(f(x), f(x^{-e_{supporting}}))\leq\tau
\big)
\]

表示 supporting 被删除后，模型是否保持主要结论稳定。

理想状态是：

- Critical Sensitivity 高；
- Supporting Stability 高。

## 7. 一个完整例子

病例：

```text
患者女性，持续咳嗽两个月，夜间明显，无发热。胸部CT提示肺部结节。请问可能是什么原因？
```

Teacher 标注：

```json
[
  {
    "id": "E1",
    "span": "持续咳嗽两个月",
    "importance": "critical"
  },
  {
    "id": "E2",
    "span": "夜间明显",
    "importance": "supporting"
  },
  {
    "id": "E3",
    "span": "无发热",
    "importance": "supporting"
  },
  {
    "id": "E4",
    "span": "胸部CT提示肺部结节",
    "importance": "critical"
  }
]
```

完整病例输出：

```text
需要结合肺部结节的大小、形态、位置及既往吸烟史等判断。持续咳嗽与肺部结节需要进一步由呼吸科评估，但目前不能仅凭这些信息确定病因。
```

Mask E4：

```text
患者女性，持续咳嗽两个月，夜间明显，无发热。
```

合理输出：

```text
目前只能根据持续咳嗽进行一般性鉴别，缺少影像学结果，不能判断是否存在肺部结节及其性质。
```

这是明显变化，说明 E4 确实关键。

Mask E2：

```text
患者女性，持续咳嗽两个月，无发热。胸部CT提示肺部结节。
```

合理输出：

```text
主要判断方向仍是结合肺部结节和咳嗽进一步评估，只是无法判断夜间加重这一特征的意义。
```

变化较小，符合 supporting。

Mask E3：

```text
患者女性，持续咳嗽两个月，夜间明显。胸部CT提示肺部结节。
```

主要结论也应基本保持。

## 8. 这个实验和普通消融实验有什么区别

普通模型消融通常问：

> 去掉某个输入字段后，整体性能下降多少？

例如去掉：

- evidence 字段；
- query_intent；
- missing_information。

而关键证据 Mask 消融问的是：

> 删除病例中的某个具体临床事实后，模型的判断是否按照医学逻辑发生变化？

它更接近因果干预：

```text
不是删除一个模型模块
而是删除一个病例事实
```

因此它可以验证：

- 标签是否正确；
- 模型是否依赖正确证据；
- 不确定性是否能够随信息缺失而调整；
- 反事实训练是否有效。

## 9. 这个实验不能证明什么

需要注意，Mask 消融不能单独证明：

- 模型真的理解了医学因果关系；
- 删除某证据后的新答案一定医学正确；
- Teacher 标注一定是正确的；
- 模型在现实临床中安全；
- 所有 critical 证据都必须导致巨大输出变化。

因为病例中的证据可能存在冗余。

例如：

```text
医生诊断为肺炎
胸片提示肺部感染
使用抗生素后好转
```

这三条证据可能都支持同一结论。删除一条后，模型仍然可以判断肺炎，并不一定说明该证据不是 critical，而可能说明它和其他证据高度重叠。

因此需要把结果分成：

- 单证据独立性；
- 证据冗余；
- 证据交互作用。

## 10. 对当前项目最实际的实验版本

建议第一版不要做得过于复杂：

### 数据范围

从 test 集抽取约 100–200 条：

- 有至少 1 条 critical；
- 有至少 1 条 supporting；
- evidence span 不重叠；
- 排除明显结构异常样本。

### 每条病例构造

```text
1 个完整版本
1 个 critical mask 版本
1 个 supporting mask 版本
1 个 random mask 版本
```

### 模型

先比较：

```text
Full-SFT
Full-SFT + Evidence-SFT
```

### 重点指标

- critical mask 后主要结论变化率；
- supporting mask 后主要结论保持率；
- sufficiency 降级率；
- critical/supporting 的变化距离比；
- masked 后是否指出缺失信息；
- 是否出现新的无依据诊断。

最终要验证的核心关系是：

```text
删除 critical：
模型应显著降低确定性

删除 supporting：
模型应基本保持主要方向

删除 random：
模型不应出现系统性大幅变化
```

如果这个关系成立，就说明 `critical_evidence_ids` 不只是格式字段，而是具有实际反事实价值的训练信号。