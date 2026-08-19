# Evidence-Med Strict Answer-level DPO

## 1. 本轮实验结论

本目录实现不经过 Evidence-Mask 的主实验：

```text
Qwen2.5-7B-Instruct
  -> Direct Evidence-SFT
  -> D0（合并后的固定起点）
  -> Strict Answer-level DPO
  -> D1
```

本轮只回答：

> DPO 是否能在保持 Evidence 结构能力和医学知识的前提下，改善最终回答的完整性、证据忠实性、安全性和校准？

本轮不能回答 Mask 是否有效，也不能把 D1 的结果写成 Evidence + Mask + DPO。

默认 D0 选择：

```text
base:    Qwen/Qwen2.5-7B-Instruct
adapter: outputs/evidence-sft-frombase
```

`outputs/evidence-sft` 的 Two-stage 路线保留为候选或后续消融，不作为本轮默认起点。

## 2. Strict Answer-level 定义

每个 DPO pair 必须共享以下字段：

- `task_type`
- `query_intent`
- `evidence_sufficiency`
- `evidence`
- `critical_evidence_ids`

只允许以下字段不同：

- `missing_information`
- `clinical_reasoning`
- `final_answer`

Strict Answer-level 约束的是最终进入 Judge 和训练的 pair：pair 两端必须共享上述冻结字段。自然候选无需自行复现 validated target 的 task type、sufficiency 或 evidence 原子边界；只要三个回答字段可解析且投影后通过 Schema/硬失败审计，就统一投影到 target 的结构骨架。

候选的原始结构漂移、Schema 错误、hard failure 和对齐指标仍完整保留用于诊断，但不替代对自然回答的医学质量判断。无法解析 JSON，或三个回答字段缺失、类型错误、投影后仍非法的候选才直接过滤。医学正确性、过度断言、病例外治疗建议等由后续 MiMo Judge 判断。

受控负样本也只制造回答级缺陷：

- 证据不足时过度确定；
- 证据充分时无必要拒答；
- 只给笼统建议、不直接回答问题。

## 3. 数据流

```text
03_validated_full.jsonl
  -> 00_sources.jsonl
  -> 01_sft_candidates.jsonl
  -> 02_pair_candidates.jsonl
  -> 03_judgments.jsonl
  -> train/train.jsonl + validation/validation.jsonl + audit/*.jsonl
  -> D0 merged checkpoint
  -> D1 Answer-level DPO adapter
  -> held-out Evidence/C-Eval + D0-vs-D1 swap A/B
```

所有新数据使用独立目录：

```text
data/dpo/answer_v1/
```

不要与旧的 `data/dpo/v1` 混用。

## 4. 脚本

### `dpo_common.py`

提供 Evidence Schema 审计、Answer-level 结构对齐与投影、确定性 ID、评分 Schema、受控回答级负样本和 JSONL 工具。

其中 `make_controlled_negative()` 是受控负样本的唯一构造入口（见
`process_dpo/dpo_common.py:454`）。它不会调用模型，也不会改写病例或 Evidence
字段，而是先复制并清理 `T`，再只修改 `missing_information`、
`clinical_reasoning`、`final_answer` 三个 Answer-level 字段。缺陷类型由
`source_id + seed` 确定性选择，因此同一来源可以复现相同结果：

- `partial/insufficient`：在 `overconfident_answer`（删除缺失信息并声称资料足以
  得出确定结论）和 `generic_non_answer`（只给笼统建议）之间选择；
- `sufficient`：在 `unnecessary_uncertainty`（明明资料充分却拒绝判断）和
  `generic_non_answer` 之间选择。

返回对象会保留 `candidate_id`、`origin`、`intended_error`、`text` 和
`parsed_output`，便于后续统计实际覆盖的缺陷类型。

### `select_dpo_sources.py`

从 Evidence-SFT v2.2 的 validated full 数据选择来源：

- 只允许原 train 和 validation；
- 永久锁定 test；
- 默认使用分级 warning 策略，而不是三个标签一票否决；
- 按 sufficiency 目标分布抽样，并尽量平衡 task type。

warning 策略分为：

- `tiered`（默认）：教师修改 `task_type` 只作为低风险审计信息；强断言结合
  evidence span、否定和限定语重新判断；纯分点序号忽略，新增时长或无单位数字
  视为中风险，新增剂量、浓度、生命体征阈值等医学参数视为高风险。高风险排除，
  每个 split 的中风险占比默认不超过 15%；
- `strict`：复现旧实验，只要命中上述任一历史 warning 就排除；
- `all`：通过 target Schema/grounding 硬审计后不再按 warning 排除，仅用于审计或消融，不建议直接作为主实验默认配置。

输出样本保留 `target_warnings`，并新增 `warning_risk`、具体原因、涉及的evidence id 和数字；stats 同时报告评估前、eligible 和最终 selected 的风险分布。

### `generate_sft_candidates.py`

从 Direct Evidence-SFT 为每个来源病例自适应生成自然回答。默认先生成两个主候选：

- `greedy`：不采样，提供稳定基线；
- `sample_t0.8`：中温采样，提供与 greedy 有区分度的自然回答。

生成后立即提取 `missing_information`、`clinical_reasoning` 和 `final_answer`，投影到 validated target 的冻结结构并审计。如果某个病例不足两个投影可用且 Answer-level 内容不同的候选，才额外生成一次 `sample_t0.6`。默认每个病例生成两个回答，最多三个；`sample_t1` 不再是默认档位，但仍可通过 CLI 自定义温度。

这里的多个候选是自然回答池，不是一个 DPO 样本包含多个回答。它们分别用于：

- 至少一个 `M` 时构造 `target_vs_model`；
- 至少两个不同 `M` 时构造 `model_vs_model`；
- 主候选解析失败或内容重复时由 fallback 恢复覆盖。

每个候选保留生成参数和原始审计；每条来源新增 `generation_policy`，记录主候选、fallback 配置、最少不同候选数和是否实际触发 fallback。生成结束后 stats 报告各 profile 数量、投影可用候选数、拥有至少一个/两个候选的来源数、fallback 率、耗时和吞吐。

当前 10 条 smoke 表明 `greedy + sample_t0.8` 已覆盖 10/10 来源，因此相较四回答预计接近减半生成时间。`max-new-tokens` 仍保持 1152；正常 EOS 的序列不会因为上限较高而额外生成，贸然降低只会增加长回答截断风险。

### `build_dpo_pairs.py`

该脚本先定义三种回答来源：

- `T`（validated target）：原 Evidence-SFT 教师回答经过验证后得到的 target；
- `M`（model candidate）：当前 Direct Evidence-SFT 模型生成的自然回答；
- `N`（controlled negative）：脚本根据 `T` 确定性构造的回答级受控负样本。

`M` 通过以下 Answer-level 门禁进入 pair：

- 原始输出能够解析为 JSON 对象；
- `missing_information` 是字符串列表，`clinical_reasoning` 和 `final_answer` 是非空字符串；
- 三个回答字段投影到 `T` 的共享结构后，通过完整 Schema 和硬失败审计；
- 与 pair 的另一端至少有一个 Answer-level 字段不同。

原始 task type、sufficiency、evidence span/importance 和 critical evidence 是否与 `T` 对齐，只进入 `raw_audit_summary` 和 stats，不再作为准入条件。这样保留严格的最终 pair 隔离，同时避免将 evidence 原子切分差异误判成自然回答不可用。

投影后，pair 两端共享 `task_type`、`query_intent`、
`evidence_sufficiency`、`evidence` 和 `critical_evidence_ids`，只允许
`missing_information`、`clinical_reasoning`、`final_answer` 不同。

每个 DPO pair 始终只有两个回答。默认每个病例最多构造下面三种二元比较：

| pair type | 两端 | 默认选择方式 | 主要作用 |
| --- | --- | --- | --- |
| `target_vs_model` | `T vs M` | `T` 与自动预排序较低的一个合法 `M` 比较 | 检验 validated target 是否确实优于当前模型回答，同时让 Judge 有权推翻教师 target |
| `model_vs_model` | `M_best vs M_worst` | 从同一病例的合法自然候选中选择自动预排序最高和最低的两个 | 学习当前 policy 自身输出之间的细粒度质量差异，减少所有偏好都锚定教师文风 |
| `controlled_negative` | `T vs N` | `T` 与一个只修改回答字段的受控缺陷回答比较 | 覆盖过度确定、无必要拒答和笼统非回答等已知失败模式，并检查 Judge 是否能识别明显错误 |

例如，一个病例拥有 `T`、两个主候选 `M0/M1`（必要时还有 fallback `M2`）和一个 `N`，并不意味着把这些回答放进同一个训练样本。脚本可能输出三个独立记录：

```text
pair_1 = (T, M_worst)       # target_vs_model
pair_2 = (M_best, M_worst)  # model_vs_model
pair_3 = (T, N)             # controlled_negative
```

每个记录都只有两个回答，独立调用一次 Judge，也可能分别得到不同结果。一个 pair
的结果不会自动决定另外两个 pair 的结果。

`M_best` 和 `M_worst` 不是最终的 chosen/rejected。它们只是脚本为了挑选有
对比度的自然候选所做的自动预排序，排序依据包括自动审计结果、warning/review
数量以及 missing information 是否与 target 对齐。如果两个候选排序完全相同，
使用 `source_id + candidate_id + seed` 的确定性哈希打破平局，避免候选 ID 字典序
系统性偏向 `greedy`；stats 会报告 `target_vs_model_candidate_profiles`。该预排序不
声称判断了完整的医学质量，最终偏好必须由 Judge 决定。

`N` 根据 evidence sufficiency 制造不同缺陷：

- partial/insufficient：删除缺失信息并给出过度确定结论，或生成笼统非回答；
- sufficient：无必要地拒绝形成判断，或生成笼统非回答；
- 所有 `N` 都保持 Evidence 结构不变，不能通过伪造 evidence 制造容易区分的负样本。

具体调用位于 `process_dpo/build_dpo_pairs.py:230`：每个来源按
`--controlled-per-source`（默认 1）调用 `make_controlled_negative()`，然后在
`process_dpo/build_dpo_pairs.py:234` 重新执行完整 `audit_response()`，并在
`process_dpo/build_dpo_pairs.py:241` 通过 `is_isolated_answer_pair()` 检查只发生
Answer-level 差异。审计失败的 `N` 不会写入 pair 文件。成功记录的
`candidate_B`（或经 A/B 位置随机化后的另一侧）会带有 `intended_error`，但这个
字段不会暴露给 Judge；Judge 只看到病例、共享结构和两个回答字段。

`_make_pair` 会按照 seed 将两端确定性随机放入 Candidate A/B，因此
`validated_target`、`sft_model` 或 `controlled_negative` 都不会固定出现在
A 或 B。pair 保存 `shared_structure` 供 Judge 理解共同前提，但不会向 Judge
提供 target reference anchor，也不会暴露候选来源。

### `judge_dpo_pairs.py`

对每个二元 pair 做一次来源盲化 Judge。构造 pair 时还不存在
`chosen`/`rejected`，只有 Candidate A/B；Judge 作出明确裁决后才建立：

```text
若 decision = A_better：chosen = A，rejected = B
若 decision = B_better：chosen = B，rejected = A
若 decision = tie/both_bad/unjudgeable：不能导出为 DPO pair
```

Judge 只看到：

- 病例原文；
- A/B 共用结构；
- 两边的 answer-level 字段。

Judge 不知道哪边是教师 target、模型自然回答或受控负样本。它需要输出
medical correctness、evidence faithfulness、answer completeness、calibration、
missing information、actionability/safety 和 expression 的分项得分，同时报告
硬错误、胜负、理由与 confidence。

原始训练 Judge 先对全部 pair 做一次盲评；随机 A/B 只负责构造盲化位置，不能被
视为已经抵消了 Judge 的位置偏差。自然 pair 在导出前还必须由
`judge_dpo_swaps.py` 使用同一 Judge、同一 rubric 和 temperature=0 交换 A/B
复审，并由 `reconcile_dpo_judgments.py` 映射回底层候选。controlled negative
不重复调用，但在最终数据中严格限制为 10%。三个阶段的结果写入不同文件，原始
`03_judgments.jsonl` 永不原地改写。

Schema 校验保持严格，不会把缺字段或语义不完整的返回静默转换成有效裁决。
`A_better`、`B_better` 和 `both_bad` 必须给出至少一个
`decisive_dimensions`；`tie` 与 `unjudgeable` 可以为空，因为这两种裁决可能本来就
不存在决定性差异。`03_judgments_failed.jsonl` 保存当前仍未解决的失败 pair，每条
记录包含：

- `attempt_count`：本轮实际尝试次数；
- `attempt_failures`：每次尝试的错误详情；
- Schema 失败时的 `raw_response`、`parsed_judgment`、`finish_reason` 和
  `validation_errors`；
- API 异常时的异常类型与信息。

同一 pair 重试失败时，只保留最新失败记录；重试成功后会从 failed 文件移除。
`--retry-failed-only` 还会先用当前 Schema 重新校验 failed 中已经保存的
`parsed_judgment`：若返回现在已经合法，则直接恢复到 judgment 文件，不再次调用
Judge。恢复记录带有 `recovered_from_failed=true` 和 `recovery` 审计信息，原始
`decision`、scores、reason 和 confidence 完全保留。因此 `03_judgments.jsonl`
表示已完成状态，`03_judgments_failed.jsonl` 表示尚待处理状态。

### `judge_dpo_swaps.py`

该脚本只复审原始 judgment 中已满足明确胜负、confidence≥0.85、分差≥2、
chosen 无 Judge 硬错误、两端自动审计与 Answer-level 隔离通过的自然 pair。
`controlled_negative` 不进入 swap。请求只交换 `candidate_A` 和
`candidate_B`，病例、共享结构、pair_id、source_id 与 split 保持不变，也不会
泄露候选来源。

`--limit-per-type 100` 会按稳定哈希分别抽取 100 条
`target_vs_model` 和 100 条 `model_vs_model`，用于 200 条 smoke；去掉该参数
后处理全部合格自然 pair。脚本支持断点续跑、failed 本地恢复、逐字段 Schema
诊断和连续失败熔断。正式全量运行可直接复用 smoke 输出，已成功的 200 条不会
重复请求。

### `reconcile_dpo_judgments.py`

该脚本把原始与 swap judgment 映射到同一底层候选：

- Tier A：两次明确胜负均指向同一候选，两次 confidence≥0.85、分差≥2，winner
  在两个方向均无 Judge 硬错误；
- Tier B：仅在 Tier A 容量不足时作为候选；两次映射后的总分都指向同一
  winner、各方向分差≥1、平均分差≥2、两次 confidence≥0.90，winner 无硬错误；
- tie、both_bad、unjudgeable、方向冲突或缺少 swap 的自然 pair 不进入
  reconciled 文件。

Tier B 的 20% 上限在导出阶段按最终自然 pair 计算；reconcile 文件保留所有满足
Tier B 定义的候选，便于审计容量。

### `export_dpo_dataset.py`

该脚本才真正把 Candidate A/B 转换为训练需要的 `chosen`/`rejected`。导出前
硬性要求：

- judgment 状态为 ok；
- decision 为明确胜负；
- confidence ≥ 0.85；
- chosen 至少高 2 分；
- chosen 无 Judge 硬错误；
- chosen/rejected 都通过自动 Schema 与 grounding；
- chosen/rejected 结构完全相同且至少一个回答字段不同；
- pair_id 与 judgment 不重复；
- train/validation source_id 不重叠。
- 自然 pair 带有通过的 Tier A 或 Tier B 双向一致性记录；
- controlled negative 带有单向 controlled reconciliation 标记。

通过质量门槛后，按 pair type 严格配额选择。初始规划配比为：

- `target_vs_model`：55%；
- `model_vs_model`：35%；
- `controlled_negative`：10%。

配比可通过 CLI 显式配置，但默认禁止跨类型补位；这意味着调整后的配比仍是固定
配额，而不是用某一类型临时填补另一类型缺口。controlled 永不超过 10%，Tier B 不超过最终自然 pair 的
20%，每个来源最多贡献两条。在每一种自然 pair 内，原始 A 胜比例必须处于
45%–55%；脚本只做分层下采样，不通过翻转 chosen/rejected 修正分布。

导出器从 train≤3000、validation≤200 向下搜索满足所有约束的最大规模；数量
下限由 `--min-train/--min-validation` 明确指定。若未达到下限，则只更新 `04_export.stats.json` 为
`export_ready=false`，不覆盖现有正式 train/validation/audit 文件。
`--allow-type-backfill` 和 `--no-require-swap-consistency` 仅用于兼容诊断，
不得用于最终训练数据。

### `prepare_dpo_start.py`

把 Direct Evidence-SFT LoRA 合并到 Qwen base，生成固定 D0 full checkpoint，并写入 `dpo_start_manifest.json`。

### `training/dpo_training.py`

在合并 D0 上创建新的 DPO LoRA：

- policy：D0 + 新 DPO LoRA；
- reference：禁用新 DPO LoRA 后的原始 D0；
- 默认 beta 0.1；
- 默认学习率 5e-6；
- 默认 1 epoch；
- 使用 tokenizer token 数而非字符数做长度过滤；
- 拒绝无 D0 manifest 的起点。

### `judge_dpo_evaluation.py`

在 held-out test 上比较 D0 与 D1 的回答质量。每条病例评两次，第二次交换 A/B；只有两次映射到同一模型结论时才计入一致结果。

## 5. 环境变量

Judge 沿用项目根目录 `.teacher_env`：

```bash
JUDGE_MODEL=你的教师模型名
JUDGE_BASE_URL=https://你的OpenAI兼容接口/v1
JUDGE_API_KEY=你的密钥
```

也兼容 `TEACHER_*` 和标准 `OPENAI_*`。不要把密钥写入数据、日志或文档。

## 6. 完整运行流程

所有命令从项目根目录执行。请使用实际安装了 torch、transformers、peft、trl、datasets 和 openai 的项目 Python 环境。

### 步骤 0：运行测试

```bash
python3 -m unittest discover -s tests -v
```

### 步骤 1：合并 Direct Evidence-SFT，冻结 D0

```bash
CUDA_VISIBLE_DEVICES=0 python3 process_dpo/prepare_dpo_start.py \
  --base-model ./Qwen/Qwen2.5-7B-Instruct \
  --sft-adapter ./outputs/evidence-sft-frombase \
  --output ./outputs/dpo-start-direct-merged \
  --torch-dtype bfloat16 \
  --device-map cuda:0
```

输出目录必须为空或不存在。不要用原始 Qwen base 直接训练 DPO，也不要把 adapter-only 目录直接传给训练器。

### 步骤 2：选择 3000/200 个来源病例

```bash
python3 process_dpo/select_dpo_sources.py \
  --input data/evidence_sft/validated_v2_2/03_validated_full.jsonl \
  --output data/dpo/answer_v1/00_sources.jsonl \
  --stats-output data/dpo/answer_v1/00_sources.stats.json \
  --train-limit 3000 \
  --validation-limit 200 \
  --warning-policy tiered \
  --max-medium-risk-fraction 0.15 \
  --seed 42
```

当前数据可产出 3000 train 和 200 validation；`conflicting` 只有 1 条，不能用本轮结果声称该类别已被充分优化。

如需复现旧的一票否决或做放开全部 warning 的消融，只改
`--warning-policy strict` 或 `--warning-policy all`，不要修改源数据。正式生成候选前，
检查 stats 中的 `selected_warning_risk`、`assessed_warning_reasons` 和
`rejected.warning_risk:high`。

### 步骤 3：生成 Direct Evidence-SFT 候选

先用新策略做 50 条 smoke。默认主候选是 `greedy + t0.8`，不足两个不同投影候选时只补 `t0.6`：

```bash
CUDA_VISIBLE_DEVICES=0 python3 process_dpo/generate_sft_candidates.py \
  --sources data/dpo/answer_v1/00_sources.jsonl \
  --output data/dpo/answer_v1/01_sft_candidates.smoke50.jsonl \
  --stats-output data/dpo/answer_v1/01_sft_candidates.smoke50.stats.json \
  --base-model ./Qwen/Qwen2.5-7B-Instruct \
  --adapter ./outputs/evidence-sft-frombase \
  --batch-size 2 \
  --max-new-tokens 1152 \
  --temperatures 0.8 \
  --fallback-temperature 0.6 \
  --min-distinct-candidates 2 \
  --top-p 0.9 \
  --seed 42 \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --limit 50
```

正式生成：

```bash
CUDA_VISIBLE_DEVICES=0 python3 process_dpo/generate_sft_candidates.py \
  --sources data/dpo/answer_v1/00_sources.jsonl \
  --output data/dpo/answer_v1/01_sft_candidates.jsonl \
  --stats-output data/dpo/answer_v1/01_sft_candidates.stats.json \
  --base-model ./Qwen/Qwen2.5-7B-Instruct \
  --adapter ./outputs/evidence-sft-frombase \
  --batch-size 2 \
  --max-new-tokens 1152 \
  --temperatures 0.8 \
  --fallback-temperature 0.6 \
  --min-distinct-candidates 2 \
  --top-p 0.9 \
  --seed 42 \
  --device-map cuda:0 \
  --torch-dtype bfloat16 \
  --resume
```

`--resume` 只跳过已经完整写入的 source，不会重复追加。旧版四候选 JSONL 仍可交给新版 pair builder；不要把旧 smoke 文件直接作为正式输出续跑，因为其 profile 配置不同。

50 条 smoke 的默认验收线：至少一个投影可用自然候选的来源覆盖率不低于 95%，至少两个不同候选的覆盖率不低于 90%。fallback 率应单独查看；若高于 10%，先分析解析失败和重复原因，不直接恢复四候选。

### 步骤 4：构造严格 Answer-level pair

```bash
python3 process_dpo/build_dpo_pairs.py \
  --sources data/dpo/answer_v1/00_sources.jsonl \
  --candidates data/dpo/answer_v1/01_sft_candidates.jsonl \
  --output data/dpo/answer_v1/02_pair_candidates.jsonl \
  --stats-output data/dpo/answer_v1/02_pair_candidates.stats.json \
  --target-model-per-source 1 \
  --model-model-per-source 1 \
  --controlled-per-source 1 \
  --seed 42
```

先检查 stats 中的 `natural_candidate_alignment.answer_projection_rate`、`sources_with_at_least_one_projectable_candidate` 和 `sources_with_at_least_two_distinct_projectable_candidates`。`structure_compatibility_rate` 只诊断原始结构漂移，低值不再阻止候选进入 Judge。

对 smoke 文件还应查看顶层 `candidate_sources_without_target_vs_model` 和 `candidate_sources_without_model_vs_model`；它们只以实际生成候选的来源为分母，不会把尚未生成的正式来源误报成失败。

### 步骤 5：预览 10 条训练 Judge 请求

```bash
python3 process_dpo/judge_dpo_pairs.py \
  --pairs data/dpo/answer_v1/02_pair_candidates.jsonl \
  --preview-output data/dpo/answer_v1/03_judge_requests.preview.jsonl \
  --limit 10 \
  --dry-run
```

人工确认：

- 请求中没有 `validated_target`、`sft_model`、`controlled_negative`；
- 没有 `reference_anchor`；
- A/B 的 shared structure 相同；
- A/B 只展示 missing information、clinical reasoning 和 final answer。

### 步骤 6：执行训练数据 Judge

```bash
python3 process_dpo/judge_dpo_pairs.py \
  --pairs data/dpo/answer_v1/02_pair_candidates.jsonl \
  --output data/dpo/answer_v1/03_judgments.jsonl \
  --failed-output data/dpo/answer_v1/03_judgments_failed.jsonl \
  --model "$JUDGE_MODEL" \
  --workers 8 \
  --temperature 0 \
  --max-tokens 2048
```

脚本按 pair_id 断点续跑：`03_judgments.jsonl` 中已有 `status=ok` 的 pair
不会重复请求。普通模式下，`--limit` 在跳过成功项后应用，所以不能用普通
`--limit 50` 来表达“只重试原 smoke 的 12 个失败项”。

只重试 failed 文件中当前未解决的 pair：

```bash
python3 process_dpo/judge_dpo_pairs.py \
  --pairs data/dpo/answer_v1/02_pair_candidates.jsonl \
  --output data/dpo/answer_v1/03_judgments.jsonl \
  --failed-output data/dpo/answer_v1/03_judgments_failed.jsonl \
  --model "$JUDGE_MODEL" \
  --workers 8 \
  --temperature 0 \
  --max-tokens 2048 \
  --max-retries 1 \
  --retry-failed-only
```

`--retry-failed-only` 会先本地恢复当前 Schema 已经接受的保存裁决，再按 failed
pair_id 筛选真正需要请求的项，最后应用 `--limit`；因此它不会夹带后续尚未 Judge
的新 pair。加 `--dry-run` 时不会改写数据，输出中的 `locally_recoverable` 是可本地
恢复数量，preview 只包含仍需 API 的请求。

第一次诊断性重跑建议使用 `--max-retries 1`：温度为 0 时重复请求通常不会修复
确定性的格式偏差，单次调用足以收集原始返回。连续失败保护默认阈值为 10，可通过
`--max-consecutive-failures` 调整；仅在明确需要完整收集一个有界诊断批次时设为 0
关闭。重跑后先汇总 `validation_errors`，再决定是否对某一种明确、无损的格式偏差
做规范化；不要直接放宽整个 Judge Schema。

### 步骤 6B：200 条 swap Judge smoke

先预览稳定哈希抽取的两类各 100 条：

```bash
python3 process_dpo/judge_dpo_swaps.py \
  --limit-per-type 100 \
  --preview-output data/dpo/answer_v1/03_swap_requests.smoke.preview.jsonl \
  --stats-output data/dpo/answer_v1/03_swap_judgments.smoke.stats.json \
  --dry-run
```

确认请求只交换回答 A/B 且不泄露来源后执行：

```bash
python3 process_dpo/judge_dpo_swaps.py \
  --limit-per-type 100 \
  --output data/dpo/answer_v1/03_swap_judgments.jsonl \
  --failed-output data/dpo/answer_v1/03_swap_judgments_failed.jsonl \
  --stats-output data/dpo/answer_v1/03_swap_judgments.smoke.stats.json \
  --model "$JUDGE_MODEL" \
  --workers 8 \
  --temperature 0 \
  --max-retries 3
```

smoke 必须满足 Schema coverage≥98%，并在重试后 failed=0。达标后去掉
`--limit-per-type` 完成全量；同一路径会按 pair_id 续跑，不重复前 200 条。

### 步骤 6C：映射双向裁决

```bash
python3 process_dpo/reconcile_dpo_judgments.py \
  --pairs data/dpo/answer_v1/02_pair_candidates.jsonl \
  --judgments data/dpo/answer_v1/03_judgments.jsonl \
  --swap-judgments data/dpo/answer_v1/03_swap_judgments.jsonl \
  --output data/dpo/answer_v1/03_reconciled_judgments.jsonl \
  --stats-output data/dpo/answer_v1/03_reconciled_judgments.stats.json
```

先查看两类的 swap coverage、Tier A/Tier B 数量、方向冲突原因和原始胜者位置。
全量 swap 未覆盖时，缺失自然 pair 会明确记为 `swap:missing`，不会静默进入导出。

### 步骤 7：严格容量搜索并导出 DPO 数据

```bash
python3 process_dpo/export_dpo_dataset.py \
  --pairs data/dpo/answer_v1/02_pair_candidates.with_recovery.jsonl \
  --judgments data/dpo/answer_v1/03_reconciled_judgments.with_recovery.jsonl \
  --output-root data/dpo/answer_v1 \
  --train-limit 3000 \
  --validation-limit 200 \
  --min-train 1200 \
  --min-validation 100 \
  --target-vs-model-weight 0.62 \
  --model-vs-model-weight 0.28 \
  --controlled-weight 0.10 \
  --min-confidence 0.85 \
  --min-score-margin 2 \
  --max-pairs-per-source 2 \
  --recovery-status data/dpo/answer_v1/recovery/validation_exhaustion_probe.stats.json
```

`answer_v1` 的全量双向 Judge 与四轮恢复完成后，初始 55%/35%/10% 配比最多只能
形成 1,054 train + 81 validation；瓶颈是 `model_vs_model`，不是 Judge 或医学质量
失败。现有池在固定 62%/28%/10% 配比下可形成 1,243 train + 101 validation，且
仍满足所有质量、位置、Tier 和 controlled 上限。因此本版将经验性数量下限调整为
1,200/100。该调整不降低 confidence、分差、硬错误、Answer-level 隔离或双向一致性
门槛，也没有跨类型补位。

导出后检查：

- `export_ready=true`，train≥1200、validation≥100；
- 两个 split 的类型数量精确符合 62%/28%/10%，无跨类型补位；
- controlled≤10%、Tier B≤自然 pair 的 20%；
- 每个自然类型的 `original_a_win_fractions` 均在 45%–55%；
- `04_export.stats.json` 的淘汰原因；
- difference fields 分布；
- train audit 至少人工抽查 50 条；
- validation audit 建议全部检查；
- 是否出现“更长即更好”“一律更保守”或固定模板偏好；
- chosen 是否引入病例外数字、剂量、阈值或治疗建议。

正式导出后运行：

```bash
python3 process_dpo/validate_dpo_export.py \
  --root data/dpo/answer_v1 \
  --train-sample 50 \
  --min-train 1200 \
  --min-validation 100 \
  --max-per-source 2
```

脚本复核 pair/source 唯一性、split 零泄漏、Answer-level 隔离、类型比例、
controlled/Tier B 上限和原始位置比例，并写出
`audit/final_review_sample.jsonl`（稳定哈希 50 条 train + 全部 validation）与
`04_export.validation.json`。自动检查不能替代对 Judge 医学理由的人工/模型审阅。

若 `export_ready=false`，不要放宽医学、置信度、分差或位置门槛。先从尚无
Tier A/B 自然 pair 的来源中稳定抽取 225 条 train + 25 条 validation：

```bash
python3 process_dpo/select_dpo_recovery_sources.py \
  --pairs data/dpo/answer_v1/02_pair_candidates.jsonl \
  --reconciled data/dpo/answer_v1/03_reconciled_judgments.jsonl \
  --output data/dpo/answer_v1/recovery/round_01_sources.jsonl

python3 process_dpo/generate_sft_candidates.py \
  --sources data/dpo/answer_v1/recovery/round_01_sources.jsonl \
  --output data/dpo/answer_v1/recovery/round_01_candidates.jsonl \
  --base-model ./Qwen/Qwen2.5-7B-Instruct \
  --adapter ./outputs/evidence-sft-frombase \
  --temperatures 0.6 \
  --no-greedy \
  --min-distinct-candidates 1 \
  --batch-size 2 \
  --max-new-tokens 1152 \
  --top-p 0.9 \
  --device-map cuda:0 \
  --torch-dtype bfloat16

python3 process_dpo/build_dpo_recovery_pairs.py \
  --sources data/dpo/answer_v1/recovery/round_01_sources.jsonl \
  --new-candidates data/dpo/answer_v1/recovery/round_01_candidates.jsonl \
  --output data/dpo/answer_v1/recovery/round_01_pairs.jsonl \
  --round 1
```

随后对 `round_01_pairs.jsonl` 先运行 `judge_dpo_pairs.py`，再运行
`judge_dpo_swaps.py` 和 `reconcile_dpo_judgments.py`；用
`merge_dpo_jsonl.py` 分别合并基础与恢复 pairs、reconciled judgments 后重新
导出。下一轮选择来源时，把上一轮 candidates 通过 `--exclude-candidates` 排除。
每轮最多补一次 `sample_t0.6`，达到当次实验预先指定的数量下限即停止；来源
耗尽仍不达标时，将最后一次来源选择 stats 通过
`--recovery-status path/to/exhaustion.stats.json` 传给导出器；最终
`04_export.stats.json` 会标记
`final_status=preference_data_insufficient`。同时检查
`04_export.status.json` 中 `formal_files_written_this_run=false`，不得使用目录中
可能残留的旧 train/validation，也不得用更多 controlled 补量。若缺口仅来自自然
pair 类型容量不均，可在完整记录容量证据后重新设定固定类型权重和经验性数量下限；
不得降低质量门槛、突破 controlled 10% 或启用跨类型补位。

### 步骤 8：200 对训练 smoke

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/dpo_training.py \
  --model_name_or_path ./outputs/dpo-start-direct-merged \
  --tokenizer_name_or_path ./outputs/dpo-start-direct-merged \
  --train_file_dir ./data/dpo/answer_v1/train \
  --validation_file_dir ./data/dpo/answer_v1/validation \
  --output_dir ./outputs/evidence-dpo-answer-v1-smoke \
  --do_train \
  --do_eval \
  --use_peft \
  --target_modules all \
  --max_train_samples 200 \
  --max_steps 20 \
  --beta 0.1 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16
```

smoke 只验证显存、数据加载、reference 语义、loss 和保存格式，不用于报告最终效果。
设置 `--max_train_samples` 时，脚本会先用 `--seed` 对全量训练集做确定性打乱，
再截取 smoke 子集；不能直接截取导出文件开头，因为正式文件按 pair type 分块排列，
会造成 smoke 只覆盖 `model_vs_model`。运行后应对照 audit 检查子集的 pair type 与
原始 A/B 胜者位置分布。

### 步骤 9：正式 DPO

从 D0 重新开始，不要从 smoke adapter 续训：

```bash
CUDA_VISIBLE_DEVICES=0 python3 training/dpo_training.py \
  --model_name_or_path ./outputs/dpo-start-direct-merged \
  --tokenizer_name_or_path ./outputs/dpo-start-direct-merged \
  --train_file_dir ./data/dpo/answer_v1/train \
  --validation_file_dir ./data/dpo/answer_v1/validation \
  --output_dir ./outputs/evidence-dpo-answer-v1 \
  --do_train \
  --do_eval \
  --use_peft \
  --target_modules all \
  --max_steps -1 \
  --num_train_epochs 1 \
  --beta 0.1 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --bf16
```

第一轮只跑 1 epoch。不要在查看 held-out test 后反复调 beta、epoch 或学习率。

### 步骤 10：复跑 Evidence held-out test

D0：

```bash
CUDA_VISIBLE_DEVICES=0 python3 evi-sft-traing/evaluate_evidence.py \
  --base-model ./Qwen/Qwen2.5-7B-Instruct \
  --adapter ./outputs/evidence-sft-frombase \
  --tokenizer-name-or-path ./Qwen/Qwen2.5-7B-Instruct \
  --test-file ./data/evidence_sft/validated_v2_2/test.jsonl \
  --output-dir ./results/evidence_d0_direct_eval \
  --batch-size 8 \
  --max-new-tokens 1152
```

D1：

```bash
CUDA_VISIBLE_DEVICES=0 python3 evi-sft-traing/evaluate_evidence.py \
  --base-model ./outputs/dpo-start-direct-merged \
  --adapter ./outputs/evidence-dpo-answer-v1 \
  --tokenizer-name-or-path ./Qwen/Qwen2.5-7B-Instruct \
  --test-file ./data/evidence_sft/validated_v2_2/test.jsonl \
  --output-dir ./results/evidence_dpo_answer_eval \
  --batch-size 8 \
  --max-new-tokens 1152
```

注意：D1 adapter 的 base 是合并后的 D0，不是原始 Qwen。D0/D1 显式使用同一
原始 Qwen tokenizer，避免合并目录中重新序列化的 tokenizer 文件给对照评测引入
额外变量；正式评测前可逐条核对两套 tokenizer 的 prompt token IDs。

至少比较：

- Strict JSON / Schema valid；
- evidence grounding；
- evidence exact/boundary 指标；
- critical evidence；
- task type 与 sufficiency；
- C-Eval。

### 步骤 11：D0-vs-D1 双向 A/B

先预览：

```bash
python3 process_dpo/judge_dpo_evaluation.py \
  --baseline-predictions results/evidence_d0_direct_eval/predictions.jsonl \
  --dpo-predictions results/evidence_dpo_answer_eval/predictions.jsonl \
  --preview-output results/evidence_dpo_answer_eval/answer_ab_preview.jsonl \
  --limit 10 \
  --dry-run
```

正式评测：

```bash
python3 process_dpo/judge_dpo_evaluation.py \
  --baseline-predictions results/evidence_d0_direct_eval/predictions.jsonl \
  --dpo-predictions results/evidence_dpo_answer_eval/predictions.jsonl \
  --output results/evidence_dpo_answer_eval/answer_ab_judgments.jsonl \
  --failed-output results/evidence_dpo_answer_eval/answer_ab_failed.jsonl \
  --summary-output results/evidence_dpo_answer_eval/answer_ab_summary.json \
  --model "$JUDGE_MODEL" \
  --workers 8 \
  --temperature 0
```

每条病例调用两次 Judge。只有交换 A/B 后仍映射到同一模型的结果才计入一致结果。

## 7. 首轮成功标准

D1 值得保留，需要同时满足：

- swap consistency 足够高，建议至少 80%；
- 在一致且有明确胜负的样本中，D1 win rate 建议达到约 60%；
- 提升来自完整性、忠实性、安全性或校准，而不是单纯变长；
- Strict JSON、Schema 和 grounding 基本保持；
- task type、sufficiency、critical evidence 不明显下降；
- C-Eval 总分下降不超过约 1 个百分点。

A/B 结果只能称为“模型裁判偏好胜率”，不能称为临床正确率。

## 8. 停止条件

出现以下情况时不要继续增加 epoch：

- D1 win rate 没有稳定超过随机基线；
- swap consistency 很低；
- 回答普遍变长但质量没有提升；
- 信息充分场景中过度拒答；
- Schema、grounding 或 critical evidence 明显下降；
- C-Eval 明显回退；
- audit 显示 chosen/rejected 主要差异不是回答质量。

此时应回到偏好数据和 Judge rubric，而不是继续堆训练步数。

## 9. 关键中间文件

| 文件 | 内容 | 用于训练 |
| --- | --- | --- |
| `00_sources.jsonl` | 来源病例与 clean target | 否 |
| `01_sft_candidates.jsonl` | Direct SFT 多候选及审计 | 否 |
| `02_pair_candidates.jsonl` | 严格 Answer-level A/B pair | 否 |
| `03_judgments.jsonl` | 训练 pair 单次来源盲评 | 否 |
| `03_swap_judgments.jsonl` | 合格自然 pair 的 A/B 交换复审 | 否 |
| `03_reconciled_judgments.jsonl` | Tier A/B 双向一致裁决与 controlled 标记 | 导出输入 |
| `train/train.jsonl` | DPO train | 是 |
| `validation/validation.jsonl` | DPO validation | 是 |
| `audit/*.jsonl` | 来源、差异字段、Judge 理由 | 否 |
| `04_export.stats.json` | 门禁和最终分布 | 否 |
| `04_export.status.json` | 本轮是否实际写入正式数据及阻断原因 | 否 |
| `dpo_start_manifest.json` | D0 合并 provenance | 否 |
| `dpo_training_manifest.json` | D1 起点/reference/超参数 | 否 |
| `answer_ab_summary.json` | D0-vs-D1 双向 A/B 汇总 | 否 |
