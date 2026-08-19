# Evidence Mask 最简消融实验运行说明（实习项目版）

> 版本：minimal-v1  
> 日期：2026-08-13  
> 父模型：`outputs/evidence-sft-frombase`  
> 实验定位：单 checkpoint、单 seed 的配对概念验证，不估计完整训练流程方差。

## 1. 要回答的问题

本实验只回答两个问题：

1. 删除关键证据后，critical-mask 训练是否让模型更合理地降低结论强度、补充缺失信息，并减少对已删除事实的依赖？
2. 这种改善是否主要发生在 critical 删除上，而不是删除 supporting 文本后也一律变得保守？

实验结论必须拆成两个口径：

- **病例内干预差异：** 同一模型在 unmasked、critical、supporting 版本上的响应差异；
- **训练增量：** M1 相对等训练预算 M0 的改善，即 `M1 - M0`。

本实验不能证明 Teacher 的 critical 标签是临床金标准，也不能证明完整训练流程在不同初始化下稳定复现。

## 2. 最简实验矩阵

| 组别 | 父 checkpoint | 继续训练数据 | Epoch / Seed | 作用 |
|---|---|---|---|---|
| M0-active | Direct Evidence-SFT | 原始 7,671 条 Evidence | 1 / 42 | 匹配额外训练量 |
| M1-critical | 同一个 Direct Evidence-SFT | 同一批 7,671 个 parent，其中 1,000 条替换为 critical-mask target | 1 / 42 | Mask 实验组 |

M0 和 M1 必须：

- 从同一个父 adapter 分叉；
- 使用相同 parent ID、顺序、训练步数和超参数；
- 只在 1,000 条训练记录是否替换为 critical-mask target 上存在差异；
- 使用同一组 100 个 held-out test parent 做配对评测。

测试只保留三种版本：

- `unmasked`：完整病例；
- `critical`：删除唯一 critical evidence；
- `supporting`：删除长度最接近的 supporting evidence。

本版不做 random mask、3 seeds、P0 单独推理和 unmasked final-answer 全量盲评。

## 3. 数据规模与目录

候选规模：

| 数据 | 候选数 | 正式使用数 |
|---|---:|---:|
| Train critical | 1,200 | 1,000 replacements |
| Validation mask | 0 | 0 replacements |
| Test parents | 120 | 100 pairs |

validation 不做 Mask 替换。M0/M1 均使用原始 880 条 validation，使 validation loss 具有直接可比性；Mask 泛化由独立 test pairs 衡量。

输出目录固定为：

```text
data/evidence_mask/minimal_v1/
├── 00_candidates.jsonl
├── 00_candidates.stats.json
├── 01_teacher_requests.preview.jsonl
├── 01_teacher_raw.jsonl
├── 01_teacher_failed.jsonl
├── 02_judgments.jsonl
├── 02_judgments_failed.jsonl
└── validated/
    ├── 03_validated_full.jsonl
    ├── 03_review.jsonl
    ├── 03_rejected.jsonl
    ├── 03_validation.stats.json
    ├── test_pairs.jsonl
    ├── m0/train.jsonl
    ├── m0/validation.jsonl
    ├── m1/train.jsonl
    └── m1/validation.jsonl
```

## 4. 指标与判定口径

### 4.1 主指标

1. `critical appropriate response`：删除 critical 后，final answer 是否做出与剩余证据匹配的降级或保持；
2. `critical removed-fact leakage`：是否继续把已删除事实作为患者已知事实。

### 4.2 选择性诊断

```text
Selectivity = (M1-M0)_critical appropriate response
            - (M1-M0)_supporting appropriate response
```

`Selectivity > 0` 才能支持“收益主要来自关键证据缺失处理”，而不是“删任何文本都变保守”。

### 4.3 原能力保护指标

- 原始 444 Evidence：Schema、span grounding、boundary-compatible F1；
- C-Eval 三个医学子任务总分。

### 4.4 结果分级

- **强结果：** critical appropriate response 提升至少 10 pp、配对 95% CI 下界大于 0、Selectivity > 0，且 leakage 不增加；
- **方向性结果：** critical 与 Selectivity 均为正，但 CI 跨 0；只能称为单 seed 探索性增益；
- **无证据/负结果：** critical 无改善、leakage 增加，或 supporting 的变化与 critical 同样大。

保护门槛：

- 原始 Schema、span grounding 下降不超过 1 pp；
- boundary-compatible F1 下降不超过 2 pp；
- C-Eval 总分下降不超过 1 pp（约 8/818 题）。

## 5. 环境与固定资产

服务器项目根目录：

```bash
cd /home/medgpt
export PYTHON_BIN=/root/miniconda3/envs/medgpt/bin/python
export PARENT_ADAPTER="$PWD/outputs/evidence-sft-frombase"
export MIN_MASK_ROOT="$PWD/data/evidence_mask/minimal_v1"
```

正式运行前记录父 adapter：

```bash
sha256sum "$PARENT_ADAPTER/adapter_model.safetensors"
```

本轮锁定哈希：

```text
585029bb15c128e5bb3ff8c5e82492847e6ad92313bf39293f16d5d16afeb090
```

Teacher 配置从项目根目录 `.teacher_env` 读取。若没有独立 Judge，可复用同一模型完成概念验证，但报告必须标注 Judge 非独立，并人工盲审至少 20 个 critical test pairs。

## 6. 构造候选与防泄漏预览

```bash
$PYTHON_BIN process/build_evidence_mask_candidates.py \
  --input data/evidence_sft/validated_v2_2/03_validated_full.jsonl \
  --output-dir "$MIN_MASK_ROOT" \
  --train-candidates 1200 \
  --validation-candidates 0 \
  --test-candidates 120 \
  --no-random-control \
  --seed 42
```

生成 20 条 teacher 请求预览：

```bash
$PYTHON_BIN process/build_evidence_mask_targets.py \
  --input "$MIN_MASK_ROOT/00_candidates.jsonl" \
  --preview-output "$MIN_MASK_ROOT/01_teacher_requests.preview.jsonl" \
  --limit 20 \
  --dry-run
```

门禁：预览只能包含 masked case，不得包含 `original_case_text`、原 target 或 removed span。

## 7. 生成并审核 Mask targets

正式生成：

```bash
$PYTHON_BIN process/build_evidence_mask_targets.py \
  --input "$MIN_MASK_ROOT/00_candidates.jsonl" \
  --output "$MIN_MASK_ROOT/01_teacher_raw.jsonl" \
  --failed-output "$MIN_MASK_ROOT/01_teacher_failed.jsonl" \
  --workers 8
```

审核预览：

```bash
$PYTHON_BIN process/judge_evidence_mask_targets.py \
  --candidates "$MIN_MASK_ROOT/00_candidates.jsonl" \
  --teacher "$MIN_MASK_ROOT/01_teacher_raw.jsonl" \
  --preview-output "$MIN_MASK_ROOT/02_judge_requests.preview.jsonl" \
  --model "$JUDGE_MODEL" \
  --limit 20 \
  --dry-run
```

正式审核：

```bash
$PYTHON_BIN process/judge_evidence_mask_targets.py \
  --candidates "$MIN_MASK_ROOT/00_candidates.jsonl" \
  --teacher "$MIN_MASK_ROOT/01_teacher_raw.jsonl" \
  --output "$MIN_MASK_ROOT/02_judgments.jsonl" \
  --failed-output "$MIN_MASK_ROOT/02_judgments_failed.jsonl" \
  --model "$JUDGE_MODEL" \
  --workers 8
```

使用同一模型时，把 `.teacher_env` 中的 `TEACHER_MODEL` 同时赋给 `JUDGE_MODEL`。不得在日志或文档中写入 API key。

## 8. 验证并导出等预算 manifests

```bash
$PYTHON_BIN process/validate_evidence_mask.py \
  --candidates "$MIN_MASK_ROOT/00_candidates.jsonl" \
  --teacher "$MIN_MASK_ROOT/01_teacher_raw.jsonl" \
  --judgments "$MIN_MASK_ROOT/02_judgments.jsonl" \
  --output-dir "$MIN_MASK_ROOT/validated" \
  --train-replacements 1000 \
  --validation-replacements 0 \
  --test-pairs 100 \
  --seed 42
```

必须满足：

- `m0_train = m1_train = 7671`；
- `m1_train_replacements = 1000`；
- `m0_validation = m1_validation = 880`；
- M0/M1 train parent ID 集合和顺序一致；
- M0/M1 validation 完全相同；
- train、validation、test parent 无交集；
- `test_pairs.jsonl` 恰好为 100 组 U/C/S，即 300 条记录。

若 accepted 数不足，补跑 teacher/Judge 或扩大候选；正式实验不使用 `--allow-missing-judgments`。

## 9. 测试与训练

```bash
$PYTHON_BIN -m unittest discover -s tests -p 'test_evidence_sft*.py' -v
$PYTHON_BIN -m unittest discover -s tests -p 'test_evidence_mask*.py' -v
```

训练：

```bash
DATA_ROOT="$MIN_MASK_ROOT/validated" \
OUTPUT_DIR="$PWD/outputs/evidence-mask-min-m0-seed42" \
ARM=m0 SEED=42 bash evi-sft-traing/run_evidence_mask_sft.sh

DATA_ROOT="$MIN_MASK_ROOT/validated" \
OUTPUT_DIR="$PWD/outputs/evidence-mask-min-m1-seed42" \
ARM=m1 SEED=42 bash evi-sft-traing/run_evidence_mask_sft.sh
```

主分析固定使用训练结束 checkpoint，不根据 test 结果选择 step。

## 10. Mask 配对评测与 Judge

```bash
$PYTHON_BIN evi-sft-traing/evaluate_evidence_mask.py \
  --adapter outputs/evidence-mask-min-m0-seed42 \
  --test-file "$MIN_MASK_ROOT/validated/test_pairs.jsonl" \
  --output-dir results/evidence_mask_min_eval/m0-seed42 \
  --batch-size 128

$PYTHON_BIN evi-sft-traing/evaluate_evidence_mask.py \
  --adapter outputs/evidence-mask-min-m1-seed42 \
  --test-file "$MIN_MASK_ROOT/validated/test_pairs.jsonl" \
  --output-dir results/evidence_mask_min_eval/m1-seed42 \
  --batch-size 128
```

分别运行 response Judge：

```bash
$PYTHON_BIN evi-sft-traing/judge_evidence_mask_predictions.py \
  --predictions results/evidence_mask_min_eval/m0-seed42/predictions.jsonl \
  --output results/evidence_mask_min_eval/m0-seed42/judgments.jsonl \
  --model "$JUDGE_MODEL" \
  --workers 8

$PYTHON_BIN evi-sft-traing/judge_evidence_mask_predictions.py \
  --predictions results/evidence_mask_min_eval/m1-seed42/predictions.jsonl \
  --output results/evidence_mask_min_eval/m1-seed42/judgments.jsonl \
  --model "$JUDGE_MODEL" \
  --workers 8
```

聚合：

```bash
$PYTHON_BIN evi-sft-traing/aggregate_evidence_mask.py \
  --predictions results/evidence_mask_min_eval/m1-seed42/predictions.jsonl \
  --judgments results/evidence_mask_min_eval/m1-seed42/judgments.jsonl \
  --baseline-predictions results/evidence_mask_min_eval/m0-seed42/predictions.jsonl \
  --baseline-judgments results/evidence_mask_min_eval/m0-seed42/judgments.jsonl \
  --output results/evidence_mask_min_eval/comparison-seed42.json \
  --bootstrap-iters 5000 \
  --seed 42
```

## 11. 原能力保护评测

M0/M1 都在原始 444 条 Evidence test 上评测：

```bash
$PYTHON_BIN evi-sft-traing/evaluate_evidence.py \
  --adapter outputs/evidence-mask-min-m0-seed42 \
  --test-file data/evidence_sft/validated_v2_2/test.jsonl \
  --output-dir results/evidence_mask_min_eval/m0-seed42-unmasked \
  --batch-size 128

$PYTHON_BIN evi-sft-traing/evaluate_evidence.py \
  --adapter outputs/evidence-mask-min-m1-seed42 \
  --test-file data/evidence_sft/validated_v2_2/test.jsonl \
  --output-dir results/evidence_mask_min_eval/m1-seed42-unmasked \
  --batch-size 128
```

C-Eval 对 M0/M1 各运行一次，并使用 `--log_samples` 保存逐题结果。这样可以区分 Mask 训练影响与“额外继续训练一轮”的影响。

## 12. 对外表述边界

推荐表述：

> 在同一个 Direct Evidence-SFT checkpoint 上构造等训练预算 M0/M1，使用 100 个 held-out 病例的 unmasked/critical/supporting 反事实版本进行配对评测，验证 critical-mask 增强是否改善关键证据缺失时的安全不确定性响应，并检查原始 Evidence 与 C-Eval 能力是否保持。

必须主动说明：

- 单 seed 只支持概念验证，配对 bootstrap 反映测试样本不确定性，不包含训练 seed 方差；
- Mask target 和 Judge 来自模型评审，不是临床金标；
- Teacher 与 Judge 若复用同一模型，会产生相关偏差；
- 100 对测试的置信区间可能较宽，正向但跨 0 时只能称为方向性结果；
- 不把 grounding、teacher agreement 或 Judge agreement 表述为临床正确率。
