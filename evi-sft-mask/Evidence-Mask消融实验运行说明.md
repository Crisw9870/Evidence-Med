# Evidence Mask 消融实验：文件说明与运行命令

> 版本：v1  
> 日期：2026-08-12  
> 主实验父模型：`outputs/evidence-sft-frombase`  
> 目标：比较等训练预算的 M0-active 与 M1-critical，验证关键证据缺失时的安全不确定性响应是否改善。

## 1. 实验定义

本实验中的 Mask 是病例级反事实干预，不是 attention mask，也不是 `train_on_inputs` 的 loss mask。

| 组别 | 相同父 checkpoint | 继续训练数据 | 作用 |
|---|---|---|---|
| P0 | Direct Evidence-SFT | 不继续训练 | 训练前参考，不是主因果对照 |
| M0-active | P0 | 7,671 条原始 Evidence | 匹配额外训练量 |
| M1-critical | P0 | 同一批 7,671 个 parent，其中 1,918 条替换为 critical-mask 反事实 | Mask 实验组 |

M0 与 M1 都从同一个 Direct adapter 分叉，训练 1 epoch，约 240 steps。M1 不是在 M0 之后继续训练，Mask 样本也不是简单追加到原训练集末尾。

测试以 `pair_id` 为病例单位，包含：

- `unmasked`：完整病例；
- `critical`：删除唯一 critical evidence；
- `supporting`：删除长度最接近的 supporting evidence；
- `random`：尽可能删除长度相近、未被标注为 evidence 的普通文本。

首版训练只使用“恰好 1 条 critical 且至少 1 条 supporting”的 accepted 样本。现有合格 parent 数为 train 4,008、validation 454、test 250。

## 2. 新增文件及作用

### 2.1 数据构造与验证

| 文件 | 作用 |
|---|---|
| `process/evidence_mask_common.py` | JSONL、确定性抽样、offset 校验、自然删除、supporting/random 对照选择等公共函数 |
| `process/build_evidence_mask_candidates.py` | 从 `03_validated_full.jsonl` 选择候选，生成 critical/supporting/random Mask 变体并保留 parent、split 和 removed-span provenance |
| `process/build_evidence_mask_targets.py` | 调用 teacher 为 masked case 重新生成完整 Evidence JSON；teacher prompt 不包含原病例、原 target 或 `original_answer` |
| `process/judge_evidence_mask_targets.py` | 独立审核 full/masked pair，输出预期确定性变化、缺失概念、允许结论范围和禁止断言 |
| `process/validate_evidence_mask.py` | 复用 Evidence Schema/grounding 校验，增加 parent/split、隐藏事实和 Judge gate，并导出等预算 M0/M1 manifest 与 test pairs |

### 2.2 训练与评测

| 文件 | 作用 |
|---|---|
| `evi-sft-traing/run_evidence_mask_sft.sh` | 从同一个 Direct LoRA 继续训练 M0 或 M1；默认 1 epoch、LR 5e-6、有效 batch 32 |
| `evi-sft-traing/evaluate_evidence_mask.py` | 按 `variant_id` 生成 U/C/S/R 输出，复用 JSON、Schema、grounding 和 teacher 指标，并保留配对元数据 |
| `evi-sft-traing/judge_evidence_mask_predictions.py` | 对模型身份盲化，审核 final answer 的适当不确定性、删除事实泄漏、结论范围、缺失信息与安全性 |
| `evi-sft-traing/aggregate_evidence_mask.py` | 汇总 sufficiency macro-F1、逐类 recall、Judge 指标，并对 M1/M0 做配对 bootstrap 和 exact McNemar 检验 |
| `tests/test_evidence_mask_pipeline.py` | 覆盖唯一删除、split 继承、teacher prompt 防泄漏、M0/M1 等量 manifest、U/C/S 配对和配对统计 |

### 2.3 生成目录

```text
data/evidence_mask/v1/
├── 00_candidates.jsonl
├── 00_candidates.stats.json
├── 01_teacher_raw.jsonl
├── 01_teacher_failed.jsonl
├── 02_judgments.jsonl
├── 02_judgments_failed.jsonl
└── validated/
    ├── 03_validated_full.jsonl
    ├── 03_review.jsonl
    ├── 03_rejected.jsonl
    ├── 03_validation.stats.json
    ├── masked_train.jsonl
    ├── masked_validation.jsonl
    ├── masked_test.jsonl
    ├── test_pairs.jsonl
    ├── m0/train.jsonl
    ├── m0/validation.jsonl
    ├── m1/train.jsonl
    └── m1/validation.jsonl
```

`m0` 和 `m1` 目录只放训练器应读取的 manifest。不要把审计、teacher raw 或 test JSONL 放入这两个目录，因为训练器会递归读取目录中的全部 JSONL。

## 3. Windows：数据构造、审核与验证

所有本地 Python 命令使用项目指定解释器：

```powershell
Set-Location 'D:\DeepLearning\Evidence-Med'
$MEDGPT_PY = 'D:\miniconda3\envs\medgpt\python.exe'
```

### 3.1 构造候选

```powershell
& $MEDGPT_PY process\build_evidence_mask_candidates.py `
  --input data\evidence_sft\validated_v2_2\03_validated_full.jsonl `
  --output-dir data\evidence_mask\v1 `
  --train-candidates 2300 `
  --validation-candidates 240 `
  --test-candidates 250 `
  --seed 42
```

默认进行自然删除，不向模型输入插入 `[MASK]`。当前实际统计为 train critical 2,300、validation critical 240、test critical/supporting 各 250。random 对照只在找到安全候选时生成，缺失数量会写入 stats，不会强行补齐。

### 3.2 配置并运行 teacher

可以通过环境变量或项目根目录的 `.teacher_env` 配置。PowerShell 示例：

```powershell
$env:TEACHER_MODEL = '<teacher-model>'
$env:TEACHER_API_KEY = '<api-key>'
$env:TEACHER_BASE_URL = '<openai-compatible-base-url>'
```

先检查 20 个请求，确认 prompt 只有 masked case：

```powershell
& $MEDGPT_PY process\build_evidence_mask_targets.py `
  --input data\evidence_mask\v1\00_candidates.jsonl `
  --preview-output data\evidence_mask\v1\01_teacher_requests.preview.jsonl `
  --limit 20 `
  --dry-run
```

正式生成 target：

```powershell
& $MEDGPT_PY process\build_evidence_mask_targets.py `
  --input data\evidence_mask\v1\00_candidates.jsonl `
  --output data\evidence_mask\v1\01_teacher_raw.jsonl `
  --failed-output data\evidence_mask\v1\01_teacher_failed.jsonl `
  --model $env:TEACHER_MODEL `
  --workers 8
```

脚本支持断点续跑；已有 `status=ok` 的 `source_id` 会跳过。

### 3.3 独立审核反事实 target

建议 Judge 与 teacher 使用不同模型；如果资源有限，也可以先使用同一模型完成流程试跑，但报告中需注明不独立。

```powershell
$env:JUDGE_MODEL = '<judge-model>'
$env:JUDGE_API_KEY = '<api-key>'
$env:JUDGE_BASE_URL = '<openai-compatible-base-url>'
```

先预览：

```powershell
& $MEDGPT_PY process\judge_evidence_mask_targets.py `
  --candidates data\evidence_mask\v1\00_candidates.jsonl `
  --teacher data\evidence_mask\v1\01_teacher_raw.jsonl `
  --preview-output data\evidence_mask\v1\02_judge_requests.preview.jsonl `
  --limit 20 `
  --dry-run
```

正式审核：

```powershell
& $MEDGPT_PY process\judge_evidence_mask_targets.py `
  --candidates data\evidence_mask\v1\00_candidates.jsonl `
  --teacher data\evidence_mask\v1\01_teacher_raw.jsonl `
  --output data\evidence_mask\v1\02_judgments.jsonl `
  --failed-output data\evidence_mask\v1\02_judgments_failed.jsonl `
  --model $env:JUDGE_MODEL `
  --workers 8
```

Judge 不会机械要求每个 critical 删除都降低 sufficiency。存在证据冗余时可输出 `expected_certainty_change=stay`。

### 3.4 验证并导出 M0/M1

```powershell
& $MEDGPT_PY process\validate_evidence_mask.py `
  --candidates data\evidence_mask\v1\00_candidates.jsonl `
  --teacher data\evidence_mask\v1\01_teacher_raw.jsonl `
  --judgments data\evidence_mask\v1\02_judgments.jsonl `
  --output-dir data\evidence_mask\v1\validated `
  --train-replacements 1918 `
  --validation-replacements 200 `
  --test-pairs 200 `
  --seed 42
```

正式实验不要使用 `--allow-missing-judgments`。该参数只用于开发期验证 Schema 和 manifest 代码。

验证后重点检查：

```powershell
Get-Content data\evidence_mask\v1\validated\03_validation.stats.json
```

必须满足：

- `m0_train = m1_train = 7671`；
- `m1_train_replacements = 1918`；
- `m0_validation = m1_validation = 880`；
- M0/M1 parent ID 集合完全相同；
- train、validation、test 的 parent 没有交集；
- test 至少有 200 组完整 critical/supporting pair。

如果 accepted 数不足，先检查 `03_review.jsonl` 和 `03_rejected.jsonl`，修复 teacher/Judge 质量后补跑；不要降低验证门槛或减少正式 test 数来迁就失败数据。

## 4. 运行测试

只运行 Mask 测试：

```powershell
& $MEDGPT_PY -m unittest discover -s tests -p 'test_evidence_mask*.py' -v
```

同时运行原 Evidence 回归测试：

```powershell
& $MEDGPT_PY -m unittest discover -s tests -p 'test_evidence_sft*.py' -v
& $MEDGPT_PY -m unittest discover -s tests -p 'test_evidence_mask*.py' -v
```

当前验证结果：原 Evidence 15/15 通过，Mask 5/5 通过。

## 5. Linux GPU：训练 M0 与 M1

训练脚本沿用此前 Linux GPU 环境。以下命令假设项目位于 `/home/medgpt`；如路径不同，请先进入实际项目根目录。

```bash
cd /home/medgpt
export PYTHON_BIN=/root/miniconda3/envs/medgpt/bin/python
export PARENT_ADAPTER="$PWD/outputs/evidence-sft-frombase"
export DATA_ROOT="$PWD/data/evidence_mask/v1/validated"
```

先记录父 adapter 哈希，所有实验臂必须一致：

```bash
sha256sum outputs/evidence-sft-frombase/adapter_model.safetensors
```

单 seed 试跑：

```bash
ARM=m0 SEED=42 bash evi-sft-traing/run_evidence_mask_sft.sh
ARM=m1 SEED=42 bash evi-sft-traing/run_evidence_mask_sft.sh
```

正式三个配对 seed：

```bash
for seed in 42 43 44; do
  ARM=m0 SEED="$seed" bash evi-sft-traing/run_evidence_mask_sft.sh
  ARM=m1 SEED="$seed" bash evi-sft-traing/run_evidence_mask_sft.sh
done
```

默认输出：

```text
outputs/evidence-mask-m0-seed42
outputs/evidence-mask-m1-seed42
outputs/evidence-mask-m0-seed43
outputs/evidence-mask-m1-seed43
outputs/evidence-mask-m0-seed44
outputs/evidence-mask-m1-seed44
```

主分析固定使用训练结束 checkpoint，不根据 test 结果选择 step。每 40 steps 的 validation 仅用于监控 NaN、Schema 崩溃或 grounding 大幅下降。

## 6. Linux GPU：生成 Mask 配对预测

以 seed 42 为例：

```bash
$PYTHON_BIN evi-sft-traing/evaluate_evidence_mask.py \
  --adapter outputs/evidence-mask-m0-seed42 \
  --test-file data/evidence_mask/v1/validated/test_pairs.jsonl \
  --output-dir results/evidence_mask_eval/m0-seed42 \
  --batch-size 128

$PYTHON_BIN evi-sft-traing/evaluate_evidence_mask.py \
  --adapter outputs/evidence-mask-m1-seed42 \
  --test-file data/evidence_mask/v1/validated/test_pairs.jsonl \
  --output-dir results/evidence_mask_eval/m1-seed42 \
  --batch-size 128
```

显存不足时降低 `--batch-size`，不要改变 decoding。评测固定使用 greedy 和 `max_new_tokens=1152`。

## 7. Final answer 盲评

先在 Linux 评测环境中配置 Judge。建议与 target Judge 使用同一套兼容 OpenAI 协议的服务，但不要把密钥写进脚本或结果文件：

```bash
export JUDGE_MODEL="your-judge-model"
export JUDGE_API_KEY="your-api-key"
export JUDGE_BASE_URL="https://your-endpoint/v1"
```

先预览 Judge 请求：

```bash
$PYTHON_BIN evi-sft-traing/judge_evidence_mask_predictions.py \
  --predictions results/evidence_mask_eval/m1-seed42/predictions.jsonl \
  --output results/evidence_mask_eval/m1-seed42/judgments.jsonl \
  --model "$JUDGE_MODEL" \
  --limit 20 \
  --dry-run
```

正式评审 M0/M1：

```bash
$PYTHON_BIN evi-sft-traing/judge_evidence_mask_predictions.py \
  --predictions results/evidence_mask_eval/m0-seed42/predictions.jsonl \
  --output results/evidence_mask_eval/m0-seed42/judgments.jsonl \
  --model "$JUDGE_MODEL" \
  --workers 8

$PYTHON_BIN evi-sft-traing/judge_evidence_mask_predictions.py \
  --predictions results/evidence_mask_eval/m1-seed42/predictions.jsonl \
  --output results/evidence_mask_eval/m1-seed42/judgments.jsonl \
  --model "$JUDGE_MODEL" \
  --workers 8
```

建议从 Judge 结果中随机抽取 10%～20% 做人工校准，并报告人工/Judge 一致性。

## 8. 配对统计

```bash
$PYTHON_BIN evi-sft-traing/aggregate_evidence_mask.py \
  --predictions results/evidence_mask_eval/m1-seed42/predictions.jsonl \
  --judgments results/evidence_mask_eval/m1-seed42/judgments.jsonl \
  --baseline-predictions results/evidence_mask_eval/m0-seed42/predictions.jsonl \
  --baseline-judgments results/evidence_mask_eval/m0-seed42/judgments.jsonl \
  --output results/evidence_mask_eval/comparison-seed42.json \
  --bootstrap-iters 5000 \
  --seed 42
```

聚合器会输出：

- critical/supporting/random 各自的 JSON、Schema、grounding；
- sufficiency majority baseline、macro-F1、balanced accuracy 和逐类 recall；
- final-answer appropriate response、removed-fact leakage、missing information 与安全性；
- M1−M0 的绝对百分点差、净改善样本数、配对 bootstrap 95% CI；
- exact McNemar p 值；
- critical 增量相对 supporting/random 增量的选择性差值。

三个 seed 应分别生成 comparison 文件，然后汇总均值、标准差和逐 seed 方向。三个 continuation seed 共享同一个父 checkpoint，因此结论应写成“条件于当前 Direct checkpoint 的稳定增益”，不能外推为完整训练流程方差。

## 9. 原始能力保护指标

每个 M0/M1 checkpoint 还要在原始 444 条 Evidence test 上评测：

```bash
$PYTHON_BIN evi-sft-traing/evaluate_evidence.py \
  --adapter outputs/evidence-mask-m1-seed42 \
  --test-file data/evidence_sft/validated_v2_2/test.jsonl \
  --output-dir results/evidence_mask_eval/m1-seed42-unmasked \
  --batch-size 128
```

M0 使用相同命令，只替换 adapter 与 output-dir。

C-Eval 示例：

```bash
lm_eval \
  --model hf \
  --model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft=outputs/evidence-mask-m1-seed42,dtype=bfloat16,trust_remote_code=True \
  --tasks ceval-test_basic_medicine,ceval-test_clinical_medicine,ceval-test_physician \
  --include_path ./data/lm_eval_tasks \
  --num_fewshot 5 \
  --batch_size 8 \
  --log_samples \
  --output_path ./results/evidence_mask_eval/m1-seed42-ceval
```

本轮应使用 `--log_samples` 保存逐题结果，避免再次只能比较聚合分数。

## 10. 暂定成功标准

M1 进入后续 DPO 前建议同时满足：

1. critical Mask 的 final-answer appropriate response 相对 M0 提升至少 10 pp；
2. 配对 95% CI 下界大于 0，三个 seed 方向一致；
3. M1−M0 在 critical 上的增量高于 supporting/random，证明不是“删任何文本都变保守”；
4. removed-fact semantic leakage 不增加，最好下降至少 5 pp；
5. masked Schema ≥99%，span grounding ≥98.5%；
6. 原始 444 Evidence 的 Schema、grounding 下降不超过 0.5 pp，boundary F1 下降不超过 1 pp；
7. C-Eval 总分下降不超过 1 pp，即不超过约 8/818 题；
8. unmasked final answer 盲评不出现超过 5 pp 的退化。

如果 Mask 指标提升但原始能力下降，应先降低替换比例；如果 Mask 指标不提升，应优先检查删除质量、teacher target 和 Judge rubric，不应直接增加训练轮数或提前用 DPO 掩盖问题。

## 11. 与 DPO 的边界

本轮代码只完成 Mask 消融，不启动 DPO。Mask 结果和 checkpoint 锁定后：

- 复制同一个获胜 Mask checkpoint；
- 一份冻结为 reference；
- 一份作为 policy 训练；
- 不得用 Direct 作为 Two-stage policy 的 reference，或反过来；
- DPO 不能复用 Mask test 调参。

这样才能分别解释 Mask 与 DPO 的独立增量。
