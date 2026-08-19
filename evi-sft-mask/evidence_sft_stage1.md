# Evidence-SFT 第一阶段（V2.2）

本阶段只使用病例原文和原始低质量回答进行强教师响应蒸馏，不使用指南、检索或 DPO。正式验证采用 `evidence-sft-v2.2` 数据契约，并将结果分为 accepted、review、rejected 三类。

## 1. 选择病例候选

```bash
/root/miniconda3/envs/medgpt/bin/python process/prepare_evidence_candidates.py \
  --input data/sft/medical_100k.jsonl \
  --output data/evidence_sft/00_candidates.jsonl \
  --stats-output data/evidence_sft/00_candidates.stats.json \
  --target-size 10000 \
  --min-score 6
```

候选在教师调用前分为 train/validation/test。稳定 ID、划分和抽样均由固定 seed 决定。

## 2. 预览教师请求

```bash
/root/miniconda3/envs/medgpt/bin/python process/build_evidence_sft.py \
  --input data/evidence_sft/00_candidates.jsonl \
  --preview-output data/evidence_sft/01_teacher_requests.preview.jsonl \
  --dry-run \
  --limit 3
```

预览不会调用外部 API。

## 3. 调用强教师

密钥只通过环境变量传入：

```bash
export TEACHER_API_KEY='...'
export TEACHER_BASE_URL='https://provider.example/v1'
export TEACHER_MODEL='teacher-model-name'

/root/miniconda3/envs/medgpt/bin/python process/build_evidence_sft.py \
  --input data/evidence_sft/00_candidates.jsonl \
  --output data/evidence_sft/01_teacher_raw.jsonl \
  --failed-output data/evidence_sft/01_teacher_failed.jsonl \
  --workers 8
```

成功响应写入 `01_teacher_raw.jsonl`，API 或 JSON 解析失败写入 `01_teacher_failed.jsonl`。重新执行时会按 `source_id` 跳过成功样本。

## 4. 校验并导出训练数据

```bash
/root/miniconda3/envs/medgpt/bin/python process/validate_evidence_sft.py \
  --input data/evidence_sft/01_teacher_raw.jsonl \
  --candidates data/evidence_sft/00_candidates.jsonl \
  --output-dir data/evidence_sft/validated_v2_2
```

主要产物：

- `03_validated_full.jsonl`：通过硬规则和复核规则的 clean 样本；
- `03_review.jsonl`：结构可恢复、但需人工确认 Atomicity 或证据边界的样本；
- `03_rejected.jsonl`：存在不可训练硬错误的样本；
- `train.jsonl`、`validation.jsonl`、`test.jsonl`：只由 clean accepted 样本生成；
- `03_validation.stats.json`：完整性、分流原因、警告和字段分布统计。

验证不会覆盖 `01_teacher_raw.jsonl`。`critical_evidence_ids` 为空是合法情况；`task_type_hint` 与 teacher 输出不一致仅作为抽检警告，不会自动拒绝。

## 5. 本地测试

```bash
/root/miniconda3/envs/medgpt/bin/python -m unittest discover -v -s tests
```
