# Evidence-SFT 数据构造与训练流水线

将低质量医疗 QA 重构为**证据可追溯、结构化**的 SFT 训练数据，并完成微调和评估。

## 整体流程

```
原始医疗 QA (medical_100k.jsonl)
    │
    ▼
┌─────────────────────────────────────────┐
│  Step 0: prepare_evidence_candidates.py │  筛选高质量病例候选项
└─────────────────┬───────────────────────┘
                  │ 00_candidates.jsonl
                  ▼
┌─────────────────────────────────────────┐
│  Step 1: build_evidence_sft.py          │  Teacher LLM 蒸馏结构化证据
└─────────────────┬───────────────────────┘
                  │ 01_teacher_raw.jsonl
                  ▼
┌─────────────────────────────────────────┐
│  Step 2: validate_evidence_sft.py       │  三级审计 → accepted / review / rejected
└─────────────────┬───────────────────────┘
                  │ train.jsonl / validation.jsonl / test.jsonl
                  ▼
┌─────────────────────────────────────────┐
│  Step 3: run_evidence_sft.sh            │  LoRA 微调 Qwen2.5-7B-Instruct
└─────────────────┬───────────────────────┘
                  │ outputs/evidence-sft-frombase/
                  ▼
┌─────────────────────────────────────────┐
│  Step 4: evaluate_evidence.py           │  测试集推理 + 自动评分
└─────────────────────────────────────────┘
                  │ results/evidence_eval/metrics.json
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `evidence_sft_common.py` | 共享工具库：常量定义、JSONL 读写、病例评分、任务分类、确定性数据划分 |
| `prepare_evidence_candidates.py` | **Step 0** — 从原始 SFT 数据中筛选高质量病例，按诊断/管理任务比例采样，输出候选集 |
| `build_evidence_sft.py` | **Step 1** — 调用 Teacher LLM（多线程并发），将候选病例蒸馏为结构化 evidence-grounded JSON |
| `validate_evidence_sft.py` | **Step 2** — 对 Teacher 输出做三级审计（accepted / review / rejected），导出训练集 |
| `run_evidence_sft.sh` | **Step 3** — 启动 LoRA SFT 训练脚本 |
| `evaluate_evidence.py` | **Step 4** — 加载微调后模型，在测试集上推理并计算 evidence F1、schema 合法率等指标 |

## 各步骤详解

### Step 0: 筛选候选病例

```bash
python process_sft/prepare_evidence_candidates.py \
  --input data/sft/medical_100k.jsonl \
  --output data/evidence_sft/00_candidates.jsonl \
  --target-size 5000 \
  --min-score 6 \
  --diagnostic-ratio 0.6
```

- 对每条 QA 计算启发式质量评分（患者信号、症状、检查、时间、临床问题等）
- 按 `diagnostic_reasoning : confirmed_management = 6:4` 配比采样
- 单科室上限 25%，防止类别偏斜
- 确定性划分 train (85%) / validation (10%) / test (5%)

### Step 1: Teacher 蒸馏

```bash
python process_sft/build_evidence_sft.py \
  --input data/evidence_sft/00_candidates.jsonl \
  --output data/evidence_sft/01_teacher_raw.jsonl \
  --model <teacher_model> \
  --workers 8 \
  --temperature 0.2
```

- 使用精心设计的 System Prompt（v2.2），要求 Teacher 输出：
  - `query_intent` — 用户真正想问什么
  - `evidence` — 从病例原文逐字抽取的最小原子证据（含 importance / role）
  - `critical_evidence_ids` — 删除后会改变主要结论的关键证据
  - `evidence_sufficiency` — sufficient / partial / insufficient / conflicting
  - `missing_information` — 补充后会改变判断的缺失信息
  - `clinical_reasoning` — 证据边界摘要
  - `final_answer` — 基于证据重构的安全回答
- 支持断点续跑（已完成的 source_id 自动跳过）
- API Key 从 `.teacher_env` 或环境变量读取
- 连续失败 10 次自动停止

### Step 2: 验证与导出

```bash
python process_sft/validate_evidence_sft.py \
  --input data/evidence_sft/01_teacher_raw.jsonl \
  --candidates data/evidence_sft/00_candidates.jsonl \
  --output-dir data/evidence_sft/validated_v2_2
```

三级审计：
- **rejected** — 结构性错误（字段缺失、span 不在原文、ID 重复等），不可用于训练
- **review** — 结构合法但有可疑点（span 过长、ID 不连续、unexpected 字段等），需人工审核
- **accepted** — 通过所有硬性检查，直接可用于训练

输出：
- `train.jsonl` / `validation.jsonl` / `test.jsonl` — SFT 训练格式
- `03_validated_full.jsonl` — 全量 accepted 记录
- `03_review.jsonl` — 待人工审核记录
- `03_rejected.jsonl` — 被拒绝记录
- `03_validation.stats.json` — 统计报告

### Step 3: LoRA 微调

```bash
./process_sft/run_evidence_sft.sh
```

默认配置：
- 基座模型：`Qwen/Qwen2.5-7B-Instruct`
- LoRA rank=16, alpha=32, dropout=0.05
- batch size=8, gradient accumulation=4
- epochs=2, learning rate=1e-5
- max sequence length=1536
- 输出：`outputs/evidence-sft-frombase/`

### Step 4: 评估

```bash
python process_sft/evaluate_evidence.py \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --adapter outputs/evidence-sft-frombase \
  --test-file data/evidence_sft/test.jsonl
```

评估指标：
- **JSON 合法率** — strict / recoverable 两种解析模式
- **Schema 合法率** — 字段完整性、枚举值正确性
- **Evidence F1** — 预测 span 与 gold span 的精确匹配 precision/recall/F1
- **Critical F1** — 关键证据 span 的精确匹配
- **Grounding 率** — 预测的 evidence 是否真出现在原文中
- **Critical 一致性** — `critical_evidence_ids` 与 `importance=critical` 是否匹配
- **任务分类准确率** — `task_type` 和 `evidence_sufficiency` 的 confusion matrix

支持 `--resume` 断点续跑。

## Prompt 迭代历程

详见 [prompt改进.md](../docs/prompt改进.md)，核心演进：

| 版本 | 解决的问题 | 关键改进 |
|------|-----------|---------|
| V1.1→V2.0 | sufficiency 定义过宽；无重要性分级 | 加 `partial`；加 `critical/supporting` |
| V2.0→V2.1 | 问题混入 evidence；critical 偏多；role 过度推断 | 独立 `query_intent`；反事实删除原则收紧 critical；限制 role 推断强度 |
| V2.1→V2.2 | 单条 evidence 包含多个临床事实 | Atomic Evidence + Minimal Critical Span |

## 依赖

- Python 3.10+
- `transformers`, `peft`, `torch`, `tqdm`, `openai`
- 训练需要 GPU（推荐 A100 / 4090）
