# Evidence-Med

基于 Qwen2.5-7B-Instruct 的**证据可追溯医疗问答**训练与评估项目。

核心思想：让模型在回答医疗问题时，不只是给出答案，还要输出**可检查、可追踪的结构化证据** — 从病例原文中逐字抽取、标记重要性、判断充分性，为后续反事实训练和偏好优化提供基础。

## 项目架构

```
                  原始医疗 QA
                      │
                      ▼
┌──────────────────────────────────────────────────┐
│  Evidence-SFT 数据构造 (process_sft/)             │
│  候选筛选 → Teacher 蒸馏 → 三级审计 → 导出训练集    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  Evidence-SFT 训练 (training/supervised_finetuning.py) │
│  Qwen2.5-7B + LoRA → 学会结构化证据输出            │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  DPO 偏好优化 (process_dpo/)                      │
│  来源选择 → 候选生成 → 构造 pair → 双向 Judge       │
│  → 严格导出 → DPO 训练                             │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│  评估 (cmb_eval/ + process_sft/evaluate_evidence.py) │
│  Evidence 指标 / CEval / CMB-Exam / CMB-Clin      │
└──────────────────────────────────────────────────┘
```

## 目录结构

```
Evidence-Med/
├── process_sft/              # Evidence-SFT 数据构造流水线
│   ├── evidence_sft_common.py        # 共享工具库
│   ├── prepare_evidence_candidates.py # Step 0: 筛选高质量病例
│   ├── build_evidence_sft.py         # Step 1: Teacher LLM 蒸馏
│   ├── validate_evidence_sft.py      # Step 2: 三级审计 (accepted/review/rejected)
│   ├── evaluate_evidence.py          # Step 4: 测试集推理 + 自动评分
│   └── run_evidence_sft.sh           # Step 3: LoRA 微调启动脚本
│
├── process_dpo/              # DPO 偏好训练流水线
│   ├── dpo_common.py                 # 共享工具库
│   ├── select_dpo_sources.py         # 来源选择
│   ├── generate_sft_candidates.py    # 模型候选生成
│   ├── build_dpo_pairs.py            # 构造严格 Answer-level pair
│   ├── judge_dpo_pairs.py            # 来源盲化 Judge
│   ├── judge_dpo_swaps.py            # A/B 交换复审
│   ├── reconcile_dpo_judgments.py    # 双向裁决映射
│   ├── export_dpo_dataset.py         # 严格导出训练数据
│   ├── validate_dpo_export.py        # 导出验证
│   ├── prepare_dpo_start.py          # 合并 SFT adapter → D0
│   └── train_dpo.py                  # DPO 训练
│
├── training/                 # 训练脚本
│   ├── supervised_finetuning.py      # SFT 训练器
│   ├── dpo_training.py               # DPO 训练器
│   ├── reward_modeling.py            # 奖励模型训练
│   └── grpo_training.py              # GRPO 训练
│
├── cmb_eval/                 # CMB 医学评测套件
│   ├── run_exam_four_models.sh       # CMB-Exam 四模型对比
│   ├── run_clin_dpo_effect.sh        # CMB-Clin DPO 效果评测
│   ├── generate_exam.py / score_exam.py / compare_exam.py
│   ├── generate_clin.py / judge_clin.py / aggregate_clin_dpo.py
│   └── model_runner.py / cmb_utils.py
│
├── tools/                    # 工具脚本
│   ├── merge_peft_adapter.py         # 合并 LoRA adapter
│   ├── model_quant.py / eval_quantize.py # 量化
│   ├── build_domain_tokenizer.py     # 领域 tokenizer
│   └── validate_jsonl.py             # JSONL 校验
│
├── scripts/                  # 辅助脚本
│   ├── lm_eval.sh / lm_eval_local.sh # CEval 评测
│   ├── run_sft.sh / run_dpo.sh       # 训练启动
│   ├── vllm_deployment.sh            # vLLM 部署
│   └── rewrite_answers.py            # 答案改写
│
├── tests/                    # 测试用例
│   ├── test_evidence_sft_pipeline.py
│   ├── test_dpo_pipeline.py
│   └── test_evidence_mask_pipeline.py
│
├── docs/                     # 文档
│   ├── analysis.md                   # 训练与评测分析报告
│   ├── 后续实验指导.md                 # 实验路线说明
│   └── prompt改进.md                 # Prompt 迭代记录
│
├── data/                     # 数据集 (gitignored)
├── outputs/                  # 模型权重 (gitignored)
└── results/                  # C-eval评测结果 (gitignored)
```

## Evidence-SFT 输出结构

模型学习输出的结构化 JSON：

```json
{
  "task_type": "diagnostic_reasoning",
  "query_intent": ["评估持续干咳的可能病因"],
  "evidence_sufficiency": "partial",
  "evidence": [
    {
      "id": "E1",
      "span": "感冒好了以后",
      "importance": "supporting",
      "role": "提供起病背景"
    },
    {
      "id": "E2",
      "span": "一直干咳，有两个月了",
      "importance": "critical",
      "role": "核心症状，决定诊断方向"
    }
  ],
  "critical_evidence_ids": ["E2"],
  "missing_information": ["胸部影像学检查", "肺功能检查"],
  "clinical_reasoning": "已有证据支持慢性咳嗽需要进一步评估...",
  "final_answer": "根据您描述的情况..."
}
```

## 训练路线

### 路线 1: Direct Evidence-SFT

```
Qwen2.5-7B-Instruct → Evidence-SFT LoRA  → DPO → D1
```

### 路线 2: Two-stage Evidence-SFT

```
Qwen2.5-7B-Instruct → Full-SFT LoRA → Evidence-SFT → DPO → D1
```

两条路线使用相同 Evidence 数据，区别在于初始化方式。实验表明 Direct 路线成本更低且指标差距很小。

## 评测体系

### Evidence 指标 (held-out test, 444 条)

| 指标 | 说明 |
|------|------|
| JSON 合法率 | strict / recoverable 两种解析模式 |
| Schema 合法率 | 字段完整性、枚举值正确性 |
| Evidence F1 | 预测 span 与 gold span 精确匹配 |
| Critical F1 | 关键证据 span 精确匹配 |
| Grounding 率 | 预测 evidence 是否出现在原文中 |
| 任务分类准确率 | task_type / evidence_sufficiency 混淆矩阵 |

### CEval 医学选择题 (lm_eval, 5-shot)

覆盖 basic_medicine、clinical_medicine、physician 三个子任务。

### CMB 评测 (cmb_eval/)

- **CMB-Exam**: 11,200 道选择题，四模型 (Base/Full-SFT/Evidence-SFT/DPO) 知识能力全景
- **CMB-Clin**: 74 个病例、208 个多轮问题，Evidence-SFT vs DPO 临床回答质量对比

### DPO A/B 评测

双向匿名 Judge，交换 A/B 位置各评一次，只有两次映射到同一模型结论才计入一致结果。

## Prompt 迭代

经过三轮 Error-driven 优化（详见 [docs/prompt改进.md](docs/prompt改进.md)）：

| 版本 | 解决的问题 | 关键改进 |
|------|-----------|---------|
| V1.1→V2.0 | sufficiency 定义过宽 | 加 `partial`；加 `critical/supporting` |
| V2.0→V2.1 | 问题混入 evidence；critical 偏多 | 独立 `query_intent`；反事实删除原则收紧 critical |
| V2.1→V2.2 | evidence 粒度太粗 | Atomic Evidence + Minimal Critical Span |

## 快速开始

### 1. 环境

```bash
pip install torch transformers peft trlm datasets openai tqdm
```

### 2. 准备数据

```bash
# 筛选候选病例
python process_sft/prepare_evidence_candidates.py \
  --input data/sft/medical_100k.jsonl \
  --output data/evidence_sft/00_candidates.jsonl \
  --target-size 5000

# Teacher 蒸馏
python process_sft/build_evidence_sft.py \
  --input data/evidence_sft/00_candidates.jsonl \
  --output data/evidence_sft/01_teacher_raw.jsonl

# 验证与导出
python process_sft/validate_evidence_sft.py \
  --input data/evidence_sft/01_teacher_raw.jsonl \
  --candidates data/evidence_sft/00_candidates.jsonl \
  --output-dir data/evidence_sft/validated_v2_2
```

### 3. 训练

```bash
# Evidence-SFT
./process_sft/run_evidence_sft.sh

# DPO (详见 process_dpo/README.md)
python process_dpo/prepare_dpo_start.py ...
python process_dpo/export_dpo_dataset.py ...
python training/dpo_training.py ...
```

### 4. 评估

```bash
# Evidence 评估
python process_sft/evaluate_evidence.py --adapter outputs/evidence-sft-frombase

# CEval
bash scripts/lm_eval.sh

# CMB
bash cmb_eval/run_exam_four_models.sh results/exam
bash cmb_eval/run_clin_dpo_effect.sh results/clin
```

## 文档

- [docs/analysis.md](docs/analysis.md) — 完整训练与评测分析报告
- [docs/后续实验指导.md](docs/后续实验指导.md) — 后续实验路线（Evidence Mask、DPO、偏好优化）
- [docs/prompt改进.md](docs/prompt改进.md) — Prompt 迭代记录
- [process_sft/README.md](process_sft/README.md) — Evidence-SFT 流水线详解
- [process_dpo/README.md](process_dpo/README.md) — DPO 流水线详解
- [cmb_eval/README.md](cmb_eval/README.md) — CMB 评测套件说明

## 环境变量

Teacher / Judge API 配置放在项目根目录 `.teacher_env`（已 gitignore）：

```bash
TEACHER_MODEL=模型名
TEACHER_BASE_URL=https://接口地址/v1
TEACHER_API_KEY=密钥
```

## License

本项目仅用于学术研究。医疗数据和模型输出不构成临床建议。
