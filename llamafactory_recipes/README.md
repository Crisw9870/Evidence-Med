# Evidence-SFT and Answer-level DPO with LLaMA-Factory

这个目录提供与项目现有 Evidence-SFT、Answer-level DPO 实验对齐的 LLaMA-Factory recipes。
它们可以替代两段训练脚本的启动入口，但不会删除原脚本，也不会自动启动训练。

## 文件

- `data/dataset_info.json`：注册现有 ShareGPT JSONL；不复制训练数据。
- `evidence_sft_lora.yaml`：Qwen2.5-7B-Instruct 的 LoRA-SFT 配置。
- `evidence_dpo_lora.yaml`：从合并后的 Evidence-SFT checkpoint 开始训练新的 DPO LoRA。

数据仍然来自：

- `data/evidence_sft/train/train.jsonl`
- `data/evidence_sft/validation/validation.jsonl`
- `data/dpo/answer_v1/train/train.jsonl`
- `data/dpo/answer_v1/validation/validation.jsonl`

每条数据的 `conversations` 使用 `human/gpt` 角色；`source_id`、`split`、
`schema_version` 等审计字段不会映射为模型输入。

DPO JSONL 中的 `prompt`、`chosen`、`rejected` 都是字符串，所以在注册表中按
Alpaca preference 格式映射；`ranking: true` 告诉 LLaMA-Factory 这是偏好对。

## 环境假设

先单独安装 LLaMA-Factory，并确保 `llamafactory-cli` 可用。建议固定实际使用的
LLaMA-Factory、Transformers、PEFT 和 PyTorch 版本，再保存到实验 manifest。

本配置假设从项目根目录执行，因此模型、数据和输出路径均相对于
`D:/DeepLearning/Evidence-Med`。

## 查看配置但不训练

```powershell
Get-Content .\llamafactory_recipes\evidence_sft_lora.yaml
Get-Content .\llamafactory_recipes\evidence_dpo_lora.yaml
Get-Content .\llamafactory_recipes\data\dataset_info.json
```

## 真正执行时的命令

当前不要执行；以后确认环境和版本后，从项目根目录运行：

```powershell
llamafactory-cli train .\llamafactory_recipes\evidence_sft_lora.yaml
```

DPO 不能直接从原始 Qwen base 开始。先把 Evidence-SFT LoRA 合并成固定 D0：

```powershell
& 'D:\miniconda3\envs\medgpt\python.exe' .\process_dpo\prepare_dpo_start.py `
  --base-model .\Qwen\Qwen2.5-7B-Instruct `
  --sft-adapter .\outputs\evidence-sft-frombase `
  --output .\outputs\dpo-start-direct-merged `
  --torch-dtype bfloat16 `
  --device-map cuda:0
```

再启动 DPO：

```powershell
llamafactory-cli train .\llamafactory_recipes\evidence_dpo_lora.yaml
```

Linux 下对应为：

```bash
llamafactory-cli train ./llamafactory_recipes/evidence_sft_lora.yaml
```

```bash
llamafactory-cli train ./llamafactory_recipes/evidence_dpo_lora.yaml
```

## 配置与原实验的对应关系

| 项目 | 配置 |
| --- | --- |
| Base model | Qwen2.5-7B-Instruct |
| Training | LoRA-SFT |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA target | all linear modules |
| Max sequence length | 1536 |
| Per-device batch | 8 |
| Gradient accumulation | 4 |
| Effective single-GPU batch | 32 |
| Epochs | 2 |
| Learning rate | 1e-5 |
| Warmup ratio | 0.03 |
| Precision | BF16 |
| Gradient checkpointing | enabled |
| Seed | 42 |

## DPO 配置

| 项目 | 配置 |
| --- | --- |
| DPO start / reference | 合并 Evidence-SFT 后的固定 D0 |
| Training | Answer-level LoRA-DPO |
| Preference loss / beta | sigmoid / 0.1 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Max sequence length | 1536 |
| Per-device batch | 2 |
| Gradient accumulation | 8 |
| Effective single-GPU pair batch | 16 |
| Epochs | 1 |
| Learning rate | 5e-6 |
| Precision | BF16 |

DPO 的 policy 是 `D0 + 新 DPO LoRA`。训练器计算 reference 时禁用这层新 LoRA，
因此 reference 回到固定 D0。先合并 SFT 再创建 DPO LoRA，是为了避免 reference
意外退回原始 Qwen base。

## 训练前必须检查

1. `llamafactory-cli` 版本是否接受当前 YAML 字段。
2. 数据预览中 `human` 被映射为 user、`gpt` 被映射为 assistant。
3. user prompt 不参与 loss，assistant 的完整 Evidence JSON 参与 loss。
4. 随机解码若干 tokenized 样本，确认 `template: qwen` 与推理一致。
5. 先跑少量样本 smoke test，再进行正式训练。

DPO 还要额外检查：

1. 每条记录都有非空且不相同的 `chosen/rejected`。
2. `chosen/rejected` 只在 `clinical_reasoning` 和 `final_answer` 上存在目标差异。
3. DPO 的起点是 `outputs/dpo-start-direct-merged`，不是原始 Qwen。
4. 先用 `max_samples` 或单独的小数据文件做 smoke，再从 D0 重新跑正式实验。

LLaMA-Factory 只替代通用训练执行层。Teacher 蒸馏、Evidence 审计、数据划分、
held-out 评测和 Answer-level DPO 数据构造仍使用项目现有流水线。
