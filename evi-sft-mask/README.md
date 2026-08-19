# Evidence-SFT training

在现有 `outputs/sft-base` LoRA adapter 上继续进行 Evidence-SFT，只使用 accepted 的 train 和 validation 数据。

## 启动训练

```bash
cd /home/medgpt
./evi-sft-traing/run_evidence_sft.sh
```

默认配置：

- train：7,671 条；
- validation：880 条；
- batch size：2；
- gradient accumulation：16；
- epochs：2；
- learning rate：`1e-5`；
- context length：1536；
- 输出目录：`outputs/evidence-sft-v2-2`。

脚本创建两个数据链接，是为了让现有训练器只读取 train 和 validation，避免递归读入 test、review、rejected 等 JSONL 文件。
