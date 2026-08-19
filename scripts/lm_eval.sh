#!/usr/bin/env bash
# 评测 Retrieval SFT 模型在 C-Eval 医学任务上的表现
set -euo pipefail

cd "$(dirname "$0")/.."


lm_eval \
	--model hf \
	--model_args pretrained=Qwen/Qwen2.5-7B-Instruct,peft=outputs/sft-base,dtype=bfloat16,trust_remote_code=True \
	--tasks ceval-test_basic_medicine,ceval-test_clinical_medicine,ceval-test_physician \
	--include_path ./data/lm_eval_tasks \
	--num_fewshot 5 \
	--batch_size 8 \
	--output_path ./results/sft_ceval.json

# 去掉lm_eval自动添加的时间戳，重命名为固定文件名
latest=$(ls -t ./results/sft_ceval_*.json 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    mv "$latest" ./results/sft_ceval.json
    echo "结果已保存至: ./results/sft_ceval.json"
fi
