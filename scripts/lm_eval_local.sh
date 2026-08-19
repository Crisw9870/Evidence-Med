#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."


lm_eval \
	--model hf \
	--model_args pretrained=Qwen/Qwen2.5-7B-Instruct \
	--tasks ceval-test_basic_medicine,ceval-test_clinical_medicine,ceval-test_physician \
	--include_path ./data/lm_eval_tasks \
	--num_fewshot 5 \
	--batch_size 8 \
	--output_path ./results/base_ceval.json

# 去掉lm_eval自动添加的时间戳，重命名为固定文件名
latest=$(ls -t ./results/base_ceval_*.json 2>/dev/null | head -1)
if [ -n "$latest" ]; then
    mv "$latest" ./results/base_ceval.json
    echo "结果已保存至: ./results/base_ceval.json"
fi