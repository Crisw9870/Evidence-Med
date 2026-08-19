#!/usr/bin/env bash
# Retrieval-Augmented SFT 数据召回脚本
# 从 C-Eval dev 题目出发，用 BGE-M3 + FAISS 从 SFT 数据中召回最相关的训练样本

set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/retrieval_sft.py \
    --sample_size -1 \
    --top_k 10 \
    --batch_size 32 \
    --model_path ./bge-m3 \
    --ceval_dir ./data/ceval \
    --sft_dir ./data/sft \
    --output_dir ./data/sft_retrieval
