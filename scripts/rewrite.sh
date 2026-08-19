#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/rewrite_answers.py \
    --input ./data/sft_retrieval/retrieval_sft.jsonl \
    --output ./data/sft_retrieval/retrieval_sft_rewritten.jsonl \
    --api_key "${TEACHER_API_KEY:-${OPENAI_API_KEY:-}}" \
    --base_url https://opencode.ai/zen/v1 \
    --model deepseek-v4-flash-free \
    --min_rewrite_len 100
