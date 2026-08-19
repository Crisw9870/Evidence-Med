#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/medgpt/bin/python \
    "$PROJECT_ROOT/training/supervised_finetuning.py" \
    --model_name_or_path "$PROJECT_ROOT/Qwen/Qwen2.5-7B-Instruct" \
    --train_file_dir "$PROJECT_ROOT/data/evidence_sft/train" \
    --validation_file_dir "$PROJECT_ROOT/data/evidence_sft/validation" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --model_max_length 1536 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 2 \
    --preprocessing_num_workers 6 \
    --output_dir "$PROJECT_ROOT/outputs/evidence-sft-frombase" \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --seed 42 \
    --cache_dir "$PROJECT_ROOT/cache"
