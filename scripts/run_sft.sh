#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/../training/supervised_finetuning.py" \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --train_file_dir ./data/test_sft/train \
    --validation_file_dir ./data/test_sft/validation \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples 250000 \
    --max_eval_samples 500 \
    --model_max_length 1024 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_steps 2 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 200 \
    --eval_strategy steps \
    --save_steps 500 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 6 \
    --output_dir ./outputs/test-sft-v1 \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --cache_dir ./cache
