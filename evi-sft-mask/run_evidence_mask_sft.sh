#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/medgpt/bin/python}"
BASE_MODEL="${BASE_MODEL:-$PROJECT_ROOT/Qwen/Qwen2.5-7B-Instruct}"
PARENT_ADAPTER="${PARENT_ADAPTER:-$PROJECT_ROOT/outputs/evidence-sft-frombase}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/evidence_mask/v1/validated}"
ARM="${ARM:-m1}"
SEED="${SEED:-42}"

if [[ "$ARM" != "m0" && "$ARM" != "m1" ]]; then
    echo "ARM must be m0 or m1" >&2
    exit 2
fi
if [[ ! -f "$DATA_ROOT/$ARM/train.jsonl" || ! -f "$DATA_ROOT/$ARM/validation.jsonl" ]]; then
    echo "Missing validated $ARM manifests under $DATA_ROOT" >&2
    exit 2
fi
if [[ ! -f "$PARENT_ADAPTER/adapter_config.json" ]]; then
    echo "Missing parent adapter: $PARENT_ADAPTER" >&2
    exit 2
fi

STAGE_DIR="$SCRIPT_DIR/.staged_mask_${ARM}_seed${SEED}"
mkdir -p "$STAGE_DIR/train" "$STAGE_DIR/validation"
ln -sfn "$DATA_ROOT/$ARM/train.jsonl" "$STAGE_DIR/train/train.jsonl"
ln -sfn "$DATA_ROOT/$ARM/validation.jsonl" "$STAGE_DIR/validation/validation.jsonl"

OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/evidence-mask-${ARM}-seed${SEED}}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$PYTHON_BIN" \
    "$PROJECT_ROOT/training/supervised_finetuning.py" \
    --model_name_or_path "$BASE_MODEL" \
    --peft_path "$PARENT_ADAPTER" \
    --train_file_dir "$STAGE_DIR/train" \
    --validation_file_dir "$STAGE_DIR/validation" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --do_train \
    --do_eval \
    --use_peft True \
    --model_max_length 1536 \
    --num_train_epochs 1 \
    --learning_rate 5e-6 \
    --warmup_ratio 0.03 \
    --weight_decay 0.01 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy steps \
    --eval_steps 40 \
    --save_strategy steps \
    --save_steps 40 \
    --save_total_limit 2 \
    --preprocessing_num_workers 6 \
    --output_dir "$OUTPUT_DIR" \
    --logging_first_step True \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --seed "$SEED" \
    --data_seed "$SEED" \
    --cache_dir "$PROJECT_ROOT/cache"
