#!/usr/bin/env bash
# 四模型 CMB-Exam 正式评测流水线
# 模型：Base / Full-SFT / Evidence-SFT / Evidence-DPO
# 输出：四模型 predictions + accuracy + 四组配对比较 + summary 总表
set -euo pipefail

# ── 路径变量 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"                     # /home/medgpt
OUTPUT_ROOT="${1:-$SCRIPT_DIR/results/four_models_test}"       # 可选第 1 参数覆盖输出目录

# ── 模型与超参（均可通过环境变量覆盖）─────────────────────
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/medgpt/bin/python}"
BASE_MODEL="${BASE_MODEL:-$WORKSPACE/Qwen/Qwen2.5-7B-Instruct}"               # Qwen 基座
FULL_SFT_ADAPTER="${FULL_SFT_ADAPTER:-$WORKSPACE/outputs/sft-base}"            # 全量 SFT LoRA
EVIDENCE_SFT_ADAPTER="${EVIDENCE_SFT_ADAPTER:-$WORKSPACE/outputs/evidence-sft-frombase}"  # 证据增强 SFT LoRA
EVIDENCE_DPO_ADAPTER="${EVIDENCE_DPO_ADAPTER:-$WORKSPACE/outputs/evidence-dpo-answer-v1}" # DPO LoRA（叠加在 Evidence-SFT 上）
BATCH_SIZE="${BATCH_SIZE:-64}"                                # 批量推理大小
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"                        # 每题最多生成 token 数（选择题只需几个）

# ── 前置文件检查：确保四个模型权重都存在 ──────────────────
for required_path in \
  "$BASE_MODEL/config.json" \
  "$FULL_SFT_ADAPTER/adapter_config.json" \
  "$EVIDENCE_SFT_ADAPTER/adapter_config.json" \
  "$EVIDENCE_DPO_ADAPTER/adapter_config.json"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Required model file not found: $required_path" >&2
    exit 1
  fi
done

# ── Step 1: 数据校验 ──────────────────────────────────────
"$PYTHON_BIN" "$SCRIPT_DIR/validate_exam_test.py" \
  --output "$OUTPUT_ROOT/test_data_validation.json"

# ── Step 2–3: Base 模型（裸基座，无适配器）─────────────────
"$PYTHON_BIN" "$SCRIPT_DIR/generate_exam.py" \
  --model "$BASE_MODEL" \
  --tokenizer "$BASE_MODEL" \
  --model-label Base \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output "$OUTPUT_ROOT/base/predictions.jsonl" \
  --resume

"$PYTHON_BIN" "$SCRIPT_DIR/score_exam.py" \
  --predictions "$OUTPUT_ROOT/base/predictions.jsonl" \
  --output-dir "$OUTPUT_ROOT/base"

# ── Step 4–5: Full-SFT 模型（Qwen + 全量 SFT LoRA）────────
"$PYTHON_BIN" "$SCRIPT_DIR/generate_exam.py" \
  --model "$BASE_MODEL" \
  --adapter "$FULL_SFT_ADAPTER" \
  --tokenizer "$FULL_SFT_ADAPTER" \
  --model-label Full-SFT \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output "$OUTPUT_ROOT/full_sft/predictions.jsonl" \
  --resume

"$PYTHON_BIN" "$SCRIPT_DIR/score_exam.py" \
  --predictions "$OUTPUT_ROOT/full_sft/predictions.jsonl" \
  --output-dir "$OUTPUT_ROOT/full_sft"

# ── Step 6–7: Evidence-SFT 模型（Qwen + 证据增强 SFT LoRA）──
"$PYTHON_BIN" "$SCRIPT_DIR/generate_exam.py" \
  --model "$BASE_MODEL" \
  --adapter "$EVIDENCE_SFT_ADAPTER" \
  --tokenizer "$EVIDENCE_SFT_ADAPTER" \
  --model-label Evidence-SFT \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output "$OUTPUT_ROOT/evidence_sft/predictions.jsonl" \
  --resume

"$PYTHON_BIN" "$SCRIPT_DIR/score_exam.py" \
  --predictions "$OUTPUT_ROOT/evidence_sft/predictions.jsonl" \
  --output-dir "$OUTPUT_ROOT/evidence_sft"

# ── Step 8–9: Evidence-DPO 模型（Qwen + Evidence-SFT LoRA + DPO LoRA 叠加）──
"$PYTHON_BIN" "$SCRIPT_DIR/generate_exam.py" \
  --model "$BASE_MODEL" \
  --adapter "$EVIDENCE_SFT_ADAPTER" \
  --additional-adapter "$EVIDENCE_DPO_ADAPTER" \
  --tokenizer "$EVIDENCE_SFT_ADAPTER" \
  --model-label Evidence-DPO \
  --batch-size "$BATCH_SIZE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --output "$OUTPUT_ROOT/evidence_dpo/predictions.jsonl" \
  --resume

"$PYTHON_BIN" "$SCRIPT_DIR/score_exam.py" \
  --predictions "$OUTPUT_ROOT/evidence_dpo/predictions.jsonl" \
  --output-dir "$OUTPUT_ROOT/evidence_dpo"

# ── Step 10: 四组配对比较（McNemar 检验）──────────────────
# 比较 1: Base vs Full-SFT → SFT 训练是否有效？
"$PYTHON_BIN" "$SCRIPT_DIR/compare_exam.py" \
  --baseline "$OUTPUT_ROOT/base/scored_items.jsonl" \
  --candidate "$OUTPUT_ROOT/full_sft/scored_items.jsonl" \
  --baseline-label Base \
  --candidate-label Full-SFT \
  --output "$OUTPUT_ROOT/base_vs_full_sft.json"

# 比较 2: Base vs Evidence-SFT → 证据增强 SFT 是否有效？
"$PYTHON_BIN" "$SCRIPT_DIR/compare_exam.py" \
  --baseline "$OUTPUT_ROOT/base/scored_items.jsonl" \
  --candidate "$OUTPUT_ROOT/evidence_sft/scored_items.jsonl" \
  --baseline-label Base \
  --candidate-label Evidence-SFT \
  --output "$OUTPUT_ROOT/base_vs_evidence_sft.json"

# 比较 3: Evidence-SFT vs Evidence-DPO → DPO 对齐是否进一步提升？
"$PYTHON_BIN" "$SCRIPT_DIR/compare_exam.py" \
  --baseline "$OUTPUT_ROOT/evidence_sft/scored_items.jsonl" \
  --candidate "$OUTPUT_ROOT/evidence_dpo/scored_items.jsonl" \
  --baseline-label Evidence-SFT \
  --candidate-label Evidence-DPO \
  --output "$OUTPUT_ROOT/evidence_sft_vs_dpo.json"

# 比较 4: Base vs Evidence-DPO → 最终模型相比原始基座提升多少？
"$PYTHON_BIN" "$SCRIPT_DIR/compare_exam.py" \
  --baseline "$OUTPUT_ROOT/base/scored_items.jsonl" \
  --candidate "$OUTPUT_ROOT/evidence_dpo/scored_items.jsonl" \
  --baseline-label Base \
  --candidate-label Evidence-DPO \
  --output "$OUTPUT_ROOT/base_vs_evidence_dpo.json"

# ── Step 11: 汇总四模型总表（JSON + CSV）─────────────────
"$PYTHON_BIN" "$SCRIPT_DIR/summarize_exam_suite.py" \
  --run "Base=$OUTPUT_ROOT/base/metrics.json" \
  --run "Full-SFT=$OUTPUT_ROOT/full_sft/metrics.json" \
  --run "Evidence-SFT=$OUTPUT_ROOT/evidence_sft/metrics.json" \
  --run "Evidence-DPO=$OUTPUT_ROOT/evidence_dpo/metrics.json" \
  --output-json "$OUTPUT_ROOT/summary.json" \
  --output-csv "$OUTPUT_ROOT/summary.csv"

echo "Four-model CMB-test evaluation saved under $OUTPUT_ROOT"
