#!/usr/bin/env bash
# CMB-Clin DPO 效果评测流水线
# 对比 Evidence-SFT vs Evidence-DPO 在临床病例多轮问答上的表现
# 流程：生成回答 → LLM Judge 盲评 → 汇总胜负与 bootstrap 置信区间
set -euo pipefail

# ── 路径变量 ──────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE="$(cd "$SCRIPT_DIR/.." && pwd)"                     # /home/medgpt
OUTPUT_ROOT="${1:-$SCRIPT_DIR/results/clin_dpo}"   # 可选第 1 参数覆盖输出目录

# ── 模型与超参（均可通过环境变量覆盖）─────────────────────
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/envs/medgpt/bin/python}"
BASE_MODEL="${BASE_MODEL:-$WORKSPACE/Qwen/Qwen2.5-7B-Instruct}"               # Qwen 基座
EVIDENCE_SFT_ADAPTER="${EVIDENCE_SFT_ADAPTER:-$WORKSPACE/outputs/evidence-sft-frombase}"  # 证据增强 SFT LoRA
EVIDENCE_DPO_ADAPTER="${EVIDENCE_DPO_ADAPTER:-$WORKSPACE/outputs/evidence-dpo-answer-v1}" # DPO LoRA（叠加在 Evidence-SFT 上）
CLIN_MAX_NEW_TOKENS="${CLIN_MAX_NEW_TOKENS:-640}"              # 覆盖最长参考答案，并给正常结束留余量
CLIN_LIMIT_CASES="${CLIN_LIMIT_CASES:-0}"                      # 限制病例数（0=全部，>0 用于 smoke test）
DEFAULT_CLIN_SYSTEM_PROMPT="你是一名严谨的临床医生。请只根据给定病例回答当前问题，结论优先、分点作答。严格只输出本轮问题明确要求的部分：本轮未询问的鉴别诊断、治疗、建议或总结不得输出。不要复述病例，不虚构病例中未提供的既有症状、检查结果或治疗经过；核对患者性别及异常检验值，不得列出与患者性别矛盾的疾病。答案应简洁完整，通常不超过600个汉字。"
CLIN_SYSTEM_PROMPT="${CLIN_SYSTEM_PROMPT-$DEFAULT_CLIN_SYSTEM_PROMPT}" # 显式设为空字符串可关闭

# ── Judge 配置 ─────────────────────────────────────────────
RUN_JUDGE="${RUN_JUDGE:-0}"                                    # 是否运行 Judge（0=只生成回答，1=生成+评审）
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"                            # Judge 并发线程数
JUDGE_LIMIT="${JUDGE_LIMIT:-0}"                                # 限制评审条数（0=全部）
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-5000}"                     # bootstrap 重采样次数（用于置信区间）
SEED="${SEED:-42}"                                             # 随机种子（保证可复现）

# ── Step 1: 生成 Evidence-SFT 回答 ─────────────────────────
"$PYTHON_BIN" "$SCRIPT_DIR/generate_clin.py" \
  --model "$BASE_MODEL" \
  --adapter "$EVIDENCE_SFT_ADAPTER" \
  --tokenizer "$EVIDENCE_SFT_ADAPTER" \
  --model-label Evidence-SFT \
  --system-prompt "$CLIN_SYSTEM_PROMPT" \
  --max-new-tokens "$CLIN_MAX_NEW_TOKENS" \
  --limit-cases "$CLIN_LIMIT_CASES" \
  --output "$OUTPUT_ROOT/evidence_sft/predictions.jsonl" \
  --resume

# ── Step 2: 生成 Evidence-DPO 回答（SFT LoRA + DPO LoRA 叠加）──
"$PYTHON_BIN" "$SCRIPT_DIR/generate_clin.py" \
  --model "$BASE_MODEL" \
  --adapter "$EVIDENCE_SFT_ADAPTER" \
  --additional-adapter "$EVIDENCE_DPO_ADAPTER" \
  --tokenizer "$EVIDENCE_SFT_ADAPTER" \
  --model-label Evidence-DPO \
  --system-prompt "$CLIN_SYSTEM_PROMPT" \
  --max-new-tokens "$CLIN_MAX_NEW_TOKENS" \
  --limit-cases "$CLIN_LIMIT_CASES" \
  --output "$OUTPUT_ROOT/evidence_dpo/predictions.jsonl" \
  --resume

# ── Judge 开关：默认只生成回答，需显式开启评审 ────────────
if [[ "$RUN_JUDGE" != "1" ]]; then
  echo "Primary CMB-Clin DPO generation complete: $OUTPUT_ROOT"
  echo "Set RUN_JUDGE=1 after configuring the fixed Judge model/API."
  exit 0
fi

# ── Step 3: LLM Judge 盲评（匿名 A/B + 交换位置验证）──────
"$PYTHON_BIN" "$SCRIPT_DIR/judge_clin.py" \
  --baseline-predictions "$OUTPUT_ROOT/evidence_sft/predictions.jsonl" \
  --candidate-predictions "$OUTPUT_ROOT/evidence_dpo/predictions.jsonl" \
  --baseline-label Evidence-SFT \
  --candidate-label Evidence-DPO \
  --output "$OUTPUT_ROOT/judgments.jsonl" \
  --workers "$JUDGE_WORKERS" \
  --limit "$JUDGE_LIMIT" \
  --seed "$SEED" \
  --resume

# ── Step 4: 汇总胜负、四维分数及 bootstrap 置信区间 ───────
"$PYTHON_BIN" "$SCRIPT_DIR/aggregate_clin_dpo.py" \
  --judgments "$OUTPUT_ROOT/judgments.jsonl" \
  --baseline-predictions "$OUTPUT_ROOT/evidence_sft/predictions.jsonl" \
  --candidate-predictions "$OUTPUT_ROOT/evidence_dpo/predictions.jsonl" \
  --bootstrap-iters "$BOOTSTRAP_ITERS" \
  --seed "$SEED" \
  --output "$OUTPUT_ROOT/evidence_sft_vs_dpo_summary.json"

echo "Primary CMB-Clin DPO comparison saved under $OUTPUT_ROOT"
