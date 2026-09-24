-- DuckDB SQL reference for the native tables and charts in the DPO report.
-- All paths are relative to the workspace root (/home/medgpt).

-- 1. Final export size and pair-type mix.
WITH export_stats AS (
  SELECT *
  FROM read_json_auto('data/dpo/answer_v1/04_export.stats.json')
)
SELECT
  exported.train AS train_pairs,
  exported.validation AS validation_pairs,
  pair_types.train.target_vs_model AS train_target_vs_model,
  pair_types.train.model_vs_model AS train_model_vs_model,
  pair_types.train.controlled_negative AS train_controlled_negative
FROM export_stats;

-- 2. Formal DPO checkpoint metrics.
WITH state AS (
  SELECT *
  FROM read_json_auto('outputs/evidence-dpo-answer-v1/trainer_state.json')
), history AS (
  SELECT entry
  FROM state, UNNEST(log_history) AS t(entry)
)
SELECT
  entry.step AS step,
  entry.eval_loss AS eval_loss,
  entry."eval_rewards/accuracies" AS preference_accuracy,
  entry."eval_rewards/margins" AS reward_margin
FROM history
WHERE entry.eval_loss IS NOT NULL
ORDER BY step;

-- 3. Fair held-out comparison: merged D0 start versus DPO D1.
WITH d0 AS (
  SELECT * FROM read_json_auto('results/evidence_d0_merged_control_eval/metrics.json')
), d1 AS (
  SELECT * FROM read_json_auto('results/evidence_dpo_answer_eval/metrics.json')
)
SELECT * FROM (
  VALUES
    ('Strict JSON', d0.sample_level.strict_json_valid_rate, d1.sample_level.strict_json_valid_rate),
    ('Schema valid', d0.sample_level.schema_valid_rate, d1.sample_level.schema_valid_rate),
    ('Sample fully grounded', d0.sample_level.all_evidence_grounded_rate, d1.sample_level.all_evidence_grounded_rate),
    ('Micro grounding', d0.micro_evidence.grounding_rate, d1.micro_evidence.grounding_rate),
    ('Evidence exact F1', d0.micro_evidence.teacher_exact_f1, d1.micro_evidence.teacher_exact_f1),
    ('Critical span F1', d0.micro_critical_span.teacher_exact_f1, d1.micro_critical_span.teacher_exact_f1),
    ('Task type', d0.sample_level.task_type_correct_rate, d1.sample_level.task_type_correct_rate),
    ('Sufficiency', d0.sample_level.sufficiency_correct_rate, d1.sample_level.sufficiency_correct_rate)
) AS metrics(metric, merged_d0, dpo_d1);

-- 4. Paired C-Eval comparison.
SELECT *
FROM read_json_auto('results/dpo_ceval_comparison.json');
