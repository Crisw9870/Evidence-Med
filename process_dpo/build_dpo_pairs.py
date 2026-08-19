#!/usr/bin/env python3
"""Build source-blind pairs whose completions differ only at answer level."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from dpo_common import (
    ANSWER_LEVEL_FIELDS,
    DPO_SCHEMA_VERSION,
    answer_level_differences,
    answer_level_signature,
    answer_structure_compatible,
    audit_rank,
    audit_response,
    compact_json,
    is_isolated_answer_pair,
    make_controlled_negative,
    project_answer_fields_onto_target,
    read_jsonl,
    shared_structure,
    stable_id,
    stable_int,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/dpo/answer_v1/00_sources.jsonl")
    parser.add_argument("--candidates", default="data/dpo/answer_v1/01_sft_candidates.jsonl")
    parser.add_argument("--output", default="data/dpo/answer_v1/02_pair_candidates.jsonl")
    parser.add_argument(
        "--stats-output", default="data/dpo/answer_v1/02_pair_candidates.stats.json"
    )
    parser.add_argument("--target-model-per-source", type=int, default=1)
    parser.add_argument("--model-model-per-source", type=int, default=1)
    parser.add_argument("--controlled-per-source", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _answer_view(value: dict[str, Any]) -> dict[str, Any]:
    return {field: value.get(field) for field in ANSWER_LEVEL_FIELDS}


def _target_candidate(source: dict[str, Any]) -> dict[str, Any]:
    target = source["target"]
    text = compact_json(target)
    audit = audit_response(source["case_text"], text, target)
    return {
        "candidate_id": "validated_target",
        "origin": "validated_target",
        "text": text,
        "answer_view": _answer_view(target),
        "audit": audit,
    }


def _normalized_model_candidate(
    source: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any] | None:
    raw_audit = candidate.get("audit")
    if not isinstance(raw_audit, dict):
        raw_audit = audit_response(
            source["case_text"], str(candidate.get("text", "")), source["target"]
        )
    parsed = raw_audit.get("parsed_output")
    if not isinstance(parsed, dict):
        return None

    try:
        projected = project_answer_fields_onto_target(source["target"], parsed)
    except ValueError:
        return None
    projected_audit = audit_response(source["case_text"], projected, source["target"])
    if projected_audit.get("schema_errors") or projected_audit.get("hard_failures"):
        return None
    return {
        "candidate_id": str(candidate.get("candidate_id", "model_candidate")),
        "origin": "sft_model",
        "text": compact_json(projected),
        "answer_view": _answer_view(projected),
        "audit": projected_audit,
        "raw_audit_summary": {
            "schema_errors": raw_audit.get("schema_errors", []),
            "hard_failures": raw_audit.get("hard_failures", []),
            "review_reasons": raw_audit.get("review_reasons", []),
            "warnings": raw_audit.get("warnings", []),
            "error_tags": raw_audit.get("error_tags", []),
            "metrics": raw_audit.get("metrics", {}),
            "strict_structure_compatible": answer_structure_compatible(
                parsed, source["target"]
            ),
        },
        "generation": candidate.get("generation", {}),
    }


def _answer_rank(candidate: dict[str, Any], target: dict[str, Any]) -> tuple[float, ...]:
    audit = candidate["audit"]
    warning_count = len(audit.get("warnings", [])) + len(audit.get("review_reasons", []))
    missing_match = candidate["answer_view"].get("missing_information") == target.get(
        "missing_information"
    )
    return (
        *audit_rank(audit),
        1.0 if warning_count == 0 else 0.0,
        float(missing_match),
        -float(warning_count),
    )


def _candidate_order_key(
    source: dict[str, Any],
    candidate: dict[str, Any],
    target: dict[str, Any],
    seed: int,
) -> tuple[float | int, ...]:
    """Rank candidates with a reproducible, source-specific random tie-break."""
    tie_break = stable_int(
        "answer_rank_tie",
        source["source_id"],
        candidate["candidate_id"],
        seed=seed,
    )
    return (*_answer_rank(candidate, target), tie_break)


def _make_pair(
    source: dict[str, Any],
    pair_type: str,
    first: dict[str, Any],
    second: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    first_parsed = first["audit"].get("parsed_output")
    second_parsed = second["audit"].get("parsed_output")
    if not isinstance(first_parsed, dict) or not isinstance(second_parsed, dict):
        raise ValueError("pair candidates must contain parsed projected outputs")
    if not is_isolated_answer_pair(first_parsed, second_parsed):
        raise ValueError("DPO pair is not isolated to answer-level fields")

    pair_id = stable_id(
        "pair",
        source["source_id"],
        pair_type,
        first["candidate_id"],
        second["candidate_id"],
        seed=seed,
    )
    if stable_int(pair_id, "blind_position", seed=seed) % 2:
        candidate_a, candidate_b = second, first
    else:
        candidate_a, candidate_b = first, second
    return {
        "schema_version": DPO_SCHEMA_VERSION,
        "preference_scope": "answer_level",
        "pair_id": pair_id,
        "pair_type": pair_type,
        "source_id": source["source_id"],
        "split": source["split"],
        "category": source.get("category", "未标注医学主题"),
        "case_text": source["case_text"],
        "prompt": source["prompt"],
        "shared_structure": shared_structure(source["target"]),
        "difference_fields": answer_level_differences(first_parsed, second_parsed),
        "candidate_A": candidate_a,
        "candidate_B": candidate_b,
    }


def build_pairs(
    sources: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    by_source = {row["source_id"]: row for row in candidate_rows}
    pairs: list[dict[str, Any]] = []
    for source in sources:
        row = by_source.get(source["source_id"])
        if not row:
            continue
        target = _target_candidate(source)
        normalized = [
            _normalized_model_candidate(source, item)
            for item in row.get("candidates", [])
            if isinstance(item, dict) and str(item.get("text", "")).strip()
        ]
        unique_models: dict[str, dict[str, Any]] = {}
        target_signature = answer_level_signature(source["target"])
        for item in normalized:
            if item is None:
                continue
            signature = answer_level_signature(item["answer_view"])
            if signature != target_signature:
                unique_models.setdefault(signature, item)
        valid_models = list(unique_models.values())
        valid_models.sort(
            key=lambda item: _candidate_order_key(
                source, item, source["target"], args.seed
            )
        )

        for candidate in valid_models[: max(0, args.target_model_per_source)]:
            pairs.append(_make_pair(source, "target_vs_model", target, candidate, args.seed))

        if len(valid_models) >= 2 and args.model_model_per_source > 0:
            best_to_worst = sorted(
                valid_models,
                key=lambda item: _candidate_order_key(
                    source, item, source["target"], args.seed
                ),
                reverse=True,
            )
            used: set[tuple[str, str]] = set()
            for index in range(args.model_model_per_source):
                best = best_to_worst[min(index, len(best_to_worst) - 1)]
                worst = best_to_worst[max(0, len(best_to_worst) - 1 - index)]
                key = tuple(sorted((best["candidate_id"], worst["candidate_id"])))
                if best["text"] == worst["text"] or key in used:
                    continue
                used.add(key)
                pairs.append(_make_pair(source, "model_vs_model", best, worst, args.seed))

        for index in range(max(0, args.controlled_per_source)):
            controlled = make_controlled_negative(
                source["target"], source["source_id"], seed=args.seed + index
            )
            controlled["audit"] = audit_response(
                source["case_text"], controlled["text"], source["target"]
            )
            controlled["answer_view"] = _answer_view(controlled["parsed_output"])
            if (
                not controlled["audit"].get("schema_errors")
                and not controlled["audit"].get("hard_failures")
                and is_isolated_answer_pair(
                    target["audit"]["parsed_output"], controlled["parsed_output"]
                )
            ):
                pairs.append(
                    _make_pair(
                        source,
                        "controlled_negative",
                        target,
                        controlled,
                        args.seed + index,
                    )
                )
    return pairs


def candidate_alignment_stats(
    sources: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    source_by_id = {row["source_id"]: row for row in sources}
    total = raw_schema_valid = strict_compatible = projectable = 0
    candidate_sources: set[str] = set()
    strict_sources: set[str] = set()
    projectable_signatures: dict[str, set[str]] = {}
    profile_counts: dict[str, Counter[str]] = {}
    rejection_reasons: Counter[str] = Counter()

    for row in candidate_rows:
        source = source_by_id.get(row.get("source_id"))
        if not source:
            continue
        source_id = source["source_id"]
        candidate_sources.add(source_id)
        projectable_signatures.setdefault(source_id, set())
        for candidate in row.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            total += 1
            profile = str(candidate.get("candidate_id", "unknown"))
            counts = profile_counts.setdefault(profile, Counter())
            counts["total"] += 1
            raw_audit = candidate.get("audit")
            if not isinstance(raw_audit, dict):
                raw_audit = audit_response(
                    source["case_text"],
                    str(candidate.get("text", "")),
                    source["target"],
                )
            parsed = raw_audit.get("parsed_output")
            if not isinstance(parsed, dict):
                rejection_reasons["unparseable_output"] += 1
                continue

            raw_valid = not raw_audit.get("schema_errors") and not raw_audit.get(
                "hard_failures"
            )
            if raw_valid:
                raw_schema_valid += 1
                counts["raw_schema_valid"] += 1
                if answer_structure_compatible(parsed, source["target"]):
                    strict_compatible += 1
                    counts["strict_structure_compatible"] += 1
                    strict_sources.add(source_id)

            try:
                projected = project_answer_fields_onto_target(source["target"], parsed)
            except ValueError:
                rejection_reasons["invalid_answer_fields"] += 1
                continue
            projected_audit = audit_response(
                source["case_text"], projected, source["target"]
            )
            if projected_audit.get("schema_errors") or projected_audit.get(
                "hard_failures"
            ):
                rejection_reasons["projected_schema_or_hard_failure"] += 1
                continue

            projectable += 1
            counts["answer_projectable"] += 1
            projectable_signatures[source_id].add(answer_level_signature(projected))

    profile_stats = {}
    for profile, counts in sorted(profile_counts.items()):
        profile_total = counts["total"]
        profile_stats[profile] = {
            **dict(counts),
            "answer_projection_rate": (
                counts["answer_projectable"] / profile_total if profile_total else 0.0
            ),
        }
    sources_with_one = sum(bool(values) for values in projectable_signatures.values())
    sources_with_two = sum(
        len(values) >= 2 for values in projectable_signatures.values()
    )
    return {
        "candidate_sources": len(candidate_sources),
        "total_candidates": total,
        "schema_valid_candidates": raw_schema_valid,
        "raw_schema_valid_candidates": raw_schema_valid,
        "structure_compatible_candidates": strict_compatible,
        "compatible_sources": len(strict_sources),
        "structure_compatibility_rate": strict_compatible / total if total else 0.0,
        "answer_projectable_candidates": projectable,
        "answer_projection_rate": projectable / total if total else 0.0,
        "sources_with_at_least_one_projectable_candidate": sources_with_one,
        "sources_with_at_least_two_distinct_projectable_candidates": sources_with_two,
        "candidate_sources_without_projectable_candidates": (
            len(candidate_sources) - sources_with_one
        ),
        "candidate_sources_without_two_distinct_projectable_candidates": (
            len(candidate_sources) - sources_with_two
        ),
        "profiles": profile_stats,
        "projection_rejections": dict(rejection_reasons),
    }


def main() -> None:
    args = parse_args()
    sources = read_jsonl(args.sources)
    candidate_rows = read_jsonl(args.candidates)
    pairs = build_pairs(sources, candidate_rows, args)
    write_jsonl(args.output, pairs)
    paired_source_ids = {row["source_id"] for row in pairs}
    selected_source_ids = {row["source_id"] for row in sources}
    candidate_source_ids = {
        row.get("source_id")
        for row in candidate_rows
        if row.get("source_id") in selected_source_ids
    }
    target_model_source_ids = {
        row["source_id"] for row in pairs if row["pair_type"] == "target_vs_model"
    }
    model_model_source_ids = {
        row["source_id"] for row in pairs if row["pair_type"] == "model_vs_model"
    }
    target_model_profiles: Counter[str] = Counter()
    for pair in pairs:
        if pair["pair_type"] != "target_vs_model":
            continue
        for candidate in (pair["candidate_A"], pair["candidate_B"]):
            if candidate.get("origin") == "sft_model":
                target_model_profiles[str(candidate.get("candidate_id", "unknown"))] += 1
    stats = {
        "schema_version": DPO_SCHEMA_VERSION,
        "preference_scope": "answer_level",
        "pairs": len(pairs),
        "pair_types": dict(Counter(row["pair_type"] for row in pairs)),
        "splits": dict(Counter(row["split"] for row in pairs)),
        "sources": len(paired_source_ids),
        "sources_without_pairs": len(selected_source_ids - paired_source_ids),
        "candidate_sources": len(candidate_source_ids),
        "candidate_sources_with_target_vs_model": len(target_model_source_ids),
        "candidate_sources_without_target_vs_model": len(
            candidate_source_ids - target_model_source_ids
        ),
        "candidate_sources_with_model_vs_model": len(model_model_source_ids),
        "candidate_sources_without_model_vs_model": len(
            candidate_source_ids - model_model_source_ids
        ),
        "target_vs_model_candidate_profiles": dict(target_model_profiles),
        "candidate_rank_tie_breaker": "stable_hash(source_id,candidate_id,seed)",
        "natural_candidate_alignment": candidate_alignment_stats(sources, candidate_rows),
        "single_blind_judge": True,
        "swap_consistency_check": False,
        "frozen_fields": [
            "task_type",
            "query_intent",
            "evidence_sufficiency",
            "evidence",
            "critical_evidence_ids",
        ],
        "answer_fields": list(ANSWER_LEVEL_FIELDS),
    }
    stats_path = Path(args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Pairs: {args.output}")


if __name__ == "__main__":
    main()
