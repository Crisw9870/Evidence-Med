#!/usr/bin/env python3
"""Generate adaptive natural-answer candidates from the current Evidence-SFT model."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dpo_common import (
    answer_level_signature,
    audit_response,
    project_answer_fields_onto_target,
    read_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="data/dpo/answer_v1/00_sources.jsonl")
    parser.add_argument("--output", default="data/dpo/answer_v1/01_sft_candidates.jsonl")
    parser.add_argument("--stats-output", default="")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", default="outputs/evidence-sft-frombase")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1152)
    parser.add_argument("--temperatures", default="0.8")
    parser.add_argument(
        "--no-greedy",
        action="store_true",
        help="Generate only the requested sampling profiles (recovery batches).",
    )
    parser.add_argument("--fallback-temperature", type=float, default=0.6)
    parser.add_argument("--min-distinct-candidates", type=int, default=2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _profiles(raw: str, *, include_greedy: bool = True) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    if include_greedy:
        profiles.append({"candidate_id": "greedy", "do_sample": False})
    seen = {profile["candidate_id"] for profile in profiles}
    for value in raw.split(","):
        value = value.strip()
        if not value:
            continue
        temperature = float(value)
        if temperature <= 0:
            raise ValueError("Sampling temperatures must be positive")
        candidate_id = f"sample_t{temperature:g}"
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        profiles.append(
            {
                "candidate_id": candidate_id,
                "do_sample": True,
                "temperature": temperature,
            }
        )
    return profiles


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row["source_id"]: row for row in read_jsonl(path) if row.get("source_id")}


def _projectable_answer_signature(
    source: dict[str, Any], candidate: dict[str, Any]
) -> str | None:
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
    return answer_level_signature(projected)


def _fallback_source_indices(
    batch: list[dict[str, Any]],
    generated_by_source: list[list[dict[str, Any]]],
    min_distinct_candidates: int,
) -> list[int]:
    deficient: list[int] = []
    for index, (source, candidates) in enumerate(zip(batch, generated_by_source)):
        signatures: set[str] = set()
        for candidate in candidates:
            signature = _projectable_answer_signature(source, candidate)
            if signature is not None:
                signatures.add(signature)
        if len(signatures) < min_distinct_candidates:
            deficient.append(index)
    return deficient


def _trim_generated_tokens(
    token_ids: list[int], eos_token_id: int | None, pad_token_id: int | None
) -> tuple[list[int], bool]:
    if eos_token_id is not None and eos_token_id in token_ids:
        end = token_ids.index(eos_token_id) + 1
        return token_ids[:end], True
    if pad_token_id is not None and pad_token_id != eos_token_id:
        while token_ids and token_ids[-1] == pad_token_id:
            token_ids.pop()
    return token_ids, False


def _stats_path(output: Path, raw: str) -> Path:
    return Path(raw) if raw else output.with_suffix(".stats.json")


def generation_stats(
    sources: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    min_distinct_candidates: int,
    elapsed_seconds: float,
    generated_sources: int,
) -> dict[str, Any]:
    source_by_id = {row["source_id"]: row for row in sources}
    profile_counts: Counter[str] = Counter()
    projectable_candidates = 0
    sources_with_one = 0
    sources_with_minimum = 0
    fallback_sources = 0
    matched_records = 0
    for record in records:
        source = source_by_id.get(record.get("source_id"))
        if source is None:
            continue
        matched_records += 1
        policy = record.get("generation_policy")
        if isinstance(policy, dict) and policy.get("fallback_triggered"):
            fallback_sources += 1
        signatures: set[str] = set()
        for candidate in record.get("candidates", []):
            if not isinstance(candidate, dict):
                continue
            profile_counts[str(candidate.get("candidate_id", "unknown"))] += 1
            signature = _projectable_answer_signature(source, candidate)
            if signature is not None:
                projectable_candidates += 1
                signatures.add(signature)
        if signatures:
            sources_with_one += 1
        if len(signatures) >= min_distinct_candidates:
            sources_with_minimum += 1
    return {
        "sources_requested": len(sources),
        "source_records": matched_records,
        "generated_sources_this_run": generated_sources,
        "candidate_profiles": dict(sorted(profile_counts.items())),
        "answer_projectable_candidates": projectable_candidates,
        "sources_with_at_least_one_projectable_candidate": sources_with_one,
        "sources_with_minimum_distinct_candidates": sources_with_minimum,
        "min_distinct_candidates": min_distinct_candidates,
        "fallback_sources": fallback_sources,
        "fallback_rate": fallback_sources / matched_records if matched_records else 0.0,
        "elapsed_seconds_this_run": round(elapsed_seconds, 3),
        "sources_per_second_this_run": (
            generated_sources / elapsed_seconds if elapsed_seconds > 0 else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.min_distinct_candidates <= 0:
        raise ValueError("min-distinct-candidates must be positive")

    sources = read_jsonl(args.sources)
    if args.limit > 0:
        sources = sources[: args.limit]
    output = Path(args.output)
    completed = _load_completed(output) if args.resume else {}
    pending = [row for row in sources if row["source_id"] not in completed]
    print(f"Sources={len(sources)}, completed={len(completed)}, pending={len(pending)}")

    primary_profiles = _profiles(
        args.temperatures, include_greedy=not args.no_greedy
    )
    if not primary_profiles:
        raise ValueError("at least one primary generation profile is required")
    primary_ids = {profile["candidate_id"] for profile in primary_profiles}
    fallback_profiles = [
        profile
        for profile in _profiles(
            str(args.fallback_temperature), include_greedy=False
        )
        if profile["candidate_id"] not in primary_ids
    ]
    profile_ids = [profile["candidate_id"] for profile in primary_profiles]
    fallback_ids = [profile["candidate_id"] for profile in fallback_profiles]

    output.parent.mkdir(parents=True, exist_ok=True)
    if not pending:
        summary = generation_stats(
            sources,
            list(completed.values()),
            min_distinct_candidates=args.min_distinct_candidates,
            elapsed_seconds=0.0,
            generated_sources=0,
        )
        summary["primary_candidate_ids"] = profile_ids
        summary["fallback_candidate_ids"] = fallback_ids
        stats_path = _stats_path(output, args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    import torch
    from peft import PeftModel
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, use_fast=True, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    model.eval()

    mode = "a" if args.resume and output.exists() else "w"
    started = time.monotonic()
    run_records: list[dict[str, Any]] = []
    all_profiles = primary_profiles + fallback_profiles
    with output.open(mode, encoding="utf-8") as handle, tqdm(
        total=len(pending), desc="DPO candidates", unit="source", dynamic_ncols=True
    ) as progress:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            rendered = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            inputs = tokenizer(
                rendered, return_tensors="pt", padding=True, truncation=False
            ).to(model.device)
            input_width = inputs["input_ids"].shape[1]
            generated_by_source: list[list[dict[str, Any]]] = [[] for _ in batch]
            fallback_triggered: set[int] = set()

            def generate_profile(
                profile: dict[str, Any], source_indices: list[int], profile_index: int
            ) -> None:
                if not source_indices:
                    return
                torch.manual_seed(
                    args.seed + start * max(1, len(all_profiles)) + profile_index
                )
                generation_args: dict[str, Any] = {
                    "do_sample": profile["do_sample"],
                    "max_new_tokens": args.max_new_tokens,
                    "eos_token_id": tokenizer.eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                    "use_cache": True,
                }
                if profile["do_sample"]:
                    generation_args.update(
                        temperature=profile["temperature"], top_p=args.top_p
                    )
                selected_inputs = {
                    key: value[source_indices] for key, value in inputs.items()
                }
                with torch.inference_mode():
                    sequences = model.generate(**selected_inputs, **generation_args)
                generated = sequences[:, input_width:]
                for output_index, source_index in enumerate(source_indices):
                    token_ids, ended_with_eos = _trim_generated_tokens(
                        generated[output_index].tolist(),
                        tokenizer.eos_token_id,
                        tokenizer.pad_token_id,
                    )
                    text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
                    source = batch[source_index]
                    generated_by_source[source_index].append(
                        {
                            "candidate_id": profile["candidate_id"],
                            "origin": "sft_model",
                            "text": text,
                            "generation": {
                                "do_sample": profile["do_sample"],
                                "temperature": profile.get("temperature"),
                                "top_p": args.top_p if profile["do_sample"] else None,
                                "generated_tokens": len(token_ids),
                                "ended_with_eos": ended_with_eos,
                            },
                            "audit": audit_response(
                                source["case_text"], text, source["target"]
                            ),
                        }
                    )

            all_indices = list(range(len(batch)))
            for profile_index, profile in enumerate(primary_profiles):
                generate_profile(profile, all_indices, profile_index)

            for fallback_offset, profile in enumerate(fallback_profiles):
                deficient = _fallback_source_indices(
                    batch,
                    generated_by_source,
                    args.min_distinct_candidates,
                )
                if not deficient:
                    break
                fallback_triggered.update(deficient)
                generate_profile(
                    profile,
                    deficient,
                    len(primary_profiles) + fallback_offset,
                )

            for index, (source, candidates) in enumerate(
                zip(batch, generated_by_source)
            ):
                record = {
                    "source_id": source["source_id"],
                    "split": source["split"],
                    "base_model": args.base_model,
                    "adapter": args.adapter,
                    "generation_policy": {
                        "primary_candidate_ids": profile_ids,
                        "fallback_candidate_ids": fallback_ids,
                        "min_distinct_candidates": args.min_distinct_candidates,
                        "fallback_triggered": index in fallback_triggered,
                    },
                    "candidates": candidates,
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                run_records.append(record)
            handle.flush()
            progress.update(len(batch))

    elapsed = time.monotonic() - started
    records = list(completed.values()) + run_records
    summary = generation_stats(
        sources,
        records,
        min_distinct_candidates=args.min_distinct_candidates,
        elapsed_seconds=elapsed,
        generated_sources=len(run_records),
    )
    summary["primary_candidate_ids"] = profile_ids
    summary["fallback_candidate_ids"] = fallback_ids
    stats_path = _stats_path(output, args.stats_output)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Candidates: {output}; stats={stats_path}; elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
