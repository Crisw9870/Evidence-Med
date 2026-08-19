#!/usr/bin/env python3
"""Merge the Direct Evidence-SFT adapter into Qwen to create the frozen D0.
CUDA_VISIBLE_DEVICES=0 python process_dpo/prepare_dpo_start.py \
  --base-model ./Qwen/Qwen2.5-7B-Instruct \
  --sft-adapter ./outputs/evidence-sft-frombase \
  --output ./outputs/dpo-start-direct-merged \
  --torch-dtype bfloat16 \
  --device-map cuda:0

"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MANIFEST_NAME = "dpo_start_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-adapter", default="outputs/evidence-sft-frombase")
    parser.add_argument("--output", default="outputs/dpo-start-direct-merged")
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(
    base_model: str | Path, sft_adapter: str | Path, output: str | Path
) -> dict[str, Any]:
    adapter_path = Path(sft_adapter)
    output_path = Path(output)
    adapter_config_path = adapter_path / "adapter_config.json"
    adapter_weights = adapter_path / "adapter_model.safetensors"
    if not adapter_config_path.is_file():
        raise ValueError(f"missing SFT adapter config: {adapter_config_path}")
    if not adapter_weights.is_file():
        raise ValueError(f"missing SFT adapter weights: {adapter_weights}")
    if output_path.exists() and not output_path.is_dir():
        raise ValueError(f"output exists and is not a directory: {output_path}")
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError(
            f"output already exists and is not empty: {output_path}; "
            "use a new D0 directory"
        )

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    configured_base = str(adapter_config.get("base_model_name_or_path", ""))
    requested_base = str(base_model)
    if configured_base and Path(configured_base).name != Path(requested_base).name:
        raise ValueError(
            "SFT adapter base does not match --base-model: "
            f"{configured_base!r} != {requested_base!r}"
        )
    return {
        "adapter_config": adapter_config,
        "adapter_config_sha256": _sha256(adapter_config_path),
        "adapter_weights_sha256": _sha256(adapter_weights),
    }


def main() -> None:
    args = parse_args()
    provenance = validate_inputs(args.base_model, args.sft_adapter, args.output)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.torch_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    policy_start = PeftModel.from_pretrained(
        base,
        args.sft_adapter,
        is_trainable=False,
    )
    merged = policy_start.merge_and_unload(safe_merge=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.sft_adapter, use_fast=True, trust_remote_code=True
        )
        tokenizer_source = args.sft_adapter
    except (OSError, ValueError):
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, use_fast=True, trust_remote_code=True
        )
        tokenizer_source = args.base_model
    tokenizer.save_pretrained(output)

    manifest = {
        "schema_version": "evidence-dpo-start-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "objective": "direct_evidence_sft_to_answer_level_dpo",
        "base_model": str(args.base_model),
        "sft_adapter": str(args.sft_adapter),
        "tokenizer_source": str(tokenizer_source),
        "merged_sft_adapter": True,
        "reference_semantics": (
            "DPOTrainer disables the fresh DPO adapter; the merged D0 weights "
            "therefore serve as the frozen reference."
        ),
        "torch_dtype": args.torch_dtype,
        "adapter_config_sha256": provenance["adapter_config_sha256"],
        "adapter_weights_sha256": provenance["adapter_weights_sha256"],
        "adapter_base_model": provenance["adapter_config"].get(
            "base_model_name_or_path"
        ),
    }
    (output / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Merged D0 checkpoint: {output}")


if __name__ == "__main__":
    main()
