#!/usr/bin/env python3
"""Hugging Face/PEFT model loading and deterministic batched generation."""

from __future__ import annotations

import argparse
from typing import Any


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Full model or base-model directory.")
    parser.add_argument("--adapter", default="", help="Optional primary PEFT adapter.")
    parser.add_argument(
        "--additional-adapter",
        default="",
        help="Optional additive second adapter, e.g. DPO LoRA on top of an SFT LoRA.",
    )
    parser.add_argument(
        "--tokenizer",
        default="",
        help="Tokenizer directory; defaults to --model.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional fixed system prompt. Keep identical across compared checkpoints.",
    )


class ModelRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        except ImportError as exc:
            raise RuntimeError(
                "Model inference requires torch and transformers. Use "
                "/root/miniconda3/envs/medgpt/bin/python."
            ) from exc

        self.torch = torch
        dtype = getattr(torch, args.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or args.model,
            use_fast=True,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=dtype,
            device_map=args.device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        if args.adapter:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("--adapter requires peft") from exc
            model = PeftModel.from_pretrained(base, args.adapter, adapter_name="primary")
            if args.additional_adapter:
                model.load_adapter(args.additional_adapter, adapter_name="additional")
                model.base_model.set_adapter(["primary", "additional"])
        else:
            if args.additional_adapter:
                raise ValueError("--additional-adapter requires --adapter")
            model = base
        model.eval()
        generation_config = GenerationConfig.from_model_config(model.config)
        generation_config.do_sample = False
        generation_config.num_beams = 1
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None
        generation_config.repetition_penalty = 1.0
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model
        self.generation_config = generation_config
        self.input_device = model.get_input_embeddings().weight.device

    def render(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_rendered(
        self, rendered: list[str], *, max_new_tokens: int
    ) -> list[dict[str, Any]]:
        inputs = self.tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=False
        ).to(self.input_device)
        input_width = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            sequences = self.model.generate(
                **inputs,
                generation_config=self.generation_config,
                max_new_tokens=max_new_tokens,
                use_cache=True,
            )
        generated = sequences[:, input_width:]
        outputs: list[dict[str, Any]] = []
        for sequence in generated:
            token_ids = sequence.tolist()
            ended_with_eos = (
                self.tokenizer.eos_token_id is not None
                and self.tokenizer.eos_token_id in token_ids
            )
            if ended_with_eos:
                end = token_ids.index(self.tokenizer.eos_token_id) + 1
                token_ids = token_ids[:end]
            elif self.tokenizer.pad_token_id is not None:
                while token_ids and token_ids[-1] == self.tokenizer.pad_token_id:
                    token_ids.pop()
            outputs.append(
                {
                    "text": self.tokenizer.decode(
                        token_ids, skip_special_tokens=True
                    ).strip(),
                    "generated_tokens": len(token_ids),
                    "ended_with_eos": ended_with_eos,
                }
            )
        return outputs

    def messages(self, user_content: str, system_prompt: str = "") -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_content})
        return messages
