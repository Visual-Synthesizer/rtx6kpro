#!/usr/bin/env python3
"""Collect dense prefill logits through vLLM return_prompt_logits."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time

import torch
from datasets import load_dataset
from safetensors.torch import save_file
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--load-format", default="safetensors")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--quantization", default="auto")
    parser.add_argument("--attention-backend", default="B12X_MLA_SPARSE")
    parser.add_argument("--hf-overrides", default="{}")
    parser.add_argument("--llm-extra-json", default="{}")
    args = parser.parse_args()

    if "return_prompt_logits" not in inspect.signature(SamplingParams).parameters:
        raise RuntimeError("This vLLM build does not expose return_prompt_logits")

    os.makedirs(args.output_dir, exist_ok=True)
    hf_overrides = json.loads(args.hf_overrides)
    extra = json.loads(args.llm_extra_json)

    print(
        "collect_prefill_ref_start",
        json.dumps(
            {
                "model": args.model,
                "output_dir": args.output_dir,
                "context_length": args.context_length,
                "stride": args.stride,
                "max_windows": args.max_windows,
                "hf_overrides": hf_overrides,
                "extra": extra,
                "quantization": args.quantization,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x.get("text") and x["text"].strip()]
    text = "\n\n".join(texts)
    max_tokens = args.context_length + max(0, args.max_windows - 1) * args.stride
    text = text[: max_tokens * 5]

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    print(
        "tokenized",
        json.dumps({"num_tokens": len(token_ids), "first16": token_ids[:16]}),
        flush=True,
    )

    llm_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "dtype": args.dtype,
        "kv_cache_dtype": args.kv_cache_dtype,
        "load_format": args.load_format,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "attention_backend": args.attention_backend,
        "hf_overrides": hf_overrides,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "max_logprobs": -1,
    }
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        llm_kwargs["quantization"] = args.quantization
    llm_kwargs.update(extra)
    llm = LLM(**llm_kwargs)

    params = SamplingParams(
        prompt_logprobs=1,
        max_tokens=1,
        return_prompt_logits=True,
        detokenize=False,
    )

    manifest = {
        "model": args.model,
        "context_length": args.context_length,
        "stride": args.stride,
        "max_windows": args.max_windows,
        "token_first16": token_ids[:16],
        "windows": [],
    }
    written = 0
    t0 = time.time()
    for window_idx, start_idx in enumerate(
        range(0, len(token_ids) - args.context_length + args.stride, args.stride)
    ):
        if window_idx >= args.max_windows:
            break
        end_idx = start_idx + args.context_length
        if end_idx > len(token_ids):
            break
        window_tokens = token_ids[start_idx:end_idx]
        prompt: TokensPrompt = {
            "prompt_token_ids": window_tokens,
            "target_token_ids": window_tokens[1:],
        }
        out = llm.generate([prompt], sampling_params=params)[0]
        raw_logits = out.prompt_logits
        if raw_logits is None:
            raise RuntimeError("vLLM returned no prompt_logits")
        npos = len(window_tokens) - 1
        logits = raw_logits[:npos].detach().to("cpu", dtype=torch.float32, copy=True)
        del raw_logits, out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        path = os.path.join(args.output_dir, f"logits_{window_idx}.safetensors")
        save_file({"logits": logits}, path)
        record = {
            "window_idx": window_idx,
            "path": path,
            "shape": list(logits.shape),
            "dtype": str(logits.dtype),
        }
        manifest["windows"].append(record)
        print("wrote_prefill_ref", json.dumps(record, sort_keys=True), flush=True)
        written += 1

    if written == 0:
        raise RuntimeError("No reference windows written")

    manifest["elapsed_sec"] = time.time() - t0
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        "collect_prefill_ref_done",
        json.dumps(
            {
                "output_dir": args.output_dir,
                "windows": written,
                "elapsed_sec": manifest["elapsed_sec"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
