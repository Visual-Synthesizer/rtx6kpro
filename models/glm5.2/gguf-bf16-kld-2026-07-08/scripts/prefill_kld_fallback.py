#!/usr/bin/env python3
"""Fallback prefill KLD runner for vLLM builds without SamplingParams.kld_mode."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from safetensors.torch import safe_open
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def _dense_from_flat_prompt_logprobs(prompt_logprobs, npos: int,
                                    vocab: int) -> torch.Tensor:
    dense = torch.empty((npos, vocab), dtype=torch.float32)
    if hasattr(prompt_logprobs, "start_indices"):
        for pos in range(npos):
            src_pos = pos + 1
            start = prompt_logprobs.start_indices[src_pos]
            end = prompt_logprobs.end_indices[src_pos]
            ids = torch.tensor(prompt_logprobs.token_ids[start:end],
                               dtype=torch.long)
            vals = torch.tensor(prompt_logprobs.logprobs[start:end],
                                dtype=torch.float32)
            row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
            valid = (ids >= 0) & (ids < vocab)
            row[ids[valid]] = vals[valid]
            dense[pos] = row
        return dense

    for pos in range(npos):
        src_pos = pos + 1
        row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
        for token_id, lp in prompt_logprobs[src_pos].items():
            tid = int(token_id)
            if 0 <= tid < vocab:
                row[tid] = float(lp.logprob)
        dense[pos] = row
    return dense


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--max-windows", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument("--load-format", default="fastsafetensors")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument(
        "--quantization",
        default="modelopt_fp4",
        help=(
            "Quantization override passed to vLLM. Use auto/none/null/empty to "
            "let the checkpoint quantization_config select the method."
        ),
    )
    parser.add_argument("--attention-backend", default="SPARSE_MLA_SM120")
    parser.add_argument("--hf-overrides", default="{}")
    parser.add_argument("--llm-extra-json", default="{}")
    parser.add_argument("--kld-chunk-rows", type=int, default=32)
    args = parser.parse_args()

    hf_overrides = json.loads(args.hf_overrides)
    extra = json.loads(args.llm_extra_json)

    print(
        "fallback_prefill_kld_start",
        json.dumps(
            {
                "model": args.model,
                "reference_logits": args.reference_logits,
                "context_length": args.context_length,
                "stride": args.stride,
                "max_windows": args.max_windows,
                "hf_overrides": hf_overrides,
                "extra": extra,
                "env": {
                    "VLLM_WORKER_MULTIPROC_METHOD": os.getenv(
                        "VLLM_WORKER_MULTIPROC_METHOD"
                    ),
                    "VLLM_USE_B12X_MOE": os.getenv("VLLM_USE_B12X_MOE"),
                    "VLLM_USE_B12X_SPARSE_INDEXER": os.getenv(
                        "VLLM_USE_B12X_SPARSE_INDEXER"
                    ),
                    "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE": os.getenv(
                        "VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE"
                    ),
                    "VLLM_B12X_FORCE_MOE_A16": os.getenv(
                        "VLLM_B12X_FORCE_MOE_A16"
                    ),
                    "B12X_MOE_FORCE_A16": os.getenv("B12X_MOE_FORCE_A16"),
                    "B12X_W4A16_TC_DECODE": os.getenv("B12X_W4A16_TC_DECODE"),
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as exc:
        print(
            "wikitext_alias_failed",
            json.dumps({"error": repr(exc), "fallback": "Salesforce/wikitext"}),
            flush=True,
        )
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if x.get("text") and x["text"].strip()]
    text = "\n\n".join(texts)
    max_tokens = args.context_length + max(0, args.max_windows - 1) * args.stride
    text = text[: max_tokens * 5]

    tokenizer_path = args.tokenizer if args.tokenizer else args.model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
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
    if args.tokenizer:
        llm_kwargs["tokenizer"] = args.tokenizer
    if args.quantization.lower() not in ("", "auto", "none", "null"):
        llm_kwargs["quantization"] = args.quantization
    llm_kwargs.update(extra)
    llm = LLM(**llm_kwargs)

    kld_sum = 0.0
    kld_count = 0
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
        supports_prompt_logits = (
            "return_prompt_logits" in inspect.signature(SamplingParams).parameters
        )
        if supports_prompt_logits:
            params = SamplingParams(
                prompt_logprobs=1,
                max_tokens=1,
                return_prompt_logits=True,
                detokenize=False,
            )
        else:
            params = SamplingParams(
                prompt_logprobs=-1,
                flat_logprobs=True,
                max_tokens=1,
                detokenize=False,
            )
        out = llm.generate([prompt], sampling_params=params)[0]

        ref_file = os.path.join(
            args.reference_logits, f"logits_{window_idx}.safetensors"
        )
        with safe_open(ref_file, framework="pt", device="cpu") as f:
            ref_logits = f.get_tensor("logits")

        ref_shape = list(ref_logits.shape)
        vocab = int(ref_logits.shape[-1])
        if supports_prompt_logits:
            raw_model_logits = out.prompt_logits
            if raw_model_logits is None:
                raise RuntimeError("vLLM returned no prompt_logits")
            model_shape = list(raw_model_logits.shape)
            npos = min(raw_model_logits.shape[0], ref_logits.shape[0])
            # Keep the KLD reduction off GPU. The model still owns almost all
            # GPU memory here, so even a single full-vocab temporary can OOM.
            model_logits = (
                raw_model_logits[:npos, :vocab].detach().to("cpu", copy=True)
            )
            del raw_model_logits
        else:
            prompt_logprobs = out.prompt_logprobs
            if prompt_logprobs is None:
                raise RuntimeError("vLLM returned no prompt_logprobs")
            npos = min(len(window_tokens) - 1, ref_logits.shape[0])
            model_logits = _dense_from_flat_prompt_logprobs(prompt_logprobs,
                                                            npos, vocab)
            model_shape = list(model_logits.shape)
        ref_logits = ref_logits[:npos, :vocab]
        del out
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        win_sum = 0.0
        win_count = 0
        chunk_rows = max(1, args.kld_chunk_rows)
        for chunk_start in range(0, npos, chunk_rows):
            chunk_end = min(npos, chunk_start + chunk_rows)
            model_chunk = model_logits[chunk_start:chunk_end].to(torch.float32)
            ref_chunk = ref_logits[chunk_start:chunk_end].to(torch.float32)
            log_probs_model = F.log_softmax(model_chunk, dim=-1)
            log_probs_ref = F.log_softmax(ref_chunk, dim=-1)
            # Same direction as score_mode_kld fallback: KL(ref || model).
            kld_chunk = F.kl_div(
                log_probs_model,
                log_probs_ref,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
            win_sum += float(kld_chunk.sum().item())
            win_count += int(kld_chunk.numel())
        kld_sum += win_sum
        kld_count += win_count
        print(
            "window_done",
            json.dumps(
                {
                    "window_idx": window_idx,
                    "positions": win_count,
                    "mean_kld": win_sum / win_count,
                    "model_logits_shape": model_shape,
                    "ref_logits_shape": ref_shape,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if kld_count == 0:
        raise RuntimeError("No valid KLD positions")

    elapsed = time.time() - t0
    mean = kld_sum / kld_count
    print("\nResults:")
    print(f"  Mean KLD: {mean:.6f}")
    print(f"  Total positions: {kld_count}")
    print(f"  Time elapsed: {elapsed:.2f} seconds")
    print(f"  Positions/second: {kld_count / elapsed:.2f}")
    print(
        "fallback_prefill_kld_done",
        json.dumps(
            {"mean_kld": mean, "total_positions": kld_count, "elapsed_sec": elapsed},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
