#!/usr/bin/env python3
"""Compare an EXL3 checkpoint overlay with saved BF16 teacher logits."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F
from safetensors.torch import safe_open, save_file
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def dense_prompt_logprobs(
    prompt_logprobs: object, npos: int, vocab: int
) -> torch.Tensor:
    dense = torch.empty((npos, vocab), dtype=torch.float32)
    if hasattr(prompt_logprobs, "start_indices"):
        for pos in range(npos):
            start = prompt_logprobs.start_indices[pos + 1]
            end = prompt_logprobs.end_indices[pos + 1]
            ids = torch.as_tensor(
                prompt_logprobs.token_ids[start:end], dtype=torch.long
            )
            values = torch.as_tensor(
                prompt_logprobs.logprobs[start:end], dtype=torch.float32
            )
            row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
            valid = (ids >= 0) & (ids < vocab)
            row[ids[valid]] = values[valid]
            dense[pos] = row
        return dense

    for pos in range(npos):
        row = torch.full((vocab,), float("-inf"), dtype=torch.float32)
        for token_id, logprob in prompt_logprobs[pos + 1].items():
            token_id = int(token_id)
            if 0 <= token_id < vocab:
                row[token_id] = float(logprob.logprob)
        dense[pos] = row
    return dense


def positional_kld(
    model_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, reference_logits.shape[0], chunk_rows):
        end = min(reference_logits.shape[0], start + chunk_rows)
        model_log_probs = F.log_softmax(model_logits[start:end].float(), dim=-1)
        reference_log_probs = F.log_softmax(
            reference_logits[start:end].float(), dim=-1
        )
        chunks.append(
            F.kl_div(
                model_log_probs,
                reference_log_probs,
                reduction="none",
                log_target=True,
            ).sum(dim=-1)
        )
    return torch.cat(chunks).cpu()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-logits", required=True)
    parser.add_argument("--tokens-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quantization-config", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--kv-cache-dtype", default="nvfp4_ds_mla")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--chunk-rows", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_dir = Path(args.reference_logits)
    manifest = json.loads((reference_dir / "manifest.json").read_text())
    token_ids = json.loads(Path(args.tokens_file).read_text())
    if len(token_ids) != args.context_length:
        raise RuntimeError(
            f"expected {args.context_length} tokens, file has {len(token_ids)}"
        )
    if token_ids[:16] != manifest["token_first16"]:
        raise RuntimeError("tokenizer output does not match BF16 reference")

    with safe_open(
        str(reference_dir / "logits_0.safetensors"),
        framework="pt",
        device="cpu",
    ) as handle:
        reference_logits = handle.get_tensor("logits")

    quantization_config = json.loads(args.quantization_config)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.95,
        # Avoid filling spare VRAM with KV: full prompt-logprob extraction
        # needs several temporary full-vocabulary buffers on every TP rank.
        kv_cache_memory_bytes=512 * 1024 * 1024,
        dtype="bfloat16",
        kv_cache_dtype=args.kv_cache_dtype,
        load_format="safetensors",
        max_model_len=4096,
        # r19 exposes full prompt logits through prompt_logprobs=-1. Chunking
        # keeps its full-vocabulary top-k workspace bounded without changing
        # the causal logits for the 2048-token sequence.
        max_num_batched_tokens=256,
        max_num_seqs=1,
        attention_backend="B12X_MLA_SPARSE",
        moe_backend="b12x",
        quantization="exl3",
        **(
            {"quantization_config": quantization_config}
            if quantization_config is not None
            else {}
        ),
        decode_context_parallel_size=1,
        enforce_eager=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=-1,
        hf_overrides={
            "use_index_cache": True,
            "index_topk_pattern": (
                "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSF"
                "SSSFSSSFSSSFSSSFSSSFSSS"
            ),
        },
    )

    prompt: TokensPrompt = {
        "prompt_token_ids": token_ids,
        "target_token_ids": token_ids[1:],
    }
    supports_prompt_logits = (
        "return_prompt_logits" in inspect.signature(SamplingParams).parameters
    )
    results = []
    for run in range(1, args.repeats + 1):
        if supports_prompt_logits:
            sampling_params = SamplingParams(
                prompt_logprobs=1,
                max_tokens=1,
                return_prompt_logits=True,
                detokenize=False,
            )
        else:
            sampling_params = SamplingParams(
                prompt_logprobs=-1,
                flat_logprobs=True,
                max_tokens=1,
                detokenize=False,
            )

        started = time.monotonic()
        output = llm.generate([prompt], sampling_params=sampling_params)[0]
        npos = min(args.context_length - 1, reference_logits.shape[0])
        vocab = reference_logits.shape[-1]
        if supports_prompt_logits:
            if output.prompt_logits is None:
                raise RuntimeError("vLLM returned no prompt logits")
            model_logits = output.prompt_logits[:npos, :vocab].detach().cpu()
        else:
            if output.prompt_logprobs is None:
                raise RuntimeError("vLLM returned no prompt logprobs")
            model_logits = dense_prompt_logprobs(
                output.prompt_logprobs, npos, vocab
            )

        finite_logits = torch.isfinite(model_logits)
        if not bool(finite_logits.all()):
            bad_logits = int((~finite_logits).sum().item())
            raise RuntimeError(
                f"candidate returned {bad_logits} non-finite logits; "
                "this runtime configuration is invalid for KLD"
            )

        kld = positional_kld(
            model_logits,
            reference_logits[:npos, :vocab],
            args.chunk_rows,
        )
        if not bool(torch.isfinite(kld).all()):
            bad_positions = int((~torch.isfinite(kld)).sum().item())
            raise RuntimeError(
                f"KLD contains {bad_positions} non-finite positions; "
                "refusing to publish this run"
            )
        result = {
            "run": run,
            "mean_kld": float(kld.mean()),
            "sd_across_positions": float(kld.std(unbiased=True)),
            "positions": int(kld.numel()),
            "elapsed_sec": time.monotonic() - started,
        }
        results.append(result)
        save_file(
            {"kld": kld},
            str(output_dir / f"kld_positions_run{run}.safetensors"),
        )
        print("kld_run " + json.dumps(result, sort_keys=True), flush=True)
        del output, model_logits, kld

    run_means = [result["mean_kld"] for result in results]
    summary = {
        "label": args.label,
        "model": args.model,
        "reference_logits": str(reference_dir),
        "quantization_config": quantization_config,
        "kv_cache_dtype": args.kv_cache_dtype,
        "supports_prompt_logits": supports_prompt_logits,
        "token_first16": token_ids[:16],
        "results": results,
        "mean_kld": statistics.fmean(run_means),
        "run_sd": statistics.stdev(run_means) if len(run_means) > 1 else 0.0,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print("kld_done " + json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
