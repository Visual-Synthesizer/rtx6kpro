# DeepSeek-V4-Flash-0731 on Infernal Invocation r14

**Status: qualified.** This page
specifies fixed probabilistic DSpark K5 serving for
`deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell GPUs. The
runtime validates accepted speculative structured output before committing it
to request history.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm91e2adf-b12xec2f97d-fi1ac6942-cu133-torch213-20260815-r14` |
| Registry digest | `sha256:9fe1d1b0e8370df578f92c6431af41a9fc36076ccbebac18c77a8ae99a3454e3` |
| Image ID | `sha256:e53afda79ad5133b583d3594e5f825c8ecacd4f8444748b0000a818854696a38` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@ad848fc4141f201489db18d5453c50b312245a0a` |
| vLLM integration tree | `91e2adf75b271a88bd4d18a081f40fa65e82d469` |
| B12X integration tree | `ec2f97d2fec10dc93e5bf4e7675821016f5419ff` |
| LMCache integration tree | `5fdf59cfa184bc15dc5414df0bd633da9e49aaae` |
| Image build source | [`945205e`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/945205e00dc7b28900f75b01e919c1bf849b0071) |
| Qualification receipt | [`validation/infernal-invocation-r14-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r14-remote-gpu.json) |
| Speculative grammar validation | [vLLM PR #320](https://github.com/local-inference-lab/vllm/pull/320) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The generated vLLM, B12X, and LMCache locks record each base commit, ordered
pull-request head, resulting Git tree, and patch digest. Every
`source_patches` array is empty.

## Start The Server

Download the release Compose profile and start TP2/DCP1 fixed probabilistic
K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/d1f2cafee04179b2b6890d0e3d7339cdfad299bb/examples/docker-compose-ds4-infernal-invocation-cu133-r14.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r14.yml up -d
```

The profile defaults to B12X W4A8, FP8 compressed MLA KV, InstantTensor
`BUFFERED`, prefix caching, fixed probabilistic DSpark K5, and release-scoped
JIT storage. Native vLLM KV offload and LMCache are disabled unless explicitly
selected.

The implemented 1M-token deployment envelope is:

```bash
MAX_MODEL_LEN=1048576 \
MAX_NUM_SEQS=8 \
MAX_NUM_BATCHED_TOKENS=4096 \
GRAPH=auto \
GPU_MEMORY_UTILIZATION=0.975 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r14.yml up -d
```

`GRAPH=auto` derives the graph cap from scheduler-reachable verifier rows. A
fixed K5 profile with eight request slots requires
`8 * (1 + 5) = 48` all-decode graph rows.

## Serving Contract

| Component | Qualified behavior |
|---|---|
| Checkpoint | Official `deepseek-ai/DeepSeek-V4-Flash-0731` revision |
| Target and draft quantization | `deepseek_v4_fp8` |
| Attention | B12X compressed MLA |
| MoE and linear layers | B12X W4A8 |
| KV cache | FP8 compressed target MLA plus FP32 sliding-window compressor state |
| Speculative decoding | Fixed probabilistic DSpark K5 |
| Structured output | XGrammar 0.2.5 with pre-commit speculative-prefix validation |
| Target and verifier decode | FULL CUDA graph for captured all-decode rows |
| DSpark proposal | FULL CUDA graph for captured verifier rows |
| Prefill | PIECEWISE or uncaptured model path |
| Model loading | InstantTensor `BUFFERED` |

Target-only, fixed K7, and confidence-controlled K7 are implemented by the
entrypoint but are outside this qualification receipt. Their selectors are
`MODE=dspark-mtp0`, `DSPARK_TOKENS=7`, and `DSPARK_DEPTH_MODE=dynamic`.

## Speculative Structured Output

Grammar masks are created for positions scheduled before speculative
verification. One accepted block can cross the reasoning-end marker that
activates a deferred grammar, or it can continue after the grammar reaches a
terminal state. A mask created before either transition is not sufficient to
authorize every token in that block.

The scheduler therefore enforces this invariant:

> A token enters request history only after the effective structured-output
> grammar accepts the complete prefix ending at that token.

The scheduler validates an accepted block without advancing grammar state. It
commits the longest valid prefix and rolls back rejected scheduler and
asynchronous-output accounting so the suffix can be sampled again. Tokens
through a deferred reasoning-end marker remain unconstrained. The grammar
manager advances exactly once, when the scheduler commits the accepted prefix.

The contract in [vLLM PR #320](https://github.com/local-inference-lab/vllm/pull/320)
complements three independent structured-output fixes:

- [vLLM PR #294](https://github.com/local-inference-lab/vllm/pull/294)
  preserves grammar-mask source width after speculative draft trimming;
- [vLLM PR #295](https://github.com/local-inference-lab/vllm/pull/295)
  stops XGrammar draft validation at matcher termination;
- [vLLM PR #302](https://github.com/local-inference-lab/vllm/pull/302)
  activates reasoning-aware structural tool grammars at token zero.

## Qualification Evidence

Source validation for [vLLM PR #320](https://github.com/local-inference-lab/vllm/pull/320)
at head `b33319edbbe1daa03a14bee6acd896aa32e0a6df` passed Ruff check,
Ruff format, `git diff --check`, and 30 focused scheduler, XGrammar, Guidance,
reasoning-marker, synchronous-accounting, and asynchronous-accounting tests.

The deterministic runtime discriminator uses eight growing strict-tool
sessions with asynchronous scheduling and fixed probabilistic K5:

| Condition | Result |
|---|---|
| Scheduler without pre-commit grammar validation | Request processing fails by round 11 |
| Scheduler with pre-commit grammar validation | 32 rounds and 256/256 HTTP responses through 225,634 prompt tokens |
| Schema result | 255 single valid tool calls; one request reached its 768-token limit while still reasoning |
| 282,817-byte strict-tool payload | Eight concurrent replays each returned one valid JSON tool call |

Published-image validation:

| Workload | Result |
|---|---|
| Buffered and streaming strict tools at C1 | 8/8 valid single tool calls; no violations |
| Concurrent strict tools at C8 | 16/16 valid buffered or streaming calls; no violations |
| Long-prompt strict tools at C8 | 8/8 valid at 225,330-225,634 prompt tokens; 1,803,671 prompt tokens total |
| Ordinary no-tool CC1 | 191.21 and 188.95 aggregate tok/s; 63.17 and 63.25 engine steps/s |
| Runtime log errors | 0 |

## Source Merge Contract

The source composition includes these vLLM responsibilities in addition to the
model and runtime integration recorded by the release lock:

| Repository | Pull request | Responsibility |
|---|---|---|
| vLLM | [#309](https://github.com/local-inference-lab/vllm/pull/309) | Materialize direct MLA DCP workspaces after deferred model weights reach CUDA |
| vLLM | [#320](https://github.com/local-inference-lab/vllm/pull/320) | Validate speculative structured-output prefixes before scheduler commit |

The complete ordered vLLM, B12X, and LMCache pull-request heads are maintained
in [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).
The release process does not merge vLLM or B12X pull requests.

## Qualification Limits

- **Qualified:** TP2/DCP1, fixed probabilistic DSpark K5, B12X W4A8, FP8
  compressed MLA KV, `MAX_NUM_SEQS=8`, graph cap 48,
  `MAX_MODEL_LEN=524288`, native KV offload disabled, and LMCache disabled.
- **Implemented:** target-only serving, K7 modes, native vLLM KV offload,
  LMCache, and the GLM-5.2 launcher profiles included in the image.
- **Unsupported claim:** one finite growing-session workload does not establish
  stability for every multi-hour agent workload or every tool schema.
- **Unsupported:** DS4 DCP greater than one and alternate DeepSeek checkpoints
  are outside this receipt.
- Performance measurements use direct-root-port GPUs and are not directly
  comparable with switched-PCIe results.
- A fresh cache compiles context-dependent B12X kernels during the first long
  request. Persist the release-scoped `/cache` mount to avoid repeating that
  startup latency.
