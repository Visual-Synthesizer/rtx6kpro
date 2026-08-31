# DeepSeek-V4-Flash-0731 Infernal Invocation r12

**Status: qualified.** This page specifies fixed probabilistic DSpark K5
serving for `deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell
GPUs. Qualification covers source-locked startup, FULL target and DSpark CUDA
graph capture, heterogeneous attention-cache block recycling, concurrent
structured-tool requests, sustained plain decode, and a near-capacity pair of
long-context requests.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllmdc2934e-b12xd48c62b-fi1ac6942-cu133-torch213-20260814-r12` |
| Registry digest | `sha256:7bb6994afe2b9b2307afb87f926ffe2fdc938254dc98f45692f836bc85654849` |
| Image ID | `sha256:f65f357adf5c880a7fc5f121d37e912d150568d320fbe0e5b8b0be7a169bf76f` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@ad848fc4141f201489db18d5453c50b312245a0a` |
| vLLM integration tree | `dc2934ef69de7dc4d35c2ec13e088db47ce1d7d7` |
| B12X integration tree | `d48c62bbbdcac90ae5a9e85888ee0be3f8abeafb` |
| LMCache integration tree | `5fdf59cfa184bc15dc5414df0bd633da9e49aaae` |
| Image build source | [`1401571`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/1401571754a7633582904f9d885d1139f643b432) |
| Qualification receipt | [`validation/infernal-invocation-r12-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r12-remote-gpu.json) |
| Recycled-block zeroing | [vLLM PR #308](https://github.com/local-inference-lab/vllm/pull/308) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The vLLM, B12X, and LMCache lock directories record each base commit, ordered
pull-request head, resulting Git tree, and patch digest. Every
`source_patches` array is empty.

## Start The Server

Download the immutable Compose profile and start TP2/DCP1 fixed probabilistic
K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/1401571754a7633582904f9d885d1139f643b432/examples/docker-compose-ds4-infernal-invocation-cu133-r12.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r12.yml up -d
```

The profile defaults to B12X W4A8, FP8 compressed MLA KV, InstantTensor
`BUFFERED`, prefix caching, fixed probabilistic DSpark K5, and release-scoped
JIT storage. Native vLLM KV offload and LMCache are disabled unless explicitly
selected.

The qualified 1M-token configuration uses a smaller scheduler envelope so two
long requests fit concurrently:

```bash
MAX_MODEL_LEN=1048576 \
MAX_NUM_SEQS=8 \
MAX_NUM_BATCHED_TOKENS=4096 \
GRAPH=auto \
GPU_MEMORY_UTILIZATION=0.975 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r12.yml up -d
```

`GRAPH=auto` derives the graph cap from scheduler-reachable verifier rows. For
fixed K5, the required all-decode row envelope is
`MAX_NUM_SEQS * (1 + DSPARK_TOKENS)`. The qualified configuration therefore
captures 48 rows.

## Serving Contract

| Component | Qualified behavior |
|---|---|
| Checkpoint | Official `deepseek-ai/DeepSeek-V4-Flash-0731` revision |
| Target and draft quantization | `deepseek_v4_fp8` |
| Attention | B12X compressed MLA |
| MoE and linear layers | B12X W4A8 |
| KV cache | FP8 compressed target MLA plus FP32 sliding-window compressor state |
| Speculative decoding | Fixed probabilistic DSpark K5 |
| Target/verifier decode | FULL CUDA graph for captured all-decode rows |
| DSpark proposal | FULL CUDA graph for captured verifier rows |
| Prefill | PIECEWISE or uncaptured model path |
| Model loading | InstantTensor `BUFFERED` |

Target-only, fixed K7, and confidence-controlled K7 are implemented by the
entrypoint but are not qualified by this receipt. Their selectors are
`MODE=dspark-mtp0`, `DSPARK_TOKENS=7`, and `DSPARK_DEPTH_MODE=dynamic`.

## Heterogeneous Cache Recycling

DeepSeek-V4-Flash uses more than one attention-cache specification in a single
request lifecycle. The target model stores FP8 MLA pages, while DSpark's
compressed context includes FP32 sliding-window pages. A physical block that
returns to the allocator must contain no state from its previous request
before either cache consumer reuses it.

The scheduler records recycled blocks for every `AttentionSpec` cache group
that requires zeroing. The worker traverses every attention-cache tensor and
zeros the recycled physical pages with a bounded CUDA launch over three axes:

1. physical block identifier;
2. cache tensor identifier;
3. 1024-element page chunk.

Mamba state tensors are outside this contract and are not passed to the
attention-page zeroing kernel. The chunked launch bounds the grid even when a
model exposes many heterogeneous cache tensors. The qualified runtime reported
1,232,956 KV tokens at `GPU_MEMORY_UTILIZATION=0.975`; no release-specific KV
capacity reduction was observed.

Focused validation for [vLLM PR #308](https://github.com/local-inference-lab/vllm/pull/308):

| Condition | Result |
|---|---|
| Scheduler and cache-manager suite | 13 passed |
| CUDA zeroing suite | 9 passed |
| Pre-fix discriminator | Three scheduler and three CUDA cases fail without the generalized contract |
| Ruff check and format | Passed |
| Source composition | Locked trees; no source patches |

## Long-Context Qualification

The exact image ran on two direct-root-port RTX PRO 6000 Blackwell GPUs as
TP2/DCP1. Native CPU KV offload, native filesystem L2, and LMCache were
disabled. Each request used a distinct deterministic corpus so prefix sharing
could not conceal request-state contamination.

| Workload | Prompt tokens | Completion tokens | Result |
|---|---:|---:|---|
| Concurrent strict structured tools | 150,117 / 300,125 | 447 / 236 | Both ended with tool calls; zero integrity indicators |
| Concurrent plain decode with EOS ignored | 150,089 / 299,991 | 4,096 / 4,096 | Both reached the requested length; zero integrity indicators |
| Concurrent near-capacity plain decode | 479,950 / 499,946 | 1,024 / 1,024 | Both reached the requested length; zero integrity indicators |

The validation checks cross-request identity markers, raw-token-like text,
replacement characters, non-printable output, sustained CJK runs, HTTP
errors, runtime errors, and post-run server health. All checks passed. Peak
reported KV usage in the 480k/500k pair was 79.6%, and the server returned to
zero active KV usage after both requests completed.

The visible multi-hour token corruption reported by a community workload was
not reproduced on unmodified Infernal Invocation r11, Gilded Gnosis r33, or
Infernal Invocation r12 during finite deterministic tests. The evidence above
qualifies the recycled-block zeroing invariant. It does not establish that
every long-horizon output anomaly has the same cause.

## Source Merge Contract

Infernal Invocation r12 adds one source responsibility to the r11 merge set:

| Repository | Pull request | Responsibility |
|---|---|---|
| vLLM | [#308](https://github.com/local-inference-lab/vllm/pull/308) | Record and zero recycled blocks for all heterogeneous attention-cache specifications with bounded CUDA launch geometry |

The complete ordered vLLM, B12X, and LMCache pull-request heads are maintained
in [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).
PR #308 is ready for review and is not merged by this release process.

## Qualification Limits

- **Qualified:** official-checkpoint startup, fixed probabilistic K5, FULL
  target and DSpark graph capture, heterogeneous attention-cache block
  recycling, two concurrent requests up to approximately 500k prompt tokens
  each, and forced 4096-token decode at 150k/300k.
- **Implemented:** target-only serving, K7 modes, native vLLM KV offload, and
  LMCache.
- **Unsupported claim:** multi-hour agent stability is not established by the
  approximately 130-second near-capacity run.
- **Unsupported claim:** r12 does not qualify GLM-5.2, TP4, DCP greater than
  one, or alternate DeepSeek checkpoints.
- Performance was not swept. Diagnostic throughput from the long-context runs
  must not be used as a topology-independent benchmark.
