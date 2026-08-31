# DeepSeek-V4-Flash-0731 on Infernal Invocation r15

**Status: qualified.** This page specifies target-only and fixed probabilistic
DSpark K5 serving for `deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000
Blackwell GPUs. B12X consumes inactive MoE route identifiers without assigning
padding rows to expert zero.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm068fc8e-b12x96e5d3d-fi1ac6942-cu133-torch213-20260815-r15` |
| Registry digest | `sha256:f1b13c8604b274212e1164def7d4ed7a4cac9e4f7fa06fa1739730195eca4e18` |
| Image ID | `sha256:0a0d27738abb70db77142c9f83267e80a763c0fccf91cadb6a15ea4bafd2b925` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@ad848fc4141f201489db18d5453c50b312245a0a` |
| vLLM integration tree | `068fc8e7270b92077ba753d002da179c865e444d` |
| B12X base | `master@d3fc4bdbc797f6094e12c2009958cd3939c51668` |
| B12X integration tree | `96e5d3d5c2057fa5d4f542e2368951ddbdcb5b42` |
| LMCache integration tree | `5fdf59cfa184bc15dc5414df0bd633da9e49aaae` |
| Image build source | [`2c301121c`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/2c301121c8680f02a91443f502d13ca1fccb51c2) |
| Qualification receipt | [`validation/infernal-invocation-r15-remote-gpu.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r15-remote-gpu.json) |
| Runtime evidence | [`validation/infernal-invocation-r15/runtime-summary.json`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/infernal-invocation-r15/runtime-summary.json) |
| Inactive-route implementation | [B12X PR #214](https://github.com/local-inference-lab/b12x/pull/214) |
| Speculative grammar validation | [vLLM PR #320](https://github.com/local-inference-lab/vllm/pull/320) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, cuBLAS 13.6.0.2, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The vLLM, B12X, and LMCache source locks record each base commit, ordered
pull-request head, resulting Git tree, and patch digest. Every `source_patches`
array is empty.

## Start The Server

Download the qualified Compose profile and start TP2/DCP1 fixed probabilistic
K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/2c301121c8680f02a91443f502d13ca1fccb51c2/examples/docker-compose-ds4-infernal-invocation-cu133-r15.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r15.yml up -d
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
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r15.yml up -d
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
| Inactive MoE routes | Native B12X elimination with `VLLM_MOE_SKIP_PADDING=1` |
| KV cache | FP8 compressed target MLA plus FP32 sliding-window compressor state |
| Speculative decoding | Fixed probabilistic DSpark K5 |
| Structured output | XGrammar 0.2.5 with pre-commit speculative-prefix validation |
| Target and verifier decode | FULL CUDA graph for captured all-decode rows |
| DSpark proposal | FULL CUDA graph for captured verifier rows |
| Prefill | PIECEWISE or uncaptured model path |
| Model loading | InstantTensor `BUFFERED` |

Target-only, fixed K7, and confidence-controlled K7 are implemented by the
entrypoint but are outside the r15 physical-GPU receipt. Their selectors are
`MODE=dspark-mtp0`, `DSPARK_TOKENS=7`, and `DSPARK_DEPTH_MODE=dynamic`.

## Inactive MoE Routes

Scheduler padding uses route identifier `-1` to state that a row has no active
expert. B12X PR #214 at
`321c24a7ef60174cd6131d932f43bb84a4f3a60f` enforces these invariants:

- inactive identifiers do not contribute to expert histograms;
- compact route storage and weight access contain only valid expert IDs;
- a token without an active route skips input quantization and route fanout;
- valid routes retain the existing arithmetic and output layout.

`VLLM_MOE_SKIP_PADDING=1` is the runtime default. Set
`VLLM_MOE_SKIP_PADDING=0` only as a compatibility discriminator; it restores
padded-row execution and gives up the optimization.

vLLM PR #291 is not part of this image and must not be merged for this
contract. It maps inactive rows to expert zero, which avoids invalid IDs but
performs expert-zero work for padding. B12X PR #214 consumes the inactive
identifier directly.

## Speculative Structured Output

The scheduler validates every accepted speculative token against the effective
structured-output grammar before the token enters request history. An accepted
block can cross the reasoning-end marker that activates a deferred grammar or
continue after a grammar reaches terminal state, so one mask created before
verification cannot authorize the complete block.

The scheduler commits the longest grammar-valid prefix and rolls back the
rejected suffix. Grammar state advances exactly once when the prefix is
committed. The contract is implemented by vLLM PR #320 and complements PRs
[#294](https://github.com/local-inference-lab/vllm/pull/294),
[#295](https://github.com/local-inference-lab/vllm/pull/295), and
[#302](https://github.com/local-inference-lab/vllm/pull/302).

## Qualification Evidence

Two direct-root-port RTX PRO 6000 Blackwell GPUs ran TP2/DCP1 with
`MAX_NUM_SEQS=8`, `MAX_NUM_BATCHED_TOKENS=4096`, and
`MAX_MODEL_LEN=131072`.

| Gate | Result |
|---|---|
| B12X physical-GPU tests | 29 passed across dynamic Trellis, W4A8 dynamic, and standard MoE corpora |
| Docker release tests | 15 passed; DeepSeek and three GLM Compose profiles passed static validation |
| Target-only CUDA graphs | FULL capture passed |
| Target-only CC1 | 125.7 aggregate tok/s over 20 seconds |
| Fixed probabilistic K5 CUDA graphs | FULL target and draft capture passed |
| Fixed probabilistic K5 CC1 | 191.3 aggregate tok/s; 184.8 active-user tok/s over 20 seconds |
| Arithmetic content probe | Exact response `{"a":42,"b":126}` |
| Runtime errors after startup | 0 |

The isolated B12X PR #214 A/B composition used commit
`321c24a7ef60174cd6131d932f43bb84a4f3a60f` on the r14 dependency stack and
excluded vLLM PR #291:

| Workload | Result |
|---|---|
| Target-only C1/C4/C8 | 125.8 / 339.9 / 481.1 tok/s |
| Target-only 8K/64K prefill | 12,646 / 13,195 tok/s |
| Fixed probabilistic K5 CC1 | 199.3 tok/s |

DSpark aggregate throughput depends on the generated trajectory and draft
acceptance. Compare DSpark speed only with matched prompts, sampling, and
acceptance accounting.

## Source Merge Contract

| Repository | Pull request | Responsibility |
|---|---|---|
| B12X | [#214](https://github.com/local-inference-lab/b12x/pull/214) | Eliminate inactive dynamic-MoE routes and skip producer work for tokens without an active route |
| vLLM | [#309](https://github.com/local-inference-lab/vllm/pull/309) | Materialize direct MLA DCP workspaces after deferred model weights reach CUDA |
| vLLM | [#320](https://github.com/local-inference-lab/vllm/pull/320) | Validate speculative structured-output prefixes before scheduler commit |

The complete ordered vLLM, B12X, and LMCache pull-request heads are maintained
in [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).
The release process does not merge vLLM or B12X pull requests.

## Qualification Limits

- **Qualified:** TP2/DCP1 target-only and fixed probabilistic DSpark K5,
  B12X W4A8, FP8 compressed MLA KV, `MAX_NUM_SEQS=8`,
  `MAX_MODEL_LEN=131072`, native KV offload disabled, and LMCache disabled.
- **Implemented:** fixed and dynamic K7, native vLLM KV offload, LMCache, the
  1,048,576-token launcher envelope, and the GLM-5.2 profiles included in the
  image.
- **Unsupported by the r15 receipt:** DCP greater than one, TP other than two,
  native KV offload, LMCache, and the 1,048,576-token serving profile.
- Performance measurements use direct-root-port GPUs and are not directly
  comparable with switched-PCIe results.
- A fresh cache compiles context-dependent B12X kernels during the first long
  request. Persist the release-scoped `/cache` mount.
