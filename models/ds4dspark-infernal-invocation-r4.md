# DeepSeek-V4-Flash-0731 Infernal Invocation r4

**Status: qualified.** This page specifies the reproducible serving profile for
`deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell GPUs. Fixed
probabilistic DSpark K5 is the general-purpose profile. Target-only serving,
fixed probabilistic K7, and confidence-controlled K7 use the same image with
explicit runtime settings.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm3226eb7-b12x1584743-fi1ac6942-cu133-torch213-20260812-r4` |
| Registry digest | `sha256:21f048058375ccf00ea555f37addad326a7ee33bc2b4699ae53370f25af4ecb6` |
| Image ID | `sha256:b0cac4ef4037ed8880809df87c14ddc592ef234d59499864e1468448eb928cbf` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@ce5f50f6d01b02336c4207f11277fd7bedacb4d6` |
| vLLM integration tree | `3226eb7ff642702908f502a2402f9d083d16511c` |
| B12X base | `master@184d7d52ad630841d0c6caf962f8b9d36f38992a` |
| B12X integration tree | `1584743fd972ead81619e8f8934cb7bca61571db` |
| LMCache integration tree | `ccccdfc37f108ab674ac0418b5ac5fc1c8b0857e` |
| FlashInfer revision | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| Docker build commit | [`0040f0a`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/0040f0af0670d0e5bb0f6bea6ee7cd2de2990b01) |
| Qualification receipt | [`1429cb3`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/1429cb3010be38e68bcaa069322bc0a587db452f/validation/infernal-invocation-r4-local-gpu.json) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, cuDNN 9.24.0.43, CUTLASS DSL 4.6.2, FlashInfer 0.6.18, XGrammar 0.2.5 |

The image uses immutable integration locks for vLLM, B12X, and LMCache. Each
lock records the base commit, ordered pull-request heads, resulting Git tree,
and integration-patch digest. Every lock has an empty `source_patches` list.

Infernal Invocation revisions identify images built from
`dev/infernal-invocation`. Gilded Gnosis `v20-r*` pages specify a different
source branch and remain separate deployment records.

## Start The Server

Download the immutable Compose profile and start TP2/DCP1 fixed K5:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/1429cb3010be38e68bcaa069322bc0a587db452f/examples/docker-compose-ds4-infernal-invocation-cu133-r4.yml
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r4.yml up -d
```

The profile defaults to B12X W4A8, FP8 compressed MLA KV, InstantTensor
`BUFFERED`, persistent JIT storage, and FULL target and DSpark CUDA graphs.

The qualified TP4/DCP1 K5 profile is:

```bash
GPUS=0,1,2,3 \
TP_SIZE=4 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=8 GRAPH=auto \
MAX_MODEL_LEN=131072 MAX_NUM_BATCHED_TOKENS=8192 \
GPU_MEMORY_UTILIZATION=0.95 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r4.yml up -d
```

`GRAPH=auto` derives the verifier-row envelope from
`MAX_NUM_SEQS * (1 + DSPARK_TOKENS)`. A smaller graph cap cannot represent all
scheduler-reachable verifier rows.

## DSpark Profiles

| Purpose | Environment | Status |
|---|---|---|
| Fixed probabilistic K5 | `MODE=dspark DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5` | Qualified default |
| Fixed probabilistic K7 | `MODE=dspark DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=7` | Functionally and graph qualified |
| Confidence-controlled K7 | `MODE=dspark DSPARK_DEPTH_MODE=dynamic DSPARK_TOKENS=7 DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=0` | Functionally and graph qualified |
| Target-only 0731 checkpoint | `MODE=dspark-mtp0` | Qualified target and cache baseline |

Confidence-controlled K7 uses compact variable-length verification. It can
reduce useful draft depth when the confidence policy predicts that deeper
proposals are unlikely to survive. Its performance depends on workload entropy
and acceptance, so it is not promoted over fixed K5 by this release.

The standard `DeepSeek-V4-Flash` checkpoint and its MTP head use `MODE=mtp0`,
`MODE=mtp2`, or `MODE=mtp3`. Those modes do not select DSpark depth for the
0731 checkpoint.

## CUDA Graph Contract

| Stage | Execution contract |
|---|---|
| Target/verifier decode | FULL CUDA graph for captured all-decode scheduler rows |
| DSpark proposal | FULL CUDA graph for captured rows and draft depths |
| DSpark context-KV compression/update | Dedicated FULL CUDA graph |
| Prefill | PIECEWISE or non-FULL model path |
| Rejection sampling | Post-verification orchestration outside the model FULL graph |
| Request metadata and output bookkeeping | Host path |

The model runner dispatches a FULL uniform-decode graph only when every
scheduled request is in decode state. Token shape alone is insufficient: a
prefill chunk can contain the same number of rows as a speculative decode
batch. The explicit state predicate prevents a prefill batch from replaying a
decode graph with incompatible metadata. The implementation is tracked by
[vLLM PR #298](https://github.com/local-inference-lab/vllm/pull/298) and
[upstream vLLM PR #51865](https://github.com/vllm-project/vllm/pull/51865).

The qualified graph coverage is:

| Profile | Target FULL graphs | DSpark FULL graphs | Context-KV FULL graphs |
|---|---:|---:|---:|
| TP2 target-only, graph cap 4 | 4 | n/a | n/a |
| TP2 fixed K7, graph cap 32 | 4 | 4 | 7 |
| TP2 confidence-controlled K7, graph cap 32 | 17 | 28 | 7 |
| TP4 fixed K5, graph cap 48 | 8 | 8 | 9 |

## Correctness Qualification

The fixed K5 long-agent workload used TP2/DCP1, `MAX_NUM_SEQS=4`, graph cap
24, 40 GiB native CPU KV offload, and a 200 GiB filesystem L2 tier.

| Gate | Result |
|---|---:|
| Estonia long-context requests | 160/160 pass |
| Errors or maximum-token-limit hits | 0 |
| Response-integrity violations | 0 |
| C4 aggregate generation | 132.20 tok/s |
| Mean TTFT | 0.686 s |
| 134,217-token prefill scout | 11,131 tok/s |
| GPU-to-offload movement | 11,303,262,720 bytes |
| Filesystem-L2 writes | 11,317,698,560 bytes |

Two concurrent strict-tool requests used 150,003-token and 300,128-token
prompts. Both returned valid `tool_calls`, retained request-local grammar and
output state, and left the server healthy. A short control request then
returned `DS4 SHORT CONTROL READY`.

The same 160-request workload reproduced two response-integrity failures with
the r3 image
`voipmonitor/vllm:infernal-invocation-vllm6c50b0a-b12x1584743-fi1ac6942-cu133-torch213-20260812-r3`.
The r4 result therefore exercises the interval that exposed the classifier
defect rather than relying on a short smoke test.

## Performance Qualification

The table was measured on isolated GPU subsets of a 16-GPU RTX PRO 6000
Blackwell PCIe-switch host. Direct-root-port and dual-socket systems require
their own topology measurements.

| Profile | Result |
|---|---:|
| TP2 target-only C1 | 151.88 tok/s |
| TP2 target-only C4 | 397.97 tok/s |
| TP4 fixed K5 C8 | 905.68 tok/s |
| TP4 fixed K5 coding median | 404.40 tok/s |
| TP4 fixed K5 coding range | 396.33-428.52 tok/s |
| TP4 fixed K5 8,191-token prefill | 16,444 tok/s |
| TP4 fixed K5 63,984-token prefill | 16,011 tok/s |
| TP4 GPU KV capacity at 131,072 max length | 894,638 tokens |

Arithmetic validation returned `80235` for `317 * 253 + 34` in target-only,
fixed K5, fixed K7, and confidence-controlled K7 modes.

Single-stream DSpark throughput is phase-sensitive because draft acceptance
changes with generated content. Two temperature-zero TP4 K5 C1 samples were
231.74 tok/s at 1.46% strict acceptance and 289.59 tok/s at 47.96% strict
acceptance. Use the coding workload and concurrency measurements for release
comparisons instead of treating one arbitrary long C1 stream as a stable
hardware limit.

## Native KV Offload

Native vLLM offload provides a CPU cache with an optional bounded filesystem
tier:

```bash
KV_OFFLOADING_SIZE=40 \
NATIVE_L2_GB=200 \
NATIVE_L2_PATH=/cache/native-kv/8000 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r4.yml up -d
```

The 160-request K5 qualification exercised both tiers and wrote more than 11
GB to the filesystem tier. Native offload and LMCache are independent KV
ownership models and must not be enabled for one engine at the same time.

## LMCache KV Offload

LMCache uses one cache worker per visible GPU and exposes health, control, and
Prometheus interfaces through one HTTP port:

```bash
MODE=dspark-mtp0 \
LMCACHE_MODE=disk \
LMCACHE_L1_GB=2 \
LMCACHE_L2_GB=8 \
LMCACHE_L2_PATH=/cache/lmcache/8000 \
LMCACHE_HTTP_PORT=8099 \
docker compose -f docker-compose-ds4-infernal-invocation-cu133-r4.yml up -d
```

The TP2/DCP1 qualification seeded a 16,092-token prompt, cleared the local
prefix cache and LMCache L1, and replayed 15,872 tokens from filesystem L2.
Seed latency was 2.949 seconds and replay latency was 0.380 seconds. The
wrapper enforces `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`, which is
required by the native-memory cache connector.

Health and metrics are available at:

```text
http://127.0.0.1:8099/healthcheck
http://127.0.0.1:8099/metrics
```

## Source Merge Contract

The image composes these source responsibilities:

| Repository | Pull requests | Purpose |
|---|---|---|
| vLLM | #285-#293 | Immutable model identity, DS4 launch contracts, B12X tensor contracts, sparse metadata bounds, serving-shape memory profiling, package identity, and hybrid KV recovery |
| vLLM | #294-#296 | Structured-output bitmask widths, grammar termination, and DeepSeek V4 tool-call delimiter handling |
| vLLM | #298 | Decode-state classification for FULL CUDA graph dispatch |
| B12X | #145-#146 | CUTLASS DSL 4.6.2 qualification and the wave-balanced W4A16 FC2 tile |
| LMCache | #7-#17 and #22 | Bounded worker errors, retrieval recovery, durable bounded tiers, filesystem key compatibility, writeback, and hybrid object-group replay |

The merge-ready status and exact head commits are maintained in
[`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67).

## Validation And Limits

- Docker release composition and CUDA 13.3 runtime-contract scripts passed.
- Docker repository tests passed: 27 tests.
- vLLM request-state classification passed 50 focused tests.
- Qualified logs contain no traceback, engine initialization failure, CUDA
  runtime error, or response-integrity violation.
- K7 profiles have exact-arithmetic and CUDA-graph evidence. They do not have a
  general quality or throughput promotion over fixed probabilistic K5.
- K7 functional tests used an 8,192-token maximum model length because the TP2
  memory budget could not hold a 131,072-token KV target together with the K7
  graph envelope.
- LMCache replay was measured without speculative decoding so cache movement
  and draft acceptance remained independent variables.
- A complete context/concurrency performance sweep is not part of the r4
  correctness gate.

Machine-readable evidence is stored in the
[Infernal Invocation r4 qualification receipt](https://github.com/local-inference-lab/blackwell-llm-docker/blob/1429cb3010be38e68bcaa069322bc0a587db452f/validation/infernal-invocation-r4-local-gpu.json).
