# DeepSeek-V4-Flash-0731 Infernal Invocation r2

**Status: qualified.** This page specifies the reproducible serving profile for
`deepseek-ai/DeepSeek-V4-Flash-0731` on RTX PRO 6000 Blackwell GPUs. The
checkpoint contains the DSpark draft head. Fixed probabilistic K5 is the
general-purpose profile; fixed K7 and confidence-controlled K7 are supported
alternatives.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:infernal-invocation-vllm344438d-b12x1584743-fi1ac6942-cu132-20260812-r2` |
| Registry digest | `sha256:2fc077dc7d790d6d27f76e8f7d32ffdc278ff4eb47c927b94d26a7d17d9313cc` |
| Image ID | `sha256:b80779ca73bebb6de11f7611f204112e81ad16ee155feeeed61ee3aa7779a1d5` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| vLLM base | `dev/infernal-invocation@c8d04a543e0e8b0896e60b8b11bec0bb2d780860` |
| vLLM integration tree | `344438d742b3cb3f3bd1851a0e9f33f4ebac64e0` |
| B12X base | `master@184d7d52ad630841d0c6caf962f8b9d36f38992a` |
| B12X integration tree | `1584743fd972ead81619e8f8934cb7bca61571db` |
| LMCache integration tree | `ccccdfc37f108ab674ac0418b5ac5fc1c8b0857e` |
| FlashInfer revision | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| Docker source | [`blackwell-llm-docker@70ea936`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/70ea936) |
| Source merge contract | [`rtx6kpro` issue #67](https://github.com/local-inference-lab/rtx6kpro/issues/67) |
| Runtime | CUDA 13.2.1, PyTorch 2.13.0+cu132, NCCL 2.30.4, CUTLASS DSL 4.6.2, XGrammar 0.2.5 |

The image uses integration locks for vLLM, B12X, and LMCache. Each lock records
the base commit, ordered pull-request heads, resulting tree, and patch digest.
All three locks have an empty `source_patches` list; no untracked source overlay
is part of the runtime.

Infernal Invocation uses its own revision sequence because its vLLM base is
`dev/infernal-invocation`. Gilded Gnosis `v20-r*` pages remain immutable
historical specifications for the `dev/gilded-gnosis` source line.

## Start The Server

Download the immutable Compose file:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/70ea936/examples/docker-compose-ds4-infernal-invocation-r2.yml
docker compose -f docker-compose-ds4-infernal-invocation-r2.yml up -d
```

The Compose defaults to TP2/DCP1, fixed probabilistic K5, B12X W4A8, FP8
compressed MLA KV, InstantTensor `BUFFERED`, and persistent JIT storage.

The qualified TP4/DCP1 performance profile is:

```bash
GPUS=0,1,2,3 \
TP_SIZE=4 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=8 GRAPH=auto \
MAX_MODEL_LEN=131072 MAX_NUM_BATCHED_TOKENS=8192 \
GPU_MEMORY_UTILIZATION=0.95 \
docker compose -f docker-compose-ds4-infernal-invocation-r2.yml up -d
```

`GRAPH=auto` derives the verifier-row envelope from
`MAX_NUM_SEQS * (1 + DSPARK_TOKENS)`. Do not set a smaller graph cap than that
physical requirement.

## DSpark Profiles

| Purpose | Environment | Status |
|---|---|---|
| Fixed probabilistic K5 | `MODE=dspark DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5` | Qualified default |
| Fixed probabilistic K7 | `MODE=dspark DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=7` | Functionally and graph qualified |
| Confidence-controlled K7 | `MODE=dspark DSPARK_DEPTH_MODE=dynamic DSPARK_TOKENS=7 DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE=0` | Functionally and graph qualified |
| Target-only 0731 checkpoint | `MODE=dspark-mtp0` | Qualified cache and target baseline |

The confidence-controlled profile uses compact variable-length verification.
It can reduce useful draft depth when the DSpark confidence policy predicts
that deeper proposals are unlikely to survive. Its benefit depends on prompt
entropy and acceptance; the release does not claim that it outperforms fixed
K5 for every workload.

The standard `DeepSeek-V4-Flash` checkpoint and its MTP head use `MODE=mtp0`,
`MODE=mtp2`, or `MODE=mtp3`. Do not use those mode names to select speculative
depth for the 0731 checkpoint.

## CUDA Graph Contract

| Stage | Execution contract |
|---|---|
| Target/verifier forward | FULL CUDA graph for captured scheduler rows |
| DSpark proposal | FULL CUDA graph for captured rows and draft depths |
| DSpark context-KV compression/update | Dedicated FULL CUDA graph |
| Prefill | PIECEWISE CUDA graph |
| Rejection sampling | Post-verification host/device orchestration outside the model FULL graph |
| Request metadata and output bookkeeping | Host path |

With `MAX_NUM_SEQS=8`, fixed K5 captured eight target graph sizes and eight
DSpark graph sizes through a 48-row envelope. Fixed K7 captured eight target
and eight DSpark graph sizes through a 64-row envelope. Confidence-controlled
K7 captured 30 target graphs and 56 DSpark graphs across scheduler rows and
draft depths.

## TP4 Qualification

The table was measured on GPUs 4-7 of `192.168.0.69`. The four RTX PRO 6000
Blackwell GPUs attach through independent CPU root ports. Results from a PCIe
switch host are not substituted into this table.

The decode client omitted an explicit temperature, ignored EOS, used a
five-second warmup, and measured each cell for 15 seconds.

| Fixed probabilistic K5 | Result |
|---|---:|
| C1 aggregate decode | 234.57 tok/s |
| C1 active user | 231.48 tok/s |
| C1 strict draft acceptance | 41.86% |
| C8 aggregate decode | 781.56 tok/s |
| C8 active user average | 98.46 tok/s |
| C8 strict draft acceptance | 35.99% |
| Uncached 63,988-token prefill | 15,225 tok/s |
| 64k prefill median TTFT | 4.203 s |

Seven prefill samples were used. Arithmetic validation returned `80235` for
`317 * 253 + 34` in fixed K5, fixed K7, and confidence-controlled K7 modes.

## Native KV Offload

Native vLLM offload provides a CPU cache with an optional bounded filesystem
tier. It is independent from LMCache.

```bash
MODE=dspark-mtp0 \
KV_OFFLOADING_SIZE=16 \
NATIVE_L2_GB=1024 \
NATIVE_L2_PATH=/cache/native-kv/8000 \
docker compose -f docker-compose-ds4-infernal-invocation-r2.yml up -d
```

The qualified TP4/DCP1 test used a 1 GiB CPU cache and an 8 GiB filesystem
tier. After six churn requests and local GPU-prefix eviction, replay restored
13,568 of 13,571 prompt tokens. Observed movement was 300,672,000 bytes from
CPU to GPU, 4,125,219,840 bytes from GPU to CPU, 565,407,744 filesystem-read
bytes, and 4,126,273,536 filesystem-write bytes.

## LMCache KV Offload

LMCache uses one cache worker per visible GPU. The generic launcher derives the
worker count and DCP topology, then exposes control, health, and Prometheus
metrics through one HTTP endpoint.

```bash
MODE=dspark-mtp0 \
LMCACHE_MODE=disk \
LMCACHE_L1_GB=24 \
LMCACHE_L2_GB=1024 \
LMCACHE_L2_PATH=/cache/lmcache/8000 \
LMCACHE_HTTP_PORT=8099 \
docker compose -f docker-compose-ds4-infernal-invocation-r2.yml up -d
```

Health and metrics are available at:

```text
http://127.0.0.1:8099/healthcheck
http://127.0.0.1:8099/metrics
```

Port `8099` avoids the commonly occupied Asterisk HTTP port `8089`. Set a
different `LMCACHE_HTTP_PORT` when `8099` is in use.

The qualified TP4/DCP1 test used 4 GiB aggregate L1, 8 GiB filesystem L2,
512-token chunks, and four workers. After L1 eviction, all four workers loaded
52 chunks and replay restored 13,312 of 13,571 prompt tokens. Filesystem usage
was 3,582,699,264 bytes. Seed and replay latency were 1.179 and 0.306 seconds.

Do not enable LMCache and native vLLM KV offload in one profile. Choose one
ownership model for external KV blocks.

## Validation And Limits

- Release composition: 9 tests passed.
- LMCache integration: 222 tests passed and 131 environment-dependent tests
  were skipped.
- Helper, Compose, patched-NCCL, source-lock, and 14 focused Python tests
  passed.
- Qualified logs contain no traceback, engine initialization failure, CUDA
  runtime error, or GPU fault signature.
- K5 has same-host performance evidence. K7 profiles have correctness and CUDA
  graph evidence, not a general throughput promotion over K5.
- KV-offload replay was qualified with speculative decoding disabled so cache
  movement and draft acceptance remained independent variables.
- `VLLM_SERVER_DEV_MODE=1` was used only by validation to clear local prefix
  state. The Compose profile leaves it disabled.

The complete machine-readable evidence is the
[Infernal Invocation r2 qualification receipt](https://github.com/local-inference-lab/blackwell-llm-docker/blob/70ea936/validation/infernal-invocation-r2-remote-gpu.json).
