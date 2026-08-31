# DeepSeek-V4-Flash-0731 DSpark r30

This page documents the Gilded Gnosis r30 image for
`deepseek-ai/DeepSeek-V4-Flash-0731`. The release uses the renamed B12X package
and serves the 0731 checkpoint with its native DSpark draft head. Standard MTP
belongs to the older `DeepSeek-V4-Flash` checkpoint and is a separate helper
mode. r30 keeps the r29 performance paths and fixes final-store ordering for
native tiered KV offload.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:gilded-gnosis-v20-vllm20ed7f9-b12x6bc35fd-fi801d57a-cu132-20260807-r30` |
| Registry digest | `sha256:a8a7a05fe8a8599b2ff86cfa2814e4a3b1ec1a28fb154b496f891115f28a6c84` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| GG base | `e2666d9a65f41fc376607531453cbd57c4c71016` |
| vLLM integration tree | `20ed7f98b1ab2c0e6f008a6ad34306fd3b72b33f` |
| B12X integration tree | `6bc35fdb76b6f9d11601009fe413720b461fb444` |
| Runtime | CUDA 13.2.1, PyTorch 2.12.0, B12X 1.1.0, InstantTensor 0.1.9 |
| Default DSpark profile | TP2/DCP1, B12X W4A8, fixed probabilistic K5 |

The immutable source receipts and generated patches are stored in
`patches/releases/gilded-gnosis-v20-r30/` in
[blackwell-llm-docker PR #18](https://github.com/local-inference-lab/blackwell-llm-docker/pull/18).

## Start The Server

Use the release Compose file from the Docker repository:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/2479220615ba04c3e16b69549147e53f35733bc8/examples/docker-compose-ds4-v20-r30.yml
docker compose -f docker-compose-ds4-v20-r30.yml up -d
```

The Compose file delegates all vLLM arguments to
`/usr/local/bin/serve-ds4-flash.sh`. Normal configuration is therefore done
with environment variables rather than a duplicated command line:

```bash
GPUS=0,1 \
TP_SIZE=2 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=16 MAX_MODEL_LEN=131072 \
docker compose -f docker-compose-ds4-v20-r30.yml up -d
```

InstantTensor `BUFFERED` loading is the default. Keep the same mounted JIT
cache across restarts so AOT and CUDA kernel artifacts are reused.

### High-Concurrency Gate

The verifier row envelope grows with concurrency and draft depth. The helper
derives the required graph cap when `GRAPH=auto`:

```bash
MAX_NUM_SEQS=64 MAX_MODEL_LEN=12288 GRAPH=auto \
docker compose -f docker-compose-ds4-v20-r30.yml up -d
```

For fixed K5 this selects 384 physical verifier rows. Fixed K7 requires 512
rows at MNS64. The short `MAX_MODEL_LEN` above is only the reproducible C64
correctness/throughput gate; production context length can be raised according
to the available KV budget.

## Native Tiered KV Offload

Native offload is optional and independent from LMCache.
`KV_OFFLOADING_SIZE` is the total host L1 capacity in GiB across all TP ranks.
For a persistent filesystem L2, pass a compact transfer configuration through
`EXTRA_VLLM_ARGS`:

```bash
export EXTRA_VLLM_ARGS='--kv-transfer-config={"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":"/cache/native-l2","n_read_threads":4,"n_write_threads":4}]}}'

GPUS=0,1 TP_SIZE=2 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=16 GRAPH=128 \
MAX_MODEL_LEN=500000 MAX_NUM_BATCHED_TOKENS=4096 \
KV_OFFLOADING_SIZE=16 SHM_SIZE=32gb PYTHONHASHSEED=0 \
docker compose -f docker-compose-ds4-v20-r30.yml up -d
```

The helper automatically changes `expandable_segments:True` to
`expandable_segments:False` while native offload is enabled. Its shared host
region requires stable registrations. Keep `PYTHONHASHSEED` and the mounted
cache directory stable when L2 objects must survive a container restart. The
release profile was qualified without privileged container access; GPU runtime,
host IPC, memlock, and the mounted cache are sufficient.

r30 fixes a deterministic lifecycle error in this path. The old scheduler
could finalize `TieringOffloadingManager` request state before the following
step built the final partial-block store. Under a deep queue, `prepare_store()`
then raised `KeyError` and terminated EngineCore. The new order builds the
final store first and retains request state until submitted transfers finish.

### 500k/C256 Qualification

The exact profile above was tested on physical GPUs 4-5:

| Gate | Result |
|---|---:|
| Concurrent stress requests | 1,792 / 1,792 passed |
| Peak capacity queue observed | 237 |
| Peak deferred queue observed | 218 |
| Store path | 1,184 operations / 38.93 GB |
| Filesystem replay load path | 408 operations / 18.46 GB |
| Async lookup completions | 454 |
| Final running / waiting queues | 0 / 0 |
| HTTP / EngineCore errors | 0 / 0 |
| Post-stress response | exact `42` |

The unique-prefix wave forced fresh final stores. Replaying the same prefixes
forced filesystem-tier loads, so this gate covers the failing lifecycle and
both transfer directions rather than only startup or cache hits.

## DSpark Modes

| Mode | Environment | Status |
|---|---|---|
| Fixed K5 | `DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5` | Release default; best proven mixed-workload choice. |
| Fixed K7 | `DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=7` | Optional; can win during highly predictable code generation. |
| Dynamic depth | `DSPARK_DEPTH_MODE=dynamic DSPARK_TOKENS=7` | Diagnostic/experimental; correct, but slower in the release sweep. |
| No speculation | `MODE=dspark-mtp0` | 0731 target-only baseline. |

K7 is functional. It reached about 499 tok/s and used all seven positions in a
low-entropy code phase. On matched official-max reasoning, however, strict K7
acceptance was commonly about 19-40%. The apparent public SGLang advantage was
caused by comparing chat-mode SGLang prompts with official-max vLLM prompts;
after matching prompt encoding both were in the same approximately 39.8%
acceptance regime. K5 therefore remains the production default.

## CUDA Graph Coverage

r30 captures all device-heavy DSpark decode stages that currently have a
stable replay contract:

| Stage | Execution |
|---|---|
| Target/verifier model forward | FULL CUDA graph |
| DSpark proposal model | FULL CUDA graph |
| DSpark context-KV compression/update | Dedicated FULL CUDA graph |
| Prefill | PIECEWISE CUDA graph |
| Host metadata and input preparation | Eager host path |
| Rejection sampling and output bookkeeping | Eager path |

The context-KV manager maps the actual row count to the smallest captured
bucket. Padding uses position `0` and slot `-1`, so padded rows cannot write to
live request state. Profiling, dummy runs, unsupported shapes, and non-DSpark
models continue through the eager fallback.

The relevant implementation is the consolidated
[vLLM PR #251](https://github.com/local-inference-lab/vllm/pull/251). It owns
both semantic B12X graph-channel lifecycle and context-KV capture. r30 also
updates the DFlash and autoregressive call sites to GG's current contract:
`channel_id` belongs to `capture()`, not the graph-manager constructor.

## Performance

The matched TP2/DCP1 fixed-K5 A/B below is the inherited r29 performance
qualification. r30 changes offload lifecycle and GG graph-manager integration;
it does not change model arithmetic or the qualified decode path.

| Test | Eager context-KV | FULL context-KV | Delta |
|---|---:|---:|---:|
| CC1 server decode | 182.82 tok/s | 190.68 tok/s | +4.3% |
| CC32 aggregate median | 1,257.73 tok/s | 1,253.69 tok/s | -0.3% |

The final clean release image reproduced `190.66 tok/s` at CC1 and
`1,259.85 tok/s` at CC32. At CC32, target and MoE compute already saturate the
server, so eliminating context-KV launch overhead improves the trace but not
end-to-end throughput. Over four profiled steps:

| Trace item | Eager | FULL context-KV |
|---|---:|---:|
| DSpark proposal time per step | 2.408 ms | 1.510 ms |
| Eager CUDA launches | 296 | 188 |
| GPU span | 68.444 ms | 63.738 ms |

The FULL context-KV path increased estimated graph memory from 2.14 to 2.17
GiB and reduced the matched KV pool by only 74 tokens.

### Inherited r29 Performance Gate

| Gate | Result |
|---|---:|
| Fixed K5 C64 exact final answer | 192/192 |
| Fixed K7 C32 exact final answer | 96/96 |
| K5 CC1 / CC32 | 190.66 / 1,259.85 tok/s |
| K7 smoke CC1 / CC32 | 168.5 / 818.8 tok/s |
| Model load | 80.97 GiB/GPU |
| InstantTensor | 72,317 tensors loaded in about 7 seconds |

The K5 short-context test had 14,001 usable KV tokens. The vLLM metrics also
reported 5,868 raw blocks with block size 4 (23,472 raw block slots); those are
not equivalent to usable hybrid-cache sequence capacity, so use
`kv_cache_size_tokens`, not `num_gpu_blocks * block_size`, for capacity reports.

## What r30 Fixes

- Declares scheduler-reachable compressed-MLA verifier capacity instead of a
  fixed 256-row assumption. This fixes the correctness failure above C24.
- Keeps B12X split planning and scratch allocation on the same physical-row
  contract as vLLM.
- Isolates semantic PCIe graph channels and prewarms the CuTe one-shot path
  before graph capture.
- Captures DSpark context-KV work in FULL CUDA graphs without changing model,
  logits, sampling, or verification math.
- Updates all graph owners to the current GG capture-time channel API and adds
  exact-signature tests to prevent constructor/capture drift.
- Builds native tiered-offload final stores before request-state finalization
  and retains state until the transfer completes.
- Uses the canonical `b12x` package name. Legacy `SPARKINFER_*` variables are
  compatibility aliases, not the preferred spelling.

Rejection sampling is not part of this release's FULL graph coverage. A future
optimization must capture the whole post-verification chain, including RNG
state and variable request bookkeeping; graphing only the approximately 3 us
rejection kernel would not provide a meaningful gain.
