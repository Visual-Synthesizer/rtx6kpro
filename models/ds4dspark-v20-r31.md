# DeepSeek-V4-Flash-0731 DSpark r31

This page documents the validated Gilded Gnosis r31 image for
`deepseek-ai/DeepSeek-V4-Flash-0731`. The 0731 checkpoint contains the native
DSpark draft head; standard MTP belongs to the older
`DeepSeek-V4-Flash` checkpoint. Fixed probabilistic K5 remains the production
default.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12xacee6e5-fi1ac6942-cu132-20260807-r31` |
| Registry digest | `sha256:3230c25ff95f8678a8eeb52a463f0d3b9f96f6ad550418cc51ea12177a55b41c` |
| Local validation image | `sha256:b162476b0b3432096e9dd1d0b0d8c825ba71bf33635423c511d9bac533b12a9f` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| GG base | `e2666d9a65f41fc376607531453cbd57c4c71016` |
| vLLM integration tree | `fa13d334a2962756f9f7e9b562deb85387359f42` |
| B12X integration tree | `acee6e504209068bd0cbb01cb2b98966bddcf042` |
| FlashInfer integration | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| LMCache integration tree | `9a05c8818bae48d15b79c7e876418bb813c08cd0` |
| Runtime | CUDA 13.2.1, PyTorch 2.12.0+cu132, B12X 1.1.0, FlashInfer 0.6.18+cu132, InstantTensor 0.1.9 |
| Default profile | TP2/DCP1, B12X W4A8, fixed probabilistic K5, FP8 DS-MLA KV |

The exact release composition and machine-readable validation are in
[blackwell-llm-docker PR #19](https://github.com/local-inference-lab/blackwell-llm-docker/pull/19).
The canonical source landing order is tracked in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).

## Start The Server

Download the immutable release Compose:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/d281c51cf494cfcac8eee5ce6c14c6b112b07771/examples/docker-compose-ds4-v20-r31.yml
docker compose -f docker-compose-ds4-v20-r31.yml up -d
```

The Compose delegates vLLM arguments to
`/usr/local/bin/serve-ds4-flash.sh`. Configure normal serving choices through
environment variables:

```bash
GPUS=0,1 \
TP_SIZE=2 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=16 MAX_MODEL_LEN=131072 GRAPH=auto \
docker compose -f docker-compose-ds4-v20-r31.yml up -d
```

InstantTensor `BUFFERED` loading and `PYTHONHASHSEED=0` are defaults. Preserve
the mounted JIT cache between restarts.

### Target-Only Baseline

Disable speculation without changing the 0731 target checkpoint:

```bash
MODE=dspark-mtp0 \
docker compose -f docker-compose-ds4-v20-r31.yml up -d
```

Do not select `mtp2` for 0731. That mode is only meaningful for the older
checkpoint with its MTP head.

## All-Reduce Selection

`ALLREDUCE_MODE=auto` resolves by TP size:

| TP | Automatic backend | Explicit override |
|---|---|---|
| TP2 | FlashInfer PCIe IPC | `ALLREDUCE_MODE=b12x` |
| TP4 or larger | B12X | `ALLREDUCE_MODE=flashinfer-ipc` |

The policy is reversible and logs the chosen backend. The two TP2 paths are
close at high concurrency on the validation host. B12X was faster at C1 and
prefill in the final target-only gate, so latency/prefill-focused deployments
should benchmark `ALLREDUCE_MODE=b12x` as well. Keep the rest of the recipe
identical when comparing them.

## DSpark Modes

| Mode | Environment | Status |
|---|---|---|
| Fixed K5 | `DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5` | Default; best proven mixed-workload choice. |
| Fixed K7 | `DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=7` | Optional; can win in predictable code phases. |
| Dynamic depth | `DSPARK_DEPTH_MODE=dynamic DSPARK_TOKENS=7` | Diagnostic; correct but not selected by the release sweep. |
| Target-only | `MODE=dspark-mtp0` | MTP0 performance/correctness baseline. |

K7 is functional and has reached about 499 tok/s while using all seven draft
positions in a low-entropy code phase. On matched official-max reasoning its
strict acceptance was commonly about 19-40%. The earlier apparent SGLang
advantage was a prompt mismatch; matching official-max prompt encoding put
both runtimes in the same approximately 39.8% acceptance regime. This is why
r31 keeps fixed K5 as the general default.

### High-Concurrency Row Envelope

The physical verifier-row requirement is:

```text
max_num_seqs * (1 + draft_tokens)
```

At MNS64, K5 requires 384 rows and K7 requires 512. `GRAPH=auto` derives the
correct cap. The r31 release gate used TP4/K5/MNS64:

```bash
GPUS=0,1,2,3 TP_SIZE=4 DCP_SIZE=1 \
MAX_NUM_SEQS=64 GRAPH=auto MAX_MODEL_LEN=196608 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
docker compose -f docker-compose-ds4-v20-r31.yml up -d
```

The target, proposal, and context-KV graph families all captured the complete
384-row envelope. There was no row-capacity eager fallback.

## CUDA Graph Coverage

| Stage | Execution |
|---|---|
| Target/verifier forward | FULL CUDA graph |
| DSpark proposal | FULL CUDA graph |
| DSpark context-KV compression/update | Dedicated FULL CUDA graph |
| Prefill | PIECEWISE CUDA graph |
| Host metadata/input preparation | Eager host path |
| Rejection sampling/output bookkeeping | Eager path |

The device-heavy decode path is captured. Rejection sampling is still part of
the variable host-side post-verification chain. Capturing its small device
kernel alone would not remove the host bookkeeping or provide a useful
end-to-end gain.

## Native L1/L2 KV Offload

Native offload is optional and independent from LMCache.
`KV_OFFLOADING_SIZE` is total host L1 capacity in GiB. r31 adds direct
environment-only filesystem L2 configuration:

```bash
GPUS=0,1 TP_SIZE=2 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=16 GRAPH=128 \
MAX_MODEL_LEN=500000 MAX_NUM_BATCHED_TOKENS=4096 \
KV_OFFLOADING_SIZE=16 \
NATIVE_L2_PATH=/native-l2 NATIVE_L2_GB=1024 \
NATIVE_L2_HOST_PATH=./cache/ds4-r31/native-l2 \
SHM_SIZE=32gb \
docker compose -f docker-compose-ds4-v20-r31.yml up -d
```

`NATIVE_L2_PATH` is the path inside the container and must be paired with
`NATIVE_L2_GB`. `NATIVE_L2_HOST_PATH` controls the persistent host mount. The
helper builds the transfer JSON and disables expandable CUDA allocator
segments because the shared host region requires stable registrations.
Privileged container access is not required.

The restart gate used a decimal 4.5 GiB L1 and bounded 4 GiB filesystem L2.
After a full engine restart discarded GPU and process-local L1 state, replaying
the same 32k prompt loaded 303,586,560 bytes from L2 and completed in 0.415 s.

## Performance

All final and reference numbers in this section come from `192.168.0.69`,
physical GPUs 4-7. Those GPUs are attached through CPU root ports. No number
from the local 16-GPU PCIe-switch host is used as a baseline.

### Target-Only

| Profile | C1 tok/s | C32 tok/s | Prefill 8k tok/s | Prefill 64k tok/s |
|---|---:|---:|---:|---:|
| r31 TP2, FlashInfer PCIe IPC | 126.8 | 1,139.5 | 13,366 | 12,669 |
| r31 TP2, B12X | 129.9 | 1,135.7 | 14,197 | 13,421 |
| r31 TP4, B12X | 148.4 | 1,511.0 | 16,360 | 15,511 |
| Previous TP4 B12X, same host | 144.5 | 1,499.2 | 15,406 | 14,721 |

Only the last two rows are the direct r31 regression comparison. TP2 and TP4
rows describe different parallelism and should not be compared as a backend
A/B.

### Fixed K5 TP4

| Gate | Result |
|---|---:|
| GPU KV capacity | 797,049 tokens |
| Sustained C64 aggregate decode | 2,540.5 tok/s |
| Strict draft acceptance in C64 window | 31.36% |
| Long-context Estonia | 64/64 pass |
| Output-cap hits / runtime errors | 0 / 0 |
| Estonia completion p50 / p90 | 1,918 / 3,475 tokens |
| 134,217-token prefill scout | 13,352 tok/s |

Acceptance is workload-dependent; the Estonia gate is a correctness and
high-concurrency stability test, not a universal acceptance estimate.

## What r31 Changes

- Builds FlashInfer from the qualified current source plus upstream PR #4393
  and exposes PCIe IPC all-reduce through vLLM.
- Guards persistent FlashInfer decode wrappers by their planned query length.
- Emits packed UE8M0 scales correctly from compiled QuantFP8 and preserves
  activation dtype for int32-packed MLA weights.
- Uses vLLM's canonical speculative `attention_backend` field for the DSpark
  draft backend.
- Registers supported GG backend controls instead of reporting them as unknown
  environment variables.
- Deduplicates lockstep native-offload cleanup, bounds filesystem storage, and
  treats stale secondary-tier hits as misses that can be recomputed.
- Prewarms target and native-MTP mixed-Trellis route packing before KV sizing.
- Retains the r30 final-store ordering fix and all r29 compressed-MLA/FULL-graph
  row-capacity work.

The complete pinned PR list, dependencies, exclusions, and test evidence are
in [issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).
