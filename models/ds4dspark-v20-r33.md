# DeepSeek-V4-Flash-0731 DSpark r33

This page documents the validated Gilded Gnosis r33 image for
`deepseek-ai/DeepSeek-V4-Flash-0731`. The 0731 checkpoint contains the native
DSpark draft head. Standard MTP belongs to the older `DeepSeek-V4-Flash`
checkpoint. Fixed probabilistic K5 remains the production default.

## Release Identity

| Item | Value |
|---|---|
| Image | `voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12x06db0f4-fi1ac6942-cu132-20260809-r33` |
| Registry digest | `sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82` |
| Local validation image | `sha256:60944a4ea1fbb2d1f35d7972f685d8fb0b91e77dd5aeca1dcafa3bcc29846d12` |
| Model revision | `9e165c30e2704aec5d9d593cce3eebd58bbef1cb` |
| GG base | `e2666d9a65f41fc376607531453cbd57c4c71016` |
| vLLM integration tree | `fa13d334a2962756f9f7e9b562deb85387359f42` |
| B12X integration tree | `06db0f4b27dbd19eb934da0da27eff7a7c49d8c4` |
| FlashInfer integration | `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| LMCache integration tree | `9a05c8818bae48d15b79c7e876418bb813c08cd0` |
| Runtime | CUDA 13.2.1, PyTorch 2.12.0+cu132, CUTLASS DSL 4.6.0, XGrammar 0.2.5 |
| Default profile | TP2/DCP1, B12X W4A8, fixed probabilistic K5, FP8 DS-MLA KV |

The exact release composition and machine-readable validation are in
[blackwell-llm-docker PR #20](https://github.com/local-inference-lab/blackwell-llm-docker/pull/20)
and its
[r33 receipt](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/validation/gilded-gnosis-v20-r33-remote-gpu.json).
The canonical source landing order is tracked in
[rtx6kpro issue #33](https://github.com/local-inference-lab/rtx6kpro/issues/33).

## Start The Server

Download the immutable release Compose:

```bash
curl -LO https://raw.githubusercontent.com/local-inference-lab/blackwell-llm-docker/426da51285d0666508003b03a75a442139fb7979/examples/docker-compose-ds4-v20-r33.yml
docker compose -f docker-compose-ds4-v20-r33.yml up -d
```

The Compose delegates vLLM arguments to
`/usr/local/bin/serve-ds4-flash.sh`. The measured fixed-K5 profile is:

```bash
GPUS=0,1 \
TP_SIZE=2 DCP_SIZE=1 \
MODE=dspark BACKEND=b12x-a8 \
DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5 \
MAX_NUM_SEQS=16 MAX_MODEL_LEN=131072 GRAPH=auto \
docker compose -f docker-compose-ds4-v20-r33.yml up -d
```

InstantTensor `BUFFERED` loading, FP8 DS-MLA KV, and a persistent JIT cache
are defaults. Disable speculation without changing the 0731 target checkpoint
with `MODE=dspark-mtp0`. Do not select `mtp2` for 0731; that mode uses the
older checkpoint with its standard MTP head.

## All-Reduce Selection

`ALLREDUCE_MODE=auto` resolves by TP size:

| TP | Automatic backend | Diagnostic override |
|---|---|---|
| TP2 | FlashInfer PCIe IPC | `ALLREDUCE_MODE=b12x` |
| TP4 | B12X | `ALLREDUCE_MODE=flashinfer-ipc` |
| TP8 | B12X owner reduction | `ALLREDUCE_MODE=flashinfer-ipc` |

B12X #133 also contains TP2/TP4 remote-push paths. They remain opt-in through
`B12X_PCIE_TP2_REMOTE_PUSH=1` or `B12X_PCIE_TP4_REMOTE_PUSH=1`. Same-host
matched runs showed workload-dependent gains and losses, not a consistent
end-to-end improvement, so r33 does not promote either path to automatic
selection. TP8 owner reduction remains the qualified automatic path.

## DSpark Modes

| Mode | Environment | Status |
|---|---|---|
| Fixed K5 | `DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=5` | Default; best proven mixed-workload choice. |
| Fixed K7 | `DSPARK_DEPTH_MODE=fixed DSPARK_TOKENS=7` | Optional; can win in predictable code phases. |
| Dynamic depth | `DSPARK_DEPTH_MODE=dynamic DSPARK_TOKENS=7` | Optional load-aware policy. |
| Target-only | `MODE=dspark-mtp0` | MTP0 performance and correctness baseline. |

K7 is functional and can approach 500 tok/s in a predictable code phase, but
its acceptance on matched official reasoning is much lower and workload
dependent. Fixed K5 therefore remains the general default.

## CUDA Graph Coverage

| Stage | Execution |
|---|---|
| Target/verifier forward | FULL CUDA graph |
| DSpark proposal | FULL CUDA graph |
| DSpark context-KV compression/update | Dedicated FULL CUDA graph |
| Prefill | PIECEWISE CUDA graph |
| Host metadata and input preparation | Eager host path |
| Rejection sampling and output bookkeeping | Eager post-verification path |

The physical verifier-row requirement is
`MAX_NUM_SEQS * (1 + DSPARK_TOKENS)`. `GRAPH=auto` derives the required
envelope. The r33 TP2/K5 validation captured through 96 rows; TP4/K5 captured
through 192 rows. No device-heavy decode stage relied on an eager fallback.

## Performance

All r33 rows below were measured on physical GPUs 4-7 of `192.168.0.69`.
Those GPUs attach through CPU root ports. Results from the local switched
16-GPU server are not used as references.

The decode client used the standard `llm_decode_bench` encyclopedia prompt,
temperature 1.0, and a 20-second sustained window.

| Fixed probabilistic K5 | C1 tok/s | C4 tok/s | C8 tok/s | Strict acceptance C1/C4/C8 |
|---|---:|---:|---:|---:|
| TP2, FlashInfer PCIe IPC auto | 180.6 | 397.1 | 580.7 | 29.4% / 34.5% / 33.3% |
| TP4, B12X auto | 247.0 | 541.9 | 804.5 | 37.4% / 34.7% / 32.0% |

TP2 uncached 8k prefill reached 12,849 tok/s across 15 samples. A separate
correctness request returned exactly `42`. Acceptance is prompt-dependent, so
the TP2 and TP4 C1 values are not a pure backend microbenchmark.

The same-host TP2/TP4 remote-push policy comparison was:

| Profile | C1 | C4 | C8 | C16 | C32 |
|---|---:|---:|---:|---:|---:|
| TP2 stock | 170.2 | 401.2 | 588.1 | - | - |
| TP2 remote push | 171.3 | 391.7 | 595.3 | - | - |
| TP4 stock | 217.2 | 545.7 | 785.9 | 1,113.4 | 1,750.1 |
| TP4 remote push | 239.7 | 534.0 | 793.1 | 1,132.9 | 1,718.5 |

This is why TP2/TP4 remote push is available for further tuning but is not the
r33 default.

## What r33 Changes

- B12X #133 adds topology-scoped fused all-reduce paths and TP8 owner
  reduction without changing the qualified TP2/TP4 defaults.
- B12X #135 preserves dense GEMM API contracts for block-FP8 callers.
- B12X #136 restores capture-safe K6 small-M dispatch and gates it to exact
  SM120 capability before launch.
- B12X #137 aligns mixed-Trellis execution with the QSRT ABI.
- The release retains the r31 FlashInfer PCIe IPC integration, compressed-MLA
  row contract, FULL graph DSpark context-KV path, native tiered offload,
  reasoning/tool contract, and LMCache stack.

Focused #136 coverage passed 9 tests on an SM120 GPU, including unsupported
architecture gating, numerical comparison, and CUDA graph replay. The complete
release helper, manifest, source-label, launcher-hash, and remote-receipt gates
also passed. No vLLM or B12X PR was merged as part of publishing this image.

## Native L1/L2 KV Offload

Native offload remains optional and independent from LMCache. For example:

```bash
KV_OFFLOADING_SIZE=16 \
NATIVE_L2_PATH=/native-l2 NATIVE_L2_GB=1024 \
NATIVE_L2_HOST_PATH=./cache/ds4-r33/native-l2 \
docker compose -f docker-compose-ds4-v20-r33.yml up -d
```

The helper disables expandable CUDA allocator segments when stable host
registrations are required. See the [r31 runbook](ds4dspark-v20-r31.md) for
the full restart and filesystem-L2 qualification details retained by r33.
