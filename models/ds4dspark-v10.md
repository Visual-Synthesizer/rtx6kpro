# DeepSeek-V4-Flash v10 Fathomless Validation

This page documents the reduced DeepSeek-V4-Flash standard-checkpoint
validation on the `dev/fathomless-firmament` vLLM line. It is the v10 follow-up
to [DeepSeek-V4-Flash and DSpark v9](ds4dspark-v9.md).

The requested fast validation uses only GPUs `0-7`, leaves the existing GLM
container on GPUs `8-15` untouched, and runs with:

```text
TP=4
MAX_NUM_SEQS=1
GRAPH=6
DECODE_CONCURRENCY=1
PREFILL_CONTEXTS=8k,64k
```

The DSpark checkpoint itself was not revalidated in this reduced pass. The
primary target here is the base `DeepSeek-V4-Flash` checkpoint with B12X
standard MTP off and MTP2.

## Image

The DS4 v10 validation uses the same reproducible Fathomless image as the GLM
5.2 v15 page:

```text
voipmonitor/vllm:fathomless-firmament-v15-vllmf5f4af3-b12x90172a5-cu132-20260709
voipmonitor/vllm@sha256:2dbc40a1fd104168226f46eb31f14301967a37aca95fed71fd23ff4f74b10698
```

Runtime version reported by vLLM:

```text
0.11.2.dev279+fathomless.firmament.f5f4af3.b12x90172a5.cu132.20260709
```

Pinned source stack:

| Component | Ref |
|---|---|
| vLLM | `local-inference-lab/vllm codex/ff-v15-mxfp4-online-mxfp8-20260709 @ f5f4af357e26643b355eb1190de7df1163bbcd98` |
| vLLM upstream base | `dev/fathomless-firmament @ 4cf20be8682749d0cca18639304a1693b00ce421` |
| vLLM FF PR | [`#84 Support MXFP4 online MXFP8 dense overlay`](https://github.com/local-inference-lab/vllm/pull/84) |
| B12X | `voipmonitor/b12x codex/ff-v15-cute-compile-fallback-20260709 @ 90172a504e96d246e07cb1ebad3b291532445560` |
| B12X upstream base | `lukealonso/b12x master @ 97b3d642b8ce08ce23184a36882710ce3b60ba13` |
| B12X FF PR | [`lukealonso/b12x#28 CuTe compile fallback`](https://github.com/lukealonso/b12x/pull/28) |
| FlashInfer | `5a73a36a7169ec5533ba474bb9204bed765dd297` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| InstantTensor | `scitix/InstantTensor @ 85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL runtime | `2.30.4`, unified at `/opt/libnccl-local-inference.so.2.30.4` |
| CUDA / PyTorch | CUDA `13.2.1`, PyTorch `2.12.0+cu132` |

Build command:

```bash
cd /root/rtx6kpro
PUSH_IMAGE=1 ./scripts/build-glm52-v15-final-image.sh
```

## Checkpoints

Standard checkpoint:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

DSpark checkpoint from v9, not remeasured here:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7
```

## Runtime Contract

The v10 wrappers are thin defaults over the v9 DS4 helpers:

```text
scripts/run-ds4-v10-server.sh
scripts/run-ds4-v10-sweep.sh
```

They set the Fathomless image above and call the existing v9 launch/sweep
logic. Container names inside the sweep still use the `ds4-v9-*` prefix because
the synchronized wave scheduler is shared with v9.

Common server settings are inherited from `scripts/run-ds4-v9-server.sh`:

```text
--kv-cache-dtype fp8
--block-size 256
--load-format auto
--decode-context-parallel-size 1
--max-model-len 262144
--max-num-batched-tokens 8192
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}
--async-scheduling
--no-scheduler-reserve-full-isl
--enable-chunked-prefill
--enable-flashinfer-autotune
--enable-prompt-tokens-details
--enable-force-include-usage
--enable-request-id-headers
```

B12X common env:

```text
VLLM_USE_B12X_WO_PROJECTION=1
VLLM_USE_B12X_MHC=1
VLLM_USE_B12X_MOE=1
VLLM_USE_B12X_SPARSE_INDEXER=1
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
B12X_MLA_SM120_UNIFIED=1
B12X_MHC_MAX_TOKENS=16384
B12X_DENSE_SPLITK_TURBO=1
B12X_W4A16_TC_DECODE=1
VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
```

Backend rows:

| Backend | Attention | MoE / linear |
|---|---|---|
| `b12x-a16` | `B12X_MLA_SPARSE` | `--moe-backend=b12x --linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=1`, `B12X_MOE_FORCE_A8=0`, `B12X_MOE_FORCE_A16=1` |
| `b12x-a8` | `B12X_MLA_SPARSE` | `--moe-backend=b12x --linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=1`, `B12X_MOE_FORCE_A8=1`, `B12X_MOE_FORCE_A16=0` |
| `b12x-a8-dglin` | `B12X_MLA_SPARSE` | `--moe-backend=b12x`, no `--linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=0`, `B12X_MOE_FORCE_A8=1`, `B12X_MOE_FORCE_A16=0` |

The force paths were verified in logs:

```text
B12X MoE force-A16 enabled: using quant_mode=w4a16.
B12X MoE force-A8 enabled: using quant_mode=w4a8_mx for E8M0 FP4 weights.
```

## Reproduction Commands

The reduced validation was split into two waves so only GPUs `0-7` were used.
Do not put four TP4 cases into one wave on this 16-GPU host unless you also want
the helper to use GPUs `8-15`.

Wave 1, A16 MTP0 and MTP2:

```bash
cd /root/rtx6kpro
OUT=/root/bench-results/ds4-v10-ff-reduced-20260709T040432Z
mkdir -p "$OUT"

OUT="$OUT" \
PROGRESS_FILE="$OUT/progress.log" \
TPS=4 \
BACKENDS=b12x-a16 \
MODES=standard-mtp0,standard-mtp2 \
DECODE_CONCURRENCY=1 \
DECODE_CONTEXTS=0 \
PREFILL_CONTEXTS=8k,64k \
MAX_NUM_SEQS=1 \
GRAPH=6 \
PORT_BASE=7300 \
scripts/run-ds4-v10-sweep.sh
```

Wave 2, A8 full B12X and historical A8+DeepGEMM-linear hybrid:

```bash
cd /root/rtx6kpro
OUT=/root/bench-results/ds4-v10-ff-reduced-20260709T040432Z

OUT="$OUT" \
PROGRESS_FILE="$OUT/progress.log" \
TPS=4 \
BACKENDS=b12x-a8,b12x-a8-dglin \
MODES=standard-mtp0 \
DECODE_CONCURRENCY=1 \
DECODE_CONTEXTS=0 \
PREFILL_CONTEXTS=8k,64k \
MAX_NUM_SEQS=1 \
GRAPH=6 \
PORT_BASE=7340 \
scripts/run-ds4-v10-sweep.sh
```

Result root:

```text
/root/bench-results/ds4-v10-ff-reduced-20260709T040432Z
```

Progress log:

```text
/root/bench-results/ds4-v10-ff-reduced-20260709T040432Z/progress.log
```

## Reduced Validation Results

Sustained decode is aggregate tok/s from `llm_decode_bench.py`, `ctx=0`,
`cc1`, 30 seconds per cell. `coding peak` is median generation-only tok/s over
five Sieve-of-Eratosthenes cc1 runs.

| TP | Backend | Mode | cc1 tok/s | coding peak median | CJK runs | Prefill 8k tok/s | Prefill 64k tok/s |
|---:|---|---|---:|---:|---:|---:|---:|
| 4 | `b12x-a16` | `standard-mtp0` | 176.1 | 176.0 | 0 | 15,146 | 14,595 |
| 4 | `b12x-a16` | `standard-mtp2` | 253.7 | 272.9 | 0 | 14,690 | 14,080 |
| 4 | `b12x-a8` | `standard-mtp0` | 175.8 | 175.6 | 0 | 16,711 | 15,866 |
| 4 | `b12x-a8-dglin` | `standard-mtp0` | 165.3 | 165.2 | 0 | 16,683 | 15,934 |

## v9 Reference

The v9 reference below is from the synchronized full sweep in
`ds4dspark-v9.md`. It used `MAX_NUM_SEQS=64`, graph `256` for MTP0 and graph
`512` for MTP2/MTP3, plus decode concurrencies up to cc64. Therefore the table
is a reference, not a strict apples-to-apples comparison with the reduced v10
`MAX_NUM_SEQS=1`, `GRAPH=6` run.

| TP | Backend | Mode | v9 cc1 | v10 cc1 | v9 8k | v10 8k | v9 64k | v10 64k |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 4 | `b12x-a16` | `standard-mtp0` | 174.6 | 176.1 | 14,360 | 15,146 | 13,894 | 14,595 |
| 4 | `b12x-a16` | `standard-mtp2` | 302.3 | 253.7 | 13,938 | 14,690 | 13,436 | 14,080 |
| 4 | `b12x-a8` | `standard-mtp0` | 174.4 | 175.8 | 15,733 | 16,711 | 15,080 | 15,866 |
| 4 | `b12x-a8-dglin` | `standard-mtp0` | 177.0 | 165.3 | 15,723 | 16,683 | 15,130 | 15,934 |

Readout:

- Full B12X A16 and A8 MTP0 match or slightly exceed the v9 cc1 rows while
  using the intentionally reduced graph settings.
- Prefill 8k/64k is higher than the v9 reference for every measured reduced row.
- The MTP2 reduced decode row is lower than the v9 full-graph reference. A
  follow-up full-graph `MAX_NUM_SEQS=64`, `GRAPH=512` MTP2 check was attempted
  at `/root/bench-results/ds4-v10-ff-mtp2-fullgraph-check-20260709T042200Z`,
  but the server did not reach `/v1/models` after piecewise graph capture and
  was stopped. Treat that as an open full-graph MTP2 issue, not as part of the
  reduced validation result.
- The historical A8+DeepGEMM-linear hybrid still has strong prefill but lower
  cc1 decode than full B12X in this reduced run. The primary all-B12X A8 row is
  the preferred A8 comparison.

## Artifacts

```text
/root/bench-results/ds4-v10-ff-reduced-20260709T040432Z/
/root/bench-results/ds4-v10-ff-reduced-20260709T040432Z/repro/
/root/bench-results/ds4-v10-ff-reduced-20260709T040432Z/progress.log
/root/bench-results/ds4-v10-ff-mtp2-fullgraph-check-20260709T042200Z/
/root/rtx6kpro/scripts/run-ds4-v10-server.sh
/root/rtx6kpro/scripts/run-ds4-v10-sweep.sh
/root/rtx6kpro/scripts/run-ds4-v9-server.sh
/root/rtx6kpro/scripts/run-ds4-v9-sweep.sh
/root/rtx6kpro/scripts/render-ds4-v9-results.py
```

## Caveats

- This page validates the standard `DeepSeek-V4-Flash` checkpoint only. The
  DSpark checkpoint remains documented by v9 until it is explicitly remeasured
  on Fathomless.
- `standard-mtp0` disables speculative decoding. `standard-mtp2` uses the base
  checkpoint MTP heads with two draft tokens and `moe_backend=b12x`.
- The helper scripts assume the model snapshots already exist under
  `/root/.cache/huggingface/hub`. Override `STANDARD_MODEL` or `DSPARK_MODEL`
  if your path differs.
- Use the synchronized sweep helper for comparisons. It launches every server in
  a wave, waits until all are ready, and only then starts benchmark clients.
