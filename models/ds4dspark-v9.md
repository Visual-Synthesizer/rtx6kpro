# DeepSeek-V4-Flash and DSpark v9

This page documents the DS4 standard-checkpoint and DSpark v9 validation on the
Eldritch Enlightenment line. The DSpark rows compare the new vLLM+B12X branches
against the v8 DSpark rows. The standard-checkpoint rows add the requested MTP
off, MTP2, and MTP3 comparison.

The tested checkpoints are:

```text
deepseek-ai/DeepSeek-V4-Flash
deepseek-ai/DeepSeek-V4-Flash-DSpark
```

Standard MTP rows use the base checkpoint with `method=mtp` and either `2` or
`3` draft tokens. `standard-mtp0` disables speculative decoding completely.
DSpark uses `method=dspark` with its native block size of `5` draft tokens.

## What Changed From v8

- vLLM now uses `dev/eldritch-enlightenment` at
  `45c1582e9b80ba83e71c3a6458e71da4736fbdc4`.
- The image applies the local warmup fallback patch
  `vllm-b12x-indexer-warmup-fallback-20260704.patch` (`c1441b5...`) so B12X
  decode warmup has `fused_indexer_decode_warmup_rows` available.
- B12X uses current master at
  `f3686b555d639823b276c2080f173145eed7f007`, including the A8 force fix.
- The sweep helper now launches each TP wave, waits until every server in the
  wave is ready, and only then starts decode on all instances. This avoids the
  invalid mixed boot/decode run that produced the old `132-133 tok/s` no-MTP
  B12X A8 rows.
- The launch helper can pin TP2/TP4 GPU groups to stable non-overlapping CPU
  core ranges with `ENABLE_TOPO_PIN=1`. This host has one NUMA memory node, so
  every pinned group uses `--cpuset-mems=0`; the synchronized sweep used this
  path.
- The image defaults now enable OpenAI usage/request metadata:
  `--enable-prompt-tokens-details`, `--enable-force-include-usage`, and
  `--enable-request-id-headers`.
- The image and helper set `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1`. With this
  attention-aware memory profile, DSpark TP2 needs `gpu_memory_utilization=0.93`
  to keep `max_model_len=262144`.
- B12X remains split into explicit `A16`, `A8`, and `A8 + DeepGEMM linear`
  rows. The v8 `b12x` row is effectively the `A16` comparison point.

## Docker Image

```text
voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704
voipmonitor/vllm@sha256:7703639ae9532759d180f26b649c4dd10064a84e6b7bb1767510fab900e6c468
```

Local image config digest:

```text
sha256:c2b6c780aadc7c3cc5e93a32ad4293d2aec1d17934e1eb3fc71e41fbb61a1297
```

Runtime version:

```text
0.11.2.dev279+eldritch.enlightenment.v45c1582.b12xf3686b5.patchc1441b5.fi25dd814.cu132.20260704
```

Component pins from image labels; the patch hash is the verified local file SHA:

| Component | Commit / branch |
|---|---|
| vLLM | `dev/eldritch-enlightenment` @ `45c1582e9b80ba83e71c3a6458e71da4736fbdc4` |
| vLLM patch | `vllm-b12x-indexer-warmup-fallback-20260704.patch` @ `c1441b53348d57e92782f3e07c379cda0d01caaedaf83b9b1b1bbaf3e44a19be` |
| B12X | `master` @ `f3686b555d639823b276c2080f173145eed7f007` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` (`nv_dev`) |
| NCCL | `2.30.4`, `local-inference-lab/nccl-canonical`, `canonical/cu132-nccl2304-amd-noxml` |
| CUDA / PyTorch | CUDA `13.2.1`, PyTorch `2.12.0+cu132` |

Installed package versions:

```text
vllm 0.11.2.dev279+eldritch.enlightenment.v45c1582.b12xf3686b5.patchc1441b5.fi25dd814.cu132.20260704
b12x 0.23.0
flashinfer-python 0.6.13+cu132
deep_gemm 2.5.0+2073ddb
torch 2.12.0+cu132
```

## Build Command

The image was built from the existing DSpark TP4 CUDA 13.2 Docker helper, with
source commits pinned and the local vLLM patch applied:

```bash
cd /root/vllm/blackwell-llm-docker

sha256sum -c <<'EOF'
c1441b53348d57e92782f3e07c379cda0d01caaedaf83b9b1b1bbaf3e44a19be  patches/vllm-b12x-indexer-warmup-fallback-20260704.patch
EOF

IMAGE=voipmonitor/vllm:eldritch-enlightenment-v45c1582-b12xf3686b5-pc1441b5-cu132-20260704 \
BUILD_BASE_IMAGE=0 \
PIN_SOURCE_COMMITS=1 \
VLLM_REPO=https://github.com/local-inference-lab/vllm.git \
VLLM_REF=dev/eldritch-enlightenment \
VLLM_COMMIT=45c1582e9b80ba83e71c3a6458e71da4736fbdc4 \
LAUNCHER_REPO=https://github.com/local-inference-lab/vllm.git \
LAUNCHER_REF=dev/eldritch-enlightenment \
LAUNCHER_COMMIT=45c1582e9b80ba83e71c3a6458e71da4736fbdc4 \
B12X_REPO=https://github.com/lukealonso/b12x.git \
B12X_REF=master \
B12X_COMMIT=f3686b555d639823b276c2080f173145eed7f007 \
VLLM_PATCH_FILE=vllm-b12x-indexer-warmup-fallback-20260704.patch \
VLLM_BUILD_VERSION=0.11.2.dev279+eldritch.enlightenment.v45c1582.b12xf3686b5.patchc1441b5.fi25dd814.cu132.20260704 \
./build-eldritch-enlightenment-dspark-tp4-cu132.sh
```

No separate tee build log was retained for this patch rebuild; the pushed image
digest and labels above are the canonical build identity.

## Models

Standard checkpoint:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/6976c7ff1b30a1b2cb7805021b8ba4684041f136
```

DSpark checkpoint:

```text
/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-DSpark/snapshots/913f0657a874f76844e2e91cbe706dbcaceeb6d7
```

DSpark checkpoint settings:

```text
dspark_block_size=5
dspark_target_layer_ids=[40,41,42]
n_mtp_layers=3
dspark_noise_token_id=128799
dspark_markov_rank=256
```

## Runtime Matrix

| Mode | Checkpoint | Speculative config | Graph cap |
|---|---|---|---:|
| `standard-mtp0` | `DeepSeek-V4-Flash` | none | 256 |
| `standard-mtp2` | `DeepSeek-V4-Flash` | `{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}` plus `moe_backend=b12x` on B12X rows | 512 |
| `standard-mtp3` | `DeepSeek-V4-Flash` | `{"method":"mtp","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}` plus `moe_backend=b12x` on B12X rows | 512 |
| `dspark` | `DeepSeek-V4-Flash-DSpark` | `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}` | 512 |

| Backend | Attention | MoE / linear |
|---|---|---|
| `b12x-a16` | `B12X_MLA_SPARSE` | `--moe-backend=b12x --linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=1`, `B12X_MOE_FORCE_A8=0`, `B12X_MOE_FORCE_A16=1` |
| `b12x-a8` | `B12X_MLA_SPARSE` | `--moe-backend=b12x --linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=1`, `B12X_MOE_FORCE_A8=1`, `B12X_MOE_FORCE_A16=0` |
| `b12x-a8-dglin` | `B12X_MLA_SPARSE` | `--moe-backend=b12x`, no `--linear-backend=b12x`, `VLLM_USE_B12X_FP8_GEMM=0`, `B12X_MOE_FORCE_A8=1`, `B12X_MOE_FORCE_A16=0` |
| `lucifer-default` | `FLASHINFER_MLA_SPARSE_DSV4` | default DS4 MoE path, B12X PCIe one-shot all-reduce for small decode tensors |
| `lucifer-cutlass` | `FLASHINFER_MLA_SPARSE_DSV4` | `--kernel-config.moe_backend=flashinfer_cutlass`, B12X PCIe one-shot all-reduce for small decode tensors |

Common B12X env:

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

OpenAI server defaults are enabled in the image and explicitly passed by the
helper:

```text
--enable-prompt-tokens-details
--enable-force-include-usage
--enable-request-id-headers
```

The B12X force modes were verified in the server logs:

```text
B12X MoE force-A16 enabled: using quant_mode=w4a16.
B12X MoE force-A8 enabled: using quant_mode=w4a8_mx for E8M0 FP4 weights.
```

## Launch Helper

The recommended v9 launch path is:

```bash
cd /root/rtx6kpro

IMAGE=voipmonitor/vllm@sha256:7703639ae9532759d180f26b649c4dd10064a84e6b7bb1767510fab900e6c468 \
NAME=ds4-v9 \
PORT=8000 \
GPUS=0,1,2,3 \
TP=4 \
BACKEND=b12x-a8 \
MODE=dspark \
MAX_NUM_SEQS=64 \
ENABLE_TOPO_PIN=1 \
scripts/run-ds4-v9-server.sh
```

Examples:

```bash
# Full B12X A16, closest to the v8 B12X row.
TP=4 GPUS=0,1,2,3 BACKEND=b12x-a16 MODE=dspark scripts/run-ds4-v9-server.sh

# Full B12X A8.
TP=4 GPUS=0,1,2,3 BACKEND=b12x-a8 MODE=dspark scripts/run-ds4-v9-server.sh

# B12X attention+MoE A8 with DeepGEMM FP8 linear.
TP=4 GPUS=0,1,2,3 BACKEND=b12x-a8-dglin MODE=dspark scripts/run-ds4-v9-server.sh

# Lucifer CUTLASS reference path.
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=dspark scripts/run-ds4-v9-server.sh

# Standard checkpoint with MTP disabled.
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=standard-mtp0 scripts/run-ds4-v9-server.sh

# Standard checkpoint with MTP2 / MTP3.
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=standard-mtp2 scripts/run-ds4-v9-server.sh
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=standard-mtp3 scripts/run-ds4-v9-server.sh
```

The helper uses `gpu_memory_utilization=0.93` for DSpark unless `GPU_MEM` is
overridden. This is required with `VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1` at
`max_model_len=262144`.

With `ENABLE_TOPO_PIN=1`, TP2 pairs and TP4 groups are pinned to fixed
non-overlapping CPU core ranges, and memory stays pinned to NUMA node `0` on
this one-node host. Override `PIN_CPUSET_CPUS` or `PIN_CPUSET_MEMS` for a
different topology.

## Full Sweep Commands

The v9 benchmark was run as a single synchronized full sweep on the 16-GPU host.
Each TP wave launched all instances first, waited for all servers to become
ready, and only then started the parallel decode clients. This is the valid run
for the tables below.

```bash
cd /root/rtx6kpro

OUT=/root/bench-results/ds4-v9-refresh-pc1441b5-syncwave-20260704-102844 \
IMAGE=voipmonitor/vllm@sha256:7703639ae9532759d180f26b649c4dd10064a84e6b7bb1767510fab900e6c468 \
PROGRESS_FILE=/root/vllm/prubezne_vysledky \
TPS=2,4 \
BACKENDS=b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass \
MODES=standard-mtp0,standard-mtp2,standard-mtp3,dspark \
MAX_NUM_SEQS=64 \
DECODE_CONCURRENCY=1,16,32,64 \
DECODE_CONTEXTS=0 \
DECODE_DURATION=30 \
PREFILL_CONTEXTS=8k,64k,128k \
PREFILL_DURATION=10 \
PORT_BASE=7100 \
STARTUP_TIMEOUT=2400 \
SYNC_WAVE_READY=1 \
ENABLE_TOPO_PIN=1 \
scripts/run-ds4-v9-sweep.sh
```

Progress log:

```text
/root/vllm/prubezne_vysledky
```

The earlier `/root/bench-results/ds4-v9-refresh-pc1441b5-20260704-101250`
artifact is intentionally ignored. It started some decode clients before every
server in the wave was ready and reproduced the invalid `132-133 tok/s` B12X A8
no-MTP number.

The current `scripts/run-ds4-v9-sweep.sh` defaults are set to this full matrix
and image. Every new sweep copies the launcher, sweep script, result renderer,
the local vLLM patch, image labels, script hashes, git status, and NVIDIA
topology into `$OUT/repro/`.

To regenerate the markdown result tables from any completed sweep directory:

```bash
cd /root/rtx6kpro
scripts/render-ds4-v9-results.py /root/bench-results/ds4-v9-refresh-pc1441b5-syncwave-20260704-102844
```

## Decode Throughput

Sustained decode is aggregate tok/s from `llm_decode_bench.py`, `ctx=0`, 30
seconds per cell. `coding peak` is the median generation-only tok/s over five
Sieve-of-Eratosthenes cc1 runs; every valid row had `0` CJK runs.

### DSpark Checkpoint

| TP | Backend | Mode | cc1 tok/s | cc16 tok/s | cc32 tok/s | cc64 tok/s | coding peak median | CJK runs |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | b12x-a16 | dspark | 219.1 | 1029.3 | 826.6 | 2065.4 | 305.8 | 0 |
| 2 | b12x-a8 | dspark | 216.9 | 1014.5 | 1051.7 | 2149.9 | 269.5 | 0 |
| 2 | b12x-a8-dglin | dspark | 193.8 | 971.4 | 948.3 | 2143.1 | 263.9 | 0 |
| 2 | lucifer-default | dspark | 201.5 | 1049.4 | 1576.7 | 2296.3 | 286.2 | 0 |
| 2 | lucifer-cutlass | dspark | 239.5 | 1154.2 | 1744.7 | 2500.0 | 314.7 | 0 |
| 4 | b12x-a16 | dspark | 292.4 | 1446.3 | 2090.1 | 2616.7 | 389.4 | 0 |
| 4 | b12x-a8 | dspark | 268.6 | 1399.5 | 2116.9 | 2367.7 | 373.5 | 0 |
| 4 | b12x-a8-dglin | dspark | 251.7 | 1379.4 | 2108.8 | 1058.7 | 333.9 | 0 |
| 4 | lucifer-default | dspark | 274.7 | 1514.7 | 2268.5 | 3128.6 | 359.6 | 0 |
| 4 | lucifer-cutlass | dspark | 277.9 | 1681.5 | 2452.0 | 3326.0 | 393.3 | 0 |

### Standard Checkpoint

| TP | Backend | Mode | cc1 tok/s | cc16 tok/s | cc32 tok/s | cc64 tok/s | coding peak median | CJK runs |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | b12x-a16 | standard-mtp0 | 142.6 | 845.4 | 1264.5 | 1884.6 | 141.9 | 0 |
| 2 | b12x-a16 | standard-mtp2 | 227.0 | 1137.4 | 1669.8 | 2508.8 | 237.7 | 0 |
| 2 | b12x-a16 | standard-mtp3 | 213.1 | 1032.3 | 1552.4 | 2304.1 | 234.4 | 0 |
| 2 | b12x-a8 | standard-mtp0 | 140.7 | 767.6 | 1193.4 | 1838.2 | 140.2 | 0 |
| 2 | b12x-a8 | standard-mtp2 | 217.4 | 1083.4 | 1620.5 | 2568.1 | 229.4 | 0 |
| 2 | b12x-a8 | standard-mtp3 | 197.7 | 1000.0 | 1518.7 | 2383.0 | 221.0 | 0 |
| 2 | b12x-a8-dglin | standard-mtp0 | 140.8 | 774.7 | 1175.4 | 1820.5 | 140.5 | 0 |
| 2 | b12x-a8-dglin | standard-mtp2 | 227.5 | 1085.1 | 1622.6 | 2592.8 | 235.9 | 0 |
| 2 | b12x-a8-dglin | standard-mtp3 | 212.3 | 1018.0 | 1552.5 | 2413.3 | 234.0 | 0 |
| 2 | lucifer-default | standard-mtp0 | 127.0 | 791.8 | 1171.7 | 1790.8 | 128.1 | 0 |
| 2 | lucifer-default | standard-mtp2 | 204.3 | 1065.6 | 1702.1 | 2573.5 | 220.4 | 0 |
| 2 | lucifer-default | standard-mtp3 | 195.2 | 1006.7 | 1597.6 | 2419.9 | 215.0 | 0 |
| 2 | lucifer-cutlass | standard-mtp0 | 127.7 | 864.1 | 1280.1 | 1971.2 | 128.7 | 0 |
| 2 | lucifer-cutlass | standard-mtp2 | 218.5 | 1199.0 | 1856.5 | 2841.5 | 228.1 | 0 |
| 2 | lucifer-cutlass | standard-mtp3 | 210.6 | 1113.3 | 1719.1 | 2658.7 | 223.0 | 0 |
| 4 | b12x-a16 | standard-mtp0 | 174.6 | 1212.8 | 1852.5 | 2700.2 | 173.4 | 0 |
| 4 | b12x-a16 | standard-mtp2 | 302.3 | 1705.9 | 2454.3 | 3571.8 | 318.9 | 0 |
| 4 | b12x-a16 | standard-mtp3 | 254.3 | 1493.2 | 2211.5 | 3104.2 | 290.2 | 0 |
| 4 | b12x-a8 | standard-mtp0 | 174.4 | 1070.2 | 1670.6 | 2528.5 | 173.7 | 0 |
| 4 | b12x-a8 | standard-mtp2 | 290.0 | 1554.9 | 2354.5 | 3593.9 | 309.5 | 0 |
| 4 | b12x-a8 | standard-mtp3 | 249.8 | 1392.2 | 2155.0 | 3260.1 | 277.3 | 0 |
| 4 | b12x-a8-dglin | standard-mtp0 | 177.0 | 1080.3 | 1677.7 | 2556.7 | 176.6 | 0 |
| 4 | b12x-a8-dglin | standard-mtp2 | 297.0 | 1578.7 | 2359.4 | 3622.1 | 321.5 | 0 |
| 4 | b12x-a8-dglin | standard-mtp3 | 255.0 | 1403.4 | 2172.9 | 3243.3 | 280.0 | 0 |
| 4 | lucifer-default | standard-mtp0 | 158.6 | 1153.2 | 1717.9 | 2634.0 | 159.7 | 0 |
| 4 | lucifer-default | standard-mtp2 | 265.9 | 1575.9 | 2467.0 | 3789.7 | 276.7 | 0 |
| 4 | lucifer-default | standard-mtp3 | 266.2 | 1524.9 | 2380.4 | 3520.9 | 296.5 | 0 |
| 4 | lucifer-cutlass | standard-mtp0 | 152.7 | 1219.2 | 1915.0 | 2900.7 | 154.1 | 0 |
| 4 | lucifer-cutlass | standard-mtp2 | 279.6 | 1813.3 | 2805.1 | 4146.2 | 303.9 | 0 |
| 4 | lucifer-cutlass | standard-mtp3 | 261.8 | 1651.9 | 2578.0 | 3774.9 | 295.1 | 0 |

## Prefill Throughput

Client-side prompt tokens / TTFT, `standalone-prefill`, prefix cache enabled but
non-repeating prompts.

### DSpark Checkpoint

| TP | Backend | Mode | 8k tok/s | 64k tok/s | 128k tok/s | Note |
|---:|---|---|---:|---:|---:|---|
| 2 | b12x-a16 | dspark | 11141 | 11105 | 10437 |  |
| 2 | b12x-a8 | dspark | 12902 | 12765 | 11982 |  |
| 2 | b12x-a8-dglin | dspark | 12717 | 12664 | 11844 | DeepGEMM linear |
| 2 | lucifer-default | dspark | 12637 | 12585 | 11609 |  |
| 2 | lucifer-cutlass | dspark | 12328 | 12186 | 11216 |  |
| 4 | b12x-a16 | dspark | 13549 | 13335 | 12571 |  |
| 4 | b12x-a8 | dspark | 14780 | 14581 | 13550 |  |
| 4 | b12x-a8-dglin | dspark | 14936 | 14707 | 13613 | DeepGEMM linear |
| 4 | lucifer-default | dspark | 14841 | 14561 | 13334 |  |
| 4 | lucifer-cutlass | dspark | 14729 | 14304 | 12271 |  |

### Standard Checkpoint

| TP | Backend | Mode | 8k tok/s | 64k tok/s | 128k tok/s | Note |
|---:|---|---|---:|---:|---:|---|
| 2 | b12x-a16 | standard-mtp0 | 11904 | 11499 | 10709 |  |
| 2 | b12x-a16 | standard-mtp2 | 11539 | 11208 | 10416 |  |
| 2 | b12x-a16 | standard-mtp3 | 11643 | 11294 | 10523 |  |
| 2 | b12x-a8 | standard-mtp0 | 13623 | 13135 | 12140 |  |
| 2 | b12x-a8 | standard-mtp2 | 13032 | 12574 | 11623 |  |
| 2 | b12x-a8 | standard-mtp3 | 13109 | 12657 | 11686 |  |
| 2 | b12x-a8-dglin | standard-mtp0 | 13624 | 13103 | 12101 | DeepGEMM linear |
| 2 | b12x-a8-dglin | standard-mtp2 | 13339 | 12807 | 11785 | DeepGEMM linear |
| 2 | b12x-a8-dglin | standard-mtp3 | 13330 | 12865 | 11864 | DeepGEMM linear |
| 2 | lucifer-default | standard-mtp0 | 13354 | 12757 | 11723 |  |
| 2 | lucifer-default | standard-mtp2 | 12674 | 12237 | 11222 |  |
| 2 | lucifer-default | standard-mtp3 | 12800 | 12094 | 11262 |  |
| 2 | lucifer-cutlass | standard-mtp0 | 13081 | 12435 | 11422 |  |
| 2 | lucifer-cutlass | standard-mtp2 | 12915 | 12302 | 11270 |  |
| 2 | lucifer-cutlass | standard-mtp3 | 12783 | 12238 | 11250 |  |
| 4 | b12x-a16 | standard-mtp0 | 14360 | 13894 | 12868 |  |
| 4 | b12x-a16 | standard-mtp2 | 13938 | 13436 | 12434 |  |
| 4 | b12x-a16 | standard-mtp3 | 13702 | 13226 | 12279 |  |
| 4 | b12x-a8 | standard-mtp0 | 15733 | 15080 | 13866 |  |
| 4 | b12x-a8 | standard-mtp2 | 15229 | 14665 | 13432 |  |
| 4 | b12x-a8 | standard-mtp3 | 14910 | 14413 | 13242 |  |
| 4 | b12x-a8-dglin | standard-mtp0 | 15723 | 15130 | 13874 | DeepGEMM linear |
| 4 | b12x-a8-dglin | standard-mtp2 | 15287 | 14708 | 13478 | DeepGEMM linear |
| 4 | b12x-a8-dglin | standard-mtp3 | 15016 | 14439 | 13273 | DeepGEMM linear |
| 4 | lucifer-default | standard-mtp0 | 15562 | 14821 | 13547 |  |
| 4 | lucifer-default | standard-mtp2 | 15014 | 14426 | 13138 |  |
| 4 | lucifer-default | standard-mtp3 | 14897 | 14267 | 13029 |  |
| 4 | lucifer-cutlass | standard-mtp0 | 15271 | 14578 | 13332 |  |
| 4 | lucifer-cutlass | standard-mtp2 | 14874 | 14192 | 12961 |  |
| 4 | lucifer-cutlass | standard-mtp3 | 14562 | 13991 | 12822 |  |

## Quick Read

- The synchronized sweep fixes the invalid B12X A8 no-MTP result. The valid
  `tp2-b12x-a8-standard-mtp0` row is `140.7 tok/s` cc1 and `140.2 tok/s` coding
  median; the hybrid row is `140.8` and `140.5`.
- The v8 B12X DSpark comparison row is `A16`. In this image, TP4 B12X A16
  DSpark reaches `292.4 tok/s` cc1, `2616.7 tok/s` cc64, and `389.4 tok/s`
  coding median.
- DSpark full B12X A8 keeps the prefill advantage: TP4 reaches `14780`,
  `14581`, and `13550 tok/s` at 8k/64k/128k.
- On the standard checkpoint, MTP2 is still the best sustained decode setting in
  the important cc64 rows. The strongest standard row is
  `tp4-lucifer-cutlass-standard-mtp2` at `4146.2 tok/s` cc64 and `303.9 tok/s`
  coding median.
- B12X A8 + DeepGEMM linear is competitive on the standard checkpoint. At TP4
  MTP2 it reaches `3622.1 tok/s` cc64 and `321.5 tok/s` coding median.
- The historical hybrid is not a preferred DSpark path. TP4 DSpark cc64 remains
  low at `1058.7 tok/s`, even though its prefill is strong.
- DSpark Lucifer CUTLASS remains the strongest DSpark sustained decode row:
  TP2 cc64 `2500.0 tok/s`, TP4 cc64 `3326.0 tok/s`, and TP4 coding peak
  `393.3 tok/s`.
- Current numbers are from the synchronized wave-ready run only. Ignore the
  aborted patch-missing run and the unsynchronized `pc1441b5-20260704-101250`
  artifact.

## Artifacts

```text
/root/bench-results/ds4-v9-refresh-pc1441b5-syncwave-20260704-102844/
/root/bench-results/ds4-v9-refresh-pc1441b5-syncwave-20260704-102844/repro/
/root/vllm/prubezne_vysledky
/root/vllm/blackwell-llm-docker/patches/vllm-b12x-indexer-warmup-fallback-20260704.patch
/root/rtx6kpro/scripts/run-ds4-v9-server.sh
/root/rtx6kpro/scripts/run-ds4-v9-sweep.sh
/root/rtx6kpro/scripts/render-ds4-v9-results.py
```

Source worktrees used around the image refresh:

```text
/root/vllm/worktrees/b12x-master-latest
```

The vLLM source identity is recorded in the Docker labels as
`dev/eldritch-enlightenment` @ `45c1582e9b80ba83e71c3a6458e71da4736fbdc4` plus
the local patch above.

## Caveats

- Standard rows use the base `DeepSeek-V4-Flash` checkpoint. DSpark rows use
  the DSpark checkpoint, not the standard checkpoint with an extra flag.
- `standard-mtp0` disables speculative decoding completely. `standard-mtp2` and
  `standard-mtp3` use the base checkpoint MTP heads with `2` and `3` draft
  tokens.
- DSpark's native tested draft count is `5`; do not treat it as equivalent to
  standard MTP2 or MTP3.
- The helper scripts assume the model snapshot already exists under
  `/root/.cache/huggingface/hub`. Override `STANDARD_MODEL` or `DSPARK_MODEL`
  if your path differs.
- Valid v9 comparisons should use the synchronized sweep artifacts above. The
  unsynchronized aborted run is useful only as the reproduction of the bad
  `132-133 tok/s` no-MTP measurement.
