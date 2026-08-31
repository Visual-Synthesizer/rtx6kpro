# DeepSeek-V4-Flash and DSpark v8

This page documents the DS4 v8 validation on the shared Eldritch Enlightenment
image. Unlike v7, the benchmark matrix covers both checkpoints:

```text
deepseek-ai/DeepSeek-V4-Flash
deepseek-ai/DeepSeek-V4-Flash-DSpark
```

The standard checkpoint is tested with MTP off and MTP2. The DSpark checkpoint
is tested with `method=dspark`; DSpark is not regular MTP and uses its native
block size of `5` draft tokens.

## What Changed From v7

- The Docker image is now the unified Eldritch build with the DeepSeek V4 MTP
  RoPE FP32 fix: DS4 MTP2 no longer fails with `cos_sin_cache must be float32`.
- The v8 page includes a full TP2/TP4 sweep for B12X, Lucifer default, and
  Lucifer CUTLASS across standard no-MTP, standard MTP2, and DSpark.
- Lucifer default and Lucifer CUTLASS now enable B12X PCIe one-shot all-reduce
  by default for decode-sized tensors, with PyNCCL fallback for larger prefill
  all-reduces.
- Helper launch scripts are checked into this repo so the compose/runtime layer
  does not need to duplicate the full vLLM command by hand.
- TileLang, TVM, TorchInductor, Torch extensions, FlashInfer, and vLLM caches
  are all mounted under `/cache` so repeated starts do not lose kernel caches.

## Docker Image

```text
voipmonitor/vllm:eldritch-enlightenment-v2226f26-b12x15cd38c-cu132-20260629
voipmonitor/vllm@sha256:72c2dd96310b6e9cea5c6e33982586d64a4ca7a9b66921867879309ee1aa58f6
```

Runtime version:

```text
0.11.2.dev279+eldritch.enlightenment.2226f26.b12x15cd38c.fi25dd814.cu132.20260629
```

Component pins from image labels:

| Component | Commit / branch |
|---|---|
| vLLM | `2226f261e9a6befef7a344997fb3b6769baa3bf7` |
| B12X | `15cd38ce3f10ee5cb7db1179cbc7c88fd15e37b7` |
| FlashInfer | `25dd814e03791e370f96c3148242f0dc8de504ac` |
| DeepGEMM | `2073ddb2814892014c33ef4cd1c7d4c148baf1fe` (`nv_dev`) |
| CUDA / PyTorch | CUDA `13.2.1`, PyTorch `2.12.0+cu132` |

Important vLLM PRs included in this build:

| PR | Purpose |
|---|---|
| local-inference-lab/vllm#64 | GLM TP6/head66 and B12X sparse-MLA compatibility stack used by the Eldritch line. |
| local-inference-lab/vllm#66 | DSpark TP4 native TileLang sparse-attention path. |
| local-inference-lab/vllm#67 | Keep DeepSeek V4 RoPE cache in FP32 so DS4 MTP2 starts cleanly. |

See [`eldritch-enlightenment-docker.md`](./eldritch-enlightenment-docker.md) for
the broader image build notes.

## Models

Standard DS4 checkpoint:

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

| Mode | Checkpoint | Speculative config | Graph cap used in v8 sweep |
|---|---|---|---:|
| `standard-mtp0` | `DeepSeek-V4-Flash` | none | 256 |
| `standard-mtp2` | `DeepSeek-V4-Flash` | `{"method":"mtp","num_speculative_tokens":2,"draft_sample_method":"probabilistic"}` | 512 |
| `dspark` | `DeepSeek-V4-Flash-DSpark` | `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}` | 512 |

| Backend | Attention | MoE / linear |
|---|---|---|
| `b12x` | `B12X_MLA_SPARSE` | `--moe-backend=b12x --linear-backend=b12x`, B12X PCIe all-reduce on, `B12X_MOE_FORCE_A16=1` |
| `lucifer-default` | `FLASHINFER_MLA_SPARSE_DSV4` | default DS4 MoE path, B12X PCIe one-shot all-reduce for small decode tensors |
| `lucifer-cutlass` | `FLASHINFER_MLA_SPARSE_DSV4` | `--kernel-config.moe_backend=flashinfer_cutlass`, B12X PCIe one-shot all-reduce for small decode tensors |

For Lucifer modes, the helper sets:

```text
VLLM_ENABLE_PCIE_ALLREDUCE=1
VLLM_PCIE_ALLREDUCE_BACKEND=b12x
VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
```

This selects `['B12X_PCIE_ONESHOT', 'PYNCCL']` for the TP group. Large prefill
all-reduces are intentionally rejected by the 64 KiB one-shot limit and fall
back to PyNCCL.

## Launch Helper

The recommended v8 launch path is the host-side helper script:

```bash
cd /root/rtx6kpro

IMAGE=voipmonitor/vllm:eldritch-enlightenment-v2226f26-b12x15cd38c-cu132-20260629 \
NAME=ds4-v8 \
PORT=8000 \
GPUS=0,1 \
TP=2 \
BACKEND=b12x \
MODE=standard-mtp0 \
MAX_NUM_SEQS=64 \
scripts/run-ds4-v8-server.sh
```

Common overrides:

```bash
# Standard DS4 + MTP2 on TP4, Lucifer CUTLASS.
TP=4 GPUS=0,1,2,3 BACKEND=lucifer-cutlass MODE=standard-mtp2 scripts/run-ds4-v8-server.sh

# DSpark checkpoint on TP2, Lucifer default.
TP=2 GPUS=0,1 BACKEND=lucifer-default MODE=dspark scripts/run-ds4-v8-server.sh

# B12X DSpark on TP4.
TP=4 GPUS=0,1,2,3 BACKEND=b12x MODE=dspark scripts/run-ds4-v8-server.sh
```

The helper always unsets `NCCL_GRAPH_FILE`, `NCCL_GRAPH_DUMP_FILE`, and
`VLLM_B12X_MLA_EXTEND_MAX_CHUNKS` before `vllm serve`. Do not run with
`NCCL_GRAPH_FILE=` set to an empty string.

## Docker Compose Wrapper

This compose file keeps only the knobs users usually change. It delegates the
full command construction to `scripts/run-ds4-v8-server.sh`. The default below
starts the preferred DSpark path: Lucifer CUTLASS MoE with
`FLASHINFER_MLA_SPARSE_DSV4` attention and B12X PCIe one-shot all-reduce for
small decode all-reduces.

```yaml
services:
  ds4:
    image: docker:27-cli
    container_name: ${WRAPPER_NAME:-ds4-v8-launcher}
    network_mode: host
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - /root/rtx6kpro:/workspace:ro
      - /root/.cache:/root/.cache
    working_dir: /workspace
    environment:
      IMAGE: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v2226f26-b12x15cd38c-cu132-20260629}
      NAME: ${NAME:-ds4-v8}
      PORT: ${PORT:-8000}
      GPUS: ${CUDA_VISIBLE_DEVICES:-0,1}
      TP: ${TP:-2}
      BACKEND: ${BACKEND:-lucifer-cutlass}
      MODE: ${MODE:-dspark}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      CACHE: ${CACHE:-/root/.cache/vllm-ds4-v8/ds4-v8}
    command: ["scripts/run-ds4-v8-server.sh"]
```

Examples:

```bash
# Preferred DSpark path: Lucifer CUTLASS MoE.
BACKEND=lucifer-cutlass MODE=dspark TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose up

# Lucifer default MoE DSpark path.
BACKEND=lucifer-default MODE=dspark TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose up

# B12X DSpark path.
BACKEND=b12x MODE=dspark TP=4 CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose up
```

## Full Sweep Command

The v8 benchmark was run with all 16 GPUs in TP-sized waves:

```bash
cd /root/rtx6kpro

OUT=/root/bench-results/ds4-v8-v2226f26-20260630 \
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v2226f26-b12x15cd38c-cu132-20260629 \
TPS=2,4 \
MAX_NUM_SEQS=64 \
DECODE_CONCURRENCY=1,16,32,64 \
DECODE_CONTEXTS=0 \
DECODE_DURATION=30 \
PREFILL_CONTEXTS=8k,64k,128k \
PREFILL_DURATION=10 \
PORT_BASE=6100 \
scripts/run-ds4-v8-sweep.sh
```

For the two B12X MTP2 8k prefill cells, the first full-sweep run included a
cold artefact. Those two cells were rerun warm with the same server cache and
are marked below.

The Lucifer rows below were retested after enabling B12X PCIe one-shot
all-reduce by default:

```text
/root/bench-results/ds4-v8-v2226f26-b12xar-lucifer-20260630/
```

## Decode Throughput

Sustained decode is aggregate tok/s from `llm_decode_bench.py`, `ctx=0`, 30
seconds per cell. `coding peak` is the median generation-only tok/s over five
Sieve-of-Eratosthenes cc1 runs; every row had `0` CJK runs.

| TP | Backend | Mode | cc1 tok/s | cc16 tok/s | cc32 tok/s | cc64 tok/s | coding peak median | CJK runs |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 2 | b12x | standard-mtp0 | 130.6 | 732.4 | 1020.1 | 1371.4 | 131.5 | 0 |
| 2 | b12x | standard-mtp2 | 218.1 | 886.6 | 1168.7 | 1483.3 | 229.4 | 0 |
| 2 | b12x | dspark | 204.9 | 708.9 | 890.1 | 963.7 | 279.6 | 0 |
| 2 | lucifer-default | standard-mtp0 | 118.8 | 771.3 | 1158.8 | 1767.4 | 119.7 | 0 |
| 2 | lucifer-default | standard-mtp2 | 195.0 | 1051.4 | 1692.8 | 2580.8 | 211.0 | 0 |
| 2 | lucifer-default | dspark | 191.5 | 1027.7 | 1550.4 | 2263.9 | 268.3 | 0 |
| 2 | lucifer-cutlass | standard-mtp0 | 128.0 | 864.4 | 1299.4 | 1977.8 | 129.2 | 0 |
| 2 | lucifer-cutlass | standard-mtp2 | 216.1 | 1215.2 | 1857.4 | 2837.2 | 228.0 | 0 |
| 2 | lucifer-cutlass | dspark | 222.6 | 1129.2 | 1712.9 | 2458.6 | 307.1 | 0 |
| 4 | b12x | standard-mtp0 | 157.7 | 1073.8 | 1563.2 | 2129.4 | 159.4 | 0 |
| 4 | b12x | standard-mtp2 | 279.0 | 1399.2 | 1826.9 | 2332.2 | 305.9 | 0 |
| 4 | b12x | dspark | 267.8 | 1091.8 | 1354.2 | 1357.0 | 363.7 | 0 |
| 4 | lucifer-default | standard-mtp0 | 146.8 | 1110.5 | 1684.9 | 2573.1 | 147.7 | 0 |
| 4 | lucifer-default | standard-mtp2 | 267.8 | 1574.4 | 2468.5 | 3802.1 | 276.7 | 0 |
| 4 | lucifer-default | dspark | 254.6 | 1474.8 | 2179.3 | 3044.3 | 328.7 | 0 |
| 4 | lucifer-cutlass | standard-mtp0 | 152.9 | 1217.1 | 1899.4 | 2905.9 | 154.2 | 0 |
| 4 | lucifer-cutlass | standard-mtp2 | 280.0 | 1805.8 | 2800.2 | 4140.0 | 298.6 | 0 |
| 4 | lucifer-cutlass | dspark | 287.9 | 1693.9 | 2461.3 | 3342.0 | 370.2 | 0 |

## Prefill Throughput

Client-side prompt tokens / TTFT, `standalone-prefill`, prefix cache enabled but
non-repeating prompts.

| TP | Backend | Mode | 8k tok/s | 64k tok/s | 128k tok/s | Note |
|---:|---|---|---:|---:|---:|---|
| 2 | b12x | standard-mtp0 | 7763 | 5526 | 4121 |  |
| 2 | b12x | standard-mtp2 | 7629 | 5483 | 4096 | warm rerun |
| 2 | b12x | dspark | 7745 | 5609 | 4191 |  |
| 2 | lucifer-default | standard-mtp0 | 13363 | 12760 | 11688 | B12X one-shot AR retest |
| 2 | lucifer-default | standard-mtp2 | 13136 | 12583 | 11534 | B12X one-shot AR retest |
| 2 | lucifer-default | dspark | 12573 | 12563 | 11551 | B12X one-shot AR retest |
| 2 | lucifer-cutlass | standard-mtp0 | 13021 | 12483 | 11468 | B12X one-shot AR retest |
| 2 | lucifer-cutlass | standard-mtp2 | 12723 | 12148 | 11143 | B12X one-shot AR retest |
| 2 | lucifer-cutlass | dspark | 12295 | 12092 | 11141 | B12X one-shot AR retest |
| 4 | b12x | standard-mtp0 | 9564 | 6393 | 4604 |  |
| 4 | b12x | standard-mtp2 | 9446 | 6356 | 4592 | warm rerun |
| 4 | b12x | dspark | 9295 | 6264 | 4497 |  |
| 4 | lucifer-default | standard-mtp0 | 15483 | 14795 | 13509 | B12X one-shot AR retest |
| 4 | lucifer-default | standard-mtp2 | 15138 | 14515 | 13259 | B12X one-shot AR retest |
| 4 | lucifer-default | dspark | 14572 | 14412 | 13228 | B12X one-shot AR retest |
| 4 | lucifer-cutlass | standard-mtp0 | 15179 | 14587 | 13324 | B12X one-shot AR retest |
| 4 | lucifer-cutlass | standard-mtp2 | 14909 | 14243 | 13021 | B12X one-shot AR retest |
| 4 | lucifer-cutlass | dspark | 14860 | 14477 | 13258 | B12X one-shot AR retest |

## Quick Read

- For raw cc1 decode without speculation, B12X is fastest in this matrix:
  TP2 `130.6 tok/s`, TP4 `157.7 tok/s`. Lucifer CUTLASS with B12X one-shot
  all-reduce is now close: TP2 `128.0 tok/s`, TP4 `152.9 tok/s`.
- For high-concurrency standard MTP2, Lucifer CUTLASS is strongest:
  TP2 cc64 `2837.2 tok/s`, TP4 cc64 `4140.0 tok/s`.
- DSpark improves cc1 heavily, especially TP4: B12X `363.7 tok/s` coding peak
  and Lucifer CUTLASS `370.2 tok/s` coding peak.
- B12X prefill is still materially slower than the Lucifer SM120 path. TP4
  standard no-MTP 128k prefill is `4604 tok/s` on B12X versus `13509 tok/s` on
  Lucifer default and `13324 tok/s` on Lucifer CUTLASS.
- Lucifer default has the best prefill in this run; CUTLASS generally wins more
  at high-concurrency decode with MTP2.

## Artifacts

```text
/root/bench-results/ds4-v8-v2226f26-20260630/
/root/bench-results/ds4-v8-v2226f26-20260630-rerun-b12x-mtp2-8k/
/root/bench-results/ds4-v8-v2226f26-b12xar-lucifer-20260630/
/root/rtx6kpro/scripts/run-ds4-v8-server.sh
/root/rtx6kpro/scripts/run-ds4-v8-sweep.sh
```

## Caveats

- DSpark rows use the DSpark checkpoint, not the standard checkpoint with an
  extra flag.
- DSpark's native tested draft count is `5`; do not treat it as equivalent to
  standard MTP2.
- B12X MTP2 8k prefill values in the first full sweep were cold-start artefacts
  and were replaced with same-cache warm reruns. The raw files are retained in
  the artifact directory.
- The helper scripts assume the model snapshots already exist under
  `/root/.cache/huggingface/hub`. Override `STANDARD_MODEL` or `DSPARK_MODEL`
  if your paths differ.
