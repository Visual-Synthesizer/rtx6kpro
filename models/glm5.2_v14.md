# GLM-5.2 v14 NVFP4 / Online MXFP8 Overlay

This page documents the July 2026 GLM-5.2 serving recipe and benchmark sweep for
RTX 6000 Pro Blackwell. The baseline checkpoint is Luke Alonso's NVFP4 model.
The online variant starts from the same checkpoint and converts eligible BF16
linear weights to MXFP8 during model load.

The benchmark run for this revision was designed to avoid the earlier low
decode outlier: for paired 8-GPU tests the script starts both servers, waits for
both `/v1/models` endpoints to respond, waits another 30 seconds, and only then
starts benchmark clients. No client is started while the second model is still
loading.

## Image And Model

Final reproducible image:

```text
voipmonitor/vllm:dev-eldritch-enlightenment-vllmc382f1d-fp8d005934-b12xe44cb77-it85e7c5f-cu132-20260706
```

Docker Hub manifest digest:

```text
sha256:2dfb16d1e890dbe637e13fc259dc596704974e2868f0d0679f734ad51eaa2934
```

Build script:

```text
scripts/build-glm52-v14-final-image.sh
```

Pinned source stack:

| Component | Ref |
|---|---|
| vLLM base | `local-inference-lab/vllm dev/eldritch-enlightenment @ c382f1d28d5be2f867c216609408bdb424d6049a` |
| fp8.py bridge patch | `d00593416aeb3925553ccd589d91df7075d618f6` |
| B12X | `local-inference-lab/b12x master @ e44cb77777a075790ebe9f7aa9f225d073aea109` |
| InstantTensor | `scitix/InstantTensor @ 85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| FlashInfer | `5a73a36a7169ec5533ba474bb9204bed765dd297` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |

FlashInfer cubin wheels are intentionally not built by this image script
(`FLASHINFER_BUILD_CUBIN=0`); the GLM 5.2 path here uses the B12X/DeepGEMM
kernels, while `--enable-flashinfer-autotune` remains enabled for compatibility.

Exact build and push:

```bash
cd /root/rtx6kpro
PUSH_IMAGE=1 ./scripts/build-glm52-v14-final-image.sh
```

The final image defaults InstantTensor to buffered I/O:

```bash
--load-format instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

InstantTensor's upstream default for regular disk files is direct I/O
(`URING,AIO`), which is good for cold one-time loads but bypasses the Linux page
cache. `BUFFERED` expands to `URING_BUFFERED,AIO_BUFFERED,MMAP`, so repeated
loads can reuse cached pages when the model is already hot in memory. See
[`scitix/InstantTensor`](https://github.com/scitix/InstantTensor) and the
[vLLM InstantTensor loader docs](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/).

Previous benchmark images, kept here only for historical comparison:

Clean Luke NVFP4 image:

```text
voipmonitor/vllm:eldritch-enlightenment-v56a5c3e-b12x7bfc945-glm52-dcp-fp8nvfp4fix-cu132-20260705
```

Online MXFP8 overlay image:

```text
voipmonitor/vllm:eldritch-enlightenment-v56a5c3e-b12x7bfc945-pr74-mxfp8overlay-cu132-20260705
```

Model:

```text
lukealonso/GLM-5.2-NVFP4
/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522
```

Result roots:

```text
/root/bench-results/glm52-v14-todo-and-sweep-20260706T0150Z
/root/kld/glm52_v14_todo_20260706T0150Z
/root/bench-results/glm52-v14-codingpeak-dcp1-mtp3-20260706T130151Z
```

## Runtime Contract

| Setting | Direct DCP1 rows | Full sweep |
|---|---:|---:|
| TP | `8` | `8` |
| GPUs per instance | `8` | `8` |
| Paired instances | yes, `0-7` and `8-15` | yes, `0-7` and `8-15` |
| DCP | `1` | `1,2,4,8` |
| MTP | `0` | `0,3` |
| Max num seqs | `64` | `32` |
| CUDA graph capture | `256` | `128` |
| Decode concurrency | table profile | `1,2,4,8,16,32` |
| Decode context | table profile | `0,16k,32k,64k,128k` |
| Standalone prefill | `30k,64k,120k` | `8k,64k` |
| KV cache | `fp8` | `fp8` |
| Attention backend | `B12X_MLA_SPARSE` | `B12X_MLA_SPARSE` |
| MoE backend | `b12x` | `b12x` |

The index-cache pattern is intentionally pinned and validated by the script. It
must be exactly 78 characters:

```text
FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS
```

The runner exits before launch if this is shortened.

## Minimal Docker Compose

The final image contains the GLM 5.2 launcher:

```text
/usr/local/bin/run-glm52-v14-server
```

So users do not need to download any helper script. A standalone compose file is
enough:

```yaml
name: glm52-v14

services:
  server:
    image: ${IMAGE:-voipmonitor/vllm:dev-eldritch-enlightenment-vllmc382f1d-fp8d005934-b12xe44cb77-it85e7c5f-cu132-20260706}
    container_name: ${NAME:-glm52-v14}
    network_mode: host
    ipc: host
    privileged: true
    init: true
    shm_size: 32g
    gpus: all
    ulimits:
      memlock: -1
      stack: 67108864
      nofile:
        soft: 1048576
        hard: 1048576
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: compute,utility
      GPUS: ${GPUS:-0,1,2,3,4,5,6,7}
      MODEL: ${MODEL:-/root/.cache/huggingface/hub/models--lukealonso--GLM-5.2-NVFP4/snapshots/8a1f4a13204acf2b7ac840375efaed64c231c522}
      PORT: ${PORT:-8000}
      TP: ${TP:-8}
      DCP: ${DCP:-1}
      DCP_BACKEND: ${DCP_BACKEND:-a2a}
      MTP: ${MTP:-0}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      GRAPH: ${GRAPH:-}
      MOE_MODE: ${MOE_MODE:-a4}
      ONLINE_MXFP8: ${ONLINE_MXFP8:-0}
      F8_DMA: ${F8_DMA:-0}
      LOAD_FORMAT: ${LOAD_FORMAT:-instanttensor}
      INSTANTTENSOR_BACKEND: ${INSTANTTENSOR_BACKEND:-BUFFERED}
    volumes:
      - /root/models:/root/models:ro
      - /root/.cache/huggingface:/root/.cache/huggingface:rw
      - ${CACHE_DIR:-/root/.cache/vllm-glm52-v14/glm52-v14}:/cache:rw
      - ${TMP_DIR:-/root/vllm/tmp/glm52-v14}:/container-tmp:rw
    command:
      - run-glm52-v14-server
```

Example overrides:

```bash
# Online MXFP8, A16 force, MTP3, graph auto = 4 * MAX_NUM_SEQS.
GPUS=8,9,10,11,12,13,14,15 \
PORT=8001 \
MOE_MODE=a16 \
ONLINE_MXFP8=1 \
MTP=3 \
MAX_NUM_SEQS=32 \
docker compose up -d
```

User-facing knobs:

| Env | Default | Meaning |
|---|---|---|
| `MODEL` | Luke NVFP4 snapshot | HF checkpoint path or repo. |
| `IMAGE` | final v14 image | Docker image to run. |
| `GPUS` | `0,1,2,3,4,5,6,7` | GPU set for one 8-GPU instance. |
| `PORT` | `8000` | OpenAI API port. |
| `DCP` | `1` | Decode context parallel size. |
| `MTP` | `0` | Native MTP speculative token count; `0` disables MTP. |
| `MAX_NUM_SEQS` | `64` | vLLM concurrency cap. |
| `GRAPH` | `4 * MAX_NUM_SEQS` | CUDA graph capture cap; normally leave unset. |
| `MOE_MODE` | `a4` | `a4` = checkpoint-native NVFP4/A4, `a16` = force W4A16, `force-a8-experimental` = explicit A8 force for non-NVFP4 experiments. |
| `ONLINE_MXFP8` | `0` | `1` adds `--quantization-config '{"linear":{"weight":"mxfp8"}}'`. |
| `F8_DMA` | `0` | FP8 PCIe DMA allreduce mode: `0`, `ag`, or `ring`. |
| `LOAD_FORMAT` | `instanttensor` | Set `fastsafetensors` to disable InstantTensor for comparison. |
| `INSTANTTENSOR_BACKEND` | `BUFFERED` | Buffered mode uses Linux page cache on hot reloads. |

`GRAPH` can be omitted; the launcher computes it as `4 * MAX_NUM_SEQS`. The
launcher validates the 78-character index-cache pattern before starting vLLM.

The repository also includes the same compose file and an optional host helper:

```text
compose/glm52-v14.yml
scripts/run-glm52-v14-compose.sh
```

## A4 And A16

For the Luke NVFP4 checkpoint, `A4` means the checkpoint-native NVFP4 MoE path:

```bash
export B12X_MOE_FORCE_A8=0
export B12X_MOE_FORCE_A16=0
```

`A16` forces B12X MoE onto the W4A16 path: BF16 activations with the NVFP4/W4
expert weights.

```bash
export B12X_MOE_FORCE_A8=0
export B12X_MOE_FORCE_A16=1
export B12X_W4A16_TC_DECODE=1
```

`A8` force is not the comparison axis for this NVFP4 checkpoint.

## Online MXFP8 Conversion

The online variant uses the overlay image and passes this vLLM quantization
config:

```bash
--quantization modelopt_fp4 \
--quantization-config '{"linear":{"weight":"mxfp8"}}'
```

In this PR overlay, `linear.weight=mxfp8` means: when a module is treated as a
linear layer and has a BF16 weight that is eligible for online requant, load that
weight as MXFP8 instead of leaving it BF16. Layers that the overlay keeps BF16,
including the sparse indexer projection path, are not forced to MXFP8 by the
generic `linear` rule.

The baseline variant does not pass `--quantization-config`; it loads the Luke
checkpoint as `--quantization modelopt_fp4`.

## f8 DMA Mode

`f8` selects the FP8 PCIe DMA allreduce mode:

```bash
export VLLM_PCIE_DMA_FP8=0      # or ag/ring
export B12X_PCIE_DMA_FP8=0      # same value
```

| f8 | Meaning |
|---|---|
| `0` | Disable FP8-compressed PCIe DMA payloads. |
| `ag` | Enable all-gather style FP8 DMA mode. |
| `ring` | Enable ring FP8 DMA mode. |

This mode affects large PCIe allreduce payloads, so it is a prefill knob. It is
not expected to move decode throughput in a meaningful way; decode deltas with
`ag`/`ring` are treated as measurement noise and are not shown in the main
decode tables. `ag` and `ring` remain listed for prefill/KLD where the transport
actually matters.

## Direct DCP1 Measurements

These rows replace the older placeholders. KLD is 5 runs against:

```text
/root/kld/glm52_refs/bf16-b12xmlasparse-w1-ctx2048-s512-20260618
context_length=2048
stride=512
max_windows=1
```

Primary DCP1 rows (`f8=0`):

| Variant | Mode | Decode agg tok/s | Coding peak tok/s | Prefill 30k | Prefill 64k | Prefill 120k | KLD mean +/- sd |
|---|---|---:|---:|---:|---:|---:|---|
| base | A4 | 88.53 | 88.81 | 6,491 | 6,238 | 5,883 | 0.10680 +/- 0.00323 |
| base | A16 | 85.73 | 86.21 | 5,975 | 5,752 | 5,448 | 0.06842 +/- 0.00197 |
| online | A4 | 95.51 | 96.12 | 6,598 | 6,307 | 5,967 | 0.10761 +/- 0.00430 |

Online A4 f8 DMA impact, keeping decode out of the comparison:

| f8 | Prefill 30k | Prefill 64k | Prefill 120k | KLD mean +/- sd |
|---|---:|---:|---:|---|
| `0` | 6,598 | 6,307 | 5,967 | 0.10761 +/- 0.00430 |
| `ag` | 7,165 | 6,870 | 6,444 | 0.11171 +/- 0.00542 |
| `ring` | 8,092 | 7,676 | 7,142 | 0.12100 +/- 0.00886 |

## Coding Peak Rerun

Result root:

```text
/root/bench-results/glm52-v14-codingpeak-dcp1-mtp3-20260706T130151Z
```

DCP1, MTP3, `f8=0`. `cc1` and `cc32` are decode aggregate tok/s from the
same launch family; coding peak is the mean of the three coding task positions.

| Variant | Mode | cc1 decode tok/s | cc32 decode tok/s | Coding peak mean | Median | Min | Max |
|---|---|---:|---:|---:|---:|---:|---:|
| base | A4 | 136.50 | 1,409.11 | 180.52 | 178.90 | 176.75 | 185.86 |
| base | A16 | 126.60 | 1,319.22 | 166.55 | 169.51 | 160.15 | 171.27 |
| online | A4 | 144.46 | 1,479.78 | 177.18 | 178.76 | 166.59 | 184.13 |
| online | A16 | 130.07 | 1,386.32 | 166.26 | 166.44 | 158.90 | 172.76 |

## Full Sweep Decode

Values are `llm_decode_bench` ctx0 aggregate tok/s for concurrency
`1,2,4,8,16,32`. The published decode sweep is intentionally consolidated to
`f8=0`; the raw `ag/ring` decode runs are omitted because FP8 DMA is not a
decode path knob.

### base A4 f8=0

| MTP | DCP | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 87.99 | 146.28 | 252.23 | 372.84 | 615.69 | 934.07 |
| 0 | 2 | 68.72 | 110.48 | 183.96 | 308.64 | 517.44 | 770.13 |
| 0 | 4 | 67.48 | 107.97 | 179.56 | 298.21 | 484.50 | 722.11 |
| 0 | 8 | 62.95 | 101.12 | 165.40 | 269.64 | 422.81 | 632.34 |
| 3 | 1 | 125.90 | 208.47 | 351.03 | 547.58 | 867.20 | 1,427 |
| 3 | 2 | 100.78 | 177.56 | 301.14 | 475.80 | 756.94 | 1,186 |
| 3 | 4 | 99.30 | 167.94 | 289.98 | 454.40 | 691.08 | 1,070 |
| 3 | 8 | 95.84 | 159.88 | 265.06 | 410.45 | 600.76 | 827.86 |

### base A16 f8=0

| MTP | DCP | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 86.56 | 142.02 | 235.72 | 324.45 | 616.97 | 932.72 |
| 0 | 2 | 67.65 | 104.27 | 169.32 | 273.86 | 517.73 | 782.30 |
| 0 | 4 | 66.31 | 102.63 | 165.97 | 264.11 | 482.12 | 722.91 |
| 0 | 8 | 61.77 | 96.06 | 154.14 | 241.11 | 422.22 | 635.66 |
| 3 | 1 | 119.62 | 182.71 | 350.45 | 553.79 | 843.71 | 1,345 |
| 3 | 2 | 90.69 | 154.19 | 304.37 | 481.78 | 735.46 | 1,134 |
| 3 | 4 | 89.44 | 150.41 | 296.73 | 453.77 | 685.09 | 1,030 |
| 3 | 8 | 90.48 | 152.68 | 263.93 | 413.46 | 584.04 | 793.75 |

### online A4 f8=0

| MTP | DCP | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 94.96 | 152.04 | 254.11 | 378.62 | 641.43 | 953.24 |
| 0 | 2 | 72.99 | 113.89 | 187.98 | 312.84 | 532.22 | 802.18 |
| 0 | 4 | 71.82 | 111.84 | 184.29 | 300.67 | 498.74 | 739.96 |
| 0 | 8 | 66.51 | 104.36 | 170.11 | 270.55 | 429.92 | 638.57 |
| 3 | 1 | 129.37 | 211.16 | 359.88 | 557.84 | 902.52 | 1,461 |
| 3 | 2 | 104.96 | 179.20 | 307.62 | 486.24 | 779.92 | 1,225 |
| 3 | 4 | 100.28 | 172.18 | 286.85 | 452.26 | 715.04 | 1,085 |
| 3 | 8 | 98.23 | 164.85 | 270.51 | 413.74 | 611.94 | 842.56 |

### online A16 f8=0

| MTP | DCP | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 93.30 | 148.15 | 237.03 | 326.64 | 632.38 | 954.52 |
| 0 | 2 | 71.67 | 107.49 | 173.35 | 274.93 | 526.95 | 789.85 |
| 0 | 4 | 70.44 | 106.09 | 169.77 | 267.38 | 496.04 | 735.91 |
| 0 | 8 | 65.36 | 99.20 | 156.32 | 242.38 | 430.73 | 643.02 |
| 3 | 1 | 120.69 | 184.78 | 358.23 | 559.65 | 873.98 | 1,378 |
| 3 | 2 | 92.47 | 159.22 | 310.52 | 481.54 | 752.39 | 1,163 |
| 3 | 4 | 95.56 | 154.51 | 293.70 | 460.27 | 692.15 | 1,051 |
| 3 | 8 | 92.51 | 146.57 | 266.80 | 420.01 | 597.00 | 803.96 |

## Full Sweep Standalone Prefill

Standalone prefill stores only contexts that fit under the 131,072-token model
length; the requested 128k row is skipped by `llm_decode_bench` for this model.
Values are tok/s for 8k and 64k contexts.

The comparison tables below show cells as `8k / 64k` tok/s. Percent deltas use
the 64k value because that is the more useful long-prefill comparison point.
The smaller per-variant tables are kept below for exact raw rows.

### f8=0 comparison

| MTP | DCP | base A4 | online A4 | online vs base | base A16 | online A16 | online vs base |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 6,557 / 6,257 | 6,681 / 6,351 | +1.5% | 6,140 / 5,849 | 6,239 / 5,941 | +1.6% |
| 0 | 2 | 4,597 / 4,675 | 4,599 / 4,724 | +1.0% | 4,369 / 4,439 | 4,360 / 4,477 | +0.9% |
| 0 | 4 | 3,402 / 3,457 | 3,403 / 3,492 | +1.0% | 3,279 / 3,335 | 3,280 / 3,355 | +0.6% |
| 0 | 8 | 2,175 / 2,195 | 2,173 / 2,209 | +0.6% | 2,121 / 2,140 | 2,121 / 2,156 | +0.7% |
| 3 | 1 | 6,441 / 6,136 | 6,546 / 6,222 | +1.4% | 6,016 / 5,740 | 6,109 / 5,833 | +1.6% |
| 3 | 2 | 4,487 / 4,570 | 4,492 / 4,618 | +1.1% | 4,262 / 4,335 | 4,261 / 4,392 | +1.3% |
| 3 | 4 | 3,328 / 3,392 | 3,325 / 3,422 | +0.9% | 3,211 / 3,267 | 3,209 / 3,294 | +0.8% |
| 3 | 8 | 2,133 / 2,156 | 2,132 / 2,166 | +0.5% | 2,079 / 2,100 | 2,081 / 2,114 | +0.7% |

### A4 MTP3 FP8 DMA comparison

This is the only full-sweep prefill slice where `ag` and `ring` were tested.
The percentage in the online columns is the 64k gain versus the same online
`f8=0` DCP row.

| DCP | base f8=0 | base ag | base ring | online f8=0 | online ag | online ring |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6,441 / 6,136 | 7,130 / 6,738 | 7,912 / 7,435 | 6,546 / 6,222 | 7,235 / 6,843 (+10.0%) | 8,035 / 7,564 (+21.6%) |
| 2 | 4,487 / 4,570 | 4,804 / 4,894 | 5,147 / 5,272 | 4,492 / 4,618 | 4,806 / 4,963 (+7.5%) | 5,144 / 5,328 (+15.4%) |
| 4 | 3,328 / 3,392 | 3,501 / 3,571 | 3,682 / 3,757 | 3,325 / 3,422 | 3,505 / 3,602 (+5.3%) | 3,689 / 3,791 (+10.8%) |
| 8 | 2,133 / 2,156 | 2,203 / 2,226 | 2,272 / 2,300 | 2,132 / 2,166 | 2,206 / 2,240 (+3.4%) | 2,275 / 2,314 (+6.8%) |

### base A4 f8=0

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 0 | 1 | 6,557 | 6,257 |
| 0 | 2 | 4,597 | 4,675 |
| 0 | 4 | 3,402 | 3,457 |
| 0 | 8 | 2,175 | 2,195 |
| 3 | 1 | 6,441 | 6,136 |
| 3 | 2 | 4,487 | 4,570 |
| 3 | 4 | 3,328 | 3,392 |
| 3 | 8 | 2,133 | 2,156 |

### base A4 f8=ag

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 3 | 1 | 7,130 | 6,738 |
| 3 | 2 | 4,804 | 4,894 |
| 3 | 4 | 3,501 | 3,571 |
| 3 | 8 | 2,203 | 2,226 |

### base A4 f8=ring

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 3 | 1 | 7,912 | 7,435 |
| 3 | 2 | 5,147 | 5,272 |
| 3 | 4 | 3,682 | 3,757 |
| 3 | 8 | 2,272 | 2,300 |

### base A16 f8=0

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 0 | 1 | 6,140 | 5,849 |
| 0 | 2 | 4,369 | 4,439 |
| 0 | 4 | 3,279 | 3,335 |
| 0 | 8 | 2,121 | 2,140 |
| 3 | 1 | 6,016 | 5,740 |
| 3 | 2 | 4,262 | 4,335 |
| 3 | 4 | 3,211 | 3,267 |
| 3 | 8 | 2,079 | 2,100 |

### online A4 f8=0

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 0 | 1 | 6,681 | 6,351 |
| 0 | 2 | 4,599 | 4,724 |
| 0 | 4 | 3,403 | 3,492 |
| 0 | 8 | 2,173 | 2,209 |
| 3 | 1 | 6,546 | 6,222 |
| 3 | 2 | 4,492 | 4,618 |
| 3 | 4 | 3,325 | 3,422 |
| 3 | 8 | 2,132 | 2,166 |

### online A4 f8=ag

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 3 | 1 | 7,235 | 6,843 |
| 3 | 2 | 4,806 | 4,963 |
| 3 | 4 | 3,505 | 3,602 |
| 3 | 8 | 2,206 | 2,240 |

### online A4 f8=ring

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 3 | 1 | 8,035 | 7,564 |
| 3 | 2 | 5,144 | 5,328 |
| 3 | 4 | 3,689 | 3,791 |
| 3 | 8 | 2,275 | 2,314 |

### online A16 f8=0

| MTP | DCP | 8k | 64k |
|---:|---:|---:|---:|
| 0 | 1 | 6,239 | 5,941 |
| 0 | 2 | 4,360 | 4,477 |
| 0 | 4 | 3,280 | 3,355 |
| 0 | 8 | 2,121 | 2,156 |
| 3 | 1 | 6,109 | 5,833 |
| 3 | 2 | 4,261 | 4,392 |
| 3 | 4 | 3,209 | 3,294 |
| 3 | 8 | 2,081 | 2,114 |

## Reproduction

Build the final serving image:

```bash
cd /root/rtx6kpro
PUSH_IMAGE=1 ./scripts/build-glm52-v14-final-image.sh
```

Start a server with the image-owned launcher:

```bash
docker compose -f compose/glm52-v14.yml up -d
```

With a repo checkout, the helper is only a convenience wrapper around the same
compose file:

```bash
cd /root/rtx6kpro
NAME=glm52-v14-online-a16 PORT=8000 GPUS=0,1,2,3,4,5,6,7 \
MOE_MODE=a16 ONLINE_MXFP8=1 MTP=3 MAX_NUM_SEQS=32 \
./scripts/run-glm52-v14-compose.sh up
```

The reproducible runner is checked in at:

```text
scripts/bench-glm52-v14-todo-and-sweep.sh
```

Run the complete sweep:

```bash
cd /root/rtx6kpro
RUN_ID=20260706T0150Z \
RESULT_ROOT=/root/bench-results/glm52-v14-todo-and-sweep-20260706T0150Z \
KLD_ROOT=/root/kld/glm52_v14_todo_20260706T0150Z \
./scripts/bench-glm52-v14-todo-and-sweep.sh all
```

The runner supports narrower reruns:

```bash
./scripts/bench-glm52-v14-todo-and-sweep.sh table-todos
./scripts/bench-glm52-v14-todo-and-sweep.sh kld-todos
./scripts/bench-glm52-v14-todo-and-sweep.sh full-sweep
./scripts/bench-glm52-v14-todo-and-sweep.sh summarize
```

Progress is appended to:

```text
/root/vllm/prubezne_vysledky
```

The summary rendered into this page came from:

```bash
RUN_ID=20260706T0150Z \
RESULT_ROOT=/root/bench-results/glm52-v14-todo-and-sweep-20260706T0150Z \
KLD_ROOT=/root/kld/glm52_v14_todo_20260706T0150Z \
/root/vllm/dspark/bench_glm52_v14_todo_and_sweep.sh summarize
```

Validation counts from the final run:

```text
decode_full.json: 48
prefill_full.json: 48
```
