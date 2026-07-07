# GLM-5.2 v14 NVFP4 / Online FP8-MXFP8 Overlay

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
voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707
```

Docker Hub manifest digest:

```text
sha256:9b7d005575670ed3d007d7783932c1140d86f2b8ef66a2d9104b7ac609353466
```

Build script:

```text
scripts/build-glm52-v14-final-image.sh
```

Pinned source stack:

| Component | Ref |
|---|---|
| vLLM | `local-inference-lab/vllm fable/dcp-b12x-contiguous-lse-20260707 @ e2e2eaf61d05834fb5f7f529b75ce75c4cafc289` |
| vLLM upstream base | `dev/eldritch-enlightenment @ c382f1d28d5be2f867c216609408bdb424d6049a` |
| vLLM stacked PRs | `#76 fp8.py bridge`, `#77 online dense FP8/MXFP8 overlay + shared-expert fix`, `#78 DCP A2A hybrid token cap`, `#79 DCP warmup unsupported-world-size guard`, `#80 TP6 MXFP4 W4A8 padding`, `#81 contiguous LSE for B12X DCP pool` |
| Docker build repo | `local-inference-lab/blackwell-llm-docker main @ d25e34953b76e201676c29590be1b4d1079f56b0` |
| B12X | `local-inference-lab/b12x master @ 26144c0eda970ce7e30bf7c64a2f094abe1fea4d` |
| InstantTensor | `scitix/InstantTensor @ 85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL runtime | unified `/opt/libnccl.so.2.30.4` |
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

Hybrid-DCP follow-up sweep helpers:

```bash
cd /root/rtx6kpro
./scripts/bench-glm52-v14-dcp-hybrid-v5.sh tp8-decode
./scripts/bench-glm52-v14-tp6-mxfp4-v7.sh run
./scripts/bench-glm52-v14-tp8-hybrid-table-v7.sh run
```

The helpers start all servers through `scripts/run-glm52-v14-compose.sh`, wait
until paired instances are fully loaded before benchmarking, write progress to
`${RESULT_ROOT}/progress.log` by default, measures TP6 MXFP4/online-MXFP8 for
`DCP=1,2,3,6`, measures TP8 NVFP4/A16/MTP3 decode for `DCP=2,4,8`, and builds
the TP8/MTP0 comparison table for NVFP4 and MXFP4 across `DCP=1,2,4,8`.
The benchmark client is run with `--no-hw-monitor`; the helpers record their own
thermal CSV snapshots before and after each measured phase.

The final image defaults InstantTensor to buffered I/O:

```bash
--load-format instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

For `DCP>1`, the helper defaults to the hybrid DCP policy from
[`local-inference-lab/vllm#78`](https://github.com/local-inference-lab/vllm/pull/78):

```bash
DCP_BACKEND=a2a
DCP_A2A_MAX_TOKENS=64
DCP_A2A_LARGE_BACKEND=ag_rs
```

This keeps the low-latency B12X A2A path for small decode steps and routes
larger prefill/extend batches through AG+RS. `DCP=1` still has backend `n/a`.

InstantTensor's upstream default for regular disk files is direct I/O
(`URING,AIO`), which is good for cold one-time loads but bypasses the Linux page
cache. `BUFFERED` expands to `URING_BUFFERED,AIO_BUFFERED,MMAP`, so repeated
loads can reuse cached pages when the model is already hot in memory. See
[`scitix/InstantTensor`](https://github.com/scitix/InstantTensor) and the
[vLLM InstantTensor loader docs](https://docs.vllm.ai/en/latest/models/extensions/instanttensor/).

The image also replaces PyTorch's bundled `libnccl.so.2` and the historical
`/opt/libnccl-local-inference.so.2.30.4` path with symlinks to the same
`/opt/libnccl.so.2.30.4` runtime. This keeps PyTorch distributed, vLLM PyNCCL,
and InstantTensor on one NCCL library when InstantTensor receives a PyTorch
process group.

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
/root/kld/glm52_v14_keypoints_20260707Tkeypoints-v5
/root/kld/glm52_v14_keypoints_20260707Tkeypoints-v7
/root/bench-results/glm52-v14-dcp-hybrid-v5-tp8-fixed-20260707T0330Z
/root/bench-results/glm52-v14-dcp-hybrid-v5-tp6-fixed-20260707T0345Z
/root/bench-results/glm52-v14-v7-tp6-mxfp4-a8-20260707T115913Z
/root/bench-results/glm52-v14-v7-tp8-mxfp4-a8-dcp1-mtp0-20260707T133344Z
/root/bench-results/glm52-v14-v7-tp8-hybrid-table-mtp0-20260707T135950Z
```

## Runtime Contract

| Setting | Direct DCP1 rows | Full sweep |
|---|---:|---:|
| TP | `8` | `8` |
| GPUs per instance | `8` | `8` |
| Paired instances | yes, `0-7` and `8-15` | yes, `0-7` and `8-15` |
| DCP | `1` | `1,2,4,8` |
| DCP backend | `n/a` for DCP1 | hybrid: `a2a` up to 64 tokens, `ag_rs` above |
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
    image: ${IMAGE:-voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707}
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
      SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-GLM-5.2-NVFP4}
      PORT: ${PORT:-8000}
      TP: ${TP:-8}
      DCP: ${DCP:-1}
      DCP_BACKEND: ${DCP_BACKEND:-a2a}
      DCP_A2A_MAX_TOKENS: ${DCP_A2A_MAX_TOKENS:-64}
      DCP_A2A_LARGE_BACKEND: ${DCP_A2A_LARGE_BACKEND:-ag_rs}
      MTP: ${MTP:-0}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      GRAPH: ${GRAPH:-}
      MAX_MODEL_LEN: ${MAX_MODEL_LEN:-131072}
      MAX_BATCHED_TOKENS: ${MAX_BATCHED_TOKENS:-8192}
      GPU_MEMORY_UTILIZATION: ${GPU_MEMORY_UTILIZATION:-0.90}
      MOE_MODE: ${MOE_MODE:-a4}
      MOE_BACKEND: ${MOE_BACKEND:-b12x}
      LINEAR_BACKEND: ${LINEAR_BACKEND:-auto}
      ONLINE_MXFP8: ${ONLINE_MXFP8:-0}
      ONLINE_FP8: ${ONLINE_FP8:-0}
      ONLINE_FP8_MXFP4: ${ONLINE_FP8_MXFP4:-0}
      ONLINE_QUANT: ${ONLINE_QUANT:-}
      F8_DMA: ${F8_DMA:-0}
      LOAD_FORMAT: ${LOAD_FORMAT:-instanttensor}
      INSTANTTENSOR_BACKEND: ${INSTANTTENSOR_BACKEND:-BUFFERED}
      QUANTIZATION: ${QUANTIZATION:-modelopt_fp4}
      QUANTIZATION_CONFIG_JSON: ${QUANTIZATION_CONFIG_JSON:-}
      GLM52_INDEX_TOPK_PATTERN: ${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}
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
| `DCP_BACKEND` | `a2a` | DCP communication mode for `DCP>1`; paired with the token cap below for hybrid dispatch. |
| `DCP_A2A_MAX_TOKENS` | `64` | Hybrid cutoff: B12X one-shot A2A for batches up to this many tokens, large-backend above it. `0` disables the cap and restores pure A2A. |
| `DCP_A2A_LARGE_BACKEND` | `ag_rs` | Backend for batches above the cap; `ag_rs` is the prefill-preserving default. |
| `MTP` | `0` | Native MTP speculative token count; `0` disables MTP. |
| `MAX_NUM_SEQS` | `64` | vLLM concurrency cap. |
| `GRAPH` | `4 * MAX_NUM_SEQS` | CUDA graph capture cap; normally leave unset. |
| `MAX_BATCHED_TOKENS` | `8192` | Chunked-prefill scheduler token cap; lower values can free some KV memory. |
| `MAX_MODEL_LEN` | `131072` | Maximum model context length. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | vLLM memory budget fraction. |
| `MOE_MODE` | `a4` | `a4` = checkpoint-native NVFP4/A4, `a16` = force W4A16, `force-a8-experimental` = explicit A8 force for non-NVFP4 experiments. |
| `ONLINE_QUANT` | `none` | `mxfp8` converts eligible BF16 dense linears to MXFP8; `fp8` converts eligible BF16 dense linears to FP8 block format while leaving checkpoint MXFP4 experts intact. |
| `ONLINE_MXFP8` | `0` | Backward-compatible alias for `ONLINE_QUANT=mxfp8`. |
| `ONLINE_FP8` | `0` | Backward-compatible alias for `ONLINE_QUANT=fp8`. |
| `ONLINE_FP8_MXFP4` | `0` | Legacy alias for `ONLINE_QUANT=fp8`. |
| `QUANTIZATION` | `modelopt_fp4` | Use `mxfp4` for the AMD MXFP4 experts checkpoint. |
| `QUANTIZATION_CONFIG_JSON` | generated | Optional advanced override for the online quantization JSON. |
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

## Online FP8/MXFP8 Conversion

The online variants use the overlay image and pass one of these vLLM
quantization configs:

```bash
ONLINE_QUANT=mxfp8
```

```bash
--quantization modelopt_fp4 \
--quantization-config '{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}'
```

```bash
ONLINE_QUANT=fp8
```

```bash
--quantization modelopt_fp4 \
--quantization-config '{"linear":{"weight":"fp8_per_block_static"},"ignore":["lm_head","model.layers.78.eh_proj","re:.*kv_b_proj","re:.*\\.mlp\\.gate$","re:.*\\.self_attn\\.indexer\\.weights_proj$","re:.*\\.self_attn\\.indexers_proj$"]}'
```

In this overlay, `linear.weight=mxfp8` or `linear.weight=fp8_per_block_static`
means: when a module is treated as a linear layer and has a BF16 weight that is
eligible for online requant, load that weight in the selected online format
instead of leaving it BF16. Existing checkpoint MXFP4 experts stay MXFP4.
Layers that the overlay keeps BF16, including the sparse indexer projection path,
are not forced by the generic `linear` rule.

Both presets now ignore `kv_b_proj` (added 2026-07-07): MLA absorb dequantizes
`kv_b_proj` at load into BF16 `W_UK`/`W_UV` copies and the quantized GEMM never
runs, so quantizing it buys zero speed and only bakes rounding noise into every
attention read. Measured with the paired teacher-forced logprob probe
(GLM-5.2, 3.5k tokens, vs the clean NVFP4 baseline): mean|Δlogprob| 0.152 →
0.144 and max 5.5 → 4.5 at identical throughput. At KLD level this is below
the harness noise floor (the whole dense overlay measures 0.1068 → 0.1076
± 0.004), so it is a free-quality detail, not a regression fix. The same
reasoning applies to the offline zai-style FP8 checkpoints, which historically
quantized `kv_b_proj` for checkpoint size only.

Shared experts: the `linear` spec never quantizes shared-expert projections on
any checkpoint format (as of `6a784b94` in
[PR #77](https://github.com/local-inference-lab/vllm/pull/77), the MXFP4
overlay path matches the ModelOpt behavior). Quantizing shared experts measured
strictly worse on GLM-5.2 — quality 0.156 vs 0.152 mean|Δlogprob| **and**
decode 90.1 vs 92.5 tok/s — so they stay BF16 unless an explicit
`"shared_experts"` spec is passed.

The baseline variant does not pass `--quantization-config`; it loads the Luke
checkpoint as `--quantization modelopt_fp4`.

## BF16 MXFP4 Experts Checkpoint

The same final image can also serve the AMD MXFP4 experts checkpoint:

```text
festr2/GLM-5.2-BF16-AMDMXFP4experts
/root/models/GLM-5.2-BF16-AMDMXFP4experts
```

This checkpoint keeps MXFP4 experts and can convert eligible BF16 dense linears
online at load time:

```bash
MODEL=/root/models/GLM-5.2-BF16-AMDMXFP4experts \
SERVED_MODEL_NAME=GLM-5.2-BF16-AMDMXFP4experts-online \
QUANTIZATION=mxfp4 \
MOE_MODE=force-a8-experimental \
ONLINE_QUANT=mxfp8   # or fp8
```

The measured prefill speed is effectively the same for online MXFP8 and online
FP8 dense conversion. `f8=ring` is what changes prefill throughput materially.

| Online quant | f8 DMA | Prefill 30k | Prefill 64k | Prefill 120k |
|---|---|---:|---:|---:|
| `mxfp8` | `0` | 6,706 | 6,396 | 6,058 |
| `fp8` | `0` | 6,638 | 6,350 | 5,984 |
| `mxfp8` | `ring` | 8,303 | 7,841 | 7,284 |
| `fp8` | `ring` | 8,304 | 7,837 | 7,271 |

Decode cc1/coding probes show the FP8 dense format as the faster decode choice:
the FP8-MXFP4 decode probe measured 101.9 aggregate tok/s and 102.5 coding
peak, while the comparable MXFP8-MXFP4 probe measured 97.2 aggregate tok/s and
97.6 coding peak. In other words, FP8 dense is about +5 tok/s on this A8 MXFP4
experts path, while long prefill remains parity within noise.

KV cache probe for the same MXFP4 checkpoint, online MXFP8 dense conversion,
DCP1, MTP3, `MOE_MODE=force-a8-experimental`, `MAX_MODEL_LEN=262144`,
`F8_DMA=0`, and `GPU_MEMORY_UTILIZATION=0.98`:

| TP | Max seqs | Graph | Max batched tokens | Model load | Available KV memory | GPU KV cache size | Max concurrency at 262,144 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 64 | 256 | 8192 | 50.33 GiB | 40.63 GiB | 785,920 | 3.00x |
| 8 | 32 | 128 | 4096 | 50.21 GiB | 41.60 GiB | 816,064 | 3.11x |

Lowering `MAX_NUM_SEQS` from 64 to 32 and `MAX_BATCHED_TOKENS` from 8192 to
4096 freed enough scheduler/cudagraph budget to add about 30k KV tokens on TP8.
`GPU_MEMORY_UTILIZATION=0.99` failed the startup free-memory check on this node.

The same A8/MXFP4 setup used to be invalid at `TP=6`: B12X W4A8-MX rejected
the TP6 shard shape during weight preparation with
`W4A8-MX QMMA layout requires hidden_size % 256 == 0 and intermediate_size % 128 == 0`,
and the packed W4A16 e8m0 path failed at profile run with
`no valid W4A16 tile config for M/N/K=16384/6144/352` (GLM-5.2
`moe_intermediate 2048 / TP6 -> 352 per rank`, a multiple of 32 but not 128;
NVFP4 checkpoints were unaffected because modelopt sources take the
SOURCE_NATIVE w4a16 layout). Fixed by
[vllm PR #80](https://github.com/local-inference-lab/vllm/pull/80): the
vLLM/B12X weight handoff zero-pads e8m0 expert shards to the next 128 multiple
(352 → 384, bit-exact — padded gate/up rows produce `silu(0)*0 = 0` and the
padded w2 columns only multiply those zeros; ~9% extra expert GEMM work on the
padded rank), plus [b12x PR #26](https://github.com/lukealonso/b12x/pull/26)
so the `tiny_decode` M≤4 kernel accepts the padded 384 (odd FC2 K-tile
counts). See the TP6 Notes section for measured numbers.

Retested on the v7 image with `MAX_NUM_SEQS=16`, `MAX_BATCHED_TOKENS=2048`,
`MAX_MODEL_LEN=128000`, `MTP=0`, hybrid DCP defaults, `ONLINE_QUANT=mxfp8`,
and `MOE_MODE=force-a8-experimental`. `DCP=1` used
`GPU_MEMORY_UTILIZATION=0.957`; `DCP>1` used `0.950` because `0.957` OOMs in
`warmup_dynamic_launches`. All rows booted and completed decode/prefill.

| TP | DCP | KV cache tokens | Max conc at 128k | Decode ctx3k cc1 | Prefill 8k | Prefill 64k |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 335,168 | 2.62x | 82.12 | 5,628 | 5,315 |
| 6 | 2 | 639,616 | 5.00x | 66.13 | 4,010 | 3,862 |
| 6 | 3 | 958,944 | 7.49x | 63.97 | 3,167 | 3,218 |
| 6 | 6 | 1,904,670 | 14.88x | 49.54 | 2,150 | 2,133 |

Result root:

```text
/root/bench-results/glm52-v14-v7-tp6-mxfp4-a8-20260707T115913Z
```

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

## TP6 Notes

Validated 2026-07-07 on the v14/v7 image (`vllme2e2eaf-b12x26144c0`). TP6
works, but not with the original v14 helper defaults. Three independent failure
modes were hit and resolved:

1. **KV-cache OOM with default memory settings.** TP6 stores 1/6 of the
   weights per GPU instead of 1/8, so with the helper defaults
   (`GPU_MEMORY_UTILIZATION=0.90`, `MAX_MODEL_LEN=131072`,
   `MAX_BATCHED_TOKENS=8192`, `MAX_NUM_SEQS=32`) startup dies with:

   ```text
   ValueError: To serve at least one request with the model's max seq len
   (131072), (6.68 GiB KV cache is needed, which is larger than the available
   KV cache memory (2.18 GiB). ... estimated maximum model length is 42752.
   ```

   Use the v13-era memory shape instead: `GPU_MEMORY_UTILIZATION=0.957`,
   `MAX_MODEL_LEN=128000`, `MAX_BATCHED_TOKENS=2048`, and either a small
   `MAX_NUM_SEQS` or DCP to spread the KV cache.

2. **DCP3/DCP6 boot regression (fixed).** The B12X PCIe DCP channel only
   supports world sizes 2/4/8. The DCP collective warmup introduced after v13
   ("Optimize B12X DCP collectives and warmup") treated that as fatal and
   killed EngineCore with
   `RuntimeError: B12X PCIe DCP query all-gather is unavailable for the
   configured attention geometry`, even though the runtime falls back to NCCL
   per call (which is why the same config booted on v13). Fixed by
   [PR #79](https://github.com/local-inference-lab/vllm/pull/79)
   (`b75e72993c`, cherry-picked into this image as `cd272c7b1a`): unsupported
   DCP world sizes log a warning and use NCCL collectives. In the v5 image,
   TP6 with `DCP=3` or `DCP=6` no longer needs site-packages patch mounts.

3. **TP6 DCP2 B12X pool contiguity regression (fixed).** With TP6 virtual head
   padding, sparse MLA returns LSE as a sliced view. The NCCL and AG+RS paths
   accept strided views, but the B12X PCIe DCP pool validates contiguity. That
   made the unique combination `TP=6`, `DCP>1`, and B12X A2A fail with:

   ```text
   partial_lse must be contiguous
   ```

   Fixed by [PR #81](https://github.com/local-inference-lab/vllm/pull/81):
   after cheap reject checks and before the B12X pool call, `cp_attn_lse` is
   made contiguous. The LSE tensor is tiny (`[B,H]` fp32), so the copy is only
   paid on the padded TP6/DCP path and graph capture allocates it in the graph
   pool.

Still true on v14:

- The head66 virtual-TP padding is automatic (`attention heads 64 -> 66` in
  the boot log); no extra switch is needed.
- Use `B12X_MLA_SPARSE` and DCP values that divide 6: `1`, `2`, `3`, `6`.
  `DCP=2` keeps the B12X PCIe DCP channel (world size 2); `DCP=3`/`DCP=6`
  use NCCL DCP collectives.
- The v13 workaround `VLLM_ENABLE_PCIE_ALLREDUCE=0` is **no longer needed**:
  PCIe oneshot/fused-RMS allreduce at world size 6 started and ran cleanly in
  both validated configurations.
- The A8/MXFP4 shard-shape rejection (and the matching packed-W4A16 e8m0 tile
  failure) is fixed by two stacked changes:
  [vllm PR #80](https://github.com/local-inference-lab/vllm/pull/80) zero-pads
  e8m0 expert shards 352 → 384 at the vLLM/B12X handoff (bit-exact — padded
  gate/up rows produce `silu(0)*0 = 0`; ~9% extra expert GEMM work), and
  [b12x PR #26](https://github.com/lukealonso/b12x/pull/26) teaches the
  `tiny_decode` M≤4 kernel odd FC2 K-tile counts (`n % 128` instead of
  `n % 256`), so A8 decode keeps its fast path at the padded 384. Measured on
  the AMD MXFP4 experts checkpoint, TP6/DCP1, MTP0, online MXFP8 dense overlay,
  v13 memory shape (all outputs 0 CJK in the smoke probes). In the v7 full TP6
  helper run the fixed A8 row measured 335,168 KV tokens, 82.12 tok/s decode at
  ctx3k, 5,628 tok/s prefill at 8k, and 5,315 tok/s prefill at 64k. Earlier
  intermediate validation without the b12x #26 tiny-decode fix measured about
  71 tok/s decode, so #26 is required to keep the padded 384 path fast.

Reproduce the current TP6/MXFP4 A8 table with the checked-in helper:

```bash
cd /root/rtx6kpro
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707 \
RESULT_ROOT=/root/bench-results/glm52-v14-v7-tp6-mxfp4-a8-20260707T115913Z \
./scripts/bench-glm52-v14-tp6-mxfp4-v7.sh run
```

The helper uses GPUs `0-5` and `8-13` in pairs, starts both servers, waits for
both `/v1/models` endpoints, sleeps another 30 seconds, and only then starts
benchmarks. It writes progress and summaries under `${RESULT_ROOT}` and keeps
`DCP=1` at GMU `0.957` while using `0.950` for `DCP>1`.

`DCP=1` is the fast single-stream shape; `DCP=6` multiplies KV capacity about
5.7x for long-context or multi-user serving at roughly 60% of the DCP1 decode
speed. `DCP=2` is the middle ground and keeps the B12X DCP fast path.

## Hybrid DCP v5 Decode And KV

Retested on the v5 image with the helper script after fixing the benchmark
client invocation (`--host/--port`, `--no-hw-monitor`, and required JSON
existence checks):

```text
/root/bench-results/glm52-v14-dcp-hybrid-v5-tp8-fixed-20260707T0330Z
```

Config: Luke NVFP4 checkpoint, `TP=8`, `MOE_MODE=a16`, `MTP=3`, `F8_DMA=0`,
`MAX_MODEL_LEN=131072`, `MAX_NUM_SEQS=32`, `MAX_BATCHED_TOKENS=8192`,
`GRAPH=128`, `GPU_MEMORY_UTILIZATION=0.90`, `DCP_BACKEND=a2a`,
`DCP_A2A_MAX_TOKENS=64`, `DCP_A2A_LARGE_BACKEND=ag_rs`.

Decode values are `llm_decode_bench` ctx0 `aggregate_tps` for concurrency
`1,2,4,8,16,32`.

| TP | DCP | KV cache tokens | Max conc at 131k | cc1 | cc2 | cc4 | cc8 | cc16 | cc32 | cc32 accept |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 2 | 1,079,424 | 8.24x | 111.89 | 176.03 | 339.21 | 493.84 | 763.46 | 1,144.29 | 0.579 |
| 8 | 4 | 2,158,848 | 16.47x | 108.15 | 164.63 | 302.63 | 458.65 | 669.88 | 1,011.82 | 0.594 |
| 8 | 8 | 4,302,336 | 32.82x | 101.50 | 155.71 | 277.93 | 401.52 | 532.59 | 793.33 | 0.576 |

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

## KLD Keypoint Rerun

Retested on the v7 image with 5 runs per case:

```text
/root/kld/glm52_v14_keypoints_20260707Tkeypoints-v7
```

Reference logits:

```text
/root/kld/glm52_refs/bf16-b12xmlasparse-w1-ctx2048-s512-20260618
```

Harness settings: `context_length=2048`, `stride=512`, `max_windows=1`,
`load_format=instanttensor`, `INSTANTTENSOR_BACKEND=BUFFERED`, and
`VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4`. Online MXFP8 used:

```json
{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}
```

| Checkpoint | MoE mode | Online MXFP8 | Runs | KLD mean +/- sd | Min | Max |
|---|---|---:|---:|---:|---:|---:|
| Luke NVFP4 | A4 | no | 5 | 0.10734 +/- 0.00416 | 0.10154 | 0.11087 |
| Luke NVFP4 | A4 | yes | 5 | 0.10901 +/- 0.00564 | 0.10490 | 0.11724 |
| Luke NVFP4 | A16 | no | 5 | 0.06662 +/- 0.00130 | 0.06535 | 0.06838 |
| Luke NVFP4 | A16 | yes | 5 | 0.07188 +/- 0.00203 | 0.06964 | 0.07457 |
| BF16 AMD MXFP4 experts | A8 force | no | 5 | 0.07610 +/- 0.00087 | 0.07486 | 0.07730 |
| BF16 AMD MXFP4 experts | A8 force | yes | 5 | 0.07741 +/- 0.00060 | 0.07638 | 0.07782 |

## TP8 Hybrid DCP MTP0 Comparison

Current TP8, MTP0, `f8=0` comparison for the key GLM 5.2 variants. The `DCP2/4/8`
columns use the v7 hybrid DCP policy (`a2a` for small decode batches, `ag_rs`
for larger prefill/extend batches). `DCP1` has no DCP collective; its values
reuse the already-published DCP1 measurements where applicable, plus the missing
MXFP4 `cc32` rerun from the table helper. No server was benchmarked while the
paired 8-GPU instance was still loading.

Settings: `TP=8`, `MTP=0`, `MAX_NUM_SEQS=32`, `GRAPH=128`,
`MAX_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=131072`, `F8_DMA=0`,
`LOAD_FORMAT=instanttensor`, `INSTANTTENSOR_BACKEND=BUFFERED`. `cc1` and `cc32`
are `llm_decode_bench` ctx0 aggregate tok/s; prefill values are standalone prompt
tok/s.

Result roots:

```text
/root/bench-results/glm52-v14-v7-tp8-mxfp4-a8-dcp1-mtp0-20260707T133344Z
/root/bench-results/glm52-v14-v7-tp8-hybrid-table-mtp0-20260707T135950Z
```

Reproduce:

```bash
cd /root/rtx6kpro
./scripts/bench-glm52-v14-tp8-hybrid-table-v7.sh run
```

### Best Readout

This table keeps the accuracy/speed tradeoff on one screen. `Best decode DCP`
uses `cc32`; `Best prefill DCP` uses 64k prefill. For this particular TP8/MTP0
sweep both are still `DCP1`, while `DCP2/4/8` are mainly useful for larger KV
capacity.

| Case | KLD mean | DCP1 cc1 | DCP1 cc32 | DCP1 prefill 8k | DCP1 prefill 64k | Best decode DCP | Best prefill DCP |
|---|---:|---:|---:|---:|---:|---|---|
| Luke NVFP4 A4 orig | 0.10734 | 87.99 | 934.07 | 6,557 | 6,257 | DCP1 | DCP1 |
| Luke NVFP4 A4 online MXFP8 | 0.10901 | 94.96 | 953.24 | 6,681 | 6,351 | DCP1 | DCP1 |
| Luke NVFP4 A16 orig | 0.06662 | 86.56 | 932.72 | 6,140 | 5,849 | DCP1 | DCP1 |
| Luke NVFP4 A16 online MXFP8 | 0.07188 | 93.30 | 954.52 | 6,239 | 5,941 | DCP1 | DCP1 |
| BF16 AMD MXFP4 experts A8 orig | 0.07610 | 88.72 | 938.10 | 6,698 | 6,307 | DCP1 | DCP1 |
| BF16 AMD MXFP4 experts A8 online MXFP8 | 0.07741 | 94.03 | 956.30 | 6,731 | 6,364 | DCP1 | DCP1 |

### Decode cc1

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
|---|---:|---:|---:|---:|
| Luke NVFP4 A4 orig | 87.99 | 72.44 | 71.65 | 67.29 |
| Luke NVFP4 A4 online MXFP8 | 94.96 | 76.26 | 75.32 | 70.84 |
| Luke NVFP4 A16 orig | 86.56 | 71.48 | 70.74 | 66.11 |
| Luke NVFP4 A16 online MXFP8 | 93.30 | 74.85 | 73.99 | 69.45 |
| BF16 AMD MXFP4 experts A8 orig | 88.72 | 71.84 | 71.73 | 67.15 |
| BF16 AMD MXFP4 experts A8 online MXFP8 | 94.03 | 75.66 | 75.37 | 71.01 |

### Decode cc32

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
|---|---:|---:|---:|---:|
| Luke NVFP4 A4 orig | 934.07 | 838.57 | 747.11 | 606.35 |
| Luke NVFP4 A4 online MXFP8 | 953.24 | 847.24 | 760.87 | 617.18 |
| Luke NVFP4 A16 orig | 932.72 | 828.30 | 750.20 | 610.88 |
| Luke NVFP4 A16 online MXFP8 | 954.52 | 837.81 | 752.91 | 610.40 |
| BF16 AMD MXFP4 experts A8 orig | 938.10 | 832.28 | 745.91 | 613.70 |
| BF16 AMD MXFP4 experts A8 online MXFP8 | 956.30 | 840.02 | 761.43 | 607.69 |

### Prefill 8k

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
|---|---:|---:|---:|---:|
| Luke NVFP4 A4 orig | 6,557 | 4,679 | 3,415 | 2,197 |
| Luke NVFP4 A4 online MXFP8 | 6,681 | 4,636 | 3,402 | 2,188 |
| Luke NVFP4 A16 orig | 6,140 | 4,455 | 3,301 | 2,147 |
| Luke NVFP4 A16 online MXFP8 | 6,239 | 4,385 | 3,270 | 2,132 |
| BF16 AMD MXFP4 experts A8 orig | 6,698 | 4,747 | 3,450 | 2,206 |
| BF16 AMD MXFP4 experts A8 online MXFP8 | 6,731 | 4,702 | 3,427 | 2,200 |

### Prefill 64k

| Case | DCP1 | DCP2 | DCP4 | DCP8 |
|---|---:|---:|---:|---:|
| Luke NVFP4 A4 orig | 6,257 | 4,710 | 3,455 | 2,209 |
| Luke NVFP4 A4 online MXFP8 | 6,351 | 4,718 | 3,468 | 2,212 |
| Luke NVFP4 A16 orig | 5,849 | 4,481 | 3,326 | 2,157 |
| Luke NVFP4 A16 online MXFP8 | 5,941 | 4,471 | 3,331 | 2,155 |
| BF16 AMD MXFP4 experts A8 orig | 6,307 | 4,786 | 3,491 | 2,220 |
| BF16 AMD MXFP4 experts A8 online MXFP8 | 6,364 | 4,781 | 3,495 | 2,223 |

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

Reproducible runners are checked in at:

```text
scripts/bench-glm52-v14-todo-and-sweep.sh
scripts/bench-glm52-v14-kld-keypoints.sh
scripts/bench-glm52-v14-dcp-hybrid-v5.sh
scripts/bench-glm52-v14-tp6-mxfp4-v7.sh
```

Run the original complete v14 sweep:

```bash
cd /root/rtx6kpro
RUN_ID=20260706T0150Z \
RESULT_ROOT=/root/bench-results/glm52-v14-todo-and-sweep-20260706T0150Z \
KLD_ROOT=/root/kld/glm52_v14_todo_20260706T0150Z \
./scripts/bench-glm52-v14-todo-and-sweep.sh all
```

Run the v7 KLD keypoint rerun:

```bash
cd /root/rtx6kpro
IMAGE=voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707 \
RUN_ID=keypoints-v7 \
KLD_ROOT=/root/kld/glm52_v14_keypoints_20260707Tkeypoints-v7 \
RUNS=5 \
./scripts/bench-glm52-v14-kld-keypoints.sh all
```

Run the v5/v7 hybrid-DCP follow-up sweeps:

```bash
cd /root/rtx6kpro
RESULT_ROOT=/root/bench-results/glm52-v14-dcp-hybrid-v5-tp8-fixed-20260707T0330Z \
./scripts/bench-glm52-v14-dcp-hybrid-v5.sh tp8-decode

IMAGE=voipmonitor/vllm:eldritch-enlightenment-v7-vllme2e2eaf-b12x26144c0-cu132-20260707 \
RESULT_ROOT=/root/bench-results/glm52-v14-v7-tp6-mxfp4-a8-20260707T115913Z \
./scripts/bench-glm52-v14-tp6-mxfp4-v7.sh run
```

The runner supports narrower reruns:

```bash
./scripts/bench-glm52-v14-todo-and-sweep.sh table-todos
./scripts/bench-glm52-v14-todo-and-sweep.sh kld-todos
./scripts/bench-glm52-v14-todo-and-sweep.sh full-sweep
./scripts/bench-glm52-v14-todo-and-sweep.sh summarize
```

The v5/v7 DCP helpers write progress to `${RESULT_ROOT}/progress.log` by default.
The original 2026-07-06 full-sweep helper appends progress to:

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
