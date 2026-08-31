# DeepSeek-V4-Pro TP16 Lucifer on 16x RTX PRO 6000 Blackwell

Status: live smoke-tested on 2026-06-04 on `10.229.14.14`.

This page records the exact Lucifer-based DS4-Pro TP16 run that starts on all
16 GPUs with FP8 KV and MTP enabled. The published image is our DockerHub tag
`voipmonitor/vllm:ds4-pro-tp16-lucifer-vllm1967a5627bc3-fp8pad-wsfix-20260604`.
It is not a clean source build: it is `cstechdev/dsv4-flash:latest` plus two
site-packages overlay patches needed for true TP16.

## Current Instance

| Field | Value |
|---|---|
| Host | `10.229.14.14` |
| Container | `ds4-pro-lucifer-tp16-fp8pad` |
| Image | `voipmonitor/vllm:ds4-pro-tp16-lucifer-vllm1967a5627bc3-fp8pad-wsfix-20260604` |
| Model path | `/root/models/DeepSeek-V4-Pro` |
| Served model | `deepseek-v4-pro` |
| Port | `8400` |
| GPUs | `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15` |
| TP | `16` |
| MTP | enabled, `num_speculative_tokens=2` |
| KV cache dtype | `fp8` |
| Max model len | `393216` |
| Prefix caching | enabled |

Runtime log checks:

```text
vLLM is using nccl==2.28.9
Custom allreduce is disabled due to an unsupported world size: 16. Supported world sizes: [2, 4, 6, 8].
GPU KV cache size: 494,190 tokens
Maximum concurrency for 393,216 tokens per request: 1.26x
Graph capturing finished in 61 secs, took 2.24 GiB
```

At TP16 this image uses PYNCCL for allreduce because the Lucifer custom
allreduce path does not support `world_size=16`.

## Image Metadata

| Component | Revision |
|---|---|
| Base image | `cstechdev/dsv4-flash:latest` |
| Base image digest | `sha256:27b80536a36212cef21664699aee35acbc14b37f147d41cd9b12361154f3c4db` |
| Base image ID | `sha256:70c3d635217676d1f9d5789e3ee67d6343be0ad04d2ca1a660668b891a2269e3` |
| TP16 FP8 pad image | `cstechdev/dsv4-flash:latest-tp16-fp8pad` |
| TP16 FP8 pad image ID | `sha256:edfb0b70a98e325c7b0a428376931703842a89154cde8d35d32957300ae548ec` |
| Final published image | `voipmonitor/vllm:ds4-pro-tp16-lucifer-vllm1967a5627bc3-fp8pad-wsfix-20260604` |
| Final published digest | `sha256:bedfe52fc8d7ec41e34d84b342691603a6e70bf9b9705b2222e158281d81cb83` |
| Local build tag on `10.229.14.14` | `cstechdev/dsv4-flash:latest-tp16-fp8pad-wsfix` |
| Final wsfix image ID | `sha256:a6584693ce12b0f625ffcd80da2a65710ecab87bdb4f0c6f1b9ed60a72339e90` |
| vLLM package | `0.21.1rc1.dev339+g1967a5627bc3` |
| vLLM full commit resolved from GitHub | `1967a5627bc3710b680bbec24ecb99aaddedf22b` |
| vLLM commit title | `fix(sm120): pass scratch for small prefill chunks` |
| PyTorch | `2.11.0+cu130` |
| FlashInfer wheel | `flashinfer-python 0.6.12` |
| NCCL Python package | `nvidia-nccl-cu13 2.28.9` |

The overlay images have no registry digest in `docker image inspect`, so they
were local tags on `10.229.14.14` when recorded.

## Recovering vLLM Git From The Docker

The installed image does not contain a vLLM `.git` checkout and the Docker
labels do not include source branch metadata. The recoverable in-image facts
are:

```bash
docker exec ds4-pro-lucifer-tp16-fp8pad /opt/env/bin/python - <<'PY'
import importlib.metadata as md
import vllm
print(md.version("vllm"))
print(vllm.__file__)
PY
```

Expected output:

```text
0.21.1rc1.dev339+g1967a5627bc3
/opt/env/lib/python3.12/site-packages/vllm/__init__.py
```

The wheel metadata points only to a local wheelhouse file:

```text
file:///wheelhouse/vllm-0.21.1rc1.dev339%2Bg1967a5627bc3-cp312-cp312-linux_x86_64.whl
sha256=7bf39ee8cecd6480488c2cb943b457e558c1aa32067086e53386154251c93986
```

The full SHA was resolved externally from the `g1967a5627bc3` prefix:

```text
https://github.com/voipmonitor/vllm/commit/1967a5627bc3710b680bbec24ecb99aaddedf22b
https://github.com/local-inference-lab/vllm/commit/1967a5627bc3710b680bbec24ecb99aaddedf22b
https://github.com/vllm-project/vllm/commit/1967a5627bc3710b680bbec24ecb99aaddedf22b
```

To create a matching `voipmonitor/vllm` branch later, branch from
`1967a5627bc3710b680bbec24ecb99aaddedf22b` and then port the two overlay
patches below as normal source commits. The Docker alone is not sufficient to
recover the original build branch name.

## Launch Command

This is the exact command used for the live instance:

```bash
docker run -d --name ds4-pro-lucifer-tp16-fp8pad \
  --gpus all --runtime nvidia --ipc=host --shm-size=32g --network=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/luci-official:/cache \
  -v /root/.cache/luci-official/huggingface:/root/.cache/huggingface \
  -v /root/models/DeepSeek-V4-Pro:/model:ro \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -e NCCL_P2P_LEVEL=SYS \
  voipmonitor/vllm:ds4-pro-tp16-lucifer-vllm1967a5627bc3-fp8pad-wsfix-20260604 \
  serve /model \
  --served-model-name deepseek-v4-pro \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8400 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --kv-cache-dtype fp8 \
  --block-size 256 \
  --tensor-parallel-size 16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 393216 \
  --enable-prefix-caching \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config.reasoning_start_str '<think>' \
  --reasoning-config.reasoning_end_str '</think>' \
  --default-chat-template-kwargs.thinking true \
  --default-chat-template-kwargs.reasoning_effort max \
  --enable-flashinfer-autotune \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 2
```

Do not set `NCCL_GRAPH_FILE=` to an empty value. If it exists in the shell,
unset it before launching.

## Overlay Patch 1: TP16 FP8 Shard Padding

Local source files used to build the first overlay:

```text
/root/vllm/ds4-tp16-fp8patch/Dockerfile
/root/vllm/ds4-tp16-fp8patch/patch_tp16_fp8_row_parallel.py
```

Build recipe:

```dockerfile
FROM cstechdev/dsv4-flash:latest

COPY patch_tp16_fp8_row_parallel.py /tmp/patch_tp16_fp8_row_parallel.py
RUN /opt/env/bin/python /tmp/patch_tp16_fp8_row_parallel.py
```

Files patched inside site-packages:

```text
/opt/env/lib/python3.12/site-packages/vllm/model_executor/layers/linear.py
/opt/env/lib/python3.12/site-packages/vllm/model_executor/parameter.py
/opt/env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/fp8.py
```

Why it was needed:

```text
ValueError: Weight input_size_per_partition = 192 is not divisible by weight quantization block_k = 128
```

DS4-Pro has `moe_intermediate_size=3072`; with `TP=16`, local MoE shards are
`192`, which is not aligned to the FP8 `[128, 128]` block shape. The patch keeps
true TP16 and pads local FP8 row-parallel and merged-column shards to the
required block boundary instead of falling back to DP-style workarounds.

What the patch changes:

| Area | Change |
|---|---|
| `RowParallelLinear` | Tracks unpadded local input size, global shard start, pad start, and pad end. |
| Row-parallel forward | Pads the local input tensor before GEMM when the local shard is not block aligned. |
| `MergedColumnParallelLinear` | Pads per-shard local output partitions so FP8 block-N scale/weight shapes stay valid. |
| Weight loading | Zero-fills padded parameter regions and copies only the real shard into the padded view. |
| FP8 scales | Adjusts block scale slicing for padded row-parallel and merged-column shards. |

Resulting intermediate image:

```text
cstechdev/dsv4-flash:latest-tp16-fp8pad
sha256:edfb0b70a98e325c7b0a428376931703842a89154cde8d35d32957300ae548ec
```

## Overlay Patch 2: Workspace Lock For All DBO Ubatches

Local source files used to build the final overlay:

```text
/root/vllm/ds4-tp16-fp8patch/Dockerfile.wsfix
/root/vllm/ds4-tp16-fp8patch/patch_workspace_lock_all_ubatches.py
```

Build recipe:

```dockerfile
FROM cstechdev/dsv4-flash:latest-tp16-fp8pad

COPY patch_workspace_lock_all_ubatches.py /tmp/patch_workspace_lock_all_ubatches.py
RUN /opt/env/bin/python /tmp/patch_workspace_lock_all_ubatches.py
```

File patched inside site-packages:

```text
/opt/env/lib/python3.12/site-packages/vllm/v1/worker/workspace.py
```

Why it was needed:

```text
AssertionError: Workspace is locked but allocation from 'sm120.py:75:_get_decode_scratch' was attempted.
Workspace growth is not allowed after locking.
```

The first TP16 overlay could load and start, but runtime requests could hit a
different DBO ubatch slot than the one that had been materialized during
warmup. After `lock_workspace()`, that slot attempted to allocate a larger
scratch buffer and the engine died. A secondary CUDA illegal memory access and
NCCL watchdog error followed the worker failure.

What the patch changes:

| Area | Change |
|---|---|
| `WorkspaceManager.lock()` | Calls `_materialize_all_ubatches_to_max_workspace()` before setting `_locked=True`. |
| `_materialize_all_ubatches_to_max_workspace()` | Finds the largest profiled workspace allocation and creates an equal-size `torch.uint8` workspace for every ubatch slot. |
| Runtime effect | Later ubatches no longer grow workspace after the manager is locked. |

Resulting final image:

```text
voipmonitor/vllm:ds4-pro-tp16-lucifer-vllm1967a5627bc3-fp8pad-wsfix-20260604
dockerhub digest: sha256:bedfe52fc8d7ec41e34d84b342691603a6e70bf9b9705b2222e158281d81cb83
local build tag: cstechdev/dsv4-flash:latest-tp16-fp8pad-wsfix
sha256:a6584693ce12b0f625ffcd80da2a65710ecab87bdb4f0c6f1b9ed60a72339e90
```

## Startup Notes

`--gpu-memory-utilization 0.90` was too conservative for the requested
`--max-model-len 393216` after the workspace materialization patch:

```text
available KV cache memory: 6.02 GiB
needed KV cache memory: 6.75 GiB
```

The live instance uses `--gpu-memory-utilization 0.92`, which passes the KV
check and reports `494,190` KV tokens.

## Validation

Smoke tests performed after the wsfix image started:

| Test | Result |
|---|---|
| `GET /v1/models` | HTTP 200, served model `deepseek-v4-pro` |
| 10 short parallel chat requests | 10/10 HTTP 200 |
| 5 long-prefill parallel chat requests | 5/5 HTTP 200, about 21s per request |
| Runtime log after tests | no `Workspace`, `CUDA illegal memory`, or HTTP 500 errors observed |

Live log tail:

```bash
ssh root@10.229.14.14 'docker logs -f ds4-pro-lucifer-tp16-fp8pad'
```

Live health check:

```bash
curl -s http://10.229.14.14:8400/v1/models | python3 -m json.tool
```

## Benchmark Artifact: `/root/benchmark_results.json`

Important: this JSON does not identify the live DS4-Pro TP16 instance. It
reports `model=Kimi-K2.6-NVFP4-B12X`, `server=localhost:8403`, and
`gpu_count=8`, so treat these numbers as an attached benchmark artifact, not as
DS4-Pro TP16 performance.

| Field | Value |
|---|---|
| timestamp | `2026-06-04T16:46:44.431166` |
| benchmark version | `0.4.24` |
| reported model | `Kimi-K2.6-NVFP4-B12X` |
| reported server | `localhost:8403` |
| decode mode | `duration` |
| duration per cell | `30s` |
| max tokens | `8192` |
| ignore eos | `True` |
| contexts | `0, 16k, 32k, 64k, 128k` |
| concurrency levels | `1, 2, 4, 8, 16, 32, 64, 128` |
| P2P override effective | `True` |
| burst/E2E | `not_run_use_--run-burst` |

Prefill, client-measured TTFT path:

| ctx | prompt tokens | TTFT s | client tok/s | server validation tok/s |
|---:|---:|---:|---:|---:|
| 8k | 8,187 | 1.073 | 7,630 | 0 |
| 16k | 16,229 | 2.229 | 7,280 | 7,503 |
| 32k | 32,307 | 4.819 | 6,704 | 6,876 |
| 64k | 64,468 | 11.124 | 5,795 | 5,916 |
| 128k | 128,785 | 27.947 | 4,608 | 4,684 |

Sustained decode aggregate tok/s, 30s cells:

| ctx \ conc | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 90.2 | 150.8 | 223.6 | 333.4 | 469.4 | 572.6 | 828.4 | 1097.4 |
| 16k | 90.3 | 139.2 | 181.8 | 265.2 | 360.8 | 412.3 | — | — |
| 32k | 77.8 | 116.9 | 154.3 | 215.7 | 282.0 | — | — | — |
| 64k | 71.7 | 95.7 | 125.0 | 152.2 | — | — | — | — |
| 128k | 52.3 | 67.8 | 83.7 | 98.7 | — | — | — | — |

Aggregate hardware summary for the benchmark run:

| Metric | Value |
|---|---:|
| GPU count | 8 |
| GPU util avg | 90.1% |
| GPU util max | 100.0% |
| VRAM used avg | 99.0% |
| Power avg W | 1,970 |
| Power max W | 3,344 |
| PCIe RX avg MB/s | 72,903 |
| PCIe TX avg MB/s | 72,057 |

## Known Limitations

The current fix is an overlay, not a clean upstream-quality branch. For a
maintainable branch, port both patches into source with tests around block-FP8
TP partitioning and DBO workspace locking.

The vLLM wheel records the source commit prefix but not the original branch.
Use full commit `1967a5627bc3710b680bbec24ecb99aaddedf22b` as the base for a
matching branch, then apply the TP16 FP8 padding and workspace materialization
changes.
