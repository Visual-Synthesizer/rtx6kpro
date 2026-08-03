# DeepSeek-V4-Flash-0731 DSpark: Gilded Gnosis r24

This is the current DeepSeek-V4-Flash-0731 runbook for RTX PRO 6000
Blackwell. r24 keeps the fixed-K5 SparkInfer profile from r16 and adds the
runtime fixes accumulated during the July 31 to August 3 investigation.

> **Release status:** published. The exact image passed clean source
> composition, build and helper tests, TP2 model startup, a coherent chat
> request, and six concurrent approximately 120k-token native-offload requests.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si2b9bf2a-fi801d57a-cu132-20260803-r24
Docker manifest: sha256:64b94299abdd3bcf5bb5050ca91b378f9ee4e0b0eff4748375b95352371d7cb2
Local image ID: sha256:dc0bc459b8c1d59f84e945a4b77f65ea474778b58f4a95d3e3d1c97632daeb1d
```

## Quick Start

The helper is included in the image. No external launcher or source mount is
required.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout build/gilded-gnosis-r21-ds4-runtime-20260802

GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r24.yml up -d
```

Readiness and logs:

```bash
curl -fsS http://127.0.0.1:8000/health
docker logs -f ds4-0731-r24
```

The release defaults are:

| Setting | Default | Meaning |
|---|---:|---|
| `MODE` | `dspark` | Native DSpark serving for the 0731 checkpoint |
| `BACKEND` | `b12x-a8` | SparkInfer W4A8 target path |
| `TP_SIZE` / `DCP_SIZE` | `2` / `1` | Validated DSpark topology |
| `DSPARK_DEPTH_MODE` | `fixed` | Fixed draft depth |
| `DSPARK_TOKENS` | `5` | Recommended K5 profile |
| `MAX_NUM_SEQS` | `16` | Scheduler concurrency |
| `MAX_MODEL_LEN` | `131072` | Conservative release envelope |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Prefill scheduler budget |
| `GPU_MEMORY_UTILIZATION` | `0.975` | Default TP2 memory target |
| `LOAD_FORMAT` | `instanttensor` | Default model loader |
| `INSTANTTENSOR_BACKEND` | `BUFFERED` | Page-cache-backed loading |
| `KV_OFFLOADING_SIZE` | `0` | Native CPU KV offload disabled |

The graph cap is derived from concurrency and physical verifier width. For
fixed K5 with 16 sequences, the default is `16 * (5 + 1) = 96`.

## Native CPU KV Offload

Native offload is independent from LMCache. Enable it with the total host
capacity in GiB across all TP ranks:

```bash
KV_OFFLOADING_SIZE=48.5 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r24.yml up -d
```

Positive decimal and non-power-of-two values are supported. The backing region
is one process-shared mmap in host `/dev/shm`. r24 unlinks its pathname after
all workers map it, so normal shutdown and abrupt process death both release
the storage when the final mapping closes.

r24 also fixes segmented registration. Registration boundaries are aligned to
both the physical KV row stride and 64 KiB. This matters because one batched
GPU copy descriptor can span many KV rows; a descriptor must never cross from
one independently registered host segment into the next.

The final TP2 test used:

```text
TP2 / DCP1 / fixed K5 / B12X A8
MAX_NUM_SEQS=16
MAX_CUDAGRAPH_CAPTURE_SIZE=16
MAX_MODEL_LEN=131072
GPU_MEMORY_UTILIZATION=0.97
KV_OFFLOADING_SIZE=5.5
```

Six independent prompts were submitted concurrently. Each contained
119,957-119,963 input tokens. All 6 requests completed, the wall time was
62.153 seconds, the server remained healthy, and the log contained no CUDA,
batched-copy, or registration error. Store intervals moved approximately
2.5-2.66 GB in 44-47 ms, about 55-56 GB/s.

## Why K5

K7 is still available, but it is not the release default. Matched TP2 tests
favored K5:

| Draft depth | Sustained decode | Coding median |
|---|---:|---:|
| K5 | 217.8 tok/s | 289.4 tok/s |
| K7 | 192.1 tok/s | 281.2 tok/s |

Community reports also associate K7 with intermittent long-context BOS/tool
pollution. That quality investigation is not closed. Use K7 only as an
explicit experiment:

```bash
DSPARK_TOKENS=7 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r24.yml up -d
```

Load-aware draft depth remains opt-in:

```bash
DSPARK_DEPTH_MODE=dynamic GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r24.yml up -d
```

## TP4 Lucifer CUTLASS

```bash
GPUS=0,1,2,3 TP_SIZE=4 BACKEND=lucifer-cutlass \
  docker compose -f examples/docker-compose-ds4-v20-r24.yml up -d
```

The exact r24 image started successfully with TP4, DCP1, fixed K5,
`MAX_NUM_SEQS=8`, graph 48, and the Lucifer CUTLASS backend. It selected
`FLASHINFER_CUTLASS_MXFP4_MXFP8`, provisioned 771,031 GPU KV tokens, returned
exactly `42` from the chat sanity request, and logged no
`tf32_hc_prenorm_gemm`, CUDA, or startup error.

## Changelog Since r16

### DS4 and vLLM runtime

| Change | Practical effect |
|---|---|
| [vLLM #229](https://github.com/local-inference-lab/vllm/pull/229) | Sizes compressed MLA workspaces from the physical cache contract and prevents TP2/K5 long-concurrency under-reservation. |
| [vLLM #217](https://github.com/local-inference-lab/vllm/pull/217) | Shared native-offload region, decimal sizing, unlink-after-map lifetime, isolated registration failures, and row/64-KiB-aligned segmented registration. |
| [vLLM #218](https://github.com/local-inference-lab/vllm/pull/218) | Preserves SWA, MTP/EAGLE tails, replay boundaries, retention intervals, and shared-prefix tails during native offload. |
| [vLLM #216](https://github.com/local-inference-lab/vllm/pull/216) | Separates semantic PCIe graph channels so replay state is not reused across incompatible operations. |
| [vLLM #230](https://github.com/local-inference-lab/vllm/pull/230) | Keeps broadcast mHC pre-processing behind the compile boundary. |
| [vLLM #228](https://github.com/local-inference-lab/vllm/pull/228) | Consolidates the mixed-prefill and dense online-K6 runtime used by the unified image. |
| [SparkInfer #113](https://github.com/local-inference-lab/sparkinfer/pull/113) | Hardens replay and CUDA IPC graph-state lifetime. |
| [SparkInfer #112](https://github.com/local-inference-lab/sparkinfer/pull/112) | Consolidates mixed prefill and dense K6 kernels. |
| [SparkInfer #106](https://github.com/local-inference-lab/sparkinfer/pull/106) | Makes compressed MLA kernels honor the physical cache-page stride. |
| [InstantTensor #19](https://github.com/scitix/InstantTensor/pull/19) | Retries large BUFFERED host registration as bounded segments instead of failing the whole model load. |

### LMCache integration

r24 includes the current LMCache recovery and capacity stack: bounded RPC
errors, invalid-block recomputation, largest-exact-prefix recovery, concurrent
key deletion protection, per-key store status, aligned O_DIRECT I/O,
current/legacy object keys, durable capacity-bounded stores, restart-time
capacity reconstruction, and bounded L1 writeback.

This stack passed 222 tests with 131 skipped. It does **not** close the reported
DS4 long-context output-correctness issue. LMCache remains experimental for
DS4; native offload is the qualified host-cache path in r24.

## Validation Summary

| Gate | Result |
|---|---|
| Clean source composition and immutable archive | Pass |
| Build/helper shell suites | Pass |
| Runtime dependency/import checks | Pass |
| Focused native-offload tests | 8 passed |
| LMCache integrated suite | 222 passed, 131 skipped |
| TP2/DCP1 K5 model load | Pass |
| Chat correctness sanity | Returned exactly `42` |
| Six concurrent approximately 120k prompts | 6/6 pass, 62.153 s |
| Post-test server health and error scan | Pass |
| Shared mmap cleanup on shutdown | Pass |
| TP4 Lucifer CUTLASS startup and chat | Pass; 771,031 GPU KV tokens |

## Known Open Items

- K7 long-context output/tool/BOS corruption is not closed; fixed K5 is the
  release default.
- LMCache long-context correctness is not closed for DS4. Its storage and
  capacity failures are fixed, but this is not a quality certification.
- Tool/reasoning anomalies reported beyond roughly 200k context have not been
  conclusively attributed to one runtime component.
- Native L2 NIXL/S3 offload has shown a prefill penalty relative to POSIX on
  current measurements.

## First-Run Compile Cache

The first run compiles SparkInfer, TileLang, and CUDA graph artifacts. Reuse
the same `JIT_CACHE` directory. If a process is interrupted during extension
compilation, PyTorch can leave an empty `lock` file. Only remove such a lock
after confirming that no compiler process holds it; deleting an active lock
can corrupt the shared cache.

## Source Provenance

| Component | Ref |
|---|---|
| Canonical GG base | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| Composed vLLM tree | `f5981f14b4d39979bc0d799c020d42002b707257` |
| SparkInfer base | `59216fa25f3d5fc9d4df2d052e02d05f763906e9` |
| Composed SparkInfer tree | `2b9bf2a4d15770c0c23e19cc13a75843f2f0a995` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| LMCache composed tree | `9a05c8818bae48d15b79c7e876418bb813c08cd0` |
| InstantTensor | `25b3f268ea95b76bd03c825a1681872c9b615428` |
| XGrammar | `0.2.5` |
| PyTorch / CUDA | `2.12.0+cu132` / `13.2.1` |
| cuDNN | `9.20.0.48` runtime package |

The reproducible source archive and Compose file are on
[`build/gilded-gnosis-r21-ds4-runtime-20260802`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/build/gilded-gnosis-r21-ds4-runtime-20260802).

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r24 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The archive verifies every base commit, PR head, composed tree, and patch hash
before building.
