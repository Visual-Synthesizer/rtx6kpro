# DeepSeek-V4-Flash-0731 DSpark: Gilded Gnosis r27

This is the current DeepSeek-V4-Flash-0731 runbook for RTX PRO 6000
Blackwell. r27 retains the fixed-K5 SparkInfer serving profile and fixes the
native tiered-offload backing lifetime, the official 0731 reasoning/tool
prompt contract, and InstantTensor host-registration fallback behavior.

> **Release status:** published. The exact registry image passed clean source
> composition, TP2 model startup, reasoning/tool E2E checks, controlled decode
> comparison, concurrent short and long requests, uncached 64k prefill, and
> native filesystem L2 replay after a complete container restart.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm966d57c-sibbbdccc-fi801d57a-cu132-20260803-r27
Docker manifest: sha256:2605fda01797f33239af4c95ec7449505fe57d9b9de9687792f5b8273d3201a7
Local validation image ID: sha256:ab32600a628d9ea4a7ec4a7e3e4ff4779bd33397010b14d926546b214a3e7556
```

## Quick Start

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout main

GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r27.yml up -d
```

Readiness and logs:

```bash
curl -fsS http://127.0.0.1:8000/health
docker logs -f ds4-0731-r27
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
| `PYTHONHASHSEED` | `0` | Stable persistent native-L2 keys across restarts |
| `KV_OFFLOADING_SIZE` | `0` | Native CPU KV offload disabled |
| `SHM_SIZE` | `32gb` | Container `/dev/shm`; increase with native L1 |

The graph cap is derived from concurrency and physical verifier width. Fixed
K5 with 16 sequences uses `16 * (5 + 1) = 96`.

## Native CPU KV Offload

Native offload is independent from LMCache. `KV_OFFLOADING_SIZE` is the total
host capacity in GiB across all TP ranks. Positive decimal and non-power-of-two
values are supported.

### L1 only

The shared host region lives in `/dev/shm`, so `SHM_SIZE` must exceed the L1
capacity with room for normal runtime IPC:

```bash
SHM_SIZE=64gb KV_OFFLOADING_SIZE=48.5 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r27.yml up -d
```

L1-only mode unlinks the shared-region pathname after every worker has mapped
it. The mapping remains valid while processes use it and is reclaimed when the
final mapping closes.

### Persistent filesystem L2

Tiered mode must keep the named L1 backing available until the delayed
EngineCore scheduler maps the same region. r27 enforces that lifetime and
skips the scheduler's redundant full-region prefault. The filesystem root
below is under the Compose `/cache` mount and therefore survives container
replacement.

`EXTRA_VLLM_ARGS` is parsed as an argument list. Keep the JSON compact and
whitespace-free:

```bash
export EXTRA_VLLM_ARGS='--kv-transfer-config={"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":"/cache/native-l2","n_read_threads":4,"n_write_threads":4}]}}'

SHM_SIZE=8gb \
KV_OFFLOADING_SIZE=4 \
PYTHONHASHSEED=0 \
GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r27.yml up -d
```

Keep `PYTHONHASHSEED` unchanged when persistent objects must be reused after a
full restart.

The exact TP2/DCP1 fixed-K5 validation wrote four independent approximately
70k-token prefixes:

| Check | Result |
|---|---:|
| L2 files | 2,748 non-empty objects |
| L2 bytes | 5,920,555,008 |
| Warm replay cached tokens | 69,888 / 70,018 |
| Warm replay CPU-to-GPU bytes | 635,500,800 |
| Cold-restart cached tokens | 69,888 / 70,018 |
| Cold-restart CPU-to-GPU bytes | 635,500,800 |

The cold result was measured after stopping and recreating the entire
container while preserving only the filesystem directory and hash seed. A 1
GiB L1 cannot hold one complete approximately 70k DS4 prefix group (about 1.48
GB in this profile), so a miss at that capacity is expected rather than an L2
failure.

## Why K5

K7 remains available but is not the release default. Matched historical TP2
tests favored K5:

| Draft depth | Sustained decode | Coding median |
|---|---:|---:|
| K5 | 217.8 tok/s | 289.4 tok/s |
| K7 | 192.1 tok/s | 281.2 tok/s |

K7 has also been associated with intermittent long-context BOS/tool output.
Use it only as an explicit experiment:

```bash
DSPARK_TOKENS=7 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r27.yml up -d
```

Load-aware draft depth remains opt-in:

```bash
DSPARK_DEPTH_MODE=dynamic GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r27.yml up -d
```

## TP4 Lucifer CUTLASS

```bash
GPUS=0,1,2,3 TP_SIZE=4 BACKEND=lucifer-cutlass \
  docker compose -f examples/docker-compose-ds4-v20-r27.yml up -d
```

The last matched TP4 validation was performed on r24 with fixed K5,
`MAX_NUM_SEQS=8`, graph 48, and the Lucifer CUTLASS backend. It selected
`FLASHINFER_CUTLASS_MXFP4_MXFP8`, provisioned 771,031 GPU KV tokens, and
returned the expected chat result. r27 validation focused on the TP2 release
default; this TP4 recipe is retained but was not rerun for r27.

## What Changed Since r24

| Change | Practical effect |
|---|---|
| [vLLM #217](https://github.com/local-inference-lab/vllm/pull/217) | Tiering workers and the delayed scheduler map one named backing; scheduler prefault is removed; timeout and rendezvous cleanup are covered by tests. |
| [vLLM #235](https://github.com/local-inference-lab/vllm/pull/235) | Aligns low/high/max reasoning prompts and system/developer/request tool ordering with the official 0731 template in both Python and Rust. |
| [InstantTensor #19](https://github.com/scitix/InstantTensor/pull/19) | Falls back from whole-region registration to bounded segments, then to a runtime-pinned allocation where segmented registration is unavailable. |
| r27 Compose | Exposes `SHM_SIZE`, `EXTRA_VLLM_ARGS`, and a stable `PYTHONHASHSEED` for reproducible native L1/L2 deployment. |

The image also retains the r24 runtime safety stack: vLLM #229, #218, #216,
#228, and #230 plus SparkInfer #106. SparkInfer #117 is present for the unified
GLM mixed-Trellis path but is not part of DS4 execution.

## Validation Summary

| Gate | Result |
|---|---|
| Clean source composition and immutable archive | Pass |
| Release shell and Python tests | Pass; 12 Python composition tests |
| vLLM #217 focused tests | 36 passed plus lint/format |
| vLLM #235 tokenizer tests | 38 Python + 11 Rust plus lint/format |
| TP2/DCP1 fixed-K5 model load | Pass |
| Official low/high/max prompt contract | Pass |
| Required single tool call | Pass |
| Repeated CC2 output correctness | 100 / 100 pass in 12.014 s |
| Six concurrent approximately 50k prefills | 6 / 6 pass in 23.684 s |
| Final uncached prefill | 65,539 tokens in 5.110 s |
| Native L2 warm replay | 69,888 cached tokens |
| Native L2 replay after container restart | 69,888 cached tokens |
| Controlled raw decode A/B | r27 305.18 vs r15 304.68 tok/s (+0.16%, noise) |
| Post-test health and error scan | Pass |

The decode A/B used the same literal raw prompt and the same remote GPU/NUMA
class. An earlier apparent regression came from comparing different rendered
prompt lengths and probabilistic DSpark acceptance, not from the runtime fix.

## Known Open Items

- Rare batch-wide BOS output under deep concurrent DSpark workloads remains
  open in [rtx6kpro #53](https://github.com/local-inference-lab/rtx6kpro/issues/53).
  The r27 CC2 and CC6 tests passed but do not close that report.
- K7 long-context output/tool quality remains unresolved. Fixed K5 is the
  release default.
- LMCache storage and capacity fixes are included, but DS4 long-context output
  correctness is not qualified. Track
  [rtx6kpro #26](https://github.com/local-inference-lab/rtx6kpro/issues/26).
  Native offload is the qualified host-cache path for r27.
- Persistent L2 was qualified with the POSIX filesystem tier. NIXL and S3 were
  not part of this release gate.
- TP4 Lucifer remains inherited from the r24 validation and should receive a
  fresh r27 sweep before publishing new TP4 performance claims.

## First-Run Compile Cache

The first run compiles SparkInfer, TileLang, and CUDA graph artifacts. Reuse
the same `JIT_CACHE` directory. If a process is interrupted during extension
compilation, PyTorch can leave an empty `lock` file. Remove such a lock only
after confirming that no compiler process holds it.

## Source Provenance

| Component | Ref |
|---|---|
| Canonical GG base | `30038602b71395f481ef4a6edfe4fcf8551d9c15` |
| Composed vLLM tree | `966d57c8c1d9f643eaac8aa231c6e1027936ef2a` |
| SparkInfer base | `272a84bd97ce791a1e92d1f3a0da3dd5f3c6565f` |
| Composed SparkInfer tree | `bbbdccc338a2691d780ed160db54ef121c3a61c9` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| LMCache composed tree | `9a05c8818bae48d15b79c7e876418bb813c08cd0` |
| InstantTensor | `49b4010afc1cae0441e71fe0b0bffc24fa05e932` |
| XGrammar | `0.2.5` |
| PyTorch / CUDA | `2.12.0+cu132` / `13.2.1` |
| cuDNN reported by PyTorch | `9.20.0` |

The reproducible source locks and Compose file are on Docker repository
`main` at commit
[`74563bb`](https://github.com/local-inference-lab/blackwell-llm-docker/commit/74563bb).

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r27 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The archive verifies every base commit, PR head, composed tree, and patch hash
before building.
