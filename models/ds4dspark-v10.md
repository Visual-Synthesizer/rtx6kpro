# DeepSeek-V4-Flash and DSpark v10

This page documents the DS4 standard-checkpoint and DSpark v10 release on the
`dev/fathomless-firmament` line. It is a reproducible successor to
[v9](ds4dspark-v9.md): the image is built from reviewed source refs, the launch
contract lives inside the image, and the benchmark scheduler does not start a
client until every server in the current GPU wave is ready.

The tested checkpoints are:

```text
deepseek-ai/DeepSeek-V4-Flash
deepseek-ai/DeepSeek-V4-Flash-DSpark
```

Standard MTP rows use `method=mtp` with two or three draft tokens. `mtp0`
disables speculative decoding. DSpark uses its dedicated draft module with the
validated fixed K=5 probabilistic path by default.

## Current Unified Image

New deployments can use the same Gilded Gnosis v18 image as GLM-5.2. It
contains the v10 DS4 helper and DSpark stack together with the consolidated GG
runtime fixes. The original v10 benchmark image remains pinned below so its
published measurements stay reproducible.

```text
voipmonitor/vllm:gilded-gnosis-v18-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718
Docker manifest: sha256:1a6c388b76dee43969760ca700ddaf222dc133f5d603a2e32124fcccdfd9c15e
```

Use the v18 Compose file; the launch helper is already inside the image:

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout 7f3cbc6

MODE=mtp2 BACKEND=b12x-a8 TP_SIZE=2 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v18.yml up -d
```

The full image provenance and build command are on the
[GLM-5.2 v18 release page](glm5.2_v18.md#release-image). The v18 Compose file
defaults to InstantTensor `BUFFERED`; the image helper still computes the CUDA
graph cap from mode, draft depth, and `MAX_NUM_SEQS`.

## What Changed From v9

- DSpark draft randomness is independent from acceptance/recovery randomness,
  including deterministic request state during padded CUDA-graph replays.
- Prefix-cache-restored tokens are excluded from draft context and running
  prefill chunks no longer allocate invalid lookahead slots.
- DCP1 verifier causal lengths, B12X auxiliary-stream ordering, TP sampling
  state, stale request slots, and CUDA-graph backbone-output lifetime are fixed.
- Standard non-speculative requests no longer consume the stale CUDA-graph
  padding buffer when building KV slot mappings. The old behavior could assign
  `PAD_SLOT_ID` to live tokens, skip KV writes, and eventually produce garbled
  or CJK output.
- Capacity-aware and variable-length DSpark verification, load-aware physical
  depth, online SPS/STS profiling, block rejection, and a rowwise-FP8 draft head
  are retained as opt-in research paths. They are not release defaults.
- Variable-length capacity is canonicalized from TP rank 0 before any rank
  selects a compact verifier graph. This prevents rank-local capacity estimates
  from entering different shape-sensitive collectives and deadlocking after a
  cold C1-to-C64 graph transition.
- The SM120 PCIe serving work from upstream vLLM PR #47979 is included. Its
  sequence-parallel/async-TP path cannot be used by DSpark yet because this
  revision rejects sequence parallelism under the required V2 runner.
- FlashInfer includes PR #3871 plus the canonical DS4 `topk=256` SM120
  sparse-MLA decode and prefill fixes from PRs #3817 and #3896.
- `/usr/local/bin/serve-ds4-flash.sh` is installed in the image. Compose and
  benchmark wrappers pass environment settings to this helper instead of
  duplicating the complete `vllm serve` command.
- The original v10 release image was shared with GLM-5.2 v16. The current v18
  image keeps the same model-specific helpers behind one `MODEL_FAMILY`
  dispatcher.
- The v10 sweep uses all GPUs `0-15` by default: eight TP2 instances or four
  TP4 instances per wave. `GPU_GROUPS_TP2` and `GPU_GROUPS_TP4` can restrict
  the allocation without changing the synchronized load/benchmark ordering.

## Pull Requests

| Component | Pull request | Purpose |
|---|---|---|
| vLLM | [local-inference-lab/vllm#88](https://github.com/local-inference-lab/vllm/pull/88) | DSpark correctness/capacity work, SM120 PCIe stack, and env launcher |
| B12X | [lukealonso/b12x#28](https://github.com/lukealonso/b12x/pull/28) | CuTe compiler compatibility fallback required by the pinned stack |
| B12X | [lukealonso/b12x#32](https://github.com/lukealonso/b12x/pull/32) | Cooperative resident-grid launch for the dynamic W4A8 kernel |
| FlashInfer | [flashinfer-ai/flashinfer#3871](https://github.com/flashinfer-ai/flashinfer/pull/3871) | Graph-safe uniform multi-token FA2 decode |
| FlashInfer | [flashinfer-ai/flashinfer#3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) | SM120 DSV4 `topk=256` decode instantiation |
| FlashInfer | [flashinfer-ai/flashinfer#3896](https://github.com/flashinfer-ai/flashinfer/pull/3896) | SM120 DSV4 `topk=256` prefill dispatch |
| upstream vLLM | [vllm-project/vllm#47979](https://github.com/vllm-project/vllm/pull/47979) | SM120 PCIe serving stack |

The release PRs created in the local vLLM and B12X forks (#88, #28, and #32) were
opened ready for review, not as drafts. The three pinned upstream FlashInfer
PRs are also non-draft PRs.

## Original v10 Benchmark Image

```text
voipmonitor/vllm:fathomless-firmament-v16-vllm8f86f42-b12xfe06f49-fi801d57a-cu132-20260714
sha256:7a0ed4f956bc2f753fd8c67d32d4ee7358e71922794a471abdb9ae6513cabc54
```

Pinned source stack:

| Component | Ref / commit |
|---|---|
| vLLM | `codex/fathomless-firmament-v16-unified-20260712` @ `8f86f425102cee08745462615d54115eee275f9f` |
| vLLM base | `dev/fathomless-firmament` @ `c649d41bd2d8f1cbb85075d1cf3027eb29cac2ea` when PR #88 was opened |
| B12X | `codex/fathomless-firmament-v16-integration-20260714` @ `fe06f494719267fa3b399878b67caffb915dbdc4` |
| FlashInfer combined source | `codex/sm120-dspark-stack-20260711` @ `801d57a08958c13d375ddbb6be3be4808f48a708` |
| FlashInfer PR heads | #3871 `547ae8e42d9994d930ccd48713a178390f374a82`; #3817 `76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9`; #3896 `1125246e4b2f19f6a77d42d937c8785a1f687445` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL | `2.30.4`, `canonical/cu132-nccl2304-amd-noxml` @ `dfab7c1ace32da250ba97757879429c341b7bcf9` |
| CUDA / PyTorch | CUDA `13.2.1`, PyTorch `2.12.0+cu132` |
| Docker build repo | `local-inference-lab/blackwell-llm-docker main` @ `d104659` |

## Rebuild The Image

The canonical build recipe is
[`build-fathomless-firmament-v16-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/build-fathomless-firmament-v16-cu132.sh).
It pins every source commit, requires both GLM and DS4 helpers, validates the
TP2/TP4 DSpark memory policies and cooperative W4A8 launch, unifies PyTorch and
vLLM on the patched NCCL 2.30.4 runtime, and can push the final tag.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout d104659
PUSH_IMAGE=1 ./build-fathomless-firmament-v16-cu132.sh
```

No runtime source overlay or patch mount is used.

## Start A Server

The helper is already in the image; users do not need to download a launch
script. The maintained minimal Compose example is
[`examples/docker-compose-ds4-v10.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/examples/docker-compose-ds4-v10.yml).

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

MODE=dspark \
BACKEND=lucifer-cutlass \
TP_SIZE=2 \
GPUS=0,1 \
docker compose -f examples/docker-compose-ds4-v10.yml up -d
```

The equivalent minimal service is:

```yaml
services:
  ds4:
    image: voipmonitor/vllm:fathomless-firmament-v16-vllm8f86f42-b12xfe06f49-fi801d57a-cu132-20260714
    entrypoint: ["/usr/local/bin/serve-fathomless-firmament.sh"]
    network_mode: host
    ipc: host
    shm_size: 32gb
    gpus: all
    environment:
      MODEL_FAMILY: ds4
      CUDA_VISIBLE_DEVICES: ${GPUS:-0,1}
      MODE: ${MODE:-dspark}
      BACKEND: ${BACKEND:-lucifer-cutlass}
      TP_SIZE: ${TP_SIZE:-2}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
      LOAD_FORMAT: ${LOAD_FORMAT:-instanttensor}
      INSTANTTENSOR_BACKEND: ${INSTANTTENSOR_BACKEND:-BUFFERED}
```

The helper derives the graph cap automatically: `4 * MAX_NUM_SEQS` for MTP0,
`8 * MAX_NUM_SEQS` for MTP2/MTP3, and `(DSPARK_TOKENS + 1) * MAX_NUM_SEQS`
for DSpark. The default K=5 DSpark configuration therefore uses 6x, with an
absolute minimum graph cap of 6.
All model variants default to InstantTensor with buffered I/O. `BUFFERED`
expands to `URING_BUFFERED,AIO_BUFFERED,MMAP`, allowing hot checkpoint pages to
be reused from the Linux page cache; another loader must be selected explicitly.
Lucifer MTP2/MTP3 uses `GPU_MEMORY_UTILIZATION=0.912` by default; this preserves
the documented 262,144-token limit with 266,246 profiled KV tokens. For DSpark,
Lucifer default uses `0.953` (264,543 profiled KV tokens), Lucifer CUTLASS uses
`0.9465` at TP2 and `0.94` at TP4 or larger, and B12X uses `0.95`. TP4 at
`0.9465` had only 777 MiB free when its first real prefill requested a 764 MiB
FlashInfer MoE workspace, while TP2 at `0.94` had only 7.28 GiB of KV memory
and could not satisfy the 7.89 GiB required for 262,144 tokens. The topology-
aware defaults preserve both constraints without a runtime override. Other
non-DSpark profiles remain at `0.91`.

### Stable Controls

| Environment | Values / default | Meaning |
|---|---|---|
| `MODE` | `mtp0`, `mtp2`, `mtp3`, `dspark` (`dspark`) | Checkpoint/speculative mode |
| `BACKEND` | five profiles below (`b12x-a8`) | Attention, MoE, linear, and force-mode profile |
| `TP_SIZE` | positive integer (`2`) | Tensor parallel size |
| `DCP_SIZE` | positive integer (`1`) | Decode-context parallel size; DSpark currently requires 1 |
| `MAX_NUM_SEQS` | positive integer (`64`) | Scheduler concurrency and automatic graph input |
| `MAX_MODEL_LEN` | tokens (`262144`) | Maximum sequence length |
| `MAX_NUM_BATCHED_TOKENS` | tokens (`8192`) | Scheduler token budget |
| `LOAD_FORMAT` | loader (`instanttensor`) | Model loader; override only for explicit loader comparisons |
| `INSTANTTENSOR_BACKEND` | backend policy (`BUFFERED`) | Buffered InstantTensor policy shared with the GLM helper |
| `ALLREDUCE_MODE` | `b12x`, `vllm-custom`, `vllm-custom-2stage`, `nccl` (`b12x`) | TP collective implementation |
| `B12X_PCIE_DMA` | `0`, `1` (`0`) | Opt-in large-tensor B12X DMA; decode keeps the B12X oneshot path either way |
| `INDEXER_BACKEND` | `auto`, `b12x`, `native` (`auto`) | Sparse indexer; auto follows the backend profile |
| `CUDAGRAPH_CAPTURE_SIZES` | `default`, `auto`, `none`, or a list (`default`) | Optional explicit graph-capture pattern |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | positive integer (`auto`) | Override the derived graph cap |

### Backend Profiles

| Backend | Attention | MoE / linear and activation force |
|---|---|---|
| `b12x-a16` | `B12X_MLA_SPARSE` | B12X MoE + B12X linear, force W4A16 |
| `b12x-a8` | `B12X_MLA_SPARSE` | B12X MoE + B12X linear, force W4A8 MX |
| `b12x-a8-dglin` | `B12X_MLA_SPARSE` | B12X MoE W4A8 MX + DeepGEMM linear |
| `lucifer-default` | `FLASHINFER_MLA_SPARSE_DSV4` | default model MoE/linear selection |
| `lucifer-cutlass` | `FLASHINFER_MLA_SPARSE_DSV4` | FlashInfer CUTLASS MoE |

The helper also enables the serving defaults used by v9/v10:

```text
--enable-flashinfer-autotune
--enable-prompt-tokens-details
--enable-force-include-usage
--enable-request-id-headers
VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1
```

### Experimental DSpark Controls

These controls are preserved for future work but are disabled by default:

| Environment | Default | Purpose |
|---|---:|---|
| `DSPARK_CAPACITY` | `0` | Enable capacity-aware logical draft lengths |
| `DSPARK_CAPACITY_VERIFICATION_MODE` | auto | `varlen` with B12X indexer, otherwise masked padded verification |
| `DSPARK_DYNAMIC_DRAFT_DEPTH` | `0` | Load-aware physical draft-depth controller |
| `DSPARK_FP8_DRAFT_HEAD` | `0` | Rowwise-FP8 DSpark draft LM head |
| `DSPARK_DRAFT_ATTENTION_BACKEND` | `auto` | Explicit draft-only attention backend experiment |
| `REJECTION_SAMPLE_METHOD` | `standard` | Optional `block` rejection experiment |
| `DSPARK_CONFIDENCE_THRESHOLD` | `0.0` | Capacity confidence cutoff |
| `DSPARK_BUDGET_FRAC` | `1.0` | Capacity budget fraction |
| `DSPARK_ONLINE_STS` | `1` when capacity is on | Online service-time profiling |
| `DSPARK_SPS_CURVE` | `auto` | Measured service-time curve or explicit list |

The masked CUTLASS capacity path lowers logical K but still executes padded
target compute. In the measured C1-C128 range, dynamic K therefore did not beat
fixed K=5. The FP8 draft head was approximately neutral at C1 and only about
1.2% faster at C64 on RTX 6000 Pro. These are useful implementation points,
not reasons to change the release default.

`SP_ASYNC_TP=1` is available only for compatible non-DSpark V1 runs. The helper
fails explicitly if it is requested with DSpark instead of pretending that SP
is active.

## Unified v16 Image Canaries

Only two DS4 canaries were run after unifying the GLM and DS4 image; the
published v10 sweep below was not repeated. Both used TP2, InstantTensor
`BUFFERED`, B12X all-reduce, max model length 262,144, and the same helper and
backend defaults documented above.

| Checkpoint / mode | Metric | v10 reference | Unified v16 | Delta |
|---|---|---:|---:|---:|
| Standard, B12X A16, MTP0 | Decode C1 | 143.5 | 143.4 | -0.1% |
| Standard, B12X A16, MTP0 | Prefill 64k | 11,524 | 11,635 | +1.0% |
| DSpark, Lucifer CUTLASS, fixed K5 | Decode C1 | 238.9 | 239.95 | +0.4% |
| DSpark, Lucifer CUTLASS, fixed K5 | Prefill 64k | 6,642 | 6,892 | +3.8% |

The DSpark row was run on the exact final image with no memory override. It
profiled 7.89 GiB of KV memory and 262,144 GPU KV tokens. The first 64k request
compiled `_prepare_dflash_inputs_kernel` and measured 6,489 tok/s; the clean
post-JIT repetition shown in the table measured 6,892 tok/s. The server log
contained no CUDA, NCCL, or Xid failure.

The standard row used the immediately preceding image. The final vLLM commit
changes only the `lucifer-cutlass + dspark` TP-aware memory branch, so the
standard B12X/MTP0 command and all code it reaches are identical.

Canary artifacts:

```text
/root/bench-results/ds4-v16-canary-b12xfe06-20260714/standard-tp2-b12x-a16-mtp0
/root/bench-results/ds4-v16-canary-b12xfe06-20260714/dspark-tp2-lucifer-cutlass
```

## Clean-Cache Release Validation

Warm kernel caches hid the original TP failure, so the release image was
validated from an empty compile/autotune cache. The tested TP2 profile used
Lucifer CUTLASS attention/MoE, the B12X indexer and all-reduce, varlen capacity,
capacity activation at 32 requests, dynamic depth with an eight-step window,
online STS, and the FP8 draft head. The image was allowed to finish model load,
autotuning, target FULL capture, and all 216 DSpark compact FULL captures before
the client started.

| Phase | Aggregate output tok/s | Result |
|---|---:|---|
| Cold C1 | 262.3 | passed |
| C64 after the cold graph transition | 2718.3 | passed |
| Recovery C1 after C64 | 239.5 | passed |
| C64, continuous 120-second soak | 2763.4 | passed |

After the soak, the server reported zero running and waiting requests,
`431289` generated tokens, and no error or traceback. The focused capacity test
module also passed all 22 tests inside the immutable image.

The root cause was not an NCCL bandwidth limit. Before overlap is accounted
for, NCCL kernels occupied about 42.8 ms of a 267.6 ms C64 GPU trace (roughly
16%). A standalone same-pair test measured 56.4 GB/s CUDA peer copy, 39.47 GB/s
NCCL, and 48.67 GB/s B12X DMA. Enabling large-message B12X DMA improved the
three-run C128 mean from 3620.9 to 3670.2 tok/s (+1.36%). This remains a useful
secondary optimization, but it neither explains nor fixes the TP graph-shape
deadlock, so `B12X_PCIE_DMA=0` remains the release default.

## Full Synchronized Sweep

The public scripts are:

- [`run-ds4-v10-server.sh`](../scripts/run-ds4-v10-server.sh): Docker placement,
  cache mounts, and CPU/NUMA pinning; all vLLM configuration is delegated to the
  image helper.
- [`run-ds4-v10-sweep.sh`](../scripts/run-ds4-v10-sweep.sh): v10 allocation and
  synchronized wave entry point.
- [`run-ds4-v9-sweep.sh`](../scripts/run-ds4-v9-sweep.sh): shared wave scheduler,
  result validation, resume support, and reproducibility capture.
- [`render-ds4-v9-results.py`](../scripts/render-ds4-v9-results.py): full tables
  and optional v9 percentage deltas.

Every wave follows this order:

1. Start every server assigned to the current GPU allocation.
2. Wait for `/v1/models` from every server.
3. Wait 30 seconds after the final server becomes ready.
4. Run an unreported warmup over C1/C16/C32/C64 and 8k/64k/128k prefill.
5. Wait another 30 seconds, then record the server-log measurement boundary.
6. Start benchmark clients only after the entire wave is ready and warmed.
7. Validate every JSON cell and reject any JIT cache miss after the boundary.

The fixed post-ready settle is required. Starting C1 seven seconds after the
long CUDA graph capture reproducibly measured about `133 tok/s`; waiting 30
seconds restored `141 tok/s` on the same image, server configuration, GPU pair,
and request. This was a post-capture boost/settling artifact, not a vLLM or B12X
regression. Loading another model during measurement remains prohibited as an
independent source of interference.

The unreported runtime warmup is also required because some Triton and CuTeDSL
shapes can compile after the API reports ready. The sweep stores
`warmup-decode.json`, `warmup-prefill.json`, `warmup-server.log`, and
`runtime-log-start-line.txt` for every case. A result is not reusable unless its
recorded phase contains no `JIT compilation during inference` or post-engine
disk-cache miss.

```bash
cd /root/rtx6kpro

OUT=/root/bench-results/ds4-v10-final-bbcc06f-sweep-20260712 \
SHARED_CACHE=/root/.cache/vllm-ds4-v10-final-bbcc06f-sweep \
TPS=2,4 \
BACKENDS=b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass \
MODES=standard-mtp0,standard-mtp2,standard-mtp3,dspark \
MAX_NUM_SEQS=64 \
DECODE_CONCURRENCY=1,16,32,64 \
DECODE_CONTEXTS=0 \
DECODE_DURATION=30 \
PREFILL_CONTEXTS=8k,64k,128k \
PREFILL_DURATION=10 \
STARTUP_TIMEOUT=3600 \
ENABLE_TOPO_PIN=1 \
POST_READY_SETTLE_SECONDS=30 \
RUNTIME_WARMUP=1 \
POST_WARMUP_SETTLE_SECONDS=30 \
scripts/run-ds4-v10-sweep.sh
```

Render the completed run and compare it with v9:

```bash
scripts/render-ds4-v9-results.py \
  /root/bench-results/ds4-v10-final-bbcc06f-sweep-20260712 \
  --baseline /root/bench-results/ds4-v9-refresh-pc1441b5-syncwave-20260704-102844
```

The 40-case numerical sweep was collected on the `bbcc06f` release candidate.
The original v10 `adf15ca` commit changes only the embedded helper's default
`GPU_MEMORY_UTILIZATION` for `lucifer-cutlass + dspark`; no packaged Python,
CUDA, B12X, FlashInfer, or DeepGEMM implementation changed. That one TP4 row was
rerun at `GPU_MEMORY_UTILIZATION=0.94`, exactly matching the final helper. The
original v10 image was also started without an override and revalidated through the
same C1/C64 and 128k-prefill path. That exact-image validation profiled
1,523,880 KV tokens and measured C1/C16/C32/C64 at
`298.5/1702.6/2499.6/3454.8 tok/s`, coding median `416.7 tok/s`, and completed
8k/64k/128k prefill without OOM.

## Decode Throughput

Sustained decode is aggregate output tok/s from `llm_decode_bench.py`, context
0, 30 seconds per cell. Coding peak is the median generation-only tok/s from
five Sieve-of-Eratosthenes runs.

### DSpark Checkpoint

| TP | Backend | C1 | C16 | C32 | C64 | Coding median |
|---:|---|---:|---:|---:|---:|---:|
| 2 | `b12x-a16` | 236.2 | 1062.5 | 1567.5 | 2161.5 | 315.0 |
| 2 | `b12x-a8` | 219.0 | 1041.2 | 1584.8 | 2210.0 | 294.7 |
| 2 | `b12x-a8-dglin` | 227.2 | 1055.7 | 1609.3 | 2225.3 | 292.9 |
| 2 | `lucifer-default` | 220.6 | 1065.4 | 1614.9 | 2403.1 | 296.8 |
| 2 | `lucifer-cutlass` | 238.9 | 1190.1 | 1788.5 | 2606.9 | 324.8 |
| 4 | `b12x-a16` | 309.4 | 1532.6 | 2178.0 | 2802.1 | 404.9 |
| 4 | `b12x-a8` | 288.3 | 1469.9 | 2226.6 | 2942.2 | 391.1 |
| 4 | `b12x-a8-dglin` | 283.5 | 1475.9 | 2227.2 | 2932.3 | 376.1 |
| 4 | `lucifer-default` | 309.6 | 1596.2 | 2358.1 | 3306.5 | 389.3 |
| 4 | `lucifer-cutlass` | 299.7 | 1770.7 | 2555.4 | 3524.9 | 404.4 |

All coding probes completed five of five runs with zero CJK/garbled runs.

### Standard Checkpoint

| TP | Backend | Mode | C1 | C16 | C32 | C64 | Coding median |
|---:|---|---|---:|---:|---:|---:|---:|
| 2 | `b12x-a16` | MTP0 | 143.5 | 848.4 | 1260.7 | 1887.0 | 143.1 |
| 2 | `b12x-a16` | MTP2 | 232.7 | 1162.3 | 1664.4 | 2518.6 | 245.5 |
| 2 | `b12x-a16` | MTP3 | 218.4 | 1056.3 | 1568.4 | 2306.9 | 237.1 |
| 2 | `b12x-a8` | MTP0 | 141.4 | 767.0 | 1184.7 | 1828.8 | 141.3 |
| 2 | `b12x-a8` | MTP2 | 219.1 | 1081.4 | 1631.5 | 2545.0 | 236.7 |
| 2 | `b12x-a8` | MTP3 | 224.8 | 1024.9 | 1558.6 | 2401.2 | 238.2 |
| 2 | `b12x-a8-dglin` | MTP0 | 141.6 | 771.8 | 1188.2 | 1833.5 | 141.3 |
| 2 | `b12x-a8-dglin` | MTP2 | 222.1 | 1081.1 | 1639.7 | 2595.0 | 245.3 |
| 2 | `b12x-a8-dglin` | MTP3 | 221.1 | 1010.5 | 1560.7 | 2406.4 | 237.4 |
| 2 | `lucifer-default` | MTP0 | 128.0 | 791.8 | 1182.2 | 1793.1 | 129.3 |
| 2 | `lucifer-default` | MTP2 | 213.0 | 1071.5 | 1693.4 | 2581.8 | 218.4 |
| 2 | `lucifer-default` | MTP3 | 199.2 | 998.3 | 1600.0 | 2431.0 | 221.9 |
| 2 | `lucifer-cutlass` | MTP0 | 129.2 | 866.5 | 1275.2 | 1980.7 | 130.3 |
| 2 | `lucifer-cutlass` | MTP2 | 222.1 | 1200.4 | 1859.7 | 2838.7 | 237.8 |
| 2 | `lucifer-cutlass` | MTP3 | 220.9 | 1119.3 | 1742.8 | 2686.3 | 237.7 |
| 4 | `b12x-a16` | MTP0 | 176.0 | 1211.8 | 1840.8 | 2689.8 | 175.6 |
| 4 | `b12x-a16` | MTP2 | 299.2 | 1706.6 | 2431.9 | 3576.8 | 325.7 |
| 4 | `b12x-a16` | MTP3 | 280.9 | 1541.5 | 2274.2 | 3176.4 | 308.1 |
| 4 | `b12x-a8` | MTP0 | 175.7 | 1083.9 | 1664.1 | 2541.7 | 175.2 |
| 4 | `b12x-a8` | MTP2 | 297.2 | 1591.4 | 2366.3 | 3628.6 | 313.7 |
| 4 | `b12x-a8` | MTP3 | 276.2 | 1441.4 | 2215.3 | 3270.2 | 304.0 |
| 4 | `b12x-a8-dglin` | MTP0 | 178.2 | 1093.4 | 1687.9 | 2531.0 | 177.9 |
| 4 | `b12x-a8-dglin` | MTP2 | 295.9 | 1602.1 | 2392.5 | 3625.0 | 326.1 |
| 4 | `b12x-a8-dglin` | MTP3 | 279.3 | 1470.4 | 2207.2 | 3271.2 | 307.8 |
| 4 | `lucifer-default` | MTP0 | 160.2 | 1148.6 | 1722.9 | 2637.3 | 162.0 |
| 4 | `lucifer-default` | MTP2 | 287.8 | 1635.2 | 2515.6 | 3847.8 | 304.3 |
| 4 | `lucifer-default` | MTP3 | 277.7 | 1540.2 | 2364.4 | 3519.5 | 305.4 |
| 4 | `lucifer-cutlass` | MTP0 | 154.9 | 1237.2 | 1910.3 | 2903.2 | 156.4 |
| 4 | `lucifer-cutlass` | MTP2 | 290.8 | 1818.8 | 2809.4 | 4166.6 | 310.1 |
| 4 | `lucifer-cutlass` | MTP3 | 281.7 | 1689.0 | 2608.9 | 3810.5 | 302.4 |

## Prefill Throughput

Standalone prefill is client prompt tokens divided by TTFT, with non-repeating
prompts and 10 seconds per context.

### DSpark Checkpoint

| TP | Backend | 8k | 64k | 128k |
|---:|---|---:|---:|---:|
| 2 | `b12x-a16` | 11192 | 11327 | 10598 |
| 2 | `b12x-a8` | 12834 | 12933 | 12036 |
| 2 | `b12x-a8-dglin` | 12140 | 12967 | 12028 |
| 2 | `lucifer-default` | 12662 | 7929 | 5460 |
| 2 | `lucifer-cutlass` | 13070 | 6642 | 5599 |
| 4 | `b12x-a16` | 13613 | 13443 | 12650 |
| 4 | `b12x-a8` | 14906 | 14689 | 13639 |
| 4 | `b12x-a8-dglin` | 13636 | 14818 | 13711 |
| 4 | `lucifer-default` | 14861 | 5608 | 5149 |
| 4 | `lucifer-cutlass` | 15320 | 13553 | 6009 |

Lucifer DSpark remains unsuitable when sustained 64k-128k prefill is the primary
workload. B12X DSpark keeps long-prefill throughput close to the standard model.

### Standard Checkpoint

| TP | Backend | Mode | 8k | 64k | 128k |
|---:|---|---|---:|---:|---:|
| 2 | `b12x-a16` | MTP0 | 11906 | 11524 | 10758 |
| 2 | `b12x-a16` | MTP2 | 11592 | 11245 | 10482 |
| 2 | `b12x-a16` | MTP3 | 11675 | 11330 | 10542 |
| 2 | `b12x-a8` | MTP0 | 13171 | 13153 | 12170 |
| 2 | `b12x-a8` | MTP2 | 12921 | 12577 | 11663 |
| 2 | `b12x-a8` | MTP3 | 13052 | 12652 | 11704 |
| 2 | `b12x-a8-dglin` | MTP0 | 13649 | 13139 | 12100 |
| 2 | `b12x-a8-dglin` | MTP2 | 13361 | 12900 | 11885 |
| 2 | `b12x-a8-dglin` | MTP3 | 13399 | 12902 | 11927 |
| 2 | `lucifer-default` | MTP0 | 13026 | 12817 | 11744 |
| 2 | `lucifer-default` | MTP2 | 12793 | 12277 | 11271 |
| 2 | `lucifer-default` | MTP3 | 12896 | 12371 | 11353 |
| 2 | `lucifer-cutlass` | MTP0 | 13304 | 13034 | 11966 |
| 2 | `lucifer-cutlass` | MTP2 | 13469 | 12864 | 11759 |
| 2 | `lucifer-cutlass` | MTP3 | 13424 | 12854 | 11763 |
| 4 | `b12x-a16` | MTP0 | 14371 | 13935 | 12913 |
| 4 | `b12x-a16` | MTP2 | 14036 | 13583 | 12586 |
| 4 | `b12x-a16` | MTP3 | 13784 | 13355 | 12353 |
| 4 | `b12x-a8` | MTP0 | 15158 | 15139 | 13920 |
| 4 | `b12x-a8` | MTP2 | 15281 | 14748 | 13552 |
| 4 | `b12x-a8` | MTP3 | 14937 | 14457 | 13333 |
| 4 | `b12x-a8-dglin` | MTP0 | 15858 | 15176 | 13953 |
| 4 | `b12x-a8-dglin` | MTP2 | 15361 | 14815 | 13614 |
| 4 | `b12x-a8-dglin` | MTP3 | 15111 | 14540 | 13382 |
| 4 | `lucifer-default` | MTP0 | 14974 | 14868 | 13598 |
| 4 | `lucifer-default` | MTP2 | 15214 | 14581 | 13319 |
| 4 | `lucifer-default` | MTP3 | 14881 | 14306 | 13048 |
| 4 | `lucifer-cutlass` | MTP0 | 15417 | 15191 | 13848 |
| 4 | `lucifer-cutlass` | MTP2 | 15564 | 14871 | 13523 |
| 4 | `lucifer-cutlass` | MTP3 | 15217 | 14590 | 13291 |

## v9 Comparison

The compact comparison reports percentage change for the latency-sensitive
decode endpoint (`cc1`), the tested high-concurrency endpoint (`cc64`), and the
representative long prefill cell (`64k`).

### DSpark Checkpoint

| TP | Backend | C1 | C64 | 64k prefill |
|---:|---|---:|---:|---:|
| 2 | `b12x-a16` | +7.8% | +4.7% | +2.0% |
| 2 | `b12x-a8` | +1.0% | +2.8% | +1.3% |
| 2 | `b12x-a8-dglin` | +17.3% | +3.8% | +2.4% |
| 2 | `lucifer-default` | +9.5% | +4.7% | -37.0% |
| 2 | `lucifer-cutlass` | -0.3% | +4.3% | -45.5% |
| 4 | `b12x-a16` | +5.8% | +7.1% | +0.8% |
| 4 | `b12x-a8` | +7.4% | +24.3% | +0.7% |
| 4 | `b12x-a8-dglin` | +12.7% | +177.0% | +0.8% |
| 4 | `lucifer-default` | +12.7% | +5.7% | -61.5% |
| 4 | `lucifer-cutlass` | +7.8% | +6.0% | -5.3% |

### Standard Checkpoint

| TP | Backend | Mode | C1 | C64 | 64k prefill |
|---:|---|---|---:|---:|---:|
| 2 | `b12x-a16` | MTP0 | +0.6% | +0.1% | +0.2% |
| 2 | `b12x-a16` | MTP2 | +2.5% | +0.4% | +0.3% |
| 2 | `b12x-a16` | MTP3 | +2.5% | +0.1% | +0.3% |
| 2 | `b12x-a8` | MTP0 | +0.5% | -0.5% | +0.1% |
| 2 | `b12x-a8` | MTP2 | +0.8% | -0.9% | +0.0% |
| 2 | `b12x-a8` | MTP3 | +13.7% | +0.8% | -0.0% |
| 2 | `b12x-a8-dglin` | MTP0 | +0.6% | +0.7% | +0.3% |
| 2 | `b12x-a8-dglin` | MTP2 | -2.4% | +0.1% | +0.7% |
| 2 | `b12x-a8-dglin` | MTP3 | +4.1% | -0.3% | +0.3% |
| 2 | `lucifer-default` | MTP0 | +0.8% | +0.1% | +0.5% |
| 2 | `lucifer-default` | MTP2 | +4.2% | +0.3% | +0.3% |
| 2 | `lucifer-default` | MTP3 | +2.0% | +0.5% | +2.3% |
| 2 | `lucifer-cutlass` | MTP0 | +1.1% | +0.5% | +4.8% |
| 2 | `lucifer-cutlass` | MTP2 | +1.7% | -0.1% | +4.6% |
| 2 | `lucifer-cutlass` | MTP3 | +4.9% | +1.0% | +5.0% |
| 4 | `b12x-a16` | MTP0 | +0.8% | -0.4% | +0.3% |
| 4 | `b12x-a16` | MTP2 | -1.0% | +0.1% | +1.1% |
| 4 | `b12x-a16` | MTP3 | +10.4% | +2.3% | +1.0% |
| 4 | `b12x-a8` | MTP0 | +0.8% | +0.5% | +0.4% |
| 4 | `b12x-a8` | MTP2 | +2.5% | +1.0% | +0.6% |
| 4 | `b12x-a8` | MTP3 | +10.6% | +0.3% | +0.3% |
| 4 | `b12x-a8-dglin` | MTP0 | +0.7% | -1.0% | +0.3% |
| 4 | `b12x-a8-dglin` | MTP2 | -0.4% | +0.1% | +0.7% |
| 4 | `b12x-a8-dglin` | MTP3 | +9.5% | +0.9% | +0.7% |
| 4 | `lucifer-default` | MTP0 | +1.1% | +0.1% | +0.3% |
| 4 | `lucifer-default` | MTP2 | +8.2% | +1.5% | +1.1% |
| 4 | `lucifer-default` | MTP3 | +4.3% | -0.0% | +0.3% |
| 4 | `lucifer-cutlass` | MTP0 | +1.4% | +0.1% | +4.2% |
| 4 | `lucifer-cutlass` | MTP2 | +4.0% | +0.5% | +4.8% |
| 4 | `lucifer-cutlass` | MTP3 | +7.6% | +0.9% | +4.3% |

MTP0 is effectively unchanged or faster than v9 in every latency-sensitive C1
cell. The largest negative standard-checkpoint delta is `-2.4%` on TP2
`b12x-a8-dglin` MTP2 C1; that is not an MTP0 regression.

## Development Findings

- The validated default is probabilistic fixed K=5.
- Greedy draft sampling, separate draft Q/K/V, FP32 draft head, fake-FP8 main
  projection input, and reference draft attention did not improve the result.
- Dynamic/load-aware K is functionally implemented, but the masked CUTLASS
  verifier does not skip target compute. A true compact target kernel is the
  remaining opportunity for load-aware speedups.
- The old draft-pass compaction experiment in local PR #71 targeted an earlier
  architecture and is superseded. The FP8-head work from local PR #73 is
  retained in PR #88 as an opt-in path.
- B12X indexer plus Lucifer attention can enable true variable-length metadata,
  but the hybrid was not a general default performance win.
- Alternative vLLM custom, two-stage, symmetric-memory, and NCCL collectives are
  selectable. B12X PCIe all-reduce remains the validated release default.
- DSpark still requires the V2 model runner, so PR #47979 SP/async-TP cannot
  accelerate DSpark until V2 sequence parallelism exists.

## Artifacts

```text
/root/bench-results/ds4-v10-final-bbcc06f-sweep-20260712
/root/bench-results/ds4-v10-final-bbcc06f-sweep-20260712/repro/
/root/bench-results/ds4-v10-final-bbcc06f-sweep-20260712/progress.log
/root/bench-results/ds4-v10-final-adf15ca-validation-20260712
```

The `repro/` directory contains the exact launcher and sweep scripts, SHA256
hashes, image labels, image inspection JSON, repository state, benchmark hash,
GPU inventory, and NVIDIA topology captured before the run.
