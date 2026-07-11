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
- The SM120 PCIe serving work from upstream vLLM PR #47979 is included. Its
  sequence-parallel/async-TP path cannot be used by DSpark yet because this
  revision rejects sequence parallelism under the required V2 runner.
- FlashInfer includes PR #3871 plus the canonical DS4 `topk=256` SM120
  sparse-MLA decode and prefill fixes from PRs #3817 and #3896.
- `/usr/local/bin/serve-ds4-flash.sh` is installed in the image. Compose and
  benchmark wrappers pass environment settings to this helper instead of
  duplicating the complete `vllm serve` command.
- The v10 sweep is hard-limited to GPUs `0-7`. GPUs `8-15` are not present in
  its scheduler allocation.

## Pull Requests

| Component | Pull request | Purpose |
|---|---|---|
| vLLM | [local-inference-lab/vllm#88](https://github.com/local-inference-lab/vllm/pull/88) | DSpark correctness/capacity work, SM120 PCIe stack, and env launcher |
| B12X | [lukealonso/b12x#28](https://github.com/lukealonso/b12x/pull/28) | CuTe compiler compatibility fallback required by the pinned stack |
| FlashInfer | [flashinfer-ai/flashinfer#3871](https://github.com/flashinfer-ai/flashinfer/pull/3871) | Graph-safe uniform multi-token FA2 decode |
| FlashInfer | [flashinfer-ai/flashinfer#3817](https://github.com/flashinfer-ai/flashinfer/pull/3817) | SM120 DSV4 `topk=256` decode instantiation |
| FlashInfer | [flashinfer-ai/flashinfer#3896](https://github.com/flashinfer-ai/flashinfer/pull/3896) | SM120 DSV4 `topk=256` prefill dispatch |
| upstream vLLM | [vllm-project/vllm#47979](https://github.com/vllm-project/vllm/pull/47979) | SM120 PCIe serving stack |

The release PRs created in the local vLLM and B12X forks (#88 and #28) were
opened ready for review, not as drafts. The three pinned upstream FlashInfer
PRs are also non-draft PRs.

## Docker Image

```text
voipmonitor/vllm:fathomless-firmament-ds4-v10-vllm2a62b49-b12x90172a5-fi2cba2f7-cu132-20260711
sha256:55ac0a6bcebb11dafe8d1d1a0964d41c88f3768d9edcc0eb70e741073d0ba51b
```

Pinned source stack:

| Component | Ref / commit |
|---|---|
| vLLM | `codex/fathomless-firmament-dspark-pr47979-combined-20260710` @ `2a62b4909c081013feb4fe1bfd8c7980802b88b3` |
| vLLM base | `dev/fathomless-firmament` @ `c649d41bd2d8f1cbb85075d1cf3027eb29cac2ea` when PR #88 was opened |
| B12X | `codex/ff-v15-cute-compile-fallback-20260709` @ `90172a504e96d246e07cb1ebad3b291532445560` |
| FlashInfer combined source | `codex/sm120-dspark-stack-20260711` @ `2cba2f7bbe8335fcabe18d29e6eb99de2093f991` |
| FlashInfer PR heads | #3871 `547ae8e42d9994d930ccd48713a178390f374a82`; #3817 `76fd3daf7064b73924ebb3bcb1e93a8a26fc6da9`; #3896 `1125246e4b2f19f6a77d42d937c8785a1f687445` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| CUTLASS | `d80a4e53b52b42550659a8696dab32705265e324` |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| NCCL | `2.30.4`, `canonical/cu132-nccl2304-amd-noxml` @ `dfab7c1ace32da250ba97757879429c341b7bcf9` |
| CUDA / PyTorch | CUDA `13.2.1`, PyTorch `2.12.0+cu132` |

## Rebuild The Image

The canonical build recipe is
[`build-fathomless-firmament-ds4-v10-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/main/build-fathomless-firmament-ds4-v10-cu132.sh).
It pins every source commit, requires `serve-ds4-flash.sh` to be present, checks
the helper in `DRY_RUN` mode, unifies PyTorch and vLLM on the patched NCCL
2.30.4 runtime, and can push the final tag.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout c149645

PUSH_IMAGE=1 ./build-fathomless-firmament-ds4-v10-cu132.sh
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
    image: voipmonitor/vllm:fathomless-firmament-ds4-v10-vllm2a62b49-b12x90172a5-fi2cba2f7-cu132-20260711
    command: ["/usr/local/bin/serve-ds4-flash.sh"]
    network_mode: host
    ipc: host
    shm_size: 32gb
    gpus: all
    environment:
      CUDA_VISIBLE_DEVICES: ${GPUS:-0,1}
      MODE: ${MODE:-dspark}
      BACKEND: ${BACKEND:-lucifer-cutlass}
      TP_SIZE: ${TP_SIZE:-2}
      MAX_NUM_SEQS: ${MAX_NUM_SEQS:-64}
```

The helper derives the graph cap automatically: `4 * MAX_NUM_SEQS` for MTP0
and `8 * MAX_NUM_SEQS` for MTP2, MTP3, and DSpark, with a minimum of 6.

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

1. Start every server assigned to GPUs `0-7`.
2. Wait for `/v1/models` from every server.
3. Start benchmark clients only after the entire wave is ready.
4. Validate JSON cells before a case can be reused by `RESUME=1`.

This ordering prevents the false `132-133 tok/s` result caused by benchmarking
one instance while another model was still loading.

```bash
cd /root/rtx6kpro

OUT=/root/bench-results/ds4-v10-full-20260711 \
TPS=2,4 \
BACKENDS=b12x-a16,b12x-a8,b12x-a8-dglin,lucifer-default,lucifer-cutlass \
MODES=standard-mtp0,standard-mtp2,standard-mtp3,dspark \
MAX_NUM_SEQS=64 \
DECODE_CONCURRENCY=1,16,32,64 \
DECODE_CONTEXTS=0 \
DECODE_DURATION=30 \
PREFILL_CONTEXTS=8k,64k,128k \
PREFILL_DURATION=10 \
STARTUP_TIMEOUT=2400 \
ENABLE_TOPO_PIN=1 \
scripts/run-ds4-v10-sweep.sh
```

Render the completed run and compare it with v9:

```bash
scripts/render-ds4-v9-results.py \
  /root/bench-results/ds4-v10-full-20260711 \
  --baseline /root/bench-results/ds4-v9-refresh-pc1441b5-syncwave-20260704-102844
```

## Decode Throughput

Sustained decode is aggregate output tok/s from `llm_decode_bench.py`, context
0, 30 seconds per cell. Coding peak is the median generation-only tok/s from
five Sieve-of-Eratosthenes runs.

TBD_DECODE_TABLES

## Prefill Throughput

Standalone prefill is client prompt tokens divided by TTFT, with non-repeating
prompts and 10 seconds per context.

TBD_PREFILL_TABLES

## v9 Comparison

The compact comparison reports percentage change for the latency-sensitive
decode endpoint (`cc1`), the tested high-concurrency endpoint (`cc64`), and the
representative long prefill cell (`64k`).

TBD_V9_COMPARISON

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
TBD_RESULT_ROOT
TBD_RESULT_ROOT/repro/
TBD_RESULT_ROOT/progress.log
```

The `repro/` directory contains the exact launcher and sweep scripts, SHA256
hashes, image labels, image inspection JSON, repository state, benchmark hash,
GPU inventory, and NVIDIA topology captured before the run.
