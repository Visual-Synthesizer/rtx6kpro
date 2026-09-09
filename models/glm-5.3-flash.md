# GLM-5.3-Flash

<p align="center">
  <img src="../images/glm-5.3-flash-jovian-judgement-branch-logo.png"
       width="520" alt="Gold Jovian Judgement emblem with an eye, scales, and a star">
</p>
<p align="center"><em>Jovian Judgement branch logo, published by Luke for
Local Inference Lab.</em></p>

This page specifies the qualified GLM-5.3-Flash deployment for four NVIDIA RTX
PRO 6000 Blackwell Workstation Edition GPUs. The runtime serves the
`local-inference-lab/GLM-5.3-Flash-NVFP4` target checkpoint without
speculation, with three-token Multi-Token Prediction (MTP), or with the
`local-inference-lab/GLM-5.3-Flash-DFlash2` draft checkpoint.

The commands use Hugging Face repository names and named Docker volumes. They
do not require checkpoint paths or source-code bind mounts.

## Status

| Capability | Status |
|---|---|
| Tensor parallelism of four with one decode-context rank | **qualified** for no speculation, MTP depth 3, and DFlash2 depth 7 |
| Tensor parallelism of four with four decode-context ranks | **qualified** for the same three serving modes, including complete-KV prefill |
| Two decode-context ranks | **implemented**; not independently performance-qualified for this artifact |
| Tensor parallelism of eight | **implemented**; not independently hardware-qualified for this artifact |
| Target checkpoint | `local-inference-lab/GLM-5.3-Flash-NVFP4`; Hugging Face `main` unless `MODEL_REVISION` is set |
| QAD step-1,750 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step1750`](../kld/glm-5.3-flash-qad-step1750.md); distribution fidelity, verifier-backed behavior, and AA-LCR are measured, but the checkpoint is not a qualified serving target |
| QAD step-2,500 research checkpoint | [`GLM-5.3-Flash-NVFP4-QAD-step2500`](../kld/glm-5.3-flash-qad-step2500.md); distribution fidelity is measured, and the [9,856-task VBF report](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md) qualifies practical equivalence on its primary semantic score; production serving remains unqualified |
| AA-LCR capability evaluation | **qualified** for the exact BF16, published-NVFP4, and QAD checkpoint-and-runtime configurations in the [three-configuration report](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) |
| Verifier-backed behavioral fidelity | **qualified** practical equivalence for QAD step 2,500 versus published NVFP4 on the primary 9,856-task semantic score; **qualified execution with inconclusive one-point decisions** in the [TP8 BF16/published-NVFP4/QAD-step-1,750 report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md) |
| DFlash2 checkpoint | `local-inference-lab/GLM-5.3-Flash-DFlash2`; Hugging Face `main` unless `DFLASH_MODEL_REVISION` is set |
| Target routed experts | ModelOpt NVFP4 using B12X 4-bit weights and 4-bit activations |
| DFlash2 weights | Offline-serialized ModelOpt MXFP8; no online weight quantization |
| Target KV cache | **qualified** FP8; packed NVFP4 is implemented but not qualified for R29 |
| MTP proposal vocabulary head | NVFP4 draft-only copy by default; the target verifier vocabulary head remains BF16 |
| GPU prefix cache | **qualified** request/SYSTEM boundaries in all six TP4 mode/DCP combinations; fine aligned retention is selectable |
| Native DRAM offload | **implemented** and opt-in with `CACHE_MODE=native`; not independently requalified for R29 |
| LMCache DRAM and filesystem tiers | **qualified** and opt-in with `CACHE_MODE=lmcache`; asynchronous engine-driven pinned shared memory is the default transfer path |
| CUDA graphs | **qualified** with launcher default `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` for target and speculative decode |
| Scheduler | 4,096 target tokens per step; fixed prefill compute share 0.4; interval 1; one prefill lane by default, optional bounded interleaving |
| Root filesystem | Two layers: flattened runtime foundation and committed source installation |
| FlashKDA numerical stability | **qualified** with the stable FP32 forward-substitution inverse |
| Qwen3.8-Flash-Next serving | **qualified** separately for TP1/MTP3 text, GPU prefix cache and stock-clock performance; see the [Qwen deployment page](qwen38-flash-next.md) for launch, PLE offload, TP2 limitations and results |
| DeepSeek V4 serving | **qualified** for bounded TP2/DCP1 FP8 text and Vision checks; see the [DS4 runbook](ds4-jovian-community-r29.md) |
| Qualification date | 2026-09-09 |

The [BF16-to-NVFP4 distribution-fidelity report](../kld/glm-5.3-flash-bf16-nvfp4.md),
[QAD step 1,750 comparison](../kld/glm-5.3-flash-qad-step1750.md), and
[QAD step 2,500 progression comparison](../kld/glm-5.3-flash-qad-step2500.md)
are research-only. They measure a reproducible FlashInfer CUTLASS path rather
than the B12X serving path specified here.

The [AA-LCR result](glm-5.3-flash/aa-lcr-bf16-vs-nvfp4.md) qualifies the BF16,
published NVFP4, and QAD configurations on 100 long-context questions with
three independent generations each. The published checkpoint scores 74.00%,
QAD scores 73.00%, and BF16 scores 71.67%; paired evidence does not distinguish
the three complete configurations. The accompanying
[reproduction specification](glm-5.3-flash/aa-lcr-reproduction.md) fixes the
dataset, prompt, sampling, runtime, equality checker, and receipt validation.

The [Verifier-Backed Behavioral Fidelity report](glm-5.3-flash/verifier-backed-behavioral-fidelity.md)
uses 224 deterministic tasks with executable answer keys and no language-model
judge. BF16 scores 93.11%, published NVFP4 scores 91.63%, and QAD step 1,750
scores 92.35% on the primary fractional metric. All paired one-point decisions
are inconclusive; the report separates qualified execution provenance from the
statistical power required to claim behavioral equivalence or improvement.

The [QAD step-2,500 VBF report](glm-5.3-flash/qad-step2500-verifier-backed-behavioral-fidelity.md)
pools 9,856 non-overlapping task pairs from a two-replica TP4 component and an
independent one-replica Max-Q TP4 component. Published NVFP4 scores 91.001%
and QAD step 2,500 scores 91.333%; the +0.332-point difference has a paired
95% interval from -0.317 to +0.982 points, entirely inside the predeclared
±1-point equivalence band. Exact-task accuracy and exploratory family results
remain mixed, so the qualified claim is limited to the primary VBF semantic
score.

## Docker artifact

```text
localinferencelab/vllm:jovian-judgement-community-20260909-r29
localinferencelab/vllm@sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49
```

The image contains two filesystem layers: a flattened CUDA 13.3/PyTorch 2.13
runtime foundation and one installation of complete committed vLLM, B12X and
LMCache sources. FlashInfer, the DS4-compatible native vLLM operator and the
authenticated FlashKDA extension are source-locked. It is not built by adding
layers to a preceding community release.

The [embedded source lock](glm-5.3-flash/validation/shared-serving-r29.source.lock)
has SHA-256 `3307f3372213496e5b7de4fc485ef5b8f7fc43ff99df896ac40fa68d4dd3f80c`.
The [R29 qualification and changelog](glm-5.3-flash/validation/shared-serving-r29.md)
records the immutable image identity, differences from R28.1, measurements,
cache migration requirements and known test limitations.

The same installed runtime supports
[Qwen3.8-Flash-Next](qwen38-flash-next.md) and
[DeepSeek V4 text/Vision](ds4-jovian-community-r29.md) through separate launch
profiles. DS4 backend defaults do not replace the GLM settings below.

## Runtime backends

| Operation | Implementation |
|---|---|
| Target sparse attention and C4 index selection | B12X |
| Target recurrent prefill | FlashKDA with packed checkpoint exports |
| Target recurrent decode | B12X when eligible, with the supported Triton path otherwise |
| Target routed experts | B12X NVFP4 W4A4 |
| Target dense projections | B12X |
| Tensor-parallel all-reduce | B12X PCIe one-shot/two-shot for supported sizes; PyNCCL for the remaining sizes |
| MTP attention / experts | B12X / Marlin |
| MTP vocabulary projection | Private NVFP4 draft copy; target vocabulary projection remains BF16 |
| DFlash2 weights and linear projections | Offline MXFP8 checkpoint and B12X |
| DFlash2 local attention | Graph-safe split-KV FlashAttention |
| Sampling | FlashInfer-compatible probability dispatch and standard rejection |
| External cache | LMCache through worker-owned asynchronous pinned SHM |

The target uses FP8 KV cache. DFlash's local attention uses its compatible
automatic cache dtype; it is not advertised as an MXFP8 KV format.
DeepGEMM and TileLang are installed dependencies but are not selected for the
qualified GLM target, MTP or DFlash2 hot paths. FlashKDA is the prefill default;
`GLM53_KDA_PREFILL_BACKEND=b12x` selects the retained B12X alternative. The
performance table below uses FlashKDA, not that alternative.

## Measured performance

Stock RTX PRO 6000 Blackwell Workstation GPUs, TP4/DCP1, FP8 target KV,
4096-token scheduler budget, OMP1, NCCL 16 channels/2 MiB buffers and
`FULL_AND_PIECEWISE` graphs. The R28.1 control and shared-runtime candidate
were measured on the same physical GPUs3/12/13/14. C1 is context-zero decode,
temperature 1, 15 seconds of warmup and 30 seconds measured.

Prefill is the client TTFT-derived input rate for a cold nominal 32K bucket
(about 32,315 actual tokens), with 12 samples per image. It includes
first-output work rather than timing only an attention kernel.

| Mode | 32K prefill, R28.1 → shared tok/s | C1 output, R28.1 → shared tok/s | C1 verifier, R28.1 → shared steps/s |
|---|---:|---:|---:|
| No-spec | 14,650 → 14,733 (+0.57%) | 169.38 → 169.35 (−0.02%) | Same as output rate |
| MTP3 | 14,279 → 14,320 (+0.29%) | 258.02 → 265.19 (+2.78%) | 102.39 → 108.69 (+6.15%) |
| DFlash2 K7 | 14,519 → 14,529 (+0.07%) | 226.01 → 230.98 (+2.20%) | 89.40 → 89.55 (+0.16%) |

These are bounded observations, not proof of a universal speedup. MTP accepted
length changes 2.520 → 2.440; DFlash changes 2.528 → 2.579. The MTP verifier
difference was not isolated to a particular change, and host-side concurrent
work differed between cells. Source/image boundaries are explicit in the
[qualification report](glm-5.3-flash/validation/shared-serving-r29.md): these
performance cells precede the LMCache metadata and DS4 BF16-router corrections;
their GLM GPU-cache hot path is unchanged by those corrections.

C8/C64 and Sieve were not rerun for R29. The
[R28.1 report](glm-5.3-flash/validation/scheduler-serving-r28.1.md) retains its
C8/C64 results and the short DFlash C8 decrease without relabelling them as R29.
The [R28 report](glm-5.3-flash/validation/fp8-serving-r28.md) retains the
six-mode/DCP matrix and Sieve results. None of the table above uses a +6000
VRAM offset.

## Start the server

Use model names and named volumes; source-code mounts are not needed.
The defaults already select full-and-piecewise graphs, the B12X paths,
FlashInfer sampling, NCCL 16 channels/2 MiB and OMP1.

```bash
IMAGE=localinferencelab/vllm:jovian-judgement-community-20260909-r29
GPU_DEVICES=0,1,2,3
PORT=8000
docker pull "$IMAGE"
```

Choose one mode. No speculation:

```bash
NAME=jovian-judgement-nospec
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=0)
```

MTP3:

```bash
NAME=jovian-judgement-mtp3
MODE_ARGS=(-e SPECULATOR=mtp -e MTP_DEPTH=3)
```

DFlash2 with seven draft tokens:

```bash
NAME=jovian-judgement-dflash2
MODE_ARGS=(-e SPECULATOR=dflash2 -e DFLASH_DEPTH=7
  -e DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2)
```

Run the common command after assigning the chosen mode's variables:

```bash
docker run -d --name "$NAME" --init \
  --gpus "\"device=${GPU_DEVICES}\"" --network host --ipc host \
  -v jovian-judgement-r29-runtime-cache:/cache \
  -v jovian-judgement-huggingface-cache:/root/.cache/huggingface \
  -e MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4 \
  -e CACHE_MODE=vram -e KV_CACHE_QUANT=fp8_ds_mla \
  -e TP=4 -e DCP=1 -e PORT="$PORT" \
  -e MAX_MODEL_LEN=1048576 -e MAX_NUM_SEQS=32 \
  -e MAX_NUM_BATCHED_TOKENS=4096 \
  -e PREFILL_SCHEDULE_INTERVAL=1 -e PREFILL_COMPUTE_SHARE=0.4 \
  -e GPU_MEMORY_UTILIZATION=0.93 \
  "${MODE_ARGS[@]}" "$IMAGE"
```

For DCP4, change `-e DCP=1` to `-e DCP=4`. Full-CKV prefill is selected
automatically; it is not DFlash-only. DCP2 and TP8 are implemented but are not
independently qualified by the TP4/DCP1/DCP4 measurements on this page.

Model names resolve Hugging Face `main` at startup. Optional `MODEL_REVISION`
and `DFLASH_MODEL_REVISION` variables provide reproducible model selection.
The runtime authenticates the resolved revisions before external checkpoint
reuse; changing a model or source identity produces safe cache misses.

### Scheduler and cache geometry

`PREFILL_COMPUTE_SHARE=0.4` targets 40% of measured execution time for prefill
only while prefill and decode contend. `PREFILL_SCHEDULE_INTERVAL=1` is required.
Set `FAIRNESS_ENGINE=none` to disable this policy. Model execution is indivisible,
so the realized share can oscillate over short windows. External-cache transfers
do not consume local-prefill compute credit or a prefill lane while waiting.

The scheduler launcher exposes these controls. Explicit native CLI arguments
after the image name override corresponding environment values without duplicate
flags.

| Environment | Native CLI option | Values and default |
|---|---|---|
| `PREFILL_COMPUTE_SHARE` | `--prefill-compute-share` | Finite number strictly between 0 and 1, or `auto`; launcher default `0.4` |
| `PREFILL_COMPUTE_HALF_LIFE` | `--prefill-compute-half-life` | `smooth`, `responsive`, or positive finite seconds; valid only with share `auto` |
| `MAX_PARALLEL_PREFILLS` | `--max-parallel-prefills` | Positive integer or `auto`; default `1`; `auto` selects at most four lanes, capped by `MAX_NUM_SEQS` |
| `PREFILL_POLICY` | `--prefill-policy` | `round-robin` (default) or `decode-aware` |
| `DECODE_REFILL_TARGET` | `--decode-refill-target` | Positive integer or `auto` (default); automatic target equals the effective lane count |

The lane count is independent of attention pages, recurrent checkpoints and
LMCache object size. All lanes share the same global 4096-token scheduler budget;
four lanes do not multiply that budget by four.

For concurrent long prefills and latency-sensitive short requests, opt into:

```bash
# Add before "$IMAGE" in the common docker run command:
-e MAX_PARALLEL_PREFILLS=auto \
-e PREFILL_POLICY=decode-aware \
-e DECODE_REFILL_TARGET=auto
```

Keep fixed share `0.4` initially. Interleaving can bring a short request to decode
sooner by distributing service among long requests; it can increase their
individual time to first token. One lane remains the image default. Automatic
compute share is implemented for experiments, not selected as the production
default. `FAIRNESS_ENGINE=micro_slicing` is rejected.

MTP uses a private NVFP4
proposal vocabulary head by default, costing 85.08 MiB per TP4 rank. The target
vocabulary head remains BF16; `VLLM_GLM53_MTP_DRAFT_HEAD=bf16` selects the
unquantized draft head for independent comparison.

The launcher owns page geometry. GPU-local serving keeps 2048-token attention
pages; fine recurrent retention is selected separately below. LMCache derives
per-rank pages from 4096-token storage objects and the DCP width. The public
vLLM attention block argument remains 256. Weighted allocation groups layers
by their actual cache cost, with a bounded number of groups, to reduce padding.
Normal deployments do not need to override these layout settings.

### Reasoning effort

The GLM image launcher defaults chat reasoning to **high**, not max. An API
request can override it with `"reasoning_effort":"max"` or
`"reasoning_effort":"low"`. High still enables reasoning; it is not a
no-thinking mode. To change the server default, pass
`--default-chat-template-kwargs '{"reasoning_effort":"max"}'` after the image
name. Running `vllm serve` directly bypasses the image launcher's default.

The GLM chat profile also sets `clear_thinking=true`: completed-turn reasoning
is omitted when rendering later requests, while visible answers, tool exchanges
and reasoning within the active tool cycle are retained. This follows the
[checkpoint author's chat recommendation](https://huggingface.co/zai-org/GLM-5.3-Flash#note).
It reduces repeated reasoning-history growth; it is not a change to target
weights or proof that all long-context generation problems are fixed. Explicit
request template options override the profile. When replacing the whole server
JSON, include `"clear_thinking":true` if that behavior is desired.

### Recurrent checkpoint policy

The default `auto` policy selects request-boundary retention for the qualified
GLM no-spec/MTP3/DFlash2 configurations, including DCP4. It preserves exact
request endpoints and leading SYSTEM/developer instruction endpoints. It does
not promise a hit at every arbitrary byte or token inside a changed user turn.

For arbitrary shared token prefixes, GPU-local aligned retention is selectable.
Add the following environment setting before the image name and arguments
after the image name:

```bash
# Before "$IMAGE":
-e GLM53_MAMBA_BLOCK_SIZE=256

# After "$IMAGE":
--recurrent-checkpoint-policy aligned --prefix-match-unit 256
```

Attention pages remain 2048 tokens. Packed exports retain interior recurrent
states without forcing a target forward per checkpoint. This trades additional
retained recurrent state for finer prefix reuse. Request-boundary retention
stores fewer states for ordinary shared-instruction and turn-boundary reuse.
The fine-aligned MTP3/DCP4 interval comparison measured −0.20% prefill,
+0.32% C1 verifier and +0.35% C8 verifier rate; that comparison precedes the
disjoint-MLA projection change. It is not a fresh speed claim for every mode.

## LMCache RAM and filesystem storage

LMCache is opt-in. In the common command, replace `-e CACHE_MODE=vram` with:

```bash
-e CACHE_MODE=lmcache \
-e LMCACHE_TRANSFER_MODE=engine_driven \
-e LMCACHE_L1_SIZE_GB=64 \
-e LMCACHE_L2_ENABLED=1 \
-v jovian-judgement-r29-lmcache-l2:/lmcache-l2
```

The host shared-memory filesystem must have at least 96 GiB available for the
default 64 GiB RAM pool and transfer buffers. With `--ipc host`, the host's
`/dev/shm` capacity applies; `--shm-size` does not enlarge it. The sidecar runs
inside the container and has no CUDA context. Existing vLLM workers perform
GPU gather/scatter through asynchronous pinned SHM. Do not manually remove a
shared-memory pool while either service is using it.

The launcher chooses geometry compatible with `LMCACHE_CHUNK_SIZE=4096` and
`LMCACHE_TARGET_TOKEN_BUDGET=4096`. Semantic target, recurrent and draft state
are published only as complete all-rank bundles. Payload locks protect active
copies, and incompatible source/model/layout identities safely miss. RAM
pressure evicts unlocked payloads; filesystem storage remains reusable after
both services restart. Incomplete or incompatible semantic generations are
recomputed, never partially imported. Version-1 semantic payload files do not
match the version-2 storage keys used here.

Set `LMCACHE_L2_ENABLED=0` for RAM-only operation. If multiple instances share
the host network, give each distinct API and LMCache HTTP/MP/metrics ports.
Do not share a writable cache directory across independent sidecars.

The retained recurrent-checkpoint architecture has
[all-six FP8 mode/DCP million-token qualification](glm-5.3-flash/validation/fp8-serving-r28.md#checkpoint-storage).
The exact R28.1 image separately passes MTP3/DCP4 with four prefill lanes:

| Mode / DCP | 1M cold, seconds | RAM restore, seconds | Restore after both services restart, seconds |
|---|---:|---:|---:|
| MTP3 / 4, four lanes | 99.284 | 0.855 | 0.970 |

Each restore attributes all one million prompt tokens to external storage and
zero to local compute. Times include API/first-output work. The OS page cache
was not flushed; restart results are not cold-device storage benchmarks.
The R28.1 image also passes 54K literal lookup answers across cache tiers,
shared-SYSTEM reuse, C4 all-rank bytes and C8 cancellation/live-read eviction.
These tests qualify storage correctness, not universal bitwise generation
equivalence across different floating-point prefill partitions.

A separate one-observation, same-quartet R28/R28.1 RAM comparison records
0.725 → 0.813 seconds for 1M tokens. That 88 ms difference is retained in the
report; it is insufficient to establish steady-state transfer-speed equivalence.

Native DRAM offload remains implemented through `CACHE_MODE=native`; it is not
independently requalified for R29. The qualified external path above is LMCache.
Packed NVFP4 target KV and Qwen LMCache are outside this release's qualification.

R29 additionally qualifies DFlash2/DCP4 after the paged-gather metadata fix:
54K cold/RAM/restart-filesystem answers, shared SYSTEM reuse, exact comparison
of 3,639,803,904 transferred bytes across four ranks, and three C8 cancellation
and live-read eviction rounds. This bounded check is not a repeat of the
one-million-token timing matrix.

Use an empty external-cache namespace for R29. Its immutable pinned block-ID
snapshots prevent asynchronous gathers from copying a later batch's pages.
The correction cannot repair payloads written without that guarantee. Atomic
GLM checkpoint identities reject incompatible sources; fresh named volumes
also isolate ordinary external-cache objects. Preserve existing volumes until
their owner chooses to remove them.

For independent native-offload testing, use `-e CACHE_MODE=native` and
`-e NATIVE_KV_OFFLOADING_SIZE_GB=64` in the common command. The launcher selects
the required shareable allocator. These settings do not enable LMCache.

## Source and review contract

The [portable source-locked build recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/codex/glm53-source-locked-build/recipes/glm53)
includes the native FlashKDA build, source-bundle verification and CPU build
tests. It lists the exact source revisions; no chain of preceding community
images is needed. Runtime ABI dependencies are supplied by its pinned base.

Complete Git mirrors preserve authorship and integration resolutions:
[vLLM](https://github.com/voipmonitor/vllm/tree/integration/jovian-shared-serving-20260909-r29),
[B12X](https://github.com/voipmonitor/b12x/tree/release/jovian-judgement-20260909-r29),
[LMCache](https://github.com/local-inference-lab/LMCache/tree/release/jovian-judgement-20260909-r29).
The [open merge checklist](https://github.com/local-inference-lab/vllm/issues/651)
describes each PR and integration caveat. Source locks, not tag-name inference,
identify the measured packages. The timing matrix and exact packaged storage
checks have their source boundaries recorded in the validation evidence.

Git histories retain the original contributions and author attribution,
including Luke Alonso, MadeBy561, Giancarlo Delfin, Derek Yates, Thien Tran,
logprobz, Apple FCU Fleet, Martin Vit, Codex and the other recorded authors.
