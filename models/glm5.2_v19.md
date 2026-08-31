# GLM-5.2 v19: Gilded Gnosis

v19 is the previous unified GLM-5.2 and DeepSeek-V4/DSpark image. It moved the
release onto the canonical `dev/gilded-gnosis` branch, adopts the canonical
B12X cache-key fix, and preserves the reviewed v18 feature set. This page is a
standalone successor to [v17](glm5.2_v17.md) and [v18](glm5.2_v18.md): it
contains exact source provenance, launch controls, accuracy references, and a
new full performance campaign. The current release is [v20](glm5.2_v20.md).

There is no private vLLM or B12X source overlay in this image. Its three vLLM
release deltas are public, non-draft PRs: the SM120 CUTLASS DSL pin, the DCP
packed-A2A prewarm-buffer lifetime fix, and corrected MRV2 CUDA-graph memory
profiling. The deterministic CuTe cache-key fix is merged in canonical B12X
master.

## Release Image

```text
voipmonitor/vllm:gilded-gnosis-v19-vllmf879d86-b12xc7dc733-fi801d57a-cu132-20260719
Docker manifest: sha256:dfd3b7a4f2e02ecf1b2ad826d03293de9648dbefcd29b195422c38166b01fe8b
Local image ID: sha256:5014f02f99143a16121018dcfc3cf11fce101c5edfd7e06f971f94490c839a89
```

Pinned source stack:

| Component | Ref / commit |
|---|---|
| Canonical GG base | `local-inference-lab/vllm dev/gilded-gnosis` @ `371085e9e4ee3471125d69cfbfcfc66864634ee4` |
| vLLM release source | `voipmonitor/vllm build/gilded-gnosis-v19-final3-20260719` @ `f879d8633e872703bb7aae409d06e34269364625` |
| Canonical B12X base | `lukealonso/b12x master` @ `c7dc73322cc50609f843fa2bbcc53283a90003b3` |
| B12X release source | `voipmonitor/b12x build/gilded-gnosis-v19-final-b12x-canonical-20260719` @ `c7dc73322cc50609f843fa2bbcc53283a90003b3` |
| FlashInfer | `801d57a08958c13d375ddbb6be3be4808f48a708` |
| CUTLASS C++ source | `e6233cbac5d7c7a865c19c91cd684ceece19513c` |
| CUTLASS DSL runtime | `4.5.3`; the CUDA 13 wheel is installed last and file-hash verified |
| InstantTensor | `85e7c5f5539d9c006ee0c26bc1b5233c65251b6b` |
| DeepGEMM | `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| NCCL | local-inference `2.30.4` |
| PyTorch / CUDA | PyTorch `2.12.0+cu132`, CUDA `13.2.1` |
| Build repository | `local-inference-lab/blackwell-llm-docker` @ `beaf6e5` |

The canonical build script is
[`build-gilded-gnosis-v19-cu132.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/beaf6e5/build-gilded-gnosis-v19-cu132.sh).
It verifies immutable source commits, the CUTLASS CUDA-wheel file hash, SM120
runtime symbols, NVFP4 MLA CUDA writes, the current NF3 hybrid API,
InstantTensor, NCCL, and all GLM/DS4 helper modes before an optional push.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout beaf6e5
PUSH_IMAGE=1 ./build-gilded-gnosis-v19-cu132.sh
```

## Source Audit

Luke's canonical `dev/gilded-gnosis` branch now contains the previously separate
GG integration stack: DCP query split and selected-CKV gather, NF3 Grid188,
NVFP4 KV support, DCP graph-buffer lifetime fixes, MTP revision inheritance,
DSpark, and the SM120 PCIe serving work. The obsolete staging PRs were closed
after verifying their commits were represented in the canonical branch.

Only these release deltas remain:

| Component | Review | Purpose |
|---|---|---|
| vLLM | [PR #128](https://github.com/local-inference-lab/vllm/pull/128) | Pin CUTLASS DSL 4.5.3 after the SM120 W4A16 regression in 4.6.0. |
| vLLM | [PR #130](https://github.com/local-inference-lab/vllm/pull/130) | Keep packed NCCL DCP A2A staging buffers alive across descriptor prewarm and CUDA graph replay. |
| vLLM | [PR #131](https://github.com/local-inference-lab/vllm/pull/131) | Profile MRV2 target and draft CUDA graphs plus sparse-attention transient memory before sizing the KV cache. |
| B12X | [merged PR #41](https://github.com/lukealonso/b12x/pull/41) | Make CuTe DSL option cache keys deterministic across worker processes and restarts. |

The release has no `VLLM_PATCH_URL`, `VLLM_PATCH_FILE`, source bind mount, or
hidden Docker patch. Image labels identify both release branches and full
commits.

## Changes From v18

- vLLM is rebuilt from canonical `dev/gilded-gnosis` plus the three reviewed
  release deltas in PR #128, PR #130, and PR #131.
- B12X is canonical master `c7dc733`, including the merged deterministic CuTe
  cache key and the newer high-row sparse-indexer path.
- DCP query split and transient full-CKV gather are enabled by default for all
  measured-beneficial TP4/TP8 `DCP>1` topologies, including DCP2.
- Virtual TP6 keeps the faster borrowed-workspace path after direct A/B tests
  rejected full CKV and query split for its 11-head virtual shard.
- NF3 Grid188 remains enabled by default with an explicit disable switch.
- InstantTensor `BUFFERED`, local NCCL 2.30.4, FlashInfer autotune, request
  usage details, and the exact 78-character indexer pattern remain helper
  defaults.
- The release is validated by a new CC1-to-CC32 and 8k/64k full campaign, not
  by relabeling inherited v17/v18 cells.

## Persistent Compile Cache Fix

The previous cache key included `repr(dsl_compile_options)`. CUTLASS
`OptLevel`'s representation can contain process-specific identity, so identical
TP ranks or a second server run could generate different persistent keys. That
made a warm run compile again and could make it slower than the first run.

B12X PR #41 serializes the option structurally instead. For `OptLevel(2)`, the
stable semantic value is:

```text
["object", "cutlass.base_dsl.compiler", "OptLevel", [["_value", 2]]]
```

The exact TP6/DCP3/MTP3 MXFP4 A8 configuration was tested with an empty cache
and then restarted against the same cache:

| Observation | Cold cache | Exact restart |
|---|---:|---:|
| CuTe object files before first request | 181 | 182 |
| CuTe object files after first request | 182 | 182 |
| First-request wall time | 3.38 s | 1.23 s |
| Dynamic W4A8 compile behavior | one miss; five ranks wait and reuse it | six direct disk hits |
| Cache key | `73ca22a7961bbfe2` | `73ca22a7961bbfe2` |

The restart reached readiness in 105.839 seconds and did not create another
object. The PR's CUDA 13.2/CUTLASS 4.5.3 test suite passed `35 passed, 1
skipped`.

## CUTLASS 4.6 Regression

The rebased upstream requirements moved to CUTLASS DSL 4.6.0. A controlled
TP8/DCP1/MTP0 comparison found that this left decode unchanged but reduced 64k
prefill from about 5,909 to 5,052 tok/s for Luke NVFP4 A16.

The generated B12X W4A16 fused-MoE prefill kernel changed from 242 registers
and no stack to 255 registers and 256 stack bytes per thread. Pinning only the
runtime DSL to 4.5.3 restored the baseline without a B12X source workaround.

| Case | DSL 4.6.0 | DSL 4.5.3 | Change from 4.6.0 |
|---|---:|---:|---:|
| A16 original, decode CC1 | 87.36 | 87.18 | noise |
| A16 original, prefill 64k | 5,052 | 5,909 | +16.96% |
| A16 online MXFP8, decode CC1 | 92.64 | 92.18 | noise |
| A16 online MXFP8, prefill 64k | 5,060 | 5,909 | +16.78% |

The profiler evidence, raw repeats, rejected route-block workaround,
wheel-order trap, and exact reproduction procedure are in
[CUTLASS DSL 4.6 SM120 W4A16 Prefill Regression](glm5.2/cutlass-dsl-46-w4a16-regression-2026-07-18.md).

## Start The Server

The image contains `/usr/local/bin/serve-gilded-gnosis.sh`. The Compose file
keeps backend details inside the image helper:

- [`serve-glm52-v19.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/beaf6e5/launchers/serve-glm52-v19.sh)
  owns the topology-aware CKV/query-split dispatch;
- [`serve-glm52-v16.sh`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/beaf6e5/launchers/serve-glm52-v16.sh)
  owns the common GLM arguments and backend defaults;
- [`docker-compose-glm52-v19.yml`](https://github.com/local-inference-lab/blackwell-llm-docker/blob/beaf6e5/examples/docker-compose-glm52-v19.yml)
  is the minimal user-facing Compose file.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker
git checkout beaf6e5

# Highest-accuracy standard mode: Luke NVFP4, A16, DCP1, MTP off.
MOE_MODE=a16 MTP=0 DCP=1 \
  docker compose -f examples/docker-compose-glm52-v19.yml up -d
```

Stable user-facing controls:

| Variable | Values / meaning |
|---|---|
| `MODEL` | Checkpoint path or Hugging Face repository |
| `GPUS` | Visible physical GPU list |
| `TP` | `8`, virtual-sharded `6`, or hybrid-checkpoint `4` |
| `DCP` | `1`, `2`, `3`, `4`, `6`, or `8` where topology permits |
| `DCP_QUERY_SPLIT` | `auto` by default; `0` or `1` is an experimental override |
| `DCP_CKV_GATHER` | `auto` by default; `0` or `1` is an experimental override |
| `DCP_PREFILL_WORKSPACE` | `auto`; keeps the borrowed-workspace path for topologies that do not use full CKV |
| `MTP` | `0` or speculative token count such as `3` |
| `MAX_NUM_SEQS` | Scheduler concurrency; default graph cap is `4 * MAX_NUM_SEQS` |
| `MOE_MODE` | `a4`, `a16`, or `force-a8-experimental` |
| `ONLINE_QUANT` | `none`, `mxfp8`, `fp8`, or the hybrid helper default |
| `NF3_GRID188` | `1` by default for the NF3 hybrid; set `0` only for a matched A/B test |

The helper owns the 78-character sparse-indexer pattern, B12X attention and
MoE selection, hybrid DCP communication, CUDA graph sizing, FP8 KV cache,
FlashInfer autotune, local NCCL, and request-usage API defaults. InstantTensor
is the default loader:

```text
LOAD_FORMAT=instanttensor
INSTANTTENSOR_BACKEND=BUFFERED
```

Checkpoint modes and online-quant semantics are unchanged from
[v18](glm5.2_v18.md#checkpoint-modes).

### DCP Prefill Dispatch

`auto` selects the measured fastest path rather than treating every `DCP>1`
topology as equivalent:

| TP / DCP | Query split | Full-CKV gather | Selected prefill path |
|---|---:|---:|---|
| TP8 / DCP1 | off | off | ordinary DCP1 path |
| TP8 / DCP2, DCP4, DCP8 | on | on | query split plus transient full CKV |
| TP4 / DCP1 | off | off | ordinary DCP1 path |
| TP4 / DCP2, DCP4 | on | on | query split plus transient full CKV |
| virtual TP6 / DCP1, DCP2, DCP3, DCP6 | off | off | borrowed B12X DCP workspace |

Virtual TP6 pads 64 attention heads to 66 and therefore has 11 local heads.
The B12X full-CKV kernel is 8-head aligned, so an experimental 11-to-16 head
padding path increased compute and communication. At TP6/DCP3, full CKV
reduced 64k prefill from `3,394` to `2,483 tok/s` (`-26.8%`); query split by
itself reduced it to `2,326 tok/s` (`-31.5%`). Both remain available as
explicit experiments, but they are not release defaults.

On aligned topologies the optimization is material:

| Matched A/B | Baseline 64k | Query split + full CKV | Change |
|---|---:|---:|---:|
| TP8/DCP2, Luke NVFP4 A16 | 4,756 | 5,522 | +16.1% |
| TP4/DCP4, NF3 hybrid A16 | 2,362 | 3,548 | +50.2% |

The full-sweep runner asserts the actual runtime messages for these paths; it
does not infer activation merely from environment variables.

## Accuracy Reference

Speed tests do not change model quality, so v19 retains the corrected-reference
KLD campaign produced with the same checkpoints. Lower KLD is better.

| Case | KLD mean +/- sample SD | Accuracy role |
|---|---:|---|
| Luke NVFP4 A4 original | 0.10228 +/- 0.00634 | Native A4 activation path |
| Luke NVFP4 A4 online MXFP8 | 0.10800 +/- 0.00697 | Faster dense linears, small KLD cost |
| Luke NVFP4 A16 original | 0.05994 +/- 0.00129 | Highest-accuracy tested mode |
| Luke NVFP4 A16 online MXFP8 | 0.06587 +/- 0.00253 | A16 accuracy/speed balance |
| AMD MXFP4 experts A8 original | 0.08160 +/- 0.00432 | Native BF16 dense linears |
| AMD MXFP4 experts A8 online MXFP8 | 0.08030 +/- 0.00309 | Faster dense linears; same measured distribution |

`A4` and `A16` select the activation path for Luke's NVFP4 routed experts;
they do not rewrite the checkpoint weights. `force-a8-experimental` selects
the MXFP4 expert A8 path and does not apply to the NVFP4 checkpoint. Online
MXFP8 only converts eligible BF16 linears selected by the helper's explicit
quantization configuration.

## Validation Method

The full v19 runner is
[`scripts/bench-glm52-v19-full.sh`](../scripts/bench-glm52-v19-full.sh). The
smaller release smoke test remains in
[`scripts/bench-glm52-v19-validation.sh`](../scripts/bench-glm52-v19-validation.sh).

- all 16 GPUs are used as two isolated servers where topology permits;
- both models finish loading and settle before the first client starts;
- clients run serially, never while another model is loading;
- each case checks short-output correctness and expected backend markers;
- decode covers CC1, CC2, CC4, CC8, CC16, and CC32 at context zero;
- 8k and 64k standalone prefill use one discarded 64k warmup and three
  measured runs; tables report medians;
- thermal snapshots are saved before and after every case;
- topology-specific assertions require full CKV on aligned TP4/TP8 DCP paths,
  the borrowed workspace on virtual TP6, and Grid188 execution on NF3;
- an incremental gate stops the campaign if CC1, CC32, or 64k prefill regresses
  by more than 5% against the corresponding v17/v18 result;
- the runner rejects an unexpected image ID or any source bind mount.

## v19 TP6 Regression Gate

The final image was tested with
`/root/models/GLM-5.2-BF16-AMDMXFP4experts`, force A8, MTP3, InstantTensor
`BUFFERED`, FP8 KV, `MAX_NUM_SEQS=16`, and graph cap 64. DCP3 and DCP6 pairs
were loaded concurrently; measurements started only after both servers were
ready and then ran serially.

| Dense mode | TP / DCP | Decode CC1 | Acceptance | Prefill 64k runs | Median | KV tokens |
|---|---:|---:|---:|---|---:|---:|
| Original BF16 dense | 6 / 3 | 87.29 | 0.513 | 3495 / 3493 / 3494 | 3494 | 680,491 |
| Online MXFP8 dense | 6 / 3 | 98.06 | 0.601 | 3477 / 3479 / 3478 | 3478 | 852,245 |
| Original BF16 dense | 6 / 6 | 86.60 | 0.631 | 2366 / 2369 / 2366 | 2366 | 1,343,233 |
| Online MXFP8 dense | 6 / 6 | 87.79 | 0.595 | 2367 / 2369 / 2368 | 2368 | 1,686,227 |

Prefill matches v18 within 0.4%. DCP6 decode also matches the five-run v18
distribution (mean 86.84, median 86.84 tok/s).

The DCP3 original one-shot was followed by five more decode runs without a
server restart. The six values were `47.90 / 96.62 / 98.52 / 70.19 / 90.62 /
61.76` tok/s. This workload leaves temperature at the model default and MTP
uses probabilistic draft sampling, so different continuations produced very
different acceptance and throughput. The same final process reached 98.52
tok/s, proving that 87.29 is not a fixed kernel ceiling; its 64k prefill stayed
at 3.48-3.49k tok/s throughout. Do not use one sampled MTP CC1 cell as a kernel
regression gate without also reporting acceptance. Use MTP0 for the
acceptance-independent gate and repeated MTP runs for user-visible behavior.

Raw result root on the validation host:

```text
/root/bench-results/glm52-v19-final-newmaster-tp6-20260718
/root/bench-results/glm52-v19-final-newmaster-dcp3-orig-reruns-20260719
```

## Sparse Indexer Change

The final B12X base includes commits `39d7404`, `531f2b5`, and `a93ffcc`. They
add a warp-owns-tokens direct-L2 path for the fused DSV4/GLM sparse indexer and
keep `rows <= 2` on the old staged path.

This is primarily a higher-concurrency **decode** optimization:

- CC1 and normally CC2 are controls and should remain unchanged;
- the new path becomes relevant from about three active decode rows;
- the reported 20% applies to the indexer kernel, not the complete server step;
- expected end-to-end gain is smaller according to the indexer's share of the
  decode step;
- standalone prefill is not expected to change from this patch.

Validate it with matched old/new decode sweeps at CC1/2/4/8/16/32 and longer
contexts. Do not infer its effect from CC1 or from a prefill-only benchmark.

<!-- BEGIN GENERATED V19 FULL RESULTS -->

Full exact-image results are generated after the 64-case campaign completes.

<!-- END GENERATED V19 FULL RESULTS -->

## Reproduce The Validation

```bash
cd rtx6kpro

# Complete resumable campaign. Reusing RESULT_ROOT skips finished cases.
RESULT_ROOT=/root/bench-results/glm52-v19-full-20260719 \
  TOKEN_TARGETING=estimate scripts/bench-glm52-v19-full.sh all

# Refuse publication unless all 64 expected configurations exist, then update
# only the generated-results block on this page.
scripts/render-glm52-v19-results.py \
  /root/bench-results/glm52-v19-full-20260719 \
  --check-complete --update-page models/glm5.2_v19.md

# Smaller release smoke tests remain available independently.
TOKEN_TARGETING=estimate scripts/bench-glm52-v19-validation.sh dcp1-mtp0
TOKEN_TARGETING=exact scripts/bench-glm52-v19-validation.sh tp6-mtp3
```

The full runner stores every client JSON, server log, container inspection,
thermal snapshot, correctness response, and per-case summary below its result
root. It is safe to resume: a case is skipped only when both `summary.json` and
the `complete` marker exist. Set `FORCE_RERUN=1` only for a deliberate matched
retest.
