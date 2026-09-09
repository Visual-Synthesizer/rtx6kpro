# Shared Jovian Judgement serving qualification: R30

Status: **qualified for the bounded checks and artifact boundaries below**.
This report includes the historical R29-to-R30 changelog. It does not extend
hardware or precision qualification beyond the explicitly measured cases.

## Artifact

Image: `localinferencelab/vllm:jovian-judgement-community-20260909-r30`.
Registry digest: `sha256:5f6fcbc681f20b7c052815ca17511d9fe789aea314a17723c202789dd7adc131`.
Tested image ID:
`sha256:1be0022694c3a2dcd9df9cacebf53e09b7a777815d7b5755ff7a54bd23b5249c`.
The [source lock](shared-serving-r30.source.lock) has SHA-256
`a293571bd5c0e5b18b04e6e42e5122b4783e64e61ad3f71031fead99fdab7d98`.

| Component | Source revision and complete history |
|---|---|
| vLLM | [`60e72555e755e094e0c0c0ddfd65917514cc2151`](https://github.com/voipmonitor/vllm/tree/integration/jovian-immutable-cache-serving-20260909) |
| B12X | [`3edbcbce70f491741b82f5eab9c1b30b39447228`](https://github.com/voipmonitor/b12x/tree/release/jovian-judgement-20260909-r29) |
| LMCache | [`a3a230c8f655749a8d220aabebea5deec4c66497`](https://github.com/local-inference-lab/LMCache/tree/integration/jovian-checkpoint-dedup-20260909) |
| Recipe | [`071c998ec`](https://github.com/local-inference-lab/blackwell-llm-docker/tree/071c998ec/recipes/glm53) |

There are exactly two filesystem layers: a flattened runtime foundation and
one committed source installation. Fourteen native-library hashes are unchanged
from the source-qualified artifact identified below. CUDA, PyTorch, FlashInfer,
FlashKDA, B12X and native LMCache extensions are not replaced by this release.
Installed component Git trees are clean and the source-lock hash matches the
OCI label. Displayed vLLM version: `0.26.1rc0+glm53.r30.vllm60e72555`.

The [registry verification receipt](shared-serving-r30.registry.json) confirms
the remote configuration digest, two manifest layers and pull-by-digest match
to the tested image.

## Changelog from R29

- GLM defaults to `temperature=1`, `top_p=0.95`, `reasoning_effort=high` and
  `clear_thinking=false`. Agent history is preserved. Fixed top-k remains
  disabled, matching the absence of a publisher top-k recommendation.
- The DS4 agent launcher supplies its publisher-recommended temperature 1 and
  top-p 0.95. Qwen retains its own temperature 1/top-p 0.95/top-k 20 metadata.
  API requests can override sampling; explicit native generation/configuration
  arguments take precedence over the launcher preset.
- GLM sparse attention keeps short partial-pool entries inside the active
  selection prefix ([vLLM #715](https://github.com/local-inference-lab/vllm/pull/715)).
- Recurrent cleanup retires obsolete blocks across null gaps instead of
  rescanning or retaining them ([#718](https://github.com/local-inference-lab/vllm/pull/718)).
- Local DCP1 warm admission protects checkpoints needed by bounded queued
  requests, while unrelated aged entries remain evictable
  ([#721](https://github.com/local-inference-lab/vllm/pull/721)). This guard is
  disabled with a KV connector, DCP4 or unsupported pipeline geometry.
- Identical LMCache checkpoints avoid duplicate payload copies. Growing
  histories share complete attention pages between immutable endpoints. Derek
  Yates's [LMCache #64](https://github.com/local-inference-lab/LMCache/pull/64)
  is incorporated into the canonical
  [#62](https://github.com/local-inference-lab/LMCache/pull/62) feature branch;
  this does not mean #62 is merged into LMCache dev.
- `LMCACHE_HTTP_HOST` has matching readiness checks for loopback, wildcard and
  IPv6 bindings. Checkpoint policy is forwarded once. Aligned filesystem-cache
  namespaces include immutable model/draft identity, including positional model
  selection. The source recipe records complete source and CUTLASS provenance.

Request/SYSTEM boundaries, optional fine aligned retention, atomic all-rank
checkpoint ownership, full-and-piecewise graphs, GLM B12X paths and the separate
Qwen/DS4 model profiles remain in the image. No new backend switch or GPU clock
override is introduced.

## Cache and source correctness

The installed admission suites pass **245 tests**; the earlier composed cache
suites pass 160 vLLM and 138 LMCache tests. The byte-comparison helper passes
14 tests. The complete publisher-default recipe passes **118 tests**, including
explicit-CLI precedence. Four installed API sampling-precedence cases pass.
The Hugging Face metadata proposal changes only temperature and top-p; weights,
tokenizer, template and special-token IDs are unchanged:
[HF PR #4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/discussions/4).
The Docker preset does not require that PR to be merged.

The five-token GPU sparse-selection oracle fails against R29 and passes with
#715; all 235 eager/graph/ownership cases pass. All seven recurrent-cleanup
regressions fail against R29 and pass with #718.

LMCache conditions: stock RTX PRO 6000 Workstation GPUs 3/12/13/14, TP4,
FP8 target KV, 4096 scheduled tokens, OMP1, NCCL 16 channels/2 MiB buffers,
full-and-piecewise graphs and a CPU-only sidecar with 64 GiB pinned-SHM L1.
Transfer hashing is enabled only for integrity tests, not performance cells.

| Check | Result |
|---|---|
| DFlash2/DCP4, cold 32,768 tokens | 80 payload objects, 505,528,320 bytes |
| Exact local, RAM and filesystem replay | Zero additional payload objects; exact greedy answers |
| 256-token continuation | Reuses 32,768 tokens, computes 256, adds 40 objects |
| 54,643-token literal lookup, cold/RAM/restart-filesystem | Exact answers; restores recompute zero prompt tokens |
| Shared SYSTEM, different USER | 11,340 instruction tokens reused; changed SYSTEM misses |
| DFlash2/DCP4, four concurrent restores | 576 transfers / 3,639,803,904 bytes match on all ranks |
| MTP3/DCP1, four concurrent restores | 448 transfers / 6,176,636,928 bytes match on all ranks |
| No-spec/DCP4, four concurrent restores | 416 transfers / 2,628,747,264 bytes match on all ranks |
| Three C8 cancellation/live-read-eviction rounds | 24 generations / 21,838,823,424 restored bytes match |
| MTP3/DCP1, three 262,144-token cold/warm needle pairs | All six answers exact; warm prompts fully cached |

The 54K restarted literal request takes 0.268 seconds including answer
generation, with warm OS filesystem pages. This is not cold-disk bandwidth.

### Aged local-cache admission

The final R30 image uses MTP3/DCP1 with GPU-local caching, a deliberately
limited 2 GiB KV allocation per rank, 65,536 maximum context, 16 sequence slots
and the normal 4096-token scheduler budget. The engine reports 216,129 logical
KV tokens. No serving source or scheduler observation hooks are mounted.

After 300 independent cold cache keys, each containing 8,189 prompt tokens,
C2 and C4 warmed requests each restore all 8,189 tokens and generate 4096
tokens. All literal-record checks pass. Four requests execute concurrently;
measured active KV usage reaches **88.96%**, with **zero additional
preemptions** and a healthy engine. An earlier 1024-output C4 wave reaches
90.91% and also preserves all four complete prefixes.

Warm requests ignore EOS to hold the requested allocation/output horizon.
Only the literal record is scored; the forced continuation after a natural
answer is not a generated-text quality evaluation.

These GPU checks qualify aged-cache progress and reuse under the observed
pressure. They do not instrument the exact deferral branch; the 245 CPU
admission tests establish its queued-dependency and 80%/95% pressure behavior.

**Retained overload observations:** a four-client 40K cold-holder setup cannot
admit all holders before one finishes. Three 32K cold holders followed by four
8K replay requests reach high occupancy, but those unqueued checkpoints are
no longer reusable: all four safely recompute and the run adds eight
preemptions. No API/engine failure occurs. #721 does not reserve capacity for
requests that have not arrived, nor guarantee arbitrary overload retention.
Those controls are not counted as passing exact-restore tests.

### Evidence boundaries

The three-mode LMCache and C8 cancellation checks use image
`78911161c0ee73edd7b9b71c5fa32ef4efdacd416bd2d752b2a9090eb2bdedb7`,
with the same LMCache source but before #721. The corrected-admission artifact,
`509a7276f6b6659d001864e4a5018f9094a77bcab3cdcee776c5af907e4714c3`,
repeats DFlash2/DCP4 storage/byte checks and adds local MTP3/DCP1 warm and needle
checks. Its vLLM/B12X/LMCache source trees are identical to the published R30
image; only the launcher sampling preset and its recorded inputs differ.
Packaged-default long-history and aged-pressure tests use the final R30 image.
Earlier results are not relabelled as repeated final-image measurements.

## Matched performance

The [machine-readable measurement extracts](shared-serving-r30.measurements.json)
retain duration cells, prefill aggregates, all five Sieve samples per image
and the aged-cache replay records. Private agent history is not published.

Same physical stock-clock quartet and settings above. C1 duration cells use
15 seconds of warmup and 30 seconds measured, temperature 1 and the control's
top-p 1 sampling policy. Nominal 32K prefill uses 12 unique cold requests and
client TTFT, not an isolated kernel timer. The comparison isolates the serving
source; it does not benchmark the final publisher-default sampling preset.

| DFlash2 / DCP4 / LMCache | R29 | R30 source composition | Change |
|---|---:|---:|---:|
| Cold 32K input tok/s | 13,294 | 13,285 | −0.07% |
| C1 output tok/s | 201.73 | 212.33 | +5.26% |
| C1 verifier steps/s | 81.04 | 81.26 | +0.27% |
| Emitted tokens per verifier step | 2.489 | 2.613 | +4.98% |
| Sieve median output tok/s | 256.40 | 336.69 | +31.31% |
| Sieve output min–max tok/s | 242.57–331.69 | 229.21–415.05 | Overlapping ranges |
| Sieve median verifier steps/s | 77.46 | 77.76 | +0.39% |

Sieve uses **temperature 1, top-p 0.95**, one separate warmup and five seeded
4096-output-token requests. Median accepted draft tokens per step, excluding
the bonus token, are 2.297 and 3.333. The broad overlapping output ranges do
**not** demonstrate a repeatable 31% implementation speedup. Prefill and
verifier execution are effectively unchanged in these bounded comparisons;
output tok/s changes with speculative acceptance.

Additional absolute measurements, without matched R29 controls:

| Mode / cache | C1 output tok/s | Verifier steps/s | Cold 32K tok/s |
|---|---:|---:|---:|
| MTP3 / DCP1 / GPU-local | 249.06 | 102.26 | 14,288 |
| MTP3 / DCP1 / LMCache | 264.40 | 108.81 | 14,259 |
| No-spec / DCP4 / LMCache | 151.64 | Not speculative | 13,470 |

GPU-local and LMCache geometries differ; these rows are not a cache-backend A/B.
MTP uses a private NVFP4 draft vocabulary head and retains the BF16 target head.
R30 does not repeat C8/C64 throughput or Qwen/DS4 inference benchmarks. Their
[R29](shared-serving-r29.md) and [R28.1](scheduler-serving-r28.1.md) reports
remain historical evidence, not new R30 measurements.

## Preserved agent history and sampling

The supplied 839,815-token capture retains all 1,428 messages, 13 tool schemas
and historical reasoning. With the same source/input and no speculation,
top-p 1 produces two degenerate responses among three samples; top-p 0.95
produces none among five. Three final-image DFlash2 K7 requests with sampling
omitted by the client use the packaged 1/0.95 defaults and show no degeneration.
All eight top-p 0.95 responses have valid tool arguments, zero cached tokens
and the same complete input-token hash. No history is removed.

These finite stochastic observations support shipping the publisher default;
they do not establish a universal numerical repair or a model-only cause.
Some publisher coding benchmarks use top-p 1 deliberately. An explicit request
can still select it. Fixed top-k is disabled for GLM; `top_p=0.95` removes the
low-probability tail by probability mass, not a fixed percentage of vocabulary.

## Migration and limits

Use a separate R30 external-cache volume or filesystem directory. Preserve
other services' caches. Atomic GLM identities reject incompatible model/source
payloads; ordinary DS4 cache keys do not provide the same image-migration
guarantee. The sidecar HTTP interface has administrative operations; keep it
on loopback or behind a trusted authenticated network boundary.

The 64 GiB L1 profile requires host shared memory for both the pool and transfer
buffers. No NVFP4 target-KV, TP8, pipeline-parallel or long-duration DS4 disk
pressure qualification is added. The one-million-token six-mode checkpoint
matrix is retained from R28, not repeated for this release. Original author
credit remains in complete Git histories and canonical PRs; image inclusion
does not imply those PRs are merged into Jovian Judgement.
