# Prepared-kernel ownership and API compatibility qualification

Status: qualified for the explicitly recorded checks below. No general quality
or throughput claim is made. The foreign process-global MHC buffer pool is
research-only and is not included in the integration.

## Source boundary

The integration branches contain vLLM PRs
[#780](https://github.com/local-inference-lab/vllm/pull/780),
[#782–#787](https://github.com/local-inference-lab/vllm/issues/773),
[#789](https://github.com/local-inference-lab/vllm/pull/789),
[#790](https://github.com/local-inference-lab/vllm/pull/790), and
[#791](https://github.com/local-inference-lab/vllm/pull/791), plus
[B12X #380](https://github.com/local-inference-lab/b12x/pull/380).
These PRs remain open against JJ/master; this qualification publishes source
only to `integration/beta`. [Issue #773](https://github.com/local-inference-lab/vllm/issues/773)
describes the changes for maintainer review.

B12X master `6a4382fb5598027a1a612f73aec758ec49eedbc0` supplies retained
compiled pooled selection. B12X #379 is closed as superseded, not kept as a
parallel implementation. The GLM caller retains and passes the prepared plan
through vLLM #789. Independent BF16 two-shot row counting remains #380;
two-shot is disabled in the model-serving comparisons.

The tested integration source is vLLM
`b4551e2abd4b47be76ece67d84090888795fcd3f` and B12X
`b9f44defd3a871f8c81d51846f3949ac15ace73d`. Source ancestry and installed-file
hashes are checked separately. The diagnostic images below bake their source
changes over the [wheel-composed runtime](../prepared-b12x-serving/); serving
uses no production-source bind mounts. They are local image IDs, not registry
pull digests or independent release qualifications.

Published vLLM beta `8eca34a03304ee681cb8131f28d118f880efb697` adds only the
segmented-replay test extension to that measured production source. All ten
public vLLM PR heads and B12X #380 are ancestors of the respective beta heads.

| Role | Local image SHA-256 |
|---|---|
| Conversation-compatible control before B12X reconciliation | `166fc10a7a9e11eb0e7ef4dfe08ce04731af48c97c8f6e6928272022b161a63d` |
| Reconciled B12X and prepared GLM caller | `cb8dd16e33433f24b2a24d014c8d12fc9606c9eb3e04bfd54a24cf5b5a490eff` |
| DS4.1 Responses and reviewed conversation handling; linear PCIe lookup | `ec81e2005bac38c252329c3c2a6fbcd84188afaef7e54080332cd54359a67ef0` |
| Identical DS4.1 runtime with indexed PCIe lookup | `0c4b935622a16713bc56994a409b5cbec84dfd008561f7e2e2f4a4aa5b2ae3ba` |

## Measurement contract

Stock RTX PRO 6000 Workstation GPUs 4–7, TP4/DCP1, 4,096-token scheduler
budget, temperature 1. Decode reports medians of three context-zero,
30-second runs after 15 seconds of warmup. C8 throughput is aggregate.
32K prefill is one warmed, uncached 30-second window measured from client
TTFT, not isolated GPU time. Receipts check errors, loops, underfilled cells,
warmup, actual arguments, GPU IDs, checkpoint identity and benchmark hash.
Stochastic acceptance changes output even when execution cost is unchanged.

### Retained pooled selection: GLM

| Profile / metric | Control → prepared caller | Change |
|---|---:|---:|
| MTP3 C8 output, tok/s | 856.05 → 857.27 | +0.14% |
| MTP3 C8 summed request verifier, steps/s | 345.99 → 346.01 | +0.01% |
| MTP3 32K prefill, tok/s | 15,486 → 15,531 | +0.29% |
| DFlash2 K7 C1 output, tok/s | 214.31 → 212.08 | −1.04% |
| DFlash2 K7 C1 request verifier, steps/s | 81.90 → 82.08 | +0.22% |
| DFlash2 K7 C8 output, tok/s | 695.88 → 708.64 | +1.83% |
| DFlash2 K7 C8 summed request verifier, steps/s | 272.33 → 276.33 | +1.47% |
| DFlash2 K7 32K prefill, tok/s | 15,702 → 15,702 | 0.00% |

All API checks and measured cells pass. The MTP control contains C8 only;
the prepared-caller C1 median of 249.28 tok/s is not a matched C1 A/B claim.
Native pooled-selection qualification has 64 passing cases, including disabled
JIT resolution, high physical page IDs, live row counts and changed-input
graph replay. Four TP4 BF16 two-shot geometries pass eager and mutated replay.

The broad B12X suite has 129 passes and 27 failures. All 27 failures reproduce
on its source parent and concern obsolete plan-less tests/device declarations.
The broad suite is not reported as wholly passing.

### Declared-plan index: DS4.1

The image pair differs only in PCIe communicator source. Adaptive DSpark K7,
RAM Engram, target temperature 1/top-p .95, and an explicit image-count limit
of two match on both sides. The limit is a benchmark condition, not the
unified profile default.

| Metric | Declaration scan → index | Change |
|---|---:|---:|
| C1 output, tok/s | 247.61 → 254.92 | +2.95% |
| C1 request verifier, steps/s | 101.42 → 99.94 | −1.45% |
| C8 aggregate output, tok/s | 799.07 → 799.57 | +0.06% |
| C8 summed request verifier, steps/s | 352.12 → 350.86 | −0.36% |
| 32K prefill, tok/s | 20,471 → 20,577 | +0.52% |

C1 accepted length is 2.443→2.551; its output ranges overlap
(247.43–266.71 versus 250.35–263.18 tok/s). These results do not isolate a
model-throughput gain or prove statistical equivalence. The benefit is bounded
constant-time metadata lookup: a 512-declaration CPU miss takes 139→0.70 µs.
All 45 composed TP2/TP4 native tests, five API checks and six decode cells pass.

The implementation preserves operation, shape, dtype, strides, normalization
tensor identity, epsilon and first-declaration precedence. The source
[fernandaspets/vllm_sm120#2](https://github.com/fernandaspets/vllm_sm120/pull/2)
omits normalization identity/epsilon from its key. The published minimal
reproducer demonstrates undeclared-call admission, not numerical corruption.
Contributor attribution is retained in #791.

### Breakable prefill: DS4.1

The indexed-lookup image is unchanged; only
`VLLM_USE_BREAKABLE_CUDAGRAPH=0/1` differs. Image-count limit two is explicit
in both arms. Full/piecewise target and draft decode graphs remain enabled.

| Metric | Prefill graph disabled → enabled | Change |
|---|---:|---:|
| 32K prefill, tok/s | 20,577 → 20,801 | +1.09% |
| C1 output, tok/s | 254.92 → 252.00 | −1.15% |
| C1 request verifier, steps/s | 99.94 → 101.52 | +1.57% |
| C8 output, tok/s | 799.57 → 816.14 | +2.07% |
| C8 summed request verifier, steps/s | 350.86 → 351.32 | +0.13% |
| Logical KV tokens | 4,763,329 → 3,975,153 | −16.55% |
| Actual graph memory per rank | 1.65–1.67 → 2.80 GiB | Increased |

Both arms pass five API checks and all six decode cells. C1 accepted length
changes 2.551→2.483. Prefill capture therefore remains opt-in: the measured
prefill difference is small and comes with materially less KV capacity.

The process-global output pool in
[fernandaspets/vllm_sm120#1](https://github.com/fernandaspets/vllm_sm120/pull/1)
is not required to pass these checks with #780's ownership correction.
Four real-kernel tests also check full/breakable shared-pool replay, lagged
outputs crossing an eager boundary, changed inputs and released transient
owners. No isolated pool-on/pool-off speedup is claimed. In particular, the
foreign PR's graph-off/graph-on comparison does not isolate its buffer change.

## DS4.1 API and image contract

The composed frontend, V4/V4.1 tokenizer and parser suite passes 187 tests.
Fifteen live conversation checks cover namespaces, reminders, tool history,
named choice and malformed scalar calls. Six Responses API checks cover
string input, `input_text`, streaming, `output_text` history, synthetic tool
history and image input with `detail: auto`.

Responses compatibility is an attributed backport of merged upstream
[vllm-project/vllm#56299](https://github.com/vllm-project/vllm/pull/56299).
It normalizes text-part names and does not change kernels, sampling or image
processing. PR #787's two reproduced review findings are corrected and the
review threads resolved. These checks are not a general language-model
quality evaluation or a repetition fix.

The unified model profiles do not set a one/two-image cap. Unspecified native
vLLM counts default to 999; image resolution, encoder budget, context and
memory remain limits. Qwen is deliberately text-only by default and requires
`LANGUAGE_MODEL_ONLY=0` for vision. Historical image-limited performance
receipts do not qualify arbitrary image counts.

GLM passes two, eight and sixteen distinct 128×128 images, conversation history,
repeat and changed-image checks. A long-history control reuses 6,144 prefix
tokens and recognizes a changed final image. A missing `detail` in an initial
Responses fixture and a wrong served-model name in an initial GLM fixture were
client-test errors, not runtime defects; only valid fixture results are used.

DS4.1 passes the same six image/history cases without a count override.
Repeated history reuses 8,192 tokens. The changed-image case returns green
but records zero prefix hits; prefix reuse across changed DS4.1 images is not
qualified. Its logical KV capacity remains 4,763,329 tokens. These 128×128
fixtures do not qualify arbitrary image resolutions or 999-image requests.

## Evidence interpretation

The export manifest records artifact checksums. Per-arm launch/runtime files
identify GPUs, checkpoint files, arguments and images. Functional checks and
benchmark outputs retain their original observations. `checks/` contains
native tests, API fixtures, source identity and comparison receipts. Counts
from overlapping suites must not be added together.

The [six-model wheel-image matrix](../prepared-b12x-serving/) remains a separate
immutable boundary. It is not relabeled as a measurement of every beta commit.
GPU packaging/import qualification of an automatically published image is
also distinct from complete model-serving qualification.
