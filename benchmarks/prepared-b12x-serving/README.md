# Wheel-composed model runtime qualification

Status: qualified for the six bounded serving comparisons and Qwen's separate
automatic-capacity checks below.

## Artifact and source contract

Measured image:
`sha256:dada56080b813ffcd38b18f4d3dc7c548058cfe2222fe3e5a49e2893cd674b92`.

Its local tag is
`ghcr.io/local-inference-lab/vllm:jovian-judgement-beta-20260917-b34094affbef6a01`.
This image has not been published by the qualification run.

The image contains vLLM `1048fc5439d693d99e73cb7f5f09b0a06ee70b6c`, B12X
`d9b572754a8a81888b355523888b778fe73e1b6b`, and Docker recipe
`24b82c2d9009ddfb97c617c679d7cf450c09b473`, on CUDA 13.4.1/PyTorch 2.14.
It has 68 filesystem layers. Serving uses image-owned production code without
source-code mounts.

Pushed vLLM integration `8edf83d96ec488f3bef7340952afdfa4cff020d7` differs
from the image source only in an attention workspace test. Production sources
match. Both repositories publish these fixes through `integration/beta`. Public heads
of vLLM PRs #780, #782–#786 and B12X PRs #379–#380 are ancestors of those
pushed branches; [issue #773](https://github.com/local-inference-lab/vllm/issues/773)
explains each fix. This does not qualify arbitrary
rebuilds with different dependencies.

Native packaging passes 22 CLI, seven import/operator and 91 LMCache tests.
Focused prepared-workspace suites pass 44 tests from standalone PR #786 and
48 from integration. Separate native attention, GDN, collective, memory,
grammar and timer evidence is recorded in the PRs. Counts from overlapping
suites must not be added together.

## Measurement conditions

Stock RTX PRO 6000 Workstation GPUs 4–7; each reference/image pair uses the
same physical cards and checkpoint files. Decode is the median of three
30-second runs at context zero and temperature 1, each after 15 seconds of
warmup. Errors, detected loops, underfilled concurrency, missing cells and
warmup timeouts invalidate measurements.

Benchmark source is [llm-inference-bench v0.6.2](https://github.com/local-inference-lab/llm-inference-bench/blob/ccd9ad8ced7e387794391bfb0ac6d99b1f66ba6f/llm_decode_bench.py),
SHA-256 `053989edff8c9c93e2b96e61342b2ffbd9851e03deba17e6d3fc96fcd6694c1e`.
The published Git blob matches the benchmark used for both arms.

32K prefill uses uncached 32,770-token requests in one sustained 30-second
window after warmup. It measures input tokens divided by client TTFT, not
pure GPU prefill. Small changes from a single window do not establish a
general speedup. Functional checks cover arithmetic, repeated/changed-answer
prefix requests and image interpretation for vision-enabled profiles.

All profiles use FP8 target KV and full-and-piecewise graphs. GLM uses runner
V2 and B12X PCIe all-reduce. Two-shot is off. Scheduler budget is 4,096 tokens
except Qwen's 6,019. Reduced-NCCL-channel memory workarounds are not used.

Reference images:

- GLM/Qwen R35:
  `localinferencelab/vllm@sha256:7a425c6864b951bbc368111490a4b0ac69d8cd0dd4987075b2c7d40b753b1bf5`.
- DS4.1 R38:
  `localinferencelab/vllm@sha256:f41ca8bb10bb3a125a50340d70d39ad4b7f5605f3fcb661bc992ed0bc4701a00`.
- DS4 text/Vision R9:
  `voipmonitor/vllm@sha256:5bea088597980b299a1df8a6f3fc6d2d22c723276088ea8583b456f27043c0cd`.

## Completed whole-image comparisons

Status: qualified within the conditions above. Rates are tok/s; concurrency
throughput is aggregate. Vision uses C4; all other profiles use C8.

| Serving profile | C1 reference → image | Change | Concurrent reference → image | Change | 32K prefill reference → image | Change |
|---|---:|---:|---:|---:|---:|---:|
| GLM MTP3, TP4/DCP1 | 247.12 → 249.33 | +0.89% | 872.67 → 875.86 | +0.37% | 15,332 → 15,580 | +1.62% |
| GLM DFlash2 K7, TP4/DCP1 | 211.23 → 219.83 | +4.07% | 673.26 → 717.84 | +6.62% | 15,538 → 15,734 | +1.26% |
| DS4.1 DSpark K7, TP4 | 250.87 → 256.59 | +2.28% | 808.17 → 830.82 | +2.80% | 20,079 → 20,234 | +0.77% |
| DS4 text DSpark K5, TP2 | 191.35 → 190.12 | −0.64% | 653.53 → 669.56 | +2.45% | 13,527 → 13,863 | +2.48% |
| DS4 Vision DSpark K3, TP2/C4 | 169.92 → 185.49 | +9.17% | 411.49 → 423.11 | +2.83% | 10,388 → 10,615 | +2.19% |
| Qwen MTP3, TP1/eight-GiB KV | 172.92 → 190.11 | +9.94% | 664.89 → 689.93 | +3.77% | 15,387 → 15,162 | −1.46% |

All API checks and six decode cells per profile pass. DS4.1 also passes image
input. GLM target sampling is temperature 1/top-p .95. DFlash2 uses the
identical MXFP8 draft with BF16 draft KV. DS4.1 uses adaptive DSpark, CPU
Engram, a 131,072-token context limit and target temperature 1/top-p .95;
its internal draft sampling is greedy.

| Serving profile | C1 request-verifier steps/s, reference → image | Concurrent summed request-verifier steps/s, reference → image | Logical KV tokens, reference → image |
|---|---:|---:|---:|
| GLM MTP3 | 99.49 → 100.25 | 349.28 → 350.86 | 3,645,144 → 4,224,149 |
| GLM DFlash2 K7 | 81.17 → 85.15 | 260.55 → 277.35 | 3,676,628 → 4,519,268 |
| DS4.1 DSpark K7 | 94.88 → 100.16 | 352.17 → 361.56 | 4,503,190 → 4,714,406 |
| DS4 text DSpark K5 | 71.92 → 73.31 | 244.46 → 250.94 | 1,192,983 → 1,293,619 |
| DS4 Vision DSpark K3/C4 | 79.50 → 87.48 | 190.78 → 196.05 | 1,215,925 → 1,285,174 |
| Qwen MTP3/eight-GiB KV | 81.90 → 83.47 | 322.23 → 333.28 | 517,581 → 517,581 |

DS4 text uses temperature 1/top-p 1, TP2 and a configured 1,048,576-token
context. Its three C1 samples span 188.73–193.24 tok/s versus 183.53–199.20
for R9. The −0.64% median output difference is retained; accepted length is
2.593 versus 2.662 and verifier throughput is +1.92%. These observations do
not establish an attention-kernel slowdown. Four API checks and six decode
cells pass.

DS4 Vision uses temperature 1/top-p 1, TP2 and the same configured context
limit. Five API checks, including image input, and six decode cells pass.
Its C1 range is 185.04–192.78 tok/s versus 165.67–174.26 for R9.

Qwen uses TP1/MTP3, CPU PLE, eight GiB of explicit KV memory, a 262,144-token
context limit, 16 maximum requests and temperature 1/top-p .95/top-k 20.
Four API checks and all six decode cells pass. C1 ranges are 180.47–198.09
versus 163.73–184.33 tok/s. Accepted length rises from 2.105 to 2.280, while
verifier throughput rises 1.92%; the 9.94% output difference is not an isolated
kernel gain. The −1.46% prefill difference is retained. One prefill window
does not establish its statistical significance or demonstrate zero regression.

The R9 DS4 text receipt stores its three C8 runs separately as
`decode-c8-supplement-*.json`: the initial sweeps skipped C8 because physical
hybrid-cache blocks undercount logical token capacity. The separate runs use
the engine's logical capacity and unchanged image/settings. All six valid
cells are mandatory; no failed or lower-throughput samples are discarded.

Five completed Sieve requests give medians of 325.75 versus 326.15 tok/s for
GLM MTP3 and 461.42 versus 458.14 for DFlash2 (image versus reference).
All Sieve requests finish normally. Stochastic acceptance affects output;
read output and verifier rates together. These are complete-configuration
comparisons, not isolated kernel gains or broad model-quality evaluations.

## Automatic KV sizing and scope

Qwen TP1/MTP3 with CPU PLE also passes full-and-piecewise graph capture and
all four API checks without an explicit KV-memory override. The same image
admits **773,216 logical KV tokens**. This is a startup/functionality check,
not another throughput sample. The measured text profile does not qualify
Qwen vision.

Configured admission is not an actual million-token request test.
Source-mounted DS4 diagnostics are retained in
[the workspace reservation PR](https://github.com/local-inference-lab/vllm/pull/786), not substituted for
image-owned serving evidence.

## Evidence layout and reproduction

- [comparison.json](comparison.json): validated medians, ranges, acceptance,
  verifier rates and relative raw-data locations for all six profiles.
- Each model directory contains `reference/` and `image/` records. Decode,
  prefill and Sieve JSON files are the benchmark outputs. Their semantic
  contents are unchanged. `functional-checks.json` retains assertions and
  response text but omits repeated synthetic request prompts.
- `runtime.json` records the actual server arguments, environment, device
  mapping, image identity and mounts. `launch.json` additionally records
  checkpoint metadata and shard file identities. GLM DFlash2's reference has
  a runtime record but no separate launch receipt.
- `test-commands.json` records exact benchmark commands and exit codes.
  Local mount/cache paths must be adapted to the reproduction host. Image
  IDs are local identities, not registry pull digests.
- [assembly.json](assembly.json) and [runtime-manifest.json](runtime-manifest.json):
  source locks, component wheel artifacts and dependency identities.
- [source-ancestry.json](source-ancestry.json) verifies public PR heads against
  both pushed integration branches and production identity with measured code.
- [image-source-identity.json](image-source-identity.json) compares the changed
  installed package files with their recorded source revisions.
- [native-packaging.json](native-packaging.json) covers CLI/import/operator
  and LMCache package tests. It does not prove model-serving or LMCache
  model-level roundtrip behavior.
- [gpu-offsets.json](gpu-offsets.json) records stock clock offsets.
- [Qwen automatic-capacity records](qwen/automatic-capacity) cover a separate
  no-KV-override startup and functional test.

Use the linked benchmark revision above. A representative decode invocation
for an already-running dedicated endpoint is:

```sh
python llm_decode_bench.py \
  --host http://127.0.0.1:5058 --model GLM-5.3-Flash \
  --display-mode plain --no-hw-monitor --no-resume --respect-eos \
  --temperature 1 --token-targeting exact --max-tokens 8192 \
  --decode-warmup-seconds 15 --skip-prefill \
  --contexts 0 --concurrency 1,8 --duration 30 \
  --kv-budget 4519268 --output decode-1.json
```

Run three times using separate output files. Select the served model,
concurrency and logical KV budget from that arm's runtime and functional
records; the example uses the measured GLM DFlash2 image. For prefill, replace
`--skip-prefill --contexts 0 --concurrency 1,8 --duration 30` with
`--prefill-only --prefill-contexts 32k --prefill-duration 30 --prefill-metric client`.
The benchmark warms and drains cells and checks exact repetition by default.
Those checks do not establish semantic output quality.

The measured image identity describes a local build, not a registry pull
digest. Automatic GHCR publications have their own immutable
[release manifests](https://github.com/local-inference-lab/blackwell-llm-docker/releases)
and packaging qualification. Do not infer a published image's complete-model
performance from a source or tag name alone.

[Prepared-kernel/API contract checks](../prepared-b12x-contracts/) record the
separately measured B12X reconciliation, PCIe lookup, breakable-prefill and
multimodal request boundaries.
