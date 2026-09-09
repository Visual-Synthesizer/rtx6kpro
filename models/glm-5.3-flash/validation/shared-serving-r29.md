# Shared Jovian Judgement serving qualification: R29

Status: **qualified for the bounded serving, transfer and regression checks
specified here**. This report is also the historical changelog from R28.1.
It does not qualify every model, precision or parallel topology supported by
the installed source.

## Artifact and source boundary

Image: `localinferencelab/vllm:jovian-judgement-community-20260909-r29`.
Registry digest:
`sha256:e44e07e615287605f87bd4db916d683e39066e72a1ba94cf4149089c1ec21b49`.
Tested image ID:
`sha256:40f97a8a366f6cd8ede3ab5b97bbeb54cd0e7a5a222f07a8f27640015a08e1ec`.

The image has two root filesystem layers: a flattened runtime foundation and
one committed source installation. Its
[source lock](shared-serving-r29.source.lock), installed at
`/opt/glm53-flash/source.lock`, has SHA-256
`3307f3372213496e5b7de4fc485ef5b8f7fc43ff99df896ac40fa68d4dd3f80c`.

| Component | Complete source revision |
|---|---|
| vLLM | [`45361846d60622cb5211b902bc893963e5a9eaa6`](https://github.com/voipmonitor/vllm/tree/integration/jovian-shared-serving-20260909-r29) |
| B12X | [`3edbcbce70f491741b82f5eab9c1b30b39447228`](https://github.com/voipmonitor/b12x/tree/release/jovian-judgement-20260909-r29) |
| LMCache | [`dcd6ec92b23c7da14a46e0b9bf23a078969ddd4d`](https://github.com/local-inference-lab/LMCache/tree/release/jovian-judgement-20260909-r29) |
| FlashInfer | `803c4664f4771ddc418f20a57f752469a237a825` |
| FlashKDA source / compiled extension SHA-256 | `3b225bf26bb8e218928a1fe14751cb48cf31d11b` / `3bb6ef2c9f0be24b2c6ae8a48eb20b53f28e632ed9cb45a32c7c0133f8cfde91` |

The source lock also records the FlashKDA checkpoint patch, native vLLM
extension, Git bundle and launcher hashes. The displayed vLLM version is
`0.26.1rc0+glm53.r29.vllm45361846`; installed Git trees are clean.

## Changelog from R28.1

- One installed runtime serves GLM-5.3-Flash, Qwen3.8-Flash-Next and DeepSeek
  V4 text/Vision. Each model has its own backend, graph, precision and loading
  profile. DS4 dense projections use DeepGEMM without changing GLM's B12X path.
- GLM chat defaults retain high reasoning and add `clear_thinking=true`.
  Completed-turn reasoning is omitted from subsequent rendered prompts;
  visible responses, tool exchanges and active tool-cycle reasoning remain.
  Explicit request and CLI template overrides remain supported.
- Qwen defaults preserve BF16 target vocabulary projection and use a private
  NVFP4 W4A16 MTP projection. `VLLM_LM_HEAD_A16=1` is the image default.
- DS4 includes speculative graph-row warmup, NCCL selector cleanup,
  caller-owned padded-query output, attention-aware memory admission and
  model-specific Vision/DSpark launcher contracts. Text uses K5, Vision K3.
- LMCache paged gathers keep host block-ID metadata immutable until asynchronous
  copies consume it. Reusing pinned metadata could otherwise export the wrong
  GPU pages. Small transfers retain registered staging; larger transfers use
  one call-owned snapshot, without a global GPU synchronization.
- The clustered BF16 router uses consumer-only reduction barriers, complete
  cluster lifetime synchronization and a single owner for global output stores.
  This fixes independently reproduced divergent-barrier errors.
- Packaging returns to two layers instead of the R28.1 Python overlay over
  the two-layer R28 image. Complete Git mirrors preserve author history and
  integration resolutions. Docker publication uses `localinferencelab/vllm`.

The R28.1 fairness, request/SYSTEM checkpoints, optional aligned retention,
packed FlashKDA exports, atomic external bundles, FP32 routing protections,
speculative optimizations and full-and-piecewise graphs remain in the source.

## GLM output-degeneration reproduction

The supplied capture has 1,428 messages and 13 tool schemas. It identifies
GLM-5.3-Flash, not a GLM-5.6 checkpoint. Archive SHA-256:
`944b4b6c9abb21c181ac7d15ac0b94bf56e614987d1b5dbd185de2a179b45a38`.
All 32 member checksums match. The capture already contains degenerate
historical reasoning; its sanitizer also breaks 628 tool-call ID associations.
The installed template retains tool observations through ordered fallback.

With R28.1 and retained historical reasoning, three stock TP4/DCP1 DFlash2
cold requests each compute 839,815 input tokens. Two responses are ordinary;
one degenerates to 41,013 output tokens. Two no-spec controls are ordinary.
These samples do not establish a DFlash-specific kernel defect. No CUDA fault
or external cache-copy error is observed.

The [model author's chat guidance](https://huggingface.co/zai-org/GLM-5.3-Flash#note)
recommends clearing completed reasoning. The R29 profile renders 514,963
tokens from the same captured messages, removing 324,852 historical reasoning
tokens. Three independent cold replays return 43, 72 and 83 output tokens;
each ends with one schema-valid tool call, no SSE error and no detected
degeneration. Generated tool calls are validated as data, never executed.
Seven template/token-prefix contracts pass.

**Scope:** this qualifies the chat profile for the captured conversation. It
does not prove a matched-length 840K kernel repair or immunity to arbitrary
long-context degeneration. Omission of completed reasoning changes the token
prefix: an input endpoint can remain reusable, but a response checkpoint cannot
be reused past the first omitted reasoning token.

## GLM performance

Stock RTX PRO 6000 Workstation GPUs3/12/13/14; TP4/DCP1, FP8 target KV,
FlashKDA, 4096 scheduler tokens, OMP1, NCCL 16 channels/2 MiB and
full-and-piecewise graphs. C1 uses temperature 1, 15 s warmup, 30 s measured
and EOS-respecting context-zero requests. Prefill uses 12 cold nominal-32K
samples, about 32,315 actual tokens; rates are input/API-TTFT.

| Mode | C1 output R28.1 → integration tok/s | Verifier steps/s | Accepted length | 32K prefill tok/s |
|---|---:|---:|---:|---:|
| No-spec | 169.379 → 169.350 (−0.02%) | Same as output | Not applicable | 14,650 → 14,733 (+0.57%) |
| MTP3 | 258.019 → 265.185 (+2.78%) | 102.388 → 108.688 (+6.15%) | 2.520 → 2.440 | 14,279 → 14,320 (+0.29%) |
| DFlash2 K7 | 226.010 → 230.980 (+2.20%) | 89.404 → 89.551 (+0.16%) | 2.528 → 2.579 | 14,519 → 14,529 (+0.07%) |

Control image ID starts `259a592c`; measured integration image ID starts
`5df01ff6`. These performance cells precede the LMCache metadata and BF16-router
fixes. Those changes do not replace the measured GLM GPU-cache hot path.
They were qualified separately, not silently counted as repeated performance
tests. Host-side concurrent activity differs between some cells; the MTP rate
change is not causally attributed. These are bounded observations, not a
universal gain claim. All three mode smokes and exact 32K endpoint checks pass.

C8/C64 and Sieve are not rerun for R29. Their earlier observations, including
the short DFlash C8 decrease, remain in the
[R28.1 report](scheduler-serving-r28.1.md) and [R28 report](fp8-serving-r28.md).
No table in this report represents +6000 VRAM clocks.

## Qwen qualification

Stock GPU14, TP1/MTP3, FP8 KV, CPU PLE, BF16 target head and private NVFP4
W4A16 draft head. Three warmed C1 repeats have median output
173.661 → 177.268 tok/s (+2.08%) and verifier rate
85.214 → 85.367 steps/s (+0.18%). C8 output is
645.486 → 644.776 tok/s (−0.11%); verifier rate is
328.827 → 328.424 steps/s (−0.12%). The uncached 32K rate differs by less
than 1%. Cache and leading-instruction probes pass.

An initial C1 cell measured 194.081 → 184.319 tok/s (−5.03%) while verifier
rate rose 84.873 → 85.542 steps/s. It is retained, not discarded. Acceptance
variation and the small repeat count prevent a universal output-speed claim.
These measurements precede the independent metadata/router corrections.
Qwen TP2, LMCache and generated-code quality are not requalified by this check.
The [Qwen page](../../qwen38-flash-next.md) preserves its ten-run R28.1 Sieve
and engine-comparison results without relabelling them as R29.

## LMCache transfer correctness

The pinned-ID defect is reproduced with deferred CPU copies and native CUDA
delayed streams. Six CPU cases fail before the correction; 12 pass afterward.
Both native CUDA cases copy wrong pages before and exact bytes afterward.
The complete focused CPU suites pass 115 tests. The implemented correction is
included in [LMCache #50](https://github.com/local-inference-lab/LMCache/pull/50).

The metadata-corrected image `76d55eb6` passes GLM DFlash2/DCP4:

- 54K cold, GPU-prefix, RAM and restart-filesystem literal answers;
- an 11K shared SYSTEM prefix with different user continuations;
- exact comparison of 3,639,803,904 copied bytes in 576 transfers across all
  four ranks, with zero externally restored prompt tokens recomputed;
- three C8 cancellation/live-read eviction rounds.

The sidecar is CPU-only. Filesystem pages may remain in the OS page cache.
These checks do not repeat the inherited six-mode one-million-token timing
matrix or measure cold-device bandwidth.

**Migration:** use an empty external-cache volume or L2 directory. The fix
cannot repair payloads already written with wrong GPU-page metadata. Atomic
GLM checkpoint identities reject incompatible sources, but DS4 ordinary
filesystem keys do not automatically reject an image update. Do not delete
another service's cache. The deployment examples use distinct R29 volumes.

## DS4 Vision launch failure and router correction

The supplied C4/four-image streaming workload reproduces `CTA Not Present`
twice, after 259 s and 55 s. The second failure already includes the metadata
fix, demonstrating that the metadata correction alone does not resolve it.
Instrumented controls also complete successfully; these do not erase the
two production-clock failures.

The clustered BF16 router independently fails synccheck with 8,000
divergent-barrier errors on its first standalone warmup. The correction in
[vLLM #723](https://github.com/local-inference-lab/vllm/pull/723) replaces
consumer-only whole-block barriers, guarantees peer-CTA existence/lifetime,
and gives one CTA ownership of global output. BF16 operands, FP32 reduction
order and the asynchronous copy pipeline remain unchanged.

Synccheck and memcheck pass with exact products. The installed shape/API/graph
suite passes 136 tests, independently repeated in the baked R29 image.
Two 600 s Vision/LMCache controls with empty tiers complete 529 and 501 HTTP
200 requests, respectively, with zero HTTP errors and a healthy engine.
The 501-request control uses the final image without source mounts, exception
waits or overclocking. The supplied client does not independently validate
SSE error events or answer quality. No fault-PC trace was obtained, so the
barrier correction is not claimed to explain every possible launch failure.

**Sanitizer limitation under investigation:** racecheck reports operand-stage
hazards also reproduced by a CUDA-only four-stage cp.async/mbarrier program,
with no vLLM, CuTe, DSMEM or cache service. Scalar reads, matrix reads and
separately assembled cubins reproduce the report; generation-specific copied
values are exact. Single-stage checks pass. The inspected ordering satisfies
the documented mbarrier visibility contract, supporting a tracking limitation,
but vendor confirmation is absent. Those diagnostics remain failed reports,
not passing racecheck. No suppression or slower fallback is included.

The [DS4 runbook](../../ds4-jovian-community-r29.md) records text K5/Vision K3
rates, 810K admission, image/tool checks and exact all-eight-group RAM/restart
transfers. Long-duration filesystem-capacity pressure remains unqualified.

## Review and reproducibility

Source review is tracked in [vLLM issue #651](https://github.com/local-inference-lab/vllm/issues/651).
DS4 additions are #628/#630/#671/#679/#720, router synchronization is #723,
and LMCache metadata/transport identity are #50/#56. Earlier GLM cache,
scheduler and performance review units remain required where not merged.
The complete mirrors retain attribution and conflict resolutions. Original
router authorship from Roberto L. Castro's upstream vLLM #42562 is preserved.

Installed tests include 236 native/model/runner cases, 82 LMCache cases,
113 Qwen cases with five skips, seven DS4 profiling/functionalization cases,
30 narrow DS4 launcher contracts, and 49 recipe contracts. The independent
metadata and router suites above cover their separate corrections.

The [source-locked recipe](https://github.com/local-inference-lab/blackwell-llm-docker/tree/codex/glm53-source-locked-build/recipes/glm53)
rebuilds committed package trees and authenticated native inputs. Source-tree
reproducibility is required; byte-identical OCI timestamps across builders are
not promised. Qualification does not extend to NVFP4 target KV, TP8, pipeline
parallelism, arbitrary DSpark research checkpoints or untested model profiles.
