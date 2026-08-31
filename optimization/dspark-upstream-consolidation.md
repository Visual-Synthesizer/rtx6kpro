# DSpark: ours vs upstream vLLM — consolidation study (2026-07-03)

Upstream merged DSpark speculative decoding on 2026-07-01 ([#46995](https://github.com/vllm-project/vllm/pull/46995),
follow-up [#47093](https://github.com/vllm-project/vllm/pull/47093) speculators/reduced-vocab). Ours predates it by 3 days
(`279de1956f`, Luke, 2026-06-28 + TP4 TileLang follow-ups). **They are parallel implementations of the same
algorithm — disjoint git histories — that even share file paths** (`vllm/models/deepseek_v4/nvidia/dspark.py`,
`spec_decode/dspark/speculator.py`). This page: E2E + acceptance measurements, code-level comparison,
and the consolidation/port plan.

## 1. E2E results (DS4-Flash-DSpark, TP2, RTX PRO 6000)

| | ours (eldritch 3f65c52 + b12x f416b75) | upstream nightly (post-#46995) |
|---|---:|---:|
| decode cc1 (512 tok) | **188.1 tok/s** | **does not start** |
| coding gen-only (2k tok) | **278.0 tok/s** | — |
| acceptance / draft token | **36.2 %** (5,773/15,940) | — (see §2) |
| accepted per draft (+bonus) | 1.81 (+1 ⇒ ~2.81 tok/step) | — |
| per-position acceptance (of 3,188 drafts) | 73.9 / 48.7 / 30.4 / 17.9 / 10.3 % | — |

Ours config: seqs 64 / graph 512 / mbt 8192 / 128k len / A8 / `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}`.
Consistent with the 2026-07-01 baseline (cc1 191.3 on b12x2387, prod cache/len deltas).

### Why upstream shows "does not start": four independent SM120 blockers

Peeled one at a time on `vllm/vllm-openai:nightly` (5 boot cycles):

1. **DSpark crash at speculator init**: `AttributeError: 'DSparkDeepseekV4ForCausalLM' object has no
   attribute 'draft_id_to_target_id'` — their #47093 reduced-vocab refactor broke the DSV4 draft;
   fix sits in **open PR [#47429](https://github.com/vllm-project/vllm/pull/47429)** (we applied it manually to proceed).
2. **Bundled DeepGEMM has no SM120 support**: `Assertion error (deepgemm-src/csrc/apis/layout.hpp:59):
   Unknown SF transformation` during FP8 weight processing. (Our stack ships an SM120-patched DeepGEMM fork —
   this is precisely the delta.)
3. `VLLM_USE_DEEP_GEMM=0` fallback → **CUTLASS c3x `dispatch_scaled_mm` RuntimeError** — upstream has no
   SM120 fp8-blockwise scaled-mm kernels either; `VLLM_TEST_FORCE_FP8_MARLIN=1` does not reroute the
   blockwise path.
4. **tilelang wheel broken in the image**: `libcudart_stub.so: undefined symbol: cudaDeviceReset`
   (the DSV4 target needs TileLang for the HC head) — worked around by mounting the real libcudart over the stub.

**Conclusion: pure upstream vLLM cannot serve DeepSeek-V4-Flash (with or without DSpark) on SM120 today.**
Its DSpark speed on this hardware is therefore N/A; every layer that makes DS4 runnable here (SM120 DeepGEMM,
b12x kernels, working TileLang) lives in our stack. A fair speed comparison would need SM90/SM100 hardware.

## 2. Acceptance: theirs vs ours

E2E acceptance for upstream is unmeasurable on this box, but the code answer is solid:

- **Identical rejection scheme by default** — both trees share the same `_rejection_kernel`
  (greedy: accept iff draft == target argmax; temp>0: Leviathan ratio test with the same seeding
  granularity). At `num_speculative_tokens == dspark_block_size` the draft layouts are token-for-token
  identical ⇒ **same expected acceptance on the same checkpoint**.
- **Upstream likely LOSES acceptance when `N < dspark_block_size`**: they truncate the query block to N
  (shape never seen in training); we always run the full trained block and slice (costs a little compute,
  keeps fidelity).
- **Upstream GAINS acceptance at temp>0 via `rejection_sample_method="block"`** (block verification,
  #46781 + int64 fix #47383) — joint block acceptance with residual-mass reweighting. **We lack this
  entirely** — the single biggest acceptance feature gap on our side.
- Gumbel keying differs by one position offset (both self-consistent; second-order).
- Our batch-wide draft suppression (prefix-cache window rebuild ⇒ 0-width drafts for the whole batch;
  `len ≤ 1` ⇒ zero-token drafts for the whole batch) depresses *effective* tokens/step in mixed batches;
  upstream proposes per-request unconditionally.

## 3. Architecture comparison (condensed from three deep-dive analyses)

**Draft model.** Ours (1,994 lines): self-contained, private per-request dense rolling KV window
(no paged cache, layers popped from the forward context), bespoke **TileLang sparse block-attention
kernel** (E2E-tuned: TP4-native h=16/block=32/threads=128 → 376.6 tok/s, beating the isolated-bench
winner by 7% — another isolated-vs-E2E case), b12x fused WO/MHC, fake-FP8 QAT numerics matching
training, confidence head, env-gated reference/debug harness. Upstream (488 lines): thin wrapper reusing
the *target's* stack — stock decoder layers, **paged SWA draft KV** (`sparse_swa.py`, fp8_ds_mla layout),
fused RoPE/quant/paged-insert CUDA ops, FlashMLA/FlashInfer sparse kernels (SM120 path *requires* fp8 KV),
torch.compile on the draft.

**Speculator.** Upstream rewrote DSpark as a subclass of their DFlash parallel-drafting framework:
one fused Triton prep kernel writes all graph inputs (ids, positions, both slot mappings, padding) in
place; per-request proposals; `pad_spec_decode` keeps uniform decode graphs when requests join;
reduced-vocab d2t; EPLB hooks. Ours: standalone speculator with a per-step Python state machine for the
private window (`_update_context_cache_state`), FlashInfer MoE autotune pinned to the draft block bucket,
numerics hardening (fp64 acceptance uniforms, gumbel eps clamps), vectorized flatten kernel.

**Verdict:** upstream's *integration architecture* is better (paged draft KV, fused prep, per-request
robustness, ~10× less DSpark-specific code); **our *kernels and DSV4 fidelity* are better** (TileLang
draft attention with fewer indirections wins single-stream; b12x fusions; full trained block; QAT-exact
numerics). On this hardware ours is faster by construction — upstream's stack doesn't even run.

## 4. Port plan (consolidation)

**Upstream → ours (priority order):**
1. **Block verification** (`rejection_sample_method="block"` kernels + config, incl. #47383 int64 casts) —
   direct accepted-length win at temp>0, orthogonal to our drafter.
2. **Scheduler `pad_spec_decode`** — uniform-decode FULL graphs survive requests joining a spec batch.
3. Base-class `idx_mapping[num_reqs:].fill_(-1)` defense-in-depth (our active-row guard covers DSpark
   only; audit eagle/mtp/autoregressive for stale-row corruption).
4. Replace batch-wide suppression with per-request `-1` draft fill (and stop proposing token id 0 for
   `len ≤ 1` anchors).
5. Skip the dead confidence-head compute in `forward_head`.
6. Longer term: paged draft-KV/SWA-group design (prefix caching + preemption without the trusted-window
   state machine).

**Ours → upstream (if Luke wants to upstream):**
- TileLang draft attention as the SM120 path (their SM120 FlashInfer sparse path needs cubins that may
  not ship); gumbel eps clamps + finite guard; fp64 acceptance uniforms; vectorized `_flatten_sampled`;
  FlashInfer MoE autotune bucket for block-M; `requires_eagle_cache_drop` refinement; "run full trained
  block when N < block_size" (fidelity); the #47429 one-liner is already ours-equivalent.

**Kernel-level headroom on ours (draft pass):**
1. Kill the two per-layer window copies (`cache_window` gather + `torch.cat`) — kernel reads split
   (cache, draft) tensors, or allocate window+block contiguously and write the block into the tail.
2. Hoist `_build_dspark_topk_idxs` out of the per-layer loop (layer-invariant; or fold the analytic
   index into the kernel — no index tensor at all).
3. m-fusion: one CTA per request processing all 5 draft queries (5× fewer KV re-reads; also solves the
   TP8 h=8 padding waste).
4. Do NOT: fp8 window in-kernel (latency-bound, not BW), PDL/relaxed atomics (−7 % E2E, measured),
   block=64/FullCol (lost E2E by 7 %, measured).

## 5. Bugs found along the way

- **Ours**: `tests/v1/spec_decode/test_dspark.py` imports `_get_tilelang_block_size`/`_get_tilelang_padded_heads`
  deleted by `29516ba9af` — the test file fails at import; needs updating.
- **Upstream**: the #47429 crash (still open); nightly tilelang wheel broken stub; no SM120 story for
  DS4 across three kernel layers.


## 6. Block verification port — implemented, measured, verdict (2026-07-03 night)

Ported upstream's block verification into our tree (branch
`fable/dspark-block-verification-20260703` @ local-inference-lab/vllm): wholesale adoption of their
refactored `rejection_sampler_utils.py` (block kernels, int64-safe offsets) with our extras re-applied
(fp64 acceptance uniforms, active-row-guarded gumbel), `rejection_sample_method: "block"` config,
and a host-side **all-greedy skip** (the block prep kernels cost ~5–7 % of a step and are inert at temp 0
— without the skip, greedy decode lost 7 % tok/s).

**Testing strategy that made this fast** (2 boots total instead of ~10):
1. *Synthetic kernel harness* (`optimization/code/tiny-decode/synth_rejection_test.py`): drives
   `rejection_sample()` directly with controlled target/draft divergence. Seconds per iteration.
   Results: temp-0 equivalence exact; distribution preservation (TVD within sampling bounds);
   accepted length **+7–47 %** over standard across temp × draft-quality on Gaussian-noise drafts.
2. *E2E with proper statistics*: 12×800-token probes ×3 per config at temp 0.7 (small probes have
   ±9-point noise — trajectories are not run-to-run deterministic on this stack due to fp32-atomic
   logit jitter; and first-run-after-boot reads low, again).

**E2E verdict: no real-workload gain.** block 54.0/56.4/54.2 % vs standard 54.1/56.0/59.3 %
(means 54.8 vs 56.5, spread ±3) — the DSpark draft's error structure is bimodal
(agree-hard / diverge-hard), not the smooth likelihood-ratio spectrum where joint-ratio pooling wins.
The +5-point gain seen in first small probes was sampling noise. **Kept opt-in, not default.**
Value delivered anyway: consolidated rejection utils (upstream refactor + our hardening in one file),
the synthetic harness as a permanent regression tool, and the measured-noise methodology
(big paired probes or bust) for all future acceptance work.

Next speed items from §4 remain open: draft-pass window-copy elimination, topk hoist, confidence-head
skip, per-request suppression fix, pad_spec_decode.


## 7. Speedup campaign results (2026-07-03)

Profiled the step precisely first (CUDA-event instrumentation inside the graph-replay hot path —
torch profiler cannot see inside FULL cudagraphs): **step ≈ 13.6–14.9 ms = target verify M=6
~11.3 ms (83 %) + draft graph 2.01 ms + prep 0.33 ms.** The verify pass is near its
weight-streaming floor (~33 distinct experts × 6.3 MB × 61 layers), which reframed the campaign:

| lever | result |
|---|---|
| k-sweep (5→3→2) | **falsified**: 251 / 218 / 192 tok/s greedy — verify scales weakly with M, shorter blocks just lose tokens |
| A16 MoE for DSpark | tie (244.5 vs 241.3) — the 0701 +9 % edge no longer reproduces on the f416b75 stack |
| draft cleanup (confidence-head skip, per-step topk sharing, persistent window+block staging) | **landed**: draft graph-forward 2.01 → **1.81 ms/step** (−10 % draft, ~+1.4 % E2E), coherence clean; commit `16140bcb` |

The timing instrumentation (`VLLM_DSPARK_TIMING=1`) stays as the permanent per-change A/B tool.
Remaining ideas are structural (fp8 draft window, cross-step expert dedup) with uncertain payoff.


## 8. Full-B12X vs Lucifer CUTLASS on DSpark (2026-07-03)

Paired fresh boots, warmed, cc64 measured twice per side (first-run cc64 is unreliable — FI autotune
cold start measured 572→1927 on the hybrid; bench order matters too: cc64 after a 128k prefill sweep
reads −15–20 %):

| DSpark TP2 A8, k=5 | full B12X (f416b75) | Lucifer CUTLASS | winner |
|---|---:|---:|---|
| decode cc1 | 197.3 | 202.2 | ~tie (+2.5 % Lucifer) |
| **decode cc64 aggregate** | 1,925 (30.1/user) | **2,331 (36.4/user)** | **Lucifer +21 %** |
| prefill 8k / 64k / 128k | **12,972 / 12,875 / 12,067** | 12,201 / 12,194 / 11,314 | **B12X +6 %** |

Hybrid (B12X attention + FI MoE) ≈ B12X at cc64 (1927) → the cc64 gap isn't MoE alone; Lucifer's
attention path contributes at batch. **Serving guidance:** throughput deployments → Lucifer config;
prefill/TTFT-heavy → full B12X. **Kernel work item for the b12x stream:** the dynamic w4a8 MoE band at
M≈64–384 (spec-decode verify batches) trails FI CUTLASS MXFP4×MXFP8 by enough to cost −21 % E2E at cc64.
