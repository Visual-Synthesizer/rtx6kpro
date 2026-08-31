# B12X Dense FP8 GEMM vs DeepGEMM — DSV4 Prefill Gap: Full Analysis & Optimization Log

Complete engineering log of the 2026-07-02 session that closed the DeepSeek-V4-Flash TP2
prefill gap between the full-B12X stack and the Lucifer/FlashInfer+DeepGEMM stack, and the
NCU deep-dive into the one component still slower than its DeepGEMM counterpart: the
**b12x dense MXFP8 GEMM at prefill M**. Written so the kernel work can be picked up with
zero re-derivation — every hypothesis here is either implemented+measured or falsified
with the exact error.

> **State at the end of the session:** the full-B12X DS4 TP2 A8 stack beats Lucifer CUTLASS
> at every context length. The absolute fastest config today is the **hybrid** (B12X
> attention + B12X A8 MoE + b12x indexer + **DeepGEMM dense linear**): config-only on the
> stock image, no patches. The only remaining B12X-vs-DeepGEMM deficit is the dense GEMM
> mainloop at M≥4096 (1.09–1.21× per shape) — instruction count, tiles, and SF loads are
> all **proven not to be the cause**; the gap is mainloop latency structure (§6).

- vLLM: `dev/eldritch-enlightenment@3f65c52`, B12X: `master@77bd50e`
- Image: `voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x77bd50e-overlay-cu132-20260702`
- HW: RTX PRO 6000 Blackwell Workstation (SM120, 188 SMs), TP2 pairs, PCIe
- All test servers: `max_num_seqs=1`, `max_cudagraph_capture_size=4`, fp8 KV, block 256,
  mbt 8192, chunked prefill + prefix caching, B12X PCIe one-shot allreduce

## Table of Contents

- [1. End-to-end results](#1-end-to-end-results)
- [2. How the gap decomposed](#2-how-the-gap-decomposed)
- [3. Fix #1: the MXFP8 input quantizer (merged into PR, 8–35×)](#3-fix-1-the-mxfp8-input-quantizer)
- [4. Fix #2 (vLLM): dual-path linear routing](#4-fix-2-vllm-dual-path-linear-routing)
- [5. Side experiment: DeepGEMM mqa-logits for the b12x indexer prefill](#5-side-experiment-deepgemm-mqa-logits-for-the-b12x-indexer-prefill)
- [6. The open item: dense GEMM mainloop at prefill M](#6-the-open-item-dense-gemm-mainloop-at-prefill-m)
  - [6.1 GEMM-only head-to-head across M](#61-gemm-only-head-to-head-across-m)
  - [6.2 NCU: speed-of-light](#62-ncu-speed-of-light)
  - [6.3 NCU: instruction mix](#63-ncu-instruction-mix)
  - [6.4 NCU: warp stall profile](#64-ncu-warp-stall-profile)
  - [6.5 Implemented: SF stage-bulk copies + SFB k-reuse (bit-exact, null perf result)](#65-implemented-sf-stage-bulk-copies--sfb-k-reuse)
  - [6.6 Falsified levers](#66-falsified-levers)
  - [6.7 What DeepGEMM does differently — rewrite spec](#67-what-deepgemm-does-differently--rewrite-spec)
- [7. Recommended configs](#7-recommended-configs)
- [8. Repro guide](#8-repro-guide)
- [9. Artifacts](#9-artifacts)

---

## 1. End-to-end results

DS4-Flash TP2, force-A8 MoE, standalone prefill (client tok/s), all measured 2026-07-02 on
the same image and machine, seqs=1/graph=4:

| # | Config | 8k | 64k | 128k | decode cc1 ITL |
|---|---|---:|---:|---:|---:|
| 1 | B12X full (start of day) | 12,468 | 11,971 | 11,133 | 7.66 ms (130.5 t/s) |
| 2 | Lucifer CUTLASS (target) | 13,442 | 12,622 | 11,716 | — |
| 3 | B12X full + quant fix (b12x `a67e5bd`) | 13,441 | 12,940 | 11,948 | 7.72 ms |
| 4 | #3 + vLLM dual-path (`9dbad237`) | 13,556–13,684 | 13,034–13,175 | 12,037–12,151 | 7.72 ms (129.6) |
| 5 | **Hybrid: B12X attn+MoE+indexer, DeepGEMM linear** | **13,811** | **13,201** | **12,182** | **7.48 ms (133.7)** |

- #3 = pure `--linear-backend b12x` stack with only the quant kernel fixed: **parity with
  Lucifer at 8k, +2.5%/+2.0% at 64k/128k** (was −7.2/−5.2/−5.0%).
- #5 needs **no patches at all**: drop `--linear-backend b12x`, set `VLLM_USE_B12X_FP8_GEMM=0`;
  `DeepGemmFp8BlockScaledMMKernel` gets selected for `Fp8LinearMethod`. Everything else
  stays B12X (attention `B12X_MLA_SPARSE`, `--moe-backend b12x` A8, b12x sparse indexer,
  PCIe one-shot AR).
- A16 decode was separately verified unregressed: coding 140.51 t/s vs reference 140.61
  (ITL 7.136 vs 7.129 ms). A16 is the decode-optimal MoE mode, A8 the prefill-optimal one —
  unchanged trade-off.
- Coherence: 30k-ctx `test.py` smoke clean (CJK 0) on every config above.

## 2. How the gap decomposed

Matched torch traces (identical 112,007-token prompt, `--profiler-config.*`, no stack
capture) over a 4-step window, rank0, B12X-full-A8 (with mqa indexer variant) minus
Lucifer:

| Component | B12X | Lucifer | Δ |
|---|---:|---:|---:|
| MLA prefill kernel | 339 ms | 411 ms | **−72 (B12X faster)** |
| MoE (B12X A8 vs FI CUTLASS MXFP4×MXFP8) | 458 ms | 567 ms | **−109 (B12X faster)** |
| indexer logits+topk | 190 ms | 234 ms | −44 |
| NCCL AR | 553 ms | 564 ms | −11 |
| mHC | 286 ms | 248 ms | +38 |
| dense GEMMs | 406 ms | 323 ms | **+83** |
| **dense-linear input quant** | **229 ms** | **14 ms** | **+215** |

Traces: `/root/bench-results/vllm-profile/ds4-{mqa2048,lucifer}-prefill-g4-20260702/`.
Both stacks are >97% GPU-busy — the gap is pure kernel time, not CPU launch overhead
(despite b12x's larger Python event counts under profilers).

**B12X attention and B12X A8 MoE already beat their Lucifer counterparts.** The whole
deficit was the dense-linear path: the activation quantizer (fixed, §3) and the GEMM
mainloop at large M (open, §6). The mHC +38 ms is a minor follow-up.

## 3. Fix #1: the MXFP8 input quantizer

`_quantize_dense_tk_to_tk_kernel` (`b12x/gemm/block_fp8_linear.py`) launched with grid
`(tokens, K/32)` — one CTA per 32-element scale group. At M=8192, K=4096 that is **1,048,576
CTAs per launch**, each loading 64 B and storing 34 B. Cost per 4-step prefill window:
229 ms vs 14 ms for DeepGEMM's per-token-group kernel.

Fix (b12x commit `a67e5bd`, branch `fable/dense-quant-tiled-20260702`): retile to
`BLOCK_M=32 × GROUPS=16` supergroups per program with masked edges, vectorized loads/stores
including the swizzled `scale_mma` layout (for aligned 32-token tiles the swizzle
coordinates `row32/row4/tile_m` become per-program constants).

| K (M=8192) | old | new | speedup |
|---|---:|---:|---:|
| 1024 | 122.5 µs | 11.9 µs | 10.3× |
| 1536 | 182.3 µs | 11.8 µs | 15.5× |
| 2048 | 242.5 µs | 11.7 µs | 20.8× |
| 4096 | 483.6 µs | 13.7 µs | 35.2× |
| 8192 | 965.5 µs | 118.1 µs | 8.2× |

**Bit-exact** vs the old kernel *and* an independent torch reference (values, `scale_rows`,
swizzled `scale_mma`), including decode shapes M=1/4 and edge cases M=5/33/100/8191,
K=1152. E2E effect: row #1 → row #3 of the table in §1 (+7.3–8.1% prefill), decode
unchanged. DS4 real dense-linear Ks are 1024/4096, i.e. the strongest rows.

## 4. Fix #2 (vLLM): dual-path linear routing

vLLM branch `fable/b12x-linear-dg-prefill-route-20260702` (commit `9dbad237`, pushed to
`local-inference-lab/vllm`). `B12xFp8BlockScaledMMKernel` keeps the b12x pack **and** runs
the DeepGEMM weight processing on the original checkpoint params (replaced in place — no
extra memory; the b12x pack always was a duplicate). Inside the opaque custom op, token
batches ≥ `VLLM_B12X_FP8_LINEAR_DG_PREFILL_MIN_TOKENS` (default 2049 = measured crossover,
§6.1) route to `DeepGemmFp8BlockScaledMMKernel.apply_weights`; smaller batches stay on the
b12x GEMM. Kill switch `VLLM_B12X_FP8_LINEAR_DG_PREFILL=0`.

Result = row #4: within 0.2–0.9% of the hybrid. The residual is inductor fusion of the DG
input quant into the compiled graph, which an opaque-op route cannot get, plus the small-M
chunks that stay b12x. **Once the mainloop work in §6.7 lands, this routing becomes
obsolete** — that is the desired end state.

## 5. Side experiment: DeepGEMM mqa-logits for the b12x indexer prefill

vLLM branch `fable/b12x-indexer-prefill-mqa-logits-20260702` (commit `b3fa995`, pushed).
`VLLM_B12X_INDEXER_PREFILL_MQA_LOGITS=1` reroutes DSV4 compressed-indexer prefill chunks
(logical top-k contract only; GLM's physical-slot contract untouched) to DeepGEMM
`fp8_fp4_mqa_logits` + `top_k_per_row_prefill` while decode keeps the b12x paged indexer.

- On the *old* full-b12x config it gained +1.6–1.9%.
- On the DG-linear config the stock b12x supertile indexer is equal or slightly better
  (−0.6%), thanks to Luke's `0d46e58`/`7045cc4`/`77bd50e` large-context indexer work.
- **Not part of any recommended config** — kept for reference.
- Gotcha discovered on the way: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=512` (default) is a
  pathological local minimum for that path at >64k ctx (128k dropped to 10,847 t/s);
  256 and 2048 both avoid it, 2048 best.

## 6. The open item: dense GEMM mainloop at prefill M

Everything in this section is about `DenseGemmKernel` (`b12x/gemm/dense.py`, CuTe DSL,
MXFP8: `mma.sync…mxf8f6f4.block_scale.scale_vec::1X.m16n8k32…ue8m0`, tile (64,128),
tile_k=128, mma_k=32, 8 MMA warps + 1 DMA warp, `ab_stage=3` (smem-capped), occupancy 1)
vs DeepGEMM `sm120_fp8_fp4_gemm_1d1d_impl` (384 threads, same 188-CTA persistent launch).

### 6.1 GEMM-only head-to-head across M

Graph-replay timed, L2-flushed, DSV4-Flash TP2 dense shapes, `b12x/dg` ratio
(>1 = b12x slower). From `benchmarks/benchmark_dense_fp8_vs_deepgemm.py`:

| M | qkv_a (1536×4096) | q_b (16384×1024) | wo_a (1024×4096) | wo_b (4096×4096) |
|---:|---:|---:|---:|---:|
| 2 | 0.96 | **0.56** | 0.88 | **0.70** |
| 8 | 1.00 | **0.56** | 1.00 | **0.67** |
| 32 | 1.00 | **0.59** | 0.91 | **0.76** |
| 128 | 1.00 | **0.69** | 0.99 | **0.70** |
| 512 | **0.75** | **0.82** | 0.92 | 1.00 |
| 1024 | 1.00 | 0.93 | **0.75** | 0.90 |
| 2048 | 0.90 | 1.05 | 1.06 | 1.13 |
| 4096 | 1.05 | 1.10 | 0.90 | 1.13 |
| 8192 | 1.09 | 1.17 | 1.15 | **1.21** |

b12x **wins decisively at decode/small M** (down to 0.56×) and loses from M≈2048 up. The
crossover is why the vLLM dual-path threshold is 2049. Absolute numbers at M=8192:
qkv_a 204.3/187.0 µs, q_b 524.3/448.5, wo_a 141.3/122.8, wo_b 495.6/409.6.

Caveat: in-graph E2E decode ITL is nevertheless *slightly better* with DG linear
(7.48 vs 7.72 ms) even though the isolated microbench favors b12x at small M — worth its
own look (PDL / graph-context interaction), not investigated further.

### 6.2 NCU: speed-of-light

wo_b, M=8192, N=K=4096, same operands, `--set basic`:

| metric | b12x | deep_gemm |
|---|---:|---:|
| duration | 534–541 µs | 446–448 µs |
| SM (compute) throughput | 72.4–73.3% | **87.9–88.1%** |
| L1/TEX throughput | 73.6–73.8% | 49.0% |
| L2 throughput | 71.9–72.7% | 60.1–60.4% |
| DRAM | 8% | 9.3% |
| block size / grid | 288 / 188 | 384 / 188 |
| registers/thread | **84** | **168** |
| theoretical / achieved occupancy | 18.75 / 18.74% | 25 / 18.73% |

The entire 1.21× is the SM-throughput ratio (88/73). b12x co-saturates L1 with compute;
dg leaves L1 at 49% and keeps the tensor pipe full.

### 6.3 NCU: instruction mix

`sass__inst_executed_per_opcode`, one launch:

| opcode | b12x (original) | b12x (after §6.5) | deep_gemm |
|---|---:|---:|---:|
| **total** | **167.8M** | **151.4M** | **91.2M** |
| QMMA | 33.6M | 33.6M | 33.6M |
| LDS | **37.7M** | 9.4M | 1.3M |
| LDSM | 21.0M | 21.0M | 12.6M |
| LOP3 | 12.3M | 16.0M | 5.3M |
| MOV | 11.4M | 13.0M | 4.8M |
| IADD/SHF/IMAD/PRMT/… | ~30M | ~32M | ~15M |

Identical QMMA count — all the difference is operand delivery and glue. The 37.7M LDS were
per-byte scale-factor loads: the SF copy atom is `CopyUniversalOp` at `sf_dtype` width
(8 bits) over `filter_zeros` views, issued per k_block (4× per k-tile), for both SFA and
SFB, per (mt,nt) atom.

LDSM arithmetic: 5 per warp per k_block (A m16×32 = 1×LDSM.x4, B n64×32 = 4×LDSM.x4) —
already minimal for the (64,128)/(4,2,1) geometry. dg's 12.6M over its geometry = **half
the per-byte smem re-read amplification** (bigger effective register blocking).

### 6.4 NCU: warp stall profile

`smsp__average_warps_issue_stalled_*_per_issue_active.ratio` (cycles per issued
instruction):

| stall | b12x | deep_gemm |
|---|---:|---:|
| math_pipe_throttle | 2.98 | **6.44** |
| wait (fixed latency) | 2.54 | 4.92 |
| long_scoreboard | 0.85 | 0.42 |
| short_scoreboard | 0.38 | 0.19 |
| not_selected | 0.24 | 0.43 |
| barrier | 0.04 | 0.34 |

Read together with issue counts (227k vs 115.5k issued/scheduler): dg's warps mostly wait
because the **tensor pipe is full** (the ideal state); b12x's tensor pipe has bubbles and
its schedulers additionally spend ~112k extra cycles/scheduler issuing the 2× instruction
stream. Absolute stall cycles are nearly equal — the delta is extra issue cycles *plus*
worse QMMA density in the pipe.

### 6.5 Implemented: SF stage-bulk copies + SFB k-reuse

b12x branch `fable/dense-sf-hoist-20260702`, commit `49b453c` (on top of `a67e5bd`).
Two **bit-exact** mainloop changes:

1. **Stage-bulk SF copies** — SFA/SFB register fragments for the whole acquired pipeline
   stage load in one bulk `cute.copy` instead of per-k_block slices.
2. **`sfb_k_replicated` flag** (plumbed `dense_gemm` → compile key → kernel;
   `block_fp8_linear` passes it): DSV4 weight scales are 128×128 blocks expanded to per-32
   replicas, so within a 128-wide k-tile the four SFB bytes are identical **by
   construction**. Load one byte per stage; every k_block's QMMA reads the same register.
   Guarded off under `swap_ab` (B slot holds activations there).

Validated: byte-identical outputs vs the unmodified kernel on packer-built weights
(`pack_fp8_block_scaled_weight_mxfp8` from `per_block_cast_to_fp8` checkpoint-style
scales), all four shapes, M=8192; plus Luke's own test files (see PR).

**Honest result: LDS 37.7M → 9.4M, total instructions −16M, wall-clock UNCHANGED
(SM busy stays ~73%).** This falsifies "instruction count is the limiter" directly. The
change still lowers L1 LDS pressure 4× for free and is the prerequisite for any future
SF-operand restructuring, but it does not close the gap by itself.

### 6.6 Falsified levers

Every accessible knob, each measured on the DS4 shapes at M=8192:

| Lever | Result |
|---|---|
| `mma_tiler_mn` sweep (11 candidates, M=4096/8192, all four shapes) | (64,128) already optimal everywhere; the `expected_m>128` regime already selects it (incl. narrow-N, where the no-hint default (64,64) would be 1.18× worse — vLLM passes `expected_m=tokens`, so not an issue in serving) |
| `tile_k=256` | does not compile: `error: expects 'coord' and shape of view are weakly congruent … tCrSFA_copy_view_filtered` (SF layouts are written for tile_k=128) |
| `tile_k=64` | does not compile: `Expected size in shape to be strictly positive, but got 0` |
| `use_prefetch=True` | no effect (202.8 → 204.8 µs) |
| `atom_shape (2,4,1)` (the "cooperative kNWarps×kMWarps" idea from the earlier port doc) | CuTe DSL crash: `std::bad_variant_access — Unexpected index` (SM120 helpers don't support N-warp counts ≠ 2) |
| SF operand pre-binding outside the k loop (per-nt atom objects bound once per work tile; incl. a gran-128 activation-quant mode to make SFA k-invariant too) | **MLIR-illegal**: `'cute_nvgpu.atom.set_value' op using value defined outside the region` — `set` must dominate its `gemm` use within the same region. This is also the deeper reason the "Manual atom unroll: avoids hasAuxTensor address space bug" workaround exists. Implemented, then reverted. Warning: without NCU attached, the MLIR verifier did NOT flag it and the kernel silently produced (correct) results — do not trust non-verified builds of such changes. |
| deeper pipeline (`ab_stage`) | smem-capped at 3 for (64,128): epi 16KB + ~25KB/stage in ~100KB; cannot deepen without tile_k or epi changes (both blocked above) |
| occupancy 2 CTA/SM | not viable: stages would drop to ~2/CTA |

Side finding for the quant/format axis (from the earlier port doc §8 and re-confirmed):
gran-32 vs gran-128 activation quant is numerically neutral in realistic regimes; a
gran-128 mode of the new tiled quantizer (one UE8M0 per 128-elem window, emitted per-32
for layout parity) was built and works — it only pays off together with the (illegal)
pre-binding, so it was reverted with it. Recoverable from this page + the journal if the
rewrite in §6.7 makes SF operands per-k-tile-invariant useful again.

### 6.7 What DeepGEMM does differently — rewrite spec

The remaining 1.09–1.21× at M≥4096 is **mainloop latency structure**, evidenced by:
equal QMMA, near-equal absolute stall cycles, 2× register footprint on dg (168 vs 84),
half the LDSM amplification, and dg's dominant stall being math_pipe_throttle (pipe full).

Concretely, to reach/beat dg the b12x mainloop needs:

1. **K-tile-deep register blocking**: hold A and B fragments for the *whole* 128-wide
   k-tile (4 k_blocks) in registers, double-buffered across tiles, so LDSM issues once per
   k-tile per fragment instead of per k_block. Budget check: B n64×128 = 256 B/thread
   = 64 regs + A 16×128/32 = 16 regs + 32 acc regs + addressing ≈ 130–150 regs/thread —
   exactly dg's 168-reg profile. This halves LDSM wavefronts (the actual L1 saturator)
   and removes the LDSM→QMMA short-latency chains that starve the tensor pipe.
2. **QMMA-dense inner loop**: with (1), the k_block loop body becomes pure `cute.gemm`
   over pre-loaded fragments; SF operand writes (`atom.set`) must stay region-legal —
   either per-k_block sets on register fragments already resident (cheap MOVs off the
   critical path), or, if the DSL grows support, aux-operand descriptors bound per k-tile.
3. **Keep the small-M path untouched** — b12x's 16×64/16×128/32×128 decode tiles beat dg
   by up to 1.8× (§6.1); the rewrite should be the `expected_m>2048` regime only, keeping
   the one-kernel-per-(N,K,regime) freeze/reuse contract.
4. Re-run the §6.1 sweep + §6.2–6.4 NCU set after each step; the two committed changes
   (§3, §6.5) are the correct base to build on (they remove the quant and SF-load noise
   from the profile).

Expected payoff: closing SM 73→88% is worth ~17–20% on these GEMMs ≈ +2–3% E2E prefill on
the full-B12X stack (rows #3/#4 → above row #5), which would make the pure
`--linear-backend b12x` stack the single fastest config and obsolete both the vLLM
dual-path and the hybrid recommendation.

## 6.8 Related: why A8-forced MoE decodes slower than A16 (133 vs 140 tok/s)

Not a dense-linear issue, but root-caused in the same session. DS4 TP2 E2E: A16-forced
decode ITL 7.14 ms (140.5 t/s) vs A8-forced ~7.7 ms (~130). Isolated MoE benchmark
(`benchmarks/benchmark_ds4_moe.py`, E=256 K=4096 I_tp=1024 topk=6):

| M | w4a8_mx | w4a16 |
|---:|---:|---:|
| 1 | 0.308 ms | **0.276 ms** |
| 4 | 0.305 ms | **0.274 ms** |
| 8 | 0.309 ms | **0.279 ms** |
| 64 | 0.863 ms | **0.819 ms** |
| 8192 | **2.817 ms** | 4.848 ms |

Per-kernel decode profile at M=4 (torch profiler, per layer-call): w4a16 = one
`W4A16FusedMoeKernel`, **75.0 µs**; w4a8_mx = `MoEDynamicKernelSilu` **84.9 µs** + 2×DtoD
memcpy + 2×fill (**+3.2 µs**). Δ ≈ 13 µs × 61 MoE layers ≈ 0.8 ms/token — matches the E2E
ITL delta.

**Root cause is a storage-policy decision, not kernel physics**
(`b12x/integration/tp_moe.py::_resolve_workspace_layout`): `w4a8_nvfp4` keeps
source-native weights, so its tiny-decode band (m≤8) runs the **direct-micro kernel —
measured 2.2× faster than the dynamic kernel** (0.076 vs 0.169 ms at m=8/topk=10, per the
in-code comment). `w4a8_mx` repacks weights **in-place** into the dynamic kernel's
N256/K128 layout (to avoid a second copy of ~45 GB/rank of expert weights), and the micro
kernel cannot read that layout → tiny decode is hard-wired to `"dynamic"`.

### 6.8.1 Deeper decode data (graph-replay, µs/layer)

| M | a8 dynamic (default) | a8 MAC=94 | a8 MAC=188 | a16 fused |
|---:|---:|---:|---:|---:|
| 1 | 34.8 | 34.8 | 34.8 | **22.5** |
| 4 | 88.0 | 85.1 | 81.9 | **75.6** |
| 8 | 210.0 | 209.4 | 226.0 | **190.4** |

The M=1 row is the one that matters for cc1 decode (seqs=1, MTP off): **34.8 vs 22.5 µs =
+55%**, ×61 MoE layers ≈ 0.75 ms/token — the entire A8-vs-A16 E2E ITL gap. a16 at M=1 is
essentially at the BW floor (6 experts × 6.3 MB ≈ 24 µs at ~1.6 TB/s).

NCU at M=1 (both kernels, grid = 96 CTAs, work-limited — MAC has no effect here):

| | a8 `MoEDynamicKernelSilu` | a16 `W4A16FusedMoeKernel` |
|---|---:|---:|
| duration | 42.5 µs | 33.5 µs |
| DRAM throughput | 55.5% | **70.3%** |
| L2 bytes | 46.2 MB (+8.8%) | 42.4 MB |
| block size | **192 (6 warps)** | **256 (8 warps)** |
| warps active | 12.5% | 16.7% |

Falsified knobs (all measured):
- `B12X_DYNAMIC_W4A8_DECODE_MAX_ACTIVE_CLUSTERS` ∈ {94, 128, 188}: no effect at M=1
  (grid is work-limited at 96 CTAs), −7% at M=4 with 188, **regression at M=8** (226 µs).
  The tuned `((64,64),)` ladder in `b12x/moe/tuning/decode.max_active_clusters.py` is
  already near-optimal for its structure.
- `B12X_DYNAMIC_TILE_MN=16x64 / 32x64`: **does not compile** — SFB TMA partition is
  written for tile_n=128 (`expects smem and gmem have the same size in the first rank`,
  dynamic.py:3182). tile_m=16 is already the decode ladder choice.
- widening the M16 W4A8 band to 8 MMA warps (`atom_shape (1,8,1)`, mirroring a16's
  256-thread advantage): **hard compiler crash** (`Floating point exception` during CuTe
  DSL layout construction) — the SM120 helper stack only supports Luke's tested atom
  shapes ((1,4,1) works, (1,8,1) and dense's (2,4,1) do not).
- Note the N256/K128 weight repack itself (`_copy_qweight_to_w4a8_rp_inplace`) is a pure
  int32 reshape+permute — affinely invertible, so a zero-copy source-native *view* of the
  repacked buffer exists in principle; whether micro's TMA boxes can consume that stride
  pattern is the open question for path (1).
- Per-launch wrapper overhead outside the kernel: 2×`FillFunctor<int>` =
  `barrier_count.zero_() + barrier_epoch.zero_()` (volatile_launch_state=True on the
  vLLM eager-bind path) + 2× small DtoD ≈ **3.0 µs/layer ≈ 0.18 ms/token**.

### 6.8.2 Port groundwork completed (2026-07-02 evening session)

The micro-port was attempted. Status:

**Done and verified (PR #21, commit `123a16f`,
`tests/test_w4a8_rp_inverse_mapping.py`):** both in-place repacks are pure
power-of-two permutations with exact bitfield inverses, validated bit-exact against the
real prep kernels for the w13 (rows=2n, rot=n) and w2 (rows=k) orientations, weights and
e8m0 grids:

```
weights: p=(r-rot)%rows; nt=p>>8; row=p&255; n8c=row>>5; n8i=(row>>3)&3; r8=row&7
         kt=w>>4; k32=(w&15)>>2; cgrp=w&3
         off_words = (nt*k_tiles+kt)*4096 + (n8i | cgrp<<2 | r8<<4 | n8c<<7 | k32<<10)
grids:   off_bytes = kb | row8<<2 | n32<<5 | (nt*k_tiles+kt)<<10   (kb=c&3, kt=c>>2)
```

(The sfb prep clamps e8m0 ≥ 249 — unreachable for real weights.) Zero extra weight
memory: the kernel indexes the repacked buffers directly.

**The decisive constraint found:** the rp 16 B-contiguous unit spans logical rows
{r, r+8, r+16, r+24} at one k-window (the `n8i` mode). Direct-micro's FC1 dataflow is
row-per-warp / 64 B-per-lane (`ld_global_nc_v4_u32` quads) — under rp its lanes land
16 KB apart → ~12.5% sector efficiency. **A 1:1 address translation would destroy the
DRAM efficiency the port exists to gain.** The register-resident dot-product + warp
reduce choreography is built around the row-per-warp assignment, so preserving
coalescing means redesigning the lane→(rows×k) map — i.e., authoring an rp-native
tiny-M kernel, not porting micro.

**Recommendation updated:** build the tiny-decode band as an **rp-native M≤8 mode of
the dynamic-kernel family** (it already speaks this layout natively) rather than a
micro port: 8 warps/256 threads (the a16 kernel's 70% vs 55% DRAM-util edge comes
exactly from this; the (1,8,1) atom FPE in the SM120 helpers is the blocker to fix
first), lanes covering four 8-apart rows per instruction per the mapping above, the
+8.8% L2 byte overhead trimmed, and barrier re-zeroes folded into Phase-0 (drops the 2
fills + 2 DtoD ≈ 3 µs/layer from every decode graph). Target remains the a16 kernel's
22.5 µs/layer at M=1 → **A8 decode ≈141–144 t/s with the A8 prefill advantage kept**.

Falsified during the attempt (beyond §6.8.1): `atom_shape (1,8,1)` for the M16 band —
hard `Floating point exception` in CuTe DSL layout construction (helpers support only
the tested shapes). Note the M16 W4A8 band is architecturally 6-warp: "six route warps
… three two-warp token groups" with pair-owned producers — widening it is part of the
redesign, not a knob.

**Wrapper overhead root causes (for the +3 µs/layer):** the 2 DtoD are
`_refresh_dynamic_workspace_scales` copying the [E] `input_gs`/`down_input_scale`
vectors every launch — the skip memo lives on the `TPDynamicWorkspace`, which the
eager-bind path reconstructs per call, so the src-pointer check never matches. A durable
skip needs the memo on a binding cached across steps (or prefilled arena slots +
contract change), since the eager-bind contract forbids bind-time writes under graph
capture. The 2 fills are the `barrier_count/epoch` re-zeroes required by the epoch
protocol; removing them means a self-cleaning kernel exit (kernel change). Both are
~1.5 µs/layer each — worth folding into the tiny-M redesign, not standalone surgery.

Per-M routing between w4a16 and w4a8_mx kernels was evaluated and is **not viable**:
`required_weight_layout()` demands MMA_PACKED for w4a16 vs QMMA_REPACKED for w4a8_mx —
two physical copies of the MoE weights (~+45 GB/rank on DS4-Flash).

## 7. Recommended configs

**Today (no patches, stock image) — the hybrid, fastest overall:**

```bash
# handoff DS4 TP2 A8 launch, two deltas:
#   - remove --linear-backend b12x
#   - VLLM_USE_B12X_FP8_GEMM=0
# keep: --attention-backend B12X_MLA_SPARSE, --moe-backend b12x, B12X_MOE_FORCE_A8=1,
#       VLLM_USE_B12X_SPARSE_INDEXER=1, VLLM_ENABLE_PCIE_ALLREDUCE=1 (b12x), fp8 KV
```

Log confirmations: `Selected DeepGemmFp8BlockScaledMMKernel for Fp8LinearMethod`,
`Using 'B12X' Mxfp4 MoE backend.`, `B12X MoE force-A8 enabled: using quant_mode=w4a8_mx`,
`Using b12x PCIe oneshot allreduce backend`.

**Full `--linear-backend b12x` stack (needs b12x PR + vLLM branch):** quant fix
(`a67e5bd`) + dual-path (`9dbad237`) → row #4. Within ~1% of hybrid; becomes the winner
after §6.7.

## 8. Repro guide

GEMM head-to-head + tile probe (single GPU, ~10 min):

```bash
docker run --rm --gpus all --runtime nvidia --ipc host \
  -e CUDA_VISIBLE_DEVICES=6 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_120a \
  -v <b12x-checkout>/benchmarks:/bench:ro --entrypoint /bin/sh \
  voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x77bd50e-overlay-cu132-20260702 \
  -lc 'cd /bench && python3 benchmark_dense_fp8_vs_deepgemm.py'
# tile sweep: python3 probe_dense_fp8_tile_sweep.py --n 4096 --k 4096 --m-list 4096,8192
```

NCU (works in-container with `--cap-add=SYS_ADMIN`; ncu at
`/opt/nvidia/nsight-compute/*/ncu`, python API via `PYTHONPATH=<ncu-dir>/extras/python`,
`import ncu_report`):

```bash
ncu --set basic --launch-skip 4 --launch-count 2 -k "regex:DenseGemmKernel" -o out python3 repro.py
# opcode histogram: metric sass__inst_executed_per_opcode, names via metric.correlation_ids()
# stalls: smsp__average_warps_issue_stalled_{math_pipe_throttle,wait,long_scoreboard,...}_per_issue_active.ratio
```

Standalone prefill / decode benches: `llm_decode_bench.py` per the handoff
(`/root/vllm/fable/handoff.md`); decode artifacts must match `--max-tokens` between runs
(the reference GLM/DS4 cc1 artifacts used 8192/2048 — with 512 the per-request accounting
folds TTFT in and under-reads ~2–3%).

## 9. Artifacts

| What | Where |
|---|---|
| b12x commits (quant fix + SF work) | branch `fable/dense-sf-hoist-20260702` = `a67e5bd` + `49b453c`; patch files `/root/vllm/fable/0001-*.patch`, `0002-*.patch`; PR to `lukealonso/b12x` from the `voipmonitor` fork |
| vLLM dual-path | `local-inference-lab/vllm` branch `fable/b12x-linear-dg-prefill-route-20260702` (`9dbad237`) |
| vLLM indexer experiment | branch `fable/b12x-indexer-prefill-mqa-logits-20260702` (`b3fa995`) |
| Bench JSONs (all configs, §1) | `/root/bench-results/ds4-a8-{base,fastquant,dualpath,dualpath-p01,mqaprefill,mqa2048,mqa256,dglin-b12xidx,mqa2048-dglin}-*-20260702/`, `/root/bench-results/ds4-lucifer-v3f65c52-20260702/` |
| Torch traces (§2) | `/root/bench-results/vllm-profile/ds4-{mqa2048,lucifer}-prefill-g4-20260702/` |
| NCU reports | scratchpad `ncu-out/` (`b12x_wo_b`, `dg_wo_b`, `b12x_stall`, `dg_stall`, `b12x_sfbreuse`) — regenerate via §8 if the scratchpad was cleaned |
| Session log | `/root/vllm/CODEX_JOURNAL.md` (2026-07-02 sections), `/root/vllm/fable/handoff.md` UPDATE 1–4 |
| Prior context | `b12x docs/sm120_dense_fp8_deepgemm_port.md` (Luke's tile-selection port + §8 weight re-quant fix) |
