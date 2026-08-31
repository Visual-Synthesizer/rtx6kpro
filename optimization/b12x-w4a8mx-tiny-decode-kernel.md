# W4A8-MX Tiny-Decode Kernel — rp-native Triton M=1 MoE path (work in progress)

Working log of the new tiny-decode kernel that closes the DS4 A8-vs-A16 decode gap by
replacing the b12x dynamic grouped kernel at M=1 with two Triton kernels that read the
**N256/K128 in-place-repacked weights directly** (zero extra weight memory). Written so
Luke can pick this up / work in parallel — every iteration below is measured, and the
open questions are flagged.

> **FINAL STATE (2026-07-03): the kernel was rewritten in CuTe DSL inside b12x per
> Luke's review and now beats every reference — [lukealonso/b12x PR #22](https://github.com/lukealonso/b12x/pull/22),
> **on by default** as of `de15849` (rename `tiny_rp` → `tiny_decode`; kill switch
> `B12X_W4A8_TINY_DECODE=0`): **20.6 µs/layer** at M=1 (dynamic 37.8, w4a16 fused 22.5),
> E2E decode cc1 **140.2 tok/s** = A16 parity (ITL 7.131 vs 7.136 ms) with the A8
> prefill advantage intact. Zero vLLM changes; vLLM PR #70 closed as superseded.
> Ready-to-run image (no env needed): \`voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12xtinydec-de15849-overlay-cu132-20260703\`
> (E2E re-verified flagless: 140.16 tok/s).
> DSL port highlights: hardware \`fp4_dot4\` decode + v4 16 B loads of the rp quads
> (the (n8c,r8,cgrp)→four-8-apart-rows lane map), bf16x2 scatter-add epilogue with a
> lane^4 row-pairing shuffle, BF16 activations. A cooperative single-launch variant
> with a resident grid barrier **deadlocked under TP2 serving** (rank-skewed first-use
> compile broke co-residency) — the landed two-plain-launch form is deadlock-free by
> construction and measured faster (20.6 vs 30.8 µs). Sections below document the
> Triton prototype journey that produced the layout tricks.

> **State (2026-07-02 late, updated):** kernel is correct (cos vs fp32 oracle
> **0.999999**, better than the production w4a8 dynamic kernel's 0.9990), fast in
> isolation (**28.8 µs/layer at M=1** vs dynamic 34.8 µs, a16 fused 22.5 µs), and
> **E2E-reproducible at 138.7–138.8 tok/s** DS4 TP2 A8 decode cc1 (baseline 133.7,
> A16 reference 140.5). The §6 "restart variance" was solved: `sem='relaxed'` on the
> atomic scatters — isolated-neutral but −7% E2E — was the regression; removed
> (vLLM commit `8e6e417c`).

## MERGED (2026-07-03) + canonical-master validation

Both PRs are in Luke's master (`a1c1e54` mappings/quant/SF, `a5f0e0c` tiny-decode), and
Luke stacked further dense-GEMM work on top (`8de17f1` end-to-end MXFP8 GEMM,
`a611e36` quant-per-warp). Follow-up [PR #23](https://github.com/lukealonso/b12x/pull/23)
carries the rename (`tiny_rp` → `tiny_decode`) + **default-on** (kill switch
`B12X_W4A8_TINY_DECODE=0`) that the squash-merge missed — until it lands, master needs
`B12X_W4A8_TINY_RP=1`.

**Full-B12X stack on merged master `a611e36`** (no DeepGEMM anywhere, DS4 TP2 A8,
image `voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12xa611e36-overlay-cu132-20260703`):

| | 8k | 64k | 128k | decode cc1 |
|---|---:|---:|---:|---:|
| full B12X, merged master + tiny | **13,550** | **13,088** | **12,102** | **136.3** |
| Lucifer CUTLASS reference | 13,442 | 12,622 | 11,716 | — |
| hybrid (DG linear) + tiny, for comparison | 13,811 | 13,201 | 12,182 | 140.2 |

Full B12X now beats Lucifer at **all three** prefill contexts with no DeepGEMM. The
remaining deltas to the hybrid (−1.9/−0.9/−0.7 % prefill, −3.9 t/s decode) are the
dense-linear small/large-M bands — Luke's GEMM stream is actively closing them.
Coherence clean (CJK 0) at 30k.

**Update (`c7d3eaf`, "Optimize mHC prefill projection for large prompts"):** Luke's
TF32-TMA mHC projection rewrite (chunk tiles M192/K64, k-split 8, tuned at 4096/8192
tokens) closes the prefill gap entirely — full B12X now **13,801 / 13,315 / 12,314**
@8k/64k/128k vs hybrid 13,811/13,201/12,182: a tie at 8k and a **win at 64k (+0.9 %)
and 128k (+1.1 %)**. Lucifer is −2.7/−5.5/−5.1 % behind. Decode unchanged (136.2 —
mHC is prefill-only); the dense-linear M=1 band remains the hybrid's last edge
(140.2 vs 136.2). Image:
`voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12xc7d3eaf-overlay-cu132-20260703`.

## GLM-5.2 note (2026-07-03): A8 does NOT transfer, and master is GLM-clean

Asked whether GLM 5.2 (NVFP4 checkpoint, A16/w4a16 MoE) can adopt A8: **no.**
`B12X_MOE_FORCE_A8=1` → `w4a8_nvfp4`, measured on the reference test config
(TP8/DCP1/MTP off/B12X_MLA_SPARSE/seqs1/graph4): decode 84.93 vs 86.32 (−1.6 %),
prefill 5,345/5,502/5,210 vs 5,798/5,960/5,599 (−7 to −8 %), and **broken 30k
coherence (CJK 15, garbage)** — the nvfp4→MXFP8 bridge (e4m3/16 vs e8m0/32 blocks)
looks numerically broken, FYI-grade bug (nobody serves it). DS4's A8 advantage is
specific to the MXFP4-native `w4a8_mx` path. GLM stays A16; `tiny_decode` doesn't
engage there (w4a8_mx-gated; GLM's tiny band is already W4A16 TC-decode micro).
Merged master `dafb8d7` re-validated GLM-regression-free (86.32 vs 86.2–86.5 ref
band; an initial 80.35 reading was a polluted host — a stray bench container).

**Follow-up — "would a better-designed W4A8 help GLM?" No, measured:** GLM's NVFP4
default is already **W4A4** (`quant_mode=nvfp4`, activations FP4 + e4m3/16 scales,
prod-accepted numerics; decode band rides W4A16 TC micro). Forcing A16 vs the A4
default bounds the whole activation-quant lever: prefill 5,473/5,592/5,226 (A16) vs
5,798/5,960/5,599 (A4) = **+6–7 % total**; decode 85.0 vs 86.3. Since `mxf8f6f4` MMA
only accepts ue8m0/32 scales, any W4A8 over NVFP4 weights must requant e4m3/16 weight
scales (numerics ≤ A4) at an MMA rate below A4 — i.e. **W4A8 is strictly dominated by
the shipping A4 mode for NVFP4 checkpoints**. DS4 needed `w4a8_mx` only because MXFP4
checkpoints have no A4 path. The `w4a8_nvfp4` bridge stays a deletion/fix candidate.

## 2026-07-03 late: GLM joins w4a8_mx via checkpoint conversion; DS4 full-B12X reaches total parity

**GLM**: converted `zai-org/GLM-5.2-FP8` into a DS4-Flash-style mixed checkpoint
(FP8 dense/attention + **MXFP4 routed experts**, offline OCP requant, per-tensor cos
0.9935, 7.6 min GPU convert) + `store_dtype: "mxfp4"` in quantization_config → the
`fp8.py` DeepSeek-style branch → B12X Mxfp4 MoE → `w4a8_mx` **with tiny_decode
engaged** (GLM TP8 shard n=256 passes the gate). Measured vs the NVFP4-A4 baseline:
**decode 92.94 vs 86.32 (+7.7 %)**, prefill 5,768/6,140/5,750 vs 5,798/5,960/5,599
(−0.5/+3.0/+2.7 %), 30k coherence clean. KLD quality gate pending.
**Update on b12x `f416b75`** (quant-A split-K): GLM decode **99.1** (99.12/98.90,
coherence clean; prefill unchanged) — total GLM decode gain over the NVFP4-A4
baseline is now **+14.8 %**. Checkpoint:
`/root/models/GLM-5.2-FP8-MXFP4experts` (converter script in the session scratchpad;
worth landing in rtx6kpro/code).

**DS4 @ b12x `f416b75`** (WO-projection dense-GEMM port, quant-A split-K, WO PDL
chaining): full-B12X decode **140.2** (= hybrid parity; first post-boot run reads
~132, a cold-start artifact — ignore run 1), prefill **13,822/13,316/12,285** (ties
hybrid at 8k, beats it at 64k/128k). **Full B12X ≥ DG hybrid on every axis; DeepGEMM
is retired.** Luke's CuTe-side PDL chaining shows none of the Triton `launch_pdl`
−7 % trap.

## Code access (everything Luke needs)

- **vLLM PRs**: [#70 tiny-decode](https://github.com/local-inference-lab/vllm/pull/70)
  (stacked on [#69 dual-path linear](https://github.com/local-inference-lab/vllm/pull/69)).
- **vLLM branch (integrated)**: https://github.com/local-inference-lab/vllm/tree/fable/b12x-w4a8mx-tiny-decode-20260702 —
  head `8e6e417c` (`fix: drop relaxed atomics`), kernel commit `6896d418`. Files:
  [`vllm/model_executor/layers/fused_moe/b12x_tiny_decode.py`](https://github.com/local-inference-lab/vllm/blob/fable/b12x-w4a8mx-tiny-decode-20260702/vllm/model_executor/layers/fused_moe/b12x_tiny_decode.py)
  + a 10-line hook at the top of `_run_b12x_moe_fp4` in `b12x_moe.py`.
- **Standalone kit (this repo)**: [`optimization/code/tiny-decode/`](code/tiny-decode/) —
  final kernels, the fp32-oracle/timing driver, and the abandoned fused variant.
- **Docker image (code baked in, ready to serve)**:
  `voipmonitor/vllm:eldritch-enlightenment-v8e6e417c-b12x77bd50e-tinymoe-overlay-cu132-20260702`
  (pushed to Docker Hub; base v3f65c52-b12x77bd50e + the branch's vllm tree).
  Enable with `-e VLLM_B12X_W4A8_MX_TINY_DECODE=1` on the hybrid DS4 launch — no bind
  mounts needed.
- **Mappings prerequisite**: `lukealonso/b12x` PR #21 (`123a16f`,
  `tests/test_w4a8_rp_inverse_mapping.py`).

Measured with the baked image + env flag: decode cc1 **138.7–138.8 tok/s**, prefill
unchanged (12,202 tok/s @128k vs 12,182 hybrid reference), 30k coherence clean.

## Branches / commits

| What | Where |
|---|---|
| vLLM kernel + integration | `local-inference-lab/vllm` branch **`fable/b12x-w4a8mx-tiny-decode-20260702`**, commit `6896d418` — `vllm/model_executor/layers/fused_moe/b12x_tiny_decode.py` (kernels + `maybe_run_tiny_w4a8mx_moe`) and the hook at the top of `_run_b12x_moe_fp4` in `b12x_moe.py`. Branch is stacked on `9dbad237` (dual-path linear). |
| Verified inverse mappings (prerequisite) | `lukealonso/b12x` **PR #21**, commit `123a16f` — `tests/test_w4a8_rp_inverse_mapping.py` (weights + sfb grids, w13-rotated + w2 orientations, bit-exact vs the real prep kernels) |
| Related dense-linear work | PR #21 commits `a67e5bd` (tiled quant, 8–35×) + `49b453c` (SF stage-bulk/SFB k-reuse); vLLM `9dbad237` (dual-path linear) |
| Enable flag | `VLLM_B12X_W4A8_MX_TINY_DECODE=1` (default off) |

## 1. Design

Program = (routed row, w13 rp tile `nt`, k tile `kt`) for FC1; (routed row, out-row tile,
k tile) for FC2. Two kernels, FC1 stages SiLU inputs as fp32 in a small gmem scratch
([M·topk, 2n] = 48 KB at M=1), FC2 applies `silu(gate)·up` inline and scatters
router-weighted fp32 atomics into the output ([M, K] fp32 scratch → one bf16 copy).

Key decisions, all measured (§4):
- **BF16 activations, no input quantization** — at M=1 activations are 8 KB; skipping the
  MXFP8 quant both removes work and *improves numerics over the production path*
  (cos vs fp32 oracle 0.999999 vs 0.9990).
- **Whole-tile flat loads**: one rp (nt,kt) tile = 4096 contiguous int32 words whose flat
  index is `k32<<10 | n8c<<7 | r8<<4 | cgrp<<2 | n8i`. Load with a plain
  `tl.arange(0, 4096)` (provably contiguous ⇒ Triton vectorizes to 128-bit loads) and
  `tl.reshape` to `(k32, n8c, r8, cgrp, n8i)`; logical coords fall out of the axes:
  `row = n8c*32 + n8i*8 + r8`, `k = kt*128 + (k32*4+cgrp)*8 + j`, scale col `= kt*4+k32`.
- **SFU-free decodes**: e2m1 nibble → fp32 by direct bit assembly
  (`e>0: ((e+126)<<23)|(m<<22); e==0: m*(126<<23)`, plus sign bit);
  e8m0 scale byte → fp32 by `byte<<23` bitcast. No `exp2` anywhere.
- **Activation loads hoisted** out of the nibble loop: one contiguous 128-wide load per
  k-tile, per-j extraction via an 8-wide mask-sum (registers only).
- **Gate/up tile pairing**: after the prep rotation, rp tiles `[0, N/256)` hold the "up"
  rows and `[N/256, 2N/256)` the "gate" rows of the same channels **for both** the vLLM
  `w31` layout (rot=N) and the bench's up-first layout (rot=0) — the rotation exists
  exactly to normalize this. FC1 writes inter rows via `r_log = (p + N) % 2N`.
- `KT_PER_PROG=1` (one 16 KB tile per program; more programs = better latency hiding),
  `num_warps=8` (16 elems/thread; 4 warps ⇒ catastrophic spills, §4), relaxed-semantics
  atomics (kernel boundary is the only ordering needed; adds commute).

## 2. Correctness gates (all green)

- `cos(tiny, fp32 oracle) = 0.999999` at M=1/4/8 (dynamic kernel: 0.9990).
- `cos(tiny, dynamic kernel) = 0.999` (the residual is the dynamic path's MXFP8
  activation quantization).
- E2E serve: 30k-ctx coherence clean (CJK 0), sane completions.
- The gate/up pairing was additionally confirmed by breaking it (oracle cos drops to ~0.5).

Repro driver: scratchpad `tiny_moe_test.py` (synthetic experts through the *real* b12x
prep, fp32 oracle, dynamic-kernel cross-check, graph-replay timing). Ask fable/see the
journal for the file; it belongs in `benchmarks/` once this lands.

## 3. Isolated performance (graph replay, µs/layer, DS4 TP2 shapes, GPU 6)

| M | tiny (final) | b12x dynamic | a16 fused | notes |
|---:|---:|---:|---:|---|
| 1 | **28.7** | 34.8 (+3.0 wrapper) | 22.5 | tiny also has ~2 µs wrapper (2 zeros + 1 copy) |
| 4 | 195 | 88.0 | 75.6 | tiny re-reads weights per routed row — **do not use at M>1** (integration gates on M==1) |
| 8 | 390 | 210 | 190 | same |

All numbers share the same flattering condition (replayed identical routing ⇒ weights
partially L2-resident); comparisons are apples-to-apples.

## 4. Optimization journey (what moved the needle and what did not)

| Iteration | M=1 µs | Lesson |
|---|---:|---|
| v1: per-element computed-index loads | 108.1 | Triton emits scalar 4 B gathers for any index math it cannot prove contiguous |
| v2: SFU-free decodes + 2D accumulators | 92.2 | exp2-per-nibble was real but not dominant |
| v3: **flat `arange(4096)` loads + reshape** | **36.9** | the big one: provable contiguity ⇒ vectorized loads |
| fused single-kernel FC1+FC2 (3 variants incl. axis-separable affine + `<<`→`*` + num_stages) | 96–111 | per-program 512 B strips + register pressure; abandoned in favor of the 2-kernel shape |
| v7: hoisted activation loads + `KT_PER_PROG` 4→2/1 | 30.7 | fewer per-j gathers; more programs |
| `KT_PER_PROG=1` (FC1 grid = rt×8×32) | **28.8** | shorter streams win |
| relaxed atomics | 28.7 | isolated noise-level, **but −7% E2E in serving (131 vs 138.7) — do not use** |
| `num_stages=2/4` on the kt loop | 97–101 | staged 16 KB tile buffers ⇒ spills — **do not pipeline this loop** |
| `num_warps=4` | 10,966 (!) | 32 elems/thread ⇒ total spill catastrophe |
| `cache_modifier='.cs'` | n/a | not supported by this Triton build |
| **PDL overlap** (FC1 `gdc_launch_dependents` at entry; FC2 loads w2+scales, `gdc_wait`, then inter reads; `launch_pdl=True` on FC2; Triton 3.7 intrinsics) | 28.7 isolated, **131.3 E2E** | compiles + correct, but same −7% E2E as relaxed atomics — suspicious that both exotic variants land at exactly ~131.3 (a graph-replay fast path being disabled?). Reverted; E2E re-confirmed 138.69 after revert. Worth a Luke-level look at what PDL/relaxed do to the captured graph. |

NCU (isolated, M=1): FC1 27.5 µs @ ~57% DRAM, 80 regs/thread; FC2 18.6 µs @ ~42%,
77 regs/thread; no spills. Headroom to a16's 22.5 exists mainly in FC2 utilization and
FC1/FC2 overlap (they serialize at the kernel boundary; a16 is one fused kernel).

## 5. E2E serving (DS4 TP2 A8, hybrid DG-linear base = 133.7 tok/s)

Server: stock hybrid launch + `VLLM_B12X_W4A8_MX_TINY_DECODE=1` + bind-mounted
`b12x_moe.py`/`b12x_tiny_decode.py`. Warmup engages the path eagerly (log line
`b12x w4a8_mx tiny-decode path engaged`), buffers pre-allocate before graph capture,
decode graphs capture the Triton kernels fine.

| Run | decode cc1 ITL | tok/s |
|---|---:|---:|
| baseline (dynamic MoE) | 7.477 ms | 133.7 |
| tiny v1 module, first boot | 7.208 ms | 138.7 |
| tiny + fused-zero variant + relaxed atomics, after restart | 7.616 ms | 131.3 |
| tiny + relaxed atomics only, after restart | 7.635 ms | 131.0 |
| **tiny final (default-sem atomics), fresh restart, run 1/2** | **7.204 / 7.212 ms** | **138.81 / 138.66** |

## 6. SOLVED — the "restart variance" was relaxed atomics

Reverting `sem='relaxed'` → default semantics on both `tl.atomic_add` scatters restored
138.81/138.66 tok/s across two runs on a fresh restart. Lesson for kernel authors: the
isolated graph-replay microbench (identical routing every replay ⇒ L2-warm weights)
completely masked a 7% E2E effect — always confirm atomic/caching-semantics changes
end-to-end. Root cause hypothesis: relaxed semantics change the RED instruction/L2
policy Triton emits, which matters under real mixed traffic but not with a warm L2.

## 7. Next steps (parallel-friendly)

- ~~Resolve §6~~ done — 138.7 reproducible (4 measurements across 3 restarts:
  138.81/138.66/138.69 + first-boot 138.74).
- The remaining −1.8 t/s to the A16 reference (140.5) needs FC1/FC2 overlap that does
  NOT go through PDL or relaxed atomics (both measured −7% E2E, see §4) — i.e. a true
  single fused kernel; the fused CTA-shape that failed for us (§4, 512 B strips) might
  work with CuTe-DSL-style manual smem staging instead of Triton.
- FC2 utilization (42% DRAM): candidates — split FC2 differently (out-tile × expert),
  vectorize the inter reads across kt, or fold FC2 into FC1's tail for its own expert.
- M=2–4 support (per-expert row batching instead of per-routed-row weight re-reads) —
  only matters for MTP/batched decode; cc1 serving is M=1.
- Productionization: move kernels into b12x proper (or vllm officially), add the
  correctness driver to `benchmarks/`, wire a regression test vs the fp32 oracle.
- The same design trivially retargets **w4a8_nvfp4 / GLM shapes** if wanted later
  (mappings are format-generic; only scale decode differs).

## 8. Repro

```bash
# isolated (GPU 6): correctness + graph-replay timing
docker run --rm --gpus all --runtime nvidia --ipc host \
  -e CUDA_VISIBLE_DEVICES=6 -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e CUTE_DSL_ARCH=sm_120a \
  -v <b12x>/benchmarks:/bench:ro -v <scratch>:/work:ro --entrypoint /bin/sh \
  voipmonitor/vllm:eldritch-enlightenment-v3f65c52-b12x77bd50e-overlay-cu132-20260702 \
  -lc 'python3 /work/tiny_moe_test.py'

# E2E: hybrid DS4 launch (drop --linear-backend b12x, VLLM_USE_B12X_FP8_GEMM=0)
#   + -e VLLM_B12X_W4A8_MX_TINY_DECODE=1
#   + bind-mount vllm/model_executor/layers/fused_moe/{b12x_moe.py,b12x_tiny_decode.py}
# then: llm_decode_bench.py --concurrency 1 --contexts 0 --max-tokens 512
```

Related pages: `b12x-dense-fp8-gemm-vs-deepgemm.md` (the full A8/A16 decode gap analysis
that motivated this kernel, incl. why the micro-port and all dynamic-kernel knobs were
dead ends).
