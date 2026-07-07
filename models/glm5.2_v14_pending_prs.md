# GLM 5.2 v14 — Pending PRs (temporary tracking page)

Status date: 2026-07-07. This page tracks the PR stack behind the v14/v5/v6
images so nothing gets lost at merge time. Full measurements and context live
in [`glm5.2_v14.md`](./glm5.2_v14.md); numbers here are the one-line proof per
PR. Delete this page once everything is merged and a clean image builds from
`dev/eldritch-enlightenment` + `b12x master` without pins or patches.

## vLLM — `local-inference-lab/vllm`, merge into `dev/eldritch-enlightenment`

**Status 2026-07-07 (post-merge round):** #76 and #77 are **merged**. The
remaining PRs (#78, #79, #81, #83) were rebased to **exactly one commit each
on current `dev/eldritch-enlightenment`** — all report `mergeable=true /
clean`, no stacked prefixes anymore, mergeable in any order (if GitHub flags
a trivial test-file overlap after one lands, "Update branch" resolves it).

**#80 was rejected by Luke** (by design, not conflict): zero-padding e8m0
expert shards to 128 defeats the point — A4/A16 support much smaller
alignments and W4A8-MX should too. Consequence: TP6 × MXFP4 (A8 and packed
A16) stays unbootable on stock until the kernels grow native small-alignment
support; the measured TP6 A8 numbers below stand as padded-experiment results
only. b12x #26 (odd tile counts) is unaffected — its value is DS4-Pro TP8
(384/rank) and GLM TP16 (128/rank), which need no padding.

| # | PR | What it does | Proof |
|---|---|---|---|
| 0 | [#74](https://github.com/local-inference-lab/vllm/pull/74) **MERGED** | Online MXFP8 overlay for checkpoint-excluded dense linears (`--quantization-config '{"linear":{"weight":"mxfp8"},...}'`) | bit-parity with the offline MXFP8dense checkpoints; +6-8 tok/s decode vs BF16 dense |
| 1 | [#76](https://github.com/local-inference-lab/vllm/pull/76) **MERGED** | fp8.py bridge: `store_dtype: nvfp4` experts + serialized-MXFP8 dense loaders (mixed hybrid checkpoints load upstream-style) | enables the `online` variants measured in the v14 sweep (online A4 95.51 vs base 88.53 tok/s decode DCP1) |
| 2 | [#77](https://github.com/local-inference-lab/vllm/pull/77) **MERGED** | Online dense FP8/MXFP8 overlays on `mxfp4` checkpoints (`ONLINE_FP8_MXFP4`); + `6a784b94`: `linear` spec never touches shared experts (parity with ModelOpt semantics) | quantized shared experts were strictly worse: 0.156 vs 0.152 mean\|Δlogprob\| **and** 90.1 vs 92.5 tok/s; kv_b ignore preset: 0.1481 → 0.1448 mean\|Δlp\| at equal speed |
| 3 | [#78](https://github.com/local-inference-lab/vllm/pull/78) | Hybrid DCP dispatch: `VLLM_DCP_A2A_MAX_TOKENS=64` — B12X A2A ≤64 tokens/step, AG+RS above; also shrinks B12X DCP staging 0.6 GB → 5 MB/rank | prefill 2466 → 3225 tok/s (+31%), decode ≤64 tok +3-9% vs ag_rs, crossover measured exactly at 64 tokens/step; CC32 1025.7 |
| 4 | [#79](https://github.com/local-inference-lab/vllm/pull/79) | DCP warmup must not fail boot for world sizes ∉ {2,4,8} (TP6+DCP3/6 regression; runtime falls back to NCCL as on v13) | TP6/DCP1 131.1 tok/s (KV 168k), TP6/DCP6 74.9 tok/s (KV 989k), 0 CJK — previously died at warmup |
| 5 | [#80](https://github.com/local-inference-lab/vllm/pull/80) **REJECTED** | Zero-pad e8m0 expert shards to 128-column kernel tiles (GLM 2048/TP6 → 352→384; bit-exact, ~9% extra expert GEMM on padded rank). Unblocks both w4a8_mx and packed-w4a16 on TP6/MXFP4 | with b12x #26: A8 83.0/81.4 decode + 4864/5219 prefill; A16 79.2/77.9 + 4643/4816; both previously failed before ready |
| 6 | [#81](https://github.com/local-inference-lab/vllm/pull/81) | Head-sliced attention views (TP6 head66 padding) made contiguous before the B12X PCIe DCP pool — fixes TP6+DCP>1 `partial_lse must be contiguous` at capture | TP6/A8/DCP2 on v6: boots, KV 639,616 tokens, 67.1/66.9 tok/s, 0 CJK; build-patch equivalent already in `blackwell-llm-docker/patches/vllm-dcp-b12x-contiguous-lse-20260707.patch` |
| 7 | [#83](https://github.com/local-inference-lab/vllm/pull/83) | One-liner: register the `mxfp8` checkpoint alias in the quantization override list — without it every serialized ModelOpt MXFP8 checkpoint crashes auto-detection | GLM-5.2-BF16-MXFP8experts (740.6 GiB, experts BF16→MXFP8 from zai, dense byte-identical to Luke) boots TP16 with MARLIN MxFp8 MoE: KV 681,408 tokens, 69.8/68.8 tok/s, 0 CJK. Independent of the stack (targets dev directly, mergeable any time) |

## b12x — `lukealonso/b12x`, merge into `master`

| PR | What it does | Proof |
|---|---|---|
| [#26](https://github.com/lukealonso/b12x/pull/26) | `tiny_decode` supports odd FC2 K-tile counts (`n % 128` instead of `n % 256`); FC2 K-tiles-per-task becomes a configure()-time value (2 if even — unchanged binaries — else 1) | TP6 A8 decode 72.2 → **83.0** tok/s (beats A16); prefill unchanged; oracle 10/10 incl. n=384. Also newly enables tiny for **DS4-Pro TP8** (3072/8 = 384/rank) and GLM TP16 (128/rank) |
| [#27](https://github.com/lukealonso/b12x/pull/27) | **The #80 replacement Luke asked for**: native 32-aligned expert shards — ceil-tiled rp/sfb storage with half-aligned gated halves, tiny/dynamic/w4a16 all serve 352 (GLM TP6) and 192 (DS4-Pro TP16) with zero checkpoint padding. Includes #26's commits (merge either first). Also fixes a pre-existing silent-corruption bug (e8m0 small-M direct on multi-chunk tails) | E2E TP6/MXFP4/**A8**: decode **84.1/82.7** tok/s (beats the padded experiment 83.0/81.4), 30k TTFT 3.27 s, 0 CJK; A16 native 67.9/67.0; oracle n∈{1024,384,352,192}, suite parity vs master. vLLM needs NO change (dev never merged #80) |

## Image state vs PRs

- `v6` (`vllm49bed029-b12x26144c0`) already **contains #74–#80 and b12x #26**
  via branch pins; #81 rides as a build patch until merged.
- **Caveat for the next image**: v6's vLLM pin includes the rejected #80
  padding — with b12x #27 the padding is superseded, so the next build should
  pin vLLM WITHOUT #80 (current dev HEAD is fine) + b12x with #27; otherwise
  vLLM pads 352→384 before b12x ever sees the shard and #27's native path
  never engages.
- After all merges: rebuild from `dev/eldritch-enlightenment` HEAD +
  `b12x master` HEAD, drop the branch pins and the #81 patch file, and delete
  this page.

## Known follow-ups intentionally NOT in any PR (documented for Luke)

- W4A16 TC-decode has a ~2× per-call reserve vs tiny at M=1 on e8m0/packed
  (isolated graph-replay 32.8 vs 16.2 µs, outputs cos 0.99996). Closing it
  needs a packed-layout reader in tiny (weight duplication is infeasible:
  the aux copy is per-MoE-layer = +46 GB/rank). Details in
  [b12x #26 comments](https://github.com/lukealonso/b12x/pull/26).
- DCP A2A large-batch path: the proper long-term fix is a pipelined /
  copy-engine B12X exchange (pcie_twoshot direction); #78's token crossover
  is the policy until that kernel exists.
- TP6 DCP2 needs `GPU_MEMORY_UTILIZATION<=0.95` (0.957 OOMs at
  `warmup_dynamic_launches`).
