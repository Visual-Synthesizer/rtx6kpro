# GLM 5.2 v14 — Pending PRs (temporary tracking page)

Status date: 2026-07-07. This page tracks the PR stack behind the v14/v5/v6
images so nothing gets lost at merge time. Full measurements and context live
in [`glm5.2_v14.md`](./glm5.2_v14.md); numbers here are the one-line proof per
PR. Delete this page once everything is merged and a clean image builds from
`dev/eldritch-enlightenment` + `b12x master` without pins or patches.

## vLLM — `local-inference-lab/vllm`, merge into `dev/eldritch-enlightenment`

**All six PRs now target `dev/eldritch-enlightenment` directly** (retargeted
2026-07-07; GitHub reports `mergeable=true` for every one). They were
developed as a stack, so until predecessors land, a later PR's diff shows the
cumulative changes of its prefix — merge **strictly in this order** and each
remaining diff shrinks to its own commits:

`#76 → #77 → #78 → #79 → #80 → #81`

(#80/#81 carry content-identical re-picks of #76–#79 with different SHAs from
the v5 integration branch; after the originals merge these become empty
effective diffs and the merges stay clean.)

| # | PR | What it does | Proof |
|---|---|---|---|
| 0 | [#74](https://github.com/local-inference-lab/vllm/pull/74) **MERGED** | Online MXFP8 overlay for checkpoint-excluded dense linears (`--quantization-config '{"linear":{"weight":"mxfp8"},...}'`) | bit-parity with the offline MXFP8dense checkpoints; +6-8 tok/s decode vs BF16 dense |
| 1 | [#76](https://github.com/local-inference-lab/vllm/pull/76) | fp8.py bridge: `store_dtype: nvfp4` experts + serialized-MXFP8 dense loaders (mixed hybrid checkpoints load upstream-style) | enables the `online` variants measured in the v14 sweep (online A4 95.51 vs base 88.53 tok/s decode DCP1) |
| 2 | [#77](https://github.com/local-inference-lab/vllm/pull/77) | Online dense FP8/MXFP8 overlays on `mxfp4` checkpoints (`ONLINE_FP8_MXFP4`); + `6a784b94`: `linear` spec never touches shared experts (parity with ModelOpt semantics) | quantized shared experts were strictly worse: 0.156 vs 0.152 mean\|Δlogprob\| **and** 90.1 vs 92.5 tok/s; kv_b ignore preset: 0.1481 → 0.1448 mean\|Δlp\| at equal speed |
| 3 | [#78](https://github.com/local-inference-lab/vllm/pull/78) | Hybrid DCP dispatch: `VLLM_DCP_A2A_MAX_TOKENS=64` — B12X A2A ≤64 tokens/step, AG+RS above; also shrinks B12X DCP staging 0.6 GB → 5 MB/rank | prefill 2466 → 3225 tok/s (+31%), decode ≤64 tok +3-9% vs ag_rs, crossover measured exactly at 64 tokens/step; CC32 1025.7 |
| 4 | [#79](https://github.com/local-inference-lab/vllm/pull/79) | DCP warmup must not fail boot for world sizes ∉ {2,4,8} (TP6+DCP3/6 regression; runtime falls back to NCCL as on v13) | TP6/DCP1 131.1 tok/s (KV 168k), TP6/DCP6 74.9 tok/s (KV 989k), 0 CJK — previously died at warmup |
| 5 | [#80](https://github.com/local-inference-lab/vllm/pull/80) | Zero-pad e8m0 expert shards to 128-column kernel tiles (GLM 2048/TP6 → 352→384; bit-exact, ~9% extra expert GEMM on padded rank). Unblocks both w4a8_mx and packed-w4a16 on TP6/MXFP4 | with b12x #26: A8 83.0/81.4 decode + 4864/5219 prefill; A16 79.2/77.9 + 4643/4816; both previously failed before ready |
| 6 | [#81](https://github.com/local-inference-lab/vllm/pull/81) | Head-sliced attention views (TP6 head66 padding) made contiguous before the B12X PCIe DCP pool — fixes TP6+DCP>1 `partial_lse must be contiguous` at capture | TP6/A8/DCP2 on v6: boots, KV 639,616 tokens, 67.1/66.9 tok/s, 0 CJK; build-patch equivalent already in `blackwell-llm-docker/patches/vllm-dcp-b12x-contiguous-lse-20260707.patch` |
| 7 | [#83](https://github.com/local-inference-lab/vllm/pull/83) | One-liner: register the `mxfp8` checkpoint alias in the quantization override list — without it every serialized ModelOpt MXFP8 checkpoint crashes auto-detection | GLM-5.2-BF16-MXFP8experts (740.6 GiB, experts BF16→MXFP8 from zai, dense byte-identical to Luke) boots TP16 with MARLIN MxFp8 MoE: KV 681,408 tokens, 69.8/68.8 tok/s, 0 CJK. Independent of the stack (targets dev directly, mergeable any time) |

## b12x — `lukealonso/b12x`, merge into `master`

| PR | What it does | Proof |
|---|---|---|
| [#26](https://github.com/lukealonso/b12x/pull/26) | `tiny_decode` supports odd FC2 K-tile counts (`n % 128` instead of `n % 256`); FC2 K-tiles-per-task becomes a configure()-time value (2 if even — unchanged binaries — else 1) | TP6 A8 decode 72.2 → **83.0** tok/s (beats A16); prefill unchanged; oracle 10/10 incl. n=384. Also newly enables tiny for **DS4-Pro TP8** (3072/8 = 384/rank) and GLM TP16 (128/rank) |

## Image state vs PRs

- `v6` (`vllm49bed029-b12x26144c0`) already **contains #74–#80 and b12x #26**
  via branch pins; #81 rides as a build patch until merged.
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
