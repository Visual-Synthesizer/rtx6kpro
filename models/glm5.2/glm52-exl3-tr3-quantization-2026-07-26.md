# GLM-5.2 BF16 to EXL3-TR3 3.0 bpw: Calibrated-LDLQ Conversion Recipe (incl. MTP layer 78)

Documented 2026-07-26. Base encode performed 2026-07-20 on an 8x B300 node
(SM 10.3 GPUs; quantizer ops built for arch 10.0); the MTP layer-78 leg validated 2026-07-24/25 on 4x RTX PRO 6000 Blackwell.

This page documents the full conversion recipe behind
[`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw):
GLM-5.2 753B BF16 routed MoE experts to a 3.0-bpw EXL3 **Trellis**
representation via a **calibrated LDLQ** pass, TP4 rank-sliced, MCG codebook.

**This recipe deliberately differs from the checkpoint's own
`calibration_encoder/README.md` in one place: the MTP draft layer (78) routed
experts are trellis-encoded with the identical calibrated-LDLQ math as layers
3..77** — from a serving-time capture, because the draft layer never executes
in the offline prefill pass — instead of being carried BF16 and swapped in
afterwards. The published checkpoint
originally shipped with a BF16 MTP head; the trellis MTP-78 head was produced
with the same encoder + corpus, validated at BF16 acceptance parity, and
grafted on later (checkpoint commit `41a1c8ae`). This page folds that step
into the pipeline where it belongs, with patches against the published
scripts. Nothing else in the recipe is changed.

Lower KLD is better; higher acceptance length is better.

## Artifact summary

| Field | Value |
|---|---|
| Base model | [`zai-org/GLM-5.2`](https://huggingface.co/zai-org/GLM-5.2) (BF16, 78 layers + MTP, 256 routed experts/layer) |
| Quantized | routed MoE expert `gate_proj`/`up_proj`/`down_proj`, all 256 experts, **layers 3..78 (incl. MTP layer 78)** |
| Kept BF16 (byte-exact) | attention, dense MLPs (layers 0..2), shared experts, router/gates, embeddings, `lm_head`, layer-78 non-expert tensors (`eh_proj`, `enorm`, `hnorm`, `shared_head.norm`) |
| Format | EXL3 Trellis, 3.0 bpw, MCG codebook (`0xCBAC1FED`), TP4 rank-sliced (`.rank{0..3}.{trellis,suh,svh,mcg}`) |
| Hessians | calibrated LDLQ (`finalize_capture_H` -> block LDL -> blocked LDLQ with error feedback), exllamav3 0.0.43 reference math |
| Calibration corpus | `reap_recall_calib.jsonl`, 12,228 rows, 4 axes, SHA-256 `cf247acc7c5da9f0600c7d6ab3b7c2fcfc54ec30b794e3b6047559285fa44df4` |
| Capture (layers 3..77) | offline prefill capture, 1,050,468 tokens/layer, natural top-8 routing |
| Capture (layer 78) | live-serving capture with MTP enabled, 7,288,310 tokens (full corpus) |
| Checkpoint size | `total_size` 316,304,795,648 bytes (tr3 MTP-78 head; 15,662,567,424 bytes smaller than the BF16-head variant) |
| Encoder | `encode_tr3_v31.py` (SHA-256 `e9a85a47e165c8d8644354cef611efbb81dfd9ba88544ca59f0c80ee6bc75032`) + `encode_b300.py` adapter; patched per this page for layer 78 |

The `quant_method: modelopt` field in the emitted `config.json` is a
loader-dispatch shim only — the artifact contains zero NVFP4 expert payloads.

## Why quantize the MTP head too

The MTP layer is a full MoE decoder layer (256 experts, same shapes as layers
3..77). Carried BF16 it costs ~19.3 GB on disk and ~4.8 GB/GPU at TP4 —
VRAM that comes straight out of KV cache. Measured on 4x RTX PRO 6000
Blackwell (TP4/DCP4, MTP-3; decode rows from the checkpoint's published
RELEASE_TEST_SUITE A/B):

| Metric | BF16 MTP head | Trellis 3.0-bpw MTP head |
|---|---|---|
| Draft weights VRAM (TP4) | ~4.8 GB/GPU | ~0.95 GB/GPU |
| Mean acceptance length (identical 20-prompt greedy bench, 300 tok/req) | 3.054 (n=57 windows) | 3.06 (n=30 windows) — parity |
| GPU KV capacity @ util 0.96, auto-profile | ~680K tokens | **1,132,544 tokens (~ +66%)** |
| Decode C1/C4/C8 (auto-profiled KV, util 0.96, `VLLM_EXL3_TRELLIS_MIN_M=1`) | 87.5 / 219.3 / 308.1 t/s | 89.7 / 225.3 / 293.5 t/s (~neutral) |
| Long-context + doc-generation checks | pass | pass (Estonia 10/10, lavd 5E/5N/0F) |

Prompt-logit KLD of the target model is unchanged by construction: layer 78
never executes in the target forward pass (it is the speculative draft), and
MTP acceptance is lossless — rejected drafts are recomputed by the target.
The base artifact's measured KLD vs BF16 reference logits (WikiText-2, one
2048-token window, 5 fresh-boot runs) is 0.100-0.102 (mean 0.1012) with fp8 KV cache and 0.116-0.117 (mean
0.1161) with nvfp4 KV cache; those numbers apply to this checkpoint
unchanged.

## Pipeline overview

```
                 zai-org/GLM-5.2 (BF16, 282 shards)
                              |
   [A] offline prefill capture, layers 3..77
       capture_b300.py (vLLM TP8, enforce_eager; each window's layers striped
       across the 8 ranks, one owner rank per layer; >=1,048,576 tok/layer;
       <=8-layer windows in /dev/shm)
                              |
   [B] MTP layer-78 capture — one of:
       (a) the published capture dataset (malaiwah/GLM-5.2-MTP78-calibration-
           capture): reconstruct x.bin/ids.bin, no serving stack needed; or
       (b) serve an MTP-capable GLM-5.2 deployment (reference: the as-shipped
           tr3 checkpoint, BF16 draft head) with the mtp78_xcapture.py plugin
           armed, and drive the corpus once (7,288,310 tokens)
                              |
       ./convert_b300.sh mtp78-import
       (finalize_mtp78_capture.py: audits the payload, corpus-pinned
       fingerprint, writes layer_078/layer_manifest.json)
                              |
   [C] calibrated LDLQ / trellis encode, per <=8-layer window, now incl. LAYERS=78
       encode_b300.py --encode  ->  encode_tr3_v31.py (v3.1 lockstep, 8 workers x 8 GPUs)
                              |
   [D] ./convert_b300.sh assemble
       merged shards (layer 78 = 23 BF16 non-expert + 12,288 trellis tensors),
       config.json (hybrid_tr3_tail.moe_layers [3,78], ignore model.layers.78.eh_proj*),
       tier_bitmap.json (incl. "78"), calibration_manifest.json, MANIFEST.sha256
```

Stages A, C, D are the published pipeline unchanged except for the layer-78
scope; stage B and the import step are the addition. Note the loop stage B
closes: a serving capture needs an MTP-capable deployment, and the patched
one-pass build's own output does not exist yet at that point (the patched
assembly hard-requires layer 78). A from-scratch run therefore either
(a) imports the published capture dataset — the layer-78 leg becomes pure
compute — or (b) captures on an independent deployment; the reference capture
served the as-shipped tr3 checkpoint (quantized layers 3..77 + BF16 draft
head), so the captured hidden states match the deployed target stack. The
shipped `-MTP78` artifact took route (b) plus the graft; these patches fold
the same capture into one encode+assemble pass.

## Requirements

- **exllamav3 == 0.0.43** with its source package installed. The six quantizer
  ops (`had_r_128`, `pack_trellis`, `quantize_tiles`, `reconstruct`,
  `reconstruct_slice`, `unpack_trellis`) are built from its sources as an
  `sm_100` extension by `bootstrap_ext_b300.py`; the exllamav3 Python package
  is never imported at runtime. The bootstrap refuses any other version.
- NVIDIA Blackwell GPUs. Reference run: 8x B300 (SM 10.3 GPUs; quantizer ops
  built for arch 10.0, `TORCH_CUDA_ARCH_LIST=10.0`), capture at TP8, tensors
  packed for TP4. CUDA 12.9.
- The BF16 base `zai-org/GLM-5.2` (~1.51 TB), large tmpfs for capture windows,
  ~0.5 TB scratch for assembly. The orchestrator enforces disk/RAM guards.
- For stage B route (b) only: an MTP-capable serving deployment of GLM-5.2
  (the reference used a vLLM RC2-lineage image on 4x RTX PRO 6000, serving
  the as-shipped tr3 checkpoint; the capture plugin is backend-agnostic at
  the router level). Route (a) — the published capture dataset — needs no
  serving stack at all.

### exllamav3 version note (0.0.43 vs 1.x)

This pipeline pins **exllamav3 0.0.43** (released 2026-06-14). Upstream has
since moved to **1.x** — 1.0.0 (2026-07-14) changed the **default trellis
codebook from MCG to MUL1**; 1.2.0 is current at the time of writing. The
LDLQ quantization math this pipeline vendors is unchanged between 0.0.43 and
1.x, but the codebook default is not:

- This artifact (and its serving kernels) use **MCG `0xCBAC1FED`** tensors.
- Byte-for-byte reproduction of this checkpoint **requires 0.0.43** — the
  version guards in `bootstrap_ext_b300.py` / `encode_b300.py` are there on
  purpose.
- Porting the pipeline to 1.x should be possible (pin the codebook to MCG
  explicitly), but produces an unverified variant and is out of scope here.

## Calibration corpus

JSONL, one object per line, 12,228 rows balanced across four axes
(~3,057 each): `axis1_general`, `axis2_legal`, `axis3_code_agentic`,
`axis4_reasoning_termination`.

```json
{"axis": "axis2_legal", "source": "neo4j_headnote:text", "text": "{\"messages\":[...]}", "meta": {...}}
```

Routing during capture is **natural top-8**: the router's own
`top8(sigmoid(x @ W_gate^T) + e_score_correction_bias)` decides which tokens
reach which expert (recomputed in-hook; under `--verify-engine` the recompute
is additionally audited against the served router's own logits and must agree
on >=99% of tokens). Experts are never force-routed; experts that receive
fewer than 1,024 routed tokens fall back to the layer-level Hessian at encode
time (recorded in the done JSON, never fatal; the reference run had zero).

To calibrate on your own data keep the same schema and re-point `--corpus`;
the pinned corpus SHA in `capture_b300.py` exists to reproduce *this* build
byte-for-byte — relax it if you substitute a corpus.

## Stage A — plan + prefill capture (layers 3..77)

```bash
export WORK_ROOT=/workspace/tr3
export BF16_SRC=/workspace/bf16                       # zai-org/GLM-5.2 (BF16)
export OWNER_CORPUS=$WORK_ROOT/calib/reap_recall_calib.jsonl
export BASE_ENCODER_PY=$WORK_ROOT/encode_tr3_v31.py
export CUDA_HOME=/usr/local/cuda-12.9

./convert_b300.sh preflight     # env + corpus + HBM smoke checks
./convert_b300.sh ext           # build the sm_100 exllamav3 0.0.43 ops
./convert_b300.sh plan          # deterministic capture manifest
```

The capture boots vLLM TP8 with `enforce_eager`, prefix caching off,
`max_tokens=1`, `ignore_eos` — captured tokens equal the sum of prompt
lengths exactly (asserted per layer). A `forward_pre_hook` on
`model.layers.{L}.mlp.experts` streams `x.bin` (bf16-as-int16 `[N, 6144]`)
and `ids.bin` (uint8 `[N, 8]`) into `/dev/shm`, windows of at most 8 layers
at a time. Each window's layers are striped across the 8 TP ranks (one owner
rank per layer, `assigned = active_layers[rank::world]`; the MoE input is
TP-replicated, so a single rank sees the full token stream; the owner rank is
recorded in `layer_manifest.json`). Rank-0-only capture is the stage-B
plugin's behavior, not stage A's:

```bash
for W in 3-10 11-18 19-26 27-34 35-42 43-50 51-58 59-66 67-74 75-77; do
  LAYERS=$W ./convert_b300.sh capture-window
  LAYERS=$W ./convert_b300.sh encode-window
done
```

Per layer: 1,050,468 tokens, `layer_manifest.json` with payload SHA-256s and
the full 256-expert routed-count histogram.

## Stage B — MTP layer-78 capture (serving)

The MTP module never executes during the offline prefill capture — it is the
speculative draft, and it only runs when speculative decoding is enabled. Its
expert inputs are therefore captured **while serving the corpus with MTP on**,
using the `mtp78_xcapture.py` vLLM general-plugin published in
[`malaiwah/GLM-5.2-EXL3-TR3-MTP78`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-TR3-MTP78)
`tools/` (which also documents the plugin's deadlock-free ring design: the
forward path does device-side async enqueue only; a background thread drains
chunks; TP rank 0 captures; `GO`/`STOP` marker files gate the window so
boot/profiling traffic is excluded).

```bash
# serve the checkpoint with MTP enabled and the plugin armed:
VLLM_MTP_CAPTURE_DIR=/workspace/mtp78_capture \
VLLM_MTP_CAPTURE_MAX_TOKENS=8000000 \
MTP_CAPTURE_LAYER_PREFIX=model.layers.78.mlp \
vllm serve ... --speculative-config '{"method":"mtp","num_speculative_tokens":3,...}'

touch /workspace/mtp78_capture/GO      # open the capture window
python3 drive_corpus.py                # drive all 12,228 rows, prefill-only
touch /workspace/mtp78_capture/STOP    # then one tiny request to finalize
```

The corpus drive matches stage-A semantics exactly: one request per row, raw
`record["text"]`, checkpoint tokenizer, 4,096-token cap, `max_tokens=1`,
temperature 0, `ignore_eos`. Because the draft prefills every prompt token,
one pass over the corpus yields the **full-corpus** capture: **7,288,310
tokens** with the router's ground-truth top-8 ids (the reference capture is
published as
[`malaiwah/GLM-5.2-MTP78-calibration-capture`](https://huggingface.co/datasets/malaiwah/GLM-5.2-MTP78-calibration-capture)
— with it, re-encoding layer 78 is a pure-compute job and stage B can be
skipped entirely; the dataset ships safetensors shards, so reconstruct
`x.bin`/`ids.bin` with `scripts/load_capture_safetensors.py` from the
malaiwah collector repo — `capture_done.json` is included — and point
`MTP78_CAPTURE_SRC` at the result).

Two serving-image caveats observed on the reference run (RC2-lineage image):
the draft crashed on prefill steps larger than its decode-path threshold, so
the serve config set `--max-num-batched-tokens` to that threshold and let the
scheduler chunk every prefill; and prompts whose final chunk would be 1..8
tokens were trimmed to a 128-multiple to dodge a fork-kernel fault. Neither
affects capture semantics (the ids are the router's own output either way).

Import the capture into the b300 layout (validates row parity, id range,
routed histogram, bf16 sanity; writes `layer_078/layer_manifest.json` with
the same `glm52-b300-layer-capture-v1` schema stage A emits):

```bash
# one-time: the finalize script ships with this page, not the HF bundle
cp glm52-exl3-tr3-quantization-2026-07-26/scripts/finalize_mtp78_capture.py \
   <calibration_encoder>/            # next to convert_b300.sh (stage fail-closes if absent)

MTP78_CAPTURE_SRC=/workspace/mtp78_capture ./convert_b300.sh mtp78-import
```

## Stage C — calibrated LDLQ / trellis encode (now incl. layer 78)

```bash
LAYERS=78 ./convert_b300.sh encode-window
```

The patched `--layers` default is `3-78`; a full-range invocation fail-closes
up front in `check_capture` unless the layer-78 capture is already imported
(the windowed loop above never hits this).

Identical math to layers 3..77, per expert, per TP4 slice:

- gate/up (k=6144): `H_e = X_e^T X_e` over the tokens routed to expert `e`;
  all 8 gate/up slices of an expert share one 6144x6144 `H` (reference
  shared-qmap semantics — `su` drawn once per expert at first finalize, `sv`
  per slice).
- down (k=512): `I_e = silu(X_e Wg^T) * (X_e Wu^T)` computed offline from the
  stored routed `x` with the BF16 gate/up weights; slice `r` uses the
  `[512r, 512r+512)` **diagonal block** of `I_e^T I_e`.
- `finalize_capture_H` (mean + 0.025 sigma damping + su transform) -> block
  LDL(16) with damping retries -> LDLQ (128-row spans, 16-row blocks,
  bottom-up, error feedback) -> MCG trellis pack at 3 bits, exllamav3 0.0.43
  math vendored verbatim, v3.1 cross-slice lockstep + pooled GSS (oracle gate:
  byte-equality vs v2 sequential on real slices).

Layer 78's expert tensors follow the identical output schema
(`model.layers.78.mlp.experts.{E}.{proj}.rank{r}.{trellis|suh|svh|mcg}`,
12,288 tensors), so the encoder needs no naming changes — only the layer-range
guards (see patch list below).

## Stage D — assemble

```bash
./convert_b300.sh assemble
```

With the patches applied, assembly emits for layer 78 exactly what it emits
for every other MoE layer, and the artifact-level metadata comes out in its
final (post-graft-equivalent) state directly:

- `model-layer-078.safetensors`: the 23 BF16 non-expert tensors + 12,288
  trellis tensors (the BF16 expert `.weight` entries are dropped and audited
  as replaced).
- `config.json`: `hybrid_tr3_tail.moe_layers = [3, 78]`;
  `quantization_config.ignore` carries `model.layers.78.eh_proj*` instead of
  `model.layers.78*` (attention/shared-experts/router wildcards already cover
  the rest of the layer's non-expert linears).
- `tier_bitmap.json`: gains `"78"` with the real per-expert
  `expert_rel_rt_mse` from the encode (the graft route had to add this entry
  by hand).
- Audits scale to 76 layers: 933,888 trellis tensors, 58,368 replaced BF16
  weights.
- Config-level `calibration.tokens_per_layer` (1,050,468) describes the
  prefill plan; layer 78's 7,288,310-token serving capture is recorded in its
  done JSON and in the per-layer SHA maps.

## Serving requirements (tr3 MTP-78)

Loading a **trellis** layer-78 draft needs runtime support the BF16 head did
not (all carried in
[local-inference-lab/vllm#139](https://github.com/local-inference-lab/vllm/pull/139)
— commit `f57d093f` "exl3(mtp): load rank-sliced tr3 experts in the MTP
draft layer", hash as of the current v20 rebase; PR open at the time of
writing):

1. **Draft quant-config hydration** — `get_draft_quant_config` must run the
   same `maybe_update_config` the target runs, so the draft's
   `Exl3Config.rank_sliced_metadata` is populated (otherwise the MTP experts
   silently build a stock FusedMoE and fail on the tr3 tensors).
2. **Rank-slice name normalization** — `deepseek_mtp.py::load_weights` must
   call `quant_config.normalize_rank_sliced_weight_name(name)` like
   `deepseek_v2.py` does (otherwise: `KeyError ...routed_experts.w2_rank0.mcg`).
   On RC2-lineage images this is the published 1-hunk
   `deepseek_mtp_normalization.patch`.
3. **`VLLM_EXL3_TRELLIS_MIN_M=1`** — the trellis kernel's decode plan window
   defaults to `[4, 32]`; the MTP-3 draft is cudagraph-captured at m=1,2,3,
   which is illegal during capture without lowering the floor. Required for
   any tr3 MTP layer.

Everything else (`-tp 4`, the `hybrid_tr3_tail` block driving the pre-planned
CUDA-graph-safe kernel path) is unchanged from the model card's runtime
section.

## Retrofit path (already-built 3..77 checkpoints)

An existing checkpoint with a BF16 MTP head does not need a re-run: encode
layer 78 from the published capture dataset and graft the merged shard
(`build_graft.py` in the malaiwah `tools/`: hardlink clone, merged layer-078
shard, index rebuild, the ignore-list and `moe_layers` config edits). That
route — plus two by-hand touches the shipped artifact carries
(`model.layers.78.eh_proj*` retained on the ignore list, where
`build_graft.py` strips every `layers.78` entry, and the `tier_bitmap`
`"78"` entry) — produced the shipped `-MTP78` checkpoint (commit `41a1c8ae`); the patches on this page produce the same end state in one pass,
with truthful per-layer calibration metadata baked in from the start.

## Changes vs the published `calibration_encoder/` scripts

The published bundle reproduces the checkpoint **as originally shipped**
(BF16 MTP head). The patches in
[`glm52-exl3-tr3-quantization-2026-07-26/patches/`](glm52-exl3-tr3-quantization-2026-07-26/patches/)
apply cleanly to the pinned files and fold the MTP-78 leg in. Every hunk:

### `01-encode_tr3_v31-mtp78.patch` (5 hunks)

| Change | Why |
|---|---|
| `moe_layers()`: `range(3, 78)` -> `range(3, 79)` | Layer 78 joins the encode/assembly set; `hybrid_tr3_tail.moe_layers` becomes `[3, 78]` automatically, and the assembly `dropped`/`tier_bitmap` sets follow this list. |
| layer-guard assert: `3 <= L < 78` -> `3 <= L <= 78` | `--layers 78` was rejected as "not a main MoE layer". |
| `--layers` default `3-77` -> `3-78`, help text | Full-run default covers the MTP layer. |
| Header/comment scope text | Documentation consistency (`3..77` -> `3..78`). |
| Side effect: patched v31's own standalone `--assemble` now also requires an encoded layer 78 | Unused by this pipeline (the adapter assembles); noted for direct-v31 users. Two cosmetic v31 strings (a usage example reading `3-77`, a bench log reading `75 layers`) are deliberately left unpatched to keep the encoder diff and its SHA pin minimal. |

### `02-encode_b300-mtp78.patch` (13 hunks)

| Change | Why |
|---|---|
| `EXPECTED_TOTAL_TR3_TENSORS = 75 * ...` -> `76 * ...`; `EXPECTED_REPLACED_WEIGHTS = 75 * 256 * 3` -> `76 * ...` | Output audits count layer 78's 12,288 trellis tensors / 768 replaced weights. |
| `BASE_ENCODER_SHA256` re-pinned to `3bc6839fbb35074a369673748b1effa9b74346cfe7162390e5bbb943926004e4` | The adapter refuses an encoder whose bytes differ from the pin; the patched `encode_tr3_v31.py` hashes differently by design. |
| recipe scope `"layers": [3, 77]` -> `[3, 78]` | The recipe fingerprint is a whole-run property; **it must be set before encoding layer 3**, and all 76 done JSONs must carry the same fingerprint (a 3..77 work dir cannot be retrofitted — use the graft path for that). |
| dispatch-shim ignore: `model.layers.78*` -> `model.layers.78.eh_proj*` | Layer-78 experts now go down the intercepted path; only `eh_proj` (a layer-78-specific linear no generic wildcard covers) stays BF16-ignored. Matches the shipped `-MTP78` config exactly. |
| `hybrid_tr3_tail`: `moe_layers [3,77]` -> `[3,78]`, scope strings, BF16 list, dispatch note | Runtime discovers trellis layers from `moe_layers`; scope text stops claiming the MTP head is BF16. |
| done-JSON capture block: prefer `layer_manifest` `tokens`/`capture_fingerprint` over plan values | Layer 78's manifest records the serving capture truthfully (7,288,310 tokens, its own fingerprint); layers 3..77 are unaffected (their manifests carry the plan values). |
| `--layers` default `3-77` -> `3-78` | Same as the encoder. |
| New `validate_mtp78_manifest()` + constants (`MTP_LAYER`, transport/schema markers, 1,048,576-token floor); `check_capture` and `LayerCalibRAM` dispatch layer 78 to it | The serving capture carries its own corpus-pinned fingerprint (recomputed via `canonical_hash`, fail-closed — tampering with tokens or payload SHAs is detected) and its own token count; layers 3..77 keep the plan checks verbatim. Without this the adapter hard-rejects the layer-78 manifest before any encode. |
| Assemble re-hash log made dynamic (`57,600` -> `len(layers)*768`) | Count correctness at 76 layers. |

### `03-convert_b300-mtp78.patch` (3 hunks)

| Change | Why |
|---|---|
| New `mtp78-import` stage (`MTP78_CAPTURE_SRC=<dir> ./convert_b300.sh mtp78-import`) | Imports/validates the serving-time capture into `CAPTURE_DIR/layer_078/` behind the same RAM guard the prefill windows use; fail-closes (and `py_compile`s) if `finalize_mtp78_capture.py` has not been copied next to `convert_b300.sh`. |
| Case-switch + usage text | Stage discoverability. |

### New files

| File | Role |
|---|---|
| [`scripts/finalize_mtp78_capture.py`](glm52-exl3-tr3-quantization-2026-07-26/scripts/finalize_mtp78_capture.py) | Validates the plugin capture (row parity, id range, routed histogram, duplicate-id pre-check, bf16 sanity, drop budget) and writes `layer_078/layer_manifest.json` in the stage-A schema with a corpus-pinned canonical fingerprint the patched adapter recomputes and enforces; `encode_b300.py --encode --layers 78` consumes it unchanged. |

Not vendored here: `mtp78_xcapture.py`, `drive_corpus.py`,
`deepseek_mtp_normalization.patch`, `build_graft.py` — published and
maintained in `malaiwah/GLM-5.2-EXL3-TR3-MTP78` `tools/`.

## Links

- Checkpoint: [`brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw) (tr3 MTP-78 head since commit `41a1c8ae`)
- Reproduction bundle (as-shipped recipe): [`calibration_encoder/`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw/tree/main/calibration_encoder)
- MTP-78 overlays + tools: [`malaiwah/GLM-5.2-EXL3-TR3-MTP78`](https://huggingface.co/malaiwah/GLM-5.2-EXL3-TR3-MTP78)
- Layer-78 capture dataset: [`malaiwah/GLM-5.2-MTP78-calibration-capture`](https://huggingface.co/datasets/malaiwah/GLM-5.2-MTP78-calibration-capture)
- Runtime loader fixes: [local-inference-lab/vllm#139](https://github.com/local-inference-lab/vllm/pull/139); sparkinfer counterpart [local-inference-lab/sparkinfer#49](https://github.com/local-inference-lab/sparkinfer/pull/49)
- exllamav3 (LDLQ/trellis reference math, 0.0.43): [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3)

## Credits

zai-org (GLM-5.2 base, MIT) - Brandon Music (EXL3 TR3 encoder, owner corpus,
base checkpoint, runtime PRs) - Josh Cartu / jcartu (MTP78 recipe, rank-sliced
MTP runtime) - malaiwah (layer-78 capture/encode/validation + published
overlays and capture dataset) - Luke Alonso (b12x) - turboderp-org
(exllamav3, MIT).
