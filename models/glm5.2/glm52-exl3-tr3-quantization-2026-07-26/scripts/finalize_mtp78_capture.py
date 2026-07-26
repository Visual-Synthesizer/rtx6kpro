#!/usr/bin/env python3
"""Import a serving-time MTP layer-78 capture into the b300 capture layout.

The MTP draft layer (78) never executes during the offline prefill capture
pass (`capture_b300.py` hooks the target model; the draft only runs under
speculative decoding), so its expert inputs are captured while SERVING the
calibration corpus with MTP enabled — via the `mtp78_xcapture.py` vLLM
general-plugin (see malaiwah/GLM-5.2-EXL3-TR3-MTP78 `tools/`), which writes
Brandon Music's exact TR3 capture format:

    x.bin   : int16 (bf16 bit-pattern) [tokens, 6144]  MoE-input hidden states
    ids.bin : uint8                    [tokens, 8]     routed top-8 expert ids
    capture_done.json : {tokens, dropped_rows, target_prefix, ...}

This script validates that payload, computes the audit fields, and emits
`CAPTURE_DIR/layer_078/{x.bin,ids.bin,layer_manifest.json}` with the same
`glm52-b300-layer-capture-v1` schema `capture_b300.py` writes for layers
3..77, so `encode_b300.py --encode --layers 78` consumes it unchanged.

Differences vs a prefill-captured layer, recorded honestly in the manifest:
  - `tokens` is the serving-capture count (full corpus, e.g. 7,288,310), not
    the prefill plan quota; the patched `encode_b300.py` records the
    layer_manifest value into the layer's done JSON.
  - `capture_fingerprint` is derived from this payload (corpus sha + prefix +
    payload shas), not from the prefill capture plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

HIDDEN = 6144
TOPK = 8
NUM_EXPERTS = 256
TARGET_PREFIX = "model.layers.78.mlp"
LAYER = 78


def sha256_file(path: Path, chunk: int = 64 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True,
                    help="mtp78_xcapture output dir (x.bin, ids.bin, capture_done.json)")
    ap.add_argument("--capture-dir", required=True,
                    help="b300 CAPTURE_DIR; layer_078/ is created inside it")
    ap.add_argument("--corpus", required=True,
                    help="calibration corpus jsonl (sha256 recorded in the manifest)")
    ap.add_argument("--min-tokens", type=int, default=1_048_576,
                    help="refuse captures smaller than the b300 per-layer floor")
    ap.add_argument("--max-dropped-frac", type=float, default=0.001,
                    help="refuse if the ring dropped more than this fraction of rows")
    args = ap.parse_args()

    src = Path(args.src_dir)
    x_path, ids_path = src / "x.bin", src / "ids.bin"
    done_path = src / "capture_done.json"
    for path in (x_path, ids_path, done_path):
        if not path.is_file():
            raise SystemExit(f"FATAL: missing {path}")
    done = json.loads(done_path.read_text())

    if int(done.get("hidden", HIDDEN)) != HIDDEN or int(done.get("topk", TOPK)) != TOPK:
        raise SystemExit("FATAL: capture_done geometry mismatch")
    if done.get("target_prefix") != TARGET_PREFIX:
        raise SystemExit(
            f"FATAL: capture target_prefix={done.get('target_prefix')!r}, "
            f"expected {TARGET_PREFIX!r}"
        )

    x_bytes, ids_bytes = x_path.stat().st_size, ids_path.stat().st_size
    if x_bytes % (HIDDEN * 2) or ids_bytes % TOPK:
        raise SystemExit("FATAL: payload sizes are not row-aligned")
    tokens = x_bytes // (HIDDEN * 2)
    if ids_bytes // TOPK != tokens:
        raise SystemExit(
            f"FATAL: row mismatch x={tokens} ids={ids_bytes // TOPK} "
            "(x.bin and ids.bin must cover the same rows)"
        )
    if int(done.get("tokens", -1)) != tokens:
        raise SystemExit(
            f"FATAL: capture_done tokens={done.get('tokens')} != payload rows {tokens}"
        )
    if tokens < args.min_tokens:
        raise SystemExit(f"FATAL: only {tokens} tokens captured, floor is {args.min_tokens}")
    dropped = int(done.get("dropped_rows", 0))
    if dropped > args.max_dropped_frac * (tokens + dropped):
        raise SystemExit(
            f"FATAL: ring dropped {dropped} rows "
            f"(> {args.max_dropped_frac:.4%} of the stream)"
        )

    # Routed-count audit over the full ids payload (memory-mapped).
    ids = np.memmap(ids_path, dtype=np.uint8, mode="r", shape=(tokens, TOPK))
    if int(ids.max()) >= NUM_EXPERTS:
        raise SystemExit(f"FATAL: expert id {int(ids.max())} out of range")
    routed = np.bincount(ids.reshape(-1), minlength=NUM_EXPERTS).astype(int).tolist()
    # Duplicate expert id within a token = x/ids ring skew in the capture
    # plugin; catch it here, before the ~84 GiB tmpfs copy and the encoder's
    # own late hard-fail on the same condition.
    sorted_ids = np.sort(ids, axis=1)
    if not (sorted_ids[:, 1:] != sorted_ids[:, :-1]).all():
        raise SystemExit("FATAL: duplicate expert id within a token (x/ids ring skew?)")

    # bf16 sanity on a deterministic sample: finite, non-degenerate.
    rows = np.linspace(0, tokens - 1, num=min(4096, tokens), dtype=np.int64)
    x = np.memmap(x_path, dtype=np.int16, mode="r", shape=(tokens, HIDDEN))
    import torch
    sample = torch.from_numpy(np.ascontiguousarray(x[rows])).view(torch.bfloat16).float()
    if not torch.isfinite(sample).all():
        raise SystemExit("FATAL: non-finite values in sampled x rows")
    if float(sample.abs().amax(dim=1).min()) == 0.0:
        raise SystemExit("FATAL: all-zero row in sampled x rows")

    sha_x, sha_ids = sha256_file(x_path), sha256_file(ids_path)
    corpus_sha = sha256_file(Path(args.corpus))
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "schema": "glm52-mtp78-serving-capture-v1",
                "corpus_sha256": corpus_sha,
                "target_prefix": TARGET_PREFIX,
                "tokens": tokens,
                "sha256_x": sha_x,
                "sha256_ids": sha_ids,
            },
            sort_keys=True,
            separators=(",", ":"),  # must match encode_b300.canonical_hash
        ).encode()
    ).hexdigest()

    layer_dir = Path(args.capture_dir) / f"layer_{LAYER:03d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    for name, source in (("x.bin", x_path), ("ids.bin", ids_path)):
        dest = layer_dir / name
        if dest.exists():
            dest.unlink()
        try:
            os.link(source, dest)  # same filesystem: free
        except OSError:
            import shutil
            shutil.copyfile(source, dest)

    manifest = {
        "schema": "glm52-b300-layer-capture-v1",
        "layer": LAYER,
        "capture_fingerprint": fingerprint,
        "capture_transport": "mtp78_xcapture-serving",
        "tokens": tokens,
        "hidden": HIDDEN,
        "x_dtype": "bfloat16",
        "x_bytes": x_bytes,
        "ids_topk": TOPK,
        "ids_bytes": ids_bytes,
        "sha256_x": sha_x,
        "sha256_ids": sha_ids,
        "routed_counts": routed,
        "routed_min": min(routed),
        "routed_max": max(routed),
        "cold_experts_lt1024": [e for e, count in enumerate(routed) if count < 1024],
        "dropped_rows": dropped,
        "corpus_sha256": corpus_sha,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp = layer_dir / "layer_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, layer_dir / "layer_manifest.json")

    cold = manifest["cold_experts_lt1024"]
    print(
        f"MTP78 CAPTURE IMPORTED: {tokens} tokens, routed min/max="
        f"{min(routed)}/{max(routed)}, cold(<1024)={len(cold)}, "
        f"dropped={dropped}, fingerprint={fingerprint[:16]}…"
    )
    if cold:
        print(f"  cold experts fall back to layer-level H at encode: {cold[:16]}"
              + (" …" if len(cold) > 16 else ""))


if __name__ == "__main__":
    main()
