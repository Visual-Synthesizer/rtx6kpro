#!/usr/bin/env python3
"""Export and hash the canonical BF16 Kimi K3 LM-head tensor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_KEY = "language_model.lm_head.weight"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    if not tensor.is_contiguous():
        raise RuntimeError("Raw tensor hashing requires contiguous storage")
    byte_view = tensor.view(torch.uint8).numpy()
    data = memoryview(byte_view).cast("B")
    digest = hashlib.sha256()
    block_size = 16 * 1024 * 1024
    for offset in range(0, data.nbytes, block_size):
        digest.update(data[offset : offset + block_size])
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-key", default=DEFAULT_KEY)
    parser.add_argument("--expected-vocab", type=int, default=163840)
    parser.add_argument("--expected-hidden-width", type=int, default=7168)
    parser.add_argument("--expected-raw-sha256")
    args = parser.parse_args()

    index_path = args.model_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    try:
        shard_name = index["weight_map"][args.tensor_key]
    except KeyError as error:
        raise RuntimeError(
            f"Tensor {args.tensor_key} is absent from {index_path}"
        ) from error
    shard_path = args.model_dir / shard_name
    with safe_open(shard_path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(args.tensor_key)

    expected_shape = [args.expected_vocab, args.expected_hidden_width]
    if tensor.dtype != torch.bfloat16 or list(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"Expected BF16 LM head with shape {expected_shape}; "
            f"got {tensor.dtype} {list(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    raw_sha256 = sha256_tensor(tensor)
    if args.expected_raw_sha256 and raw_sha256 != args.expected_raw_sha256:
        raise RuntimeError(
            "LM-head raw tensor hash differs: "
            f"expected {args.expected_raw_sha256}, got {raw_sha256}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "weight.safetensors"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(
        {"weight": tensor},
        str(temporary),
        metadata={
            "checkpoint_tensor_key": args.tensor_key,
            "raw_tensor_sha256": raw_sha256,
        },
    )
    temporary.replace(output_path)

    manifest = {
        "checkpoint_index_sha256": sha256_file(index_path),
        "checkpoint_tensor_key": args.tensor_key,
        "dtype": "BF16",
        "file": output_path.name,
        "file_sha256": sha256_file(output_path),
        "format_version": 1,
        "key": "weight",
        "kind": "Kimi K3 canonical LM-head weight",
        "raw_tensor_sha256": raw_sha256,
        "shape": expected_shape,
        "size_bytes": output_path.stat().st_size,
        "source_shard": shard_name,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
