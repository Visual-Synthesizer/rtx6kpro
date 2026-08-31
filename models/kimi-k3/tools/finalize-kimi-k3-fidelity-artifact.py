#!/usr/bin/env python3
"""Assemble and checksum the Kimi K3 distribution-fidelity artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CAPTURE_TOOLS = (
    "capture-kimi-k3-hidden-suite.py",
    "export-kimi-k3-lm-head.py",
    "finalize-kimi-k3-fidelity-artifact.py",
    "prepare-kimi-k3-fidelity-suite.py",
    "serve-kimi-k3-fidelity-capture.sh",
    "validate-kimi-k3-capability-overlap.py",
    "validate-kimi-k3-fidelity-artifact.py",
    "validate-kimi-k3-source-separation.py",
)
COMPARATORS = (
    "compare-kimi-k3-fidelity-receipts.py",
    "compare-kimi-k3-hidden-replay.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def copy_tools(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in names:
        source_path = source / name
        if not source_path.is_file():
            raise RuntimeError(f"Required tool is missing: {source_path}")
        shutil.copy2(source_path, destination / name)


def normalize_tensor_manifest(manifest_path: Path, runtime_path: Path) -> None:
    """Remove host-local capture paths while retaining immutable identities."""
    manifest = load_json(manifest_path)
    manifest["runtime_manifest"] = Path(
        os.path.relpath(runtime_path, start=manifest_path.parent)
    ).as_posix()
    for record in manifest.get("contexts", []):
        if record.get("raw_chunks_retained") is False:
            record.pop("capture_chunk_dir", None)
            record.pop("capture_chunks", None)
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--readme", type=Path, required=True)
    args = parser.parse_args()

    copy_tools(args.tools_dir, args.artifact / "capture-tools", CAPTURE_TOOLS)
    copy_tools(args.tools_dir, args.artifact / "comparators", COMPARATORS)
    shutil.copy2(args.readme, args.artifact / "README.md")

    suite_path = args.artifact / "suite-manifest.json"
    runtime_path = args.artifact / "capture-runtime.json"
    reference_manifest_path = args.artifact / "reference-hidden" / "manifest.json"
    lm_head_manifest_path = args.artifact / "lm-head" / "manifest.json"
    tensor_manifest_paths = [
        reference_manifest_path,
        *(args.artifact / "sentinel-hidden" / repeat / "manifest.json"
          for repeat in ("repeat-01", "repeat-02")),
        *(args.artifact / "sentinel-live-logits" / repeat / "manifest.json"
          for repeat in ("repeat-00", "repeat-01", "repeat-02")),
    ]
    for tensor_manifest_path in tensor_manifest_paths:
        normalize_tensor_manifest(tensor_manifest_path, runtime_path)
    suite = load_json(suite_path)
    runtime = load_json(runtime_path)
    reference = load_json(reference_manifest_path)
    lm_head = load_json(lm_head_manifest_path)
    if len(reference.get("contexts", [])) != 1024:
        raise RuntimeError("Official reference capture does not contain 1,024 contexts")

    manifest = {
        "artifact_kind": "Kimi K3 teacher-forced distribution-fidelity reference",
        "capture_runtime": {
            "file": "capture-runtime.json",
            "sha256": sha256_file(runtime_path),
            "status": runtime["status"],
        },
        "checkpoint": runtime["checkpoint"],
        "context_count": suite["context_count"],
        "context_length": suite["context_length"],
        "created_utc": datetime.now(UTC).isoformat(),
        "format_version": 1,
        "lm_head": {
            "file": "lm-head/weight.safetensors",
            "file_sha256": lm_head["file_sha256"],
            "raw_tensor_sha256": lm_head["raw_tensor_sha256"],
            "shape": lm_head["shape"],
        },
        "partitions": suite["partition_counts"],
        "reference_hidden": {
            "context_count": len(reference["contexts"]),
            "manifest": "reference-hidden/manifest.json",
            "manifest_sha256": sha256_file(reference_manifest_path),
            "shape_per_context": [2047, 7168],
            "total_size_bytes": reference["total_size_bytes"],
        },
        "scored_positions_per_context": suite["scored_positions_per_context"],
        "sentinel_count": suite["sentinel_count"],
        "status": "qualified",
        "suite": {
            "manifest": "suite-manifest.json",
            "manifest_sha256": sha256_file(suite_path),
            "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
        },
        "total_scored_positions": suite["total_scored_positions"],
    }
    manifest_path = args.artifact / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_manifest.replace(manifest_path)

    checksum_path = args.artifact / "checksums.txt"
    files = sorted(
        path
        for path in args.artifact.rglob("*")
        if path.is_file()
        and path != checksum_path
        and path.relative_to(args.artifact).parts[0] != ".cache"
    )
    temporary_checksums = checksum_path.with_suffix(".txt.tmp")
    with temporary_checksums.open("w", encoding="utf-8") as handle:
        for path in files:
            relative = path.relative_to(args.artifact)
            handle.write(f"{sha256_file(path)}  {relative}\n")
    temporary_checksums.replace(checksum_path)
    print(
        json.dumps(
            {
                "checksummed_files": len(files),
                "manifest_sha256": sha256_file(manifest_path),
                "payload_size_bytes": sum(path.stat().st_size for path in files),
                "status": "qualified",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
