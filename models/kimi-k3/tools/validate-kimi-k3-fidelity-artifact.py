#!/usr/bin/env python3
"""Validate the Kimi K3 distribution-fidelity artifact structure and hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from safetensors import safe_open

EXPECTED_ALLOCATIONS = {
    "chinese": 96,
    "code_tests_documentation_issues": 128,
    "dialogue_instruction_assistance": 128,
    "encyclopedic_factual": 128,
    "literary_narrative_creative": 96,
    "news_history_economics_legal_essays": 96,
    "other_multilingual": 48,
    "scientific_technical": 128,
    "structured_data_tools_apis_tables": 48,
    "worked_math_science_formal_reasoning": 128,
}
SENTINEL_REPEAT_PAIRS = (
    (
        "00-vs-01",
        "reference-hidden",
        "sentinel-hidden/repeat-01",
        "sentinel-live-logits/repeat-00",
        "sentinel-live-logits/repeat-01",
    ),
    (
        "00-vs-02",
        "reference-hidden",
        "sentinel-hidden/repeat-02",
        "sentinel-live-logits/repeat-00",
        "sentinel-live-logits/repeat-02",
    ),
    (
        "01-vs-02",
        "sentinel-hidden/repeat-01",
        "sentinel-hidden/repeat-02",
        "sentinel-live-logits/repeat-01",
        "sentinel-live-logits/repeat-02",
    ),
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


def validate_tensor_directory(
    directory: Path,
    *,
    runtime_manifest_sha256: str,
    expected_contexts: set[int],
    key: str,
    dtype: str,
    shape: list[int],
    verify_hashes: bool,
) -> int:
    manifest_path = directory / "manifest.json"
    manifest = load_json(manifest_path)
    runtime_reference = manifest.get("runtime_manifest")
    if not isinstance(runtime_reference, str) or Path(runtime_reference).is_absolute():
        raise RuntimeError(f"Runtime-manifest path is not portable in {manifest_path}")
    runtime_path = (directory / runtime_reference).resolve()
    if not runtime_path.is_file():
        raise RuntimeError(f"Referenced runtime manifest is missing: {runtime_path}")
    if manifest.get("runtime_manifest_sha256") != runtime_manifest_sha256:
        raise RuntimeError(f"Runtime-manifest identity differs in {manifest_path}")
    if sha256_file(runtime_path) != runtime_manifest_sha256:
        raise RuntimeError(f"Runtime-manifest hash differs for {runtime_path}")
    records = manifest.get("contexts")
    if not isinstance(records, list):
        raise RuntimeError(f"Missing context records in {manifest_path}")
    indices = {int(record["context_index"]) for record in records}
    if indices != expected_contexts:
        raise RuntimeError(f"Unexpected context set in {manifest_path}")
    total_size = 0
    for record in records:
        if record.get("raw_chunks_retained") is False and (
            "capture_chunk_dir" in record or "capture_chunks" in record
        ):
            raise RuntimeError(f"Deleted raw chunks retain host-local metadata in {manifest_path}")
        path = directory / record["file"]
        if not path.is_file():
            raise RuntimeError(f"Missing tensor file: {path}")
        with safe_open(path, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != [key]:
                raise RuntimeError(f"Unexpected tensor keys in {path}")
            tensor_slice = handle.get_slice(key)
            if tensor_slice.get_dtype() != dtype:
                raise RuntimeError(f"Unexpected tensor dtype in {path}")
            if tensor_slice.get_shape() != shape:
                raise RuntimeError(f"Unexpected tensor shape in {path}")
        size = path.stat().st_size
        if int(record["size_bytes"]) != size:
            raise RuntimeError(f"Recorded size differs for {path}")
        if verify_hashes and record.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Recorded hash differs for {path}")
        total_size += size
    if int(manifest.get("total_size_bytes", -1)) != total_size:
        raise RuntimeError(f"Recorded total size differs in {manifest_path}")
    return total_size


def validate_receipt_source(
    receipt: dict[str, Any],
    *,
    role: str,
    expected_kind: str,
    expected_directory: Path,
) -> None:
    source = receipt.get(role)
    if not isinstance(source, dict):
        raise RuntimeError(f"Sentinel receipt has no {role} source")
    if source.get("kind") != expected_kind:
        raise RuntimeError(f"Sentinel receipt has an unexpected {role} kind")
    label = source.get("label")
    if not isinstance(label, str) or Path(label).is_absolute():
        raise RuntimeError(f"Sentinel receipt has a non-portable {role} label")
    expected_hash = sha256_file(expected_directory / "manifest.json")
    if source.get("manifest_sha256") != expected_hash:
        raise RuntimeError(f"Sentinel receipt has an unexpected {role} manifest")


def validate_sentinel_receipts(artifact: Path, suite_hash: str) -> dict[str, float]:
    validation = artifact / "validation"
    results: dict[str, float] = {}
    ignored_fields = {"reference", "candidate", "comparator"}
    for pair, hidden_a, hidden_b, live_a, live_b in SENTINEL_REPEAT_PAIRS:
        hidden = load_json(validation / f"sentinel-hidden-repeat-{pair}.json")
        live = load_json(validation / f"sentinel-live-repeat-{pair}.json")
        for receipt in (hidden, live):
            if receipt.get("suite_manifest_sha256") != suite_hash:
                raise RuntimeError(f"Sentinel receipt {pair} uses another token suite")
            if receipt.get("contexts") != 64 or receipt.get("positions") != 64 * 2047:
                raise RuntimeError(f"Sentinel receipt {pair} has an unexpected shape")
        validate_receipt_source(
            hidden,
            role="reference",
            expected_kind="hidden",
            expected_directory=artifact / hidden_a,
        )
        validate_receipt_source(
            hidden,
            role="candidate",
            expected_kind="hidden",
            expected_directory=artifact / hidden_b,
        )
        validate_receipt_source(
            live,
            role="reference",
            expected_kind="logits",
            expected_directory=artifact / live_a,
        )
        validate_receipt_source(
            live,
            role="candidate",
            expected_kind="logits",
            expected_directory=artifact / live_b,
        )
        hidden_metrics = {
            key: value for key, value in hidden.items() if key not in ignored_fields
        }
        live_metrics = {
            key: value for key, value in live.items() if key not in ignored_fields
        }
        if hidden_metrics != live_metrics:
            raise RuntimeError(
                f"Hidden replay and live logits disagree for sentinel pair {pair}"
            )
        results[pair] = float(hidden["kl_reference_to_candidate"]["mean"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--verify-payload-hashes", action="store_true")
    args = parser.parse_args()

    suite = load_json(args.artifact / "suite-manifest.json")
    contexts = suite.get("contexts")
    if not isinstance(contexts, list) or len(contexts) != 1024:
        raise RuntimeError("The suite must contain exactly 1,024 contexts")
    if int(suite.get("context_length", -1)) != 2048:
        raise RuntimeError("The suite context length must be 2,048 tokens")
    if int(suite.get("scored_positions_per_context", -1)) != 2047:
        raise RuntimeError("Each context must contain 2,047 scored positions")

    context_indices = {int(context["context_index"]) for context in contexts}
    if context_indices != set(range(1024)):
        raise RuntimeError("Context indices must be contiguous from 0 through 1,023")
    source_documents = {
        (
            context["dataset"],
            context.get("dataset_config"),
            context["dataset_split"],
            context["source_id"],
        )
        for context in contexts
    }
    if len(source_documents) != 1024:
        raise RuntimeError("Each context must come from a distinct source document")
    source_clusters = {
        (context["dataset"], context["source_cluster_id"]) for context in contexts
    }
    software_contexts = [
        context
        for context in contexts
        if context["allocation_stratum"]
        in {
            "code_tests_documentation_issues",
            "structured_data_tools_apis_tables",
        }
    ]
    software_clusters = {
        (context["dataset"], context["source_cluster_id"])
        for context in software_contexts
    }
    if len(software_clusters) != len(software_contexts):
        raise RuntimeError("Software contexts must use distinct repository clusters")
    if len({context["token_ids_json_sha256"] for context in contexts}) != 1024:
        raise RuntimeError("Every context must have a distinct token hash")
    observed_token_hashes: list[str] = []
    for context in contexts:
        token_path = args.artifact / context["token_file"]
        token_ids = json.loads(token_path.read_text(encoding="utf-8"))
        if not isinstance(token_ids, list) or len(token_ids) != 2048:
            raise RuntimeError(f"Unexpected token array in {token_path}")
        compact = json.dumps(token_ids, separators=(",", ":"))
        token_hash = hashlib.sha256(compact.encode("utf-8")).hexdigest()
        if token_hash != context["token_ids_json_sha256"]:
            raise RuntimeError(f"Token hash differs for {token_path}")
        observed_token_hashes.append(token_hash)
    suite_token_hash = hashlib.sha256(
        "\n".join(observed_token_hashes).encode("ascii")
    ).hexdigest()
    if suite_token_hash != suite["suite_token_hash_sha256"]:
        raise RuntimeError("The aggregate suite token hash differs")
    allocations = Counter(context["allocation_stratum"] for context in contexts)
    if dict(allocations) != EXPECTED_ALLOCATIONS:
        raise RuntimeError(f"Unexpected allocation counts: {dict(allocations)}")
    partitions = Counter(context["partition"] for context in contexts)
    if partitions != {"analysis": 768, "qualification": 256}:
        raise RuntimeError(f"Unexpected partition counts: {dict(partitions)}")
    sentinels = {
        int(context["context_index"])
        for context in contexts
        if context.get("sentinel") is True
    }
    if len(sentinels) != 64:
        raise RuntimeError("The suite must contain exactly 64 sentinels")

    suite_manifest_sha256 = sha256_file(args.artifact / "suite-manifest.json")
    runtime_manifest_sha256 = sha256_file(args.artifact / "capture-runtime.json")

    reference_bytes = validate_tensor_directory(
        args.artifact / "reference-hidden",
        runtime_manifest_sha256=runtime_manifest_sha256,
        expected_contexts=context_indices,
        key="hidden_states",
        dtype="BF16",
        shape=[2047, 7168],
        verify_hashes=args.verify_payload_hashes,
    )
    sentinel_hidden_bytes = 0
    for repeat in ("repeat-01", "repeat-02"):
        sentinel_hidden_bytes += validate_tensor_directory(
            args.artifact / "sentinel-hidden" / repeat,
            runtime_manifest_sha256=runtime_manifest_sha256,
            expected_contexts=sentinels,
            key="hidden_states",
            dtype="BF16",
            shape=[2047, 7168],
            verify_hashes=args.verify_payload_hashes,
        )
    sentinel_logit_bytes = 0
    for repeat in ("repeat-00", "repeat-01", "repeat-02"):
        sentinel_logit_bytes += validate_tensor_directory(
            args.artifact / "sentinel-live-logits" / repeat,
            runtime_manifest_sha256=runtime_manifest_sha256,
            expected_contexts=sentinels,
            key="logits",
            dtype="BF16",
            shape=[2047, 163840],
            verify_hashes=args.verify_payload_hashes,
        )

    lm_head_path = args.artifact / "lm-head" / "weight.safetensors"
    lm_head_manifest = load_json(args.artifact / "lm-head" / "manifest.json")
    with safe_open(lm_head_path, framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["weight"]:
            raise RuntimeError("The LM-head file must contain only the weight tensor")
        weight = handle.get_slice("weight")
        if weight.get_dtype() != "BF16" or weight.get_shape() != [163840, 7168]:
            raise RuntimeError("The LM-head tensor must be BF16 [163840, 7168]")
    if lm_head_path.stat().st_size != lm_head_manifest.get("size_bytes"):
        raise RuntimeError("The LM-head file size differs from its manifest")
    if (
        args.verify_payload_hashes
        and sha256_file(lm_head_path) != lm_head_manifest.get("file_sha256")
    ):
        raise RuntimeError("The LM-head file hash differs from its manifest")

    replay = load_json(args.artifact / "validation" / "hidden-replay-qualification.json")
    if replay.get("status") != "qualified":
        raise RuntimeError("Hidden-state LM-head replay is not qualified")
    separation = load_json(args.artifact / "validation" / "source-separation.json")
    if (
        separation.get("status") != "implemented"
        or separation.get("analysis_qualification_document_overlap") != 0
        or separation.get("compatibility_suite", {}).get("exact_token_hash_overlap") != 0
    ):
        raise RuntimeError("Source-separation validation did not pass")
    capability = load_json(args.artifact / "validation" / "capability-overlap.json")
    if capability.get("status") != "implemented" or capability.get("overlap_count") != 0:
        raise RuntimeError("Capability-benchmark overlap validation did not pass")
    sentinel_repeat_mean_kld = validate_sentinel_receipts(
        args.artifact, suite_manifest_sha256
    )

    artifact_manifest = load_json(args.artifact / "manifest.json")
    if artifact_manifest.get("status") != "qualified":
        raise RuntimeError("The artifact manifest is not qualified")
    if artifact_manifest.get("suite", {}).get("manifest_sha256") != suite_manifest_sha256:
        raise RuntimeError("The artifact manifest identifies another token suite")
    reference_manifest_sha256 = sha256_file(
        args.artifact / "reference-hidden" / "manifest.json"
    )
    if (
        artifact_manifest.get("reference_hidden", {}).get("manifest_sha256")
        != reference_manifest_sha256
    ):
        raise RuntimeError("The artifact manifest identifies another reference capture")
    if (
        artifact_manifest.get("capture_runtime", {}).get("sha256")
        != runtime_manifest_sha256
    ):
        raise RuntimeError("The artifact manifest identifies another capture runtime")
    if (
        artifact_manifest.get("lm_head", {}).get("file_sha256")
        != lm_head_manifest.get("file_sha256")
    ):
        raise RuntimeError("The artifact manifest identifies another LM head")

    result = {
        "allocation_counts": dict(sorted(allocations.items())),
        "context_count": len(contexts),
        "lm_head_size_bytes": lm_head_path.stat().st_size,
        "partition_counts": dict(partitions),
        "reference_hidden_size_bytes": reference_bytes,
        "sentinel_count": len(sentinels),
        "source_cluster_count": len(source_clusters),
        "sentinel_repeat_mean_kld": sentinel_repeat_mean_kld,
        "sentinel_hidden_size_bytes": sentinel_hidden_bytes,
        "sentinel_live_logit_size_bytes": sentinel_logit_bytes,
        "status": "qualified",
        "verified_payload_hashes": args.verify_payload_hashes,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
