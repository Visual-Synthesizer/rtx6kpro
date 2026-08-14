#!/usr/bin/env python3
"""Record source separation for a Kimi K3 distribution-fidelity suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

EXCLUDED_CAPABILITY_DATASETS = ("GPQA-Diamond", "HumanEval", "MMLU")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--compatibility-suite-manifest", type=Path, required=True)
    parser.add_argument("--compatibility-repository", required=True)
    parser.add_argument("--known-codec-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite = load_json(args.suite_manifest)
    contexts = suite["contexts"]
    compatibility = load_json(args.compatibility_suite_manifest)
    compatibility_contexts = compatibility.get(
        "contexts", compatibility.get("windows")
    )
    if not isinstance(compatibility_contexts, list):
        raise RuntimeError("Compatibility manifest contains no context records")

    def document_identity(context: dict[str, Any]) -> tuple[Any, ...]:
        return (
            context["dataset"],
            context.get("dataset_config"),
            context.get("dataset_split"),
            context.get("source_id"),
        )

    analysis_documents = {
        document_identity(context)
        for context in contexts
        if context["partition"] == "analysis"
    }
    qualification_documents = {
        document_identity(context)
        for context in contexts
        if context["partition"] == "qualification"
    }
    token_hashes = {context["token_ids_json_sha256"] for context in contexts}
    compatibility_hashes = {
        context["token_ids_json_sha256"] for context in compatibility_contexts
    }
    selected_datasets = sorted({context["dataset"] for context in contexts})

    known_codec_fitting: dict[str, Any]
    if args.known_codec_manifest is None:
        known_codec_fitting = {
            "status": "unsupported",
            "reason": "No local codec-fitting manifest was supplied.",
        }
    else:
        codec = load_json(args.known_codec_manifest)
        hessian = codec.get("hessian")
        known_codec_fitting = {
            "hessian": hessian,
            "manifest_kind": codec.get("kind"),
            "status": "implemented" if hessian == "identity" else "research-only",
            "text_or_activation_calibration_corpus": (
                None if hessian == "identity" else "not identified by this manifest"
            ),
        }

    result = {
        "analysis_contexts": len(analysis_documents),
        "analysis_qualification_document_overlap": len(
            analysis_documents & qualification_documents
        ),
        "artifact_scope": (
            "Kimi K3 distribution-fidelity token-suite source separation"
        ),
        "capability_dataset_source_check": {
            "excluded_dataset_names": list(EXCLUDED_CAPABILITY_DATASETS),
            "excluded_dataset_selected": any(
                excluded.casefold() in dataset.casefold()
                for excluded in EXCLUDED_CAPABILITY_DATASETS
                for dataset in selected_datasets
            ),
            "selected_dataset_names": selected_datasets,
        },
        "compatibility_suite": {
            "contexts": len(compatibility_contexts),
            "exact_token_hash_overlap": len(token_hashes & compatibility_hashes),
            "repository": args.compatibility_repository,
            "suite_token_hash_sha256": compatibility["suite_token_hash_sha256"],
        },
        "external_candidate_requirement": (
            "A candidate produced with an external or unavailable Hessian, "
            "activation, codec-fitting, or mode-selection corpus must provide "
            "document identities for a separate overlap check before "
            "qualification results are accepted."
        ),
        "known_local_codec_fitting": known_codec_fitting,
        "qualification_contexts": len(qualification_documents),
        "status": "implemented",
    }
    if result["analysis_qualification_document_overlap"] != 0:
        raise RuntimeError("Analysis and qualification documents overlap")
    if result["capability_dataset_source_check"]["excluded_dataset_selected"]:
        raise RuntimeError("An excluded capability dataset was selected")

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
