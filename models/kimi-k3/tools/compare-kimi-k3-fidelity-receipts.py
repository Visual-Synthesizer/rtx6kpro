#!/usr/bin/env python3
"""Compare two Kimi K3 distribution-fidelity result receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("direction") != "KL(reference || candidate)":
        raise RuntimeError(f"Unsupported KLD direction in {path}")
    if not isinstance(report.get("per_context"), list):
        raise RuntimeError(f"Missing per-context results in {path}")
    return report


def validate_common_reference(
    report_a: dict[str, Any], report_b: dict[str, Any]
) -> None:
    fields = ("suite_manifest_sha256", "positions", "contexts")
    for field in fields:
        if report_a.get(field) != report_b.get(field):
            raise RuntimeError(f"Reports disagree on {field}")
    head_a = report_a.get("comparator", {}).get("lm_head_file_sha256")
    head_b = report_b.get("comparator", {}).get("lm_head_file_sha256")
    if head_a != head_b:
        raise RuntimeError("Reports use different LM-head weights")


def bootstrap_stratified_differences(
    by_stratum_cluster: dict[str, dict[str, list[float]]], *, samples: int, seed: int
) -> dict[str, Any]:
    generator = np.random.default_rng(seed)
    sampled_stratum_means: list[np.ndarray] = []
    stratum_context_counts: list[int] = []
    stratum_cluster_counts: list[int] = []
    observed_values: list[np.ndarray] = []
    for stratum in sorted(by_stratum_cluster):
        clusters = by_stratum_cluster[stratum]
        cluster_keys = sorted(clusters)
        cluster_sums = np.asarray(
            [np.sum(clusters[key]) for key in cluster_keys], dtype=np.float64
        )
        cluster_sizes = np.asarray(
            [len(clusters[key]) for key in cluster_keys], dtype=np.float64
        )
        choices = generator.integers(
            0, len(cluster_keys), size=(samples, len(cluster_keys))
        )
        sampled_stratum_means.append(
            cluster_sums[choices].sum(axis=1)
            / cluster_sizes[choices].sum(axis=1)
        )
        stratum_context_counts.append(int(cluster_sizes.sum()))
        stratum_cluster_counts.append(len(cluster_keys))
        observed_values.append(
            np.concatenate(
                [np.asarray(clusters[key], dtype=np.float64) for key in cluster_keys]
            )
        )
    sampled_means = np.average(
        np.stack(sampled_stratum_means, axis=1),
        axis=1,
        weights=np.asarray(stratum_context_counts, dtype=np.float64),
    )
    observed = np.concatenate(
        observed_values
    )
    return {
        "bootstrap_samples": samples,
        "ci95_high": float(np.quantile(sampled_means, 0.975)),
        "ci95_low": float(np.quantile(sampled_means, 0.025)),
        "cluster_unit": "(dataset, source_cluster_id)",
        "estimate": float(observed.mean()),
        "standard_error": float(sampled_means.std(ddof=1)),
        "stratification": "allocation_stratum",
        "stratum_cluster_counts": dict(
            zip(sorted(by_stratum_cluster), stratum_cluster_counts)
        ),
        "stratum_context_counts": dict(
            zip(sorted(by_stratum_cluster), stratum_context_counts)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-a-report", type=Path, required=True)
    parser.add_argument("--candidate-b-report", type=Path, required=True)
    parser.add_argument("--candidate-a-label", required=True)
    parser.add_argument("--candidate-b-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--tie-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    report_a = load_report(args.candidate_a_report)
    report_b = load_report(args.candidate_b_report)
    validate_common_reference(report_a, report_b)

    contexts_a = {item["context_index"]: item for item in report_a["per_context"]}
    contexts_b = {item["context_index"]: item for item in report_b["per_context"]}
    if contexts_a.keys() != contexts_b.keys():
        raise RuntimeError("Reports contain different context indices")

    paired: list[dict[str, Any]] = []
    by_stratum: dict[str, list[float]] = defaultdict(list)
    by_stratum_cluster: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    wins_a = 0
    wins_b = 0
    ties = 0
    for index in sorted(contexts_a):
        context_a = contexts_a[index]
        context_b = contexts_b[index]
        identity_fields = (
            "allocation_stratum",
            "dataset",
            "dataset_config",
            "dataset_revision",
            "dataset_split",
            "source_cluster_id",
            "source_content_sha256",
            "source_id",
        )
        for field in identity_fields:
            if context_a.get(field) != context_b.get(field):
                raise RuntimeError(f"Context {index} disagrees on {field}")
        kld_a = float(context_a["kl_reference_to_candidate"]["mean"])
        kld_b = float(context_b["kl_reference_to_candidate"]["mean"])
        difference = kld_a - kld_b
        if difference < -args.tie_tolerance:
            winner = args.candidate_a_label
            wins_a += 1
        elif difference > args.tie_tolerance:
            winner = args.candidate_b_label
            wins_b += 1
        else:
            winner = "tie"
            ties += 1
        stratum = str(
            context_a.get("allocation_stratum", context_a.get("semantic_class"))
        )
        source_cluster_key = json.dumps(
            [
                context_a.get("dataset"),
                context_a.get(
                    "source_cluster_id", context_a.get("source_id", index)
                ),
            ],
            separators=(",", ":"),
        )
        by_stratum[stratum].append(difference)
        by_stratum_cluster[stratum][source_cluster_key].append(difference)
        paired.append(
            {
                "allocation_stratum": stratum,
                "candidate_a_kld": kld_a,
                "candidate_b_kld": kld_b,
                "context_index": index,
                "dataset": context_a.get("dataset"),
                "dataset_config": context_a.get("dataset_config"),
                "dataset_revision": context_a.get("dataset_revision"),
                "dataset_split": context_a.get("dataset_split"),
                "difference_a_minus_b": difference,
                "source_cluster_id": context_a.get("source_cluster_id"),
                "source_content_sha256": context_a.get("source_content_sha256"),
                "source_id": context_a.get("source_id"),
                "winner": winner,
            }
        )

    differences = np.asarray(
        [item["difference_a_minus_b"] for item in paired], dtype=np.float64
    )
    result = {
        "candidate_a": {
            "label": args.candidate_a_label,
            "report_sha256": sha256_file(args.candidate_a_report),
        },
        "candidate_b": {
            "label": args.candidate_b_label,
            "report_sha256": sha256_file(args.candidate_b_report),
        },
        "contexts": len(paired),
        "difference_direction": "candidate_a_kld - candidate_b_kld",
        "interpretation": "Negative values favor candidate A; positive values favor candidate B.",
        "macro_allocation_stratum_mean_difference": float(
            np.mean([np.mean(values) for values in by_stratum.values()])
        ),
        "micro_token_kld_difference": float(
            report_a["kl_reference_to_candidate"]["mean"]
            - report_b["kl_reference_to_candidate"]["mean"]
        ),
        "paired_context_difference": {
            "bootstrap": bootstrap_stratified_differences(
                by_stratum_cluster,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            ),
            "mean": float(differences.mean()),
            "median": float(np.median(differences)),
        },
        "per_allocation_stratum": {
            key: {
                "contexts": len(values),
                "source_clusters": len(by_stratum_cluster[key]),
                "mean_difference": float(np.mean(values)),
                "median_difference": float(np.median(values)),
            }
            for key, values in sorted(by_stratum.items())
        },
        "per_context": paired,
        "suite_manifest_sha256": report_a["suite_manifest_sha256"],
        "win_count": {
            args.candidate_a_label: wins_a,
            args.candidate_b_label: wins_b,
            "tie": ties,
            "tie_tolerance": args.tie_tolerance,
        },
        "largest_absolute_context_differences": sorted(
            paired,
            key=lambda item: abs(item["difference_a_minus_b"]),
            reverse=True,
        )[:20],
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
