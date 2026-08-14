#!/usr/bin/env python3
"""Compare complete Kimi K3 distributions from hidden states or live logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Self

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_token_ids(path: Path) -> str:
    token_ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError(f"Expected a JSON integer array in {path}")
    canonical_json = json.dumps(token_ids, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p99_9": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
    }


def bootstrap_context_mean(
    context_means: list[float], *, samples: int, seed: int
) -> dict[str, float | int | str]:
    values = np.asarray(context_means, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(values), size=(samples, len(values)))
    means = values[choices].mean(axis=1)
    return {
        "bootstrap_samples": samples,
        "ci95_high": float(np.quantile(means, 0.975)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "cluster_unit": "context",
        "estimate": float(values.mean()),
        "standard_error": float(means.std(ddof=1)),
    }


def bootstrap_source_cluster_mean(
    cluster_context_means: dict[str, list[float]], *, samples: int, seed: int
) -> dict[str, float | int | str]:
    """Bootstrap source clusters while retaining every context in a cluster."""
    generator = np.random.default_rng(seed)
    cluster_keys = sorted(cluster_context_means)
    cluster_sums = np.asarray(
        [np.sum(cluster_context_means[key]) for key in cluster_keys],
        dtype=np.float64,
    )
    cluster_sizes = np.asarray(
        [len(cluster_context_means[key]) for key in cluster_keys],
        dtype=np.float64,
    )
    choices = generator.integers(
        0, len(cluster_keys), size=(samples, len(cluster_keys))
    )
    means = cluster_sums[choices].sum(axis=1) / cluster_sizes[choices].sum(axis=1)
    return {
        "bootstrap_samples": samples,
        "ci95_high": float(np.quantile(means, 0.975)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "cluster_count": len(cluster_keys),
        "cluster_unit": "(dataset, source_cluster_id)",
        "context_count": int(cluster_sizes.sum()),
        "estimate": float(cluster_sums.sum() / cluster_sizes.sum()),
        "standard_error": float(means.std(ddof=1)),
    }


def stratified_bootstrap_source_cluster_mean(
    stratum_clusters: dict[str, dict[str, list[float]]], *, samples: int, seed: int
) -> dict[str, Any]:
    """Bootstrap source clusters within strata and preserve stratum weights."""
    generator = np.random.default_rng(seed)
    stratum_bootstraps: list[np.ndarray] = []
    stratum_sizes: list[int] = []
    stratum_cluster_counts: list[int] = []
    for stratum in sorted(stratum_clusters):
        clusters = stratum_clusters[stratum]
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
        sampled_means = cluster_sums[choices].sum(axis=1) / cluster_sizes[
            choices
        ].sum(axis=1)
        stratum_bootstraps.append(sampled_means)
        stratum_sizes.append(int(cluster_sizes.sum()))
        stratum_cluster_counts.append(len(cluster_keys))

    stacked = np.stack(stratum_bootstraps, axis=1)
    weights = np.asarray(stratum_sizes, dtype=np.float64)
    micro = np.average(stacked, axis=1, weights=weights)
    macro = stacked.mean(axis=1)
    observed_stratum_means = np.asarray(
        [
            np.mean(
                [
                    value
                    for cluster in stratum_clusters[key].values()
                    for value in cluster
                ]
            )
            for key in sorted(stratum_clusters)
        ],
        dtype=np.float64,
    )
    return {
        "bootstrap_samples": samples,
        "cluster_unit": "(dataset, source_cluster_id)",
        "macro": {
            "ci95_high": float(np.quantile(macro, 0.975)),
            "ci95_low": float(np.quantile(macro, 0.025)),
            "estimate": float(observed_stratum_means.mean()),
            "standard_error": float(macro.std(ddof=1)),
        },
        "micro": {
            "ci95_high": float(np.quantile(micro, 0.975)),
            "ci95_low": float(np.quantile(micro, 0.025)),
            "estimate": float(np.average(observed_stratum_means, weights=weights)),
            "standard_error": float(micro.std(ddof=1)),
        },
        "stratification": "allocation_stratum",
        "stratum_cluster_counts": dict(
            zip(sorted(stratum_clusters), stratum_cluster_counts)
        ),
        "stratum_sizes": dict(zip(sorted(stratum_clusters), stratum_sizes)),
    }


def context_depth_ranges(scored_rows: int) -> list[tuple[int, int]]:
    """Return stable next-token depth buckets for a 2,048-token context."""
    boundaries = (0, 256, 512, 1024, 1536, scored_rows)
    return [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if start < end
    ]


def record_index(record: dict[str, Any]) -> int:
    for key in ("context_index", "window_index", "index"):
        if key in record:
            return int(record[key])
    raise RuntimeError(f"Source manifest record has no context index: {record}")


def validate_source_manifest(
    directory: Path,
    *,
    kind: str,
    suite: dict[str, Any],
    contexts: list[dict[str, Any]],
    verify_file_hashes: bool,
) -> dict[int, Path]:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("suite_token_hash_sha256") != suite["suite_token_hash_sha256"]:
        raise RuntimeError(f"Suite token hash mismatch in {manifest_path}")
    records = manifest.get("contexts", manifest.get("windows"))
    if not isinstance(records, list):
        raise TypeError(f"Source manifest has no contexts or windows: {manifest_path}")
    by_index = {record_index(record): record for record in records}
    if len(by_index) != len(records):
        raise RuntimeError(f"Duplicate context indices in {manifest_path}")

    expected_key = "hidden_states" if kind == "hidden" else "logits"
    paths: dict[int, Path] = {}
    for context in contexts:
        index = record_index(context)
        if index not in by_index:
            raise RuntimeError(f"Context {index} is absent from {manifest_path}")
        record = by_index[index]
        if record.get("token_ids_json_sha256") != context["token_ids_json_sha256"]:
            raise RuntimeError(
                f"Token hash mismatch for context {index} in {manifest_path}"
            )
        if record.get("key") != expected_key:
            raise RuntimeError(
                f"Expected tensor key {expected_key!r} for context {index} in "
                f"{manifest_path}"
            )
        path = directory / record["file"]
        if not path.is_file():
            raise RuntimeError(
                f"Missing tensor file recorded by {manifest_path}: {path}"
            )
        if verify_file_hashes and sha256_file(path) != record.get("sha256"):
            raise RuntimeError(f"Tensor file hash mismatch: {path}")
        paths[index] = path
    return paths


def validate_suite_tokens(
    suite_dir: Path,
    contexts: list[dict[str, Any]],
) -> None:
    for context in contexts:
        index = record_index(context)
        token_path = suite_dir / context["token_file"]
        if sha256_token_ids(token_path) != context["token_ids_json_sha256"]:
            raise RuntimeError(f"Token file hash mismatch for context {index}")


class DistributionSource:
    def __init__(
        self,
        *,
        kind: str,
        directory: Path,
        tensor_paths: dict[int, Path],
        head_weight: torch.Tensor | None,
        expected_width: int,
        expected_vocab: int,
        device: torch.device,
    ) -> None:
        self.kind = kind
        self.directory = directory
        self.tensor_paths = tensor_paths
        self.head_weight = head_weight
        self.expected_width = expected_width
        self.expected_vocab = expected_vocab
        self.device = device
        self._handle: Any = None
        self._slice: Any = None
        self._hidden: torch.Tensor | None = None
        self._live_logits: torch.Tensor | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def open_context(self, index: int, scored_rows: int) -> None:
        self.close()
        path = self.tensor_paths[index]
        self._handle = safe_open(path, framework="pt", device="cpu")
        key = "hidden_states" if self.kind == "hidden" else "logits"
        self._slice = self._handle.get_slice(key)
        shape = self._slice.get_shape()
        if self.kind == "hidden":
            if shape[0] < scored_rows or shape[1] != self.expected_width:
                raise RuntimeError(f"Unexpected hidden-state shape in {path}: {shape}")
            if self._slice.get_dtype() != "BF16":
                raise RuntimeError(f"Expected BF16 hidden states in {path}")
            if self.head_weight is None:
                raise RuntimeError("Hidden-state replay requires an LM-head weight")
            self._hidden = self._slice[:scored_rows]
        else:
            if shape != [scored_rows, self.expected_vocab]:
                raise RuntimeError(f"Unexpected live-logit shape in {path}: {shape}")
            if self._slice.get_dtype() not in {"BF16", "F32"}:
                raise RuntimeError(f"Expected BF16 or F32 live logits in {path}")
            # A Kimi K3 sentinel tensor is at most 1.25 GiB in FP32. Retaining
            # one complete context on the selected device avoids reopening the
            # row-major safetensors payload for every vocabulary pass.
            self._live_logits = self._slice[:scored_rows].to(self.device)
            self._slice = None
            self._handle.__exit__(None, None, None)
            self._handle = None

    def logits(
        self,
        row_start: int,
        row_end: int,
        vocab_start: int,
        vocab_end: int,
    ) -> torch.Tensor:
        if self.kind == "logits":
            assert self._live_logits is not None
            return self._live_logits[
                row_start:row_end, vocab_start:vocab_end
            ].contiguous()
        assert self._hidden is not None
        assert self.head_weight is not None
        hidden = self._hidden[row_start:row_end].to(self.device, non_blocking=True)
        weight = self.head_weight[vocab_start:vocab_end]
        return F.linear(hidden, weight)

    def close(self) -> None:
        self._slice = None
        self._hidden = None
        self._live_logits = None
        if self._handle is not None:
            self._handle.__exit__(None, None, None)
            self._handle = None


def distribution_normalizers_and_top1(
    source: DistributionSource,
    *,
    row_start: int,
    row_end: int,
    vocab_size: int,
    vocab_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = row_end - row_start
    log_z = torch.full((rows,), -torch.inf, dtype=torch.float32, device=source.device)
    top_values = torch.full_like(log_z, -torch.inf)
    top_ids = torch.zeros((rows,), dtype=torch.int64, device=source.device)
    for vocab_start in range(0, vocab_size, vocab_chunk):
        vocab_end = min(vocab_start + vocab_chunk, vocab_size)
        logits = source.logits(row_start, row_end, vocab_start, vocab_end).float()
        log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=-1))
        local_values, local_ids = logits.max(dim=-1)
        update = local_values > top_values
        top_values = torch.where(update, local_values, top_values)
        top_ids = torch.where(update, local_ids + vocab_start, top_ids)
    return log_z, top_ids


def compare_rows(
    reference: DistributionSource,
    candidate: DistributionSource,
    *,
    row_start: int,
    row_end: int,
    vocab_size: int,
    vocab_chunk: int,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    reference_log_z, reference_top1 = distribution_normalizers_and_top1(
        reference,
        row_start=row_start,
        row_end=row_end,
        vocab_size=vocab_size,
        vocab_chunk=vocab_chunk,
    )
    candidate_log_z, candidate_top1 = distribution_normalizers_and_top1(
        candidate,
        row_start=row_start,
        row_end=row_end,
        vocab_size=vocab_size,
        vocab_chunk=vocab_chunk,
    )
    rows = row_end - row_start
    kl = torch.zeros((rows,), dtype=torch.float64, device=reference.device)
    js = torch.zeros_like(kl)
    absolute_error_sum = 0.0
    absolute_error_max = 0.0
    for vocab_start in range(0, vocab_size, vocab_chunk):
        vocab_end = min(vocab_start + vocab_chunk, vocab_size)
        reference_logits = reference.logits(
            row_start, row_end, vocab_start, vocab_end
        ).float()
        candidate_logits = candidate.logits(
            row_start, row_end, vocab_start, vocab_end
        ).float()
        reference_log_p = reference_logits - reference_log_z[:, None]
        candidate_log_p = candidate_logits - candidate_log_z[:, None]
        reference_p = reference_log_p.exp()
        candidate_p = candidate_log_p.exp()
        log_mid = torch.logaddexp(reference_log_p, candidate_log_p) - math.log(2.0)
        kl += (reference_p * (reference_log_p - candidate_log_p)).sum(
            dim=-1, dtype=torch.float64
        )
        js += 0.5 * (
            (reference_p * (reference_log_p - log_mid)).sum(dim=-1, dtype=torch.float64)
            + (candidate_p * (candidate_log_p - log_mid)).sum(
                dim=-1, dtype=torch.float64
            )
        )
        difference = (reference_logits - candidate_logits).abs()
        absolute_error_sum += float(difference.sum(dtype=torch.float64).item())
        absolute_error_max = max(absolute_error_max, float(difference.max().item()))
    top1_matches = int((reference_top1 == candidate_top1).sum().item())
    compared_logits = rows * vocab_size
    # Float32 log-probabilities can leave sub-nanoscopic negative roundoff in
    # divergences whose mathematical lower bound is zero.
    kl.clamp_min_(0)
    js.clamp_min_(0)
    return (
        kl.cpu().numpy(),
        js.cpu().numpy(),
        top1_matches,
        absolute_error_sum / compared_logits,
        absolute_error_max,
    )


def parse_source(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    prefix: str,
) -> tuple[str, Path]:
    hidden = getattr(args, f"{prefix}_hidden_dir")
    logits = getattr(args, f"{prefix}_logits_dir")
    if (hidden is None) == (logits is None):
        parser.error(
            f"Specify exactly one of --{prefix}-hidden-dir or --{prefix}-logits-dir"
        )
    return ("hidden", hidden) if hidden is not None else ("logits", logits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-hidden-dir", type=Path)
    parser.add_argument("--reference-logits-dir", type=Path)
    parser.add_argument("--reference-label")
    parser.add_argument("--candidate-hidden-dir", type=Path)
    parser.add_argument("--candidate-logits-dir", type=Path)
    parser.add_argument("--candidate-label")
    parser.add_argument("--lm-head", type=Path)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden-width", type=int, default=7168)
    parser.add_argument("--vocab-size", type=int, default=163840)
    parser.add_argument("--vocab-chunk", type=int, default=10240)
    parser.add_argument("--position-block", type=int, default=128)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=1)
    parser.add_argument("--verify-source-file-hashes", action="store_true")
    parser.add_argument(
        "--context-filter",
        choices=("all", "analysis", "qualification", "sentinel", "non-sentinel"),
        default="all",
    )
    parser.add_argument("--start-context", type=int, default=0)
    parser.add_argument("--stop-context", type=int)
    args = parser.parse_args()

    reference_kind, reference_dir = parse_source(parser, args, "reference")
    candidate_kind, candidate_dir = parse_source(parser, args, "candidate")
    if args.vocab_size % args.vocab_chunk != 0:
        parser.error("--vocab-chunk must divide --vocab-size exactly")
    if args.position_block <= 0:
        parser.error("--position-block must be positive")
    if args.bootstrap_samples <= 1:
        parser.error("--bootstrap-samples must be greater than one")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    head_weight = None
    head_file_sha256 = None
    if reference_kind == "hidden" or candidate_kind == "hidden":
        if args.lm_head is None:
            parser.error(
                "--lm-head is required when either source stores hidden states"
            )
        with safe_open(args.lm_head, framework="pt", device="cpu") as handle:
            head_weight_cpu = handle.get_tensor("weight")
        expected_shape = [args.vocab_size, args.hidden_width]
        if (
            head_weight_cpu.dtype != torch.bfloat16
            or list(head_weight_cpu.shape) != expected_shape
        ):
            raise RuntimeError(
                f"Expected BF16 LM head with shape {expected_shape}; "
                f"got {head_weight_cpu.dtype} {list(head_weight_cpu.shape)}"
            )
        head_weight = head_weight_cpu.to(device)
        head_file_sha256 = sha256_file(args.lm_head)
        head_manifest_path = args.lm_head.parent / "manifest.json"
        if head_manifest_path.exists():
            head_manifest = json.loads(head_manifest_path.read_text(encoding="utf-8"))
            if head_manifest.get("file_sha256") != head_file_sha256:
                raise RuntimeError(f"LM-head file hash mismatch: {args.lm_head}")

    suite = json.loads(args.suite_manifest.read_text(encoding="utf-8"))
    contexts = suite.get("contexts", suite.get("windows"))
    if not isinstance(contexts, list):
        raise TypeError("Suite manifest must contain contexts or windows")
    if args.context_filter == "analysis":
        contexts = [item for item in contexts if item.get("partition") == "analysis"]
    elif args.context_filter == "qualification":
        contexts = [
            item for item in contexts if item.get("partition") == "qualification"
        ]
    elif args.context_filter == "sentinel":
        contexts = [item for item in contexts if item.get("sentinel") is True]
    elif args.context_filter == "non-sentinel":
        contexts = [item for item in contexts if item.get("sentinel") is not True]
    scored_rows = int(suite["context_length"]) - 1
    stop = len(contexts) if args.stop_context is None else args.stop_context
    if args.start_context < 0 or stop > len(contexts) or args.start_context >= stop:
        parser.error("Context range must select at least one suite context")
    selected_contexts = contexts[args.start_context : stop]
    validate_suite_tokens(args.suite_manifest.parent, selected_contexts)
    reference_paths = validate_source_manifest(
        reference_dir,
        kind=reference_kind,
        suite=suite,
        contexts=selected_contexts,
        verify_file_hashes=args.verify_source_file_hashes,
    )
    candidate_paths = validate_source_manifest(
        candidate_dir,
        kind=candidate_kind,
        suite=suite,
        contexts=selected_contexts,
        verify_file_hashes=args.verify_source_file_hashes,
    )

    all_kl: list[np.ndarray] = []
    all_js: list[np.ndarray] = []
    context_reports: list[dict[str, Any]] = []
    class_context_means: dict[str, list[float]] = defaultdict(list)
    stratum_context_means: dict[str, list[float]] = defaultdict(list)
    all_cluster_context_means: dict[str, list[float]] = defaultdict(list)
    class_cluster_context_means: dict[
        str, dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    stratum_cluster_context_means: dict[
        str, dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    depth_kl: dict[str, list[np.ndarray]] = defaultdict(list)
    total_top1 = 0
    total_positions = 0
    total_absolute_error = 0.0
    maximum_absolute_error = 0.0
    with (
        DistributionSource(
            kind=reference_kind,
            directory=reference_dir,
            tensor_paths=reference_paths,
            head_weight=head_weight,
            expected_width=args.hidden_width,
            expected_vocab=args.vocab_size,
            device=device,
        ) as reference,
        DistributionSource(
            kind=candidate_kind,
            directory=candidate_dir,
            tensor_paths=candidate_paths,
            head_weight=head_weight,
            expected_width=args.hidden_width,
            expected_vocab=args.vocab_size,
            device=device,
        ) as candidate,
    ):
        for context in selected_contexts:
            index = record_index(context)
            reference.open_context(index, scored_rows)
            candidate.open_context(index, scored_rows)
            kl_parts: list[np.ndarray] = []
            js_parts: list[np.ndarray] = []
            context_top1 = 0
            context_absolute_error_sum = 0.0
            context_absolute_error_max = 0.0
            for row_start in range(0, scored_rows, args.position_block):
                row_end = min(row_start + args.position_block, scored_rows)
                kl, js, top1, mean_error, max_error = compare_rows(
                    reference,
                    candidate,
                    row_start=row_start,
                    row_end=row_end,
                    vocab_size=args.vocab_size,
                    vocab_chunk=args.vocab_chunk,
                )
                kl_parts.append(kl)
                js_parts.append(js)
                context_top1 += top1
                context_absolute_error_sum += mean_error * (
                    (row_end - row_start) * args.vocab_size
                )
                context_absolute_error_max = max(context_absolute_error_max, max_error)
            context_kl = np.concatenate(kl_parts)
            context_js = np.concatenate(js_parts)
            semantic_class = context.get("semantic_class", context.get("domain"))
            allocation_stratum = context.get(
                "allocation_stratum", semantic_class
            )
            source_cluster_key = json.dumps(
                [
                    context.get("dataset"),
                    context.get("source_cluster_id", context.get("source_id", index)),
                ],
                separators=(",", ":"),
            )
            report = {
                "allocation_stratum": allocation_stratum,
                "context_index": index,
                "dataset": context.get("dataset"),
                "dataset_config": context.get("dataset_config"),
                "dataset_revision": context.get("dataset_revision"),
                "dataset_split": context.get("dataset_split"),
                "js": stats(context_js),
                "kl_reference_to_candidate": stats(context_kl),
                "logit_absolute_error_max": context_absolute_error_max,
                "logit_absolute_error_mean": context_absolute_error_sum
                / (scored_rows * args.vocab_size),
                "positions": scored_rows,
                "partition": context.get("partition"),
                "representation_type": context.get("representation_type"),
                "semantic_class": semantic_class,
                "sentinel": context.get("sentinel", False),
                "language": context.get("language"),
                "source_cluster_id": context.get("source_cluster_id"),
                "source_content_sha256": context.get("source_content_sha256"),
                "source_id": context.get("source_id"),
                "top1_agreement": context_top1 / scored_rows,
            }
            context_reports.append(report)
            all_kl.append(context_kl)
            all_js.append(context_js)
            class_context_means[str(semantic_class)].append(float(context_kl.mean()))
            stratum_context_means[str(allocation_stratum)].append(
                float(context_kl.mean())
            )
            all_cluster_context_means[source_cluster_key].append(
                float(context_kl.mean())
            )
            class_cluster_context_means[str(semantic_class)][
                source_cluster_key
            ].append(float(context_kl.mean()))
            stratum_cluster_context_means[str(allocation_stratum)][
                source_cluster_key
            ].append(float(context_kl.mean()))
            for depth_start, depth_end in context_depth_ranges(scored_rows):
                depth_kl[f"{depth_start:04d}-{depth_end - 1:04d}"].append(
                    context_kl[depth_start:depth_end]
                )
            total_top1 += context_top1
            total_positions += scored_rows
            total_absolute_error += context_absolute_error_sum
            maximum_absolute_error = max(
                maximum_absolute_error, context_absolute_error_max
            )
            print(json.dumps(report, sort_keys=True), flush=True)

    combined_kl = np.concatenate(all_kl)
    combined_js = np.concatenate(all_js)
    context_means = [
        item["kl_reference_to_candidate"]["mean"] for item in context_reports
    ]
    high_kld_contexts = sorted(
        context_reports,
        key=lambda item: item["kl_reference_to_candidate"]["mean"],
        reverse=True,
    )[:20]
    top1_discordant_contexts = sorted(
        context_reports,
        key=lambda item: item["top1_agreement"],
    )[:20]
    result = {
        "comparator": {
            "bf16_reduced_precision_reduction": False,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "deterministic_algorithms": True,
            "device": str(device),
            "lm_head_file_sha256": head_file_sha256,
            "live_logit_input_cache": "complete-context-on-device",
            "position_block": args.position_block,
            "tf32": False,
            "two_pass_full_vocabulary": True,
            "source_file_hashes_verified": args.verify_source_file_hashes,
            "vocab_chunk": args.vocab_chunk,
        },
        "contexts": len(context_reports),
        "context_depth_buckets": {
            key: stats(np.concatenate(values)) for key, values in depth_kl.items()
        },
        "direction": "KL(reference || candidate)",
        "high_kld_contexts": high_kld_contexts,
        "js": stats(combined_js),
        "kl_context_bootstrap": bootstrap_context_mean(
            context_means,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        "kl_reference_to_candidate": stats(combined_kl),
        "kl_macro_allocation_stratum_mean": float(
            np.mean([np.mean(values) for values in stratum_context_means.values()])
        ),
        "kl_micro_token_mean": float(combined_kl.mean()),
        "kl_source_cluster_bootstrap": bootstrap_source_cluster_mean(
            all_cluster_context_means,
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed,
        ),
        "kl_stratified_source_cluster_bootstrap": (
            stratified_bootstrap_source_cluster_mean(
                stratum_cluster_context_means,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
        ),
        "logit_absolute_error_max": maximum_absolute_error,
        "logit_absolute_error_mean": total_absolute_error
        / (total_positions * args.vocab_size),
        "per_context": context_reports,
        "positions": total_positions,
        "reference": {
            "kind": reference_kind,
            "label": args.reference_label or reference_dir.name,
            "manifest_sha256": sha256_file(reference_dir / "manifest.json"),
        },
        "candidate": {
            "kind": candidate_kind,
            "label": args.candidate_label or candidate_dir.name,
            "manifest_sha256": sha256_file(candidate_dir / "manifest.json"),
        },
        "allocation_strata": {
            stratum: bootstrap_source_cluster_mean(
                stratum_cluster_context_means[stratum],
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            for stratum in sorted(stratum_context_means)
        },
        "semantic_classes": {
            semantic_class: bootstrap_source_cluster_mean(
                class_cluster_context_means[semantic_class],
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed,
            )
            for semantic_class in sorted(class_context_means)
        },
        "suite_manifest_sha256": sha256_file(args.suite_manifest),
        "top1_agreement": total_top1 / total_positions,
        "top1_discordant_contexts": top1_discordant_contexts,
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
