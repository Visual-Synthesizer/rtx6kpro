#!/usr/bin/env python3
"""Create a machine-readable analysis of direct arithmetic stability evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "local-inference-lab.direct-arithmetic-stability-analysis.v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def percentile_interval(values: np.ndarray) -> list[float]:
    return [round(float(value) * 100, 6) for value in np.quantile(values, (0.025, 0.975))]


def task_cluster_comparison(
    rows: list[dict[str, object]],
    baseline: str,
    candidate: str,
    reasoning: str,
    temperatures: tuple[float, ...],
    draws: int,
    rng: np.random.Generator,
    family: str | None = None,
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["reasoning_effort"] != reasoning or float(row["temperature"]) not in temperatures:
            continue
        if family is not None and row["family"] != family:
            continue
        grouped[str(row["task_id"])][str(row["model_label"])].append(float(bool(row["correct"])))
    task_ids = sorted(grouped)
    if not task_ids or any(set(grouped[task_id]) != {baseline, candidate} for task_id in task_ids):
        raise ValueError("every retained task must contain both compared model labels")
    baseline_values = np.array([np.mean(grouped[task_id][baseline]) for task_id in task_ids])
    candidate_values = np.array([np.mean(grouped[task_id][candidate]) for task_id in task_ids])
    difference = candidate_values - baseline_values
    bootstrap_baseline = np.empty(draws)
    bootstrap_candidate = np.empty(draws)
    bootstrap_difference = np.empty(draws)
    batch_size = 2_000
    for start in range(0, draws, batch_size):
        count = min(batch_size, draws - start)
        indices = rng.integers(0, len(task_ids), size=(count, len(task_ids)))
        bootstrap_baseline[start : start + count] = baseline_values[indices].mean(axis=1)
        bootstrap_candidate[start : start + count] = candidate_values[indices].mean(axis=1)
        bootstrap_difference[start : start + count] = difference[indices].mean(axis=1)
    attempts_per_task = {
        label: sorted({len(grouped[task_id][label]) for task_id in task_ids})
        for label in (baseline, candidate)
    }
    return {
        "reasoning_effort": reasoning,
        "temperatures": list(temperatures),
        "family": family,
        "task_clusters": len(task_ids),
        "attempts_per_task_and_model": attempts_per_task,
        "baseline_label": baseline,
        "candidate_label": candidate,
        "baseline_accuracy_percent": round(float(baseline_values.mean()) * 100, 6),
        "candidate_accuracy_percent": round(float(candidate_values.mean()) * 100, 6),
        "candidate_minus_baseline_points": round(float(difference.mean()) * 100, 6),
        "baseline_bootstrap_95_percent_interval": percentile_interval(bootstrap_baseline),
        "candidate_bootstrap_95_percent_interval": percentile_interval(bootstrap_candidate),
        "candidate_minus_baseline_bootstrap_95_percent_interval_points": percentile_interval(
            bootstrap_difference
        ),
        "bootstrap_fraction_at_or_below_zero": round(float(np.mean(bootstrap_difference <= 0)), 8),
        "bootstrap_draws": draws,
        "bootstrap_unit": "task_id; all retained temperatures and repeats remain inside a sampled task",
    }


def direct_cells(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_label"], row["reasoning_effort"], row["temperature"])].append(row)
    result = []
    for key, group in sorted(groups.items()):
        correct = sum(bool(row["correct"]) for row in group)
        result.append(
            {
                "model_label": key[0],
                "reasoning_effort": key[1],
                "temperature": key[2],
                "attempts": len(group),
                "correct": correct,
                "wrong": len(group) - correct,
                "accuracy_percent": round(100 * correct / len(group), 6),
                "answer_classes": dict(Counter(str(row["answer_class"]) for row in group)),
            }
        )
    return result


def combined_answer_classes(rows: list[dict[str, object]], reasoning: str) -> list[dict[str, object]]:
    result = []
    for model_label in sorted({str(row["model_label"]) for row in rows}):
        group = [
            row
            for row in rows
            if row["model_label"] == model_label and row["reasoning_effort"] == reasoning
        ]
        result.append(
            {
                "model_label": model_label,
                "reasoning_effort": reasoning,
                "attempts": len(group),
                "counts": dict(Counter(str(row["answer_class"]) for row in group)),
            }
        )
    return result


def task_reliability(rows: list[dict[str, object]], reasoning: str) -> list[dict[str, object]]:
    result = []
    for model_label in sorted({str(row["model_label"]) for row in rows}):
        grouped: dict[str, list[bool]] = defaultdict(list)
        for row in rows:
            if row["model_label"] == model_label and row["reasoning_effort"] == reasoning:
                grouped[str(row["task_id"])].append(bool(row["correct"]))
        distribution = Counter(sum(values) for values in grouped.values())
        result.append(
            {
                "model_label": model_label,
                "reasoning_effort": reasoning,
                "attempts_per_task": sorted({len(values) for values in grouped.values()}),
                "tasks_by_correct_attempt_count": {
                    str(key): distribution[key] for key in sorted(distribution)
                },
            }
        )
    return result


def aggregate_probe_root(root: Path, model_names: dict[str, str]) -> dict[str, object]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    files = sorted(root.rglob("*.json"))
    receipt_hashes = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(root)
        directory_label = relative.parts[0]
        model_label = model_names.get(directory_label, directory_label)
        config = data["configuration"]
        for row in data["results"]:
            groups[
                (
                    model_label,
                    config["reasoning"],
                    float(config["temperature"]),
                    row["api"],
                    bool(row["stream"]),
                )
            ].append(row)
        receipt_hashes.append({"path": str(relative), "sha256": file_sha256(path)})
    cells = []
    for key, group in sorted(groups.items()):
        completed = [row for row in group if "error" not in row]
        correct = sum(bool(row["correct"]) for row in completed)
        cells.append(
            {
                "model_label": key[0],
                "reasoning_effort": key[1],
                "temperature": key[2],
                "api": key[3],
                "stream": key[4],
                "attempts": len(group),
                "completed": len(completed),
                "correct": correct,
                "wrong": len(completed) - correct,
                "request_errors": len(group) - len(completed),
                "wrong_answers": dict(
                    Counter(str(row.get("answer")) for row in completed if not row["correct"])
                ),
            }
        )
    combined: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for cell in cells:
        combined[(cell["model_label"], cell["reasoning_effort"], cell["temperature"])].append(cell)
    temperature_cells = []
    for key, group in sorted(combined.items()):
        attempts = sum(int(cell["attempts"]) for cell in group)
        completed = sum(int(cell["completed"]) for cell in group)
        correct = sum(int(cell["correct"]) for cell in group)
        wrong_answers: Counter[str] = Counter()
        for cell in group:
            wrong_answers.update(cell["wrong_answers"])
        temperature_cells.append(
            {
                "model_label": key[0],
                "reasoning_effort": key[1],
                "temperature": key[2],
                "attempts": attempts,
                "completed": completed,
                "correct": correct,
                "wrong": completed - correct,
                "request_errors": attempts - completed,
                "wrong_answers": dict(wrong_answers),
            }
        )
    receipt_set_sha256 = hashlib.sha256(
        "".join(f"{row['path']}\0{row['sha256']}\n" for row in receipt_hashes).encode("utf-8")
    ).hexdigest()
    return {
        "root": str(root),
        "json_receipts": len(files),
        "receipt_set_sha256": receipt_set_sha256,
        "temperature_cells": temperature_cells,
        "api_stream_cells": cells,
    }


def aggregate_isolation_root(root: Path) -> dict[str, object]:
    cells = []
    receipt_hashes = []
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data["results"]
        completed = [row for row in rows if "error" not in row]
        cells.append(
            {
                "replica_role": path.parent.name,
                "profile": path.stem,
                "attempts": len(rows),
                "completed": len(completed),
                "correct": sum(bool(row["correct"]) for row in completed),
                "wrong": sum(not bool(row["correct"]) for row in completed),
                "request_errors": len(rows) - len(completed),
            }
        )
        receipt_hashes.append(
            {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
        )
    return {
        "root": str(root),
        "json_receipts": len(receipt_hashes),
        "receipt_set_sha256": hashlib.sha256(
            "".join(f"{row['path']}\0{row['sha256']}\n" for row in receipt_hashes).encode("utf-8")
        ).hexdigest(),
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sentinel-none-root", type=Path, required=True)
    parser.add_argument("--sentinel-low-root", type=Path, required=True)
    parser.add_argument("--isolation-root", type=Path, required=True)
    parser.add_argument("--baseline-label", default="NVFP4")
    parser.add_argument("--candidate-label", default="QAD")
    parser.add_argument("--bootstrap-draws", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260915)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    attempts_path = args.run_root / "attempts.jsonl"
    rows = load_jsonl(attempts_path)
    if len(rows) != 9_600 or any("error" in row for row in rows):
        raise ValueError("generalization evidence must contain 9,600 completed attempts")
    if len({str(row["task_id"]) for row in rows}) != 600:
        raise ValueError("generalization evidence must contain 600 unique task clusters")
    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = []
    for reasoning in ("none", "low"):
        for temperatures in ((0.0,), (1.0,), (0.0, 1.0)):
            comparisons.append(
                task_cluster_comparison(
                    rows,
                    args.baseline_label,
                    args.candidate_label,
                    reasoning,
                    temperatures,
                    args.bootstrap_draws,
                    rng,
                )
            )
    families = sorted({str(row["family"]) for row in rows})
    family_comparisons = [
        task_cluster_comparison(
            rows,
            args.baseline_label,
            args.candidate_label,
            "none",
            (0.0, 1.0),
            args.bootstrap_draws,
            rng,
            family,
        )
        for family in families
    ]
    report = {
        "schema": SCHEMA,
        "status": "qualified",
        "scope": "Exact direct-answer integer arithmetic under the declared served-system configurations",
        "source_artifacts": {
            name: {"path": str(args.run_root / name), "sha256": file_sha256(args.run_root / name)}
            for name in ("attempts.jsonl", "run-config.json", "suite.json", "summary.json")
        },
        "generalization": {
            "cells": direct_cells(rows),
            "task_cluster_comparisons": comparisons,
            "family_comparisons_for_reasoning_none": family_comparisons,
            "combined_answer_classes": combined_answer_classes(rows, "none"),
            "task_reliability": task_reliability(rows, "none"),
            "statistical_contract": {
                "bootstrap_seed": args.bootstrap_seed,
                "bootstrap_draws": args.bootstrap_draws,
                "interval": "two-sided 95% percentile interval",
                "resampling_unit": "task_id",
                "warning": "Matching by task controls task difficulty; independently sampled model outputs are not paired causal observations.",
            },
        },
        "external_sentinel": {
            "prompt": "Case stability3-effort-76. Compute 99 times 17. Give only the integer as your final answer.",
            "expected": "1683",
            "source_gist": "https://gist.github.com/ktsaou/cd36c6bed6a4a1947ff15c632964dd2d",
            "source_revision": "8e2d144ef0ac4287183cbc7a34b7ec22e3c993be",
            "source_script_sha256": "f1d39496647be574c7ed0cd2456c6fea0a12b04f2990f8e9ce4ff0329f9774e2",
            "reasoning_none": aggregate_probe_root(
                args.sentinel_none_root, {"non-qad": args.baseline_label, "qad": args.candidate_label}
            ),
            "reasoning_low_control": aggregate_probe_root(
                args.sentinel_low_root, {"non-qad": args.baseline_label, "qad": args.candidate_label}
            ),
            "replica_isolation": aggregate_isolation_root(args.isolation_root),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(f"Wrote {args.output} ({file_sha256(args.output)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
