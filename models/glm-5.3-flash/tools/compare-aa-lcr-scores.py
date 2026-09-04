#!/usr/bin/env python3
"""Compare two complete AA-LCR equality-checker result sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPECTED_QUESTIONS = 100
EXPECTED_REPEATS = 3


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_result(
    root: Path,
) -> tuple[dict[tuple[int, int], bool], dict[str, Any]]:
    root = root.resolve()
    manifest_path = root / "judge-manifest.json"
    summary_path = root / "pass-at-1-summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "qualified":
        raise RuntimeError(f"AA-LCR score is not qualified: {summary_path}")

    labels: dict[tuple[int, int], bool] = {}
    receipts: list[tuple[str, str]] = []
    for path in sorted((root / "judgements").glob("repeat-*/question-????.json")):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        key = (int(receipt["question_id"]), int(receipt["repeat"]))
        if key in labels:
            raise RuntimeError(f"duplicate question-repeat receipt: {path}")
        if receipt.get("status") != "qualified":
            raise RuntimeError(f"judgement receipt is not qualified: {path}")
        label = receipt.get("label")
        if label not in {"CORRECT", "INCORRECT"}:
            raise RuntimeError(f"judgement receipt has an invalid label: {path}")
        if bool(receipt.get("correct")) != (label == "CORRECT"):
            raise RuntimeError(f"judgement correctness flag is invalid: {path}")
        labels[key] = label == "CORRECT"
        receipts.append((path.relative_to(root).as_posix(), sha256_file(path)))

    expected = {
        (question_id, repeat)
        for question_id in range(1, EXPECTED_QUESTIONS + 1)
        for repeat in range(EXPECTED_REPEATS)
    }
    if set(labels) != expected:
        raise RuntimeError(f"AA-LCR result does not contain 300 pairs: {root}")
    if summary.get("attempts") != len(labels):
        raise RuntimeError(f"AA-LCR summary attempt count drifted: {summary_path}")
    if summary.get("correct") != sum(labels.values()):
        raise RuntimeError(f"AA-LCR summary correct count drifted: {summary_path}")

    identity = {
        "root": str(root),
        "judge_manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
        "receipt_manifest_sha256": canonical_sha256(receipts),
        "judge": manifest.get("judge"),
        "generation_manifest_sha256": manifest.get("generation_manifest_sha256")
        or manifest.get("judge", {}).get("generation_manifest_sha256"),
    }
    return labels, identity


def judge_contract(judge: dict[str, Any]) -> dict[str, Any]:
    """Return the judge configuration without the evaluated input identity."""
    contract = dict(judge)
    contract.pop("generation_manifest_sha256", None)
    return contract


def exact_mcnemar_two_sided(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    tail = min(first_only, second_only)
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def interpolated_quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    fraction = position - lower
    upper = min(lower + 1, len(values) - 1)
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def question_cluster_bootstrap(
    per_question_differences: list[float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    generator = random.Random(seed)
    size = len(per_question_differences)
    samples = [
        sum(per_question_differences[generator.randrange(size)] for _ in range(size))
        / size
        for _ in range(replicates)
    ]
    samples.sort()
    return (
        interpolated_quantile(samples, 0.025),
        interpolated_quantile(samples, 0.975),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=200_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_815)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap replicate count must be positive")

    reference, reference_identity = load_result(args.reference_dir)
    candidate, candidate_identity = load_result(args.candidate_dir)
    reference_judge = judge_contract(reference_identity["judge"])
    candidate_judge = judge_contract(candidate_identity["judge"])
    if reference_judge != candidate_judge:
        raise RuntimeError(
            "AA-LCR paired score comparison requires an identical judge contract"
        )

    both_correct = sum(reference[key] and candidate[key] for key in reference)
    reference_only = sum(reference[key] and not candidate[key] for key in reference)
    candidate_only = sum(not reference[key] and candidate[key] for key in reference)
    both_incorrect = sum(not reference[key] and not candidate[key] for key in reference)
    per_question_differences = [
        sum(
            int(candidate[(question_id, repeat)])
            - int(reference[(question_id, repeat)])
            for repeat in range(EXPECTED_REPEATS)
        )
        / EXPECTED_REPEATS
        for question_id in range(1, EXPECTED_QUESTIONS + 1)
    ]
    interval = question_cluster_bootstrap(
        per_question_differences,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    reference_correct = sum(reference.values())
    candidate_correct = sum(candidate.values())
    paired_attempts = len(reference)
    result = {
        "artifact_kind": "AA-LCR paired score comparison",
        "status": "qualified",
        "created_utc": utc_now(),
        "scoring": "Mean equality-checker correctness across 300 attempts",
        "reference": {
            "label": args.reference_label,
            "correct": reference_correct,
            "attempts": paired_attempts,
            "pass_at_1": reference_correct / paired_attempts,
            "identity": reference_identity,
        },
        "candidate": {
            "label": args.candidate_label,
            "correct": candidate_correct,
            "attempts": paired_attempts,
            "pass_at_1": candidate_correct / paired_attempts,
            "identity": candidate_identity,
        },
        "paired": {
            "attempts": paired_attempts,
            "candidate_minus_reference": (candidate_correct - reference_correct)
            / paired_attempts,
            "both_correct": both_correct,
            "reference_only_correct": reference_only,
            "candidate_only_correct": candidate_only,
            "both_incorrect": both_incorrect,
            "exact_mcnemar_two_sided_p": exact_mcnemar_two_sided(
                reference_only,
                candidate_only,
            ),
            "per_question_difference_counts": {
                f"{difference:.6f}": count
                for difference, count in sorted(
                    Counter(per_question_differences).items()
                )
            },
            "question_cluster_bootstrap_95": {
                "low": interval[0],
                "high": interval[1],
                "replicates": args.bootstrap_replicates,
                "seed": args.bootstrap_seed,
                "sampling_unit": "AA-LCR question with all three attempts",
            },
        },
        "interpretation_limit": (
            "The question-cluster bootstrap treats the three observed generations "
            "per checkpoint and question as fixed. It estimates variation across "
            "questions and does not include additional free-running generation or "
            "equality-checker repeat variation."
        ),
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
