#!/usr/bin/env python3
"""Render the DeepSeek-V4-Flash Infernal Invocation qualification matrix."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


TPS = (2, 4)
BACKENDS = ("b12x-a16", "b12x-a8", "b12x-a8-dglin")
PROFILES = (
    "dspark-mtp0",
    "dspark-k5",
    "dspark-k7",
    "dspark-k7-dynamic",
)
PROFILE_LABELS = {
    "dspark-mtp0": "Target only",
    "dspark-k5": "Fixed probabilistic K5",
    "dspark-k7": "Fixed probabilistic K7",
    "dspark-k7-dynamic": "Confidence-controlled K7",
}
CONCURRENCIES = (1, 16, 32, 64)
PREFILL_LENGTHS = (8192, 65536, 131072)
PREFILL_TARGET_TOLERANCE = 256


def finite(value: object, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{field} is missing or non-numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise SystemExit(f"{field} is non-finite: {value!r}")
    return result


def select_prefill_row(
    rows: dict[str, object], target: int, case_name: str
) -> tuple[int, dict[str, object]]:
    candidates: list[tuple[int, dict[str, object]]] = []
    for key, row in rows.items():
        try:
            actual = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(row, dict) and 0 <= target - actual <= PREFILL_TARGET_TOLERANCE:
            candidates.append((actual, row))
    if not candidates:
        raise SystemExit(f"{case_name} lacks nominal prefill row {target}")
    return max(candidates)


def load_case(root: Path, tp: int, backend: str, profile: str) -> dict[str, object]:
    name = f"tp{tp}-{backend}-{profile}"
    case = root / name
    try:
        decode = json.loads((case / "decode.json").read_text())
        prefill = json.loads((case / "prefill.json").read_text())
    except FileNotFoundError as exc:
        raise SystemExit(f"qualification artifact is missing: {exc.filename}") from exc

    rows = {
        int(row["concurrency"]): row
        for row in decode.get("results", ())
        if int(row.get("context_tokens", -1)) == 0
    }
    missing_concurrency = sorted(set(CONCURRENCIES) - rows.keys())
    if missing_concurrency:
        raise SystemExit(f"{name} lacks decode rows {missing_concurrency}")

    prefill_rows = prefill.get("prefill", {})
    selected_prefill = {
        length: select_prefill_row(prefill_rows, length, name)
        for length in PREFILL_LENGTHS
    }

    coding = decode.get("coding_peak", {}).get("summary", {})
    return {
        "decode": {
            concurrency: finite(
                rows[concurrency].get("aggregate_tps"), f"{name} C{concurrency}"
            )
            for concurrency in CONCURRENCIES
        },
        "acceptance": {
            concurrency: finite(
                rows[concurrency].get("server_spec_accept_rate", 0.0),
                f"{name} C{concurrency} acceptance",
            )
            for concurrency in CONCURRENCIES
        },
        "coding": finite(
            coding.get("median_generation_tok_s"), f"{name} coding median"
        ),
        "cjk_runs": int(coding.get("cjk_runs", -1)),
        "prefill": {
            length: finite(row.get("tok_per_sec"), f"{name} {length} prefill")
            for length, (_, row) in selected_prefill.items()
        },
        "prefill_target": {
            length: actual for length, (actual, _) in selected_prefill.items()
        },
        "prefill_prompt_tokens": {
            length: int(row.get("prompt_tokens", 0))
            for length, (_, row) in selected_prefill.items()
        },
    }


def validate_case_set(root: Path) -> None:
    expected = {
        f"tp{tp}-{backend}-{profile}"
        for tp in TPS
        for backend in BACKENDS
        for profile in PROFILES
    }
    actual = {
        path.name for path in root.iterdir() if re.fullmatch(r"tp[24]-.*", path.name)
    }
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SystemExit(
            f"qualification case mismatch; missing={missing}, extra={extra}"
        )


def decode_table(root: Path) -> str:
    lines = [
        "| TP | Backend | Profile | C1 | C16 | C32 | C64 | Coding median | C1 accept | C64 accept | CJK |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tp in TPS:
        for backend in BACKENDS:
            for profile in PROFILES:
                result = load_case(root, tp, backend, profile)
                decode = result["decode"]
                acceptance = result["acceptance"]
                lines.append(
                    f"| {tp} | {backend} | {PROFILE_LABELS[profile]} | "
                    f"{decode[1]:.1f} | {decode[16]:.1f} | {decode[32]:.1f} | "
                    f"{decode[64]:.1f} | {result['coding']:.1f} | "
                    f"{acceptance[1] * 100:.1f}% | {acceptance[64] * 100:.1f}% | "
                    f"{result['cjk_runs']} |"
                )
    return "\n".join(lines)


def prefill_table(root: Path) -> str:
    lines = [
        "| TP | Backend | Profile | 8k | 64k | 128k |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for tp in TPS:
        for backend in BACKENDS:
            for profile in PROFILES:
                result = load_case(root, tp, backend, profile)
                prefill = result["prefill"]
                lines.append(
                    f"| {tp} | {backend} | {PROFILE_LABELS[profile]} | "
                    f"{prefill[8192]:.0f} | {prefill[65536]:.0f} | "
                    f"{prefill[131072]:.0f} |"
                )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    validate_case_set(args.artifact_root)
    print("## Decode throughput\n")
    print(decode_table(args.artifact_root))
    print("\n## Prefill throughput\n")
    print(prefill_table(args.artifact_root))
    print(
        "\nThe 128k label is the requested benchmark class. A server may cap "
        "the target by up to 256 tokens to reserve output space; each JSON "
        "artifact records the effective target and measured prompt-token count."
    )


if __name__ == "__main__":
    main()
