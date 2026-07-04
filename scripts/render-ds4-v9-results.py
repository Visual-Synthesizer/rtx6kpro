#!/usr/bin/env python3
"""Render DS4 v9 benchmark JSON artifacts as wiki markdown tables."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


BACKENDS = [
    "b12x-a16",
    "b12x-a8",
    "b12x-a8-dglin",
    "lucifer-default",
    "lucifer-cutlass",
]
STANDARD_MODES = ["standard-mtp0", "standard-mtp2", "standard-mtp3"]
TPS = [2, 4]


def finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"missing/non-numeric {label}: {value!r}") from exc
    if not math.isfinite(result):
        raise SystemExit(f"non-finite {label}: {value!r}")
    return result


def load_case(out: Path, tp: int, backend: str, mode: str) -> dict[str, float]:
    case_dir = out / f"tp{tp}-{backend}-{mode}"
    decode_path = case_dir / "decode.json"
    prefill_path = case_dir / "prefill.json"
    if not decode_path.exists() or not prefill_path.exists():
        raise SystemExit(f"missing decode/prefill JSON for {case_dir}")

    decode = json.loads(decode_path.read_text())
    prefill = json.loads(prefill_path.read_text())

    cc: dict[int, float] = {}
    for row in decode.get("results", []):
        if int(row.get("context_tokens", -1)) == 0:
            cc[int(row.get("concurrency", 0))] = finite_float(
                row.get("aggregate_tps"),
                f"{case_dir.name} cc{row.get('concurrency')}",
            )
    missing = [value for value in (1, 16, 32, 64) if value not in cc]
    if missing:
        raise SystemExit(f"missing decode concurrency {missing} for {case_dir}")

    coding = decode.get("coding_peak", {}).get("summary", {})
    prefill_rows = prefill.get("prefill", {})
    return {
        "cc1": cc[1],
        "cc16": cc[16],
        "cc32": cc[32],
        "cc64": cc[64],
        "coding": finite_float(coding.get("median_generation_tok_s"), f"{case_dir.name} coding"),
        "cjk": int(coding.get("cjk_runs", -1)),
        "p8": finite_float(prefill_rows.get("8192", {}).get("tok_per_sec"), f"{case_dir.name} 8k"),
        "p64": finite_float(prefill_rows.get("65536", {}).get("tok_per_sec"), f"{case_dir.name} 64k"),
        "p128": finite_float(prefill_rows.get("131072", {}).get("tok_per_sec"), f"{case_dir.name} 128k"),
    }


def f1(value: float) -> str:
    return f"{value:.1f}"


def f0(value: float) -> str:
    return f"{value:.0f}"


def decode_table(out: Path, dspark: bool) -> str:
    lines = [
        "| TP | Backend | Mode | cc1 tok/s | cc16 tok/s | cc32 tok/s | cc64 tok/s | coding peak median | CJK runs |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for tp in TPS:
        for backend in BACKENDS:
            modes = ["dspark"] if dspark else STANDARD_MODES
            for mode in modes:
                row = load_case(out, tp, backend, mode)
                lines.append(
                    f"| {tp} | {backend} | {mode} | {f1(row['cc1'])} | "
                    f"{f1(row['cc16'])} | {f1(row['cc32'])} | {f1(row['cc64'])} | "
                    f"{f1(row['coding'])} | {int(row['cjk'])} |"
                )
    return "\n".join(lines)


def prefill_table(out: Path, dspark: bool) -> str:
    lines = [
        "| TP | Backend | Mode | 8k tok/s | 64k tok/s | 128k tok/s | Note |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for tp in TPS:
        for backend in BACKENDS:
            modes = ["dspark"] if dspark else STANDARD_MODES
            for mode in modes:
                row = load_case(out, tp, backend, mode)
                note = "DeepGEMM linear" if backend == "b12x-a8-dglin" else ""
                lines.append(
                    f"| {tp} | {backend} | {mode} | {f0(row['p8'])} | "
                    f"{f0(row['p64'])} | {f0(row['p128'])} | {note} |"
                )
    return "\n".join(lines)


def validate_case_set(out: Path) -> None:
    expected = {
        f"tp{tp}-{backend}-{mode}"
        for tp in TPS
        for backend in BACKENDS
        for mode in [*STANDARD_MODES, "dspark"]
    }
    actual = {path.name for path in out.iterdir() if re.match(r"tp[24]-", path.name)}
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        raise SystemExit(f"missing case directories: {', '.join(missing)}")
    if extra:
        raise SystemExit(f"unexpected case directories: {', '.join(extra)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path, help="Sweep output directory")
    args = parser.parse_args()

    validate_case_set(args.out)
    print("## Decode Throughput\n")
    print("### DSpark Checkpoint\n")
    print(decode_table(args.out, dspark=True))
    print("\n### Standard Checkpoint\n")
    print(decode_table(args.out, dspark=False))
    print("\n## Prefill Throughput\n")
    print("### DSpark Checkpoint\n")
    print(prefill_table(args.out, dspark=True))
    print("\n### Standard Checkpoint\n")
    print(prefill_table(args.out, dspark=False))


if __name__ == "__main__":
    main()
