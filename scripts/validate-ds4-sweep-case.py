#!/usr/bin/env python3
"""Validate one DeepSeek-V4 benchmark case before it can be reused."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """A benchmark artifact does not satisfy the qualification contract."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _load(case_dir: Path, name: str) -> dict[str, Any]:
    path = case_dir / name
    if not path.exists():
        raise ValidationError(f"missing {name}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_decode(data: dict[str, Any], concurrency: list[int]) -> None:
    """Require a successful, nonempty context-zero row for every load level."""
    rows: dict[int, dict[str, Any]] = {}
    for row in data.get("results", []):
        try:
            context = int(row.get("context_tokens", -1))
            row_concurrency = int(row.get("concurrency", 0))
        except (TypeError, ValueError):
            continue
        if context == 0:
            rows[row_concurrency] = row

    missing = [value for value in concurrency if value not in rows]
    if missing:
        raise ValidationError(f"missing decode result for concurrency {missing}")

    for value in concurrency:
        row = rows[value]
        throughput = row.get("aggregate_tps")
        if not _finite(throughput) or float(throughput) <= 0:
            raise ValidationError(
                f"decode concurrency {value} has nonpositive aggregate_tps"
            )
        if int(row.get("request_count", 0)) <= 0:
            raise ValidationError(f"decode concurrency {value} has no request samples")
        if int(row.get("num_completed", 0)) <= 0:
            raise ValidationError(
                f"decode concurrency {value} has no successful streams"
            )
        if int(row.get("num_errors", 0)) != 0:
            raise ValidationError(
                f"decode concurrency {value} has {row.get('num_errors')} error(s)"
            )

    coding = data.get("coding_peak", {})
    summary = coding.get("summary", {})
    if int(coding.get("runs_ok", -1)) != int(coding.get("runs_requested", -2)):
        raise ValidationError(
            "incomplete coding peak: "
            f"{coding.get('runs_ok')}/{coding.get('runs_requested')} runs"
        )
    if not _finite(summary.get("median_generation_tok_s")):
        raise ValidationError("missing coding peak median_generation_tok_s")
    if int(summary.get("cjk_runs", -1)) != 0:
        raise ValidationError(
            f"coding peak produced CJK in {summary.get('cjk_runs')} run(s)"
        )


def _valid_prefill(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if not _finite(row.get("tok_per_sec")) or not _finite(row.get("ttft_seconds")):
        return False
    # A failed streaming request can be timed as an immediate TTFT and report
    # implausibly high throughput. Qualification requires a positive sample.
    return (
        0 < float(row["tok_per_sec"]) < 1_000_000
        and float(row["ttft_seconds"]) > 0
        and int(row.get("samples", 0)) > 0
    )


def validate_prefill(data: dict[str, Any], contexts: list[int]) -> None:
    """Require a valid sample within 256 tokens below each requested length."""
    rows = data.get("prefill", {})

    def matching_row(target: int) -> tuple[int, dict[str, Any]] | None:
        candidates = []
        for key, row in rows.items():
            try:
                actual = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= target - actual <= 256 and _valid_prefill(row):
                candidates.append((actual, row))
        return max(candidates) if candidates else None

    missing = [context for context in contexts if matching_row(context) is None]
    if missing:
        raise ValidationError(f"missing prefill tok_per_sec for contexts {missing}")


def _parse_context(value: str) -> int:
    normalized = value.strip().lower()
    if normalized.endswith("k"):
        return int(float(normalized[:-1]) * 1024)
    return int(normalized)


def validate_case(
    case_dir: Path,
    decode_concurrency: list[int],
    prefill_contexts: list[int],
    run_decode: bool,
    run_prefill: bool,
) -> None:
    if run_decode:
        validate_decode(_load(case_dir, "decode.json"), decode_concurrency)
    if run_prefill:
        validate_prefill(_load(case_dir, "prefill.json"), prefill_contexts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("--decode-concurrency", default="1,16,32,64")
    parser.add_argument("--prefill-contexts", default="8k,64k,128k")
    parser.add_argument("--run-decode", action="store_true")
    parser.add_argument("--run-prefill", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    decode_concurrency = [
        int(value) for value in args.decode_concurrency.replace(",", " ").split()
    ]
    prefill_contexts = [
        _parse_context(value)
        for value in args.prefill_contexts.replace(",", " ").split()
    ]
    try:
        validate_case(
            args.case_dir,
            decode_concurrency,
            prefill_contexts,
            args.run_decode,
            args.run_prefill,
        )
    except ValidationError as error:
        print(f"invalid benchmark results: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
