#!/usr/bin/env python3
"""Render the GLM-5.2 v16 JSON sweep as compact Markdown tables."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


CASE_NAMES = {
    "nvfp4-a4-orig": "Luke NVFP4 A4 original",
    "nvfp4-a4-online-mxfp8": "Luke NVFP4 A4 online MXFP8",
    "nvfp4-a16-orig": "Luke NVFP4 A16 original",
    "nvfp4-a16-online-mxfp8": "Luke NVFP4 A16 online MXFP8",
    "mxfp4-a8-orig": "AMD MXFP4 experts A8 original",
    "mxfp4-a8-online-mxfp8": "AMD MXFP4 experts A8 online MXFP8",
    "mxfp4-a8-online-fp8": "AMD MXFP4 experts A8 online FP8",
}


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def load_record(path: Path) -> dict:
    decode = load_json(path / "decode.json", {}).get("results", [])
    decode_tps = {
        int(row["concurrency"]): float(
            row.get("aggregate_tps") or row.get("server_gen_throughput")
        )
        for row in decode
        if int(row.get("context_tokens", -1)) == 0
    }
    prefill = load_json(path / "prefill.json", {}).get("prefill", {})
    prefill_tps = {
        int(context): float(value["tok_per_sec"])
        for context, value in prefill.items()
        if value.get("tok_per_sec") is not None
    }
    runtime = load_json(path / "runtime.json", {})
    inspect = load_json(path / "container.inspect.json", [{}])
    image_id = inspect[0].get("Image", "") if inspect else ""
    return {
        "path": path,
        "tp": int(path.parents[3].name.removeprefix("tp")),
        "mtp": int(path.parents[2].name.removeprefix("mtp")),
        "f8": path.parents[1].name.removeprefix("f8-"),
        "case": path.parent.name,
        "dcp": int(path.name.removeprefix("dcp")),
        "decode": decode_tps,
        "prefill": prefill_tps,
        "runtime": runtime,
        "image_id": image_id,
        "gated": (path / "gate-passed").is_file(),
    }


def fmt_tps(value: float | None) -> str:
    return "-" if value is None else f"{value:,.1f}"


def fmt_int(value: int | float | None) -> str:
    return "-" if value is None else f"{value:,.0f}"


def print_speed_group(title: str, records: list[dict]) -> None:
    print(f"## {title}\n")
    for case in sorted({record["case"] for record in records}):
        rows = sorted(
            (record for record in records if record["case"] == case),
            key=lambda record: record["dcp"],
        )
        print(f"### {CASE_NAMES.get(case, case)}\n")
        print("| DCP | C1 | C2 | C4 | C8 | C16 | C32 | Prefill 8k | Prefill 64k | KV tokens |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in rows:
            values = [fmt_tps(row["decode"].get(cc)) for cc in (1, 2, 4, 8, 16, 32)]
            print(
                f'| {row["dcp"]} | ' + " | ".join(values)
                + f' | {fmt_int(row["prefill"].get(8192))}'
                + f' | {fmt_int(row["prefill"].get(65536))}'
                + f' | {fmt_int(row["runtime"].get("gpu_kv_cache_tokens"))} |'
            )
        print()


def print_dma(records: list[dict]) -> None:
    rows = {
        (record["dcp"], record["f8"], record["case"]): record
        for record in records
        if record["tp"] == 8
        and record["mtp"] == 3
        and record["f8"] in {"ag", "ring"}
    }
    if not rows:
        return
    print("## FP8 PCIe DMA Prefill\n")
    print("Decode is intentionally omitted because this DMA mode only affects prefill communication.\n")
    print("| DCP | AG original | AG online MXFP8 | Ring original | Ring online MXFP8 |")
    print("|---:|---:|---:|---:|---:|")
    for dcp in (1, 2, 4, 8):
        values = []
        for dma, case in (
            ("ag", "nvfp4-a4-orig"),
            ("ag", "nvfp4-a4-online-mxfp8"),
            ("ring", "nvfp4-a4-orig"),
            ("ring", "nvfp4-a4-online-mxfp8"),
        ):
            record = rows.get((dcp, dma, case))
            values.append(fmt_int(record["prefill"].get(65536)) if record else "-")
        print(f"| {dcp} | " + " | ".join(values) + " |")
    print()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument(
        "--expected-image-id",
        help="Require every rendered cell to use this full sha256 image ID.",
    )
    args = parser.parse_args()

    records = [
        load_record(path)
        for path in sorted(args.result_root.glob("tp*/mtp*/f8-*/*/dcp*"))
        if (path / "decode.json").is_file() or (path / "prefill.json").is_file()
    ]
    if not records:
        raise SystemExit(f"no benchmark cells found under {args.result_root}")

    ungated = [record["path"] for record in records if not record["gated"]]
    if ungated:
        raise SystemExit("ungated benchmark cells: " + ", ".join(map(str, ungated)))
    if args.expected_image_id:
        mismatched = [
            record["path"]
            for record in records
            if record["image_id"] != args.expected_image_id
        ]
        if mismatched:
            raise SystemExit(
                "cells from a different image: " + ", ".join(map(str, mismatched))
            )

    image_counts = Counter(record["image_id"] for record in records)
    print("# GLM-5.2 v16 Sweep Tables\n")
    print(f"Validated benchmark cells: `{len(records)}`.\n")
    print("Image IDs: " + ", ".join(f"`{key}` ({value})" for key, value in image_counts.items()) + ".\n")

    print_speed_group(
        "TP8 MTP0, FP8 DMA Off",
        [record for record in records if record["tp"] == 8 and record["mtp"] == 0 and record["f8"] == "0"],
    )
    print_speed_group(
        "TP8 MTP3, FP8 DMA Off",
        [record for record in records if record["tp"] == 8 and record["mtp"] == 3 and record["f8"] == "0"],
    )
    print_dma(records)
    print_speed_group(
        "TP6 MTP0, FP8 DMA Off",
        [record for record in records if record["tp"] == 6 and record["mtp"] == 0 and record["f8"] == "0"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
