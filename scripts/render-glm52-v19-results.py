#!/usr/bin/env python3
"""Render GLM-5.2 v19 benchmark JSONs as compact Markdown tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


LABELS = {
    "nvfp4-a4-orig": "Luke NVFP4 A4 original",
    "nvfp4-a4-online-mxfp8": "Luke NVFP4 A4 online MXFP8",
    "nvfp4-a16-orig": "Luke NVFP4 A16 original",
    "nvfp4-a16-online-mxfp8": "Luke NVFP4 A16 online MXFP8",
    "mxfp4-a8-orig": "AMD MXFP4 experts A8 original",
    "mxfp4-a8-online-mxfp8": "AMD MXFP4 experts A8 online MXFP8",
    "mxfp4-a8-online-fp8": "AMD MXFP4 experts A8 online FP8",
    "nf3-hybrid-a16": "NVFP4/NF3 hybrid A16",
}

TP8_CASES = (
    "nvfp4-a4-orig",
    "nvfp4-a4-online-mxfp8",
    "nvfp4-a16-orig",
    "nvfp4-a16-online-mxfp8",
    "mxfp4-a8-orig",
    "mxfp4-a8-online-mxfp8",
    "mxfp4-a8-online-fp8",
)
TP8_MTP3_CASES = TP8_CASES[:4]
TP6_CASES = ("mxfp4-a8-orig", "mxfp4-a8-online-mxfp8")

KLD = {
    "nvfp4-a4-orig": (0.10228, 0.00634),
    "nvfp4-a4-online-mxfp8": (0.10800, 0.00697),
    "nvfp4-a16-orig": (0.05994, 0.00129),
    "nvfp4-a16-online-mxfp8": (0.06587, 0.00253),
    "mxfp4-a8-orig": (0.08160, 0.00432),
    "mxfp4-a8-online-mxfp8": (0.08030, 0.00309),
}

BEGIN_MARKER = "<!-- BEGIN GENERATED V19 FULL RESULTS -->"
END_MARKER = "<!-- END GENERATED V19 FULL RESULTS -->"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def number(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "TBD"
    return f"{value:,.{decimals}f}"


def load_results(root: Path) -> dict[tuple[int, int, int, str], dict]:
    results: dict[tuple[int, int, int, str], dict] = {}
    pattern = re.compile(r"tp(\d+)-dcp(\d+)-mtp(\d+)")
    for path in sorted(root.glob("tp*-dcp*-mtp*/*/summary.json")):
        match = pattern.fullmatch(path.parent.parent.name)
        if not match:
            continue
        tp, dcp, mtp = map(int, match.groups())
        results[(tp, dcp, mtp, path.parent.name)] = json.loads(path.read_text())
    return results


def expected_keys() -> set[tuple[int, int, int, str]]:
    keys = {
        (8, dcp, 0, case)
        for dcp in (1, 2, 4, 8)
        for case in TP8_CASES
    }
    keys.update(
        (8, dcp, 3, case)
        for dcp in (1, 2, 4, 8)
        for case in TP8_MTP3_CASES
    )
    keys.update((8, 1, 3, case) for case in TP6_CASES)
    keys.update(
        (6, dcp, 0, case)
        for dcp in (1, 2, 3, 6)
        for case in TP6_CASES
    )
    keys.update(
        (6, dcp, 3, case)
        for dcp in (3, 6)
        for case in TP6_CASES
    )
    keys.update(
        (4, dcp, mtp, "nf3-hybrid-a16")
        for dcp in (1, 2, 4)
        for mtp in (0, 3)
    )
    return keys


def get(results, tp: int, dcp: int, mtp: int, case: str) -> dict:
    return results.get((tp, dcp, mtp, case), {})


def render_decision(results) -> str:
    rows = []
    for case in TP8_CASES[:6]:
        dcp1 = get(results, 8, 1, 0, case)
        dcp4 = get(results, 8, 4, 0, case)
        dcp8 = get(results, 8, 8, 0, case)
        mean, sd = KLD[case]
        rows.append(
            [
                LABELS[case],
                f"{mean:.5f} +/- {sd:.5f}",
                number(dcp1.get("decode_cc1"), 2),
                number(dcp1.get("decode_cc32"), 1),
                number(dcp1.get("prefill_64k_median")),
                number(dcp4.get("prefill_64k_median")),
                number(dcp8.get("prefill_64k_median")),
            ]
        )
    return table(
        [
            "Case",
            "KLD mean +/- sd",
            "DCP1 CC1",
            "DCP1 CC32",
            "DCP1 64k",
            "DCP4 64k",
            "DCP8 64k",
        ],
        rows,
    )


def render_prefill(results, tp: int, mtp: int, cases, dcps) -> str:
    parts = []
    for metric, label in (
        ("prefill_8k_median", "Prefill 8k"),
        ("prefill_64k_median", "Prefill 64k"),
    ):
        rows = []
        for case in cases:
            rows.append(
                [LABELS[case]]
                + [number(get(results, tp, dcp, mtp, case).get(metric)) for dcp in dcps]
            )
        parts.extend(
            [
                f"#### {label}",
                "",
                table(["Case"] + [f"DCP{dcp}" for dcp in dcps], rows),
                "",
            ]
        )
    return "\n".join(parts).rstrip()


def render_decode(results, tp: int, mtp: int, cases, dcps) -> str:
    parts = []
    for dcp in dcps:
        rows = []
        for case in cases:
            decode = get(results, tp, dcp, mtp, case).get("decode", {})
            rows.append(
                [LABELS[case]]
                + [number(decode.get(str(cc)), 2) for cc in (1, 2, 4, 8, 16, 32)]
            )
        parts.extend(
            [
                f"#### DCP{dcp}",
                "",
                table(["Case", "CC1", "CC2", "CC4", "CC8", "CC16", "CC32"], rows),
                "",
            ]
        )
    return "\n".join(parts).rstrip()


def render_capacity(results, tp: int, mtp: int, cases, dcps) -> str:
    rows = []
    for case in cases:
        rows.append(
            [LABELS[case]]
            + [number(get(results, tp, dcp, mtp, case).get("kv_tokens")) for dcp in dcps]
        )
    return table(["Case"] + [f"DCP{dcp}" for dcp in dcps], rows)


def render_acceptance(results, tp: int, cases, dcps) -> str:
    rows = []
    for case in cases:
        row = [LABELS[case]]
        for dcp in dcps:
            result = get(results, tp, dcp, 3, case)
            length = result.get("mean_acceptance_length")
            rate = result.get("draft_acceptance_rate")
            row.append(
                "TBD"
                if length is None
                else f"{length:.3f} / {rate:.3f}"
            )
        rows.append(row)
    return table(["Case"] + [f"DCP{dcp}" for dcp in dcps], rows)


def render_path_status(results, tp: int, mtp: int, cases, dcps) -> str:
    rows = []
    for dcp in dcps:
        values = [get(results, tp, dcp, mtp, case) for case in cases]
        measured = [value for value in values if value]
        if not measured:
            status = "TBD"
        else:
            fast = all(value.get("fast_dcp_path") for value in measured)
            workspace = all(value.get("borrowed_workspace_path") for value in measured)
            status = f"fast={str(fast).lower()}, workspace={str(workspace).lower()}"
        rows.append([f"TP{tp}/DCP{dcp}", status])
    return table(["Topology", "Observed runtime path"], rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--update-page", type=Path)
    parser.add_argument("--check-complete", action="store_true")
    args = parser.parse_args()
    results = load_results(args.result_root)
    missing = expected_keys() - results.keys()
    if args.check_complete and missing:
        details = "\n".join(
            f"  TP{tp}/DCP{dcp}/MTP{mtp} {case}"
            for tp, dcp, mtp, case in sorted(missing)
        )
        raise SystemExit(f"missing {len(missing)} expected result(s):\n{details}")

    sections = [
        "### Cross-Quant Decision Table\n\n" + render_decision(results),
        "### TP8 MTP0 Prefill\n\n" + render_prefill(results, 8, 0, TP8_CASES, (1, 2, 4, 8)),
        "<details>\n<summary>TP8 MTP0 full decode sweep</summary>\n\n"
        + render_decode(results, 8, 0, TP8_CASES, (1, 2, 4, 8))
        + "\n\n</details>",
        "### TP8 MTP0 KV Capacity\n\n" + render_capacity(results, 8, 0, TP8_CASES, (1, 2, 4, 8)),
        "### TP8 MTP0 DCP Path Verification\n\n"
        + render_path_status(results, 8, 0, TP8_CASES, (1, 2, 4, 8)),
        "### TP8 MTP3 Prefill\n\n" + render_prefill(results, 8, 3, TP8_MTP3_CASES, (1, 2, 4, 8)),
        "### TP8 MTP3 Acceptance\n\nCells are `mean accepted length / draft acceptance rate`.\n\n"
        + render_acceptance(results, 8, TP8_MTP3_CASES, (1, 2, 4, 8)),
        "<details>\n<summary>TP8 MTP3 full decode sweep</summary>\n\n"
        + render_decode(results, 8, 3, TP8_MTP3_CASES, (1, 2, 4, 8))
        + "\n\n</details>",
        "### TP6 MTP0 Prefill\n\n" + render_prefill(results, 6, 0, TP6_CASES, (1, 2, 3, 6)),
        "<details>\n<summary>TP6 MTP0 full decode sweep</summary>\n\n"
        + render_decode(results, 6, 0, TP6_CASES, (1, 2, 3, 6))
        + "\n\n</details>",
        "### TP6 MTP3 Prefill\n\n" + render_prefill(results, 6, 3, TP6_CASES, (3, 6)),
        "### TP6 MTP3 Acceptance\n\nCells are `mean accepted length / draft acceptance rate`.\n\n"
        + render_acceptance(results, 6, TP6_CASES, (3, 6)),
        "<details>\n<summary>TP6 MTP3 full decode sweep</summary>\n\n"
        + render_decode(results, 6, 3, TP6_CASES, (3, 6))
        + "\n\n</details>",
        "### TP4 NF3 Hybrid Prefill\n\n"
        + render_prefill(results, 4, 0, ("nf3-hybrid-a16",), (1, 2, 4))
        + "\n\n"
        + render_prefill(results, 4, 3, ("nf3-hybrid-a16",), (1, 2, 4)),
        "<details>\n<summary>TP4 NF3 hybrid full decode sweeps</summary>\n\n"
        + render_decode(results, 4, 0, ("nf3-hybrid-a16",), (1, 2, 4))
        + "\n\n"
        + render_decode(results, 4, 3, ("nf3-hybrid-a16",), (1, 2, 4))
        + "\n\n</details>",
    ]
    output = "## Full v19 Performance Campaign\n\n" + "\n\n".join(sections) + "\n"
    if args.update_page:
        page = args.update_page.read_text()
        if page.count(BEGIN_MARKER) != 1 or page.count(END_MARKER) != 1:
            raise SystemExit("target page must contain exactly one generated-results marker pair")
        before, remainder = page.split(BEGIN_MARKER, 1)
        _, after = remainder.split(END_MARKER, 1)
        args.update_page.write_text(
            before
            + BEGIN_MARKER
            + "\n\n"
            + output.rstrip()
            + "\n\n"
            + END_MARKER
            + after
        )
    if args.output:
        args.output.write_text(output)
    elif not args.update_page:
        print(output, end="")


if __name__ == "__main__":
    main()
