#!/usr/bin/env python3
"""Render cumulative GLM-5.2 v17 speed tables from inherited and v17 data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LABELS = {
    "nvfp4-a4-orig": "Luke NVFP4 A4 original",
    "nvfp4-a4-online-mxfp8": "Luke NVFP4 A4 online MXFP8",
    "nvfp4-a16-orig": "Luke NVFP4 A16 original",
    "nvfp4-a16-online-mxfp8": "Luke NVFP4 A16 online MXFP8",
    "mxfp4-a8-orig": "AMD MXFP4 experts A8 original",
    "mxfp4-a8-online-mxfp8": "AMD MXFP4 experts A8 online MXFP8",
}

TP8_CASES = tuple(LABELS)
NVFP4_CASES = TP8_CASES[:4]

KLD = {
    "nvfp4-a4-orig": (0.10228, 0.00634),
    "nvfp4-a4-online-mxfp8": (0.10800, 0.00697),
    "nvfp4-a16-orig": (0.05994, 0.00129),
    "nvfp4-a16-online-mxfp8": (0.06587, 0.00253),
    "mxfp4-a8-orig": (0.08160, 0.00432),
    "mxfp4-a8-online-mxfp8": (0.08030, 0.00309),
}

# v15 decode is inherited because PR #94 is gated to eager large prefill.
DECODE_CC1 = {
    "nvfp4-a4-orig": {1: 87.99, 2: 72.44, 4: 71.65, 8: 67.29},
    "nvfp4-a4-online-mxfp8": {1: 94.96, 2: 76.26, 4: 75.32, 8: 70.84},
    "nvfp4-a16-orig": {1: 86.56, 2: 71.48, 4: 70.74, 8: 66.11},
    "nvfp4-a16-online-mxfp8": {1: 93.30, 2: 74.85, 4: 73.99, 8: 69.45},
    "mxfp4-a8-orig": {1: 88.72, 2: 71.84, 4: 71.73, 8: 67.15},
    "mxfp4-a8-online-mxfp8": {1: 94.03, 2: 75.66, 4: 75.37, 8: 71.01},
}

DECODE_CC32 = {
    "nvfp4-a4-orig": {1: 934.07, 2: 838.57, 4: 747.11, 8: 606.35},
    "nvfp4-a4-online-mxfp8": {1: 953.24, 2: 847.24, 4: 760.87, 8: 617.18},
    "nvfp4-a16-orig": {1: 932.72, 2: 828.30, 4: 750.20, 8: 610.88},
    "nvfp4-a16-online-mxfp8": {1: 954.52, 2: 837.81, 4: 752.91, 8: 610.40},
    "mxfp4-a8-orig": {1: 938.10, 2: 832.28, 4: 745.91, 8: 613.70},
    "mxfp4-a8-online-mxfp8": {1: 956.30, 2: 840.02, 4: 761.43, 8: 607.69},
}

# Canonical v15 standalone-prefill sweep, plus the two MXFP4 rows from the
# v15 TP8 hybrid table. These values are retained only for DCP1 and deltas.
OLD_TP8_MTP0 = {
    "nvfp4-a4-orig": {
        1: (6557, 6257),
        2: (4597, 4675),
        4: (3402, 3457),
        8: (2175, 2195),
    },
    "nvfp4-a4-online-mxfp8": {
        1: (6681, 6351),
        2: (4599, 4724),
        4: (3403, 3492),
        8: (2173, 2209),
    },
    "nvfp4-a16-orig": {
        1: (6140, 5849),
        2: (4369, 4439),
        4: (3279, 3335),
        8: (2121, 2140),
    },
    "nvfp4-a16-online-mxfp8": {
        1: (6239, 5941),
        2: (4360, 4477),
        4: (3280, 3355),
        8: (2121, 2156),
    },
    "mxfp4-a8-orig": {
        1: (6698, 6307),
        2: (4747, 4786),
        4: (3450, 3491),
        8: (2206, 2220),
    },
    "mxfp4-a8-online-mxfp8": {
        1: (6731, 6364),
        2: (4702, 4781),
        4: (3427, 3495),
        8: (2200, 2223),
    },
}

OLD_TP8_MTP3 = {
    "nvfp4-a4-orig": {
        1: (6441, 6136),
        2: (4487, 4570),
        4: (3328, 3392),
        8: (2133, 2156),
    },
    "nvfp4-a4-online-mxfp8": {
        1: (6546, 6222),
        2: (4492, 4618),
        4: (3325, 3422),
        8: (2132, 2166),
    },
    "nvfp4-a16-orig": {
        1: (6016, 5740),
        2: (4262, 4335),
        4: (3211, 3267),
        8: (2079, 2100),
    },
    "nvfp4-a16-online-mxfp8": {
        1: (6109, 5833),
        2: (4261, 4392),
        4: (3209, 3294),
        8: (2081, 2114),
    },
}

OLD_DMA = {
    ("nvfp4-a4-orig", "ag"): {
        1: (7130, 6738),
        2: (4804, 4894),
        4: (3501, 3571),
        8: (2203, 2226),
    },
    ("nvfp4-a4-orig", "ring"): {
        1: (7912, 7435),
        2: (5147, 5272),
        4: (3682, 3757),
        8: (2272, 2300),
    },
    ("nvfp4-a4-online-mxfp8", "ag"): {
        1: (7235, 6843),
        2: (4806, 4963),
        4: (3505, 3602),
        8: (2206, 2240),
    },
    ("nvfp4-a4-online-mxfp8", "ring"): {
        1: (8035, 7564),
        2: (5144, 5328),
        4: (3689, 3791),
        8: (2275, 2314),
    },
}

OLD_TP6 = {
    "mxfp4-a8-orig": {
        1: (75.75, 5139, 5280),
        2: (61.98, 3658, 3850),
        3: (59.23, 3171, 3212),
        6: (45.88, 2118, 2135),
    },
    "mxfp4-a8-online-mxfp8": {
        1: (82.96, 4906, 5244),
        2: (66.64, 3514, 3878),
        3: (63.82, 3124, 3176),
        6: (50.05, 2105, 2133),
    },
}

# Two-run means already measured on the final v17 image before the selective
# campaign. Raw files live in /root/bench-results/pr94-generalization-20260714.
IMPORTED_V17 = {
    (8, 0, "0", "nvfp4-a16-orig", 2): (4633.0, 4641.5),
    (8, 0, "0", "nvfp4-a16-orig", 4): (3551.5, 3576.0),
    (8, 0, "0", "nvfp4-a16-orig", 8): (2378.5, 2388.0),
    (6, 0, "0", "mxfp4-a8-orig", 2): (3975.5, 3966.5),
    (6, 0, "0", "mxfp4-a8-orig", 3): (3299.0, 3326.5),
    (6, 0, "0", "mxfp4-a8-orig", 6): (2275.0, 2293.0),
}


def fmt(value: float | None, decimals: int = 0) -> str:
    if value is None:
        return "TBD"
    return f"{value:,.{decimals}f}"


def pct(new: float | None, old: float) -> str:
    if new is None:
        return "TBD"
    return f"{new / old - 1:+.1%}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_current(
    root: Path,
) -> dict[tuple[int, int, str, str, int], tuple[float, float]]:
    current = dict(IMPORTED_V17)
    summary = root / "summary.json"
    if not summary.exists():
        return current
    for row in json.loads(summary.read_text()):
        current[(row["tp"], row["mtp"], str(row["f8"]), row["case"], row["dcp"])] = (
            row["prefill_8k"],
            row["prefill_64k"],
        )
    return current


def current_pair(current, tp, mtp, dma, case, dcp, inherited):
    if dcp == 1:
        return inherited[case][dcp]
    return current.get((tp, mtp, str(dma), case, dcp), (None, None))


def render_decision(current) -> str:
    rows = []
    for case in TP8_CASES:
        p64_dcp1 = OLD_TP8_MTP0[case][1][1]
        p64_dcp4 = current_pair(current, 8, 0, 0, case, 4, OLD_TP8_MTP0)[1]
        mean, sd = KLD[case]
        rows.append(
            [
                LABELS[case],
                f"{mean:.5f} +/- {sd:.5f}",
                f"{DECODE_CC1[case][1]:.2f}",
                fmt(p64_dcp1),
                fmt(p64_dcp4),
                pct(p64_dcp4, OLD_TP8_MTP0[case][4][1]),
            ]
        )
    return table(
        [
            "Case",
            "KLD mean +/- sd",
            "DCP1 decode CC1",
            "DCP1 prefill 64k",
            "v17 DCP4 prefill 64k",
            "vs v15 DCP4",
        ],
        rows,
    )


def render_tp8_mtp0(current) -> str:
    parts = []
    for index, name in enumerate(("Prefill 8k", "Prefill 64k")):
        rows = []
        for case in TP8_CASES:
            values = [
                current_pair(current, 8, 0, 0, case, dcp, OLD_TP8_MTP0)[index]
                for dcp in (1, 2, 4, 8)
            ]
            rows.append([LABELS[case], *(fmt(value) for value in values)])
        parts.extend(
            [
                f"#### {name}",
                "",
                table(["Case", "DCP1", "DCP2", "DCP4", "DCP8"], rows),
                "",
            ]
        )
    rows = []
    for case in TP8_CASES:
        changes = []
        for dcp in (2, 4, 8):
            new = current_pair(current, 8, 0, 0, case, dcp, OLD_TP8_MTP0)[1]
            changes.append(pct(new, OLD_TP8_MTP0[case][dcp][1]))
        rows.append([LABELS[case], *changes])
    parts.extend(
        [
            "#### 64k Change Versus v15",
            "",
            table(["Case", "DCP2", "DCP4", "DCP8"], rows),
        ]
    )
    return "\n".join(parts)


def render_tp8_mtp3(current) -> str:
    rows = []
    for case in NVFP4_CASES:
        values = [
            current_pair(current, 8, 3, 0, case, dcp, OLD_TP8_MTP3)
            for dcp in (1, 2, 4, 8)
        ]
        rows.append(
            [LABELS[case], *(f"{fmt(pair[0])} / {fmt(pair[1])}" for pair in values)]
        )
    absolute = table(["Case", "DCP1", "DCP2", "DCP4", "DCP8"], rows)
    delta_rows = []
    for case in NVFP4_CASES:
        changes = []
        for dcp in (2, 4, 8):
            new = current_pair(current, 8, 3, 0, case, dcp, OLD_TP8_MTP3)[1]
            changes.append(pct(new, OLD_TP8_MTP3[case][dcp][1]))
        delta_rows.append([LABELS[case], *changes])
    delta = table(["Case", "DCP2", "DCP4", "DCP8"], delta_rows)
    return f"{absolute}\n\n#### 64k Change Versus v15\n\n{delta}"


def render_dma(current) -> str:
    rows = []
    for case in ("nvfp4-a4-orig", "nvfp4-a4-online-mxfp8"):
        for dma in ("0", "ag", "ring"):
            inherited = (
                {case: OLD_TP8_MTP3[case]}
                if dma == "0"
                else {case: OLD_DMA[(case, dma)]}
            )
            values = [
                current_pair(current, 8, 3, dma, case, dcp, inherited)
                for dcp in (1, 2, 4, 8)
            ]
            rows.append(
                [
                    LABELS[case],
                    dma,
                    *(f"{fmt(pair[0])} / {fmt(pair[1])}" for pair in values),
                ]
            )
    absolute = table(["Case", "f8", "DCP1", "DCP2", "DCP4", "DCP8"], rows)
    delta_rows = []
    for case in ("nvfp4-a4-orig", "nvfp4-a4-online-mxfp8"):
        for dma in ("0", "ag", "ring"):
            changes = []
            for dcp in (2, 4, 8):
                old = (
                    OLD_TP8_MTP3[case][dcp][1]
                    if dma == "0"
                    else OLD_DMA[(case, dma)][dcp][1]
                )
                new = current.get((8, 3, dma, case, dcp), (None, None))[1]
                changes.append(pct(new, old))
            delta_rows.append([LABELS[case], dma, *changes])
    delta = table(["Case", "f8", "DCP2", "DCP4", "DCP8"], delta_rows)
    gain_rows = []
    for case in ("nvfp4-a4-orig", "nvfp4-a4-online-mxfp8"):
        for dma in ("ag", "ring"):
            gains = []
            for dcp in (2, 4, 8):
                base = current.get((8, 3, "0", case, dcp), (None, None))[1]
                value = current.get((8, 3, dma, case, dcp), (None, None))[1]
                gains.append("TBD" if base is None else pct(value, base))
            gain_rows.append([LABELS[case], dma, *gains])
    gains = table(["Case", "f8", "DCP2", "DCP4", "DCP8"], gain_rows)
    return (
        f"{absolute}\n\n#### 64k Change Versus v15\n\n{delta}"
        f"\n\n#### 64k DMA Gain Versus `f8=0` On v17\n\n{gains}"
    )


def render_tp6(current) -> str:
    rows = []
    for case in ("mxfp4-a8-orig", "mxfp4-a8-online-mxfp8"):
        for dcp in (1, 2, 3, 6):
            old_decode, old_8k, old_64k = OLD_TP6[case][dcp]
            if dcp == 1:
                new_8k, new_64k = old_8k, old_64k
            else:
                new_8k, new_64k = current.get((6, 0, "0", case, dcp), (None, None))
            rows.append(
                [
                    LABELS[case],
                    str(dcp),
                    f"{old_decode:.2f}",
                    fmt(new_8k),
                    fmt(new_64k),
                    "inherited" if dcp == 1 else pct(new_64k, old_64k),
                ]
            )
    return table(
        [
            "Case",
            "DCP",
            "Decode CC1",
            "v17 prefill 8k",
            "v17 prefill 64k",
            "64k vs v16",
        ],
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    current = load_current(args.result_root)
    sections = [
        "### Cross-Quant Decision Table\n\n" + render_decision(current),
        "### TP8 MTP0 DCP Prefill\n\n" + render_tp8_mtp0(current),
        "### TP8 MTP3 DCP Prefill\n\nCells are `8k / 64k` tok/s.\n\n"
        + render_tp8_mtp3(current),
        "### TP8 A4 MTP3 FP8 DMA\n\nCells are `8k / 64k` tok/s.\n\n"
        + render_dma(current),
        "### TP6 MTP0\n\nDecode is inherited from v16; PR #94 does not change decode.\n\n"
        + render_tp6(current),
    ]
    output = "\n\n".join(sections) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
