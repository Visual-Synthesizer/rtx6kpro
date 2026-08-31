#!/usr/bin/env python3
"""Assign DS4 sweep cases to stable GPU waves and ports.

The assignment depends on the complete backend/mode matrix, not on which
results already exist. A resumed sweep therefore returns every pending case
to the same GPU UUIDs that populated its device-scoped compile cache.
"""

from __future__ import annotations

import argparse
import itertools


def split_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_layout(
    *,
    tp: int,
    backends: tuple[str, ...],
    modes: tuple[str, ...],
    gpu_groups: tuple[str, ...],
    port_base: int,
) -> list[tuple[int, str, str, int, int, int, str, int]]:
    if not backends or not modes or not gpu_groups:
        raise ValueError("backends, modes, and GPU groups must be non-empty")
    if len(set(gpu_groups)) != len(gpu_groups):
        raise ValueError("GPU groups must be unique within a TP layout")

    layout = []
    for ordinal, (backend, mode) in enumerate(itertools.product(backends, modes)):
        wave, slot = divmod(ordinal, len(gpu_groups))
        port = port_base + tp * 100 + wave * 20 + slot
        layout.append((tp, backend, mode, ordinal, wave, slot, gpu_groups[slot], port))
    return layout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--backends", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--gpu-groups", required=True)
    parser.add_argument("--port-base", type=int, required=True)
    args = parser.parse_args()

    layout = build_layout(
        tp=args.tp,
        backends=split_csv(args.backends),
        modes=split_csv(args.modes),
        gpu_groups=tuple(args.gpu_groups.split()),
        port_base=args.port_base,
    )
    for row in layout:
        print("\t".join(map(str, row)))


if __name__ == "__main__":
    main()
