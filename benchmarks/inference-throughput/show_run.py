#!/usr/bin/env python3
"""Render one inference-throughput benchmark JSON as tables.

Usage:
    show_run.py <run.json> [more.json ...]
    show_run.py            # defaults to the newest JSON under runs/

Prints three tables per file:
  1. throughput grid      - aggregate tok/s, context (rows) x concurrency (cols)
  2. latency detail       - per (concurrency, context) row from `results`
  3. prefill              - ttft + tok/s per context length
"""

import json
import sys
from pathlib import Path

RUNS_DIR = Path(__file__).parent / "runs"


def default_run():
    files = sorted(RUNS_DIR.glob("*/*.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        sys.exit(f"no JSON files found under {RUNS_DIR}")
    return files[-1]


def human_tokens(n):
    return f"{n // 1024}k" if n >= 1024 else str(n)


def fmt(v, width, precision=1):
    if v is None:
        return "-".rjust(width)
    return f"{v:,.{precision}f}".rjust(width)


def render_run(path):
    with open(path) as fh:
        data = json.load(fh)

    meta = data.get("metadata", {})
    print(f"=== {path.name} ===")
    print(f"model: {meta.get('model', '?')}  server: {meta.get('server', '?')}  "
          f"timestamp: {meta.get('timestamp', '?')}")

    results = data.get("results", [])
    if not results:
        print("  (no results)\n")
        return

    # --- table 1: aggregate tok/s grid, context x concurrency ------------
    contexts = sorted({r["context_tokens"] for r in results})
    concurrencies = sorted({r["concurrency"] for r in results})
    grid = {(r["context_tokens"], r["concurrency"]): r.get("aggregate_tps")
            for r in results}

    col_w = 10
    header = "ctx\\conc".ljust(9) + "".join(f"{c}".rjust(col_w) for c in concurrencies)
    print(f"\nThroughput (aggregate tok/s)\n{header}")
    for ctx in contexts:
        row = human_tokens(ctx).ljust(9)
        row += "".join(fmt(grid.get((ctx, c)), col_w) for c in concurrencies)
        print(row)

    # --- table 2: per-request latency detail -------------------------------
    print("\nLatency detail")
    cols = [("conc", 6, lambda r: f"{r['concurrency']}"),
            ("ctx", 7, lambda r: human_tokens(r["context_tokens"])),
            ("tok/s(req)", 11, lambda r: fmt(r.get("per_request_avg_tps"), 0)),
            ("ttft_avg", 9, lambda r: fmt(r.get("ttft_avg"), 0, 2)),
            ("ttft_p50", 9, lambda r: fmt(r.get("ttft_p50"), 0, 2)),
            ("ttft_p99", 9, lambda r: fmt(r.get("ttft_p99"), 0, 2)),
            ("wall_s", 8, lambda r: fmt(r.get("wall_time"), 0)),
            ("done", 6, lambda r: str(r.get("num_completed", "-"))),
            ("err", 5, lambda r: str(r.get("num_errors", "-")))]
    print("".join(name.rjust(w) for name, w, _ in cols))
    for r in sorted(results, key=lambda r: (r["context_tokens"], r["concurrency"])):
        print("".join(get(r).rjust(w) for _, w, get in cols))

    # --- table 3: prefill ----------------------------------------------------
    prefill = data.get("prefill", {})
    if prefill:
        print("\nPrefill")
        print(f"{'ctx'.rjust(7)}{'ttft_s'.rjust(9)}{'tok/s'.rjust(12)}")
        for ctx in sorted(prefill, key=lambda k: int(k)):
            p = prefill[ctx]
            print(f"{human_tokens(int(ctx)).rjust(7)}"
                  f"{fmt(p.get('ttft_seconds'), 9, 2)}"
                  f"{fmt(p.get('tok_per_sec'), 12)}")
    print()


def main():
    paths = [Path(a) for a in sys.argv[1:]] or [default_run()]
    for p in paths:
        render_run(p)


if __name__ == "__main__":
    main()
