#!/usr/bin/env python3
"""
Broadcom PEX890xx Posted-Write Arbitration Bug — Reproduction Script
ASUS ESC8000A-E13P with 8x RTX PRO 6000 Blackwell

Requirements:
  - ACS disabled on all root ports and switch downstream ports
  - PyTorch with CUDA support
  - Run as root

Usage:
  python3 reproduce_bug.py
"""
import torch
import time
import sys

SIZE = 256 * 1024 * 1024  # 256 MB per transfer
ITERS = 20

def enable_p2p():
    for i in range(8):
        torch.cuda.set_device(i)
        for j in range(8):
            if i != j:
                try:
                    torch.cuda.enable_peer_access(j)
                except:
                    pass

def concurrent_transfer(pairs, mode="write"):
    """
    mode="write": GPU pushes data to remote (posted write TLP)
    mode="read":  GPU pulls data from remote (non-posted read + completion)
    """
    bufs = {}
    streams = {}
    for s, d in pairs:
        if mode == "write":
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                           torch.empty(SIZE//4, device=f'cuda:{d}'))
        else:
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{d}'),
                           torch.empty(SIZE//4, device=f'cuda:{s}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))

    # Warmup
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]):
            bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()

    # Measure
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]):
                bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return SIZE * ITERS * len(pairs) / elapsed / 1e9

def main():
    print("=" * 70)
    print("PEX890xx Posted-Write Arbitration Bug — Reproduction")
    print("=" * 70)
    print()

    ngpu = torch.cuda.device_count()
    print(f"GPUs detected: {ngpu}")
    if ngpu < 8:
        print(f"WARNING: Need 8 GPUs, found {ngpu}. Results may differ.")

    enable_p2p()

    # BUG TRIGGER: 2 GPUs from same VS → 2 different target root ports
    # GPU0 and GPU1 are on VS0 (same switch partition, same uplink)
    # GPU4 is behind root port 80:01.1, GPU6 is behind root port f0:01.1
    collapse_pairs = [(0, 4), (1, 6)]

    # CONTROL: 2 GPUs from DIFFERENT VS → 2 different target root ports
    # GPU0 is on VS0 (uplink 10:01.1), GPU2 is on VS1 (uplink 70:01.1)
    ok_pairs = [(0, 4), (2, 6)]

    print()
    print("--- TEST 1: BUG TRIGGER ---")
    print("GPU0(VS0) → GPU4 + GPU1(VS0) → GPU6")
    print("Same source VS, same uplink, DIFFERENT target root ports")
    w = concurrent_transfer(collapse_pairs, "write")
    r = concurrent_transfer(collapse_pairs, "read")
    print(f"  Posted WRITE:     {w:6.1f} GB/s", end="")
    print(f"  {'← COLLAPSE!' if w < 10 else ''}")
    print(f"  Non-posted READ:  {r:6.1f} GB/s", end="")
    print(f"  {'← NO collapse' if r > 20 else ''}")

    print()
    print("--- TEST 2: CONTROL (no bug) ---")
    print("GPU0(VS0) → GPU4 + GPU2(VS1) → GPU6")
    print("Different source VS, different uplinks")
    w = concurrent_transfer(ok_pairs, "write")
    r = concurrent_transfer(ok_pairs, "read")
    print(f"  Posted WRITE:     {w:6.1f} GB/s")
    print(f"  Non-posted READ:  {r:6.1f} GB/s")

    print()
    print("--- TEST 3: BASELINE ---")
    w = concurrent_transfer([(0, 4)], "write")
    print(f"  Single flow GPU0→GPU4: {w:6.1f} GB/s")

    print()
    print("=" * 70)
    print("RESULT: Bug confirmed if TEST 1 Write shows ~2.7 GB/s")
    print("        while TEST 1 Read shows ~50 GB/s")
    print("        and TEST 2 Write shows ~75+ GB/s")
    print("=" * 70)

    # Return exit code based on whether bug is present
    if w > 20:  # baseline OK
        collapse_w = concurrent_transfer(collapse_pairs, "write")
        if collapse_w < 10:
            print("\n*** BUG PRESENT: Posted-write collapse detected ***")
            sys.exit(1)
    print("\n*** Bug not detected ***")
    sys.exit(0)

if __name__ == "__main__":
    main()
