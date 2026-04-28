"""
Escalating concurrency test on 16-GPU 4-root topology.

Per collapse-report.md, the original 16-GPU collapse:
- COLLAPSE-2: 2 src GPUs on SW1 -> 2 dst roots = 13.5 GB/s (75% drop)
- COLLAPSE-3: 3 src GPUs on SW1 -> 3 dst roots = 12.6 GB/s (78% drop)

Topology: 4 switches, 4 GPU each, each on its own root
  SW1 (root 00, Q0): GPU 0,1,2,3
  SW2 (root 20, Q1): GPU 4,5,6,7
  SW3 (root 40, Q2): GPU 8,9,10,11
  SW4 (root e0, Q3): GPU 12,13,14,15
"""
import torch, time

SIZE = 256 * 1024 * 1024

def concurrent_write(pairs, iters=50, size=SIZE):
    bufs, streams = {}, {}
    for s, d in pairs:
        bufs[(s,d)] = (torch.randn(size//4, device=f'cuda:{s}'),
                       torch.empty(size//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return size * iters * len(pairs) / (time.perf_counter() - t0) / 1e9

def concurrent_read(pairs, iters=50, size=SIZE):
    bufs, streams = {}, {}
    for s, d in pairs:
        bufs[(s,d)] = (torch.randn(size//4, device=f'cuda:{d}'),
                       torch.empty(size//4, device=f'cuda:{s}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return size * iters * len(pairs) / (time.perf_counter() - t0) / 1e9

print("="*100)
print("Escalating concurrency: src GPUs on SW1 -> dsts on multiple remote quadrants")
print("="*100)

tests = [
    # === Single dst root (no trigger) ===
    ("CTRL  1-pair  SW1 -> SW2 (1 dst root Q1)",                      [(0, 4)]),
    ("CTRL  2-pair  SW1 -> SW2 only",                                  [(0, 4), (1, 5)]),
    ("CTRL  4-pair  SW1 -> SW2 only (saturate uplink)",                [(0, 4), (1, 5), (2, 6), (3, 7)]),
    # === 2 dst roots (potential trigger) ===
    ("TRIG  2-pair  SW1 -> SW2 + SW3 [Q1 + Q2]",                       [(0, 4), (1, 8)]),
    ("TRIG  2-pair  SW1 -> SW2 + SW4 [Q1 + Q3]",                       [(0, 4), (1, 12)]),
    ("TRIG  2-pair  SW1 -> SW3 + SW4 [Q2 + Q3]",                       [(0, 8), (1, 12)]),
    # === 3 dst roots ===
    ("TRIG  3-pair  SW1 -> SW2 + SW3 + SW4 [Q1+Q2+Q3]",                [(0, 4), (1, 8), (2, 12)]),
    # === 4-pair, 3 dst roots ===
    ("TRIG  4-pair  SW1 -> 2×SW2 + SW3 + SW4",                         [(0, 4), (1, 5), (2, 8), (3, 12)]),
    ("TRIG  4-pair  SW1 -> SW2 + 2×SW3 + SW4",                         [(0, 4), (1, 8), (2, 9), (3, 12)]),
    ("TRIG  4-pair  SW1 -> SW2 + SW3 + 2×SW4",                         [(0, 4), (1, 8), (2, 12), (3, 13)]),
    # === 4-pair, all to different dsts in 3 different quadrants ===
    ("TRIG  4-pair  SW1 -> 2×Q1 + Q2 + Q3",                            [(0, 4), (1, 5), (2, 8), (3, 12)]),
]

print(f"\n{'Pattern':<60s}  {'WRITE':>7s}  {'READ':>7s}  {'W/R':>5s}  Per-pair  Verdict")
print("-"*120)
for label, pairs in tests:
    w = concurrent_write(pairs)
    r = concurrent_read(pairs)
    n = len(pairs)
    ratio = w/r if r > 0 else 0
    if w < r * 0.4:
        flag = " ⚠⚠ COLLAPSE!"
    elif w < r * 0.7:
        flag = " ⚠ partial collapse"
    elif w < r * 0.9:
        flag = " minor dip"
    else:
        flag = " OK"
    print(f"{label:<60s}  {w:6.1f}    {r:6.1f}   {ratio:.2f}x  {w/n:5.1f}/{r/n:5.1f}/p {flag}")
