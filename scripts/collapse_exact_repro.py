"""
Exact replica of the collapse trigger patterns from
`hardware/wrx90-cpayne-16gpu-4switch.md` Posted-Write Collapse section.

Original 16-GPU 4-switch topology had:
  SW1 -> root 00 (Q0)
  SW2 -> root 40 (Q2)
  SW3 -> root e0:01.1 (Q3)
  SW4 -> root e0:03.1 (Q3)
That gave 3 distinct root complexes.

Current topology (2-root):
  SW1 -> root 00 (Q0)  -- shares root with SW2
  SW2 -> root 00 (Q0)  -- shares root with SW1
  SW3 -> root e0 (Q3)  -- shares root with SW4
  SW4 -> root e0 (Q3)
That gives only 2 distinct root complexes.

If the collapse trigger needs 3+ root complexes, current topology
cannot reproduce. Run the EXACT same patterns and compare.
"""
import torch, time

SIZE = 256 * 1024 * 1024
ITERS = 50

# Same GPU index → switch mapping as the original page
# SW1: 0-3, SW2: 4-7, SW3: 8-11, SW4: 12-15

def concurrent_write(pairs, iters=ITERS, size=SIZE):
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

def concurrent_read(pairs, iters=ITERS, size=SIZE):
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

# Exact same patterns as in the wrx90-cpayne-16gpu-4switch.md collapse table
print("="*100)
print("EXACT REPLICA of patterns from hardware/wrx90-cpayne-16gpu-4switch.md")
print("Current topology: 4 switches but only 2 root complexes")
print("="*100)

tests = [
    # SW1 → SW3 + SW4 (both Q3 in current = same dst root) -- "OK" in original
    ("SW1->SW3+SW4 (current: both Q3, same root)", [(0, 8), (1, 12)],
     "ORIG: 53.4/57.5 OK"),
    # SW2 → SW3 + SW4 (both Q3) -- "OK" in original (dst share root e0)
    ("SW2->SW3+SW4 (current: both Q3, same root)", [(4, 8), (5, 12)],
     "ORIG: 55.3/54.1 OK"),
    # The KEY collapse pattern: SW1 → SW2 + SW3
    # ORIGINAL: SW2(root 40) + SW3(root e0) = 2 cross-roots, NEITHER is src root → COLLAPSE
    # CURRENT:  SW2(root 00, SAME as src!) + SW3(root e0) = 1 cross-root only
    ("SW1->SW2+SW3 (orig: 2 diff roots != src // current: 1 cross-root only)",
     [(0, 4), (1, 8)],
     "ORIG: 11.6/52.9 COLLAPSE"),
    # SW1 → SW2 + SW4
    ("SW1->SW2+SW4 (orig: 2 diff roots // current: 1 cross-root only)",
     [(0, 4), (1, 12)],
     "ORIG: 13.3/53.8 COLLAPSE"),
    # SW1 → SW2 only (single dst root)
    ("SW1->SW2 only (1 dst root)", [(0, 4), (1, 5)],
     "ORIG: 54.1/54.7 OK"),
    # SW1 → SW3 only
    ("SW1->SW3 only (1 dst root)", [(0, 8), (1, 9)],
     "ORIG: 56.0/53.0 OK"),
    # SW1 → SW2 + SW3 → SW4 (different src switches)
    ("SW1->SW2 + SW3->SW4 (different src switches)", [(0, 4), (8, 12)],
     "ORIG: 106.8/105.6 OK"),
]

print(f"\n{'Pattern':<70s}  {'WRITE':>7s}  {'READ':>7s}  {'W/R':>5s}  Original notes")
print("-"*120)
for label, pairs, orig_note in tests:
    w = concurrent_write(pairs)
    r = concurrent_read(pairs)
    ratio = w/r if r > 0 else 0
    flag = ""
    if w < r * 0.5:
        flag = " ⚠ COLLAPSE!"
    elif w < r * 0.9:
        flag = " ⚠ partial"
    print(f"{label:<70s}  {w:6.1f}    {r:6.1f}   {ratio:.2f}x   {orig_note}{flag}")
