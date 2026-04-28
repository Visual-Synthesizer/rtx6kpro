"""
Comprehensive collapse isolation test on 4-root 16-GPU topology.

Step 1: Verify chip mapping via 2-pair single src->single dst BW signature
Step 2: Run all 12 "1 src -> 2 dst switches" patterns (vary src + dst pair)
Step 3: Specifically all 4 patterns with 3 dst roots (max collapse trigger)
Step 4: 4-pair with 3 dst roots, multiple variants
Step 5: Sustained stress (long iter) on collapse trigger
"""
import torch, time

SIZE = 256 * 1024 * 1024

# 16-GPU 4-switch 4-root topology
# SW1 (root 00 / Q0): GPU 0,1,2,3
# SW2 (root 20 / Q1): GPU 4,5,6,7
# SW3 (root 40 / Q2): GPU 8,9,10,11
# SW4 (root e0 / Q3): GPU 12,13,14,15

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
print("STEP 1: 2-pair single src→single dst BW signature (chip mapping)")
print("="*100)
print(f"{'Src→Dst':<15s} {'WRITE':>8s} {'Comment':>30s}")
sw_gpus = {1:[0,1,2,3], 2:[4,5,6,7], 3:[8,9,10,11], 4:[12,13,14,15]}
sw_root = {1:"Q0", 2:"Q1", 3:"Q2", 4:"Q3"}
for src in range(1, 5):
    for dst in range(1, 5):
        if src == dst: continue
        sg, dg = sw_gpus[src], sw_gpus[dst]
        bw = concurrent_write([(sg[0], dg[0]), (sg[1], dg[1])])
        print(f"  SW{src}({sw_root[src]})→SW{dst}({sw_root[dst]}):  {bw:6.1f} GB/s")

print()
print("="*100)
print("STEP 2: All 1 src → 2 dst SWITCHES patterns (12 combos)")
print("="*100)
print(f"{'Pattern':<35s} {'WRITE':>8s} {'READ':>8s} {'W/R':>5s}  Quadrants involved")
results_2dst = []
for src in range(1, 5):
    others = [s for s in range(1, 5) if s != src]
    for i in range(len(others)):
        for j in range(i+1, len(others)):
            d1, d2 = others[i], others[j]
            sg = sw_gpus[src]
            d1g = sw_gpus[d1][0]
            d2g = sw_gpus[d2][0]
            pairs = [(sg[0], d1g), (sg[1], d2g)]
            w = concurrent_write(pairs)
            r = concurrent_read(pairs)
            ratio = w/r if r > 0 else 0
            quads = f"src={sw_root[src]}, dst={{{sw_root[d1]},{sw_root[d2]}}}"
            results_2dst.append((src, d1, d2, w, r))
            print(f"  SW{src}→SW{d1}+SW{d2:<14d}     {w:6.1f}    {r:6.1f}   {ratio:.2f}x  {quads}")

print()
print("="*100)
print("STEP 3: 3 dst switches (3 remote quadrants) — strongest trigger")
print("="*100)
for src in range(1, 5):
    others = [s for s in range(1, 5) if s != src]
    sg = sw_gpus[src]
    pairs = [(sg[0], sw_gpus[others[0]][0]),
             (sg[1], sw_gpus[others[1]][0]),
             (sg[2], sw_gpus[others[2]][0])]
    w = concurrent_write(pairs)
    r = concurrent_read(pairs)
    ratio = w/r if r > 0 else 0
    quads = ",".join(sw_root[d] for d in others)
    flag = " ⚠⚠ COLLAPSE!" if w < r * 0.4 else (" ⚠ PARTIAL" if w < r * 0.7 else "")
    print(f"  SW{src} → SW{others[0]}+SW{others[1]}+SW{others[2]:<10d}  {w:6.1f}   {r:6.1f}   {ratio:.2f}x  src={sw_root[src]} → {{{quads}}}{flag}")

print()
print("="*100)
print("STEP 4: 4-pair from SW1 to 4 dst GPUs (one per remote quadrant + extra)")
print("="*100)
# Pattern: 4 src GPUs on SW1 fan out to 4 different dst switches
# But we have only 3 remote quadrants, so 4 src can't all hit different remote quadrants
# Variants:
for label, pairs in [
    ("4-pair SW1→SW2+SW3+SW4 (3 dsts) + 1 to SW2 again",  [(0, 4), (1, 8), (2, 12), (3, 5)]),
    ("4-pair SW1→2*SW2+SW3+SW4", [(0, 4), (1, 5), (2, 8), (3, 12)]),
    ("4-pair SW1→SW2+2*SW3+SW4", [(0, 4), (1, 8), (2, 9), (3, 12)]),
    ("4-pair SW1→SW2+SW3+2*SW4", [(0, 4), (1, 8), (2, 12), (3, 13)]),
    ("4-pair SW1→4*SW2 (uplink-saturated control)", [(0, 4), (1, 5), (2, 6), (3, 7)]),
]:
    w = concurrent_write(pairs)
    r = concurrent_read(pairs)
    ratio = w/r if r > 0 else 0
    flag = " ⚠⚠ COLLAPSE!" if w < r * 0.4 else (" ⚠ PARTIAL" if w < r * 0.7 else "")
    print(f"  {label:<55s}  {w:6.1f}   {r:6.1f}   {ratio:.2f}x{flag}")

print()
print("="*100)
print("STEP 5: SUSTAINED stress — same patterns with 200 iter (5x longer)")
print("="*100)
for label, pairs in [
    ("3-pair SW1→Q1+Q2+Q3 (200 iter)", [(0, 4), (1, 8), (2, 12)]),
    ("4-pair SW1→2Q1+Q2+Q3 (200 iter)", [(0, 4), (1, 5), (2, 8), (3, 12)]),
]:
    w = concurrent_write(pairs, iters=200)
    r = concurrent_read(pairs, iters=200)
    ratio = w/r if r > 0 else 0
    flag = " ⚠⚠ COLLAPSE!" if w < r * 0.4 else (" ⚠ PARTIAL" if w < r * 0.7 else "")
    print(f"  {label:<55s}  {w:6.1f}   {r:6.1f}   {ratio:.2f}x{flag}")
