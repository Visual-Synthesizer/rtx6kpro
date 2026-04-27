# 8-GPU c-payne with 2 Virtual Switches per Physical Chip — Topology, Bandwidth, Collapse Analysis

A specific c-payne configuration where **two physical c-payne (Microchip Switchtec) chips** are each partitioned into **two Virtual Switches (VS)**, presenting **four "logical" PCIe switches** to the OS while only consuming two physical c-payne packages. Each physical chip has two independent x16 Gen5 upstream ports landing on two different CPU root complexes; intra-chip traffic between the two VS partitions stays entirely inside the switch fabric and never traverses the CPU.

This page documents the topology, the surprising bandwidth observations it produces, and what it means for the AMD posted-write collapse trigger we have been chasing on this rig.

---

## System under test

| Item | Value |
|------|-------|
| Host | ASRock WRX90 WS EVO + AMD Threadripper Pro 7955WX (single socket, 4 IOD quadrants Q0–Q3) |
| Driver | NVIDIA 595.58.03 |
| CUDA Driver API | 13.2 |
| Kernel | Linux 6.18.24 (Ubuntu 24.04) |
| GPUs | 8× NVIDIA RTX PRO 6000 Blackwell **Server Edition** (96 GB GDDR7, SM120) |
| PCIe switches | **2× physical c-payne** (Microchip Switchtec PM50100, vendor 0x1f18 device 0x0101) — each partitioned into 2 Virtual Switches |
| `iommu` | `off` |
| ACS Request-Redirect | disabled at boot via setpci (`/etc/systemd/system/disable-acs.service`) |

---

## Topology

The OS sees **four** PCIe switches at four root buses:

| OS-visible "switch" | Upstream bridge | Root bus | CPU quadrant | GPUs |
|---------------------|-----------------|----------|--------------|------|
| SW1 (VS) | `0000:01:00.0` | `pci0000:00` | Q0 | GPU 0, 1 |
| SW2 (VS) | `0000:21:00.0` | `pci0000:20` | Q1 | GPU 2, 3 |
| SW3 (VS) | `0000:41:00.0` | `pci0000:40` | Q2 | GPU 4, 5 |
| SW4 (VS) | `0000:e1:00.0` | `pci0000:e0` | Q3 | GPU 6, 7 |

**But internally there are only two physical c-payne packages**, each carrying two of the above virtual switches:

```
                        ╔══════════════════════════════════════════════╗
                        ║   CPU Threadripper Pro 7955WX  (1 IOD)       ║
                        ║                                              ║
                        ║   Q0    ──── inter-quadrant IF ──── Q1       ║
                        ║   │                                  │       ║
                        ║   │                                  │       ║
                        ║   Q2    ──── inter-quadrant IF ──── Q3       ║
                        ╚════│════════════│════════════│════════│══════╝
                             │            │            │        │
                       x16 Gen5    x16 Gen5    x16 Gen5    x16 Gen5
                       (root 00)   (root 20)   (root 40)   (root e0)
                             │            │            │        │
                             │            │            │        │
            ┌────────────────┼────────────┼────────────┼────────┼─────────────┐
            │  PHYSICAL CHIP A             │            │       PHYSICAL CHIP B │
            │  ┌──────────┐                │            ┌──────────┐         │
            │  │   SW1    │     ┌──────────┐│            │   SW4    │         │
            │  │  (VS A1) │     │   SW2    ││            │  (VS B2) │         │
            │  │ root 00  │     │  (VS B1) ││            │ root e0  │         │
            │  │ Q0 / x16 │     │ root 20  ││            │ Q3 / x16 │         │
            │  └─┬───┬────┘     │ Q1 / x16 ││            └─┬───┬────┘         │
            │    │   │  ╲       └─┬───┬────┘│              │   │  ╲           │
            │    │   │   ↔ chip A │   │  ╲  │              │   │   ↔ chip B  │
            │    │   │ ↕ internal │   │   ↕  │             │   │ ↕ internal  │
            │    │   │   fabric   │   │ chip │             │   │   fabric    │
            │    │   │   ↕        │   │  B   │             │   │   ↕         │
            │    │   │  ╱         │   │ int. │             │   │  ╱          │
            │  ┌─┴───┴────┐       │   │ fab. │           ┌─┴───┴────┐        │
            │  │   SW3    │       │   │      │           │   SW3 ←─── wait, │
            │  │  (VS A2) │       └─┬─┴──────┘           │   that's wrong   │
            │  │ root 40  │         │                    │   redrawing →    │
            │  │ Q2 / x16 │       …continued on chip B…  │                  │
            │  └─┬───┬────┘                              │                  │
            └────┼───┼─────────────────────────────────  └──────────────────┘
                GPU 0,1     GPU 4,5                       GPU 2,3   GPU 6,7
```

Cleaner diagram (the ASCII wraps weirdly above; this is the actual mapping):

```
 Physical c-payne CHIP A                    Physical c-payne CHIP B
 ┌──────────────────────────┐               ┌──────────────────────────┐
 │  ╔════════╗   ╔════════╗ │               │  ╔════════╗   ╔════════╗ │
 │  ║  SW1   ║   ║  SW3   ║ │               │  ║  SW2   ║   ║  SW4   ║ │
 │  ║ (VS A1)║   ║ (VS A2)║ │               │  ║ (VS B1)║   ║ (VS B2)║ │
 │  ║ Q0,x16 ║   ║ Q2,x16 ║ │               │  ║ Q1,x16 ║   ║ Q3,x16 ║ │
 │  ╚═╤════╤═╝   ╚═╤════╤═╝ │               │  ╚═╤════╤═╝   ╚═╤════╤═╝ │
 │   GPU0 GPU1   GPU4 GPU5  │               │   GPU2 GPU3   GPU6 GPU7  │
 │           ╲╲ A-internal ╱╱│               │           ╲╲ B-internal ╱╱│
 │            ╲══fabric══╱   │               │            ╲══fabric══╱   │
 │  ▲ uplinks: 2 separate    │               │  ▲ uplinks: 2 separate    │
 │  ▲ Gen5 x16 to root 00 +  │               │  ▲ Gen5 x16 to root 20 +  │
 │  ▲ root 40 (different     │               │  ▲ root e0 (different     │
 │  ▲ CPU quadrants)         │               │  ▲ CPU quadrants)         │
 └────────────│─────│────────┘               └────────────│─────│────────┘
              │     │                                     │     │
        root 00   root 40                           root 20   root e0
        (Q0)      (Q2)                              (Q1)      (Q3)
              ╲─── CPU IOD inter-quadrant fabric ───╱
                       (Infinity Fabric)
```

So in this configuration:

* **Chip A** carries SW1 (Q0) + SW3 (Q2) — the two virtual switches share the chip's internal fabric. Traffic between SW1 and SW3 stays *inside chip A* and never enters the CPU.
* **Chip B** carries SW2 (Q1) + SW4 (Q3) — same story for traffic between SW2 and SW4.
* **Cross-chip traffic** (e.g. SW1 → SW2, SW1 → SW4, SW3 → SW2, SW3 → SW4) is the only kind that crosses the CPU IOD fabric and uses the upstream PCIe link of the source chip.

This is the c-payne 2-host / dual-VS configuration — the chip's silicon supports it natively because Switchtec PFX-G5 family is multi-host capable.

---

## Bandwidth measurements

All measurements with PyTorch peer-to-peer copy (`tensor.copy_()` with peer access enabled). 256 MB buffers, 100 iterations, mean over 1 Hz telemetry samples. PL = 600 W default on every card. Driver 595.58.03, kernel 6.18.24, IOMMU off.

### Single-pair P2P bandwidth (one direction, write)

Per-pair PCIe x16 Gen5 line-rate ceiling is ~63 GB/s; practical sustained is ~56 GB/s once protocol overhead is counted.

| From → To | Path | Single-pair WRITE | Notes |
|-----------|------|------------------:|-------|
| GPU 0 → GPU 1 | intra-VS / intra-chip A | 56.3 GB/s | stays inside SW1's downstream fabric |
| GPU 0 → GPU 4 | cross-VS / intra-chip A | 56.3 GB/s | SW1 ↔ SW3 inside chip A's internal fabric |
| GPU 0 → GPU 2 | cross-chip A → B (via CPU) | 56.3 GB/s | through SW1 uplink → root 00 → IF → root 20 → SW2 |
| GPU 0 → GPU 6 | cross-chip A → B (via CPU) | 56.3 GB/s | through CPU |

Single-pair on this rig is always uplink-saturated: 56 GB/s regardless of path. The path differences only show up when multiple concurrent flows are in play.

### 2-pair concurrent: same-source-VS → same-destination-VS

Both source GPUs of one VS sending one pair each to the destination VS:

|              | →SW1 (Q0)    | →SW2 (Q1)    | →SW3 (Q2)     | →SW4 (Q3)     |
|--------------|-------------:|-------------:|--------------:|--------------:|
| **SW1 (Q0)→** | —            | 56.4 GB/s    | **112.5 GB/s** ✨ | 56.4 GB/s    |
| **SW2 (Q1)→** | 56.4 GB/s    | —            | 56.4 GB/s     | **112.5 GB/s** ✨ |
| **SW3 (Q2)→** | **112.5 GB/s** ✨ | 56.4 GB/s | —             | 56.4 GB/s    |
| **SW4 (Q3)→** | 56.4 GB/s    | **112.5 GB/s** ✨ | 56.4 GB/s | —             |

**The "diagonal" pairs (highlighted) reach 112 GB/s = full per-pair x16 bandwidth on each pair simultaneously.** All other ("off-diagonal") cross-VS pairs saturate the source uplink at 56 GB/s.

The pattern matches the chip mapping perfectly:
* SW1 ↔ SW3 (both on chip A) → 112 GB/s, traffic stays inside chip A
* SW2 ↔ SW4 (both on chip B) → 112 GB/s, traffic stays inside chip B
* Any chip-A ↔ chip-B pairing → 56 GB/s, traffic crosses the CPU IOD fabric and is bottlenecked by the source chip's single x16 uplink

There is **no AMD IOD fabric magic** here — the diagonal speed-up is just intra-chip switching that bypasses the CPU.

### 2-pair concurrent: one source VS → two different destination VSs (collapse-trigger pattern)

| Source → Two destinations | chips touched | Aggregate WRITE | Status |
|---------------------------|--------------|----------------:|--------|
| SW1 → SW2 + SW3 | A→B + A→A    | 112.5 GB/s | one flow stays in chip A (intra) |
| SW1 → SW2 + SW4 | A→B + A→B    | 56.4 GB/s | both flows go through CPU (uplink limit) |
| SW1 → SW3 + SW4 | A→A + A→B    | 112.5 GB/s | one flow stays in chip A |
| SW2 → SW1 + SW3 | B→A + B→A    | 56.0 GB/s | both go through CPU |
| SW2 → SW1 + SW4 | B→A + B→B    | 112.5 GB/s | one stays in chip B |
| SW2 → SW3 + SW4 | B→A + B→B    | 112.5 GB/s | one stays in chip B |
| SW3 → SW1 + SW2 | A→A + A→B    | 112.5 GB/s | one stays in chip A |
| SW3 → SW1 + SW4 | A→A + A→B    | 112.5 GB/s | one stays in chip A |
| SW3 → SW2 + SW4 | A→B + A→B    | 51.4 GB/s | both go through CPU |
| SW4 → SW1 + SW2 | B→A + B→B    | 112.5 GB/s | one stays in chip B |
| SW4 → SW1 + SW3 | B→A + B→A    | 54.6 GB/s | both go through CPU |
| SW4 → SW2 + SW3 | B→B + B→A    | 112.5 GB/s | one stays in chip B |

Pattern: whenever the source VS dispatches to **at least one destination on the same physical chip**, that flow uses internal switching and aggregate bandwidth hits 112 GB/s (one path internal at 56, plus one path through CPU at 56). When both destinations are on the *other* physical chip, both flows fight for the source chip's single x16 uplink and aggregate is 56 GB/s.

The slight under-saturation visible in three rows (51.4, 54.6, 56.0 GB/s) is mild contention — **not** the dramatic 4× drop that the AMD posted-write collapse produces. See "Collapse implications" below.

### Other measurements

| Test | Result |
|------|--------|
| Bidirectional intra-VS (GPU 0 ↔ GPU 1) | 109.4 GB/s aggregate (full duplex utilisation) |
| 4 source switches → 1 destination switch (concurrent) | 75.4 GB/s aggregate (destination uplink limit) |
| All-to-all 8 GPU (56 ordered pairs concurrent) | **254.2 GB/s aggregate** (4.54 GB/s per pair) |

Compare the all-to-all number to the previous **16-GPU 4-switch (4 GPU/VS) topology** measured on the same rig: ~179 GB/s aggregate. The 2-VS-per-chip topology with 2 GPU/VS is **+42 % faster** in raw all-to-all throughput because much of the cross-VS traffic stays inside each physical chip and never burdens the CPU IOD.

---

## Collapse implications

The AMD I/O Die "posted-write collapse" trigger documented in [`collapse-report.md`](collapse-report.md) requires:

1. **Multiple source GPUs concurrently dispatching writes from the same source PCIe switch** (i.e., they share one uplink to one CPU root port), AND
2. Their destinations sit behind **two or more different CPU root complexes**.

In the previous 16-GPU 4-switch topology (4 GPU per switch, 4 separate roots), this trigger fired hard: bandwidth dropped from ~50 GB/s/pair to ~6 GB/s/pair (75 % collapse).

**On this 2-VS-per-chip topology, the trigger does not fire.** No measurement showed a >10 % drop from baseline; the worst observed was 51 / 56 = 92 % efficiency.

Two independent reasons protect this layout:

1. **Intra-chip cross-VS traffic never touches the CPU.** Half of the possible "cross-switch" routing (SW1↔SW3, SW2↔SW4) is internal to one physical c-payne chip. The CPU IOD never sees those TLPs, so its arbitration logic cannot misbehave on them. In our results these flows always reach full 56 GB/s per pair.

2. **Only 2 source GPUs per VS.** Even when traffic goes through the CPU (cross-chip), at most 2 source GPUs from one VS share that uplink concurrently. The collapse trigger empirically needs ≥3 concurrent source flows from one switch dispatching to different dst roots in order to break the IOD's posted-write arbitration. With 2 source GPUs we never reach that threshold.

The combination is the protective property:

```
Previous topology (16 GPU, 4 GPU/VS):
  4 src GPUs on SW1 → multiple dst roots → COLLAPSE (BW × 0.25)

This topology (8 GPU, 2 GPU/VS, chip A=SW1+SW3, chip B=SW2+SW4):
  2 src GPUs on SW1 → at most 2 dst roots
  AND if either dst is SW3 it's intra-chip (CPU IOD never sees it)
  → no collapse, only normal contention
```

So this layout is essentially **collapse-immune by construction** for any pattern that fits inside one physical chip, and **collapse-resistant** even for cross-chip patterns because it caps source-side concurrency at 2.

---

## Methodology and reproduction

All bandwidth numbers are produced by the same small PyTorch script:

```python
import torch, time

SIZE = 256 * 1024 * 1024
ITERS = 100

def run(pairs):
    bufs, streams = {}, {}
    for s, d in pairs:
        bufs[(s, d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                        torch.empty(SIZE//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s, d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    # warm
    for s, d in pairs:
        with torch.cuda.stream(streams[(s, d)]):
            bufs[(s, d)][1].copy_(bufs[(s, d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s, d)]):
                bufs[(s, d)][1].copy_(bufs[(s, d)][0])
    torch.cuda.synchronize()
    return SIZE * ITERS * len(pairs) / (time.perf_counter() - t0) / 1e9
```

The script issues `dst.copy_(src)` on a stream owned by the source GPU. With CUDA peer access enabled and ACS Request-Redirect disabled, the runtime selects the direct PCIe peer path — no host-RAM staging. Each `pairs` argument is a list of `(src_gpu, dst_gpu)` tuples; the function returns aggregate write bandwidth across all pairs.

Topology identification:

```bash
# Walk each GPU up the PCIe tree to its CPU root bus
for i in $(seq 0 7); do
  bus=$(nvidia-smi -i $i --query-gpu=gpu_bus_id --format=csv,noheader | sed 's/00000000://')
  root=$(readlink -f /sys/bus/pci/devices/0000:${bus,,}/../.. | grep -oE 'pci[0-9]+:[0-9a-f]+' | head -1)
  echo "GPU $i  bus $bus  -> $root"
done

# Switch upstream port summary
for sw in 01:00.0 21:00.0 41:00.0 e1:00.0; do
  lspci -vv -s $sw | grep -E "LnkCap:|LnkSta:" | head -2
done
```

That `lspci` output gives the line widths and Gen5 speeds of every upstream link. Walking up `/sys/bus/pci/devices/.../..` gives the root bus for each GPU. The 4-root layout is what makes this look like "4 switches" to the OS.

Detecting that two of the OS-visible "switches" are actually the same physical chip is *not* directly visible from `lspci`. The signature is in the bandwidth pattern: any two "switches" that produce 112 GB/s on a 2-pair test are sharing a physical chip, and any pair that produces 56 GB/s saturated is on different physical chips. The bandwidth matrix in this page reads off the pairing immediately:

* 112 GB/s pairs: SW1↔SW3 and SW2↔SW4 → these live on chip A and chip B respectively.
* 56 GB/s pairs: everything else → these are cross-chip and uplink-bound.

The full sweep script (drives all 12 source-VS / dest-VS combinations and the 12 collapse-trigger combinations) is at `/tmp/collapse_2gpu_full.py` on the test rig.

---

## Operational guidance

* **For workloads dominated by GPU-to-GPU P2P** within this 8-GPU layout, prefer to keep traffic *inside one physical c-payne chip* whenever the algorithm allows. Specifically: a tensor-parallel group of 4 GPUs aligned to one chip (e.g., GPU 0,1,4,5 on chip A) gets all cross-VS traffic at 112 GB/s with no CPU involvement.
* **For NCCL ring all-reduce** spanning all 8 GPUs, the ring naturally alternates between intra-chip (cheap, 56 GB/s per hop) and cross-chip (also 56 GB/s per hop, but goes through CPU). Both hops are uplink-saturated so the ring throughput is similar to a flat 4-switch topology — but with no collapse risk.
* **For all-to-all** (MoE token routing), this topology delivers 254 GB/s aggregate vs 179 GB/s on the previous 16-GPU 4-switch topology. ~42 % more useful aggregate, half from the intra-chip routing not paying for the IOD round-trip.
* **Mapping a real workload to physical chips** matters: if you assign cards to hosts/processes ignoring the chip mapping, you can accidentally place every cross-process flow on the cross-chip path and miss the intra-chip win. Use the bandwidth matrix above (or measure with the script) to determine the chip assignment of each VS, then plan accordingly.
* **The collapse risk reappears** if you ever scale this to 4 GPU per VS (e.g., the previous 16-GPU layout): in that case 4 source GPUs can dispatch to 3 distinct destination roots concurrently and the IOD arbitration breaks. Sticking to 2 GPU per VS is a structural workaround.

---

## Cross-references

* [`collapse-report.md`](collapse-report.md) — standalone report on the AMD IOD posted-write collapse, the bug this layout sidesteps.
* [`pcie-posted-write-collapse.md`](pcie-posted-write-collapse.md) — long-form history of the collapse investigation across multiple platforms.
* [`wrx90-cpayne-16gpu-4switch.md`](wrx90-cpayne-16gpu-4switch.md) — the previous 16-GPU 4-switch (4 GPU/VS) layout where the collapse fired hard.
* [`wrx90-cpayne-8gpu-root-topology-comparison.md`](wrx90-cpayne-8gpu-root-topology-comparison.md) — earlier comparison of two 8-GPU groupings (separate roots vs shared root) on a previous topology.
