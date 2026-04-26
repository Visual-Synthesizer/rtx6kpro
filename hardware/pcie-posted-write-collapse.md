# AMD CPU Posted-Write Collapse on Multi-GPU / Multi-Switch Topologies

A reproducible bandwidth collapse (~4× to ~15×) that affects **posted PCIe writes** (but not reads) on AMD EPYC (Turin) and AMD Threadripper PRO (Zen 4) platforms when multiple GPUs on one PCIe switch simultaneously write to destinations behind **two or more different CPU root complexes**.

This page consolidates findings from multiple test platforms. For per-platform detail see:
- [ASUS ESC8000A-E13P + Broadcom Switches](asus-esc8000a-e13p-broadcom-switches.md)
- [WRX90 + 2× c-payne (flat)](wrx90-cpayne-2switch-flat.md)
- [WRX90 + 3× c-payne (hierarchy)](wrx90-cpayne-microchip-switches.md)
- [WRX90 + 4× c-payne (16 GPU)](wrx90-cpayne-16gpu-4switch.md)

## Table of Contents

- [Summary](#summary)
- [Exact Trigger](#exact-trigger)
- [Measurements](#measurements)
- [Where It Appears](#where-it-appears)
- [Where It Does NOT Appear](#where-it-does-not-appear)
- [What is and is NOT the Cause](#what-is-and-is-not-the-cause)
- [IOMMU Effect](#iommu-effect)
- [Reproduction Steps](#reproduction-steps)
- [Workarounds](#workarounds)
- [Open Questions](#open-questions)

---

## Summary

On multi-GPU PCIe topologies where **GPU-to-GPU P2P traffic crosses CPU root complexes**, large unidirectional write bandwidth collapses from **~50 GB/s to ~2–15 GB/s** under a specific flow pattern. The same GPUs **read** at full speed — the bug only affects **posted writes**.

| Scenario | Write | Read |
|---|---|---|
| Single flow cross-root | 37–53 GB/s | 40–55 GB/s |
| 2 flows, same src switch → **different dst root complexes** | **2.7–14 GB/s COLLAPSE** | 51–57 GB/s OK |
| 2 flows, same src switch → same dst root | 40–55 GB/s | 50–55 GB/s |
| 2 flows, different src switches → different dst | 75–107 GB/s | 50–105 GB/s |

---

## Exact Trigger

All of these must be true **simultaneously**:

1. **Two or more different GPUs on the same source PCIe switch** (or switch partition / Virtual Switch)
2. Sending **posted write** TLPs concurrently
3. Through **one shared upstream link** to one CPU root port
4. To destinations behind **two or more different CPU root complexes**

Posted writes in PCIe are fire-and-forget — they do not require completions. The CPU root complex / Data Fabric must route them internally. When a single source uplink injects posted writes destined for multiple different root complexes simultaneously, internal arbitration enters a pathological state.

### Canonical collapse recipe

```
GPU_A (on switch S1) → (uplink U1) → CPU → root port R1 → GPU_X (on switch S_X)
GPU_B (on switch S1) → (uplink U1) → CPU → root port R2 → GPU_Y (on switch S_Y)

where R1 and R2 live on DIFFERENT CPU root complexes
```

### Non-triggers

```
Same GPU → multiple destinations:     OK (GPU DS port is single-threaded)
Diff src switches → diff dst:         OK (diff uplinks, no contention)
Same src sw → same dst root complex:  OK (single arbitration target)
Remote → local (reversed direction):  OK
Non-posted reads (any pattern):       OK
```

---

## Measurements

### ASUS ESC8000A-E13P (dual EPYC 9575F Turin, Broadcom PEX890xx)

```
SAME switch → SAME dst switch:    ~50 GB/s write, ~50 GB/s read  (OK)
SAME switch → DIFF dst switches:   ~2.7 GB/s write, ~51 GB/s read  (COLLAPSE 18×)
```

Originally mis-attributed to a Broadcom PEX890xx firmware bug. Cross-root collapse was severe because GPU0,1 and GPU4,5 shared one Broadcom chip with a single x4 uplink at the time.

### WRX90 + 4× c-payne Microchip Switchtec (16 GPU, Threadripper PRO 7955WX, 3 root complexes)

```
SW1→SW3+SW4 (both root cplx e0):         53 GB/s write, 53 GB/s read   OK
SW1→SW2(root 40) + SW1→SW3(root e0):     11.6 GB/s write, 52.9 GB/s read   COLLAPSE
SW1→SW2(root 40) + SW1→SW4(root e0):     13.3 GB/s write, 53.8 GB/s read   COLLAPSE
SW1 → SW2 + SW3 + SW4 (3 dst sw):        19.2 GB/s write, 53.2 GB/s read   COLLAPSE
```

### WRX90 + 3× c-payne hierarchy (8 GPU, same CPU)

No collapse — the root switch routes cross-leaf traffic through switch fabric, avoiding the CPU root complex entirely.

### WRX90 + 2× c-payne flat (8 GPU, same CPU, 2 root complexes)

No collapse — only 2 root complexes exist, so the "2+ different dst root complexes" condition cannot be met from a single source switch.

---

## Where It Appears

| Platform | GPU count | CPU root complexes | Uses switches | Collapse? |
|---|---|---|---|---|
| ASUS ESC8000A-E13P Turin dual-socket | 8 | 4 (2 per socket) | Broadcom PEX890xx | **Yes** |
| ASRock WRX90 + 4× c-payne | 16 | 3 | Microchip Switchtec | **Yes** |
| ASRock WRX90 + 3× c-payne (root switch) | 8 | 2 | Microchip Switchtec | No |
| ASRock WRX90 + 2× c-payne (flat) | 8 | 2 | Microchip Switchtec | No |
| Dual Turin direct-attach (Festr's budgetserver) | 8 | 2 | None (direct PCIe) | Not observed (1 switch equivalent) |

The collapse requires both:
- 3+ independent CPU root complexes
- Traffic from one switch targeting 2+ of them

---

## Where It Does NOT Appear

- **2-root-complex systems** — there are not enough targets to trigger the pathological routing
- **Hierarchical PCIe switch topology with a root switch** (e.g. 3-switch c-payne) — cross-switch traffic routes through switch fabric without touching the CPU
- **Non-posted reads** — completion-based flow control prevents the arbitration pathology
- **Reverse direction** (remote → local) — load/pull mechanics differ from push

---

## What is and is NOT the Cause

### Ruled out

| Hypothesis | Evidence |
|---|---|
| Broadcom PEX890xx firmware bug | Reproduces on unrelated Microchip Switchtec switches |
| Microchip firmware bug | Does not appear on fewer-root-complex variants of same hardware |
| ACS `ReqRedir` forcing P2P through root port | All ACS disabled, collapse persists |
| PCIe MaxReadReq 128B on switch | Tested changing to 512B/4096B on endpoints, collapse persists |
| 10-Bit Tag disabled | Unchanged behavior |
| GPU DS port saturation | Single-GPU → many destinations does not collapse |
| NVIDIA P2P driver mode | Collapse happens with P2P on AND off (SHM mode) |
| Transfer size | Identical collapse at 4MB, 256MB, 1024MB |
| Packet rate / MaxReadReq | Changing MRRS 128/512/4096 has no effect on collapse |
| Kernel CPU mitigations | `mitigations=off` already set |

### Strongly indicated

**The bug is in the AMD CPU side — specifically in how the Data Fabric / NBIO handles posted-write arbitration when one upstream PCIe port injects writes destined for multiple different root complexes.**

Reasoning:
- Same PCIe topology + fewer root complexes = no collapse
- Same root complex set + different switches = same collapse
- Traffic never reaches the switch fabric in the cross-complex case (it goes into and out of the CPU)
- `iommu=on` (full translated mode) eliminates the collapse — IOMMU engine inserts a translation layer that reorders/buffers the posted writes, bypassing the arbitration bug

---

## IOMMU Effect

Tested three IOMMU modes on WRX90 16-GPU 4-switch setup:

| Mode | Single-flow BW | Collapse pattern (SW1→SW2+SW3) | 8-GPU allreduce |
|---|---|---|---|
| `iommu=off` | 52–54 GB/s | **11.6 GB/s COLLAPSE** | Works |
| `iommu=pt` (passthrough) | 52–54 GB/s | **14.1 GB/s COLLAPSE** | Works |
| `iommu=on` (full translated) | 43 GB/s (−15%) | **44.7 GB/s OK** | **Hangs** |

### Key observations

- `iommu=pt` uses identity mapping and **does not** mask the collapse. So the IOMMU TLB alone doesn't help — full page-table translation is required.
- `iommu=on` fixes the collapse but:
  - Single-flow BW drops from ~53 to ~43 GB/s (15%)
  - Latency measurements become unrealistic (min 0.07 µs, impossible)
  - Bandwidth numbers for "single reader all peers" exceed theoretical Gen5 x16 (>63 GB/s per GPU)
  - **8-GPU p2pmark allreduce hangs** (deadlock or DMA timeout)
  - Measurement artifacts suggest IOMMU IOTLB caching/reordering is producing ghost numbers

**`iommu=on` is not a production-safe workaround.**

---

## Reproduction Steps

### Minimal Python repro (requires torch + 8 GPUs on 2+ switches across 2+ root complexes)

```python
import torch, time

SIZE = 256 * 1024 * 1024
ITERS = 15

def enable(pairs):
    devs = set([p for pair in pairs for p in pair])
    for a in devs:
        torch.cuda.set_device(a)
        for b in devs:
            if a != b:
                try: torch.cuda.enable_peer_access(b, 0)
                except: pass

def concurrent(pairs, reads=False):
    bufs = {}; streams = {}
    for s, d in pairs:
        if reads:
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{d}'),
                           torch.empty(SIZE//4, device=f'cuda:{s}'))
        else:
            bufs[(s,d)] = (torch.randn(SIZE//4, device=f'cuda:{s}'),
                           torch.empty(SIZE//4, device=f'cuda:{d}'))
        torch.cuda.set_device(s)
        streams[(s,d)] = torch.cuda.Stream(torch.device(f'cuda:{s}'))
    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        for s, d in pairs:
            with torch.cuda.stream(streams[(s,d)]): bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()
    return SIZE * ITERS * len(pairs) / (time.perf_counter() - t0) / 1e9

# Adjust indices to match your topology:
# Pick 2 GPUs on ONE source switch, and 2 destinations each behind a DIFFERENT CPU root complex.
# Example for WRX90 16-GPU 4-switch layout (SW1=GPU0-3 root 00, SW2=GPU4-7 root 40, SW3=GPU8-11 root e0):
pairs_collapse = [(0, 4), (1, 8)]   # SW1 → SW2(root 40) + SW1 → SW3(root e0)
pairs_control  = [(0, 4), (2, 4)]   # same src switch, same dst switch

for label, pairs in [("COLLAPSE", pairs_collapse), ("CONTROL", pairs_control)]:
    enable(pairs)
    w = concurrent(pairs, reads=False)
    r = concurrent(pairs, reads=True)
    print(f"{label:10s} WRITE={w:6.1f} GB/s  READ={r:6.1f} GB/s")
```

Expected output on an affected system:
```
COLLAPSE   WRITE=  11.6 GB/s  READ=  52.9 GB/s
CONTROL    WRITE=  54.1 GB/s  READ=  54.7 GB/s
```

### Identifying the CPU root complex for each GPU

```bash
# Walk each GPU up the PCIe tree to the CPU root port bus (00, 40, e0, etc.)
for gpu in $(nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader | sed 's/00000000://'); do
  root=$(readlink -f /sys/bus/pci/devices/0000:${gpu,,}/../.. 2>/dev/null | grep -oP 'pci[0-9]+:[0-9a-f]+' | head -1)
  echo "GPU $gpu → root complex $root"
done
```

### `lspci` topology

```bash
lspci -t
# Count distinct top-level PCIe domains (root complexes):
lspci | grep "Root Port\|Host bridge" | awk '{print substr($1,1,2)}' | sort -u
```

---

## Workarounds

Ordered by practical usefulness:

### 1. Change topology (most effective)

Use a **hierarchical PCIe switch with a root switch** (e.g. Microchip PM50100-based c-payne 3-switch setup, or NVIDIA MGX PCIe boards). Cross-switch traffic is routed fabric-to-fabric and never reaches a CPU root complex.

This is the only "fix" that preserves full bandwidth.

### 2. Application-level avoidance

In NCCL / custom allreduce topologies, ensure the traffic pattern never contains "same source switch → multiple destination root complexes":
- Use **ring allreduce** ordering that does not create this pattern
- Limit TP group to GPUs on at most 2 root complexes
- For 16-GPU configurations, prefer TP within a single source switch

### 3. IOMMU = on (workaround with downsides)

```
GRUB_CMDLINE_LINUX="... amd_iommu=on iommu=on"
```

Fixes the collapse but:
- 15% single-flow bandwidth drop
- Hangs on some NCCL allreduce configurations at 8+ GPU
- Measurement artifacts (ghost bandwidths, bogus latencies)

**Not recommended for production.**

### 4. Disable P2P (fallback via DRAM staging)

```bash
export NCCL_P2P_DISABLE=1
export NCCL_MIN_NCHANNELS=16
export NCCL_MAX_NCHANNELS=32
export NCCL_BUFFSIZE=33554432
```

Routes traffic through host RAM, bypassing the cross-root arbitration. Higher latency but no collapse. Reasonable for high-concurrency serving.

### 5. BIOS (worth trying, not validated)

Worth experimenting with:
- **Data Fabric C-states → Disabled** (AMD CBS → DF Common Options)
- **NBIO PCIe → Relaxed Ordering → Enabled**
- **Preferred I/O** → set GPU devices
- **x2APIC mode → Enabled**
- **Above 4G Decoding → Enabled** (usually already set)

None of these have been confirmed to eliminate the collapse, but they adjust the Data Fabric / NBIO behavior and are cheap to try.

---

## Open Questions

1. **Kernel version** — not yet tested whether a newer kernel (e.g. 6.18) eliminates the collapse. Kernel 6.17 behavior is same with `iommu=off` and `iommu=pt`.
2. **Why ASUS cannot reproduce** — they likely run default BIOS with IOMMU enabled (probably `iommu=pt` or stock on Linux, which differs from ours). Needs clarification.
3. **Is there an AMD errata** — not publicly documented. Would need NDA-level contact with AMD GPSE.
4. **Does it affect AMD EPYC Genoa / Milan the same way** — untested.
5. **IOMMU IOTLB flush / invalidation behavior** — possible that selective IOMMU configuration (e.g. only specific IOVA ranges translated) retains the fix without the side effects.

---

## Related work

- [NCCL all_reduce_perf slower with P2P on AMD EPYC 7302](https://github.com/NVIDIA/nccl-tests/issues/74) — documented NCCL slowdown on AMD EPYC, workaround is often `NCCL_P2P_DISABLE=1`
- [Inconsistent P2P bandwidth and NCCL allreduce bus bandwidth #814](https://github.com/NVIDIA/nccl/issues/814)
- [XRT Documentation: IOMMU redirects P2P through root complex](https://xilinx.github.io/XRT/master/html/p2p.html) — explains why IOMMU routing affects P2P
- [Understanding Data Movement in AMD Multi-GPU Systems with Infinity Fabric (arXiv)](https://arxiv.org/html/2410.00801v1) — hints that DMA engines cannot always saturate interconnects at multi-destination
