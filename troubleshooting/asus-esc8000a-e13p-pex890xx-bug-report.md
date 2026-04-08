# ASUS ESC8000A-E13P — PEX890xx Bug Report

Vendor escalation report for a critical posted-write arbitration bug in the Broadcom PEX890xx PCIe Gen5 switch on the ASUS ESC8000A-E13P GPU server.

## System Information

### Question 1 — Serial Number

**DN: T9S0CG000107**

### Question 2 — BIOS / Firmware

| Component | Version |
|---|---|
| BIOS | American Megatrends Inc. v1104 (2025-10-23) |
| NVIDIA Driver | 595.58.03 (Open Kernel Module) |
| CUDA | 13.2 |
| Broadcom PEX890xx | rev b0 |

### Question 3 — Hardware Configuration

| Component | Details |
|---|---|
| **CPUs** | 2× AMD EPYC 9575F 64-Core (Turin), 2 sockets, 128 cores / 256 threads |
| **RAM** | 24× 64 GB DDR5-6400 (Kingston 9965844-029.A00G), 1536 GB total, all 24 channels populated (12 per socket) |
| **GPUs** | 8× NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7 each) |
| **PCIe Switches** | 2× Broadcom PEX890xx Gen5 (device 1000:C030, rev b0) |
| **NIC** | 2× Intel X710 10GBASE-T, 2× Emulex OneConnect (Skyhawk) |
| **Storage** | 1× Kingston FURY Renegade NVMe SSD, 2× AMD SATA (AHCI) |

### Question 5 — Operating System

```
Ubuntu 24.04.4 LTS (Noble Numbat)
Kernel: 6.17.0-20-generic #20~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC
Architecture: x86_64

Kernel parameters:
  BOOT_IMAGE=/vmlinuz-6.17.0-20-generic root=UUID=78c30682-...
  rd.auto=1 rd.md=1 rd.md.conf=1
  mitigations=off spectre_v2=off spec_store_bypass_disable=off
  l1tf=off mds=off tsx_async_abort=off srbds=off mmio_stale_data=off
  retbleed=off amd_iommu=off iommu=off
```

### Question 6 — OS Logs

Full logs attached:
- [`dmesg.log`](../logs/asus-report/dmesg.log) — kernel boot messages
- [`journal.log`](../logs/asus-report/journal.log) — systemd journal
- [`lspci_full.log`](../logs/asus-report/lspci_full.log) — complete PCIe device enumeration
- [`nvidia_smi_full.log`](../logs/asus-report/nvidia_smi_full.log) — GPU status
- [`dmidecode.log`](../logs/asus-report/dmidecode.log) — hardware inventory

No PCIe AER errors observed. No GPU errors in dmesg.

---

## Email 2 — Bandwidth Test Results

### Scenario 1–5: GPU-to-Memory Write Bandwidth

All GPUs writing to NUMA node 0 pinned host memory. Transfer size: 256 MB, 20 iterations.

| Scenario | GPUs | Total BW (GB/s) | Per-GPU BW (GB/s) |
|---|---|---|---|
| 1 | GPU0 alone | 56.3 | 56.3 |
| 2 | GPU0 + GPU1 | 56.3 | 28.1 |
| 3 | GPU0 + GPU2 | 56.3 | 28.1 |
| 4 | GPU0 + GPU1 + GPU2 | 56.3 | 18.8 |
| 5 | GPU0 + GPU1 + GPU2 + GPU3 | 56.2 | 14.1 |

**Observation:** Total bandwidth stays constant at ~56 GB/s regardless of GPU count. This is the per-uplink bandwidth limit (~38 GB/s measured P2P, ~56 GB/s to host memory). All 4 GPUs share the switch uplinks to CPU, so total bandwidth is capped by the uplink.

### GPU P2P Bandwidth Scenarios

| Scenario | Bandwidth | Status |
|---|---|---|
| GPU0→GPU4 (cross-chip baseline) | 39.6 GB/s | OK |
| GPU0→GPU4 + GPU1→GPU6 (same VS, diff target root ports) | **2.7 GB/s** | **COLLAPSE** |
| GPU0→GPU4 + GPU2→GPU6 (diff VS, diff target root ports) | 79.0 GB/s | OK |
| GPU0→GPU4 + GPU1→GPU5 (same VS, same target root port) | 43.3 GB/s | OK |
| GPU0→GPU1 (same switch) | 53.4 GB/s | OK |
| GPU0→GPU2 (same chip, cross-VS) | 53.9 GB/s | OK |

### Traffic Pattern Confirmation

**Yes, the traffic pattern is primarily memory write (posted write TLPs).**

We confirmed this by testing both directions:
- **Posted writes** (GPU pushes data to remote GPU): **2.7 GB/s COLLAPSE**
- **Non-posted reads** (GPU pulls data from remote GPU): **51.0 GB/s NO COLLAPSE**

The bug affects **only posted writes**. Non-posted reads with completions work correctly under identical conditions.

### Broadcom Switch Configuration

- **ACS:** Disabled on all root ports and switch downstream ports (required for P2P through switch fabric)
- **Virtual Switch Mode:** 2 VS per chip (4 VS total across 2 chips)
- **MaxReadReq on switch ports:** 128 bytes (firmware-locked, read-only via setpci)
- **10-Bit Tag:** Capable but disabled on switch ports (10BitTagComp+ 10BitTagReq-)
- **All switch DevCtl/DevCtl2 registers:** Read-only (firmware-locked)

We do not have access to Broadcom switch XML configuration or trace logs. The Broadcom PCI/PCIe SDK v9.81 was compiled and loaded, but PEX89000 internal registers return error 524 — the SDK's PLX 8000-series register access does not work with PEX89000's MRPC-based interface. We would need Broadcom's proprietary PEX89000 management tools to extract switch configuration.

---

## Detailed Bug Description

### Broadcom PEX890xx Posted-Write Arbitration Bug

#### Exact Trigger

When **2 or more different GPUs** on the **same Virtual Switch partition** (sharing one upstream port) simultaneously send **posted write TLPs** through that shared uplink to destinations behind **2 or more different CPU root ports**, total bandwidth collapses from ~37 GB/s to **2.7 GB/s** (93% loss).

#### Topology Context

```
    CPU Socket 0                        CPU Socket 1
    ┌──────────────┐                    ┌──────────────┐
    │ Root Port     │                    │ Root Port     │
    │ 10:01.1       │     xGMI           │ 80:01.1       │
    │ Root Port     │◄──350 GB/s───►    │ Root Port     │
    │ 70:01.1       │                    │ f0:01.1       │
    └──┬─────┬──────┘                    └──┬─────┬──────┘
       │     │                              │     │
    ┌──┴─────┴──────┐                    ┌──┴─────┴──────┐
    │ CHIP A        │                    │ CHIP B        │
    │ VS0    VS1    │                    │ VS2    VS3    │
    │ GPU0,1 GPU2,3 │                    │ GPU4,5 GPU6,7 │
    └───────────────┘                    └───────────────┘
```

#### Collapse Example

```
COLLAPSE (2.7 GB/s total):
  GPU0 (VS0, uplink 10:01.1) → writes to GPU4 (behind root port 80:01.1)
  GPU1 (VS0, uplink 10:01.1) → writes to GPU6 (behind root port f0:01.1)
  ↑ Same VS, same uplink, DIFFERENT target root ports

NO COLLAPSE (79 GB/s total):
  GPU0 (VS0, uplink 10:01.1) → writes to GPU4 (behind root port 80:01.1)
  GPU2 (VS1, uplink 70:01.1) → writes to GPU6 (behind root port f0:01.1)
  ↑ Different VS, different uplinks
```

#### Why Only Posted Writes

| TLP Type | Under Collapse Trigger | Notes |
|---|---|---|
| Posted Write (GPU pushes data) | **2.7 GB/s** | No flow control, switch must buffer/arbitrate internally |
| Non-Posted Read (GPU pulls data) | **51.0 GB/s** | Completion-based flow control prevents pathology |

#### What Does NOT Trigger Collapse

| Scenario | Result | Reason |
|---|---|---|
| Same GPU → multiple root ports | 36 GB/s | Single DMA engine, serialized |
| Different VS → different root ports | 79 GB/s | Different uplinks, no shared path |
| Same VS → same root port | 43 GB/s | Same destination, normal sharing |
| Reverse direction | 51 GB/s | Different source chip |
| Non-posted reads (any pattern) | 40–51 GB/s | Completion flow control |
| P2P disabled (SHM through host memory) | 2.7 GB/s | Same collapse — same physical path |

#### Bug Is Independent Of

- Transfer size (tested 4 MB to 1024 MB — identical collapse)
- MaxReadReq setting (tested 128B, 512B, 4096B)
- P2P enabled vs disabled (SHM uses same uplink path)
- Kernel version (tested 6.17.0-19 and 6.17.0-20)
- NVIDIA driver version (tested 595.45.04 and 595.58.03)
- ACS state (collapse occurs with ACS on or off, but ACS must be off for same-chip P2P)

---

## Reproduction Steps

### Prerequisites

```bash
# Ubuntu 24.04 with kernel 6.17.0-20-generic
# NVIDIA driver 595.58.03 (open kernel module)
# PyTorch with CUDA support
pip install --break-system-packages torch

# Disable ACS (required for P2P through switch fabric)
# This script finds all devices with ACS ReqRedir+ and disables them:
python3 -c "
import subprocess, re
out = subprocess.check_output(['lspci', '-vv'], text=True, timeout=30)
for dev in out.split('\n\n'):
    lines = dev.strip().split('\n')
    if not lines or not lines[0]: continue
    bdf = lines[0].split()[0]
    acs_offset = None
    for line in lines:
        if 'Access Control Services' in line:
            m = re.search(r'\[([0-9a-fA-F]+)\s', line)
            if m: acs_offset = m.group(1)
        if 'ACSCtl:' in line and 'ReqRedir+' in line and acs_offset:
            ctrl_offset = int(acs_offset, 16) + 6
            subprocess.run(['setpci', '-s', bdf, f'{ctrl_offset:x}.w=0x0011'], check=True)
            print(f'Disabled ACS on {bdf}')
            break
"

# Verify: should output 0
lspci -vv 2>/dev/null | grep "ACSCtl:" | grep -c "ReqRedir+"
```

### Reproduction Script

Save as `reproduce_bug.py` and run with `python3 reproduce_bug.py`:

```python
#!/usr/bin/env python3
"""
Broadcom PEX890xx Posted-Write Arbitration Bug — Reproduction Script
ASUS ESC8000A-E13P with 8x RTX PRO 6000 Blackwell

Requirements:
  - ACS disabled on all root ports and switch downstream ports
  - PyTorch with CUDA support
  - Run as root
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

    for s, d in pairs:
        with torch.cuda.stream(streams[(s,d)]):
            bufs[(s,d)][1].copy_(bufs[(s,d)][0])
    torch.cuda.synchronize()

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

    ngpu = torch.cuda.device_count()
    print(f"GPUs detected: {ngpu}")
    if ngpu < 8:
        print(f"WARNING: Need 8 GPUs, found {ngpu}")
        sys.exit(2)

    enable_p2p()

    # TEST 1: BUG TRIGGER
    # GPU0 and GPU1 on same VS (VS0, shared uplink 10:01.1)
    # GPU4 behind root port 80:01.1, GPU6 behind root port f0:01.1
    collapse = [(0, 4), (1, 6)]
    print("\n--- TEST 1: BUG TRIGGER ---")
    print("GPU0(VS0)→GPU4 + GPU1(VS0)→GPU6")
    print("Same VS, same uplink, DIFFERENT target root ports")
    w = concurrent_transfer(collapse, "write")
    r = concurrent_transfer(collapse, "read")
    print(f"  Posted WRITE:     {w:6.1f} GB/s  {'← COLLAPSE!' if w < 10 else ''}")
    print(f"  Non-posted READ:  {r:6.1f} GB/s  {'← NO collapse' if r > 20 else ''}")

    # TEST 2: CONTROL
    ok = [(0, 4), (2, 6)]
    print("\n--- TEST 2: CONTROL ---")
    print("GPU0(VS0)→GPU4 + GPU2(VS1)→GPU6")
    print("Different VS, different uplinks")
    w2 = concurrent_transfer(ok, "write")
    r2 = concurrent_transfer(ok, "read")
    print(f"  Posted WRITE:     {w2:6.1f} GB/s")
    print(f"  Non-posted READ:  {r2:6.1f} GB/s")

    # TEST 3: BASELINE
    w3 = concurrent_transfer([(0, 4)], "write")
    print(f"\n--- BASELINE: GPU0→GPU4: {w3:6.1f} GB/s ---")

    print("\n" + "=" * 70)
    if w < 10 and r > 20 and w2 > 50:
        print("RESULT: BUG CONFIRMED")
        print(f"  Collapse: {w:.1f} GB/s (expected ~37, got 93% loss)")
        print(f"  Read OK:  {r:.1f} GB/s (no collapse on reads)")
        print(f"  Control:  {w2:.1f} GB/s (different VS works fine)")
        sys.exit(1)
    else:
        print("RESULT: Bug not detected in this run")
        sys.exit(0)

if __name__ == "__main__":
    main()
```

### Expected Output When Bug Is Present

```
======================================================================
PEX890xx Posted-Write Arbitration Bug — Reproduction
======================================================================
GPUs detected: 8

--- TEST 1: BUG TRIGGER ---
GPU0(VS0)→GPU4 + GPU1(VS0)→GPU6
Same VS, same uplink, DIFFERENT target root ports
  Posted WRITE:        2.7 GB/s  ← COLLAPSE!
  Non-posted READ:    51.0 GB/s  ← NO collapse

--- TEST 2: CONTROL ---
GPU0(VS0)→GPU4 + GPU2(VS1)→GPU6
Different VS, different uplinks
  Posted WRITE:       79.0 GB/s
  Non-posted READ:    79.3 GB/s

--- BASELINE: GPU0→GPU4: 39.7 GB/s ---

======================================================================
RESULT: BUG CONFIRMED
  Collapse: 2.7 GB/s (expected ~37, got 93% loss)
  Read OK:  51.0 GB/s (no collapse on reads)
  Control:  79.0 GB/s (different VS works fine)
```

---

## Questions for Broadcom

1. Is this a known errata for PEX890xx rev b0?
2. Is there a firmware update that addresses posted-write arbitration under multi-root-port fanout?
3. Can the switch configuration (VS layout, arbitration weights, QoS) be modified to avoid this pathology?
4. Can MaxReadReq and 10-Bit Tag settings be changed through the PEX89000 management interface?
5. Is the embedded analyzer / trace capability available to capture the internal switch state during the collapse?

## Full Technical Analysis

See [ASUS ESC8000A-E13P with Broadcom PEX890xx Switches](../hardware/asus-esc8000a-e13p-broadcom-switches.md) for the complete investigation including topology discovery, ACS configuration, uplink degradation proofs, and multi-flow contention analysis.
