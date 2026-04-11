# PCIe Bandwidth & P2P Performance

Measured PCIe peer-to-peer bandwidth, latency, and NCCL AllReduce performance on RTX PRO 6000 Blackwell systems. These numbers are the foundation for understanding inference throughput limits.

## Table of Contents

- [P2P Bandwidth Measurements](#p2p-bandwidth-measurements)
- [P2P Latency Measurements](#p2p-latency-measurements)
- [NCCL AllReduce Bus Bandwidth](#nccl-allreduce-bus-bandwidth)
- [BAR1 Configuration](#bar1-configuration)
- [How PCIe Bandwidth Affects Inference](#how-pcie-bandwidth-affects-inference)
- [GRUB Kernel Parameters for PCIe Stability](#grub-kernel-parameters-for-pcie-stability)
- [Debugging Tools](#debugging-tools)

---

## P2P Bandwidth Measurements

All measurements taken using the CUDA `p2pBandwidthLatencyTest` sample or luke's `p2pmark` tool.

### 2-GPU PLX vs direct-attach: what the switch actually does (counterintuitive)

Most data on this page is from 4-8 GPU rigs where PCIe switches provide topology-enablement benefits that direct-attach physically can't match. **The 2-GPU case is different and worth understanding before you spec a build.**

**Measured data (same RTX PRO 6000 Blackwell GPUs, same optimized software stack, three platforms):**

| Platform | Topology | Gen | 122B NVFP4 best tok/s @ C=1 |
|---|---|---|---|
| TRX40 (other tester) | Direct-attach (NODE) | Gen4 | ~100 (older vLLM version) |
| WRX90E-SAGE SE (other tester) | Direct-attach (NODE) | Gen5 | 168.2 |
| B650D4U + PM50100 PLX (this repo's reference) | Switch (PIX) | Gen5 | ~198 |

At first glance this suggests the switch is worth ~18% over Gen5 direct-attach. **That framing is misleading.** The real mechanism is nvidia driver P2P routing, not fabric bandwidth.

**The ForceP2P story (2026-04-03 RTX6kPRO Discord PSA):**

SGLang's PCIe oneshot custom allreduce (`--enable-pcie-oneshot-allreduce`) is ~30% faster than NCCL on small LLM decode messages (4-120 KB, the size range that dominates transformer TP allreduce). Whether you actually capture that benefit depends entirely on how the nvidia driver routes P2P writes:

| Topology (per `nvidia-smi topo -m`) | Driver default | Custom AR @ 16 KB | Effective result |
|---|---|---|---|
| **PIX / PXB** (PCIe switch) | BAR1 P2P enabled automatically | ~7 µs | Custom AR wins up to ~120 KB, full benefit captured |
| **NODE** (direct-attach) without fix | Routes P2P via SysMem staging | ~59 µs (vs NCCL ~12 µs) | Auto-crossover detects custom AR losing, sets `max_size=4 KB`, custom AR silently disabled for LLM decode |
| **NODE** with ForceP2P modprobe override | BAR1 P2P forced on | ~7.7 µs | Auto-crossover sets `max_size=120 KB`, custom AR wins across decode range |

The direct-attach fix:

```
# /etc/modprobe.d/nvidia-p2p-override.conf  (NODE topology ONLY -- do NOT add on PIX/PXB)
options nvidia NVreg_RegistryDwords="ForceP2P=0x11;RMForceP2PType=1;RMPcieP2PType=2;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1"
```

Verify after reload: `cat /proc/driver/nvidia/params | grep RegistryDwords` should show the full string. In SGLang startup logs, look for `[PCIe oneshot allreduce] Crossover benchmark` -- the `max_size` line should be ~120 KB, not 4 KB.

**What this means for the 168 → 198 gap:**

The WRX90 direct-attach Gen5 rig that measured 168 tok/s is almost certainly running without `ForceP2P`. The PSA was published 2026-04-03 and isn't in any mainstream documentation; the workaround has to be discovered. Without it, `--enable-pcie-oneshot-allreduce` in his SGLang launch command is decorative -- the auto-crossover disables custom AR at 4 KB, and every 16 KB TP decode allreduce falls back to NCCL ring. His 130→168 optimization gain comes entirely from b12x MoE kernels + NEXTN speculative, with zero contribution from the custom allreduce kernel that the switch rigs get for free.

Estimated gap breakdown at 2 GPUs: **~15-20 tok/s from missing PCIe oneshot benefit, ~7 tok/s from Max-Q 300W vs 600W, ~3-5 tok/s run variance**. With ForceP2P added on WRX90, expected performance is ~185-195 tok/s — essentially matching PLX.

**Refined conclusion for 2-GPU builds:**

The PCIe switch doesn't create a faster fabric or lower silicon latency (both topologies measure identical 0.38 µs P2P latency). What the switch does is **skip an undocumented modprobe workaround that direct-attach users need to know about**. At 2 GPUs with the full optimization stack, switch and direct-attach deliver equivalent performance once configured correctly. The practical difference is that PIX users get the fast path "for free" while NODE users have to discover ForceP2P.

**Cost/value framing at 2 GPUs:**

Since the switch provides no inherent performance advantage at this scale, the build decision becomes pure price-performance. A budget AM5 platform with a PCIe Gen5 switch (B650D4U + c-payne PM50100 reference here: ~$500 motherboard, $700 AM5 EPYC 4564P, 128 GB DDR5 ECC) delivers inference throughput matching a full WRX90E-SAGE SE + Threadripper Pro 9965WX build (~$2000+ motherboard, $4000+ CPU, 256 GB RDIMM) once both are correctly configured. The B650D4U reference platform is ~5-7x cheaper on the CPU + motherboard + RAM line for the same 2-GPU inference throughput.

### When the switch does matter: 4+ GPUs

At 4 or more GPUs the analysis flips completely and PCIe switches become unambiguous wins -- not because of bandwidth or latency but because of topology enablement that direct-attach cannot provide:

1. **Slot density**: consumer/workstation boards max out at 2-3 Gen5 x16 slots from the CPU. A 4-GPU direct-attach build requires dual-socket (cross-NUMA penalties) or exotic bifurcation (halved lane widths). A single switch gives you 4-8 uniform Gen5 x16 downstream ports from one CPU upstream.
2. **NUMA isolation**: all GPUs behind one switch sit in a single uniform fabric. No cross-socket P2P traversal, no `__threadfence_system` stalls (see 2026-04-02 discord PSA about PCIe oneshot AR regressing -4 to -16% on dual-socket Genoa specifically because of this).
3. **All-to-all scaling**: dense peer-to-peer between many GPUs is the normal case at 4+, not the exception. Switches provide predictable cut-through routing for every pair; dual-socket direct-attach has a strict cost inequality between same-socket and cross-socket pairs.
4. **Cost inversion**: workstation boards with 4+ direct-attach Gen5 x16 slots (Threadripper Pro WRX80/WRX90, Xeon W9) cost thousands more than single-socket server boards with a c-payne/Broadcom switch.

**Do not generalize the 2-GPU ForceP2P / cost findings above to 4+ GPU builds.** At higher GPU counts, switches solve topology problems that direct-attach literally cannot, and the price comparison flips in the switch's favor.

### Unidirectional P2P Bandwidth

Measured with P2P Writes, PCIe Gen5 x16 links.

| Source | Setup | Same-NUMA | Cross-NUMA |
|--------|-------|-----------|------------|
| purplepow3r | Dual Turin, 4x 6000 Pro WS + 2x 5090 | ~55-56 GB/s | ~51 GB/s |
| orangezed | Dual EPYC 9374F, 8x 6000 Pro MaxQ | ~54 GB/s | ~39 GB/s |
| Festr | Dual Turin, 8x 6000 Pro Server | ~53 GB/s | -- |
| luke | Switches, 8x 6000 Pro MaxQ | ~54 GB/s | N/A (single CPU) |
| voipmonitor | [ASUS ESC8000A-E13P, Broadcom PEX890xx switches](asus-esc8000a-e13p-broadcom-switches.md), 8x 6000 Pro Server | ~54 GB/s | ~50 GB/s (cross-NUMA) |

**Theoretical maximum:** PCIe Gen5 x16 = 63 GB/s unidirectional. The ~56 GB/s measured represents ~89% efficiency.

### Bidirectional P2P Bandwidth

| Source | Setup | Same-NUMA | Cross-NUMA |
|--------|-------|-----------|------------|
| purplepow3r | Dual Turin | ~111 GB/s | ~99 GB/s |
| orangezed | Dual EPYC 9374F | ~103 GB/s | ~64 GB/s |
| voipmonitor | [ASUS ESC8000A-E13P + Broadcom switches](asus-esc8000a-e13p-broadcom-switches.md) (ACS off) | ~103 GB/s | ~95 GB/s |

### Full P2P Bandwidth Matrix Example

From purplepow3r's dual Turin system (4x RTX 6000 Pro WS + 2x RTX 5090):

```
Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 1488.10  56.57  56.57  50.97  55.61  55.59
     1   56.57 1416.77  56.57  51.01  55.60  55.64
     2   56.57  56.57 1415.11  50.97  55.62  55.64
     3   50.97  50.97  50.97 1375.60  50.16  50.45
     4   55.61  55.60  55.62  50.16 1408.87  55.60
     5   55.60  55.64  55.64  50.45  55.61 1489.91

Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1      2      3      4      5
     0 1485.22 111.38 111.39  99.42 111.02 111.10
     1  111.38 1416.86 111.38  99.59 111.06 111.15
     2  111.39 111.39 1415.11  99.53 111.04 111.12
     3   99.42  99.59  99.53 1377.73  99.09  99.43
     4  111.02 111.06 111.04  99.09 1409.15 111.13
     5  111.10 111.15 111.12  99.43 111.13 1487.77
```

Note: Device 3 shows lower bandwidth (~51/99 GB/s) -- this is the cross-NUMA path. Devices 0-2 are on NUMA0, devices 3-5 are on NUMA1.

### B650D4U + PLX (2x RTX PRO 6000 Blackwell, c-payne PM50100 Gen5 PLX switch, PIX topology)

Measured 2026-04-10 on AMD EPYC 4564P / B650D4U with `p2pBandwidthLatencyTest` from CUDA samples. Both GPUs behind a single Gen5 PLX switch (bus IDs 03:00, 04:00).

```
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 1498.89  22.91
     1   22.74 1536.38

Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1
     0 1502.40  52.24
     1   52.35 1525.93

Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 1487.10  23.55
     1   23.77 1501.64

Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 1486.64  51.04
     1   51.04 1499.48

P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1
     0   0.97   0.38
     1   0.44   1.00

P2P=Disabled Latency Matrix (us)
   GPU     0      1
     0   0.98  14.35
     1  14.68   1.02
```

**Summary:**
- Unidirectional P2P: **52.3 GB/s** (~83% of Gen5 x16 theoretical 63 GB/s)
- Bidirectional P2P: **51.0 GB/s** -- note this is essentially identical to unidirectional, not 2x. On multi-GPU-behind-single-switch topologies, the bidirectional benchmark saturates the single upstream PLX port, so both streams share one link's worth of bandwidth. Direct-attached topologies (Festr/purplepow3r) see ~2x bidirectional because each GPU has its own CPU root port.
- P2P enabled latency: **0.38-0.44 us** (cross-GPU)
- P2P disabled latency: **14.35-14.68 us** (~36x penalty when falling back to host-mediated copy)
- Same-GPU self-bandwidth: ~1500 GB/s (HBM/GDDR7 local, sanity check)

Raw output: [`benchmarks/p2p/2026-04-10_b650d4u_p2pBandwidthLatencyTest.txt`](../benchmarks/p2p/2026-04-10_b650d4u_p2pBandwidthLatencyTest.txt)

### TRX40 (2x RTX PRO 6000 Blackwell, CPU root complex, Gen4 x16, no switch)

Measured 2026-03 on TRX40 Threadripper with `p2pBandwidthLatencyTest` from CUDA samples. Each GPU on its own CPU root port (bus IDs 01:00, 21:00) -- no PCIe switch. Identical GPU silicon to the B650D4U system above, so all deltas below are pure platform/topology.

```
Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 1499.52  23.97
     1   24.23 1531.91

Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)
   D\D     0      1
     0 1502.45  26.60
     1   26.74 1514.05

Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 1484.49  29.53
     1   29.53 1504.53

Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)
   D\D     0      1
     0 1487.34  46.55
     1   46.58 1493.03

P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1
     0   2.07   0.38
     1   0.38   2.07

P2P=Disabled Latency Matrix (us)
   GPU     0      1
     0   2.07  14.42
     1  14.50   2.07
```

**Summary:**
- Unidirectional P2P: **26.6 GB/s** (~83% of Gen4 x16 theoretical 32 GB/s)
- Bidirectional P2P: **46.6 GB/s** (1.75x uni -- each GPU hangs off its own CPU root port, so both directions flow concurrently on independent full-duplex links)
- P2P enabled latency: **0.38 us** (symmetric, both directions)
- P2P disabled latency: **14.42-14.50 us** (~38x penalty)

Raw output: [`benchmarks/p2p/2026-03_trx40_p2pBandwidthLatencyTest.txt`](../benchmarks/p2p/2026-03_trx40_p2pBandwidthLatencyTest.txt)

### B650D4U vs TRX40 Head-to-Head

Same GPU silicon (2x RTX PRO 6000 Blackwell), opposite topologies.

| Metric | TRX40 (CPU root, Gen4) | B650D4U (PLX, Gen5) | Delta |
|---|---|---|---|
| Unidirectional P2P | 26.6 GB/s | **52.3 GB/s** | **B650 +96%** |
| Bidirectional P2P | **46.6 GB/s** | 51.0 GB/s | B650 +10% |
| P2P-enabled latency | 0.38 us | 0.38-0.44 us | tie |
| P2P-disabled latency | 14.42 us | 14.35 us | tie |
| Disabled bi (host bounce) | **29.5 GB/s** | 23.8 GB/s | TRX40 +24% |

**Architectural interpretation:**

- **Unidirectional:** B650D4U wins 2x because Gen5 x16 (~63 GB/s theoretical) vs Gen4 x16 (~32 GB/s). Both platforms hit the same ~83% efficiency; the gap is pure PCIe generation.
- **Bidirectional:** The PLX advantage collapses from 2x to 10%. On TRX40, each GPU has its own CPU root port, so bidirectional traffic flows on two independent full-duplex links (46.6 ≈ 1.75x uni). On B650D4U, even though peer-to-peer traffic stays behind the PLX switch and shouldn't touch the upstream port, the measured bi ≈ uni (51.0 ≈ 1.0x) -- the PLX internal crossbar appears to serialize simultaneous bidirectional P2P rather than running it full-duplex.
- **Latency:** Identical (0.38 us enabled, ~14 us disabled). This is a silicon/driver property, not a platform one. The ~36x enabled/disabled ratio is why `uvm_disable_hmm=1` and NCCL P2P config are critical -- without them traffic silently falls back to the 14 us host-mediated path.
- **Disabled fallback bidirectional:** TRX40 actually wins (29.5 vs 23.8 GB/s) because when P2P is off, traffic routes through host RAM -- and TRX40's dual root complexes give each GPU its own independent DMA path to memory, while B650D4U still bottlenecks on the shared PLX upstream.

**Implications for tensor-parallel inference:**

Tensor-parallel allreduce is inherently bidirectional (ring scatter-reduce + all-gather: each GPU sends and receives concurrently). The bi P2P number is the relevant ceiling, not uni. That's why NCCL allreduce caps around ~25 GB/s on both machines -- both are bumping against their respective bi ceilings minus NCCL overhead.

The PLX's big uni advantage only shows up in workloads whose cross-GPU traffic is more one-way: MTP/speculative decode (drafter -> verifier), prefill streaming, KV cache migration. This matches the observed inference deltas -- 122B MTP +30% on B650D4U (uni-friendly), while non-MTP dense decode is only +5-10%.

### TRX40 NCCL AllReduce (2-GPU, nccl-tests 2.18.2)

```
NCCL_P2P_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS NCCL_IB_DISABLE=1 NCCL_MIN_NCHANNELS=8 \
  ./build/all_reduce_perf -b 8M -e 2G -f 2 -g 2 -n 50
```

| Message size | Time (us) | algbw (GB/s) | busbw (GB/s) |
|---|---|---|---|
| 8 MB | 474 | 17.71 | 17.71 |
| 16 MB | 943 | 17.80 | 17.80 |
| 32 MB | 1,873 | 17.92 | 17.92 |
| 64 MB | 3,725 | 18.01 | 18.01 |
| 128 MB | 7,376 | 18.20 | 18.20 |
| 256 MB | 14,586 | 18.40 | 18.40 |
| 512 MB | 28,898 | 18.58 | 18.58 |
| 1 GB | 57,328 | 18.73 | 18.73 |
| 2 GB | 113,682 | 18.89 | 18.89 |

**Avg bus bandwidth: 18.26 GB/s.** Note busbw == algbw for 2-GPU allreduce (the formula is `algbw * 2(n-1)/n` = `algbw * 1` for n=2). The 18.89 GB/s peak is well below the 46.6 GB/s bidirectional P2P hardware ceiling -- NCCL's ring reduction + compute for the reduction op eats ~60% of available bandwidth. A matching B650D4U run with the same command is the missing apples-to-apples comparison.

### p2pmark Scores (8 GPUs)

luke's [p2pmark](https://github.com/lukealonso/p2pmark) tool provides a standardized comparison:

| System | PCIe Link Score | Dense Interconnect Score | Effective Latency |
|--------|----------------|------------------------|-------------------|
| luke (3x switches, single CPU) | 0.86 (54.3 GB/s) | 0.44 (191.8 / 434.7 GB/s) | 6.79 us |
| Festr (dual Turin, direct-attach) | 0.84 (52.7 GB/s) | 0.41 (173.1 / 421.3 GB/s) | 6.03 us |
| Grimulkan (4x switches, single CPU) | 0.86 (53.9 GB/s) | 0.38 (164.3 / 431.2 GB/s) | 7.04 us |
| [voipmonitor (ASUS ESC8000A-E13P, Broadcom, ACS off)](asus-esc8000a-e13p-broadcom-switches.md) | 0.85 (53.7 GB/s) | 0.12 (51.7 / 429.2 GB/s) | 7.39 us |

> **Note:** voipmonitor's 8-GPU all-to-all score (0.12) is low because the dual-socket Infinity Fabric saturates under 56 concurrent flows. Same-NUMA 4-GPU performance is excellent (0.58, comparable to Festr). See [detailed page](asus-esc8000a-e13p-broadcom-switches.md) for full analysis.

### p2pmark Scores (4 GPUs)

| System | PCIe Link Score | Dense Interconnect Score | Effective Latency |
|--------|----------------|------------------------|-------------------|
| luke (1 switch) | 0.86 | 0.64 (138.3 / 217.7 GB/s) | 4.10 us |
| Festr (Turin, same NUMA) | 0.88 | 0.59 (129.7 / 220.6 GB/s) | 2.28 us |
| [voipmonitor (ASUS ESC8000A-E13P, Broadcom, ACS off)](asus-esc8000a-e13p-broadcom-switches.md) | 0.88 | 0.58 (127.6 / 221.0 GB/s) | 2.12 us |

---

## P2P Latency Measurements

### P2P Enabled vs Disabled

| Condition | Cross-GPU Latency |
|-----------|-------------------|
| P2P Enabled (same NUMA) | 0.36-0.45 us |
| P2P Enabled (cross-NUMA, Turin) | 0.44 us |
| P2P Disabled | ~14 us |

P2P enablement provides a **30x latency reduction**. This is why the `nvidia_uvm uvm_disable_hmm=1` fix and proper NCCL P2P configuration are critical.

### Full P2P Latency Matrix Example

From purplepow3r's system:

```
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5
     0   2.07   0.37   0.36   0.38   0.44   0.36
     1   0.37   2.07   0.36   0.38   0.44   0.36
     2   0.37   0.37   2.07   0.38   0.44   0.36
     3   0.38   0.38   0.38   2.07   0.38   0.38
     4   0.44   0.44   0.44   0.38   2.07   0.36
     5   0.36   0.36   0.36   0.38   0.36   2.07
```

From luke's 8x RTX 6000 Pro on switches:

```
P2P=Enabled Latency (P2P Writes) Matrix (us)
   GPU     0      1      2      3      4      5      6      7
     0   2.07   0.45   0.51   0.45   0.45   0.45   0.45   0.44
     1   0.52   2.07   0.44   0.45   0.52   0.45   0.45   0.44
     ...
```

---

## NCCL AllReduce Bus Bandwidth

AllReduce is the dominant collective operation in tensor-parallel inference. Higher bus bandwidth means faster decode.

### 8-GPU Results

| Setup | NCCL Config | Message Size | Bus BW (GB/s) |
|-------|-------------|-------------|---------------|
| luke (3x switches) | MIN_NCHANNELS=8 | Sweep 8M-2G | **41.1** |
| Grimulkan (4x switches) | MIN_NCHANNELS=8 | Sweep 8M-2G | 39.4 |
| Grimulkan (4x switches) | Default | 32M fixed | 40.1 |
| Festr (dual Turin) | MIN_NCHANNELS=8 | Sweep 8M-2G | 37.6 |
| Festr (dual Turin) | Default | 32M fixed | 22.2 |
| purplepow3r (7 GPUs, dual Turin) | P2P_LEVEL=SYS | 4G fixed | 41.3-41.7 |

### NCCL Test Commands

```bash
# Basic test (32M message, 8 GPUs)
NCCL_P2P_LEVEL=SYS NCCL_NET_GDR_LEVEL=SYS ./all_reduce_perf -b 32M -g 8 -c 0

# Sweep test with tuned channels
NCCL_NET_GDR_LEVEL=SYS NCCL_MIN_NCHANNELS=8 ./all_reduce_perf -b 8M -e 2G -f 2 -g 8 -n 50

# Large message test
NCCL_P2P_LEVEL=SYS NCCL_IB_DISABLE=1 ./build/all_reduce_perf -b 4G -e 4G -f 2 -g 7 -n 20 -N 100
```

NCCL tests are located at `/usr/src/nccl-tests` in NVIDIA containers.

### Custom Allreduce vs NCCL (Small Messages)

luke's custom allreduce kernel (for PCIe switch topologies) compared to NCCL:

| Size | Custom (us) | NCCL (us) | Winner |
|------|-------------|-----------|--------|
| 256 B | 7.5 | 24.6 | Custom 3.3x |
| 1 KB | 7.5 | 24.1 | Custom 3.2x |
| 8 KB | 9.2 | 24.2 | Custom 2.6x |
| 32 KB | 16.5 | 24.5 | Custom 1.5x |
| 64 KB | 25.9 | 24.1 | NCCL 1.1x |
| 256 KB | 73.6 | 28.0 | NCCL 2.6x |

Custom allreduce wins big for inference-relevant message sizes (<32 KB) but loses at larger sizes. This kernel is only effective on densely-interconnected PCIe switch topologies -- it was actually **slower** than NCCL on dual-CPU systems without switches.

Custom allreduce repo: [github.com/lukealonso/sglang/commits/custom_ar/](https://github.com/lukealonso/sglang/commits/custom_ar/)

---

## BAR1 Configuration

BAR1 (Base Address Register 1) maps GPU memory into the CPU's address space for P2P transfers.

- **Resizable BAR:** Must be **enabled in BIOS** as a critical prerequisite (not just "default on most boards" -- some B650/X670 boards ship with it off, and some server boards default to 256 MB)
- **Expected size on Blackwell (RTX PRO 6000):** **128 GB (131072 MiB)** per GPU. Blackwell maps the full 96 GB VRAM plus additional reserved ranges, so BAR1 is larger than VRAM.
- **Expected size on Ada / older:** 96 GB (98304 MiB) matching VRAM
- **Common issue:** Some BIOS configurations default to 256 MB BAR1, which cripples P2P performance -- always verify after BIOS updates
- **GPU Display Mode:** Set to headless via [NVIDIA Display Mode Selector](https://developer.nvidia.com/displaymodeselector) for largest BAR1 allocation. Required for 16-GPU setups.

Verify BAR1 in `nvidia-smi`:

```
$ nvidia-smi -q | grep -A 1 "BAR1 Memory Usage"
    BAR1 Memory Usage
        Total                             : 131072 MiB  # Blackwell (RTX PRO 6000) = 128 GB
```

Sanity check: if you see `256 MiB` here, Resizable BAR is disabled or BIOS capped BAR1 -- fix in BIOS before running any multi-GPU work.

---

## How PCIe Bandwidth Affects Inference

### Tensor Parallel Decode

During decode (token generation), each TP step requires:
1. **AllReduce** after attention layer (~32-256 KB per layer)
2. **AllReduce** after MoE/FFN layer (~32-256 KB per layer)

For a 397B MoE model with ~60 layers, that is ~120 AllReduce operations per token. At 25 us per AllReduce, that is **3 ms of pure communication overhead per token**, limiting decode to ~333 tok/s even with zero compute time.

### Why Small-Message Latency Dominates

Inference AllReduce messages are small (32-256 KB). At these sizes:
- **Bandwidth** is irrelevant (56 GB/s can transfer 256 KB in 4.5 us)
- **Latency** dominates (NCCL ring setup, protocol negotiation, synchronization)

This is why:
- NCCL's LL (Low Latency) protocol gives 1.5-1.9x speedup
- Custom allreduce kernels that bypass NCCL overhead give 2-3x speedup
- PCIe switch cut-through latency (~100 ns) matters more than raw bandwidth

### P2P vs No-P2P for High Concurrency

For high-concurrency workloads (many simultaneous requests), disabling P2P and routing through DRAM can actually be faster:

| Workload | P2P Enabled | P2P Disabled | Winner |
|----------|-------------|-------------|--------|
| Single batch (low latency) | 90 tok/s | 70 tok/s | P2P |
| 100 concurrent requests | 5000 tok/s | 10000 tok/s | No-P2P |

This is because DRAM routing can use more channels and higher aggregate bandwidth for large batched operations.

---

## GRUB Kernel Parameters for PCIe Stability

### Critical Parameters

Add to `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`:

```bash
pcie_aspm=off pcie_port_pm=off
```

| Parameter | Purpose |
|-----------|---------|
| `pcie_aspm=off` | Disables Active State Power Management on all PCIe links |
| `pcie_port_pm=off` | Disables PCIe port runtime power management. **CRITICAL:** prevents root port from suspending during GPU link retrain Gen1<->Gen5 |

Without `pcie_port_pm=off`, GPU DynamicPowerManagement=3 causes "Surprise Link Down" errors (`aer_uncor_status: 0x00000020`) leading to **system lockups**.

### Additional Recommended Parameters

```bash
# Full recommended GRUB line (Festr's Turin system):
GRUB_CMDLINE_LINUX="rd.auto=1 rd.md=1 rd.md.conf=1 mitigations=off spectre_v2=off spec_store_bypass_disable=off l1tf=off mds=off tsx_async_abort=off srbds=off mmio_stale_data=off retbleed=off amd_iommu=off iommu=off"
```

| Parameter | Purpose |
|-----------|---------|
| `iommu=off` | Prevents NCCL hangs. Without this, NCCL P2P may deadlock. |
| `amd_iommu=off` | AMD-specific IOMMU disable |
| `mitigations=off` | Disable CPU security mitigations for maximum performance |
| `nvme_core.default_ps_max_latency_us=0` | Prevent NVMe power state transitions (stability) |

After editing, run:

```bash
update-grub
reboot
# Verify:
cat /proc/cmdline
```

### Modprobe Configuration

```bash
# /etc/modprobe.d/uvm.conf
options nvidia_uvm uvm_disable_hmm=1
```

Without this, NCCL P2P operations lock up. This is required on virtually all RTX PRO 6000 multi-GPU setups.

### BIOS Settings (PRO WS WRX90E-SAGE SE)

- **Resizable BAR:** Enabled (default)
- **Above 4G Decoding:** Enabled
- **SR-IOV:** Enabled
- **Slot 6 Warning:** On WRX90E-SAGE SE, slot 6 is limited to Gen5 x8 speed

### Filesystem Warning

ZFS caused system freezes for some users on these GPU workloads. **EXT4 is recommended** for the OS filesystem.

---

## Debugging Tools

| Tool | Purpose |
|------|---------|
| `nvidia-smi topo -m` | Display GPU topology matrix |
| `nvidia-smi -q` | Detailed GPU info including BAR1, power, thermals |
| [p2pmark](https://github.com/lukealonso/p2pmark) | P2P bandwidth, latency, and allreduce benchmarks |
| `p2pBandwidthLatencyTest` | CUDA samples P2P test |
| [amd-epyc-gpu-fabric-monitor](https://github.com/voipmonitor/amd-epyc-gpu-fabric-monitor) | Real-time AMD xGMI fabric transfer monitoring |
| `memtest86` | RAM testing |
| Intel MLC | Memory Latency Checker (works on AMD) |
| `rasdaemon` | AER error monitoring |
| `nvitop` / `nvtop` | GPU monitoring |
| PCIe AER counters | Check `aer_dev_correctable` and `aer_dev_fatal` in sysfs |
