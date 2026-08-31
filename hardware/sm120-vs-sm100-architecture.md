# SM120 vs. SM100 Architecture — 5th Gen Tensor Cores & tcgen05

A statement on what the RTX 6000 Pro's SM120 silicon actually does and does not
have relative to datacenter Blackwell (SM100), and why that distinction shows up
as a *software* problem rather than a hardware one for local inference.

> For the concrete kernel-level feature matrix (TMEM, TCGEN05, WGMMA) and the
> SM120 backend landscape, see [FlashInfer on SM120](../inference-engines/flashinfer.md#sm120-backend-landscape).

## Statement on the SM120 vs. SM100 Architecture (5th Gen Tensor Cores & tcgen05)

To address the speculation regarding SM120 cards missing Blackwell features, here is the technical reality based on kernel-level analysis:

- **The hardware is legitimately great.** Advertising "5th generation Tensor Cores" has real merit because SM120 introduces vital features like NVFP4 MMA. They are a massive leap from previous architectures in the ways that matter for local inference.

- **tcgen05 is not necessary for SM120.** The lack of 2-CTA and TMEM does not limit these cards. Those features are designed for huge HBM arrays and massive scaling. They simply aren't required to fully saturate the compute and memory bandwidth of SM120.

- **The actual bottleneck is the software fork.** SM120 shares plenty of excellent ISA features with SM100, but the hardware differences force a different programming model at the kernel level. This creates a split in the software ecosystem. Major labs prioritize kernel development for their SM100 production environments. For local inference setups, this means optimized SM120 kernels might initially take a back seat in the development queues of major frameworks.

The hardware isn't holding performance back; the temporary hurdle is simply where major developers are allocating their optimization resources.

## See also

- [FlashInfer on SM120](../inference-engines/flashinfer.md) — kernel feature matrix, CUTLASS/FA2 backends, sm120f compilation
- [NVFP4 Quantization](../optimization/nvfp4-quantization.md) — NVFP4 MMA in practice on SM120
- [GPU Configurations](gpu-configs.md) — RTX 6000 Pro (GB202, SM120) specs and rigs
