# Kimi-K3 on RTX PRO 6000 Blackwell

The qualified production deployment is documented in
[Official MXFP4 with DSpark, CPU LMCache, vision, and LLMConduit](production-dspark-lmcache.md).
It serves `moonshotai/Kimi-K3` on 16 RTX PRO 6000 Blackwell GPUs through a
source-locked CUDA 13.3 and PyTorch 2.13 image.

## Runtime and Evaluation Documents

| Document | Purpose | Status |
|---|---|---|
| [Production DSpark and LMCache](production-dspark-lmcache.md) | Docker, vLLM, LLMConduit, reasoning controls, native vision, tools, and Oh My Pi | qualified |
| [Source-locked serving receipt](validation/source-locked-runtime-20260816.json) | No-speculation, DSpark, and DFlash source composition and runtime evidence | qualified |
| [Distribution-fidelity reference](distribution-fidelity-1024x2048.md) | Teacher-forced hidden-state and KLD comparison over 1,024 contexts | implemented |
| [AA-LCR reproduction](aa-lcr-reproduction.md) | Reproducible capability comparison protocol | qualified |
| [Official MXFP4 versus QSRT K2](aa-lcr-official-mxfp4-vs-qsrt-k2.md) | Paired AA-LCR comparison | qualified |
| [QSRT K2 TP16/DCP8](qsrt-k2-tp16-dcp8.md) | Target-only and DSpark serving for the QSRT K2 checkpoint | qualified |
| [Red Hat DSpark DCP16](redhat-dspark-dcp16.md) | RedHatAI BF16 draft compatibility | qualified |

Machine-readable receipts are stored under [`validation/`](validation/).
Repository tools used by the evaluation documents are stored under
[`tools/`](tools/).
