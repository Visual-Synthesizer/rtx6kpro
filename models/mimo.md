# Xiaomi MiMo Runbook Hub

Use this page as the stable entry point for Xiaomi MiMo V2.5 Pro work on RTX
PRO 6000 Blackwell.

## Current Recommendation

| Need | Page |
|---|---|
| Run MiMo V2.5 Pro FP4-DFlash on vLLM | [Xiaomi MiMo V2.5 Pro FP4-DFlash v3](xiaomi-mimo-v2.5-pro-fp4-dflash_v3.md) |
| Reproduce the previous vLLM DFlash page | [MiMo FP4-DFlash v2](xiaomi-mimo-v2.5-pro-fp4-dflash_v2.md) |
| Understand the older SGLang/B12X route | [MiMo V2.5 Pro hub](mimo-v25-pro/README.md) |
| Reproduce quantization details | [MiMo quantization](mimo-v25-pro/quantization.md) |
| Reproduce SGLang launch details | [MiMo SGLang running notes](mimo-v25-pro/running.md) |

## Current Runtime Shape

| Area | Current guidance |
|---|---|
| Current model | `XiaomiMiMo/MiMo-V2.5-Pro-FP4-DFlash` |
| Current runtime | vLLM V2 with DFlash |
| Attention | `TRITON_ATTN` for the documented v3 path |
| MoE | FlashInfer CUTLASS path in the documented launch |
| Long context | Do not set `--max-model-len` unless the page explicitly says to; the checkpoint advertises its own long context length. |

## Version Map

| Page | Status | Why keep it |
|---|---|---|
| [MiMo FP4-DFlash v3](xiaomi-mimo-v2.5-pro-fp4-dflash_v3.md) | Current | Fathomless validation and seq-mask fix notes. |
| [MiMo FP4-DFlash v2](xiaomi-mimo-v2.5-pro-fp4-dflash_v2.md) | Historical | Earlier vLLM DFlash runbook. |
| [MiMo FP4-DFlash](xiaomi-mimo-v2.5-pro-fp4-dflash.md) | Historical | Initial MiMo DFlash page. |
| [MiMo V2.5 Pro SGLang/B12X hub](mimo-v25-pro/README.md) | Archive | Older SGLang and B12X integration path. |

## Operational Reminders

- Check for the fast DFlash marker in logs. A `diffkv` attention marker usually
  means the wrong path is selected for the current MiMo fast recipe.
- MiMo has historically been sensitive to attention-mask argument ordering, so
  smoke-test short decode and acceptance before running a long benchmark.
