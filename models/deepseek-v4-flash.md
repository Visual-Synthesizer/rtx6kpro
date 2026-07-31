# DeepSeek-V4-Flash Runbook Hub

Use this page as the stable entry point for DeepSeek-V4-Flash and DSpark on RTX
PRO 6000 Blackwell. The versioned pages contain exact measurements and are
intentionally preserved.

## Current Recommendation

| Need | Page |
|---|---|
| Start the current 0731 DSpark checkpoint | [DeepSeek-V4-Flash-0731 Gilded Gnosis r16](ds4dspark-v20.md) |
| Previous 0731 DSpark release | [DeepSeek-V4-Flash-0731 Gilded Gnosis r15](ds4dspark-v20-r15.md) |
| Historical standard-MTP and DSpark deployment | [DeepSeek-V4-Flash v10 Fathomless Validation](ds4dspark-v10.md) |
| Full DSpark and standard-MTP reference sweep | [DeepSeek-V4-Flash and DSpark v9](ds4dspark-v9.md) |
| Eldritch standard checkpoint reference | [DeepSeek-V4-Flash v6](ds4-flash-v6.md) |
| Tool-call empty-think failure mode | [DS4 empty think troubleshooting](ds4f-empty-think/README.md) |

## Current Runtime Shape

| Area | Current guidance |
|---|---|
| Current image line | Gilded Gnosis r16 for the 0731 checkpoint; Fathomless and Eldritch pages remain historical baselines |
| Current DSpark checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Standard checkpoint | `deepseek-ai/DeepSeek-V4-Flash` |
| Historical DSpark checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Recommended 0731 TP2 defaults | fixed K5, `MAX_NUM_SEQS=16`, `MAX_BATCHED=8192`, `GPU_MEM=0.975`, InstantTensor BUFFERED |
| Native CPU KV offload | Opt in with `KV_OFFLOADING_SIZE=<total GiB>` |
| Backend families | SparkInfer/B12X; historical pages also cover Lucifer and CUTLASS |
| Spec decode | Standard MTP, the old DSpark checkpoint, and the 0731 DSpark checkpoint are separate serving contracts |

## Version Map

| Page | Status | Why keep it |
|---|---|---|
| [DS4 DSpark Gilded Gnosis r16](ds4dspark-v20.md) | Current release | 0731 checkpoint, fixed K5, InstantTensor, and native CPU KV offload. |
| [DS4 DSpark Gilded Gnosis r15](ds4dspark-v20-r15.md) | Previous Gilded Gnosis release | Pinned 0731 K7 helper, image, and regression canary. |
| [DS4 DSpark v10](ds4dspark-v10.md) | Historical Fathomless entry | Full TP2/TP4 standard-MTP and DSpark sweep. |
| [DS4 DSpark v9](ds4dspark-v9.md) | Full reference | Full standard-MTP and DSpark sweep on the prior line. |
| [DS4 Flash v6](ds4-flash-v6.md) | Historical Eldritch reference | Shared Eldritch image, B12X/Lucifer variants, TP2/TP4 backend comparisons. |
| [DS4 Flash v5](ds4-flash-v5.md) | Historical fix checkpoint | Eldritch DS4Fix validation. |
| [DS4 Flash v4](ds4-flash-v4.md) | Historical Chthonic reference | Useful for old B12X performance comparisons. |
| [DS4 Flash v3](ds4-flash-v3.md) | Historical Lucifer/CUTLASS reference | Older Lucifer cutlass recipe and comparison baseline. |
| [DS4 Flash v1-v2](ds4-flash-v1.md), [v2](ds4-flash-v2.md) | Archive | Early bring-up and PR comparison data. |

## Operational Reminders

- Standard MTP and DSpark are not interchangeable. DSpark uses a different
  checkpoint and graph envelope.
- Always confirm backend markers in logs before trusting a speed number.
- Keep `NCCL_GRAPH_FILE` unset unless a real XML graph file is provided.
- Reuse cache directories while iterating; TileLang/CuTe compile time can
  dominate startup.
