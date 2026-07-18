# DeepSeek-V4-Flash Runbook Hub

Use this page as the stable entry point for DeepSeek-V4-Flash and
DeepSeek-V4-Flash-DSpark on RTX PRO 6000 Blackwell. The versioned pages contain
exact measurements and are intentionally preserved.

## Current Recommendation

| Need | Page |
|---|---|
| Start TP2 standard or DSpark quickly | [DeepSeek-V4-Flash v10 Fathomless Validation](ds4dspark-v10.md) |
| Full DSpark and standard-MTP reference sweep | [DeepSeek-V4-Flash and DSpark v9](ds4dspark-v9.md) |
| Eldritch standard checkpoint reference | [DeepSeek-V4-Flash v6](ds4-flash-v6.md) |
| Tool-call empty-think failure mode | [DS4 empty think troubleshooting](ds4f-empty-think/README.md) |

## Current Runtime Shape

| Area | Current guidance |
|---|---|
| Current image line | Fathomless Firmament for v10; Eldritch pages remain historical baselines |
| Standard checkpoint | `deepseek-ai/DeepSeek-V4-Flash` |
| DSpark checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark` |
| Recommended TP2 defaults | `MAX_NUM_SEQS=16`, graph = `16 * (1 + speculative_tokens)`, `MAX_BATCHED=4096`, `GPU_MEM=0.95`, retention `4096` |
| Backend families | B12X, Lucifer default, Lucifer CUTLASS |
| Spec decode | Standard MTP2/MTP3 and native DSpark checkpoint paths are separate modes |

## Version Map

| Page | Status | Why keep it |
|---|---|---|
| [DS4 DSpark v10](ds4dspark-v10.md) | Current Fathomless entry | Recommended TP2 launch defaults and reduced validation. |
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
