# Kimi Runbook Hub

Use this page as the stable entry point for Kimi models on RTX PRO 6000
Blackwell. The Kimi pages include both current Kimi-K2.7-Code DFlash work and
older Kimi-K2.6 regression/debugging notes.

## Current Recommendation

| Need | Page |
|---|---|
| Run Kimi-K2.7-Code on the current Fathomless line | [Kimi-K2.7-Code v3](kimi-k27-code_v3.md) |
| Reproduce the Eldritch Kimi-K2.7-Code recipe | [Kimi-K2.7-Code v2](kimi-k27-code_v2.md) |
| Reproduce Black Benediction Kimi-K2.7-Code | [Kimi-K2.7-Code](kimi-k27-code.md) |
| Debug older Kimi-K2.6 MTP/DFlash long-context behavior | [Kimi-K2.6 MTP long-context WIP](kimi-k26-mtp-long-ctx-wip/README.md) |

## Current Runtime Shape

| Area | Current guidance |
|---|---|
| Current target | `moonshotai/Kimi-K2.7-Code` |
| Draft path | DFlash, documented in the Kimi-K2.7 v3 page |
| Runner | vLLM V2 |
| Common parser setup | `--reasoning-parser kimi_k2`, `--tool-call-parser kimi_k2`, `--enable-auto-tool-choice` |
| DCP | Use only configurations documented in the current Kimi page; older DCP/DFlash behavior had prefix-cache and metadata fixes. |

## Version Map

| Page | Status | Why keep it |
|---|---|---|
| [Kimi-K2.7-Code v3](kimi-k27-code_v3.md) | Current | Fathomless Kimi DFlash validation and patch overlay notes. |
| [Kimi-K2.7-Code v2](kimi-k27-code_v2.md) | Historical | Eldritch Kimi runtime. |
| [Kimi-K2.7-Code](kimi-k27-code.md) | Historical | Black Benediction recipe. |
| [Kimi-K2.6 v9](kimi-k26-v9.md) | Historical | Black Benediction DFlash baseline. |
| [Kimi-K2.6 v2-v8](kimi-k26-v2.md) | Archive | Incremental Kimi bring-up and performance/debug history. |
| [Kimi-K2.6 Prometheus refresh](kimi-k26-prometheus-benchmark-refresh-2026-04-25.md) | Archive | Benchmark refresh data. |

## Operational Reminders

- Kimi tool calls depend on the Kimi parser and reasoning parser matching the
  model family.
- DFlash under DCP has stricter metadata and prefix-cache behavior than plain
  decode. If DCP changes, smoke-test DFlash before running a full sweep.
- For bug reports, keep raw streaming deltas when possible; parser failures are
  much easier to diagnose from the SSE stream than from final client output.
