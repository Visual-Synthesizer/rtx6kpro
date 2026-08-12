# Kimi Runbook Hub

Use this page as the stable entry point for Kimi models on RTX PRO 6000
Blackwell. It covers the Kimi-K3 Infernal Invocation TP16/DCP16 runtime,
Kimi-K2.7-Code DFlash, and Kimi-K2.6 regression/debugging references.

## Qualified Runtime Pages

| Need | Page |
|---|---|
| Run Kimi-K3 with a full MXFP4 target, DCP16, and target-only, DSpark, or DFlash decode | [Kimi-K3 TP16/DCP16 runtime](kimi-k3/README.md) |
| Run Kimi-K2.7-Code on the Fathomless runtime | [Kimi-K2.7-Code v3](kimi-k27-code_v3.md) |
| Reproduce the Eldritch Kimi-K2.7-Code recipe | [Kimi-K2.7-Code v2](kimi-k27-code_v2.md) |
| Reproduce Black Benediction Kimi-K2.7-Code | [Kimi-K2.7-Code](kimi-k27-code.md) |
| Research Kimi-K2.6 MTP/DFlash long-context behavior | [Kimi-K2.6 MTP long-context research](kimi-k26-mtp-long-ctx-wip/README.md) |

## Runtime Interfaces

| Area | Guidance |
|---|---|
| Kimi-K3 target | `moonshotai/Kimi-K3` at revision `2496450e92e425c886db095102a52a6682ca3970` |
| Kimi-K3 draft paths | `Inferact/Kimi-K3-DSpark` and `modal-labs/Kimi-K3-DFlash` |
| Kimi-K2.7-Code target | `moonshotai/Kimi-K2.7-Code` |
| Runner | vLLM V2 |
| Common parser setup | `--reasoning-parser kimi_k2`, `--tool-call-parser kimi_k2`, `--enable-auto-tool-choice` |
| DCP | Use only the topology and cache contracts documented by the selected model page. |

## Version Map

| Page | Status | Scope |
|---|---|---|
| [Kimi-K3 TP16/DCP16 runtime](kimi-k3/README.md) | Qualified | Reproducible Infernal Invocation/B12X Docker image and target-only, DSpark, and DFlash profiles. |
| [Kimi-K2.7-Code v3](kimi-k27-code_v3.md) | Qualified | Fathomless Kimi DFlash validation and patch overlay notes. |
| [Kimi-K2.7-Code v2](kimi-k27-code_v2.md) | Research-only | Eldritch Kimi runtime evidence. |
| [Kimi-K2.7-Code](kimi-k27-code.md) | Research-only | Black Benediction recipe. |
| [Kimi-K2.6 v9](kimi-k26-v9.md) | Research-only | Black Benediction DFlash baseline. |
| [Kimi-K2.6 v2-v8](kimi-k26-v2.md) | Research-only | Kimi-K2.6 bring-up and performance/debug evidence. |
| [Kimi-K2.6 Prometheus refresh](kimi-k26-prometheus-benchmark-refresh-2026-04-25.md) | Research-only | Benchmark refresh data. |

## Operational Reminders

- Kimi tool calls depend on the Kimi parser and reasoning parser matching the
  model family.
- DFlash under DCP has stricter metadata and prefix-cache behavior than plain
  decode. If DCP changes, smoke-test DFlash before running a full sweep.
- For bug reports, keep raw streaming deltas when possible; parser failures are
  much easier to diagnose from the SSE stream than from final client output.
