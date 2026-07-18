# Legacy Model Runbooks

These pages are not the current recommended entry points, but they are useful
for reproducing older measurements, topology experiments, and model-specific
debugging history.

## Legacy And Secondary Models

| Model | Page | Notes |
|---|---|---|
| DeepSeek-V4-Pro | [DeepSeek-V4-Pro TP16 Lucifer](deepseek-v4-pro-tp16-lucifer.md) | 16-GPU Lucifer reference. |
| GLM-4.7 | [GLM-4.7](glm47.md) | Parser/runtime reference for GLM-family behavior. |
| GLM-5 root page | [GLM-5](glm5.md) | Earlier GLM-5 page before GLM-5.1/5.2 split. |
| Kimi-K2.5 | [Kimi-K2.5](kimi-k25.md) | Older Kimi generation. |
| MiniMax M2.5 | [MiniMax M2.5](minimax-m25.md) | Setup and benchmark notes. |
| Qwen3.5 27B/35B/122B | [Qwen3.5 smaller variants](qwen35-27b.md) | Smaller Qwen runbooks. |
| Qwen3.5 397B | [Qwen3.5-397B](qwen35-397b.md) | Large Qwen runbook. |

## How To Use This Page

- Prefer the current model hub pages for active deployment work.
- Use legacy pages when you need an exact old Docker tag, an old branch pin, or
  a regression comparison.
- If a legacy result is copied into a current page, keep the original page link
  next to the copied number so the launch context remains auditable.
