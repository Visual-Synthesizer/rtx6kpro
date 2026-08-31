# DeepSeek-V4-Flash: empty `content` from an unclosed `<think>` block

**Status:** confirmed model behaviour (not a vLLM parser bug). Reproducible.
**Date:** 2026-06-10/13. **Reporter:** voipmonitor / local-inference-lab.

## TL;DR

Under concurrent load with a cold prefix cache, **DeepSeek-V4-Flash**
occasionally returns an HTTP 200 chat completion with:

- `finish_reason = "stop"`
- a **non-empty `reasoning`** field
- an **empty `content`** field

i.e. the user gets nothing. The model wrote the *entire final answer* —
including its `[1][2]` citation markers — **inside** the thinking block and hit
EOS without ever emitting the special `</think>` token (id `128822`). The
`deepseek_v4` reasoning parser then correctly routes everything into
`reasoning`, leaving `content` empty.

It is **not** a vLLM parser bug: the parser faithfully reports what the model
generated. It belongs to the known family of "model puts the answer in the
reasoning field" issues (vllm#23429, vllm#12999).

Rate: **~0.3–0.8 % of streaming synthesis requests** in our runs. It does **not**
reproduce on sequential replays of an identical request — concurrency and a
cold prefill are required. It only ever hit the **final synthesis** turn
(stream, no tools), never a tool-call round.

## Environment

| | |
|---|---|
| GPU | 2× NVIDIA RTX PRO 6000 Blackwell (SM120), no NVLink, TP2 |
| Driver | 595.58.03 |
| vLLM image | `voipmonitor/vllm:lucifer` (FlashInfer/CUTLASS, `SPARSE_MLA_SM120`) |
| Model | `deepseek-ai/DeepSeek-V4-Flash`, `--max-model-len 512000` |
| Reasoning parser | `deepseek_v4` (`--default-chat-template-kwargs.thinking=true --default-chat-template-kwargs.reasoning_effort=high`) |

Exact serve command (TP2, the config this was observed on):

```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --served-model-name DeepSeek-V4-Flash --trust-remote-code \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 2 \
  --kv-cache-dtype fp8 --block-size 256 \
  --gpu-memory-utilization 0.96 --max-model-len 512000 \
  --max-num-seqs 64 --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 64 \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  --async-scheduling --no-scheduler-reserve-full-isl \
  --enable-chunked-prefill --enable-flashinfer-autotune --enable-prefix-caching \
  --attention-backend SPARSE_MLA_SM120 \
  --kernel-config.moe_backend flashinfer_cutlass \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs.thinking=true \
  --default-chat-template-kwargs.reasoning_effort=high
```

The bug is **independent of the build, of MTP speculative decoding, and of
temperature** — see the cross-check matrix below.

## Symptom (what an affected response looks like)

Non-streaming, the message comes back like this (abbreviated):

```json
{
  "choices": [{
    "finish_reason": "stop",
    "message": {
      "role": "assistant",
      "content": "",
      "reasoning": "...the entire user-facing answer, with [1][2] markers..."
    }
  }],
  "usage": { "completion_tokens": 221 }
}
```

`completion_tokens` is normal (~150–260) — the model did generate a full answer.
It just put all of it in the wrong field.

## Root cause / mechanism

The `deepseek_v4` reasoning parser (a `DeepSeekV3ReasoningParser` that delegates
per request to the R1 parser using `<think>` / `</think>`, token ids
`128821` / `128822`) puts everything **before** `</think>` into `reasoning`. If
the model never emits `</think>`, the answer never reaches `content`.

Two concrete failure shapes seen in forensic dumps:

1. **Most common.** The model reasons normally, then at the think→answer
   boundary samples a plain text separator (e.g. `---`) **instead of** the
   special `</think>` token, and writes the final answer in the block. The tail
   of `reasoning` is a clean, finished answer (citations and all).
2. **Rarer.** The model skips visible reasoning and writes the answer straight
   into the block.

[`sample-hit.json`](sample-hit.json) is a **real production capture** on
`voipmonitor/vllm:lucifer` (TP2), in the exact schema `repro.py` writes to
`./hits/`. Note `content_chars: 0`, `finish: stop`, and that `reasoning_text` is
a complete, citation-bearing answer that should have been the `content`. The
captured text is generic legal information about consumer dispute-resolution
bodies — no personal or customer data.

### Why it is not the parser, not the build, not temperature

- **Parser:** the parser correctly mirrors generated tokens. With no `</think>`
  in the stream there is, by construction, nothing to put in `content`.
- **Logprobs.** Prefilling a real thinking trace through `/v1/completions` and
  inspecting the distribution at the critical point: `</think>` not being
  emitted is a *sampling* event, not a structural one. (At production
  `temperature=0.1`, `P(EOS)` right after `</think>` is ~1e-40; the failure is
  the model choosing a non-`</think>` continuation, not a forced stop.)
- **Cross-check matrix** (200 concurrent L+S pairs per cell, plus a 550-pair
  baseline; "hit" = streamed synthesis with empty content):

  | Config | `--speculative-config` (MTP) | t=0.1 | t=1.0 |
  |---|---|---|---|
  | `voipmonitor/vllm:lucifer` | on | 2 / ~550 | 2 / 200 |
  | `…:black-benediction` (b12x) | on | 0 / 200 | 3 / 200 |
  | `voipmonitor/vllm:lucifer` | **off** | 1 / 200 | 0 / 200 |

  Cumulative: **t=0.1 → 3/950 (~0.3 %)**, **t=1.0 → 5/600 (~0.8 %)**. Higher
  temperature (DeepSeek's officially recommended `1.0`) does **not** reduce it
  and if anything raises it slightly. Disabling MTP does not remove it. So:
  not the kernels, not speculative decoding, not temperature → **a model
  behaviour at the think→answer transition.**

## Reproduce

`repro.py` is **fully self-contained** (synthetic two-version "terms" documents,
ask to compare a dispute-resolution clause — the kind of multi-source synthesis
that triggers it). No private data. It recreates the production conditions:
`thinking=true, reasoning_effort=high`, "cite `[n]` after every paragraph",
concurrent tool-call + synthesis requests, and a unique system-prompt prefix per
request so prefix caching always misses.

```bash
pip install httpx
PB_REPRO_BASE=http://localhost:4999/v1 \
PB_REPRO_MODEL=DeepSeek-V4-Flash \
python3 repro.py 150          # 150 iters/temp => 300 streamed-synthesis requests
```

A hit prints e.g.:

```
[087] t=0.1 EMPTY content! finish=stop reasoning_chars=552 compl_tokens=197 -> ./hits/hit_t0.1_87_...json
```

and the full response is dumped to `./hits/`. Expect to need ~150–400 synthesis
requests to catch the first one. Sequential replays of an identical request will
**not** reproduce — keep the concurrency and the cache-buster.

Knobs: `PB_REPRO_BASE`, `PB_REPRO_MODEL`, `PB_REPRO_OUT`, and the iteration count
(argv[1]).

## Mitigation (application side — this is the durable fix)

Because the model *did* produce a correct answer (it is sitting in `reasoning`),
the client can recover gracefully. What we ship:

1. On `content == ""` **and** `finish == "stop"`: log forensically
   (model, think flag, token counts, reasoning tail) and **retry once at
   `temperature=0.6`** (leave the deterministic bad trajectory).
2. For the streamed synthesis fallback, pass the stuck reasoning text to a
   **non-thinking** re-synthesis as a *draft* ("keep these facts, add `[n]`
   citations") — the chain-of-thought never leaks to the user, but the facts and
   citations are preserved.
3. The non-thinking fallback uses **`min_tokens=16`** (vLLM forbids EOS for the
   first 16 tokens) so a chat-mode fallback can never itself return empty.
4. If everything still yields empty → return a short apology, never silence.

Application-side handling is treated as the permanent solution; it does not
depend on a server-side fix.

## Related upstream

- vllm#23429 — content ending up in the reasoning field
- vllm#12999 — reasoning/content separation edge cases

If you want to file this upstream, attach a `./hits/*.json` dump from `repro.py`
(it is synthetic and safe to share) showing `content == ""`, `finish == "stop"`,
and a complete answer at the end of `reasoning`.
