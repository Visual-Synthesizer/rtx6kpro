#!/usr/bin/env python3
"""Reproduce: DeepSeek-V4-Flash returns HTTP 200 with finish_reason=stop,
a non-empty `reasoning` field and an EMPTY `content` field.

Root cause (see README.md): on the think->answer transition the model
occasionally fails to emit the special `</think>` token (id 128822). It writes
the *entire final answer* — citation markers and all — inside the thinking
block and stops on EOS. The reasoning parser then faithfully routes everything
into `reasoning`, leaving `content` empty. This is a MODEL behaviour, not a
vLLM parser bug.

This harness is fully self-contained: it uses a synthetic multi-source
synthesis task (two fictional "terms & conditions" versions, ask to compare a
clause). No private data. It mirrors the exact conditions that trigger the bug
in production:

  * chat_template_kwargs = {thinking: true, reasoning_effort: high}
  * a synthesis task with "cite [n] after every paragraph" instructions
  * concurrent load: each iteration fires a tool-call round (L) and a streamed
    synthesis (S) at the same time
  * a cold prefix cache: a unique session id is prepended to the system prompt
    so prefix caching never serves the prefill

A "hit" is a streamed synthesis whose content is empty while reasoning is not.
Hits are dumped to ./hits/.

Usage:
  PB_REPRO_BASE=http://localhost:4999/v1 python3 repro.py [iters_per_temp]

  iters_per_temp defaults to 150. Each iteration fires one L+S pair at
  temperature 0.1 and one at 1.0 => 2 streamed-synthesis requests per
  iteration (plus 2 concurrent tool-call requests).

Expected (our runs on voipmonitor/vllm:lucifer, 2x RTX 6000 Pro):
  empty-content rate ~0.3-0.8% of streamed synthesis requests; you typically
  need 150-400 synthesis requests to catch the first hit. Sequential replays of
  an identical request do NOT reproduce — concurrency + cold cache matter.
"""

import asyncio
import json
import os
import pathlib
import sys
import time
import uuid

import httpx

BASE = os.environ.get("PB_REPRO_BASE", "http://localhost:4999/v1")
MODEL = os.environ.get("PB_REPRO_MODEL", "DeepSeek-V4-Flash")
OUT = pathlib.Path(os.environ.get("PB_REPRO_OUT", "./hits"))
OUT.mkdir(exist_ok=True)

# ---- synthetic synthesis task (no private data) -----------------------------
# Two fictional versions of a company's terms, long enough to mirror real
# retrieval context. The task is to compare a clause across both versions —
# exactly the multi-source synthesis shape that triggers the bug in production.

_FILLER = (
    "Poskytovatel je oprávněn jednostranně měnit tyto podmínky; o změně informuje "
    "klienta nejméně 30 dní předem trvalým nosičem dat. Klient je povinen udržovat "
    "aktuální kontaktní údaje. Veškerá oznámení se považují za doručená pátým dnem "
    "po odeslání. Smluvní vztah se řídí právním řádem České republiky. "
)

SOURCE_2024 = (
    "VŠEOBECNÉ OBCHODNÍ PODMÍNKY — verze účinná od 1. 7. 2024\n\n"
    + _FILLER * 6
    + "\nČl. 21 Řešení sporů: 1) Spory rozhodují obecné soudy České republiky. "
    "2) Jako spotřebitel můžete spor řešit mimosoudně. K mimosoudnímu řešení "
    "spotřebitelských sporů je příslušná Česká obchodní inspekce (coi.cz) a v "
    "oblasti platebních služeb Finanční arbitr (finarbitr.cz). 3) U smluv "
    "uzavřených on-line lze využít on-line platformu EU pro řešení sporů "
    "(ec.europa.eu/consumers/odr).\n\n"
    + _FILLER * 4
)

SOURCE_2025 = (
    "VŠEOBECNÉ OBCHODNÍ PODMÍNKY — verze účinná od 1. 4. 2025\n\n"
    + _FILLER * 6
    + "\nČl. 19 Řešení sporů: 1) Spory rozhodují obecné soudy České republiky. "
    "2) Mimosoudní řešení spotřebitelských sporů zajišťuje Finanční arbitr "
    "(finarbitr.cz). 3) Dohledovým orgánem je Česká národní banka (cnb.cz).\n\n"
    + _FILLER * 4
)

SYSTEM = (
    "Jsi asistent pro analýzu firemních dokumentů. Odpovídáš VÝHRADNĚ na základě "
    "poskytnutých zdrojů v blocích <source>.\n\nPRAVIDLA:\n"
    "1. Každý odstavec odpovědi ukonči citačními značkami [n] podle zdroje.\n"
    "2. Pokud si zdroje protiřečí (různé verze), uveď obě varianty s rozdílem.\n"
    "3. Nic nedomýšlej; co ve zdrojích není, neuváděj.\n"
)

USER = (
    f'ZDROJE:\n<source n="1" name="VOP_2024.txt">\n{SOURCE_2024}\n</source>\n\n'
    f'<source n="2" name="VOP_2025.txt">\n{SOURCE_2025}\n</source>\n\n'
    "OTÁZKA: S kým a jak mohu jako spotřebitel řešit spor mimosoudní cestou? "
    "Porovnej obě verze podmínek."
)

# A tiny tool spec so the concurrent "loop" request looks like an agentic round.
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search",
        "description": "Hybrid search in the enabled sources.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}]

THINK_KW = {"thinking": True, "reasoning_effort": "high"}


def bust(messages):
    """Unique system-prompt prefix => cold prefill (prefix cache miss)."""
    m = json.loads(json.dumps(messages))
    m[0]["content"] = f"[session {uuid.uuid4().hex}]\n" + m[0]["content"]
    return m


def synth_body(temp):
    return {
        "model": MODEL,
        "messages": bust([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER}]),
        "max_tokens": 19000,
        "temperature": temp,
        "chat_template_kwargs": THINK_KW,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def loop_body(temp):
    return {
        "model": MODEL,
        "messages": bust([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": USER}]),
        "max_tokens": 17500,
        "temperature": temp,
        "chat_template_kwargs": THINK_KW,
        "tools": TOOLS,
        "tool_choice": "auto",
    }


async def run_loop(client, body):
    r = await client.post(f"{BASE}/chat/completions", json=body)
    r.raise_for_status()
    data = r.json()
    ch = data["choices"][0]
    msg = ch["message"]
    content = (msg.get("content") or "").strip()
    tcs = msg.get("tool_calls") or []
    return {"empty": not content and not tcs,
            "finish": ch.get("finish_reason"),
            "reasoning": len(msg.get("reasoning") or msg.get("reasoning_content") or ""),
            "content": len(content),
            "compl": data["usage"]["completion_tokens"]}


async def run_synth(client, body):
    content, reasoning, finish, usage = "", "", None, {}
    async with client.stream("POST", f"{BASE}/chat/completions", json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                break
            ev = json.loads(payload)
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices") or []:
                d = c.get("delta") or {}
                reasoning += d.get("reasoning") or d.get("reasoning_content") or ""
                content += d.get("content") or ""
                if c.get("finish_reason"):
                    finish = c["finish_reason"]
    return {"empty": not content.strip(), "finish": finish,
            "content_chars": len(content.strip()), "reasoning_chars": len(reasoning),
            "compl_tokens": usage.get("completion_tokens"),
            "reasoning_text": reasoning}


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10, read=900, write=60, pool=10))
    hits = {0.1: 0, 1.0: 0}
    done = {0.1: 0, 1.0: 0}
    print(f"base={BASE} model={MODEL} iters/temp={n} "
          f"(=> {n*2} streamed-synthesis requests)\n", flush=True)
    for i in range(n):
        for temp in (0.1, 1.0):
            loop_res, synth_res = await asyncio.gather(
                run_loop(client, loop_body(temp)),
                run_synth(client, synth_body(temp)),
            )
            done[temp] += 1
            if synth_res["empty"]:
                hits[temp] += 1
                p = OUT / f"hit_t{temp}_{i}_{int(time.time())}.json"
                p.write_text(json.dumps(synth_res, ensure_ascii=False, indent=2))
                print(f"[{i:03d}] t={temp} EMPTY content! finish={synth_res['finish']} "
                      f"reasoning_chars={synth_res['reasoning_chars']} "
                      f"compl_tokens={synth_res['compl_tokens']} -> {p}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"--- {i+1}/{n}: t=0.1 {hits[0.1]}/{done[0.1]} | "
                  f"t=1.0 {hits[1.0]}/{done[1.0]}", flush=True)
    await client.aclose()
    total_req = (done[0.1] + done[1.0])
    total_hit = (hits[0.1] + hits[1.0])
    print(f"\nRESULT: t=0.1 -> {hits[0.1]}/{done[0.1]} ; "
          f"t=1.0 -> {hits[1.0]}/{done[1.0]} ; "
          f"overall {total_hit}/{total_req} "
          f"({100*total_hit/max(1,total_req):.2f}% of synthesis requests)")


if __name__ == "__main__":
    asyncio.run(main())
