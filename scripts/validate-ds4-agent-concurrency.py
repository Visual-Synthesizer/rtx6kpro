#!/usr/bin/env python3

"""Validate concurrent long-context DS4 agent responses.

The validator starts two streaming chat requests with distinct deterministic
documents and an OpenAI-compatible tool schema.  The shorter request should
enter decode while the longer request is still in chunked prefill.  Raw SSE,
token IDs, parsed content, timing, and text-integrity indicators are retained
for diagnosing request-state or batched-kernel corruption.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_project_file",
            "description": "Read a UTF-8 project file by repository-relative path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line_start": {"type": "integer", "minimum": 1},
                    "line_end": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_project",
            "description": "Search project text for a literal pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"},
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_project_tests",
            "description": "Run a named, preconfigured project test target.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                },
                "required": ["target"],
                "additionalProperties": False,
            },
        },
    },
]

RAW_TOKEN_PATTERNS = (
    re.compile(r"token_id:\d+"),
    re.compile(r"<\|[^>\n]{1,80}\|>"),
    re.compile(r"</?｜[^>\n]{1,120}>"),
    re.compile(r"(?:^|\s)\d{5,}(?:\s+\d{5,}){3,}(?:$|\s)"),
)

CJK_BURST_THRESHOLD = 4
CJK_CHARACTER_THRESHOLD = 16
CJK_FRACTION_THRESHOLD = 0.002


@dataclass(frozen=True)
class RequestSpec:
    identity: str
    target_tokens: int
    expected_marker: str
    forbidden_marker: str


def tools_for(strict: bool) -> list[dict[str, Any]]:
    tools = json.loads(json.dumps(TOOLS))
    if strict:
        for tool in tools:
            tool["function"]["strict"] = True
    return tools


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def server_is_healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def make_document(identity: str, character_count: int) -> str:
    records: list[str] = []
    current = 0
    index = 0
    while current < character_count:
        digest = hashlib.sha256(f"{identity}:{index}".encode()).hexdigest()
        record = (
            f"{identity} record {index:07d}; checksum {digest}; "
            f"component module_{digest[:12]}; invariant {digest[12:28]}; "
            "the implementation must preserve request ownership, tensor shape, "
            "stream order, and deterministic result attribution.\n"
        )
        records.append(record)
        current += len(record)
        index += 1
    return "".join(records)[:character_count]


def build_messages(spec: RequestSpec, document: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are auditing one software project. Answer in English. Keep "
                "the two request identities separate. Tool calls are optional."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Request identity: {spec.expected_marker}. The document below "
                "belongs only to this request. Analyze its concurrency invariants "
                "and produce a detailed technical report. Begin the final answer "
                f"with exactly '{spec.expected_marker} REPORT'. Do not mention "
                f"any other request identity.\n\n{document}"
            ),
        },
    ]


def tokenize_chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
) -> int:
    result = post_json(
        f"{base_url}/tokenize",
        {"model": model, "messages": messages, "tools": tools},
        timeout=600,
    )
    return int(result["count"])


def calibrate_messages(
    base_url: str,
    model: str,
    spec: RequestSpec,
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], int]:
    character_count = max(1, spec.target_tokens * 4)
    messages: list[dict[str, str]] = []
    measured = 0
    for _ in range(4):
        messages = build_messages(spec, make_document(spec.identity, character_count))
        measured = tokenize_chat(base_url, model, messages, tools)
        if abs(measured - spec.target_tokens) <= 128:
            break
        character_count = max(
            1,
            round(character_count * spec.target_tokens / max(1, measured)),
        )
    return messages, measured


def is_cjk(character: str) -> bool:
    return any(
        lower <= character <= upper
        for lower, upper in (
            ("\u3040", "\u30ff"),
            ("\u3400", "\u4dbf"),
            ("\u4e00", "\u9fff"),
            ("\uac00", "\ud7af"),
            ("\uf900", "\ufaff"),
        )
    )


def count_text_indicators(text: str, forbidden_marker: str) -> dict[str, Any]:
    printable = sum(
        character.isprintable() or character in "\n\r\t" for character in text
    )
    non_ascii = sum(ord(character) > 127 for character in text)
    cjk = 0
    cjk_run = 0
    max_cjk_run = 0
    for character in text:
        if is_cjk(character):
            cjk += 1
            cjk_run += 1
            max_cjk_run = max(max_cjk_run, cjk_run)
        else:
            cjk_run = 0
    return {
        "characters": len(text),
        "replacement_characters": text.count("\ufffd"),
        "non_printable_characters": len(text) - printable,
        "non_ascii_fraction": non_ascii / max(1, len(text)),
        "cjk_characters": cjk,
        "cjk_fraction": cjk / max(1, len(text)),
        "max_cjk_run": max_cjk_run,
        "forbidden_marker_count": text.count(forbidden_marker),
        "raw_token_pattern_counts": {
            pattern.pattern: len(pattern.findall(text))
            for pattern in RAW_TOKEN_PATTERNS
        },
    }


def integrity_violations(result: dict[str, Any]) -> list[str]:
    """Return evidence of response corruption or cross-request attribution."""
    violations: list[str] = []
    for stream_name in ("content", "reasoning"):
        indicators = result[f"{stream_name}_indicators"]
        for key in (
            "replacement_characters",
            "non_printable_characters",
            "forbidden_marker_count",
        ):
            if indicators[key]:
                violations.append(f"{stream_name}.{key}={indicators[key]}")
        for pattern, count in indicators["raw_token_pattern_counts"].items():
            if count:
                violations.append(f"{stream_name}.raw_token[{pattern}]={count}")
        if indicators["max_cjk_run"] >= CJK_BURST_THRESHOLD:
            violations.append(f"{stream_name}.max_cjk_run={indicators['max_cjk_run']}")
        elif (
            indicators["cjk_characters"] >= CJK_CHARACTER_THRESHOLD
            and indicators["cjk_fraction"] >= CJK_FRACTION_THRESHOLD
        ):
            violations.append(
                f"{stream_name}.cjk_fraction={indicators['cjk_fraction']:.6f}"
            )
    if not result["content_prefix"] and not result["tool_call_delta_count"]:
        violations.append("response contains neither content nor a tool call")
    return violations


def run_stream(
    *,
    base_url: str,
    model: str,
    spec: RequestSpec,
    messages: list[dict[str, str]],
    measured_prompt_tokens: int,
    max_tokens: int,
    tools: list[dict[str, Any]],
    return_token_ids: bool,
    ignore_eos: bool,
    start_barrier: threading.Barrier,
    stagger_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    start_barrier.wait()
    if stagger_seconds:
        time.sleep(stagger_seconds)
    payload = {
        "model": model,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
        "include_reasoning": True,
        "request_id": f"ds4-long-context-{spec.identity.lower()}",
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if return_token_ids:
        payload["return_token_ids"] = True
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    raw_path = output_dir / f"{spec.identity.lower()}.sse"
    started = time.monotonic()
    first_chunk_at: float | None = None
    first_content_at: float | None = None
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    token_ids: list[int] = []
    tool_calls: list[dict[str, Any]] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    chunks = 0
    with (
        urllib.request.urlopen(request, timeout=3600) as response,
        raw_path.open("wb") as raw_output,
    ):
        while True:
            line = response.readline()
            if not line:
                break
            raw_output.write(line)
            raw_output.flush()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            chunk = json.loads(data)
            chunks += 1
            now = time.monotonic()
            if first_chunk_at is None:
                first_chunk_at = now
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                reasoning = delta.get("reasoning") or ""
                if content and first_content_at is None:
                    first_content_at = now
                content_parts.append(content)
                reasoning_parts.append(reasoning)
                token_ids.extend(choice.get("token_ids") or [])
                tool_calls.extend(delta.get("tool_calls") or [])
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
    finished = time.monotonic()
    content = "".join(content_parts)
    reasoning = "".join(reasoning_parts)
    parsed = {
        "identity": spec.identity,
        "requested_prompt_tokens": spec.target_tokens,
        "measured_prompt_tokens": measured_prompt_tokens,
        "requested_output_tokens": max_tokens,
        "elapsed_seconds": finished - started,
        "time_to_first_chunk_seconds": (
            None if first_chunk_at is None else first_chunk_at - started
        ),
        "time_to_first_content_seconds": (
            None if first_content_at is None else first_content_at - started
        ),
        "stream_chunks": chunks,
        "returned_token_ids": len(token_ids),
        "finish_reason": finish_reason,
        "usage": usage,
        "tool_call_delta_count": len(tool_calls),
        "content_prefix": content[:1000],
        "content_suffix": content[-1000:],
        "reasoning_prefix": reasoning[:1000],
        "content_indicators": count_text_indicators(content, spec.forbidden_marker),
        "reasoning_indicators": count_text_indicators(reasoning, spec.forbidden_marker),
    }
    parsed["integrity_violations"] = integrity_violations(parsed)
    (output_dir / f"{spec.identity.lower()}.json").write_text(
        json.dumps(parsed, indent=2, sort_keys=True) + "\n"
    )
    (output_dir / f"{spec.identity.lower()}.content.txt").write_text(content)
    (output_dir / f"{spec.identity.lower()}.reasoning.txt").write_text(reasoning)
    (output_dir / f"{spec.identity.lower()}.token_ids.json").write_text(
        json.dumps(token_ids) + "\n"
    )
    return parsed


def run_short_control(base_url: str, model: str) -> dict[str, Any]:
    started = time.monotonic()
    result = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly: DS4 SHORT CONTROL READY",
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
        },
        timeout=300,
    )
    content = result["choices"][0]["message"].get("content") or ""
    return {
        "elapsed_seconds": time.monotonic() - started,
        "content": content,
        "usage": result.get("usage", {}),
    }


def run_agent_control(
    base_url: str,
    model: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    result = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Inspect a project conceptually. Explain in two sentences "
                        "why request state must not be shared between concurrent "
                        "agents. Do not call a tool."
                    ),
                }
            ],
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 256,
        },
        timeout=300,
    )
    message = result["choices"][0]["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning") or ""
    return {
        "elapsed_seconds": time.monotonic() - started,
        "content": content,
        "reasoning_prefix": reasoning[:1000],
        "tool_calls": message.get("tool_calls") or [],
        "usage": result.get("usage", {}),
        "content_indicators": count_text_indicators(content, "IMPOSSIBLE-MARKER"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5500")
    parser.add_argument("--model", default="DeepSeek-V4-Flash-0731")
    parser.add_argument("--short-context", type=int, default=150_000)
    parser.add_argument("--long-context", type=int, default=300_000)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--stagger-seconds", type=float, default=1.0)
    parser.add_argument("--return-token-ids", action="store_true")
    parser.add_argument("--strict-tools", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not server_is_healthy(args.base_url):
        raise RuntimeError(f"server is not healthy at {args.base_url}")
    args.output.mkdir(parents=True, exist_ok=True)
    tools = tools_for(args.strict_tools)
    short_control = run_short_control(args.base_url, args.model)
    agent_control = run_agent_control(args.base_url, args.model, tools)

    specs = (
        RequestSpec("AGENT_ALPHA", args.short_context, "ALPHA-731", "BETA-731"),
        RequestSpec("AGENT_BETA", args.long_context, "BETA-731", "ALPHA-731"),
    )
    calibrated = [
        (*calibrate_messages(args.base_url, args.model, spec, tools), spec)
        for spec in specs
    ]
    barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(
                run_stream,
                base_url=args.base_url,
                model=args.model,
                spec=spec,
                messages=messages,
                measured_prompt_tokens=measured,
                max_tokens=args.max_tokens,
                tools=tools,
                return_token_ids=args.return_token_ids,
                ignore_eos=args.ignore_eos,
                start_barrier=barrier,
                stagger_seconds=(0.0 if index == 0 else args.stagger_seconds),
                output_dir=args.output,
            )
            for index, (messages, measured, spec) in enumerate(calibrated)
        ]
        results = [future.result() for future in futures]

    summary = {
        "base_url": args.base_url,
        "model": args.model,
        "strict_tools": args.strict_tools,
        "ignore_eos": args.ignore_eos,
        "short_control": short_control,
        "agent_control": agent_control,
        "requests": results,
        "server_healthy_after": server_is_healthy(args.base_url),
    }
    summary["integrity_violations"] = {
        result["identity"]: result["integrity_violations"]
        for result in results
        if result["integrity_violations"]
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["server_healthy_after"]:
        raise RuntimeError("server became unhealthy during concurrent requests")
    if summary["integrity_violations"]:
        raise RuntimeError(
            "concurrent responses failed integrity checks: "
            + json.dumps(summary["integrity_violations"], sort_keys=True)
        )


if __name__ == "__main__":
    main()
