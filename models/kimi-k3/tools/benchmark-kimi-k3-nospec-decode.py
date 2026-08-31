#!/usr/bin/env python3
"""Measure Kimi-K3 single-request target-only decode throughput."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _validated_http_url(url: str) -> str:
    """Return an absolute HTTP or HTTPS URL or reject the input."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"expected an HTTP(S) URL, got {url!r}")
    return url


def _read_url(url: str) -> bytes:
    url = _validated_http_url(url)
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        return response.read()


def _discover_model(url: str) -> str:
    payload = json.loads(_read_url(f"{url}/v1/models"))
    models = payload.get("data") or []
    if (
        len(models) != 1
        or not isinstance(models[0], dict)
        or not isinstance(models[0].get("id"), str)
    ):
        raise RuntimeError("--model is required when /v1/models is not singular")
    return models[0]["id"]


def _run(
    *,
    url: str,
    model: str,
    prompt: list[int],
    max_tokens: int,
    allowed_token_id: int,
    output_path: Path,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "allowed_token_ids": [allowed_token_id],
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    started = time.perf_counter()
    token_times: list[float] = []
    output_parts: list[str] = []
    usage: dict[str, int] = {}
    _validated_http_url(request.full_url)
    with urllib.request.urlopen(request, timeout=3600) as response:  # noqa: S310
        for raw_line in response:
            if not raw_line.startswith(b"data: "):
                continue
            data = raw_line[6:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("usage"):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                token_times.append(time.perf_counter())
                output_parts.append(choices[0].get("text") or "")
    ended = time.perf_counter()

    completion_tokens = int(usage.get("completion_tokens", 0))
    if completion_tokens < 2 or len(token_times) < 2:
        raise RuntimeError(
            f"fewer than two streamed tokens: usage={usage}, events={len(token_times)}"
        )
    if len(token_times) != completion_tokens:
        raise RuntimeError(
            "streamed events do not match completion tokens: "
            f"events={len(token_times)}, usage={usage}"
        )
    output = "".join(output_parts)
    output_path.write_text(output, encoding="utf-8")
    decode_seconds = token_times[-1] - token_times[0]
    return {
        "formula": "(completion_tokens - 1) / (last_token_time - first_token_time)",
        "prompt_tokens": int(usage.get("prompt_tokens", len(prompt))),
        "completion_tokens": completion_tokens,
        "allowed_token_id": allowed_token_id,
        "timed_events": len(token_times),
        "ttft_seconds": token_times[0] - started,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (completion_tokens - 1) / decode_seconds,
        "wall_seconds": ended - started,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
        "output_file": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--allowed-token-id", type=int, default=13)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.warmups < 0:
        parser.error("--warmups must not be negative")

    url = _validated_http_url(args.url.rstrip("/"))
    model = args.model or _discover_model(url)
    token_file = args.token_file
    prompt = json.loads(token_file.read_text(encoding="utf-8"))
    prompt = prompt[: args.prompt_tokens]
    if len(prompt) != args.prompt_tokens or not all(
        isinstance(token, int) for token in prompt
    ):
        raise ValueError("token file lacks the requested integer-token prefix")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    measured: list[dict[str, Any]] = []
    for index in range(args.warmups + args.runs):
        is_warmup = index < args.warmups
        run_index = index + 1 if is_warmup else index - args.warmups + 1
        label = f"warmup-{run_index}" if is_warmup else f"run-{run_index}"
        result = _run(
            url=url,
            model=model,
            prompt=prompt,
            max_tokens=args.max_tokens,
            allowed_token_id=args.allowed_token_id,
            output_path=args.output_dir / f"{label}.txt",
        )
        result = {"run": label, **result}
        (args.output_dir / f"{label}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        if not is_warmup:
            measured.append(result)

    rates = [float(result["decode_tokens_per_second"]) for result in measured]
    summary = {
        "artifact_kind": "Kimi-K3 target-only normalized decode benchmark",
        "allowed_token_id": args.allowed_token_id,
        "decode_tokens_per_second_max": max(rates),
        "decode_tokens_per_second_median": statistics.median(rates),
        "decode_tokens_per_second_min": min(rates),
        "max_tokens": args.max_tokens,
        "model": model,
        "output_sha256s": [result["output_sha256"] for result in measured],
        "prompt_tokens": args.prompt_tokens,
        "runs": args.runs,
        "token_file": str(token_file),
        "token_file_sha256": hashlib.sha256(token_file.read_bytes()).hexdigest(),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
