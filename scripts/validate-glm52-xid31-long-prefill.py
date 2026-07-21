#!/usr/bin/env python3

"""Exercise the long-prefill-to-decode transition that previously hit Xid 31."""

import argparse
import json
import time
import urllib.request
from pathlib import Path


PADDING = (
    "The reference ledger records deterministic identifiers, timestamps, "
    "integer counters, and short explanatory notes for later verification. "
)


def post_json(url: str, payload: dict, timeout: int) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def get_health(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            return response.status == 200
    except Exception:
        return False


def make_exact_prompt(base_url: str, model: str, target_tokens: int) -> tuple[str, int]:
    target_chars = max(1, target_tokens * 6)
    prompt = ""
    token_count = 0
    for _ in range(3):
        repeats = target_chars // len(PADDING) + 1
        prompt = (PADDING * repeats)[:target_chars]
        tokenized = post_json(
            f"{base_url}/tokenize",
            {"model": model, "prompt": prompt},
            timeout=180,
        )
        token_count = int(tokenized["count"])
        if abs(token_count - target_tokens) <= 64:
            break
        target_chars = max(1, round(target_chars * target_tokens / token_count))
    return prompt, token_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--target-tokens", type=int, default=300_000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    if not get_health(base_url):
        raise RuntimeError("server is not healthy before the test")

    short_started = time.monotonic()
    short = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: ready"}],
            "temperature": 0,
            "max_tokens": 16,
        },
        timeout=120,
    )
    short_seconds = time.monotonic() - short_started

    reference, calibrated_tokens = make_exact_prompt(
        base_url, args.model, args.target_tokens
    )
    long_prompt = (
        "Read the following reference, then explain why a custom CUDA allocator "
        "must preserve the memory-layout contract expected by downstream kernels. "
        "Do not quote the reference.\n\n"
        + reference
    )
    long_started = time.monotonic()
    long_result = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": args.model,
            "messages": [{"role": "user", "content": long_prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "ignore_eos": True,
        },
        timeout=900,
    )
    long_seconds = time.monotonic() - long_started
    content = long_result["choices"][0]["message"].get("content") or ""
    result = {
        "server_healthy_after": get_health(base_url),
        "short_elapsed_seconds": short_seconds,
        "short_usage": short.get("usage", {}),
        "calibrated_reference_tokens": calibrated_tokens,
        "requested_decode_tokens": args.max_tokens,
        "long_elapsed_seconds": long_seconds,
        "long_usage": long_result.get("usage", {}),
        "finish_reason": long_result["choices"][0].get("finish_reason"),
        "output_characters": len(content),
        "output_prefix": content[:500],
        "output_suffix": content[-500:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["server_healthy_after"]:
        raise RuntimeError("server became unhealthy during the test")


if __name__ == "__main__":
    main()
