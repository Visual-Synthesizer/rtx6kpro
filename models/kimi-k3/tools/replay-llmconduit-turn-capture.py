#!/usr/bin/env python3
"""Replay one captured LLMConduit request and inspect its streamed payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import requests


CONTROL_MARKERS = ("<|open|>", "<|close|>", "<|sep|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8003/v1/chat/completions")
    parser.add_argument("--max-completion-tokens", type=int, default=None)
    parser.add_argument(
        "--remove-redacted-images",
        action="store_true",
        help="Deprecated compatibility flag; redacted image items are removed by default.",
    )
    parser.add_argument(
        "--replacement-image-url",
        action="append",
        default=[],
        help=(
            "Replace one captured <redacted uri> in encounter order. Repeat the "
            "option for captures containing multiple images."
        ),
    )
    parser.add_argument(
        "--fail-on-redacted-images",
        action="store_true",
        help="Fail locally instead of removing a redacted image without a replacement.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def redacted_image_url(item: Any) -> bool:
    if not isinstance(item, dict) or item.get("type") not in {
        "image_url",
        "input_image",
    }:
        return False
    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        return image_url.get("url") == "<redacted uri>"
    return image_url == "<redacted uri>"


def set_image_url(item: dict[str, Any], replacement: str) -> dict[str, Any]:
    item = dict(item)
    image_url = item.get("image_url")
    if isinstance(image_url, dict):
        image_url = dict(image_url)
        image_url["url"] = replacement
        item["image_url"] = image_url
    else:
        item["image_url"] = replacement
    return item


def make_capture_replayable(
    value: Any,
    replacements: list[str],
    fail_on_missing: bool,
) -> tuple[Any, int, int]:
    """Replace or remove image placeholders that cannot be sent to an API."""
    replacement_index = 0
    removed = 0
    replaced = 0

    def visit(node: Any) -> Any:
        nonlocal replacement_index, removed, replaced
        if isinstance(node, list):
            retained = []
            for item in node:
                if redacted_image_url(item):
                    if replacement_index < len(replacements):
                        retained.append(
                            set_image_url(item, replacements[replacement_index])
                        )
                        replacement_index += 1
                        replaced += 1
                    elif fail_on_missing:
                        raise ValueError(
                            "capture contains a redacted image without a replacement"
                        )
                    else:
                        removed += 1
                    continue
                retained.append(visit(item))
            return retained
        if isinstance(node, dict):
            return {key: visit(item) for key, item in node.items()}
        return node

    replayable = visit(value)
    if replacement_index != len(replacements):
        raise ValueError(
            f"received {len(replacements)} replacement image URLs but used "
            f"{replacement_index}"
        )
    return replayable, removed, replaced


def main() -> None:
    args = parse_args()
    capture = json.loads(args.capture.read_text(encoding="utf-8"))
    payload = capture["sections"]["inbound_request"]["content"]
    if not isinstance(payload, dict):
        raise ValueError("captured inbound request is not a JSON object")
    payload, removed_images, replaced_images = make_capture_replayable(
        payload,
        args.replacement_image_url,
        args.fail_on_redacted_images,
    )
    payload["stream"] = True
    payload.setdefault("stream_options", {})["include_usage"] = True
    if args.max_completion_tokens is not None:
        payload["max_completion_tokens"] = args.max_completion_tokens

    response = requests.post(
        args.url,
        json=payload,
        stream=True,
        timeout=(30, args.timeout),
    )
    response.raise_for_status()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    event_count = 0
    done = False
    output_text: list[str] = []
    stream_errors: list[dict[str, Any]] = []
    raw_parts: list[bytes] = []
    for line in response.iter_lines():
        raw_parts.append(line + b"\n")
        if not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            done = True
            continue
        event = json.loads(data)
        event_count += 1
        if isinstance(event.get("error"), dict):
            stream_errors.append(event["error"])
        output_text.extend(iter_strings(event.get("choices", [])))

    raw = b"".join(raw_parts)
    args.output.write_bytes(raw)
    rendered = "".join(output_text)
    marker_counts = {marker: rendered.count(marker) for marker in CONTROL_MARKERS}
    receipt = {
        "schema": "llmconduit-turn-replay-v1",
        "capture": str(args.capture.resolve()),
        "endpoint": args.url,
        "http_status": response.status_code,
        "event_count": event_count,
        "done_event": done,
        "stream_error_count": len(stream_errors),
        "stream_errors": stream_errors,
        "stream_bytes": len(raw),
        "control_marker_counts": marker_counts,
        "control_marker_free": not any(marker_counts.values()),
        "max_completion_tokens": payload.get("max_completion_tokens"),
        "removed_redacted_images": removed_images,
        "replaced_redacted_images": replaced_images,
        "output_preview": rendered[:1000],
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    failures: list[str] = []
    if not done:
        failures.append("stream ended without [DONE]")
    if stream_errors:
        failures.append(f"stream contained {len(stream_errors)} error event(s)")
    leaked_markers = [marker for marker, count in marker_counts.items() if count]
    if leaked_markers:
        failures.append(
            "stream exposed Kimi control markers: " + ", ".join(leaked_markers)
        )
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
