#!/usr/bin/env python3
"""Capture final-normalized Kimi K3 hidden states for a token-ID suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

CHUNK_RE = re.compile(r"hidden\.rows-(\d+)-(\d+)\.safetensors$")
LIVE_CHUNK_RE = re.compile(r"logits\.rows-(\d+)-(\d+)\.safetensors$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_token_ids(path: Path) -> tuple[list[int], str]:
    token_ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        raise TypeError(f"Expected a JSON integer array in {path}")
    canonical_json = json.dumps(token_ids, separators=(",", ":"))
    token_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return token_ids, token_hash


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_chunks(
    path: Path,
    *,
    expected_rows: int,
    expected_width: int,
) -> list[tuple[int, int, Path]]:
    chunks: list[tuple[int, int, Path]] = []
    for item in path.glob("hidden.rows-*.safetensors"):
        match = CHUNK_RE.fullmatch(item.name)
        if match is not None:
            chunks.append((int(match.group(1)), int(match.group(2)), item))
    chunks.sort()

    next_row = 0
    for start, end, item in chunks:
        if start != next_row or end <= start:
            raise RuntimeError(
                f"Non-contiguous hidden-state chunks in {path}; "
                f"expected row {next_row}, found [{start}, {end})"
            )
        with safe_open(item, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["hidden_states"]:
                raise RuntimeError(f"Unexpected tensor keys in {item}")
            tensor_slice = handle.get_slice("hidden_states")
            if tensor_slice.get_dtype() != "BF16":
                raise RuntimeError(f"Expected BF16 hidden states in {item}")
            if tensor_slice.get_shape() != [end - start, expected_width]:
                raise RuntimeError(f"Unexpected hidden-state shape in {item}")
            metadata = handle.metadata() or {}
            if metadata.get("semantic_point") != ("after_final_rmsnorm_before_lm_head"):
                raise RuntimeError(f"Missing semantic-point identity in {item}")
        next_row = end

    if next_row != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} hidden-state rows in {path}; got {next_row}"
        )
    return chunks


def finalize_context(
    chunks: list[tuple[int, int, Path]],
    *,
    output_path: Path,
    context_index: int,
    token_hash: str,
    expected_rows: int,
    expected_width: int,
) -> dict[str, Any]:
    tensors = [
        load_file(str(path), device="cpu")["hidden_states"] for _, _, path in chunks
    ]
    captured_hidden_states = torch.cat(tensors, dim=0)
    if captured_hidden_states.shape[0] < expected_rows:
        raise RuntimeError(
            f"Capture for {output_path} has only "
            f"{captured_hidden_states.shape[0]} rows; expected at least {expected_rows}"
        )
    # A context with N tokens has N-1 aligned next-token targets. The final
    # prompt position predicts an uncaptured continuation token and therefore
    # is not part of the distribution-fidelity artifact.
    hidden_states = captured_hidden_states[:expected_rows].contiguous()
    if hidden_states.dtype != torch.bfloat16 or list(hidden_states.shape) != [
        expected_rows,
        expected_width,
    ]:
        raise RuntimeError(f"Invalid finalized hidden-state tensor for {output_path}")

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(
        {"hidden_states": hidden_states},
        str(temporary),
        metadata={
            "context_index": str(context_index),
            "semantic_point": "after_final_rmsnorm_before_lm_head",
            "token_ids_json_sha256": token_hash,
        },
    )
    temporary.replace(output_path)
    return {
        "context_index": context_index,
        "dtype": "BF16",
        "file": output_path.name,
        "key": "hidden_states",
        "sha256": sha256_file(output_path),
        "shape": [expected_rows, expected_width],
        "size_bytes": output_path.stat().st_size,
        "token_ids_json_sha256": token_hash,
    }


def validate_finalized_context(
    path: Path,
    *,
    expected_rows: int,
    expected_width: int,
    expected_token_hash: str,
) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["hidden_states"]:
            raise RuntimeError(f"Unexpected tensor keys in {path}")
        tensor_slice = handle.get_slice("hidden_states")
        if tensor_slice.get_dtype() != "BF16":
            raise RuntimeError(f"Expected BF16 hidden states in {path}")
        if tensor_slice.get_shape() != [expected_rows, expected_width]:
            raise RuntimeError(f"Unexpected hidden-state shape in {path}")
        metadata = handle.metadata() or {}
        if metadata.get("token_ids_json_sha256") != expected_token_hash:
            raise RuntimeError(f"Token hash mismatch in {path}")


def validate_live_chunks(
    path: Path,
    *,
    expected_rows: int,
    expected_vocab: int,
) -> list[tuple[int, int, Path]]:
    chunks: list[tuple[int, int, Path]] = []
    for item in path.glob("logits.rows-*.safetensors"):
        match = LIVE_CHUNK_RE.fullmatch(item.name)
        if match is not None:
            chunks.append((int(match.group(1)), int(match.group(2)), item))
    chunks.sort()

    next_row = 0
    for start, end, item in chunks:
        if start != next_row or end <= start:
            raise RuntimeError(
                f"Non-contiguous live-logit chunks in {path}; "
                f"expected row {next_row}, found [{start}, {end})"
            )
        with safe_open(item, framework="pt", device="cpu") as handle:
            if list(handle.keys()) != ["logits"]:
                raise RuntimeError(f"Unexpected tensor keys in {item}")
            tensor_slice = handle.get_slice("logits")
            if tensor_slice.get_dtype() != "BF16":
                raise RuntimeError(f"Expected BF16 live logits in {item}")
            if tensor_slice.get_shape() != [end - start, expected_vocab]:
                raise RuntimeError(f"Unexpected live-logit shape in {item}")
            metadata = handle.metadata() or {}
            if metadata.get("semantic_point") != (
                "live_lm_head_output_before_sampling"
            ):
                raise RuntimeError(f"Missing semantic-point identity in {item}")
        next_row = end

    if next_row != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} live-logit rows in {path}; got {next_row}"
        )
    return chunks


def finalize_live_context(
    chunks: list[tuple[int, int, Path]],
    *,
    output_path: Path,
    context_index: int,
    token_hash: str,
    expected_rows: int,
    expected_vocab: int,
) -> dict[str, Any]:
    tensors = [load_file(str(path), device="cpu")["logits"] for _, _, path in chunks]
    logits = torch.cat(tensors, dim=0).contiguous()
    if logits.dtype != torch.bfloat16 or list(logits.shape) != [
        expected_rows,
        expected_vocab,
    ]:
        raise RuntimeError(f"Invalid finalized live-logit tensor for {output_path}")

    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    save_file(
        {"logits": logits},
        str(temporary),
        metadata={
            "context_index": str(context_index),
            "semantic_point": "live_lm_head_output_before_sampling",
            "token_ids_json_sha256": token_hash,
        },
    )
    temporary.replace(output_path)
    return {
        "context_index": context_index,
        "dtype": "BF16",
        "file": output_path.name,
        "key": "logits",
        "sha256": sha256_file(output_path),
        "shape": [expected_rows, expected_vocab],
        "size_bytes": output_path.stat().st_size,
        "token_ids_json_sha256": token_hash,
    }


def validate_finalized_live_context(
    path: Path,
    *,
    expected_rows: int,
    expected_vocab: int,
    expected_token_hash: str,
) -> None:
    with safe_open(path, framework="pt", device="cpu") as handle:
        if list(handle.keys()) != ["logits"]:
            raise RuntimeError(f"Unexpected tensor keys in {path}")
        tensor_slice = handle.get_slice("logits")
        if tensor_slice.get_dtype() != "BF16":
            raise RuntimeError(f"Expected BF16 live logits in {path}")
        if tensor_slice.get_shape() != [expected_rows, expected_vocab]:
            raise RuntimeError(f"Unexpected live-logit shape in {path}")
        metadata = handle.metadata() or {}
        if metadata.get("token_ids_json_sha256") != expected_token_hash:
            raise RuntimeError(f"Token hash mismatch in {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="Kimi-K3")
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--live-capture-dir", type=Path)
    parser.add_argument("--live-output-dir", type=Path)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--hidden-width", type=int, default=7168)
    parser.add_argument("--vocab-size", type=int, default=163840)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--delete-raw-chunks-after-finalize", action="store_true")
    parser.add_argument(
        "--context-filter",
        choices=("all", "analysis", "qualification", "sentinel", "non-sentinel"),
        default="all",
    )
    parser.add_argument("--start-context", type=int, default=0)
    parser.add_argument("--stop-context", type=int)
    args = parser.parse_args()
    capture_live = args.live_capture_dir is not None
    if capture_live != (args.live_output_dir is not None):
        parser.error("--live-capture-dir and --live-output-dir must be used together")

    suite_path = args.suite_dir / "suite-manifest.json"
    suite = json.loads(suite_path.read_text(encoding="utf-8"))
    runtime_manifest_sha256 = sha256_file(args.runtime_manifest)
    context_rows = int(suite["context_length"])
    scored_rows = context_rows - 1
    contexts = suite.get("contexts", suite.get("windows"))
    if not isinstance(contexts, list):
        raise TypeError("Suite manifest must contain contexts or windows")
    if args.context_filter == "analysis":
        contexts = [item for item in contexts if item.get("partition") == "analysis"]
    elif args.context_filter == "qualification":
        contexts = [
            item for item in contexts if item.get("partition") == "qualification"
        ]
    elif args.context_filter == "sentinel":
        contexts = [item for item in contexts if item.get("sentinel") is True]
    elif args.context_filter == "non-sentinel":
        contexts = [item for item in contexts if item.get("sentinel") is not True]

    args.capture_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "context_length": context_rows,
            "created_utc": datetime.now(UTC).isoformat(),
            "format_version": 1,
            "hidden_width": args.hidden_width,
            "kind": "Kimi K3 final-normalized pre-LM-head hidden states",
            "run_name": args.run_name,
            "runtime_manifest": str(args.runtime_manifest.resolve()),
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "semantic_point": "after_final_rmsnorm_before_lm_head",
            "scored_rows_per_context": scored_rows,
            "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
            "tensor_key": "hidden_states",
            "contexts": [],
        }
    if manifest.get("runtime_manifest_sha256") != runtime_manifest_sha256:
        raise RuntimeError(
            f"Runtime manifest differs from the capture recorded in {manifest_path}"
        )
    completed = {
        int(record["context_index"]): record for record in manifest["contexts"]
    }

    live_manifest: dict[str, Any] | None = None
    live_completed: dict[int, dict[str, Any]] = {}
    live_manifest_path: Path | None = None
    if capture_live:
        assert args.live_capture_dir is not None
        assert args.live_output_dir is not None
        args.live_capture_dir.mkdir(parents=True, exist_ok=True)
        args.live_output_dir.mkdir(parents=True, exist_ok=True)
        live_manifest_path = args.live_output_dir / "manifest.json"
        if live_manifest_path.exists():
            live_manifest = json.loads(live_manifest_path.read_text(encoding="utf-8"))
        else:
            live_manifest = {
                "context_length": context_rows,
                "created_utc": datetime.now(UTC).isoformat(),
                "format_version": 1,
                "kind": "Kimi K3 native live prompt logits",
                "run_name": args.run_name,
                "runtime_manifest": str(args.runtime_manifest.resolve()),
                "runtime_manifest_sha256": runtime_manifest_sha256,
                "scored_rows_per_context": context_rows - 1,
                "semantic_point": "live_lm_head_output_before_sampling",
                "suite_token_hash_sha256": suite["suite_token_hash_sha256"],
                "tensor_key": "logits",
                "vocab_size": args.vocab_size,
                "contexts": [],
            }
        if live_manifest.get("runtime_manifest_sha256") != runtime_manifest_sha256:
            raise RuntimeError(
                "Runtime manifest differs from the live-logit capture recorded in "
                f"{live_manifest_path}"
            )
        live_completed = {
            int(record["context_index"]): record for record in live_manifest["contexts"]
        }
        if set(completed) != set(live_completed):
            raise RuntimeError(
                "Hidden-state and live-logit manifests contain different contexts"
            )

    stop = len(contexts) if args.stop_context is None else args.stop_context
    for context in contexts[args.start_context : stop]:
        index = int(context.get("index", context.get("context_index")))
        token_path = args.suite_dir / context["token_file"]
        token_hash = context["token_ids_json_sha256"]
        token_ids, observed_token_hash = load_token_ids(token_path)
        if observed_token_hash != token_hash:
            raise RuntimeError(f"Token file hash mismatch for context {index}")
        if len(token_ids) != context_rows:
            raise RuntimeError(f"Invalid token IDs for context {index}")

        output_path = args.output_dir / f"hidden_{index:04d}.safetensors"
        if index in completed:
            validate_finalized_context(
                output_path,
                expected_rows=scored_rows,
                expected_width=args.hidden_width,
                expected_token_hash=token_hash,
            )
            if sha256_file(output_path) != completed[index]["sha256"]:
                raise RuntimeError(f"Hidden-state hash mismatch for context {index}")
            if capture_live:
                assert args.live_output_dir is not None
                live_output_path = (
                    args.live_output_dir / f"logits_{index:04d}.safetensors"
                )
                validate_finalized_live_context(
                    live_output_path,
                    expected_rows=context_rows - 1,
                    expected_vocab=args.vocab_size,
                    expected_token_hash=token_hash,
                )
                if sha256_file(live_output_path) != live_completed[index]["sha256"]:
                    raise RuntimeError(f"Live-logit hash mismatch for context {index}")
            print(f"context {index:04d}: already captured", flush=True)
            continue
        if output_path.exists():
            raise RuntimeError(f"Unrecorded output already exists: {output_path}")

        before = {item.name for item in args.capture_dir.iterdir() if item.is_dir()}
        live_before: set[str] = set()
        if capture_live:
            assert args.live_capture_dir is not None
            live_before = {
                item.name for item in args.live_capture_dir.iterdir() if item.is_dir()
            }
        request = {
            "ignore_eos": True,
            "max_tokens": 1,
            "model": args.model,
            "prompt": token_ids,
            "seed": 1,
            "temperature": 0,
        }
        if capture_live:
            request["prompt_logprobs"] = 1
        started = time.monotonic()
        response = requests.post(args.url, json=request, timeout=args.timeout)
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        after = {item.name for item in args.capture_dir.iterdir() if item.is_dir()}
        created = sorted(after - before)
        if len(created) != 1:
            raise RuntimeError(
                f"Expected one capture directory for context {index}; got {created}"
            )
        request_dir = args.capture_dir / created[0]
        chunks = validate_chunks(
            request_dir,
            expected_rows=context_rows,
            expected_width=args.hidden_width,
        )
        record = finalize_context(
            chunks,
            output_path=output_path,
            context_index=index,
            token_hash=token_hash,
            expected_rows=scored_rows,
            expected_width=args.hidden_width,
        )
        payload = response.json()
        record.update(
            {
                "elapsed_seconds": elapsed,
                "raw_chunks_retained": not args.delete_raw_chunks_after_finalize,
                "request_id": payload.get("id"),
                "source_class": context.get("semantic_class", context.get("domain")),
            }
        )
        if not args.delete_raw_chunks_after_finalize:
            record["capture_chunk_dir"] = str(request_dir)
            record["capture_chunks"] = [item.name for _, _, item in chunks]
        manifest["contexts"].append(record)
        manifest["contexts"].sort(key=lambda item: int(item["context_index"]))
        manifest["total_size_bytes"] = sum(
            int(item["size_bytes"]) for item in manifest["contexts"]
        )
        write_json(manifest_path, manifest)
        completed[index] = record

        if capture_live:
            assert args.live_capture_dir is not None
            assert args.live_output_dir is not None
            assert live_manifest is not None
            assert live_manifest_path is not None
            live_after = {
                item.name for item in args.live_capture_dir.iterdir() if item.is_dir()
            }
            live_created = sorted(live_after - live_before)
            if len(live_created) != 1:
                raise RuntimeError(
                    f"Expected one live capture directory for context {index}; "
                    f"got {live_created}"
                )
            live_request_dir = args.live_capture_dir / live_created[0]
            live_chunks = validate_live_chunks(
                live_request_dir,
                expected_rows=context_rows - 1,
                expected_vocab=args.vocab_size,
            )
            live_output_path = args.live_output_dir / (
                f"logits_{index:04d}.safetensors"
            )
            if live_output_path.exists():
                raise RuntimeError(
                    f"Unrecorded output already exists: {live_output_path}"
                )
            live_record = finalize_live_context(
                live_chunks,
                output_path=live_output_path,
                context_index=index,
                token_hash=token_hash,
                expected_rows=context_rows - 1,
                expected_vocab=args.vocab_size,
            )
            live_record.update(
                {
                    "elapsed_seconds": elapsed,
                    "raw_chunks_retained": not args.delete_raw_chunks_after_finalize,
                    "request_id": payload.get("id"),
                    "source_class": context.get(
                        "semantic_class", context.get("domain")
                    ),
                }
            )
            if not args.delete_raw_chunks_after_finalize:
                live_record["capture_chunk_dir"] = str(live_request_dir)
                live_record["capture_chunks"] = [
                    item.name for _, _, item in live_chunks
                ]
            live_manifest["contexts"].append(live_record)
            live_manifest["contexts"].sort(key=lambda item: int(item["context_index"]))
            live_manifest["total_size_bytes"] = sum(
                int(item["size_bytes"]) for item in live_manifest["contexts"]
            )
            write_json(live_manifest_path, live_manifest)
            live_completed[index] = live_record
        if args.delete_raw_chunks_after_finalize:
            shutil.rmtree(request_dir)
            if capture_live:
                shutil.rmtree(live_request_dir)
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
