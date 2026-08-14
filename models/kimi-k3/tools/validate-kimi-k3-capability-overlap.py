#!/usr/bin/env python3
"""Detect exact capability-benchmark text inside Kimi K3 suite contexts."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

WHITESPACE = re.compile(r"\s+")
SIGNATURE_WORDS = 8


@dataclass(frozen=True)
class BenchmarkText:
    benchmark: str
    revision: str
    item_id: str
    field: str
    normalized_text: str
    required_fragments: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return WHITESPACE.sub(" ", normalized).strip()


def benchmark_texts() -> tuple[list[BenchmarkText], dict[str, Any]]:
    texts: list[BenchmarkText] = []
    revisions = {
        "GPQA-Diamond": {
            "repository": "hendrydong/gpqa_diamond",
            "revision": "acc659161e28d4416c0ed44e3fc85c9383add471",
            "split": "test",
        },
        "HumanEval": {
            "repository": "openai/openai_humaneval",
            "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
            "split": "test",
        },
        "MMLU": {
            "repository": "cais/mmlu",
            "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
            "config": "all",
            "split": "test",
        },
    }

    humaneval = load_dataset(
        revisions["HumanEval"]["repository"],
        revision=revisions["HumanEval"]["revision"],
        split=revisions["HumanEval"]["split"],
    )
    for item in humaneval:
        for field in ("prompt",):
            text = normalize_text(item[field])
            if text:
                texts.append(
                    BenchmarkText(
                        "HumanEval",
                        revisions["HumanEval"]["revision"],
                        item["task_id"],
                        field,
                        text,
                    )
                )

    mmlu = load_dataset(
        revisions["MMLU"]["repository"],
        revisions["MMLU"]["config"],
        revision=revisions["MMLU"]["revision"],
        split=revisions["MMLU"]["split"],
    )
    for index, item in enumerate(mmlu):
        text = normalize_text(item["question"])
        choices = tuple(normalize_text(choice) for choice in item["choices"])
        if text:
            texts.append(
                BenchmarkText(
                    "MMLU",
                    revisions["MMLU"]["revision"],
                    f"{item['subject']}:{index}",
                    "question",
                    text,
                    choices,
                )
            )

    gpqa = load_dataset(
        revisions["GPQA-Diamond"]["repository"],
        revision=revisions["GPQA-Diamond"]["revision"],
        split=revisions["GPQA-Diamond"]["split"],
    )
    for index, item in enumerate(gpqa):
        text = normalize_text(item["problem"])
        if text:
            texts.append(
                BenchmarkText(
                    "GPQA-Diamond",
                    revisions["GPQA-Diamond"]["revision"],
                    f"{item['domain']}:{index}",
                    "problem",
                    text,
                )
            )
    return texts, revisions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    suite = json.loads(args.suite_manifest.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    texts, revisions = benchmark_texts()
    signatures: dict[tuple[str, ...], list[BenchmarkText]] = defaultdict(list)
    short_texts: list[BenchmarkText] = []
    for benchmark_text in texts:
        words = tuple(benchmark_text.normalized_text.split())
        if len(words) < SIGNATURE_WORDS:
            short_texts.append(benchmark_text)
        else:
            signatures[words[:SIGNATURE_WORDS]].append(benchmark_text)

    overlaps: list[dict[str, Any]] = []
    artifact_root = args.suite_manifest.parent
    for context in suite["contexts"]:
        token_ids = json.loads(
            (artifact_root / context["token_file"]).read_text(encoding="utf-8")
        )
        normalized_context = normalize_text(tokenizer.decode(token_ids))
        words = normalized_context.split()
        candidate_texts: set[BenchmarkText] = set(short_texts)
        for start in range(len(words) - SIGNATURE_WORDS + 1):
            signature = tuple(words[start : start + SIGNATURE_WORDS])
            candidate_texts.update(signatures.get(signature, ()))
        for benchmark_text in candidate_texts:
            if benchmark_text.normalized_text in normalized_context and all(
                fragment in normalized_context
                for fragment in benchmark_text.required_fragments
            ):
                overlaps.append(
                    {
                        "benchmark": benchmark_text.benchmark,
                        "benchmark_field": benchmark_text.field,
                        "benchmark_item_id": benchmark_text.item_id,
                        "benchmark_revision": benchmark_text.revision,
                        "context_index": context["context_index"],
                        "source_content_sha256": context["source_content_sha256"],
                    }
                )

    result = {
        "benchmark_revisions": revisions,
        "benchmark_text_fields": {
            "GPQA-Diamond": ["problem"],
            "HumanEval": ["prompt"],
            "MMLU": ["question and all answer choices"],
        },
        "contexts_scanned": len(suite["contexts"]),
        "exact_normalized_text_overlaps": overlaps,
        "overlap_count": len(overlaps),
        "status": "implemented" if not overlaps else "unsupported",
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    if overlaps:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
