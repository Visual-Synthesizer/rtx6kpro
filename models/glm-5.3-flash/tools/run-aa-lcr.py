#!/usr/bin/env python3
"""Run and score the pinned AA-LCR dataset through OpenAI-compatible APIs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

DATASET_REVISION = "bdae010bbce259820c0e34c1d7cce210d966fb75"
DATASET_CSV_SHA256 = "2f90d9c30cfb4dd8df2c0f46547c384065e4c76917bd347a9a97bf797235c1ea"
DOCUMENT_ZIP_SHA256 = "5e839249826f6b9bd5324f0d139089c9dc481ccb3f212a6dfad00c51045d9d8a"
EXPECTED_QUESTIONS = 100
EXPECTED_DOCUMENT_SETS = 30
EXPECTED_REFERENCED_DOCUMENTS = 229
JUDGE_OUTPUT_RE = re.compile(r"^(CORRECT|INCORRECT)[.!]?$", re.IGNORECASE)
JUDGE_PROTOCOLS = {
    "artificial-analysis-v4.1.1": (
        "Artificial Analysis AA-LCR v4.1.1 equality checker"
    ),
    "frozen-official-kimi-k3": (
        "Frozen official Kimi-K3 checkpoint used as a common equality checker"
    ),
}


@dataclass(frozen=True)
class Question:
    """One AA-LCR question and its ordered document references."""

    question_id: int
    document_category: str
    document_set_id: str
    question: str
    official_answer: str
    filenames: tuple[str, ...]
    reported_input_tokens: int


@dataclass(frozen=True)
class ResolvedDocument:
    """One document resolved from a CSV filename without changing source data."""

    requested_name: str
    relative_path: str
    path: Path
    sha256: str
    unicode_normalization_required: bool


@dataclass(frozen=True)
class GenerationTask:
    """One independently writable AA-LCR generation request."""

    question: Question
    repeat: int
    prompt: str
    prompt_sha256: str
    documents: tuple[dict[str, Any], ...]
    receipt_path: Path
    endpoint_index: int


@dataclass(frozen=True)
class JudgementTask:
    """One independently writable AA-LCR equality-checker request."""

    question: Question
    repeat: int
    generation_path: Path
    generation_sha256: str
    prompt: str
    output_path: Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_runtime_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise TypeError("AA-LCR runtime manifest must contain one JSON object")
    required_paths = (
        ("status",),
        ("checkpoint", "repository"),
        ("checkpoint", "revision"),
        ("checkpoint", "index_sha256"),
        ("container", "image"),
        ("container", "image_id"),
        ("container", "registry_digest"),
        ("source", "vllm_revision"),
        ("source", "b12x_revision"),
        ("topology", "tensor_parallel_size"),
        ("topology", "decode_context_parallel_size"),
        ("serving", "activation_dtype"),
        ("serving", "kv_cache_dtype"),
        ("serving", "attention_backend"),
        ("serving", "kda_prefill_backend"),
        ("serving", "moe_backend"),
        ("serving", "linear_backend"),
        ("serving", "weight_loader"),
        ("serving", "max_model_len"),
        ("serving", "max_num_batched_tokens"),
        ("serving", "max_num_seqs"),
        ("serving", "kv_cache_memory_bytes"),
        ("serving", "prefix_caching"),
        ("serving", "compilation_config"),
        ("server_arguments",),
        ("relevant_environment",),
    )
    for path in required_paths:
        value: Any = manifest
        for key in path:
            if not isinstance(value, dict) or key not in value:
                raise ValueError(
                    "AA-LCR runtime manifest is missing required field "
                    + ".".join(path)
                )
            value = value[key]
        if value is None or value == "":
            raise ValueError(
                "AA-LCR runtime manifest has an empty required field " + ".".join(path)
            )
    if manifest["status"] not in {
        "implemented",
        "qualified",
        "research-only",
        "unsupported",
    }:
        raise ValueError("AA-LCR runtime manifest has an invalid status")
    if (
        not isinstance(manifest["server_arguments"], list)
        or not manifest["server_arguments"]
    ):
        raise TypeError("AA-LCR runtime server_arguments must be a non-empty list")
    if not isinstance(manifest["relevant_environment"], dict):
        raise TypeError("AA-LCR runtime relevant_environment must be an object")
    return manifest


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_int(value: str, *, field: str) -> int:
    normalized = value.strip().replace(",", "")
    try:
        return int(normalized)
    except ValueError as error:
        raise ValueError(
            f"AA-LCR field {field!r} is not an integer: {value!r}"
        ) from error


def load_questions(dataset_root: Path) -> list[Question]:
    csv_path = dataset_root / "AA-LCR_Dataset.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"AA-LCR question CSV is absent: {csv_path}")
    if sha256_file(csv_path) != DATASET_CSV_SHA256:
        raise RuntimeError(
            f"AA-LCR question CSV hash does not match revision {DATASET_REVISION}: "
            f"{csv_path}"
        )

    questions: list[Question] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            questions.append(
                Question(
                    question_id=parse_int(row["question_id"], field="question_id"),
                    document_category=row["document_category"],
                    document_set_id=row["document_set_id"],
                    question=row["question"],
                    official_answer=row["answer"],
                    filenames=tuple(row["data_source_filenames"].split(";")),
                    reported_input_tokens=parse_int(
                        row["input_tokens"], field="input_tokens"
                    ),
                )
            )

    ids = [question.question_id for question in questions]
    if len(questions) != EXPECTED_QUESTIONS or sorted(ids) != list(
        range(1, EXPECTED_QUESTIONS + 1)
    ):
        raise RuntimeError(
            "AA-LCR CSV must contain question IDs 1 through 100 exactly once"
        )
    return questions


class DocumentResolver:
    """Resolve ordered CSV filenames with Unicode normalization and hash caching."""

    def __init__(self, document_root: Path) -> None:
        if not document_root.is_dir():
            raise FileNotFoundError(f"AA-LCR document root is absent: {document_root}")
        self.document_root = document_root
        self._directory_maps: dict[Path, dict[str, Path]] = {}
        self._hashes: dict[Path, str] = {}

    def _directory_map(self, directory: Path) -> dict[str, Path]:
        if directory in self._directory_maps:
            return self._directory_maps[directory]
        if not directory.is_dir():
            raise FileNotFoundError(
                f"AA-LCR document-set directory is absent: {directory}"
            )
        mapping: dict[str, Path] = {}
        for path in directory.iterdir():
            if not path.is_file():
                continue
            normalized = unicodedata.normalize("NFC", path.name)
            if normalized in mapping:
                raise RuntimeError(
                    "AA-LCR document directory has an NFC filename collision: "
                    f"{directory}"
                )
            mapping[normalized] = path
        self._directory_maps[directory] = mapping
        return mapping

    def resolve(self, question: Question) -> list[ResolvedDocument]:
        directory = (
            self.document_root / question.document_category / question.document_set_id
        )
        mapping = self._directory_map(directory)
        resolved: list[ResolvedDocument] = []
        for requested_name in question.filenames:
            normalized = unicodedata.normalize("NFC", requested_name)
            path = mapping.get(normalized)
            if path is None:
                raise FileNotFoundError(
                    "AA-LCR CSV references an absent document: "
                    f"category={question.document_category!r}, "
                    f"set={question.document_set_id!r}, file={requested_name!r}"
                )
            if path not in self._hashes:
                self._hashes[path] = sha256_file(path)
            resolved.append(
                ResolvedDocument(
                    requested_name=requested_name,
                    relative_path=path.relative_to(self.document_root).as_posix(),
                    path=path,
                    sha256=self._hashes[path],
                    unicode_normalization_required=path.name != requested_name,
                )
            )
        return resolved


def build_prompt(documents: list[ResolvedDocument], question: str) -> str:
    document_text = "\n\n".join(
        f"BEGIN DOCUMENT {index}:\n{document.path.read_text(encoding='utf-8')}\n"
        f"END DOCUMENT {index}"
        for index, document in enumerate(documents, start=1)
    )
    return f"""BEGIN INPUT DOCUMENTS

{document_text}

END INPUT DOCUMENTS

Answer the following question using the input documents provided above.

START QUESTION

{question}

END QUESTION
"""


def dataset_identity(dataset_root: Path, resolver: DocumentResolver) -> dict[str, Any]:
    zip_path = dataset_root / "extracted_text" / "AA-LCR_extracted-text.zip"
    if not zip_path.is_file():
        raise FileNotFoundError(f"AA-LCR source ZIP is absent: {zip_path}")
    zip_hash = sha256_file(zip_path)
    if zip_hash != DOCUMENT_ZIP_SHA256:
        raise RuntimeError(
            f"AA-LCR document ZIP hash does not match revision {DATASET_REVISION}: "
            f"{zip_path}"
        )

    questions = load_questions(dataset_root)
    referenced: dict[str, str] = {}
    normalization_count = 0
    prompt_hashes: list[tuple[int, str]] = []
    document_sets: set[tuple[str, str]] = set()
    for question in questions:
        documents = resolver.resolve(question)
        document_sets.add((question.document_category, question.document_set_id))
        for document in documents:
            referenced[document.relative_path] = document.sha256
            normalization_count += int(document.unicode_normalization_required)
        prompt_hashes.append(
            (
                question.question_id,
                sha256_text(build_prompt(documents, question.question)),
            )
        )

    if len(document_sets) != EXPECTED_DOCUMENT_SETS:
        raise RuntimeError(
            f"AA-LCR CSV resolves to {len(document_sets)} document sets; "
            f"expected {EXPECTED_DOCUMENT_SETS}"
        )
    if len(referenced) != EXPECTED_REFERENCED_DOCUMENTS:
        raise RuntimeError(
            f"AA-LCR CSV resolves to {len(referenced)} unique documents; "
            f"expected {EXPECTED_REFERENCED_DOCUMENTS}"
        )

    all_files = {
        path.relative_to(resolver.document_root).as_posix()
        for path in resolver.document_root.rglob("*")
        if path.is_file()
    }
    referenced_hash = canonical_sha256(sorted(referenced.items()))
    prompt_hash = canonical_sha256(prompt_hashes)
    return {
        "repository": "ArtificialAnalysis/AA-LCR",
        "revision": DATASET_REVISION,
        "question_csv_sha256": DATASET_CSV_SHA256,
        "document_zip_sha256": zip_hash,
        "questions": len(questions),
        "document_sets": len(document_sets),
        "referenced_documents": len(referenced),
        "archive_documents": len(all_files),
        "unreferenced_documents": sorted(all_files - set(referenced)),
        "unicode_normalized_references": normalization_count,
        "referenced_document_manifest_sha256": referenced_hash,
        "prompt_manifest_sha256": prompt_hash,
        "reported_input_tokens_min": min(
            question.reported_input_tokens for question in questions
        ),
        "reported_input_tokens_max": max(
            question.reported_input_tokens for question in questions
        ),
    }


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return normalized + "/chat/completions"
    return normalized + "/v1/chat/completions"


def api_headers(api_key_env: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(
    *,
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout_seconds: float,
) -> tuple[dict[str, Any], float]:
    start = time.monotonic()
    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=timeout_seconds,
    )
    elapsed = time.monotonic() - start
    if response.status_code != 200:
        body = response.text[:4096]
        raise RuntimeError(
            f"OpenAI-compatible endpoint returned HTTP {response.status_code}: {body}"
        )
    result = response.json()
    if not isinstance(result, dict):
        raise TypeError("OpenAI-compatible endpoint returned a non-object JSON value")
    return result, elapsed


def choice_message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("Chat response must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TypeError("Chat response choice has no message object")
    return message


def question_receipt_path(output_dir: Path, repeat: int, question_id: int) -> Path:
    return (
        output_dir
        / "responses"
        / f"repeat-{repeat:02d}"
        / f"question-{question_id:04d}.json"
    )


def judgement_receipt_path(output_dir: Path, repeat: int, question_id: int) -> Path:
    return (
        output_dir
        / "judgements"
        / f"repeat-{repeat:02d}"
        / f"question-{question_id:04d}.json"
    )


def selected_questions(
    questions: list[Question], start_question: int, stop_question: int
) -> list[Question]:
    if start_question < 1 or stop_question > EXPECTED_QUESTIONS + 1:
        raise ValueError("Question range must be within [1, 101)")
    if start_question >= stop_question:
        raise ValueError("Question range must select at least one question")
    return [
        question
        for question in questions
        if start_question <= question.question_id < stop_question
    ]


def command_validate(args: argparse.Namespace) -> None:
    document_root = args.document_root or (
        args.dataset_root / "extracted_text" / "unpacked" / "lcr"
    )
    identity = dataset_identity(args.dataset_root, DocumentResolver(document_root))
    print(json.dumps(identity, indent=2, sort_keys=True))


def command_token_counts(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    document_root = args.document_root or (
        args.dataset_root / "extracted_text" / "unpacked" / "lcr"
    )
    resolver = DocumentResolver(document_root)
    questions = load_questions(args.dataset_root)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        revision=args.tokenizer_revision,
        trust_remote_code=True,
    )
    dataset = dataset_identity(args.dataset_root, resolver)
    records: list[dict[str, int]] = []
    for question in questions:
        prompt = build_prompt(resolver.resolve(question), question.question)
        raw_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        chat_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(chat_tokens, Mapping):
            chat_token_ids = chat_tokens["input_ids"]
        else:
            chat_token_ids = chat_tokens
        records.append(
            {
                "question_id": question.question_id,
                "reported_cl100k": question.reported_input_tokens,
                "raw": len(raw_tokens),
                "chat": len(chat_token_ids),
            }
        )
    chat_lengths = [record["chat"] for record in records]
    result = {
        "artifact_kind": "AA-LCR prompt token counts",
        "status": "implemented",
        "dataset": {
            "repository": dataset["repository"],
            "revision": dataset["revision"],
            "prompt_manifest_sha256": dataset["prompt_manifest_sha256"],
        },
        "tokenizer_checkpoint": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "questions": len(records),
        "minimum": min(chat_lengths),
        "median": statistics.median(chat_lengths),
        "maximum": max(chat_lengths),
        "mean": statistics.mean(chat_lengths),
        "maximum_question_id": max(records, key=lambda record: record["chat"])[
            "question_id"
        ],
        "contexts": records,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_generate(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise ValueError("AA-LCR repeat count must be positive")
    if args.concurrency_per_endpoint <= 0:
        raise ValueError("AA-LCR per-endpoint client concurrency must be positive")
    if args.max_tokens <= 0:
        raise ValueError("AA-LCR maximum output token count must be positive")
    document_root = args.document_root or (
        args.dataset_root / "extracted_text" / "unpacked" / "lcr"
    )
    resolver = DocumentResolver(document_root)
    identity = dataset_identity(args.dataset_root, resolver)
    if not args.runtime_manifest.is_file():
        raise FileNotFoundError(
            f"AA-LCR runtime manifest is absent: {args.runtime_manifest}"
        )
    runtime_manifest = validate_runtime_manifest(
        json.loads(args.runtime_manifest.read_text(encoding="utf-8"))
    )
    runtime_manifest_sha256 = sha256_file(args.runtime_manifest)
    questions = selected_questions(
        load_questions(args.dataset_root), args.start_question, args.stop_question
    )
    api_urls = [chat_completions_url(base_url) for base_url in args.base_url]
    if len(set(api_urls)) != len(api_urls):
        raise ValueError("AA-LCR generation endpoints must be unique")
    generation_config = {
        "api_urls": api_urls,
        "endpoint_assignment": "question_id_modulo_endpoint_count",
        "model": args.model,
        "repeats": args.repeats,
        "client_concurrency_per_endpoint": args.concurrency_per_endpoint,
        "client_concurrency_total": args.concurrency_per_endpoint * len(api_urls),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "request_seed": None,
        "system_message": None,
        "stream": False,
        "runtime_manifest_sha256": runtime_manifest_sha256,
    }
    if args.repeat_scheduling != "independent":
        # Absence denotes independently scheduled tasks in the generation
        # manifest schema. Explicit modes identify additional ordering rules.
        generation_config["repeat_scheduling"] = args.repeat_scheduling
    run_manifest = {
        "artifact_kind": "AA-LCR generation run",
        "status": "implemented",
        "created_utc": utc_now(),
        "dataset": identity,
        "runtime": runtime_manifest,
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "generation": generation_config,
        "generation_config_sha256": canonical_sha256(generation_config),
        "prompt_template": "AA-LCR dataset revision bdae010 prompt with one user message",
    }
    manifest_path = args.output_dir / "generation-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("dataset") != identity
            or existing.get("generation_config_sha256")
            != run_manifest["generation_config_sha256"]
        ):
            raise RuntimeError(
                f"Generation output directory has incompatible identity: {args.output_dir}"
            )
    else:
        write_json(manifest_path, run_manifest)

    headers = api_headers(args.api_key_env)
    skipped = 0
    tasks: list[GenerationTask] = []
    task_groups: list[tuple[GenerationTask, ...]] = []
    for question in questions:
        documents = resolver.resolve(question)
        prompt = build_prompt(documents, question.question)
        prompt_sha256 = sha256_text(prompt)
        document_records = tuple(
            {
                "requested_name": document.requested_name,
                "relative_path": document.relative_path,
                "sha256": document.sha256,
                "unicode_normalization_required": document.unicode_normalization_required,
            }
            for document in documents
        )
        question_tasks: list[GenerationTask] = []
        for repeat in range(args.repeats):
            receipt_path = question_receipt_path(
                args.output_dir, repeat, question.question_id
            )
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if (
                    receipt.get("status") == "qualified"
                    and receipt.get("prompt_sha256") == prompt_sha256
                    and receipt.get("generation_config_sha256")
                    == run_manifest["generation_config_sha256"]
                ):
                    receipt_path.with_suffix(".error.json").unlink(missing_ok=True)
                    skipped += 1
                    continue
                raise RuntimeError(
                    f"Generation receipt is incompatible: {receipt_path}"
                )

            task = GenerationTask(
                question=question,
                repeat=repeat,
                prompt=prompt,
                prompt_sha256=prompt_sha256,
                documents=document_records,
                receipt_path=receipt_path,
                endpoint_index=(question.question_id - 1) % len(api_urls),
            )
            tasks.append(task)
            question_tasks.append(task)
        if question_tasks:
            task_groups.append(tuple(question_tasks))

    def generate_one(task: GenerationTask) -> dict[str, Any]:
        failure_path = task.receipt_path.with_suffix(".error.json")
        url = api_urls[task.endpoint_index]
        request = {
            "model": args.model,
            "messages": [{"role": "user", "content": task.prompt}],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "stream": False,
        }
        try:
            response, elapsed = post_json(
                url=url,
                payload=request,
                headers=headers,
                timeout_seconds=args.timeout_seconds,
            )
            message = choice_message(response)
            candidate_answer = message.get("content")
            if not isinstance(candidate_answer, str):
                candidate_answer = ""
            receipt = {
                "artifact_kind": "AA-LCR generation receipt",
                "status": "qualified",
                "completed_utc": utc_now(),
                "question_id": task.question.question_id,
                "repeat": task.repeat,
                "document_category": task.question.document_category,
                "document_set_id": task.question.document_set_id,
                "documents": task.documents,
                "question": task.question.question,
                "official_answer_sha256": sha256_text(task.question.official_answer),
                "reported_input_tokens_cl100k_base": task.question.reported_input_tokens,
                "prompt_chars": len(task.prompt),
                "prompt_sha256": task.prompt_sha256,
                "generation_config_sha256": run_manifest["generation_config_sha256"],
                "server_instance_index": task.endpoint_index,
                "api_url": url,
                "elapsed_seconds": elapsed,
                "candidate_answer": candidate_answer,
                "candidate_answer_sha256": sha256_text(candidate_answer),
                "response": response,
            }
            write_json(task.receipt_path, receipt)
            # A successful retry supersedes its failure sidecar. The qualified
            # response receipt remains the authoritative task outcome.
            failure_path.unlink(missing_ok=True)
            usage = response.get("usage", {})
            result = {
                "question_id": task.question.question_id,
                "repeat": task.repeat,
                "elapsed_seconds": elapsed,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": response["choices"][0].get("finish_reason"),
            }
        except Exception as error:
            write_json(
                failure_path,
                {
                    "artifact_kind": "AA-LCR generation failure",
                    "status": "unsupported",
                    "failed_utc": utc_now(),
                    "question_id": task.question.question_id,
                    "repeat": task.repeat,
                    "prompt_sha256": task.prompt_sha256,
                    "generation_config_sha256": run_manifest[
                        "generation_config_sha256"
                    ],
                    "server_instance_index": task.endpoint_index,
                    "api_url": url,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        print(json.dumps(result, sort_keys=True), flush=True)
        return result

    completed = 0
    if args.concurrency_per_endpoint == 1 and len(api_urls) == 1:
        for task in tasks:
            generate_one(task)
            completed += 1
    elif args.repeat_scheduling == "question_serial":

        def generate_question(group: tuple[GenerationTask, ...]) -> int:
            for task in group:
                generate_one(task)
            return len(group)

        executors = [
            ThreadPoolExecutor(max_workers=args.concurrency_per_endpoint)
            for _ in api_urls
        ]
        futures = [
            executors[group[0].endpoint_index].submit(generate_question, group)
            for group in task_groups
        ]
        try:
            for future in as_completed(futures):
                completed += future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
    else:
        executors = [
            ThreadPoolExecutor(max_workers=args.concurrency_per_endpoint)
            for _ in api_urls
        ]
        futures = [
            executors[task.endpoint_index].submit(generate_one, task)
            for task in tasks
        ]
        try:
            for future in as_completed(futures):
                future.result()
                completed += 1
        except BaseException:
            for future in futures:
                future.cancel()
            raise
        finally:
            for executor in executors:
                executor.shutdown(wait=True, cancel_futures=True)
    print(json.dumps({"completed": completed, "skipped": skipped}, sort_keys=True))


def percentile(values: list[int] | list[float], probability: float) -> float:
    """Return a linearly interpolated percentile over a non-empty sample."""

    if not values:
        raise ValueError("A percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("Percentile probability must be within [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def command_verify_generations(args: argparse.Namespace) -> None:
    """Verify a complete generation artifact and seal its receipt manifest."""

    document_root = args.document_root or (
        args.dataset_root / "extracted_text" / "unpacked" / "lcr"
    )
    resolver = DocumentResolver(document_root)
    dataset = dataset_identity(args.dataset_root, resolver)
    questions = load_questions(args.dataset_root)

    generation_manifest_path = args.generation_dir / "generation-manifest.json"
    runtime_manifest_path = args.generation_dir / "runtime-manifest.json"
    token_count_path = args.token_count_manifest or (
        args.dataset_root / "kimi-k3-token-counts.json"
    )
    for required_path in (
        generation_manifest_path,
        runtime_manifest_path,
        token_count_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"AA-LCR verification input is absent: {required_path}"
            )

    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    runtime_manifest = validate_runtime_manifest(
        json.loads(runtime_manifest_path.read_text(encoding="utf-8"))
    )
    token_count_manifest = json.loads(token_count_path.read_text(encoding="utf-8"))
    if generation_manifest.get("dataset") != dataset:
        raise RuntimeError(
            "AA-LCR generation manifest does not match the pinned dataset identity"
        )
    if generation_manifest.get("runtime") != runtime_manifest:
        raise RuntimeError(
            "AA-LCR generation manifest embeds a different serving runtime manifest"
        )
    runtime_manifest_sha256 = sha256_file(runtime_manifest_path)
    if generation_manifest.get("runtime_manifest_sha256") != runtime_manifest_sha256:
        raise RuntimeError(
            "AA-LCR generation manifest has a mismatched runtime-manifest hash"
        )
    generation_config = generation_manifest.get("generation")
    if not isinstance(generation_config, dict):
        raise TypeError("AA-LCR generation manifest has no generation configuration")
    generation_config_sha256 = canonical_sha256(generation_config)
    if generation_manifest.get("generation_config_sha256") != generation_config_sha256:
        raise RuntimeError(
            "AA-LCR generation manifest has a mismatched configuration hash"
        )
    repeats = generation_config.get("repeats")
    if repeats != 3:
        raise RuntimeError("Qualified AA-LCR generation requires exactly three repeats")

    token_count_sha256 = sha256_file(token_count_path)
    expected_token_dataset = {
        "repository": dataset["repository"],
        "revision": dataset["revision"],
        "prompt_manifest_sha256": dataset["prompt_manifest_sha256"],
    }
    if token_count_manifest.get("dataset") != expected_token_dataset:
        raise RuntimeError(
            "AA-LCR token-count manifest does not match the pinned prompt set"
        )
    token_records = token_count_manifest.get("contexts")
    if not isinstance(token_records, list):
        raise TypeError("AA-LCR token-count manifest has no context list")
    token_counts = {
        int(record["question_id"]): int(record["chat"]) for record in token_records
    }
    if sorted(token_counts) != list(range(1, EXPECTED_QUESTIONS + 1)):
        raise RuntimeError(
            "AA-LCR token-count manifest must contain question IDs 1 through 100"
        )

    response_root = args.generation_dir / "responses"
    api_urls = generation_config.get("api_urls")
    if (
        not isinstance(api_urls, list)
        or not api_urls
        or not all(isinstance(url, str) and url for url in api_urls)
    ):
        raise TypeError("AA-LCR generation manifest has no endpoint list")
    expected_paths = {
        question_receipt_path(args.generation_dir, repeat, question.question_id)
        for question in questions
        for repeat in range(repeats)
    }
    observed_paths = {
        path
        for path in response_root.glob("repeat-*/question-*.json")
        if re.fullmatch(r"question-\d{4}\.json", path.name)
    }
    if observed_paths != expected_paths:
        missing = sorted(
            path.relative_to(args.generation_dir).as_posix()
            for path in expected_paths - observed_paths
        )
        unexpected = sorted(
            path.relative_to(args.generation_dir).as_posix()
            for path in observed_paths - expected_paths
        )
        raise RuntimeError(
            "AA-LCR generation receipt set is incomplete or contains unexpected files: "
            f"missing_count={len(missing)}, missing_sample={missing[:10]}, "
            f"unexpected_count={len(unexpected)}, "
            f"unexpected_sample={unexpected[:10]}"
        )
    failure_paths = sorted(response_root.glob("repeat-*/*.error.json"))
    if failure_paths:
        raise RuntimeError(
            "AA-LCR generation directory contains failure sidecars: "
            + ", ".join(str(path) for path in failure_paths)
        )

    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    elapsed_seconds: list[float] = []
    finish_reasons: Counter[str] = Counter()
    receipt_hashes: list[tuple[str, str]] = []
    per_repeat: Counter[int] = Counter()
    per_question: Counter[int] = Counter()
    for question in questions:
        documents = resolver.resolve(question)
        prompt = build_prompt(documents, question.question)
        prompt_sha256 = sha256_text(prompt)
        expected_documents = [
            {
                "requested_name": document.requested_name,
                "relative_path": document.relative_path,
                "sha256": document.sha256,
                "unicode_normalization_required": (
                    document.unicode_normalization_required
                ),
            }
            for document in documents
        ]
        for repeat in range(repeats):
            path = question_receipt_path(
                args.generation_dir, repeat, question.question_id
            )
            receipt = json.loads(path.read_text(encoding="utf-8"))
            required_values = {
                "artifact_kind": "AA-LCR generation receipt",
                "status": "qualified",
                "question_id": question.question_id,
                "repeat": repeat,
                "document_category": question.document_category,
                "document_set_id": question.document_set_id,
                "documents": expected_documents,
                "question": question.question,
                "official_answer_sha256": sha256_text(question.official_answer),
                "reported_input_tokens_cl100k_base": (question.reported_input_tokens),
                "prompt_chars": len(prompt),
                "prompt_sha256": prompt_sha256,
                "generation_config_sha256": generation_config_sha256,
                "server_instance_index": (question.question_id - 1) % len(api_urls),
                "api_url": api_urls[(question.question_id - 1) % len(api_urls)],
            }
            for key, expected in required_values.items():
                if receipt.get(key) != expected:
                    raise RuntimeError(
                        f"AA-LCR generation receipt field {key!r} is invalid: {path}"
                    )

            candidate_answer = receipt.get("candidate_answer")
            if not isinstance(candidate_answer, str) or not candidate_answer:
                raise RuntimeError(
                    f"AA-LCR generation receipt has no candidate answer: {path}"
                )
            if receipt.get("candidate_answer_sha256") != sha256_text(candidate_answer):
                raise RuntimeError(
                    f"AA-LCR generation receipt has a mismatched answer hash: {path}"
                )
            response = receipt.get("response")
            if not isinstance(response, dict):
                raise TypeError(
                    f"AA-LCR generation receipt has no response object: {path}"
                )
            if response.get("model") != generation_config.get("model"):
                raise RuntimeError(
                    f"AA-LCR response model does not match the run manifest: {path}"
                )
            message = choice_message(response)
            if message.get("content") != candidate_answer:
                raise RuntimeError(
                    f"AA-LCR response content does not match the stored answer: {path}"
                )
            choice = response["choices"][0]
            finish_reason = choice.get("finish_reason")
            if not isinstance(finish_reason, str):
                raise TypeError(f"AA-LCR response has no finish reason: {path}")
            finish_reasons[finish_reason] += 1

            usage = response.get("usage")
            if not isinstance(usage, dict):
                raise TypeError(f"AA-LCR response has no usage object: {path}")
            request_prompt_tokens = usage.get("prompt_tokens")
            request_completion_tokens = usage.get("completion_tokens")
            total_tokens = usage.get("total_tokens")
            if request_prompt_tokens != token_counts[question.question_id]:
                raise RuntimeError(
                    f"AA-LCR response prompt-token count is invalid: {path}"
                )
            if (
                not isinstance(request_completion_tokens, int)
                or request_completion_tokens <= 0
                or total_tokens != request_prompt_tokens + request_completion_tokens
            ):
                raise RuntimeError(f"AA-LCR response token usage is invalid: {path}")
            elapsed = receipt.get("elapsed_seconds")
            if not isinstance(elapsed, (int, float)) or elapsed <= 0:
                raise RuntimeError(
                    f"AA-LCR generation receipt has invalid elapsed time: {path}"
                )

            prompt_tokens.append(request_prompt_tokens)
            completion_tokens.append(request_completion_tokens)
            elapsed_seconds.append(float(elapsed))
            per_repeat[repeat] += 1
            per_question[question.question_id] += 1
            receipt_hashes.append(
                (path.relative_to(args.generation_dir).as_posix(), sha256_file(path))
            )

    if finish_reasons != {"stop": EXPECTED_QUESTIONS * repeats}:
        raise RuntimeError(
            "Qualified AA-LCR generation requires every response to finish with stop: "
            f"{dict(finish_reasons)}"
        )
    if sorted(per_repeat.items()) != [(0, 100), (1, 100), (2, 100)]:
        raise RuntimeError("AA-LCR generation repeat counts are invalid")
    if any(count != repeats for count in per_question.values()):
        raise RuntimeError("AA-LCR generation question counts are invalid")

    def distribution(values: list[int] | list[float]) -> dict[str, int | float]:
        return {
            "minimum": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": round(percentile(values, 0.95), 6),
            "p99": round(percentile(values, 0.99), 6),
            "maximum": max(values),
            "sum": sum(values),
        }

    result = {
        "artifact_kind": "AA-LCR generation completeness receipt",
        "status": "qualified",
        "created_utc": utc_now(),
        "dataset": {
            "repository": dataset["repository"],
            "revision": dataset["revision"],
            "questions": EXPECTED_QUESTIONS,
            "prompt_manifest_sha256": dataset["prompt_manifest_sha256"],
            "token_count_manifest_sha256": token_count_sha256,
        },
        "generation": {
            "model": generation_config["model"],
            "configuration_sha256": generation_config_sha256,
            "runtime_manifest_sha256": runtime_manifest_sha256,
            "generation_manifest_sha256": sha256_file(generation_manifest_path),
            "questions": EXPECTED_QUESTIONS,
            "repeats": repeats,
            "receipts": len(receipt_hashes),
            "failure_sidecars": 0,
            "finish_reasons": dict(sorted(finish_reasons.items())),
            "per_repeat_receipts": {
                str(repeat): per_repeat[repeat] for repeat in sorted(per_repeat)
            },
        },
        "usage": {
            "prompt_tokens": distribution(prompt_tokens),
            "completion_tokens": distribution(completion_tokens),
            "request_elapsed_seconds": distribution(elapsed_seconds),
        },
        "receipt_manifest_sha256": canonical_sha256(sorted(receipt_hashes)),
        "qualification": {
            "all_expected_question_repeat_pairs_present": True,
            "all_receipts_match_dataset_prompts_and_documents": True,
            "all_receipts_match_generation_configuration": True,
            "all_candidate_answer_hashes_match": True,
            "all_response_prompt_token_counts_match_pinned_tokenizer": True,
            "all_responses_finished_with_stop": True,
        },
    }
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def build_judge_prompt(
    question: str, official_answer: str, candidate_answer: str
) -> str:
    return f"""Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question}
The OFFICIAL ANSWER: {official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT."""


def command_judge(args: argparse.Namespace) -> None:
    if args.concurrency <= 0:
        raise ValueError("AA-LCR equality-checker concurrency must be positive")
    if not 1 <= args.start_question < args.stop_question <= EXPECTED_QUESTIONS + 1:
        raise ValueError(
            "AA-LCR equality-checker question range must satisfy "
            f"1 <= start < stop <= {EXPECTED_QUESTIONS + 1}"
        )
    questions = {
        question.question_id: question for question in load_questions(args.dataset_root)
    }
    generation_manifest_path = args.generation_dir / "generation-manifest.json"
    if not generation_manifest_path.is_file():
        raise FileNotFoundError(
            f"AA-LCR generation manifest is absent: {generation_manifest_path}"
        )
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    generation_config_sha256 = generation_manifest.get("generation_config_sha256")
    if not isinstance(generation_config_sha256, str):
        raise TypeError(
            "AA-LCR generation manifest has no generation-configuration hash"
        )
    judge_runtime: dict[str, Any] | None = None
    judge_runtime_manifest_sha256: str | None = None
    if args.judge_runtime_manifest is not None:
        judge_runtime = json.loads(
            args.judge_runtime_manifest.read_text(encoding="utf-8")
        )
        if judge_runtime.get("status") != "qualified":
            raise RuntimeError(
                "Equality-checker runtime manifest is not qualified: "
                f"{args.judge_runtime_manifest}"
            )
        runtime_model = judge_runtime.get("serving", {}).get("served_model_name")
        if runtime_model != args.model:
            raise RuntimeError(
                "Equality-checker runtime model does not match --model: "
                f"{runtime_model!r} != {args.model!r}"
            )
        judge_runtime_manifest_sha256 = sha256_file(args.judge_runtime_manifest)
    elif args.judge_protocol == "frozen-official-kimi-k3":
        raise ValueError(
            "--judge-runtime-manifest is required for the frozen official "
            "Kimi-K3 judge protocol"
        )

    judge_config = {
        "protocol": args.judge_protocol,
        "protocol_description": JUDGE_PROTOCOLS[args.judge_protocol],
        "api_url": chat_completions_url(args.base_url),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt_contract": (
            "Compare the candidate answer with the official answer and emit only "
            "CORRECT or INCORRECT."
        ),
        "runtime_manifest_sha256": judge_runtime_manifest_sha256,
    }
    judge_manifest = {
        "artifact_kind": "AA-LCR equality-checker run",
        "status": "implemented",
        "created_utc": utc_now(),
        "dataset_revision": DATASET_REVISION,
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "generation_config_sha256": generation_config_sha256,
        "judge": judge_config,
        "judge_config_sha256": canonical_sha256(judge_config),
        "execution": {
            "client_concurrency": args.concurrency,
            "timeout_seconds": args.timeout_seconds,
        },
    }
    if judge_runtime is not None:
        judge_manifest["judge_runtime"] = {
            "checkpoint": judge_runtime.get("checkpoint"),
            "container": {
                "image": judge_runtime.get("container", {}).get("image"),
                "image_id": judge_runtime.get("container", {}).get("image_id"),
                "registry_digest": judge_runtime.get("container", {}).get(
                    "registry_digest"
                ),
            },
            "source": judge_runtime.get("source"),
            "topology": judge_runtime.get("topology"),
            "serving": judge_runtime.get("serving"),
        }
    manifest_path = args.output_dir / "judge-manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            existing.get("judge_config_sha256") != judge_manifest["judge_config_sha256"]
            or existing.get("generation_manifest_sha256")
            != judge_manifest["generation_manifest_sha256"]
        ):
            raise RuntimeError(
                f"Judge output directory has incompatible identity: {args.output_dir}"
            )
    else:
        write_json(manifest_path, judge_manifest)

    headers = api_headers(args.api_key_env)
    url = judge_config["api_url"]
    generation_receipts = sorted(
        path
        for path in (args.generation_dir / "responses").glob("repeat-*/question-*.json")
        if re.fullmatch(r"question-\d{4}\.json", path.name)
    )
    if not generation_receipts:
        raise RuntimeError(
            f"No generation receipts were found in {args.generation_dir / 'responses'}"
        )
    skipped = 0
    tasks: list[JudgementTask] = []
    for generation_path in generation_receipts:
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        if generation.get("status") != "qualified":
            raise RuntimeError(
                f"AA-LCR generation receipt is not qualified: {generation_path}"
            )
        if generation.get("generation_config_sha256") != generation_config_sha256:
            raise RuntimeError(
                "AA-LCR generation receipt has a mismatched generation-configuration "
                f"hash: {generation_path}"
            )
        question_id = int(generation["question_id"])
        repeat = int(generation["repeat"])
        if not args.start_question <= question_id < args.stop_question:
            continue
        if args.repeat is not None and repeat != args.repeat:
            continue
        if question_id not in questions:
            raise RuntimeError(
                f"AA-LCR generation receipt has an unknown question ID: {generation_path}"
            )
        question = questions[question_id]
        expected_generation_path = question_receipt_path(
            args.generation_dir, repeat, question_id
        )
        if generation_path != expected_generation_path:
            raise RuntimeError(
                "AA-LCR generation receipt path does not match its question and repeat: "
                f"{generation_path}"
            )
        candidate_answer = generation.get("candidate_answer")
        if not isinstance(candidate_answer, str) or not candidate_answer:
            raise RuntimeError(
                f"AA-LCR generation receipt has no candidate answer: {generation_path}"
            )
        output_path = judgement_receipt_path(args.output_dir, repeat, question_id)
        generation_sha256 = sha256_file(generation_path)
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            if (
                existing.get("status") == "qualified"
                and existing.get("generation_receipt_sha256") == generation_sha256
                and existing.get("judge_config_sha256")
                == judge_manifest["judge_config_sha256"]
            ):
                output_path.with_suffix(".error.json").unlink(missing_ok=True)
                skipped += 1
                continue
            raise RuntimeError(f"Judge receipt is incompatible: {output_path}")

        prompt = build_judge_prompt(
            question.question,
            question.official_answer,
            candidate_answer,
        )
        tasks.append(
            JudgementTask(
                question=question,
                repeat=repeat,
                generation_path=generation_path,
                generation_sha256=generation_sha256,
                prompt=prompt,
                output_path=output_path,
            )
        )

    def judge_one(task: JudgementTask) -> dict[str, Any]:
        failure_path = task.output_path.with_suffix(".error.json")
        request = {
            "model": args.model,
            "messages": [{"role": "user", "content": task.prompt}],
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "stream": False,
        }
        if args.temperature is not None:
            request["temperature"] = args.temperature
        try:
            response, elapsed = post_json(
                url=url,
                payload=request,
                headers=headers,
                timeout_seconds=args.timeout_seconds,
            )
            answer = choice_message(response).get("content")
            if not isinstance(answer, str):
                answer = ""
            match = JUDGE_OUTPUT_RE.fullmatch(answer.strip())
            if match is None:
                raise RuntimeError(
                    "Equality checker returned an invalid label for question "
                    f"{task.question.question_id}, repeat {task.repeat}: {answer!r}"
                )
            label = match.group(1).upper()
            write_json(
                task.output_path,
                {
                    "artifact_kind": "AA-LCR equality-checker receipt",
                    "status": "qualified",
                    "completed_utc": utc_now(),
                    "question_id": task.question.question_id,
                    "repeat": task.repeat,
                    "document_category": task.question.document_category,
                    "generation_receipt": task.generation_path.relative_to(
                        args.generation_dir
                    ).as_posix(),
                    "generation_receipt_sha256": task.generation_sha256,
                    "judge_protocol": args.judge_protocol,
                    "judge_config_sha256": judge_manifest["judge_config_sha256"],
                    "judge_prompt_sha256": sha256_text(task.prompt),
                    "elapsed_seconds": elapsed,
                    "label": label,
                    "correct": label == "CORRECT",
                    "response": response,
                },
            )
            failure_path.unlink(missing_ok=True)
            result = {
                "question_id": task.question.question_id,
                "repeat": task.repeat,
                "label": label,
            }
        except Exception as error:
            write_json(
                failure_path,
                {
                    "artifact_kind": "AA-LCR equality-checker failure",
                    "status": "unsupported",
                    "failed_utc": utc_now(),
                    "question_id": task.question.question_id,
                    "repeat": task.repeat,
                    "generation_receipt_sha256": task.generation_sha256,
                    "judge_protocol": args.judge_protocol,
                    "judge_config_sha256": judge_manifest["judge_config_sha256"],
                    "judge_prompt_sha256": sha256_text(task.prompt),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        print(json.dumps(result, sort_keys=True), flush=True)
        return result

    completed = 0
    if args.concurrency == 1:
        for task in tasks:
            judge_one(task)
            completed += 1
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(judge_one, task) for task in tasks]
            try:
                for future in as_completed(futures):
                    future.result()
                    completed += 1
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    print(json.dumps({"completed": completed, "skipped": skipped}, sort_keys=True))


def wilson_interval(
    successes: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float]:
    if total == 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (center - half_width, center + half_width)


def command_summarize(args: argparse.Namespace) -> None:
    judge_manifest_path = args.judge_dir / "judge-manifest.json"
    if not judge_manifest_path.is_file():
        raise FileNotFoundError(
            f"Equality-checker manifest is absent: {judge_manifest_path}"
        )
    judge_manifest = json.loads(judge_manifest_path.read_text(encoding="utf-8"))
    judge_config = judge_manifest.get("judge")
    if not isinstance(judge_config, dict):
        raise TypeError("Equality-checker manifest has no judge configuration")
    judge_config_sha256 = judge_manifest.get("judge_config_sha256")
    if judge_config_sha256 != canonical_sha256(judge_config):
        raise RuntimeError("Equality-checker manifest configuration hash is invalid")
    generation_manifest_path = args.generation_dir / "generation-manifest.json"
    if not generation_manifest_path.is_file():
        raise FileNotFoundError(
            f"AA-LCR generation manifest is absent: {generation_manifest_path}"
        )
    if sha256_file(generation_manifest_path) != judge_manifest.get(
        "generation_manifest_sha256"
    ):
        raise RuntimeError(
            "AA-LCR generation manifest does not match the equality-checker manifest"
        )
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    if generation_manifest.get("generation_config_sha256") != judge_manifest.get(
        "generation_config_sha256"
    ):
        raise RuntimeError(
            "AA-LCR generation-configuration identity does not match the "
            "equality-checker manifest"
        )
    questions = {
        question.question_id: question for question in load_questions(args.dataset_root)
    }
    receipts = sorted(
        path
        for path in (args.judge_dir / "judgements").glob("repeat-*/question-*.json")
        if re.fullmatch(r"question-\d{4}\.json", path.name)
    )
    if not receipts:
        raise RuntimeError(
            f"No equality-checker receipts were found in {args.judge_dir / 'judgements'}"
        )
    records = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
    attempt_pairs: set[tuple[int, int]] = set()
    completion_tokens: list[int] = []
    elapsed_seconds: list[float] = []
    for path, record in zip(receipts, records, strict=True):
        if record.get("status") != "qualified":
            raise RuntimeError(f"Equality-checker receipt is not qualified: {path}")
        if record.get("judge_config_sha256") != judge_config_sha256:
            raise RuntimeError(
                f"Equality-checker receipt has a mismatched configuration hash: {path}"
            )
        if record.get("judge_protocol") != judge_config.get("protocol"):
            raise RuntimeError(
                f"Equality-checker receipt has a mismatched judge protocol: {path}"
            )
        question_id = int(record["question_id"])
        repeat = int(record["repeat"])
        pair = (question_id, repeat)
        if pair in attempt_pairs:
            raise RuntimeError(
                "Equality-checker receipts contain a duplicate question-repeat pair: "
                f"{pair}"
            )
        attempt_pairs.add(pair)
        if question_id not in questions or repeat not in (0, 1, 2):
            raise RuntimeError(
                f"Equality-checker receipt has an invalid question-repeat pair: {path}"
            )
        if path != judgement_receipt_path(args.judge_dir, repeat, question_id):
            raise RuntimeError(
                "Equality-checker receipt path does not match its question and repeat: "
                f"{path}"
            )
        expected_generation_path = question_receipt_path(
            args.generation_dir, repeat, question_id
        )
        expected_generation_relative = expected_generation_path.relative_to(
            args.generation_dir
        ).as_posix()
        if record.get("generation_receipt") != expected_generation_relative:
            raise RuntimeError(
                f"Equality-checker receipt names the wrong generation receipt: {path}"
            )
        if not expected_generation_path.is_file():
            raise FileNotFoundError(
                f"AA-LCR generation receipt is absent: {expected_generation_path}"
            )
        if sha256_file(expected_generation_path) != record.get(
            "generation_receipt_sha256"
        ):
            raise RuntimeError(f"AA-LCR generation receipt hash does not match: {path}")
        generation = json.loads(expected_generation_path.read_text(encoding="utf-8"))
        candidate_answer = generation.get("candidate_answer")
        if not isinstance(candidate_answer, str) or not candidate_answer:
            raise RuntimeError(
                f"AA-LCR generation receipt has no candidate answer: {path}"
            )
        question = questions[question_id]
        prompt = build_judge_prompt(
            question.question,
            question.official_answer,
            candidate_answer,
        )
        if sha256_text(prompt) != record.get("judge_prompt_sha256"):
            raise RuntimeError(f"Equality-checker prompt hash does not match: {path}")
        response = record.get("response")
        if not isinstance(response, dict):
            raise TypeError(f"Equality-checker receipt has no API response: {path}")
        if response.get("model") != judge_config.get("model"):
            raise RuntimeError(
                f"Equality-checker response has a mismatched model: {path}"
            )
        message = choice_message(response)
        response_label = message.get("content")
        if not isinstance(response_label, str):
            raise TypeError(f"Equality-checker response has no textual label: {path}")
        match = JUDGE_OUTPUT_RE.fullmatch(response_label.strip())
        if match is None or match.group(1).upper() != record.get("label"):
            raise RuntimeError(
                f"Equality-checker response label does not match its receipt: {path}"
            )
        if bool(record.get("correct")) != (record.get("label") == "CORRECT"):
            raise RuntimeError(
                f"Equality-checker correctness flag does not match its label: {path}"
            )
        choices = response["choices"]
        if choices[0].get("finish_reason") != "stop":
            raise RuntimeError(
                f"Equality-checker response did not finish with stop: {path}"
            )
        usage = response.get("usage")
        if not isinstance(usage, dict) or not isinstance(
            usage.get("completion_tokens"), int
        ):
            raise TypeError(
                f"Equality-checker response has invalid completion-token usage: {path}"
            )
        completion_tokens.append(usage["completion_tokens"])
        elapsed_seconds.append(float(record["elapsed_seconds"]))
    correct = sum(bool(record["correct"]) for record in records)
    low, high = wilson_interval(correct, len(records))
    by_repeat: dict[int, list[bool]] = defaultdict(list)
    by_category: dict[str, list[bool]] = defaultdict(list)
    by_question: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        by_repeat[int(record["repeat"])].append(bool(record["correct"]))
        by_category[str(record["document_category"])].append(bool(record["correct"]))
        by_question[int(record["question_id"])].append(bool(record["correct"]))
    expected_pairs = {
        (question_id, repeat)
        for question_id in range(1, EXPECTED_QUESTIONS + 1)
        for repeat in range(3)
    }
    all_expected_pairs_present = attempt_pairs == expected_pairs
    failure_sidecars = sorted(
        (args.judge_dir / "judgements").glob("repeat-*/question-*.error.json")
    )
    complete = (
        len(records) == EXPECTED_QUESTIONS * 3
        and len(by_question) == EXPECTED_QUESTIONS
        and sorted(by_repeat) == [0, 1, 2]
        and all(len(values) == 3 for values in by_question.values())
        and all_expected_pairs_present
        and not failure_sidecars
    )

    def numeric_distribution(values: list[int] | list[float]) -> dict[str, int | float]:
        return {
            "minimum": min(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": round(percentile(values, 0.95), 6),
            "p99": round(percentile(values, 0.99), 6),
            "maximum": max(values),
            "sum": sum(values),
        }

    protocol = str(judge_config["protocol"])
    comparison_scope = (
        "Artificial Analysis AA-LCR v4.1.1 equality-checker result"
        if protocol == "artificial-analysis-v4.1.1"
        else (
            "Internal paired-comparison result produced by a frozen official "
            "Kimi-K3 judge; it is not an official Artificial Analysis result"
        )
    )
    summary = {
        "artifact_kind": "AA-LCR equality-checker pass@1 summary",
        "status": "qualified" if complete else "research-only",
        "created_utc": utc_now(),
        "comparison_scope": comparison_scope,
        "judge": {
            "protocol": protocol,
            "protocol_description": judge_config["protocol_description"],
            "model": judge_config["model"],
            "runtime_manifest_sha256": judge_config.get("runtime_manifest_sha256"),
        },
        "scoring": "mean correctness across all question-repeat attempts",
        "attempts": len(records),
        "correct": correct,
        "pass_at_1": correct / len(records),
        "wilson_95": {"low": low, "high": high},
        "questions": len(by_question),
        "repeats_observed": sorted(by_repeat),
        "per_repeat": {
            str(repeat): {
                "attempts": len(values),
                "correct": sum(values),
                "pass_at_1": sum(values) / len(values),
            }
            for repeat, values in sorted(by_repeat.items())
        },
        "per_document_category": {
            category: {
                "attempts": len(values),
                "correct": sum(values),
                "pass_at_1": sum(values) / len(values),
            }
            for category, values in sorted(by_category.items())
        },
        "question_mean_distribution": dict(
            sorted(
                Counter(
                    f"{sum(values) / len(values):.6f}"
                    for values in by_question.values()
                ).items()
            )
        ),
        "generation_repeat_agreement": {
            "unanimous_questions": sum(
                len(set(values)) == 1 for values in by_question.values()
            ),
            "mixed-label_questions": sum(
                len(set(values)) > 1 for values in by_question.values()
            ),
            "interpretation": (
                "Agreement includes candidate-generation variation and does not "
                "isolate equality-checker repeatability."
            ),
        },
        "judge_usage": {
            "completion_tokens": numeric_distribution(completion_tokens),
            "request_elapsed_seconds": numeric_distribution(elapsed_seconds),
        },
        "judge_manifest_sha256": sha256_file(judge_manifest_path),
        "receipt_manifest_sha256": canonical_sha256(
            [
                (path.relative_to(args.judge_dir).as_posix(), sha256_file(path))
                for path in receipts
            ]
        ),
        "qualification": {
            "all_expected_question_repeat_pairs_present": all_expected_pairs_present,
            "all_generation_receipt_hashes_match": True,
            "all_judge_prompt_hashes_match": True,
            "all_response_models_match": True,
            "all_response_labels_match_receipts": True,
            "all_responses_finished_with_stop": True,
            "failure_sidecars": len(failure_sidecars),
        },
    }
    if args.output:
        write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--document-root", type=Path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate pinned AA-LCR inputs")
    add_dataset_arguments(validate)
    validate.set_defaults(func=command_validate)

    token_counts = subparsers.add_parser(
        "token-counts", help="Record tokenizer counts for every AA-LCR prompt"
    )
    add_dataset_arguments(token_counts)
    token_counts.add_argument("--tokenizer", required=True)
    token_counts.add_argument("--tokenizer-revision")
    token_counts.add_argument("--output", type=Path, required=True)
    token_counts.set_defaults(func=command_token_counts)

    generate = subparsers.add_parser(
        "generate", help="Generate AA-LCR candidate answers"
    )
    add_dataset_arguments(generate)
    generate.add_argument(
        "--base-url",
        action="append",
        required=True,
        help=(
            "OpenAI-compatible endpoint. Repeat once per identical serving "
            "instance; questions are assigned by question ID modulo the endpoint count."
        ),
    )
    generate.add_argument("--model", required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--runtime-manifest", type=Path, required=True)
    generate.add_argument("--api-key-env", default=None)
    generate.add_argument("--repeats", type=int, default=3)
    generate.add_argument(
        "--concurrency-per-endpoint",
        type=int,
        default=1,
        help="Maximum number of generation workers assigned to each endpoint.",
    )
    generate.add_argument(
        "--repeat-scheduling",
        choices=("independent", "question_serial"),
        default="independent",
        help=(
            "Submit every attempt independently, or keep each question's repeats "
            "sequential within one client worker."
        ),
    )
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Model-author-supported output ceiling for the evaluated checkpoint.",
    )
    generate.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="max"
    )
    generate.add_argument("--timeout-seconds", type=float, default=7200)
    generate.add_argument("--start-question", type=int, default=1)
    generate.add_argument("--stop-question", type=int, default=101)
    generate.set_defaults(func=command_generate)

    verify_generations = subparsers.add_parser(
        "verify-generations",
        help="Verify and seal a complete AA-LCR generation artifact",
    )
    add_dataset_arguments(verify_generations)
    verify_generations.add_argument("--generation-dir", type=Path, required=True)
    verify_generations.add_argument("--token-count-manifest", type=Path)
    verify_generations.add_argument("--output", type=Path)
    verify_generations.set_defaults(func=command_verify_generations)

    judge = subparsers.add_parser(
        "judge", help="Score generated answers with an equality-checker model"
    )
    add_dataset_arguments(judge)
    judge.add_argument("--generation-dir", type=Path, required=True)
    judge.add_argument("--output-dir", type=Path, required=True)
    judge.add_argument("--base-url", required=True)
    judge.add_argument("--model", default="gpt-5.6-luna")
    judge.add_argument(
        "--judge-protocol",
        choices=tuple(JUDGE_PROTOCOLS),
        default="artificial-analysis-v4.1.1",
        help="Identity and interpretation boundary for the equality checker.",
    )
    judge.add_argument(
        "--judge-runtime-manifest",
        type=Path,
        help=(
            "Qualified serving-runtime manifest. Required when the frozen "
            "official Kimi-K3 checkpoint is the equality checker."
        ),
    )
    judge.add_argument("--api-key-env", default="OPENAI_API_KEY")
    judge.add_argument("--reasoning-effort", default="medium")
    judge.add_argument("--temperature", type=float)
    judge.add_argument("--max-tokens", type=int, default=4096)
    judge.add_argument("--timeout-seconds", type=float, default=600)
    judge.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of independent equality-checker requests submitted concurrently.",
    )
    judge.add_argument("--repeat", type=int, choices=(0, 1, 2))
    judge.add_argument("--start-question", type=int, default=1)
    judge.add_argument("--stop-question", type=int, default=101)
    judge.set_defaults(func=command_judge)

    summarize = subparsers.add_parser(
        "summarize", help="Aggregate equality-checker receipts"
    )
    summarize.add_argument("--dataset-root", type=Path, required=True)
    summarize.add_argument("--generation-dir", type=Path, required=True)
    summarize.add_argument("--judge-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    summarize.set_defaults(func=command_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
