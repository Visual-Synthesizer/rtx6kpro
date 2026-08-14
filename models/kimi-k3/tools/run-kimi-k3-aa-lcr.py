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
    records: list[dict[str, int]] = []
    for question in questions:
        prompt = build_prompt(resolver.resolve(question), question.question)
        raw_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        chat_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
        records.append(
            {
                "question_id": question.question_id,
                "reported_cl100k": question.reported_input_tokens,
                "kimi_raw": len(raw_tokens),
                "kimi_chat": len(chat_tokens),
            }
        )
    chat_lengths = [record["kimi_chat"] for record in records]
    result = {
        "artifact_kind": "AA-LCR Kimi K3 prompt token counts",
        "status": "implemented",
        "tokenizer_checkpoint": args.tokenizer,
        "tokenizer_revision": args.tokenizer_revision,
        "questions": len(records),
        "minimum": min(chat_lengths),
        "median": statistics.median(chat_lengths),
        "maximum": max(chat_lengths),
        "mean": statistics.mean(chat_lengths),
        "maximum_question_id": max(records, key=lambda record: record["kimi_chat"])[
            "question_id"
        ],
        "contexts": records,
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


def command_generate(args: argparse.Namespace) -> None:
    if args.repeats <= 0:
        raise ValueError("AA-LCR repeat count must be positive")
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
    generation_config = {
        "api_url": chat_completions_url(args.base_url),
        "model": args.model,
        "repeats": args.repeats,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "reasoning_effort": args.reasoning_effort,
        "request_seed": None,
        "system_message": None,
        "stream": False,
        "runtime_manifest_sha256": runtime_manifest_sha256,
    }
    run_manifest = {
        "artifact_kind": "Kimi K3 AA-LCR generation run",
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

    url = generation_config["api_url"]
    headers = api_headers(args.api_key_env)
    completed = 0
    skipped = 0
    for question in questions:
        documents = resolver.resolve(question)
        prompt = build_prompt(documents, question.question)
        prompt_sha256 = sha256_text(prompt)
        document_records = [
            {
                "requested_name": document.requested_name,
                "relative_path": document.relative_path,
                "sha256": document.sha256,
                "unicode_normalization_required": document.unicode_normalization_required,
            }
            for document in documents
        ]
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
                    skipped += 1
                    continue
                raise RuntimeError(
                    f"Generation receipt is incompatible: {receipt_path}"
                )

            request = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
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
                    "artifact_kind": "Kimi K3 AA-LCR generation receipt",
                    "status": "qualified",
                    "completed_utc": utc_now(),
                    "question_id": question.question_id,
                    "repeat": repeat,
                    "document_category": question.document_category,
                    "document_set_id": question.document_set_id,
                    "documents": document_records,
                    "question": question.question,
                    "official_answer_sha256": sha256_text(question.official_answer),
                    "reported_input_tokens_cl100k_base": question.reported_input_tokens,
                    "prompt_chars": len(prompt),
                    "prompt_sha256": prompt_sha256,
                    "generation_config_sha256": run_manifest[
                        "generation_config_sha256"
                    ],
                    "elapsed_seconds": elapsed,
                    "candidate_answer": candidate_answer,
                    "candidate_answer_sha256": sha256_text(candidate_answer),
                    "response": response,
                }
                write_json(receipt_path, receipt)
                completed += 1
                usage = response.get("usage", {})
                print(
                    json.dumps(
                        {
                            "question_id": question.question_id,
                            "repeat": repeat,
                            "elapsed_seconds": elapsed,
                            "prompt_tokens": usage.get("prompt_tokens"),
                            "completion_tokens": usage.get("completion_tokens"),
                            "finish_reason": response["choices"][0].get(
                                "finish_reason"
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as error:
                failure_path = receipt_path.with_suffix(".error.json")
                write_json(
                    failure_path,
                    {
                        "artifact_kind": "Kimi K3 AA-LCR generation failure",
                        "status": "unsupported",
                        "failed_utc": utc_now(),
                        "question_id": question.question_id,
                        "repeat": repeat,
                        "prompt_sha256": prompt_sha256,
                        "generation_config_sha256": run_manifest[
                            "generation_config_sha256"
                        ],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise
    print(json.dumps({"completed": completed, "skipped": skipped}, sort_keys=True))


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
    questions = {
        question.question_id: question for question in load_questions(args.dataset_root)
    }
    generation_manifest_path = args.generation_dir / "generation-manifest.json"
    generation_manifest = json.loads(
        generation_manifest_path.read_text(encoding="utf-8")
    )
    judge_config = {
        "api_url": chat_completions_url(args.base_url),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "prompt": "AA-LCR equality checker from Artificial Analysis methodology v4.1.1",
    }
    judge_manifest = {
        "artifact_kind": "AA-LCR equality-checker run",
        "status": "implemented",
        "created_utc": utc_now(),
        "dataset_revision": DATASET_REVISION,
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "generation_config_sha256": generation_manifest["generation_config_sha256"],
        "judge": judge_config,
        "judge_config_sha256": canonical_sha256(judge_config),
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
        (args.generation_dir / "responses").glob("repeat-*/question-*.json")
    )
    if not generation_receipts:
        raise RuntimeError(
            f"No generation receipts were found in {args.generation_dir / 'responses'}"
        )
    for generation_path in generation_receipts:
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        question_id = int(generation["question_id"])
        repeat = int(generation["repeat"])
        question = questions[question_id]
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
                continue
            raise RuntimeError(f"Judge receipt is incompatible: {output_path}")

        prompt = build_judge_prompt(
            question.question,
            question.official_answer,
            generation["candidate_answer"],
        )
        request = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": args.max_tokens,
            "reasoning_effort": args.reasoning_effort,
            "stream": False,
        }
        if args.temperature is not None:
            request["temperature"] = args.temperature
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
                f"Equality checker returned an invalid label for question {question_id}, "
                f"repeat {repeat}: {answer!r}"
            )
        label = match.group(1).upper()
        write_json(
            output_path,
            {
                "artifact_kind": "AA-LCR equality-checker receipt",
                "status": "qualified",
                "completed_utc": utc_now(),
                "question_id": question_id,
                "repeat": repeat,
                "document_category": question.document_category,
                "generation_receipt": generation_path.relative_to(
                    args.generation_dir
                ).as_posix(),
                "generation_receipt_sha256": generation_sha256,
                "judge_config_sha256": judge_manifest["judge_config_sha256"],
                "judge_prompt_sha256": sha256_text(prompt),
                "elapsed_seconds": elapsed,
                "label": label,
                "correct": label == "CORRECT",
                "response": response,
            },
        )
        print(
            json.dumps(
                {"question_id": question_id, "repeat": repeat, "label": label},
                sort_keys=True,
            ),
            flush=True,
        )


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
    receipts = sorted((args.judge_dir / "judgements").glob("repeat-*/question-*.json"))
    if not receipts:
        raise RuntimeError(
            f"No equality-checker receipts were found in {args.judge_dir / 'judgements'}"
        )
    records = [json.loads(path.read_text(encoding="utf-8")) for path in receipts]
    correct = sum(bool(record["correct"]) for record in records)
    low, high = wilson_interval(correct, len(records))
    by_repeat: dict[int, list[bool]] = defaultdict(list)
    by_category: dict[str, list[bool]] = defaultdict(list)
    by_question: dict[int, list[bool]] = defaultdict(list)
    for record in records:
        by_repeat[int(record["repeat"])].append(bool(record["correct"]))
        by_category[str(record["document_category"])].append(bool(record["correct"]))
        by_question[int(record["question_id"])].append(bool(record["correct"]))
    complete = (
        len(records) == EXPECTED_QUESTIONS * 3
        and len(by_question) == EXPECTED_QUESTIONS
        and sorted(by_repeat) == [0, 1, 2]
        and all(len(values) == 3 for values in by_question.values())
    )
    summary = {
        "artifact_kind": "AA-LCR pass@1 summary",
        "status": "qualified" if complete else "research-only",
        "created_utc": utc_now(),
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
        "judge_manifest_sha256": sha256_file(args.judge_dir / "judge-manifest.json"),
        "receipt_manifest_sha256": canonical_sha256(
            [
                (path.relative_to(args.judge_dir).as_posix(), sha256_file(path))
                for path in receipts
            ]
        ),
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
        "token-counts", help="Record Kimi K3 token counts for every AA-LCR prompt"
    )
    add_dataset_arguments(token_counts)
    token_counts.add_argument("--tokenizer", default="moonshotai/Kimi-K3")
    token_counts.add_argument(
        "--tokenizer-revision",
        default="2496450e92e425c886db095102a52a6682ca3970",
    )
    token_counts.add_argument("--output", type=Path, required=True)
    token_counts.set_defaults(func=command_token_counts)

    generate = subparsers.add_parser(
        "generate", help="Generate AA-LCR candidate answers"
    )
    add_dataset_arguments(generate)
    generate.add_argument("--base-url", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.add_argument("--runtime-manifest", type=Path, required=True)
    generate.add_argument("--api-key-env", default=None)
    generate.add_argument("--repeats", type=int, default=3)
    generate.add_argument("--temperature", type=float, default=1.0)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--max-tokens", type=int, default=200000)
    generate.add_argument(
        "--reasoning-effort", choices=("low", "high", "max"), default="max"
    )
    generate.add_argument("--timeout-seconds", type=float, default=7200)
    generate.add_argument("--start-question", type=int, default=1)
    generate.add_argument("--stop-question", type=int, default=101)
    generate.set_defaults(func=command_generate)

    judge = subparsers.add_parser(
        "judge", help="Score generated answers with an equality-checker model"
    )
    add_dataset_arguments(judge)
    judge.add_argument("--generation-dir", type=Path, required=True)
    judge.add_argument("--output-dir", type=Path, required=True)
    judge.add_argument("--base-url", required=True)
    judge.add_argument("--model", default="gpt-5.6-luna")
    judge.add_argument("--api-key-env", default="OPENAI_API_KEY")
    judge.add_argument("--reasoning-effort", default="medium")
    judge.add_argument("--temperature", type=float)
    judge.add_argument("--max-tokens", type=int, default=4096)
    judge.add_argument("--timeout-seconds", type=float, default=600)
    judge.set_defaults(func=command_judge)

    summarize = subparsers.add_parser(
        "summarize", help="Aggregate equality-checker receipts"
    )
    summarize.add_argument("--judge-dir", type=Path, required=True)
    summarize.add_argument("--output", type=Path)
    summarize.set_defaults(func=command_summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
