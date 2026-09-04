#!/usr/bin/env python3
"""Judge pinned AA-LCR generations through an authenticated Codex CLI."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DATASET_REVISION = "bdae010bbce259820c0e34c1d7cce210d966fb75"
DATASET_CSV_SHA256 = "2f90d9c30cfb4dd8df2c0f46547c384065e4c76917bd347a9a97bf797235c1ea"
EXPECTED_QUESTIONS = 100
EXPECTED_REPEATS = 3
LABEL_RE = re.compile(r"^(CORRECT|INCORRECT)[.!]?$", re.IGNORECASE)


@dataclass(frozen=True)
class Question:
    """Question text and answer used by one equality-checker prompt."""

    question_id: int
    question: str
    official_answer: str


@dataclass(frozen=True)
class Task:
    """One independently resumable generation judgement."""

    question: Question
    repeat: int
    generation_path: Path
    generation_sha256: str
    candidate_answer_sha256: str
    prompt: str
    prompt_sha256: str
    receipt_path: Path


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(payload: str) -> str:
    return sha256_bytes(payload.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def load_questions(dataset_root: Path) -> dict[int, Question]:
    csv_path = dataset_root / "AA-LCR_Dataset.csv"
    if sha256_file(csv_path) != DATASET_CSV_SHA256:
        raise RuntimeError(
            f"AA-LCR CSV does not match revision {DATASET_REVISION}: {csv_path}"
        )
    questions: dict[int, Question] = {}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = int(row["question_id"])
            questions[question_id] = Question(
                question_id=question_id,
                question=row["question"],
                official_answer=row["answer"],
            )
    if sorted(questions) != list(range(1, EXPECTED_QUESTIONS + 1)):
        raise RuntimeError("AA-LCR CSV must contain question IDs 1 through 100")
    return questions


def build_prompt(question: Question, candidate_answer: str) -> str:
    """Build the equality-checker prompt specified by AA-LCR v4.1.1."""

    return f"""Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.
For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.

The question, for reference only: {question.question}
The OFFICIAL ANSWER: {question.official_answer}
CANDIDATE ANSWER TO ASSESS: {candidate_answer}

Reply only with CORRECT or INCORRECT."""


def generation_path(generation_dir: Path, repeat: int, question_id: int) -> Path:
    return (
        generation_dir
        / "responses"
        / f"repeat-{repeat:02d}"
        / (f"question-{question_id:04d}.json")
    )


def receipt_path(output_dir: Path, repeat: int, question_id: int) -> Path:
    return (
        output_dir
        / "judgements"
        / f"repeat-{repeat:02d}"
        / (f"question-{question_id:04d}.json")
    )


def log_paths(output_dir: Path, repeat: int, question_id: int) -> tuple[Path, Path]:
    root = output_dir / "logs" / f"repeat-{repeat:02d}"
    stem = f"question-{question_id:04d}"
    return root / f"{stem}.stdout.log", root / f"{stem}.stderr.log"


def load_task(
    *,
    question: Question,
    repeat: int,
    generation_dir: Path,
    output_dir: Path,
) -> Task:
    source = generation_path(generation_dir, repeat, question.question_id)
    generation = json.loads(source.read_text(encoding="utf-8"))
    if generation.get("status") != "qualified":
        raise RuntimeError(f"generation receipt is not qualified: {source}")
    if generation.get("question_id") != question.question_id:
        raise RuntimeError(f"generation question identity does not match: {source}")
    if generation.get("repeat") != repeat:
        raise RuntimeError(f"generation repeat identity does not match: {source}")
    candidate_answer = generation.get("candidate_answer")
    if not isinstance(candidate_answer, str) or not candidate_answer:
        raise RuntimeError(f"generation receipt has no candidate answer: {source}")
    candidate_hash = sha256_text(candidate_answer)
    if generation.get("candidate_answer_sha256") != candidate_hash:
        raise RuntimeError(f"candidate-answer hash does not match: {source}")
    prompt = build_prompt(question, candidate_answer)
    return Task(
        question=question,
        repeat=repeat,
        generation_path=source,
        generation_sha256=sha256_file(source),
        candidate_answer_sha256=candidate_hash,
        prompt=prompt,
        prompt_sha256=sha256_text(prompt),
        receipt_path=receipt_path(output_dir, repeat, question.question_id),
    )


def validate_existing_receipt(task: Task, config_sha256: str) -> bool:
    if not task.receipt_path.is_file():
        return False
    receipt = json.loads(task.receipt_path.read_text(encoding="utf-8"))
    expected = {
        "status": "qualified",
        "question_id": task.question.question_id,
        "repeat": task.repeat,
        "generation_receipt_sha256": task.generation_sha256,
        "candidate_answer_sha256": task.candidate_answer_sha256,
        "judge_prompt_sha256": task.prompt_sha256,
        "judge_config_sha256": config_sha256,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise RuntimeError(
                f"existing Codex judgement has mismatched {key}: {task.receipt_path}"
            )
    if receipt.get("label") not in {"CORRECT", "INCORRECT"}:
        raise RuntimeError(
            f"existing Codex judgement has no label: {task.receipt_path}"
        )
    return True


def codex_version(codex_bin: str) -> str:
    completed = subprocess.run(
        [codex_bin, "--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def run_task(
    task: Task,
    *,
    output_dir: Path,
    workdir: Path,
    codex_bin: str,
    model: str,
    reasoning_effort: str,
    timeout_seconds: float,
    config_sha256: str,
) -> dict[str, Any]:
    stdout_path, stderr_path = log_paths(
        output_dir,
        task.repeat,
        task.question.question_id,
    )
    final_path = task.receipt_path.with_suffix(".final.txt")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        codex_bin,
        "exec",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--ephemeral",
        "--model",
        model,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--sandbox",
        "read-only",
        "--cd",
        str(workdir),
        "--color",
        "never",
        "--output-last-message",
        str(final_path),
        "-",
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=task.prompt,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        elapsed = time.monotonic() - started
        write_text_atomic(stdout_path, completed.stdout)
        write_text_atomic(stderr_path, completed.stderr)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Codex exited with status {completed.returncode}; "
                f"stderr_sha256={sha256_file(stderr_path)}"
            )
        final = final_path.read_text(encoding="utf-8").strip()
        match = LABEL_RE.fullmatch(final)
        if match is None:
            raise RuntimeError(f"Codex returned an invalid judge label: {final!r}")
        label = match.group(1).upper()
        receipt = {
            "artifact_kind": "AA-LCR Codex equality-checker receipt",
            "status": "qualified",
            "completed_utc": utc_now(),
            "question_id": task.question.question_id,
            "repeat": task.repeat,
            "generation_receipt": task.generation_path.relative_to(
                task.generation_path.parents[2]
            ).as_posix(),
            "generation_receipt_sha256": task.generation_sha256,
            "candidate_answer_sha256": task.candidate_answer_sha256,
            "judge_prompt_sha256": task.prompt_sha256,
            "judge_config_sha256": config_sha256,
            "label": label,
            "correct": label == "CORRECT",
            "elapsed_seconds": elapsed,
            "codex_stdout_sha256": sha256_file(stdout_path),
            "codex_stderr_sha256": sha256_file(stderr_path),
            "codex_final_message_sha256": sha256_text(final),
        }
        write_json_atomic(task.receipt_path, receipt)
        task.receipt_path.with_suffix(".error.json").unlink(missing_ok=True)
        return receipt
    except BaseException as error:
        write_json_atomic(
            task.receipt_path.with_suffix(".error.json"),
            {
                "artifact_kind": "AA-LCR Codex equality-checker failure",
                "status": "unsupported",
                "failed_utc": utc_now(),
                "question_id": task.question.question_id,
                "repeat": task.repeat,
                "generation_receipt_sha256": task.generation_sha256,
                "judge_prompt_sha256": task.prompt_sha256,
                "judge_config_sha256": config_sha256,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        final_path.unlink(missing_ok=True)


def load_reference_labels(
    reference_dir: Path,
) -> tuple[dict[tuple[int, int], str], str]:
    manifest = reference_dir / "judge-manifest.json"
    labels: dict[tuple[int, int], str] = {}
    receipt_hashes: list[tuple[str, str]] = []
    for path in sorted(
        (reference_dir / "judgements").glob("repeat-*/question-????.json")
    ):
        receipt = json.loads(path.read_text(encoding="utf-8"))
        key = (int(receipt["question_id"]), int(receipt["repeat"]))
        label = str(receipt["label"])
        if key in labels or label not in {"CORRECT", "INCORRECT"}:
            raise RuntimeError(f"reference judge receipt is invalid: {path}")
        labels[key] = label
        receipt_hashes.append(
            (path.relative_to(reference_dir).as_posix(), sha256_file(path))
        )
    if len(labels) != EXPECTED_QUESTIONS * EXPECTED_REPEATS:
        raise RuntimeError("reference judge does not contain 300 qualified labels")
    identity = canonical_sha256(
        {
            "judge_manifest_sha256": sha256_file(manifest),
            "receipts": receipt_hashes,
        }
    )
    return labels, identity


def summarize(
    *,
    output_dir: Path,
    config: dict[str, Any],
    config_sha256: str,
    reference_dir: Path | None,
) -> dict[str, Any]:
    paths = sorted((output_dir / "judgements").glob("repeat-*/question-????.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    expected_pairs = {
        (question_id, repeat)
        for question_id in range(1, EXPECTED_QUESTIONS + 1)
        for repeat in range(EXPECTED_REPEATS)
    }
    pairs: dict[tuple[int, int], dict[str, Any]] = {}
    for path, record in zip(paths, records, strict=True):
        key = (int(record["question_id"]), int(record["repeat"]))
        if key in pairs:
            raise RuntimeError(f"duplicate Codex judge receipt: {path}")
        if record.get("status") != "qualified":
            raise RuntimeError(f"Codex judge receipt is not qualified: {path}")
        if record.get("judge_config_sha256") != config_sha256:
            raise RuntimeError(f"Codex judge receipt configuration drifted: {path}")
        pairs[key] = record
    failures = sorted(
        (output_dir / "judgements").glob("repeat-*/question-????.error.json")
    )
    complete = set(pairs) == expected_pairs and not failures
    correct = sum(bool(record["correct"]) for record in records)
    by_repeat: dict[int, list[bool]] = defaultdict(list)
    elapsed: list[float] = []
    for record in records:
        by_repeat[int(record["repeat"])].append(bool(record["correct"]))
        elapsed.append(float(record["elapsed_seconds"]))
    result: dict[str, Any] = {
        "artifact_kind": "AA-LCR Codex equality-checker summary",
        "status": "qualified" if complete else "research-only",
        "created_utc": utc_now(),
        "comparison_scope": (
            "Artificial Analysis AA-LCR v4.1.1 equality-checker reproduction"
        ),
        "judge": config,
        "judge_config_sha256": config_sha256,
        "attempts": len(records),
        "correct": correct,
        "pass_at_1": correct / len(records) if records else None,
        "per_repeat": {
            str(repeat): {
                "attempts": len(values),
                "correct": sum(values),
                "pass_at_1": sum(values) / len(values),
            }
            for repeat, values in sorted(by_repeat.items())
        },
        "elapsed_seconds": {
            "minimum": min(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "maximum": max(elapsed) if elapsed else None,
            "sum": sum(elapsed),
        },
        "qualification": {
            "all_expected_question_repeat_pairs_present": set(pairs) == expected_pairs,
            "failure_sidecars": len(failures),
            "all_receipts_match_judge_configuration": all(
                record.get("judge_config_sha256") == config_sha256 for record in records
            ),
        },
        "receipt_manifest_sha256": canonical_sha256(
            [
                (path.relative_to(output_dir).as_posix(), sha256_file(path))
                for path in paths
            ]
        ),
    }
    if reference_dir is not None:
        reference, reference_identity = load_reference_labels(reference_dir)
        common = sorted(set(reference) & set(pairs))
        both_correct = sum(
            reference[key] == "CORRECT" and pairs[key]["label"] == "CORRECT"
            for key in common
        )
        reference_only = sum(
            reference[key] == "CORRECT" and pairs[key]["label"] == "INCORRECT"
            for key in common
        )
        codex_only = sum(
            reference[key] == "INCORRECT" and pairs[key]["label"] == "CORRECT"
            for key in common
        )
        both_incorrect = sum(
            reference[key] == "INCORRECT" and pairs[key]["label"] == "INCORRECT"
            for key in common
        )
        agreement = both_correct + both_incorrect
        result["reference_comparison"] = {
            "semantic_role": "Frozen official Kimi-K3 equality-checker control",
            "reference_dir": str(reference_dir.resolve()),
            "reference_identity_sha256": reference_identity,
            "paired_attempts": len(common),
            "label_agreement": agreement,
            "label_agreement_rate": agreement / len(common) if common else None,
            "both_correct": both_correct,
            "reference_only_correct": reference_only,
            "codex_only_correct": codex_only,
            "both_incorrect": both_incorrect,
        }
    write_json_atomic(output_dir / "pass-at-1-summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--generation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-judge-dir", type=Path)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="medium",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--start-question", type=int, default=1)
    parser.add_argument("--stop-question", type=int, default=101)
    args = parser.parse_args()

    if args.concurrency < 1:
        raise ValueError("Codex judge concurrency must be positive")
    if not 1 <= args.start_question < args.stop_question <= 101:
        raise ValueError("question range must satisfy 1 <= start < stop <= 101")
    questions = load_questions(args.dataset_root)
    generation_manifest = args.generation_dir / "generation-manifest.json"
    cli_version = codex_version(args.codex_bin)
    config = {
        "provider": "OpenAI Codex CLI authenticated with ChatGPT",
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "codex_cli_version": cli_version,
        "prompt_contract": (
            "Frozen AA-LCR equality-checker prompt; emit only CORRECT or INCORRECT"
        ),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "dataset_revision": DATASET_REVISION,
        "execution_isolation": {
            "ephemeral_session": True,
            "user_config_loaded": False,
            "sandbox": "read-only",
            "workspace": "empty",
        },
    }
    config_sha256 = canonical_sha256(config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    workdir = args.output_dir / "empty-workspace"
    workdir.mkdir(exist_ok=True)
    manifest_path = args.output_dir / "judge-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("judge_config_sha256") != config_sha256:
            raise RuntimeError("existing Codex judge manifest configuration drifted")
    else:
        write_json_atomic(
            manifest_path,
            {
                "artifact_kind": "AA-LCR Codex equality-checker run",
                "status": "implemented",
                "created_utc": utc_now(),
                "judge": config,
                "judge_config_sha256": config_sha256,
                "client_concurrency": args.concurrency,
                "timeout_seconds": args.timeout_seconds,
            },
        )

    tasks: list[Task] = []
    skipped = 0
    for repeat in range(EXPECTED_REPEATS):
        for question_id in range(args.start_question, args.stop_question):
            task = load_task(
                question=questions[question_id],
                repeat=repeat,
                generation_dir=args.generation_dir,
                output_dir=args.output_dir,
            )
            if validate_existing_receipt(task, config_sha256):
                skipped += 1
            else:
                tasks.append(task)

    def execute(task: Task) -> dict[str, Any]:
        result = run_task(
            task,
            output_dir=args.output_dir,
            workdir=workdir,
            codex_bin=args.codex_bin,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.timeout_seconds,
            config_sha256=config_sha256,
        )
        print(
            json.dumps(
                {
                    "label": result["label"],
                    "question_id": result["question_id"],
                    "repeat": result["repeat"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return result

    if args.concurrency == 1:
        for task in tasks:
            execute(task)
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(execute, task) for task in tasks]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                for future in futures:
                    future.cancel()
                raise

    result = summarize(
        output_dir=args.output_dir,
        config=config,
        config_sha256=config_sha256,
        reference_dir=args.reference_judge_dir,
    )
    print(json.dumps({"completed": len(tasks), "skipped": skipped}, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
