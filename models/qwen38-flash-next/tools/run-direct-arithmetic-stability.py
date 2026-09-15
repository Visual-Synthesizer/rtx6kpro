#!/usr/bin/env python3
"""Measure direct-answer arithmetic stability across served model variants.

The program creates a deterministic suite of unique integer-arithmetic tasks,
sends each task to OpenAI-compatible Chat Completions endpoints, and scores the
returned text against an algorithmically computed exact answer. It is intended
to test the narrow behavior of arithmetic with reasoning disabled or reduced;
it is not a general mathematics benchmark.

Evidence is written incrementally as JSON Lines. Re-running the same command
resumes missing attempts, while configuration or suite changes are rejected.
Python 3.10 or newer is required. Only the standard library is used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import threading
import time
import urllib.error
import urllib.request


SUITE_SCHEMA = "local-inference-lab.direct-arithmetic-stability-suite.v1"
RUN_SCHEMA = "local-inference-lab.direct-arithmetic-stability-run.v1"
ATTEMPT_SCHEMA = "local-inference-lab.direct-arithmetic-stability-attempt.v1"
INTEGER_RE = re.compile(r"^-?[0-9]+$")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    operation: str
    operands: tuple[int, ...]
    expression: str
    expected: str
    prompt: str
    difficulty: dict[str, int]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def multiplication_carries(left: int, right: int) -> int:
    """Count nonzero carries in the conventional row-wise multiplication."""
    carries = 0
    right_digits = [int(char) for char in str(right)][::-1]
    for left_digit in [int(char) for char in str(left)][::-1]:
        carry = 0
        for right_digit in right_digits:
            carry = (left_digit * right_digit + carry) // 10
            carries += carry > 0
        carries += carry > 0
    return carries


def addition_carries(left: int, right: int) -> int:
    carries = 0
    carry = 0
    left_digits = [int(char) for char in str(left)][::-1]
    right_digits = [int(char) for char in str(right)][::-1]
    width = max(len(left_digits), len(right_digits))
    for position in range(width):
        a = left_digits[position] if position < len(left_digits) else 0
        b = right_digits[position] if position < len(right_digits) else 0
        carry = (a + b + carry) // 10
        carries += carry > 0
    return carries


def subtraction_borrows(left: int, right: int) -> int:
    if left < right:
        raise ValueError("borrow counter requires a nonnegative result")
    borrows = 0
    borrow = 0
    left_digits = [int(char) for char in str(left)][::-1]
    right_digits = [int(char) for char in str(right)][::-1]
    for position, a in enumerate(left_digits):
        b = right_digits[position] if position < len(right_digits) else 0
        needs_borrow = a - borrow < b
        borrows += needs_borrow
        borrow = int(needs_borrow)
    return borrows


def make_task(
    family: str,
    ordinal: int,
    operation: str,
    operands: tuple[int, ...],
    expression: str,
    answer: int,
    difficulty: dict[str, int],
) -> Task:
    task_id = f"{family}-{ordinal:03d}"
    prompt = (
        f"Case direct-arithmetic-{task_id}. Compute {expression}. "
        "Give only the integer as your final answer."
    )
    return Task(
        task_id=task_id,
        family=family,
        operation=operation,
        operands=operands,
        expression=expression,
        expected=str(answer),
        prompt=prompt,
        difficulty=difficulty,
    )


def generate_suite(seed: int, tasks_per_family: int) -> list[Task]:
    """Generate balanced unique tasks without using a model-authored answer key."""
    rng = random.Random(seed)
    tasks: list[Task] = []

    def collect(family: str, producer) -> None:
        seen: set[tuple[object, ...]] = set()
        ordinal = 1
        while ordinal <= tasks_per_family:
            operation, operands, expression, answer, difficulty = producer()
            signature = (operation, operands)
            if signature in seen:
                continue
            seen.add(signature)
            tasks.append(
                make_task(
                    family,
                    ordinal,
                    operation,
                    operands,
                    expression,
                    answer,
                    difficulty,
                )
            )
            ordinal += 1

    def near_power_product():
        digits = rng.choice((2, 3, 4, 5))
        offset = rng.choice((1, 2, 3, 7, 8, 9, 11, 17, 19, 23))
        left = 10**digits - offset
        right = rng.randint(11, 99)
        return (
            "multiply",
            (left, right),
            f"{left} times {right}",
            left * right,
            {
                "operand_digits": len(str(left)) + len(str(right)),
                "multiplication_carries": multiplication_carries(left, right),
                "distance_from_power_of_ten": offset,
            },
        )

    def carry_product():
        while True:
            left = rng.randint(67, 9999)
            right = rng.randint(17, 999)
            carry_count = multiplication_carries(left, right)
            if carry_count >= 3:
                return (
                    "multiply",
                    (left, right),
                    f"{left} times {right}",
                    left * right,
                    {
                        "operand_digits": len(str(left)) + len(str(right)),
                        "multiplication_carries": carry_count,
                    },
                )

    def carry_addition():
        while True:
            suffix_digits = rng.randint(2, 6)
            prefix = rng.randint(11, 999)
            left = prefix * 10**suffix_digits + (10**suffix_digits - 1)
            right = rng.randint(1, 10 ** min(4, suffix_digits) - 1)
            carry_count = addition_carries(left, right)
            if carry_count >= 2:
                return (
                    "add",
                    (left, right),
                    f"{left} plus {right}",
                    left + right,
                    {
                        "operand_digits": len(str(left)) + len(str(right)),
                        "addition_carries": carry_count,
                    },
                )

    def borrow_subtraction():
        while True:
            zero_run = rng.randint(2, 6)
            prefix = rng.randint(11, 999)
            left = prefix * 10**zero_run
            right = rng.randint(1, 10 ** min(4, zero_run) - 1)
            borrow_count = subtraction_borrows(left, right)
            if borrow_count >= 2:
                return (
                    "subtract",
                    (left, right),
                    f"{left} minus {right}",
                    left - right,
                    {
                        "operand_digits": len(str(left)) + len(str(right)),
                        "subtraction_borrows": borrow_count,
                    },
                )

    def exact_division():
        divisor = rng.randint(11, 999)
        quotient = rng.randint(11, 99999)
        dividend = divisor * quotient
        return (
            "divide_exactly",
            (dividend, divisor),
            f"{dividend} divided by {divisor}",
            quotient,
            {
                "operand_digits": len(str(dividend)) + len(str(divisor)),
                "quotient_digits": len(str(quotient)),
            },
        )

    def two_operation_expression():
        left = rng.randint(17, 999)
        right = rng.randint(11, 999)
        adjustment = rng.randint(11, 9999)
        if rng.randrange(2):
            expression = f"({left} times {right}) plus {adjustment}"
            answer = left * right + adjustment
            operation = "multiply_then_add"
        else:
            product = left * right
            adjustment = min(adjustment, product - 1)
            expression = f"({left} times {right}) minus {adjustment}"
            answer = product - adjustment
            operation = "multiply_then_subtract"
        return (
            operation,
            (left, right, adjustment),
            expression,
            answer,
            {
                "operand_digits": sum(len(str(value)) for value in (left, right, adjustment)),
                "multiplication_carries": multiplication_carries(left, right),
            },
        )

    collect("near-power-product", near_power_product)
    collect("carry-product", carry_product)
    collect("carry-addition", carry_addition)
    collect("borrow-subtraction", borrow_subtraction)
    collect("exact-division", exact_division)
    collect("two-operation", two_operation_expression)

    task_ids = [task.task_id for task in tasks]
    prompts = [task.prompt for task in tasks]
    if len(set(task_ids)) != len(tasks) or len(set(prompts)) != len(tasks):
        raise AssertionError("generated task identifiers and prompts must be unique")
    if any(task.prompt.startswith("Case stability3-effort-76.") for task in tasks):
        raise AssertionError("the external sentinel prompt must not enter the generalization suite")
    return tasks


def parse_endpoint_set(value: str) -> tuple[str, tuple[str, ...]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("endpoint set must use LABEL=URL,URL syntax")
    label, raw_urls = value.split("=", 1)
    label = label.strip()
    urls = tuple(url.strip().rstrip("/") for url in raw_urls.split(",") if url.strip())
    if not label or not urls:
        raise argparse.ArgumentTypeError("endpoint set requires a nonempty label and at least one URL")
    if any(not url.startswith(("http://", "https://")) for url in urls):
        raise argparse.ArgumentTypeError("every endpoint must start with http:// or https://")
    normalized = tuple(url if url.endswith("/v1") else url + "/v1" for url in urls)
    return label, normalized


def write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")


def load_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as source:
        return json.load(source)


def attempt_key(row: dict[str, object]) -> str:
    return "|".join(
        str(row[field])
        for field in ("model_label", "reasoning_effort", "temperature", "task_id", "repeat")
    )


def classify_answer(text: str, expected: str) -> tuple[bool, str]:
    stripped = text.strip()
    if stripped == expected:
        return True, "exact"
    if not stripped:
        return False, "empty"
    if INTEGER_RE.fullmatch(stripped):
        if len(stripped.removeprefix("-")) < len(expected.removeprefix("-")):
            return False, "incorrect_integer_shorter"
        if len(stripped.removeprefix("-")) > len(expected.removeprefix("-")):
            return False, "incorrect_integer_longer"
        return False, "incorrect_integer_same_length"
    return False, "non_integer_format"


def request_attempt(
    job: dict[str, object],
    endpoint: str,
    model: str,
    max_tokens: int,
    timeout: float,
    headers: dict[str, str],
    endpoint_gate: threading.Semaphore,
) -> dict[str, object]:
    started = time.monotonic()
    row = {
        "schema": ATTEMPT_SCHEMA,
        **job,
        "endpoint": endpoint,
        "started_at": utc_now(),
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": job["prompt"]}],
        "temperature": job["temperature"],
        "reasoning_effort": job["reasoning_effort"],
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        request = urllib.request.Request(
            endpoint + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with endpoint_gate:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = json.load(response)
                row["http_status"] = response.status
        if raw.get("error"):
            raise ValueError(str(raw["error"]))
        choice = raw["choices"][0]
        message = choice["message"]
        if message.get("tool_calls"):
            raise ValueError("unexpected tool call")
        answer = message.get("content") or ""
        finish_reason = choice.get("finish_reason")
        if finish_reason == "stop":
            correct, classification = classify_answer(answer, str(job["expected"]))
        elif finish_reason == "length":
            # Exhausting the output budget after an integer-only instruction is
            # a completed behavioral failure, not a transport/protocol error.
            correct, classification = False, "generation_truncated"
        else:
            raise ValueError("unexpected finish_reason: " + str(finish_reason))
        row.update(
            response_id=raw.get("id"),
            finish_reason=finish_reason,
            answer=answer,
            correct=correct,
            answer_class=classification,
            usage=raw.get("usage"),
        )
    except urllib.error.HTTPError as error:
        row.update(
            http_status=error.code,
            error=error.read(4096).decode("utf-8", errors="replace"),
        )
    except Exception as error:  # evidence retains protocol and transport failures
        row["error"] = f"{type(error).__name__}: {error}"
    row["seconds"] = round(time.monotonic() - started, 6)
    return row


def build_jobs(
    tasks: list[Task],
    endpoint_sets: dict[str, tuple[str, ...]],
    reasoning_repeats: dict[str, int],
    temperatures: list[float],
    seed: int,
) -> list[tuple[dict[str, object], str]]:
    rng = random.Random(seed ^ 0xA17E)
    scheduled: list[tuple[dict[str, object], str]] = []
    task_values = {task.task_id: asdict(task) for task in tasks}
    for model_label, endpoints in endpoint_sets.items():
        model_jobs: list[dict[str, object]] = []
        for reasoning_effort, repeat_count in reasoning_repeats.items():
            for temperature in temperatures:
                for repeat in range(1, repeat_count + 1):
                    for task in tasks:
                        model_jobs.append(
                            {
                                "model_label": model_label,
                                "reasoning_effort": reasoning_effort,
                                "temperature": temperature,
                                "task_id": task.task_id,
                                "family": task.family,
                                "repeat": repeat,
                                "prompt": task.prompt,
                                "expected": task.expected,
                                "operation": task.operation,
                                "operands": list(task.operands),
                                "difficulty": task.difficulty,
                            }
                        )
        rng.shuffle(model_jobs)
        scheduled.extend((job, endpoints[index % len(endpoints)]) for index, job in enumerate(model_jobs))
    rng.shuffle(scheduled)
    expected_keys = {attempt_key(job) for job, _ in scheduled}
    if len(expected_keys) != len(scheduled):
        raise AssertionError("attempt identifiers must be unique")
    if set(task_values) != {job["task_id"] for job, _ in scheduled}:
        raise AssertionError("every task must enter the run")
    return scheduled


def read_attempts(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            rows.append(row)
    keys = [attempt_key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("attempt journal contains duplicate identifiers")
    return rows


def summarize(rows: list[dict[str, object]], expected_count: int) -> dict[str, object]:
    cells: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    family_cells: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        cells[(row["model_label"], row["reasoning_effort"], row["temperature"])].append(row)
        family_cells[
            (row["model_label"], row["reasoning_effort"], row["temperature"], row["family"])
        ].append(row)

    def cell_value(key: tuple[object, ...], group: list[dict[str, object]]) -> dict[str, object]:
        completed = [row for row in group if "error" not in row]
        correct = sum(row.get("correct") is True for row in completed)
        return {
            "dimensions": list(key),
            "attempts": len(group),
            "completed": len(completed),
            "correct": correct,
            "wrong": len(completed) - correct,
            "request_errors": len(group) - len(completed),
            "accuracy_percent": round(100 * correct / len(completed), 6) if completed else None,
            "answer_classes": dict(Counter(str(row.get("answer_class")) for row in completed)),
            "wrong_answers": dict(Counter(str(row.get("answer")) for row in completed if not row.get("correct"))),
            "median_seconds": round(statistics.median(float(row["seconds"]) for row in group), 6)
            if group
            else None,
        }

    return {
        "schema": RUN_SCHEMA,
        "status": "qualified" if len(rows) == expected_count and all("error" not in row for row in rows) else "unsupported",
        "generated_at": utc_now(),
        "expected_attempts": expected_count,
        "recorded_attempts": len(rows),
        "request_errors": sum("error" in row for row in rows),
        "cells": [cell_value(key, group) for key, group in sorted(cells.items())],
        "family_cells": [cell_value(key, group) for key, group in sorted(family_cells.items())],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--endpoint-set",
        action="append",
        type=parse_endpoint_set,
        required=True,
        metavar="LABEL=URL,URL",
        help="model label and one or more OpenAI-compatible /v1 roots; repeat for each model",
    )
    parser.add_argument("--model", required=True, help="served model name sent to every endpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9917)
    parser.add_argument("--tasks-per-family", type=int, default=100)
    parser.add_argument("--none-repeats", type=int, default=3)
    parser.add_argument("--low-repeats", type=int, default=1)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 1.0])
    parser.add_argument("--per-endpoint-concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    args = parser.parse_args()

    endpoint_sets = dict(args.endpoint_set)
    if len(endpoint_sets) != len(args.endpoint_set):
        parser.error("endpoint-set labels must be unique")
    if min(
        args.tasks_per_family,
        args.none_repeats,
        args.low_repeats,
        args.per_endpoint_concurrency,
        args.max_tokens,
    ) < 1:
        parser.error("task counts, repeats, concurrency, and max-tokens must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if any(not math.isfinite(value) or value < 0 for value in args.temperatures):
        parser.error("temperatures must be finite and nonnegative")

    tasks = generate_suite(args.seed, args.tasks_per_family)
    suite = {
        "schema": SUITE_SCHEMA,
        "semantic_role": "Balanced unique-task suite for exact direct-answer integer arithmetic",
        "status": "qualified",
        "generator_seed": args.seed,
        "tasks_per_family": args.tasks_per_family,
        "task_count": len(tasks),
        "external_sentinel_excluded": "Case stability3-effort-76",
        "tasks": [asdict(task) for task in tasks],
    }
    suite["task_set_sha256"] = sha256_json(suite["tasks"])
    script_path = Path(__file__).resolve()
    script_sha256 = sha256_bytes(script_path.read_bytes())
    reasoning_repeats = {"none": args.none_repeats, "low": args.low_repeats}
    jobs = build_jobs(tasks, endpoint_sets, reasoning_repeats, args.temperatures, args.seed)
    config = {
        "schema": RUN_SCHEMA,
        "semantic_role": "Direct-answer arithmetic comparison across complete checkpoint and serving configurations",
        "status": "qualified configuration",
        "created_at": utc_now(),
        "script": str(script_path),
        "script_sha256": script_sha256,
        "suite_sha256": suite["task_set_sha256"],
        "model": args.model,
        "endpoint_sets": endpoint_sets,
        "protocol": "OpenAI Chat Completions, non-streaming",
        "sampling": {
            "temperatures": args.temperatures,
            "reasoning_repeats": reasoning_repeats,
            "request_seed": None,
            "max_tokens": args.max_tokens,
        },
        "execution": {
            "per_endpoint_concurrency": args.per_endpoint_concurrency,
            "maximum_global_concurrency": args.per_endpoint_concurrency
            * sum(len(urls) for urls in endpoint_sets.values()),
            "timeout_seconds": args.timeout,
            "automatic_retries": 0,
            "expected_attempts": len(jobs),
        },
        "scoring": "Stripped response text must equal the algorithmically computed base-10 integer",
    }
    config["contract_sha256"] = sha256_json(
        {key: value for key, value in config.items() if key not in ("created_at", "contract_sha256")}
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_path = args.output_dir / "suite.json"
    config_path = args.output_dir / "run-config.json"
    attempts_path = args.output_dir / "attempts.jsonl"
    summary_path = args.output_dir / "summary.json"
    if suite_path.exists():
        if load_json(suite_path).get("task_set_sha256") != suite["task_set_sha256"]:
            parser.error("existing suite.json does not match the generated suite")
    else:
        write_json_exclusive(suite_path, suite)
    if config_path.exists():
        if load_json(config_path).get("contract_sha256") != config["contract_sha256"]:
            parser.error("existing run-config.json does not match the requested run")
    else:
        write_json_exclusive(config_path, config)

    previous_rows = read_attempts(attempts_path)
    completed_keys = {attempt_key(row) for row in previous_rows}
    pending = [(job, endpoint) for job, endpoint in jobs if attempt_key(job) not in completed_keys]
    print(
        f"Suite: {len(tasks)} unique tasks ({suite['task_set_sha256']})\n"
        f"Run: {len(jobs)} expected attempts; {len(previous_rows)} retained; {len(pending)} pending\n"
        f"Maximum concurrency: {config['execution']['maximum_global_concurrency']} "
        f"({args.per_endpoint_concurrency} per endpoint); no automatic retries",
        flush=True,
    )

    key = os.environ.get(args.api_key_env)
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    gates = {
        endpoint: threading.Semaphore(args.per_endpoint_concurrency)
        for urls in endpoint_sets.values()
        for endpoint in urls
    }
    write_lock = threading.Lock()
    progress = {"recorded": len(previous_rows), "wrong": sum(row.get("correct") is False for row in previous_rows), "errors": sum("error" in row for row in previous_rows)}
    started = time.monotonic()
    mode = "a" if attempts_path.exists() else "x"
    with attempts_path.open(mode, encoding="utf-8", buffering=1) as journal:
        with ThreadPoolExecutor(max_workers=config["execution"]["maximum_global_concurrency"]) as pool:
            futures = [
                pool.submit(
                    request_attempt,
                    job,
                    endpoint,
                    args.model,
                    args.max_tokens,
                    args.timeout,
                    headers,
                    gates[endpoint],
                )
                for job, endpoint in pending
            ]
            interval = max(1, len(pending) // 20)
            for future in as_completed(futures):
                row = future.result()
                with write_lock:
                    journal.write(canonical_json(row) + "\n")
                    progress["recorded"] += 1
                    progress["wrong"] += row.get("correct") is False
                    progress["errors"] += "error" in row
                    newly_recorded = progress["recorded"] - len(previous_rows)
                    if newly_recorded % interval == 0 or newly_recorded == len(pending):
                        print(
                            f"{progress['recorded']}/{len(jobs)} recorded | "
                            f"wrong={progress['wrong']} | errors={progress['errors']}",
                            flush=True,
                        )

    rows = read_attempts(attempts_path)
    summary = summarize(rows, len(jobs))
    summary["elapsed_seconds_for_pending_attempts"] = round(time.monotonic() - started, 3)
    summary["suite_sha256"] = suite["task_set_sha256"]
    summary["contract_sha256"] = config["contract_sha256"]
    temporary_summary = summary_path.with_suffix(".json.tmp")
    with temporary_summary.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    os.replace(temporary_summary, summary_path)

    for cell in summary["cells"]:
        model_label, reasoning_effort, temperature = cell["dimensions"]
        print(
            f"{model_label:12} reasoning={reasoning_effort:4} temp={temperature}: "
            f"{cell['correct']}/{cell['completed']} correct, "
            f"{cell['request_errors']} request errors",
        )
    print(f"Status: {summary['status']}\nEvidence: {args.output_dir}")
    return 0 if summary["status"] == "qualified" else 2


if __name__ == "__main__":
    raise SystemExit(main())
