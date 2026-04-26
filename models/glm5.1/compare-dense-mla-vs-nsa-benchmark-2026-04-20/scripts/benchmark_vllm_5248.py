#!/usr/bin/env python3
import datetime as dt
import json
import re
import statistics
import subprocess
from pathlib import Path

RUNS = 30
PORT = 5248
MODEL = "abtest"
PROMPT_FILE = "/mnt/testLuke5.txt"
MAX_TOKENS = 40000


def run_one():
    cmd = [
        "python3",
        "/mnt/test.py",
        "--port",
        str(PORT),
        "--model",
        MODEL,
        "-f",
        PROMPT_FILE,
        "--max-tokens",
        str(MAX_TOKENS),
        "--no-overlay",
        "--quiet",
        "--json-summary",
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    summary = None
    for line in reversed((result.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                summary = json.loads(line)
                break
            except json.JSONDecodeError:
                pass
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "summary": summary,
    }


def extract_final_answer(text: str) -> str:
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def is_correct(text: str) -> bool:
    return bool(re.search(r"\bestonia\b", text or "", re.IGNORECASE))


def summarize(runs):
    completed = [r for r in runs if r.get("ok")]
    if not completed:
        return {
            "attempted_runs": len(runs),
            "completed_runs": 0,
            "correct_runs": 0,
            "correct_rate": 0.0,
        }

    def vals(key):
        return [float(r[key]) for r in completed if r.get(key) is not None]

    toks = vals("completion_tokens")
    elapsed = vals("elapsed")
    gen_elapsed = vals("gen_elapsed")
    ttft = vals("ttft")
    gen_tps = [
        r["completion_tokens"] / r["gen_elapsed"]
        for r in completed
        if r["gen_elapsed"] > 0 and r["completion_tokens"] > 0
    ]
    e2e_tps = [
        r["completion_tokens"] / r["elapsed"]
        for r in completed
        if r["elapsed"] > 0 and r["completion_tokens"] > 0
    ]
    return {
        "attempted_runs": len(runs),
        "completed_runs": len(completed),
        "correct_runs": sum(1 for r in completed if r["correct"]),
        "correct_rate": sum(1 for r in completed if r["correct"]) / len(completed),
        "mean_completion_tokens": statistics.mean(toks) if toks else 0.0,
        "median_completion_tokens": statistics.median(toks) if toks else 0.0,
        "min_completion_tokens": min(toks) if toks else 0.0,
        "max_completion_tokens": max(toks) if toks else 0.0,
        "mean_elapsed_s": statistics.mean(elapsed) if elapsed else 0.0,
        "median_elapsed_s": statistics.median(elapsed) if elapsed else 0.0,
        "min_elapsed_s": min(elapsed) if elapsed else 0.0,
        "max_elapsed_s": max(elapsed) if elapsed else 0.0,
        "mean_gen_tok_s": statistics.mean(gen_tps) if gen_tps else 0.0,
        "median_gen_tok_s": statistics.median(gen_tps) if gen_tps else 0.0,
        "min_gen_tok_s": min(gen_tps) if gen_tps else 0.0,
        "max_gen_tok_s": max(gen_tps) if gen_tps else 0.0,
        "mean_ttft_s": statistics.mean(ttft) if ttft else 0.0,
        "mean_e2e_tok_s": statistics.mean(e2e_tps) if e2e_tps else 0.0,
    }


def main():
    timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(f"/root/glm/benchmarks/vllm_5248_glm51_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "runs.jsonl"
    runs = []

    print(f"results_dir={out_dir}", flush=True)
    for i in range(1, RUNS + 1):
        raw = run_one()
        summary = raw["summary"]
        record = {
            "run": i,
            "returncode": raw["returncode"],
            "ok": False,
            "correct": False,
            "completion_tokens": None,
            "elapsed": None,
            "gen_elapsed": None,
            "ttft": None,
            "gen_tok_s": 0.0,
            "e2e_tok_s": 0.0,
            "output_text": None,
            "reasoning_text": None,
            "content_text": None,
            "finish_reason": None,
            "final_answer": "",
            "stderr": raw["stderr"],
        }

        if raw["returncode"] == 0 and summary and summary.get("last_result"):
            last = summary["last_result"]
            content_text = last.get("content_text") or ""
            output_text = last.get("output_text") or ""
            final_answer = extract_final_answer(content_text or output_text)
            completion_tokens = int(last.get("completion_tokens") or 0)
            elapsed = float(last.get("elapsed") or 0.0)
            gen_elapsed = float(last.get("gen_elapsed") or 0.0)
            record.update(
                {
                    "ok": True,
                    "correct": is_correct(final_answer),
                    "completion_tokens": completion_tokens,
                    "elapsed": elapsed,
                    "gen_elapsed": gen_elapsed,
                    "ttft": float(last.get("ttft") or 0.0),
                    "gen_tok_s": (
                        completion_tokens / gen_elapsed
                        if completion_tokens > 0 and gen_elapsed > 0
                        else 0.0
                    ),
                    "e2e_tok_s": (
                        completion_tokens / elapsed
                        if completion_tokens > 0 and elapsed > 0
                        else 0.0
                    ),
                    "output_text": output_text,
                    "reasoning_text": last.get("reasoning_text") or "",
                    "content_text": content_text,
                    "finish_reason": last.get("finish_reason"),
                    "final_answer": final_answer,
                }
            )

        runs.append(record)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        status = "OK" if record["ok"] else "ERR"
        correctness = "correct" if record["correct"] else "wrong"
        print(
            f"[vllm_5248] run {i:02d}/{RUNS} {status} {correctness} "
            f"| completion_tokens={record['completion_tokens']} "
            f"| gen_tok_s={record['gen_tok_s']:.2f} "
            f"| elapsed={record['elapsed']:.2f}s "
            f"| final_answer={record['final_answer'][:120]}",
            flush=True,
        )

    final_summary = summarize(runs)
    (out_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
