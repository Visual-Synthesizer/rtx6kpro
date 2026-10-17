"""Aggregate the five aider polyglot runs: per-run and per-language pass rates with mean, standard deviation and a 95%
confidence interval of the mean (t, n=5), request/token/time totals and per-exercise agreement across runs.

Reads runN/results.jsonl (one line per finished exercise, written by lavd's aider-bench hook) and runN/requests.jsonl.
Writes aggregate.md (markdown tables) and aggregate.json next to the runs.
"""
import collections
import json
import math
import os
import statistics

RUNS = [1, 2, 3, 4, 5]
T95 = 2.776  # t(0.975, df=4)


def load(name):
    rows = [json.loads(l) for l in open(name) if l.strip()] if os.path.exists(name) else []
    return rows


def rate(xs):
    return 100 * sum(xs) / len(xs) if xs else float("nan")


def summary(values):
    m = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    half = T95 * sd / math.sqrt(len(values))
    return m, sd, m - half, m + half


runs = {}
for n in RUNS:
    res = load(f"run{n}/results.jsonl")
    req = load(f"run{n}/requests.jsonl")
    by_ex = {r["ex"]: r for r in res}  # last result per exercise
    langs = collections.defaultdict(list)
    for ex, r in by_ex.items():
        langs[ex.split("/")[0]].append(r)
    ok = [r for r in req if r.get("ok")]
    runs[n] = {
        "exercises": len(by_ex),
        "pass1": rate([r.get("try") == 1 for r in by_ex.values()]),
        "pass2": rate([bool(r.get("pass")) for r in by_ex.values()]),
        "pass2_count": sum(bool(r.get("pass")) for r in by_ex.values()),
        "langs": {l: {"n": len(v), "pass1": rate([r.get("try") == 1 for r in v]), "pass2": rate([bool(r.get("pass")) for r in v])} for l, v in langs.items()},
        "requests": len(req),
        "requests_failed": sum(1 for r in req if not r.get("ok")),
        "finish_length": sum(1 for r in ok if r.get("finish") == "length"),
        "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in ok),
        "completion_tokens": sum(r.get("completion_tokens") or 0 for r in ok),
        "malformed": sum(r.get("malformed_responses", 0) or 0 for r in by_ex.values()),
        "ctx_exhausted": sum(r.get("exhausted_context_windows", 0) or 0 for r in by_ex.values()),
        "test_timeouts": sum(r.get("test_timeouts", 0) or 0 for r in by_ex.values()),
        "sec_per_case": statistics.mean([r.get("sec") or 0 for r in by_ex.values()]),
        "by_ex": {ex: bool(r.get("pass")) for ex, r in by_ex.items()},
    }

p1 = [runs[n]["pass1"] for n in RUNS]
p2 = [runs[n]["pass2"] for n in RUNS]
m1, sd1, lo1, hi1 = summary(p1)
m2, sd2, lo2, hi2 = summary(p2)
langs = sorted({l for n in RUNS for l in runs[n]["langs"]})

out = []
out.append("| run | exercises | pass@1 | pass@2 | passed | requests | failed req | finish=length | malformed | ctx exhausted | prompt tok | completion tok | s/case |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for n in RUNS:
    r = runs[n]
    out.append(f"| run{n} | {r['exercises']} | {r['pass1']:.1f} % | {r['pass2']:.1f} % | {r['pass2_count']} | {r['requests']} | {r['requests_failed']} | {r['finish_length']} | {r['malformed']} | {r['ctx_exhausted']} | {r['prompt_tokens']:,} | {r['completion_tokens']:,} | {r['sec_per_case']:.0f} |")
out.append(f"| **mean ± sd** | | **{m1:.1f} ± {sd1:.1f} %** | **{m2:.1f} ± {sd2:.1f} %** | | | | | | | | | |")
out.append(f"| 95 % CI of mean | | {lo1:.1f}–{hi1:.1f} % | {lo2:.1f}–{hi2:.1f} % | | | | | | | | | |")
out.append(f"| min–max | | {min(p1):.1f}–{max(p1):.1f} % | {min(p2):.1f}–{max(p2):.1f} % | | | | | | | | | |")
out.append("")
out.append("| language | exercises | pass@1 mean ± sd | pass@2 mean ± sd | pass@2 per run |")
out.append("|---|---|---|---|---|")
for l in langs:
    v1 = [runs[n]["langs"][l]["pass1"] for n in RUNS]
    v2 = [runs[n]["langs"][l]["pass2"] for n in RUNS]
    out.append(f"| {l} | {runs[1]['langs'][l]['n']} | {statistics.mean(v1):.1f} ± {statistics.stdev(v1):.1f} % | {statistics.mean(v2):.1f} ± {statistics.stdev(v2):.1f} % | {' / '.join(f'{x:.1f}' for x in v2)} |")
out.append("")
exs = sorted(set().union(*[runs[n]["by_ex"] for n in RUNS]))
counts = collections.Counter(sum(runs[n]["by_ex"].get(ex, False) for n in RUNS) for ex in exs)
out.append("| exercises passed in k of 5 runs | k=5 | k=4 | k=3 | k=2 | k=1 | k=0 |")
out.append("|---|---|---|---|---|---|---|")
out.append("| count | " + " | ".join(str(counts.get(k, 0)) for k in (5, 4, 3, 2, 1, 0)) + " |")
never = [ex for ex in exs if counts and sum(runs[n]["by_ex"].get(ex, False) for n in RUNS) == 0]
out.append("")
out.append("Never solved in any run: " + ", ".join(never))
open("aggregate.md", "w").write("\n".join(out) + "\n")
json.dump({"runs": {n: {k: v for k, v in runs[n].items() if k != "by_ex"} for n in RUNS},
           "pass1": {"mean": m1, "sd": sd1, "ci95": [lo1, hi1]}, "pass2": {"mean": m2, "sd": sd2, "ci95": [lo2, hi2]},
           "agreement": {str(k): counts.get(k, 0) for k in range(6)}, "never_solved": never}, open("aggregate.json", "w"), indent=1)
print("\n".join(out))
