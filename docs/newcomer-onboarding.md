# Newcomer Onboarding Without Lowering The Signal

The goal of this wiki is to make the work reproducible, not to remove the need
for careful debugging. New users should be able to find the current runbook,
learn the acronyms, and run a known-good configuration before asking for help.
That keeps the Discord useful for both beginners and people doing kernel/runtime
work.

## The Short Version

Before asking a runtime question, please collect:

| Required | Why |
|---|---|
| Exact model page or runbook link | Avoids guessing which recipe you followed. |
| Docker image tag and digest if known | Many bugs are image-specific. |
| Full launch command or compose file | Most failures are config differences. |
| GPU layout and TP/DCP/MTP/DSpark settings | Performance and memory behavior depend on topology and parallelism. |
| Last 100-300 server log lines | Startup backend markers and stack traces matter. |
| Client command and observed output | Throughput numbers are meaningless without the client config. |

If you cannot provide those yet, start from the relevant model hub and reproduce
the documented smoke test first.

## Recommended Learning Path

1. Read [Glossary And Acronym Guide](../GLOSSARY.md).
2. Pick a model from the [front page](../README.md#start-here).
3. Run the current recommended launch exactly once without changing parameters.
4. Confirm `/v1/models` works.
5. Run the documented smoke test.
6. Only then change one parameter at a time.

## How To Ask A Good Question

Use this format:

```text
Model/runbook:
Docker image:
GPU layout:
TP/DCP/MTP/DSpark:
Backend:
Launch command:
Client command:
Expected:
Actual:
Logs:
What I already tried:
```

Good question:

```text
I followed models/ds4dspark-v10.md with TP=2, MODE=dspark, BACKEND=b12x-a16,
MAX_NUM_SEQS=16, MAX_BATCHED=4096. The server reaches /v1/models, but the first
decode request fails with this stack trace. Is this a known DSpark issue or did
I select the wrong graph size?
```

Bad question:

```text
It does not work. What should I do?
```

## What This Community Optimizes For

- Reproducible evidence over screenshots without context.
- Exact commands over descriptions like "same as the wiki".
- One variable changed at a time.
- Raw logs and benchmark JSONs when discussing correctness or speed.
- Sharing fixes back into the wiki once a problem is solved.

## What The Wiki Should Do For New Users

The wiki should make it obvious that the repository contains:

- Current runbooks.
- Historical regression references.
- Docker build recipes.
- Benchmark and KLD methodology.
- Hardware and PCIe topology notes.
- Troubleshooting pages.
- Acronym explanations.

That is different from making every experiment beginner-proof. Some pages are
research logs. The hub pages tell you which ones are current and which ones are
historical.

## Maintainer Checklist For New Pages

- Add the page to the relevant model hub.
- Expand important acronyms on first use.
- Include exact Docker tag, model snapshot, command, TP/DCP/MTP settings, and
  GPU layout.
- Mark the page as current, historical, reduced validation, or experimental.
- Regenerate the index with `python3 scripts/generate-wiki-index.py > INDEX.md`.
