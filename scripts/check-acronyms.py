#!/usr/bin/env python3
"""Report common wiki acronyms that are used without nearby expansion.

This is intentionally a linter, not an auto-rewriter. It avoids changing code
blocks, commands, logs, Docker tags, JSON, and environment variables where
automatic expansion would create noise or break copy/paste snippets.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "GLOSSARY.md"

IGNORE_DIRS = {
    ".git",
    "images",
    "logs",
    "data",
}

IGNORE_FILES = {
    "GLOSSARY.md",
    "INDEX.md",
}


def load_terms() -> dict[str, str]:
    terms: dict[str, str] = {}
    table_re = re.compile(r"^\|\s*`?([A-Za-z0-9+./-]+)`?\s*\|\s*([^|]+?)\s*\|")
    for line in GLOSSARY.read_text(encoding="utf-8").splitlines():
        m = table_re.match(line)
        if not m:
            continue
        acronym, meaning = m.groups()
        if acronym.lower() == "acronym" or set(acronym) <= {"-"}:
            continue
        if len(acronym) < 2:
            continue
        terms[acronym] = meaning.strip()
    return terms


def markdown_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT)
        if rel.name in IGNORE_FILES:
            continue
        if any(part in IGNORE_DIRS for part in rel.parts):
            continue
        files.append(path)
    return sorted(files)


def strip_code_fences(text: str) -> str:
    out = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def has_expansion_before(text: str, acronym: str, meaning: str, pos: int) -> bool:
    prefix = text[:pos]
    if f"{meaning} ({acronym})" in prefix:
        return True
    if f"{meaning} `{acronym}`" in prefix:
        return True
    return False


def find_first_use(text: str, acronym: str) -> re.Match[str] | None:
    if acronym.startswith("cc") or acronym.startswith("ctx"):
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(acronym)}(?![A-Za-z0-9_])")
    else:
        pattern = re.compile(rf"(?<![A-Za-z0-9_`/-]){re.escape(acronym)}(?![A-Za-z0-9_`/-])")
    return pattern.search(text)


def line_number(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max",
        type=int,
        default=200,
        help="maximum number of warnings to print",
    )
    args = parser.parse_args()

    terms = load_terms()
    warnings = []
    for path in markdown_files():
        raw = path.read_text(encoding="utf-8", errors="replace")
        text = strip_code_fences(raw)
        for acronym, meaning in terms.items():
            match = find_first_use(text, acronym)
            if not match:
                continue
            if has_expansion_before(text, acronym, meaning, match.start()):
                continue
            warnings.append((path.relative_to(ROOT).as_posix(),
                             line_number(text, match.start()),
                             acronym,
                             meaning))

    for rel, line, acronym, meaning in warnings[: args.max]:
        print(f"{rel}:{line}: {acronym} -> {meaning}")
    if len(warnings) > args.max:
        print(f"... {len(warnings) - args.max} more")

    return 1 if warnings else 0


if __name__ == "__main__":
    raise SystemExit(main())
