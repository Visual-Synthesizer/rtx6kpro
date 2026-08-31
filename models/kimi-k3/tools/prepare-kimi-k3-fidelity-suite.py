#!/usr/bin/env python3
"""Build the immutable 1,024-context Kimi K3 distribution-fidelity suite."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import load_dataset
from transformers import AutoTokenizer

CONTEXT_LENGTH = 2048
SELECTION_SALT = "kimi-k3-fidelity-selection-v1"
PARTITION_SALT = "kimi-k3-fidelity-partition-v1"
SENTINEL_SALT = "kimi-k3-fidelity-sentinel-v1"
OFFSET_SALT = "kimi-k3-fidelity-offset-v1"
MINHASH_SEEDS = tuple(
    int.from_bytes(hashlib.sha256(f"minhash-v1:{index}".encode()).digest()[:8], "big")
    for index in range(16)
)


@dataclass(frozen=True)
class DatasetSource:
    """Pinned dataset and extraction policy for one source allocation."""

    key: str
    dataset: str
    revision: str
    split: str = "train"
    config: str | None = None
    license: str = "per-record metadata"
    scan_limit: int = 50_000


@dataclass
class Candidate:
    """One independently sourced 2,048-token context candidate."""

    allocation_stratum: str
    semantic_type: str
    language: str
    representation_type: str
    source_key: str
    source_id: str
    source_cluster_id: str
    source_content_sha256: str
    source_license: str
    extraction: str
    token_ids: list[int]
    source_metadata: dict[str, Any]
    chat_template_applied: bool = False
    source_char_start: int | None = None
    selection_key: str = field(init=False)
    token_hash: str = field(init=False)
    shingles: frozenset[int] = field(init=False, repr=False)
    minhash: tuple[int, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.token_ids) != CONTEXT_LENGTH:
            raise ValueError(
                f"Context {self.source_key}:{self.source_id} has "
                f"{len(self.token_ids)} tokens"
            )
        self.selection_key = domain_hash(
            SELECTION_SALT,
            self.allocation_stratum,
            self.source_key,
            self.source_id,
            self.source_content_sha256,
        )
        compact = compact_json(self.token_ids)
        self.token_hash = sha256_bytes(compact.encode("utf-8"))
        self.shingles = token_shingles(self.token_ids)
        self.minhash = minhash_signature(self.shingles)


SOURCES = {
    "wikipedia_en": DatasetSource(
        key="wikipedia_en",
        dataset="wikimedia/wikipedia",
        config="20231101.en",
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        license="CC-BY-SA-3.0 and GFDL",
        scan_limit=20_000,
    ),
    "scientific_papers": DatasetSource(
        key="scientific_papers",
        dataset="common-pile/peS2o_filtered",
        revision="297747513bfb0ff1fbf61ddad3b03319d0f04597",
        scan_limit=20_000,
    ),
    "open_news": DatasetSource(
        key="open_news",
        dataset="common-pile/news_filtered",
        revision="59aaa8f104e189e6fb8033f0ed319c5c343a03b1",
        scan_limit=20_000,
    ),
    "public_domain_review": DatasetSource(
        key="public_domain_review",
        dataset="common-pile/public_domain_review_filtered",
        revision="efc7f21259d069a27d6ca3655f74fda969f82b01",
        scan_limit=20_000,
    ),
    "regulations": DatasetSource(
        key="regulations",
        dataset="common-pile/regulations_filtered",
        revision="3327364490dfc7929009226ad667eceb2441d93a",
        scan_limit=20_000,
    ),
    "public_domain_books": DatasetSource(
        key="public_domain_books",
        dataset="common-pile/pre_1929_books_filtered",
        revision="23f9d96dbb1db3324bbc9fbfe1f8299cc799c4d1",
        scan_limit=10_000,
    ),
    "wildchat": DatasetSource(
        key="wildchat",
        dataset="allenai/WildChat-1M",
        revision="7d6490e462285cf85d91eabea0f9a954fbddcd1f",
        license="ODC-BY-1.0",
        scan_limit=100_000,
    ),
    "stackv2": DatasetSource(
        key="stackv2",
        dataset="common-pile/stackv2_edu_filtered",
        revision="c354dbe88469a1153e97c6a63ac50591849654de",
        scan_limit=250_000,
    ),
    "github_code": DatasetSource(
        key="github_code",
        dataset="codeparrot/github-code",
        revision="b5661e6b17396364b2bcf8e68977b0d28e1ebd19",
        license="per-record open-source license",
        scan_limit=200_000,
    ),
    "libretexts": DatasetSource(
        key="libretexts",
        dataset="common-pile/libretexts_filtered",
        revision="70388bca52b4a93515e14b1d56618fd7944988fd",
        scan_limit=50_000,
    ),
    "wikipedia_zh": DatasetSource(
        key="wikipedia_zh",
        dataset="wikimedia/wikipedia",
        config="20231101.zh",
        revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        license="CC-BY-SA-3.0 and GFDL",
        scan_limit=20_000,
    ),
    "wikisource_zh": DatasetSource(
        key="wikisource_zh",
        dataset="erhwenkuo/zhwikisource-zhtw",
        revision="1fdebaa66ef58a9517c87e73f813f5923490b5c2",
        license="CC-BY-SA-3.0",
        scan_limit=20_000,
    ),
    **{
        f"wikipedia_{language}": DatasetSource(
            key=f"wikipedia_{language}",
            dataset="wikimedia/wikipedia",
            config=f"20231101.{language}",
            revision="b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
            license="CC-BY-SA-3.0 and GFDL",
            scan_limit=20_000,
        )
        for language in ("cs", "de", "es", "fr", "ja", "ru")
    },
}

STRATUM_COUNTS = {
    "encyclopedic_factual": (128, 32, 8),
    "scientific_technical": (128, 32, 8),
    "news_history_economics_legal_essays": (96, 24, 6),
    "literary_narrative_creative": (96, 24, 6),
    "dialogue_instruction_assistance": (128, 32, 8),
    "code_tests_documentation_issues": (128, 32, 8),
    "worked_math_science_formal_reasoning": (128, 32, 8),
    "chinese": (96, 24, 6),
    "other_multilingual": (48, 12, 3),
    "structured_data_tools_apis_tables": (48, 12, 3),
}

CODE_EXTENSIONS = {
    "c",
    "cc",
    "cpp",
    "cs",
    "go",
    "h",
    "hpp",
    "java",
    "js",
    "jsx",
    "kt",
    "lua",
    "md",
    "php",
    "py",
    "rb",
    "rs",
    "scala",
    "sh",
    "sql",
    "swift",
    "ts",
    "tsx",
}
API_TERMS = (
    " api ",
    "endpoint",
    "http request",
    "request body",
    "response body",
    "curl ",
    "openapi",
    "graphql",
    "authorization header",
    "status code",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def domain_hash(domain: str, *values: str) -> str:
    payload = "\0".join((domain, *values)).encode("utf-8")
    return sha256_bytes(payload)


def metadata_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        if isinstance(parsed, dict):
            return parsed
    return {}


def stable_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): stable_metadata(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [stable_metadata(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def quiet_encode(tokenizer: Any, text: str) -> list[int]:
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        return list(tokenizer.encode(text, add_special_tokens=False))


def splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return value ^ (value >> 31)


def token_shingles(token_ids: list[int]) -> frozenset[int]:
    values: set[int] = set()
    for index in range(len(token_ids) - 4):
        packed = struct.pack("<5I", *token_ids[index : index + 5])
        values.add(
            int.from_bytes(hashlib.blake2b(packed, digest_size=8).digest(), "big")
        )
    return frozenset(values)


def minhash_signature(shingles: frozenset[int]) -> tuple[int, ...]:
    return tuple(
        min(splitmix64(value ^ seed) for value in shingles) for seed in MINHASH_SEEDS
    )


def approximate_duplicate(left: Candidate, right: Candidate) -> bool:
    agreement = sum(a == b for a, b in zip(left.minhash, right.minhash, strict=True))
    if agreement < 8:
        return False
    intersection = len(left.shingles & right.shingles)
    union = len(left.shingles | right.shingles)
    return bool(union and intersection / union >= 0.85)


def boundary_positions(text: str, *, paragraph_only: bool) -> list[int]:
    pattern = r"\n[ \t]*\n" if paragraph_only else r"\n"
    positions = [0]
    positions.extend(match.end() for match in re.finditer(pattern, text))
    return positions


def context_from_text(
    tokenizer: Any,
    text: str,
    *,
    content_hash: str,
    paragraph_only: bool = True,
    validator: Any | None = None,
) -> tuple[list[int], int] | None:
    if len(text) < 4_000:
        return None
    positions = boundary_positions(text, paragraph_only=paragraph_only)
    positions = [position for position in positions if len(text) - position >= 4_000]
    if not positions:
        return None
    start = int(domain_hash(OFFSET_SALT, content_hash), 16) % len(positions)
    for step in range(min(len(positions), 64)):
        char_start = positions[(start + step) % len(positions)]
        sample = text[char_start : char_start + 96_000]
        token_ids = quiet_encode(tokenizer, sample)
        if len(token_ids) < CONTEXT_LENGTH:
            continue
        context = token_ids[:CONTEXT_LENGTH]
        if validator is not None and not validator(tokenizer, context):
            continue
        return context, char_start
    return None


def api_document_window_is_valid(tokenizer: Any, token_ids: list[int]) -> bool:
    with contextlib.redirect_stdout(io.StringIO()):
        text = tokenizer.decode(token_ids)
    lowered = f" {text.lower()} "
    if sum(term in lowered for term in API_TERMS) < 2:
        return False
    return text.count("```") % 2 == 0


def source_license(source: DatasetSource, metadata: dict[str, Any]) -> str:
    return str(metadata.get("license") or metadata.get("oa_license") or source.license)


def web_source_cluster(row: dict[str, Any], row_index: int) -> str:
    metadata = metadata_dict(row.get("metadata"))
    hostname = urlparse(str(metadata.get("url") or "")).netloc
    return hostname or str(row.get("id") or row_index)


def load_stream(source: DatasetSource) -> Any:
    kwargs: dict[str, Any] = {
        "path": source.dataset,
        "split": source.split,
        "streaming": True,
        "revision": source.revision,
    }
    if source.config is not None:
        kwargs["name"] = source.config
    if source.key == "github_code":
        kwargs["trust_remote_code"] = True
    return load_dataset(**kwargs)


def dataset_slice_fingerprint(source: DatasetSource) -> str:
    return domain_hash(
        "hugging-face-dataset-slice-v1",
        source.dataset,
        source.revision,
        source.config or "",
        source.split,
    )


def raw_candidate(
    tokenizer: Any,
    source: DatasetSource,
    row: dict[str, Any],
    *,
    allocation_stratum: str,
    semantic_type: str,
    language: str,
    representation_type: str,
    source_id: str,
    cluster_id: str,
    paragraph_only: bool = True,
    validator: Any | None = None,
) -> Candidate | None:
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        return None
    content_hash = sha256_bytes(text.encode("utf-8"))
    context = context_from_text(
        tokenizer,
        text,
        content_hash=content_hash,
        paragraph_only=paragraph_only,
        validator=validator,
    )
    if context is None:
        return None
    token_ids, char_start = context
    metadata = metadata_dict(row.get("metadata"))
    return Candidate(
        allocation_stratum=allocation_stratum,
        semantic_type=semantic_type,
        language=language,
        representation_type=representation_type,
        source_key=source.key,
        source_id=source_id,
        source_cluster_id=cluster_id,
        source_content_sha256=content_hash,
        source_license=source_license(source, metadata),
        extraction=(
            "contiguous source span beginning at a deterministic source boundary; "
            "the first 2,048 Kimi K3 token IDs are retained"
        ),
        token_ids=token_ids,
        source_metadata=stable_metadata(metadata),
        source_char_start=char_start,
    )


def collect_raw(
    tokenizer: Any,
    source: DatasetSource,
    *,
    allocation_stratum: str,
    semantic_type: str,
    language: str,
    representation_type: str,
    pool_size: int,
    row_identity: Any,
    row_cluster: Any | None = None,
    row_filter: Any | None = None,
    paragraph_only: bool = True,
) -> tuple[list[Candidate], str]:
    dataset = load_stream(source)
    candidates: list[Candidate] = []
    for row_index, row in enumerate(dataset):
        if row_index >= source.scan_limit or len(candidates) >= pool_size:
            break
        if row_filter is not None and not row_filter(row):
            continue
        source_id = str(row_identity(row, row_index))
        cluster_id = (
            str(row_cluster(row, row_index)) if row_cluster is not None else source_id
        )
        candidate = raw_candidate(
            tokenizer,
            source,
            row,
            allocation_stratum=allocation_stratum,
            semantic_type=semantic_type,
            language=language,
            representation_type=representation_type,
            source_id=source_id,
            cluster_id=cluster_id,
            paragraph_only=paragraph_only,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates, dataset_slice_fingerprint(source)


def collect_wildchat(tokenizer: Any, pool_size: int) -> tuple[list[Candidate], str]:
    source = SOURCES["wildchat"]
    dataset = load_stream(source)
    candidates: list[Candidate] = []
    for row_index, row in enumerate(dataset):
        if row_index >= source.scan_limit or len(candidates) >= pool_size:
            break
        if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
            continue
        raw_messages = row.get("conversation")
        if not isinstance(raw_messages, list):
            continue
        messages = [
            {"role": item.get("role"), "content": item.get("content")}
            for item in raw_messages
            if item.get("role") in {"user", "assistant"}
            and isinstance(item.get("content"), str)
            and item["content"].strip()
        ]
        if len(messages) < 2 or messages[0]["role"] != "user":
            continue
        canonical = compact_json(messages)
        content_hash = sha256_bytes(canonical.encode("utf-8"))
        starts = [
            index for index, item in enumerate(messages) if item["role"] == "user"
        ]
        offset = int(domain_hash(OFFSET_SALT, content_hash), 16) % len(starts)
        token_ids: list[int] | None = None
        selected_start = 0
        for step in range(len(starts)):
            selected_start = starts[(offset + step) % len(starts)]
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                rendered = tokenizer.apply_chat_template(
                    messages[selected_start:],
                    tokenize=True,
                    add_generation_prompt=False,
                    thinking=False,
                )
            if len(rendered) >= CONTEXT_LENGTH:
                token_ids = list(rendered[:CONTEXT_LENGTH])
                break
        if token_ids is None:
            continue
        conversation_hash = str(row.get("conversation_hash") or row_index)
        candidates.append(
            Candidate(
                allocation_stratum="dialogue_instruction_assistance",
                semantic_type="natural_multi_turn_assistance",
                language="en",
                representation_type="kimi_k3_xtml_chat",
                source_key=source.key,
                source_id=conversation_hash,
                source_cluster_id=conversation_hash,
                source_content_sha256=content_hash,
                source_license=source.license,
                extraction=(
                    "conversation suffix beginning at a user-role boundary, rendered "
                    "by Kimi K3 Python XTML encoding with thinking disabled and no "
                    "generation prompt; the first 2,048 token IDs are retained"
                ),
                token_ids=token_ids,
                source_metadata={
                    "conversation_hash": conversation_hash,
                    "message_count": len(messages),
                    "model": str(row.get("model")),
                    "selected_message_start": selected_start,
                },
                chat_template_applied=True,
            )
        )
    return candidates, dataset_slice_fingerprint(source)


def stack_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return metadata_dict(row.get("metadata"))


def collect_stackv2(
    tokenizer: Any,
    *,
    code_pool_size: int,
    api_pool_size: int,
) -> tuple[list[Candidate], list[Candidate], str]:
    source = SOURCES["stackv2"]
    dataset = load_stream(source)
    code: list[Candidate] = []
    api: list[Candidate] = []
    code_language_counts: Counter[str] = Counter()
    code_language_pool_cap = max(64, code_pool_size // 8)
    for row_index, row in enumerate(dataset):
        if row_index >= source.scan_limit:
            break
        if len(code) >= code_pool_size and len(api) >= api_pool_size:
            break
        metadata = stack_metadata(row)
        repo = str(metadata.get("repo_name") or "")
        extension = str(metadata.get("extension") or "").lower().lstrip(".")
        text = row.get("text")
        if (
            not repo
            or not isinstance(text, str)
            or len(text) < 4_000
            or metadata.get("is_generated")
            or metadata.get("is_vendor")
        ):
            continue
        source_id = str(row.get("id") or metadata.get("blob_id") or row_index)
        language = str(metadata.get("language") or extension or "und")
        lowered = f" {text[:200_000].lower()} "
        is_api = (
            extension in {"md", "markdown", "rst", "adoc", "txt"}
            and sum(term in lowered for term in API_TERMS) >= 2
        )
        if is_api and len(api) < api_pool_size:
            candidate = raw_candidate(
                tokenizer,
                source,
                row,
                allocation_stratum="structured_data_tools_apis_tables",
                semantic_type="api_documentation_with_structured_examples",
                language="en",
                representation_type="source_document",
                source_id=source_id,
                cluster_id=repo,
                paragraph_only=True,
                validator=api_document_window_is_valid,
            )
            if candidate is not None:
                candidate.source_metadata["repo_name"] = repo
                candidate.source_metadata["path"] = stable_metadata(
                    metadata.get("path")
                )
                api.append(candidate)
        if (
            extension in CODE_EXTENSIONS
            and len(code) < code_pool_size
            and code_language_counts[language] < code_language_pool_cap
        ):
            candidate = raw_candidate(
                tokenizer,
                source,
                row,
                allocation_stratum="code_tests_documentation_issues",
                semantic_type="source_code_or_technical_documentation",
                language=language,
                representation_type="source_file",
                source_id=source_id,
                cluster_id=repo,
                paragraph_only=False,
            )
            if candidate is not None:
                candidate.source_metadata["repo_name"] = repo
                candidate.source_metadata["path"] = stable_metadata(
                    metadata.get("path")
                )
                code.append(candidate)
                code_language_counts[language] += 1
    return code, api, dataset_slice_fingerprint(source)


def collect_github_code(
    tokenizer: Any,
    *,
    pool_size: int,
) -> tuple[list[Candidate], str]:
    source = SOURCES["github_code"]
    dataset = load_stream(source)
    candidates: list[Candidate] = []
    language_counts: Counter[str] = Counter()
    language_pool_cap = max(64, pool_size // 8)
    excluded_path_parts = {
        "build",
        "deps",
        "dist",
        "generated",
        "node_modules",
        "third_party",
        "vendor",
        "vendors",
    }
    for row_index, row in enumerate(dataset):
        if row_index >= source.scan_limit or len(candidates) >= pool_size:
            break
        code = row.get("code")
        repo = str(row.get("repo_name") or "")
        path = str(row.get("path") or "")
        language = str(row.get("language") or "und")
        path_parts = set(Path(path.lower()).parts)
        if (
            not repo
            or not isinstance(code, str)
            or len(code) < 4_000
            or path_parts & excluded_path_parts
            or path.lower().endswith((".min.js", ".min.css", ".lock"))
            or language_counts[language] >= language_pool_cap
        ):
            continue
        content_hash = sha256_bytes(code.encode("utf-8"))
        source_id = f"{repo}:{path}:{content_hash}"
        synthetic_row = {
            "text": code,
            "metadata": {
                "language": language,
                "license": str(row.get("license") or source.license),
                "path": path,
                "repo_name": repo,
                "size": row.get("size"),
            },
        }
        candidate = raw_candidate(
            tokenizer,
            source,
            synthetic_row,
            allocation_stratum="code_tests_documentation_issues",
            semantic_type="source_code_or_test",
            language=language,
            representation_type="source_file",
            source_id=source_id,
            cluster_id=repo,
            paragraph_only=False,
        )
        if candidate is not None:
            candidates.append(candidate)
            language_counts[language] += 1
    return candidates, dataset_slice_fingerprint(source)


def choose_candidates(
    pool: Iterable[Candidate],
    *,
    count: int,
    selected: list[Candidate],
    unique_clusters: set[str] | None = None,
    language_cap: int | None = None,
) -> list[Candidate]:
    chosen: list[Candidate] = []
    seen_content = {item.source_content_sha256 for item in selected}
    seen_tokens = {item.token_hash for item in selected}
    language_counts: Counter[str] = Counter()
    used_clusters = unique_clusters if unique_clusters is not None else set()
    for candidate in sorted(pool, key=lambda item: item.selection_key):
        if candidate.source_content_sha256 in seen_content:
            continue
        if candidate.token_hash in seen_tokens:
            continue
        if unique_clusters is not None and candidate.source_cluster_id in used_clusters:
            continue
        if (
            language_cap is not None
            and language_counts[candidate.language] >= language_cap
        ):
            continue
        if any(approximate_duplicate(candidate, item) for item in (*selected, *chosen)):
            continue
        chosen.append(candidate)
        seen_content.add(candidate.source_content_sha256)
        seen_tokens.add(candidate.token_hash)
        language_counts[candidate.language] += 1
        if unique_clusters is not None:
            used_clusters.add(candidate.source_cluster_id)
        if len(chosen) == count:
            return chosen
    available_languages = dict(Counter(item.language for item in pool))
    available_clusters = len({item.source_cluster_id for item in pool})
    raise RuntimeError(
        f"Selected {len(chosen)} of {count} required candidates; "
        f"pool languages={available_languages}, clusters={available_clusters}"
    )


def file_hashes(directory: Path, names: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in names:
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = sha256_file(path)
    return result


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def collect_sources(tokenizer: Any) -> tuple[list[Candidate], dict[str, str]]:
    pools: dict[str, list[Candidate]] = defaultdict(list)
    fingerprints: dict[str, str] = {}

    pools["encyclopedic_factual"], fingerprints["wikipedia_en"] = collect_raw(
        tokenizer,
        SOURCES["wikipedia_en"],
        allocation_stratum="encyclopedic_factual",
        semantic_type="encyclopedic_article",
        language="en",
        representation_type="raw_continuation",
        pool_size=768,
        row_identity=lambda row, _: row["id"],
    )
    pools["scientific_technical"], fingerprints["scientific_papers"] = collect_raw(
        tokenizer,
        SOURCES["scientific_papers"],
        allocation_stratum="scientific_technical",
        semantic_type="open_access_scientific_paper",
        language="en",
        representation_type="raw_continuation",
        pool_size=768,
        row_identity=lambda row, _: row["id"],
    )

    composite = (
        ("open_news", "open_license_news_article", 192),
        ("public_domain_review", "history_and_cultural_essay", 192),
        ("regulations", "legal_and_regulatory_text", 192),
    )
    for key, semantic_type, pool_size in composite:
        source = SOURCES[key]
        values, fingerprint = collect_raw(
            tokenizer,
            source,
            allocation_stratum="news_history_economics_legal_essays",
            semantic_type=semantic_type,
            language="en",
            representation_type="raw_continuation",
            pool_size=pool_size,
            row_identity=lambda row, index: row.get("id", index),
            row_cluster=web_source_cluster,
        )
        pools[f"news_history_economics_legal_essays:{key}"] = values
        fingerprints[key] = fingerprint

    (
        pools["literary_narrative_creative"],
        fingerprints["public_domain_books"],
    ) = collect_raw(
        tokenizer,
        SOURCES["public_domain_books"],
        allocation_stratum="literary_narrative_creative",
        semantic_type="public_domain_book",
        language="en",
        representation_type="raw_continuation",
        pool_size=576,
        row_identity=lambda row, _: row["id"],
    )
    pools["dialogue_instruction_assistance"], fingerprints["wildchat"] = (
        collect_wildchat(tokenizer, 768)
    )
    _, api, fingerprints["stackv2"] = collect_stackv2(
        tokenizer,
        code_pool_size=0,
        api_pool_size=384,
    )
    pools["structured_data_tools_apis_tables"] = api
    (
        pools["code_tests_documentation_issues"],
        fingerprints["github_code"],
    ) = collect_github_code(tokenizer, pool_size=1_024)
    (
        pools["worked_math_science_formal_reasoning"],
        fingerprints["libretexts"],
    ) = collect_raw(
        tokenizer,
        SOURCES["libretexts"],
        allocation_stratum="worked_math_science_formal_reasoning",
        semantic_type="worked_mathematics_or_science_textbook",
        language="en",
        representation_type="raw_continuation",
        pool_size=768,
        row_identity=lambda row, _: row["id"],
        row_cluster=lambda row, _: metadata_dict(row.get("metadata")).get(
            "book_url", row["id"]
        ),
        row_filter=lambda row: any(
            marker in str(row.get("text", ""))
            for marker in ("Exercise", "Example", "Solution", "\\(", "\\[")
        ),
    )
    pools["chinese:wikipedia"], fingerprints["wikipedia_zh"] = collect_raw(
        tokenizer,
        SOURCES["wikipedia_zh"],
        allocation_stratum="chinese",
        semantic_type="encyclopedic_article",
        language="zh",
        representation_type="raw_continuation",
        pool_size=288,
        row_identity=lambda row, _: row["id"],
    )
    pools["chinese:wikisource"], fingerprints["wikisource_zh"] = collect_raw(
        tokenizer,
        SOURCES["wikisource_zh"],
        allocation_stratum="chinese",
        semantic_type="literary_historical_or_legal_source",
        language="zh",
        representation_type="raw_continuation",
        pool_size=288,
        row_identity=lambda row, _: row["id"],
    )
    for language in ("cs", "de", "es", "fr", "ja", "ru"):
        key = f"wikipedia_{language}"
        values, fingerprint = collect_raw(
            tokenizer,
            SOURCES[key],
            allocation_stratum="other_multilingual",
            semantic_type="encyclopedic_article",
            language=language,
            representation_type="raw_continuation",
            pool_size=64,
            row_identity=lambda row, _: row["id"],
        )
        pools[f"other_multilingual:{language}"] = values
        fingerprints[key] = fingerprint

    print(
        json.dumps(
            {name: len(values) for name, values in sorted(pools.items())},
            sort_keys=True,
        ),
        flush=True,
    )
    selected: list[Candidate] = []
    selected.extend(
        choose_candidates(pools["encyclopedic_factual"], count=128, selected=selected)
    )
    selected.extend(
        choose_candidates(pools["scientific_technical"], count=128, selected=selected)
    )
    for key, _, _ in composite:
        selected.extend(
            choose_candidates(
                pools[f"news_history_economics_legal_essays:{key}"],
                count=32,
                selected=selected,
            )
        )
    selected.extend(
        choose_candidates(
            pools["literary_narrative_creative"], count=96, selected=selected
        )
    )
    selected.extend(
        choose_candidates(
            pools["dialogue_instruction_assistance"], count=128, selected=selected
        )
    )

    software_repositories: set[str] = set()
    selected.extend(
        choose_candidates(
            pools["structured_data_tools_apis_tables"],
            count=48,
            selected=selected,
            unique_clusters=software_repositories,
        )
    )
    selected.extend(
        choose_candidates(
            pools["code_tests_documentation_issues"],
            count=128,
            selected=selected,
            unique_clusters=software_repositories,
            language_cap=32,
        )
    )
    selected.extend(
        choose_candidates(
            pools["worked_math_science_formal_reasoning"],
            count=128,
            selected=selected,
        )
    )
    selected.extend(
        choose_candidates(pools["chinese:wikipedia"], count=48, selected=selected)
    )
    selected.extend(
        choose_candidates(pools["chinese:wikisource"], count=48, selected=selected)
    )
    for language in ("cs", "de", "es", "fr", "ja", "ru"):
        selected.extend(
            choose_candidates(
                pools[f"other_multilingual:{language}"],
                count=8,
                selected=selected,
            )
        )
    return selected, fingerprints


def assign_partitions(candidates: list[Candidate]) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    by_stratum: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_stratum[candidate.allocation_stratum].append(candidate)
    for stratum, (total, qualification_count, sentinel_count) in STRATUM_COUNTS.items():
        values = by_stratum[stratum]
        if len(values) != total:
            raise RuntimeError(
                f"{stratum} contains {len(values)} contexts, expected {total}"
            )
        qualification = {
            item.token_hash
            for item in sorted(
                values,
                key=lambda item: domain_hash(
                    PARTITION_SALT, stratum, item.source_content_sha256, item.token_hash
                ),
            )[:qualification_count]
        }
        analysis = [item for item in values if item.token_hash not in qualification]
        sentinels = {
            item.token_hash
            for item in sorted(
                analysis,
                key=lambda item: domain_hash(
                    SENTINEL_SALT, stratum, item.source_content_sha256, item.token_hash
                ),
            )[:sentinel_count]
        }
        for item in values:
            assignments[item.token_hash] = {
                "partition": (
                    "qualification" if item.token_hash in qualification else "analysis"
                ),
                "sentinel": item.token_hash in sentinels,
            }
    return assignments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Output directory already exists: {args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    candidates, fingerprints = collect_sources(tokenizer)
    assignments = assign_partitions(candidates)
    ordered = sorted(
        candidates,
        key=lambda item: (
            list(STRATUM_COUNTS).index(item.allocation_stratum),
            item.selection_key,
        ),
    )

    args.output_dir.mkdir(parents=True)
    token_dir = args.output_dir / "tokens"
    token_dir.mkdir()
    context_records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for index, candidate in enumerate(ordered):
        token_name = f"context-{index:04d}.json"
        compact = compact_json(candidate.token_ids)
        (token_dir / token_name).write_text(compact + "\n", encoding="utf-8")
        assignment = assignments[candidate.token_hash]
        context_records.append(
            {
                "index": index,
                "context_index": index,
                "allocation_stratum": candidate.allocation_stratum,
                "semantic_class": candidate.semantic_type,
                "language": candidate.language,
                "representation_type": candidate.representation_type,
                "chat_template_applied": candidate.chat_template_applied,
                "dataset": SOURCES[candidate.source_key].dataset,
                "dataset_config": SOURCES[candidate.source_key].config,
                "dataset_revision": SOURCES[candidate.source_key].revision,
                "dataset_split": SOURCES[candidate.source_key].split,
                "source_id": candidate.source_id,
                "source_cluster_id": candidate.source_cluster_id,
                "source_content_sha256": candidate.source_content_sha256,
                "source_char_start": candidate.source_char_start,
                "num_tokens": CONTEXT_LENGTH,
                "scored_row_start": 0,
                "scored_row_end_exclusive": CONTEXT_LENGTH - 1,
                "token_file": f"tokens/{token_name}",
                "token_ids_json_sha256": candidate.token_hash,
                **assignment,
            }
        )
        source_records.append(
            {
                "context_index": index,
                "source_key": candidate.source_key,
                "source_id": candidate.source_id,
                "source_cluster_id": candidate.source_cluster_id,
                "source_content_sha256": candidate.source_content_sha256,
                "source_license": candidate.source_license,
                "extraction": candidate.extraction,
                "metadata": candidate.source_metadata,
            }
        )

    tokenizer_files = file_hashes(
        args.tokenizer,
        (
            "encoding_k3.py",
            "tiktoken.model",
            "tokenization_kimi.py",
            "tokenizer_config.json",
        ),
    )
    suite_hash = sha256_bytes(
        "\n".join(item["token_ids_json_sha256"] for item in context_records).encode(
            "ascii"
        )
    )
    source_registry = {
        "format_version": 1,
        "kind": "Kimi K3 distribution-fidelity source registry",
        "dataset_fingerprint_method": (
            "SHA-256 over the domain-separated dataset repository, immutable "
            "revision, configuration, and split identity"
        ),
        "sources": [
            {
                "key": source.key,
                "dataset": source.dataset,
                "config": source.config,
                "split": source.split,
                "revision": source.revision,
                "dataset_fingerprint": fingerprints[source.key],
                "license": source.license,
                "scan_limit": source.scan_limit,
            }
            for source in SOURCES.values()
        ],
    }
    partitions = {
        "format_version": 1,
        "partition_salt": PARTITION_SALT,
        "sentinel_salt": SENTINEL_SALT,
        "contexts": [
            {
                "context_index": item["context_index"],
                "partition": item["partition"],
                "sentinel": item["sentinel"],
            }
            for item in context_records
        ],
    }
    manifest = {
        "format_version": 1,
        "kind": "Kimi K3 teacher-forced distribution-fidelity token suite",
        "status": "implemented",
        "context_length": CONTEXT_LENGTH,
        "context_count": len(context_records),
        "scored_positions_per_context": CONTEXT_LENGTH - 1,
        "total_scored_positions": len(context_records) * (CONTEXT_LENGTH - 1),
        "suite_token_hash_sha256": suite_hash,
        "selection_salt": SELECTION_SALT,
        "offset_salt": OFFSET_SALT,
        "exact_content_deduplication": True,
        "approximate_deduplication": {
            "method": "five-token shingle Jaccard after 16-component MinHash screen",
            "rejection_threshold": 0.85,
        },
        "tokenizer": {
            "checkpoint_revision": args.tokenizer.name,
            "class": tokenizer.__class__.__name__,
            "files": tokenizer_files,
            "vocabulary_size": len(tokenizer),
        },
        "chat_rendering": {
            "implementation": "Kimi K3 Python XTML encoding",
            "add_generation_prompt": False,
            "thinking": False,
            "jinja_template": None,
        },
        "allocation_counts": dict(
            Counter(item["allocation_stratum"] for item in context_records)
        ),
        "language_counts": dict(Counter(item["language"] for item in context_records)),
        "partition_counts": dict(
            Counter(item["partition"] for item in context_records)
        ),
        "sentinel_count": sum(bool(item["sentinel"]) for item in context_records),
        "contexts": context_records,
    }
    source_document_count = len(
        {
            (
                item["dataset"],
                item["dataset_config"],
                item["dataset_split"],
                item["source_id"],
            )
            for item in context_records
        }
    )
    validation = {
        "status": "implemented",
        "context_count": len(context_records),
        "all_contexts_have_2048_tokens": all(
            item["num_tokens"] == CONTEXT_LENGTH for item in context_records
        ),
        "all_contexts_have_distinct_source_documents": (
            source_document_count == len(context_records)
        ),
        "distinct_source_documents": source_document_count,
        "distinct_token_hashes": len(
            {item["token_ids_json_sha256"] for item in context_records}
        ),
        "software_repository_count": len(
            {
                item.source_cluster_id
                for item in ordered
                if item.allocation_stratum
                in {
                    "code_tests_documentation_issues",
                    "structured_data_tools_apis_tables",
                }
            }
        ),
        "allocation_counts": manifest["allocation_counts"],
        "partition_counts": manifest["partition_counts"],
        "sentinel_count": manifest["sentinel_count"],
    }
    write_json(args.output_dir / "source-registry.json", source_registry)
    write_json(args.output_dir / "sources.json", source_records)
    write_json(args.output_dir / "partitions.json", partitions)
    write_json(args.output_dir / "suite-manifest.json", manifest)
    validation_dir = args.output_dir / "validation"
    validation_dir.mkdir()
    write_json(validation_dir / "structural-validation.json", validation)

    checksums = []
    for path in sorted(item for item in args.output_dir.rglob("*") if item.is_file()):
        if path.name == "checksums.txt":
            continue
        checksums.append(f"{sha256_file(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "checksums.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
