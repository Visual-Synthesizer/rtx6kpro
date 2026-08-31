#!/usr/bin/env python3
"""Build a deterministic evidence archive for a paired Kimi-K3 AA-LCR run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any


RUN_EXCLUDES = {
    Path("runtime-artifacts/docker-inspect.json"),
}
TOOL_NAMES = (
    "run-kimi-k3-aa-lcr.py",
    "judge-kimi-k3-aa-lcr-codex.py",
    "compare-kimi-k3-aa-lcr-scores.py",
    "package-kimi-k3-aa-lcr-evidence.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink():
        raise RuntimeError(f"evidence input must not be a symbolic link: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_run(source: Path, destination: Path) -> list[str]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    copied: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if relative in RUN_EXCLUDES:
            continue
        copy_file(path, destination / relative)
        copied.append(relative.as_posix())
    return copied


def deterministic_tar_gz(source: Path, output: Path) -> None:
    with output.open("wb") as raw_output:
        with gzip.GzipFile(fileobj=raw_output, mode="wb", mtime=0) as compressed:
            with tarfile.open(
                fileobj=compressed,
                mode="w",
                format=tarfile.PAX_FORMAT,
            ) as archive:
                for path in [source, *sorted(source.rglob("*"))]:
                    relative = path.relative_to(source.parent)
                    information = archive.gettarinfo(str(path), arcname=str(relative))
                    information.uid = 0
                    information.gid = 0
                    information.uname = ""
                    information.gname = ""
                    information.mtime = 0
                    if path.is_dir():
                        information.mode = 0o755
                        archive.addfile(information)
                    elif path.is_file():
                        information.mode = 0o755 if path.name.endswith(".py") else 0o644
                        with path.open("rb") as handle:
                            archive.addfile(information, handle)
                    else:
                        raise RuntimeError(
                            f"evidence archive input has an unsupported type: {path}"
                        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--candidate-name", default="qsrt-k2")
    parser.add_argument("--k3-comparison", type=Path, required=True)
    parser.add_argument("--sol-comparison", type=Path, required=True)
    parser.add_argument("--token-count-manifest", type=Path, required=True)
    parser.add_argument("--tools-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive_root_name = output.name.removesuffix(".tar.gz")

    with tempfile.TemporaryDirectory(
        prefix=f".{archive_root_name}-", dir=output.parent
    ) as temporary_name:
        archive_root = Path(temporary_name) / archive_root_name
        archive_root.mkdir()
        official_files = copy_run(
            args.official_run.resolve(), archive_root / "runs/official-mxfp4"
        )
        candidate_files = copy_run(
            args.candidate_run.resolve(), archive_root / f"runs/{args.candidate_name}"
        )
        copy_file(
            args.k3_comparison.resolve(),
            archive_root / "comparisons/frozen-official-kimi-k3.json",
        )
        copy_file(
            args.sol_comparison.resolve(),
            archive_root / "comparisons/gpt-5.6-sol-max.json",
        )
        copy_file(
            args.token_count_manifest.resolve(),
            archive_root / "dataset/kimi-k3-token-counts.json",
        )
        for name in TOOL_NAMES:
            copy_file(args.tools_dir.resolve() / name, archive_root / "tools" / name)

        official_manifest = json.loads(
            (archive_root / "runs/official-mxfp4/generation-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        candidate_manifest = json.loads(
            (
                archive_root / f"runs/{args.candidate_name}/generation-manifest.json"
            ).read_text(encoding="utf-8")
        )
        k3_comparison = json.loads(
            (archive_root / "comparisons/frozen-official-kimi-k3.json").read_text(
                encoding="utf-8"
            )
        )
        sol_comparison = json.loads(
            (archive_root / "comparisons/gpt-5.6-sol-max.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = {
            "artifact_kind": "Kimi-K3 AA-LCR paired checkpoint evidence archive",
            "status": "qualified",
            "dataset": official_manifest["dataset"],
            "runs": {
                "official_mxfp4": {
                    "generation_manifest_sha256": sha256_file(
                        archive_root / "runs/official-mxfp4/generation-manifest.json"
                    ),
                    "files": len(official_files),
                    "served_model": official_manifest["generation"]["model"],
                },
                args.candidate_name: {
                    "generation_manifest_sha256": sha256_file(
                        archive_root
                        / f"runs/{args.candidate_name}/generation-manifest.json"
                    ),
                    "files": len(candidate_files),
                    "served_model": candidate_manifest["generation"]["model"],
                },
            },
            "results": {
                "frozen_official_kimi_k3": {
                    "candidate_minus_official": k3_comparison["paired"][
                        "candidate_minus_reference"
                    ],
                    "candidate_pass_at_1": k3_comparison["candidate"]["pass_at_1"],
                    "official_pass_at_1": k3_comparison["reference"]["pass_at_1"],
                },
                "gpt_5_6_sol_max": {
                    "candidate_minus_official": sol_comparison["paired"][
                        "candidate_minus_reference"
                    ],
                    "candidate_pass_at_1": sol_comparison["candidate"]["pass_at_1"],
                    "official_pass_at_1": sol_comparison["reference"]["pass_at_1"],
                },
            },
            "omissions": {
                "aa_lcr_documents": (
                    "Fetch the immutable ArtificialAnalysis/AA-LCR revision named "
                    "in the dataset object; source documents are not redistributed."
                ),
                "checkpoint_weights": "Fetch the checkpoint revisions named by each run manifest.",
                "docker_inspect": (
                    "Raw Docker inspection records are excluded because process "
                    "environments can contain credentials. Runtime manifests retain "
                    "the image digest, source revisions, arguments, and relevant "
                    "non-secret environment variables."
                ),
            },
        }
        write_json(archive_root / "manifest.json", manifest)
        readme = f"""# Kimi-K3 AA-LCR paired checkpoint evidence

This archive contains the complete generation and equality-checker receipts for
the official MXFP4 checkpoint and `{args.candidate_name}` checkpoint comparison.
It also contains the frozen runtime manifests, comparison receipts, and Python
utilities needed to validate or repeat the evaluation.

The archive does not redistribute checkpoint weights or the AA-LCR source
documents. Fetch the immutable repository revisions recorded in
`manifest.json`. Raw Docker inspection output is excluded because a process
environment may contain credentials.

Verify every archived file from the extracted archive root:

```bash
sha256sum --check checksums.sha256
```

The standalone reproduction specification is published in the
`local-inference-lab/rtx6kpro` repository under `models/kimi-k3/`.
"""
        (archive_root / "README.md").write_text(readme, encoding="utf-8")

        checksummed = [
            path
            for path in sorted(archive_root.rglob("*"))
            if path.is_file() and path.name != "checksums.sha256"
        ]
        checksum_lines = [
            f"{sha256_file(path)}  {path.relative_to(archive_root).as_posix()}"
            for path in checksummed
        ]
        (archive_root / "checksums.sha256").write_text(
            "\n".join(checksum_lines) + "\n", encoding="utf-8"
        )
        deterministic_tar_gz(archive_root, output)

    receipt = {
        "artifact_kind": "Kimi-K3 AA-LCR evidence archive receipt",
        "status": "qualified",
        "archive": output.name,
        "sha256": sha256_file(output),
        "size_bytes": output.stat().st_size,
    }
    receipt_path = output.with_suffix(output.suffix + ".json")
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
