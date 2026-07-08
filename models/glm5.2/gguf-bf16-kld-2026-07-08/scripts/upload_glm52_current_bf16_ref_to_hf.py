#!/usr/bin/env python3
"""Upload the 2026-07-08 GLM-5.2 BF16 KLD reference logits to HF."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def token_from_env_or_file(path: str) -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    p = Path(path)
    if p.exists():
        return p.read_text().strip()
    raise RuntimeError("Set HF_TOKEN or provide --token-file")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="festr2/GLM-5.2-BF16-KLD-Reference-Logits-20260708",
    )
    parser.add_argument(
        "--folder",
        default="/root/kld/hf_upload_glm52_current_bf16_ref_20260708",
    )
    parser.add_argument("--token-file", default="/root/vllm/.hf_write_token")
    args = parser.parse_args()

    token = token_from_env_or_file(args.token_file)
    api = HfApi(token=token)
    who = api.whoami()
    print(f"authenticated_as={who.get('name')}", flush=True)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=args.folder,
        path_in_repo=".",
        commit_message="Add GLM-5.2 BF16 KLD reference logits 20260708",
    )
    info = api.dataset_info(args.repo_id)
    print(f"url=https://huggingface.co/datasets/{args.repo_id}", flush=True)
    print(f"sha={info.sha}", flush=True)


if __name__ == "__main__":
    main()
