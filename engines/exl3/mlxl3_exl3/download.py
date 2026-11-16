"""Download an EXL3 checkpoint (optionally one bitrate subfolder) from Hugging Face."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path, PurePosixPath

REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
REVISION_RE = re.compile(r"^[A-Za-z0-9._/-]{1,128}$")


class DownloadError(ValueError):
    pass


def validate(repo_id: str, folder: str, revision: str | None) -> str:
    if not REPO_RE.fullmatch(repo_id) or ".." in repo_id:
        raise DownloadError("repository must look like owner/name")
    if revision is not None and (not REVISION_RE.fullmatch(revision) or ".." in revision):
        raise DownloadError("invalid revision")
    folder = folder.strip().strip("/")
    if folder:
        parts = PurePosixPath(folder).parts
        if any(p in ("", ".", "..") for p in parts) or "\\" in folder or ":" in folder:
            raise DownloadError("folder must be a relative path inside the repository")
    return folder


def _emit(event: dict) -> None:
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def download(repo_id: str, destination: str, folder: str = "", revision: str | None = None) -> Path:
    folder = validate(repo_id, folder, revision)
    from huggingface_hub import snapshot_download
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    _emit({"type": "download_started", "repo": repo_id, "folder": folder})
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(target),
                      allow_patterns=[f"{folder}/*"] if folder else None)
    model_dir = target / folder if folder else target
    if not (model_dir / "config.json").is_file():
        raise DownloadError(f"no config.json in {model_dir}")
    _emit({"type": "download_done", "path": str(model_dir.resolve())})
    return model_dir
