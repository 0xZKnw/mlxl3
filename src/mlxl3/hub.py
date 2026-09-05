"""Small Hub catalogue: metadata first, explicit revision/folder downloads only."""
from __future__ import annotations

import fcntl
import glob
import hashlib
from html.parser import HTMLParser
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import threading
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import validate_repo_id

from mlxl3.registry import RegistryError, inspect_model, load_registry, managed_models_path, register_model


def _safe_file(name: str) -> bool:
    return bool(name) and not name.startswith("/") and "\\" not in name and all(
        part not in ("", ".", "..") for part in name.split("/")
    )

def search(query: str) -> list[dict]:
    query = query.strip()[:160]
    terms = query.lower().split()
    needle = terms[0] if terms else None
    api = HfApi()
    if "/" in query:
        try:
            validate_repo_id(query)
            model = api.model_info(query)
            if "exl3" in model.id.lower() or "exl3" in (model.tags or []):
                return [{"id": model.id, "downloads": model.downloads or 0,
                         "likes": model.likes or 0, "gated": bool(model.gated)}]
        except Exception:
            pass
    # Server-side EXL3 filter plus name fallback for repositories missing the tag.
    found = {}
    for kwargs in ({"search": needle, "filter": "exl3"}, {"search": needle or "exl3"}):
        for model in api.list_models(**kwargs, sort="downloads", limit=100):
            if "exl3" not in model.id.lower() and "exl3" not in (model.tags or []):
                continue
            if not all(term in model.id.lower() for term in terms):
                continue
            found[model.id] = {"id": model.id, "downloads": model.downloads or 0,
                               "likes": model.likes or 0, "gated": bool(model.gated)}
    return sorted(found.values(), key=lambda item: (-item["downloads"], item["id"]))[:60]


def variants(files: dict[str, int]) -> list[dict]:
    """Each EXL3 descriptor owns weights in its directory; inherit shared tokenizer files."""
    files = {name: size for name, size in files.items() if _safe_file(name)}
    result = []
    descriptors = {name: re.fullmatch(r"quantization_config(?:[._-](.+))?\.json", PurePosixPath(name).name)
                   for name in files}
    for descriptor in sorted(files):
        match = descriptors[descriptor]
        if match is None:
            continue
        quant_label = match.group(1)
        folder = str(PurePosixPath(descriptor).parent)
        own = {PurePosixPath(name).name: name for name in files
               if str(PurePosixPath(name).parent) == folder}
        weights = [name for base, name in own.items() if base.startswith("model") and base.endswith(".safetensors")]
        def belongs(name, label):
            return re.search(r"(?<![A-Za-z0-9.])" + re.escape(label) + r"(?![A-Za-z0-9.])", PurePosixPath(name).stem, re.I) is not None
        if quant_label:
            weights = [name for name in weights if belongs(name, quant_label)]
        else:
            other_labels = [m.group(1) for name, m in descriptors.items()
                            if m and m.group(1) and str(PurePosixPath(name).parent) == folder]
            weights = [name for name in weights if not any(belongs(name, label) for label in other_labels)]
        if not weights:
            continue
        weight_quants = {match[0].lower() for name in weights
                         if (match := re.search(r"\d+(?:\.\d+)?bpw", name, re.I))}
        if len(weight_quants) > 1:
            # A shared descriptor cannot safely identify mixed quantizations in one directory.
            continue
        selected = {}
        # Root and nearest ancestors may contain shared config/tokenizer assets.
        parents = list(reversed(PurePosixPath(folder).parents)) + [PurePosixPath(folder)]
        for parent in parents:
            for name in files:
                path = PurePosixPath(name)
                base = path.name
                if path.parent == parent and (
                    base in ("config.json", "generation_config.json", "merges.txt", "vocab.json", "vocab.txt", "special_tokens_map.json")
                    or base.startswith(("tokenizer", "chat_template", "preprocessor", "processor", "LICENSE", "README"))
                    or base.endswith((".tiktoken", ".model"))
                ):
                    selected[base] = name
        for base, name in own.items():
            if name in weights or (not quant_label and base == "model.safetensors.index.json"):
                selected[base] = name
        selected["quantization_config.json"] = descriptor
        if quant_label:
            for base, name in own.items():
                if base.startswith("config") and base.endswith(".json") and belongs(base, quant_label):
                    selected["config.json"] = name
        if "config.json" not in selected:
            continue
        identity = "" if folder == "." else folder
        result.append({"id": identity + ("#" + PurePosixPath(descriptor).name if quant_label else ""),
                       "label": quant_label or ("Root" if folder == "." else folder),
                       "size_bytes": sum(files[name] for name in selected.values()),
                       "files": selected})
    return result


def readable_card(text: str) -> str:
    """Remove decorative HTML from model cards, keeping code examples untouched."""
    class CardHTML(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self.hidden = 0
        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self.hidden += 1
            elif not self.hidden:
                if tag in ("div", "p", "br", "table", "tr"):
                    self.parts.append("\n")
                elif re.fullmatch(r"h[1-6]", tag):
                    self.parts.append("\n" + "#" * int(tag[1]) + " ")
                elif tag == "img":
                    pass  # Decorative badges are omitted; no remote image fetches.
                elif tag not in ("a", "span", "b", "strong", "em", "i", "td", "th", "details", "summary", "code"):
                    self.parts.append(self.get_starttag_text())
        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self.hidden = max(0, self.hidden - 1)
            elif not self.hidden and (tag in ("div", "p", "tr", "table") or re.fullmatch(r"h[1-6]", tag)):
                self.parts.append("\n")
            elif not self.hidden and tag not in ("a", "span", "b", "strong", "em", "i", "td", "th", "details", "summary", "code", "br", "img"):
                self.parts.append(f"</{tag}>")
        def handle_data(self, data):
            if not self.hidden:
                self.parts.append(data)
    text = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)
    chunks = re.split(r"(```.*?```|~~~.*?~~~|`[^`\n]+`)", text, flags=re.S)
    for index in range(0, len(chunks), 2):
        parser = CardHTML()
        parser.feed(chunks[index])
        parser.close()
        chunks[index] = "".join(parser.parts)
    return "".join(chunks)


def details(repo: str, revision: str | None = None) -> dict:
    validate_repo_id(repo)
    api = HfApi()
    refs = api.list_repo_refs(repo)
    branches = [r.name for r in refs.branches] + [r.name for r in refs.tags]
    info = api.model_info(repo, revision=revision or "main", files_metadata=True)
    files = {f.rfilename: f.size or 0 for f in info.siblings or []}
    choices = variants(files)
    selected_revision = revision or "main"
    # Many EXL3 repos put only a model card on main, weights on BPW branches.
    if not choices and revision is None:
        quant_branches = [r for r in branches if re.search(r"\d.*bpw", r, re.I)]
        if quant_branches:
            selected_revision = min(quant_branches, key=lambda r: abs(float(re.search(r"\d+(?:\.\d+)?", r)[0]) - 3))
            info = api.model_info(repo, revision=selected_revision, files_metadata=True)
            files = {f.rfilename: f.size or 0 for f in info.siblings or []}
            choices = variants(files)
    readme = ""
    # Model cards are untrusted display text. Never import or execute repository code.
    for ref in dict.fromkeys([info.sha, "main"]):
        try:
            metadata = files if ref == info.sha else {f.rfilename: f.size or 0 for f in api.model_info(repo, revision=ref, files_metadata=True).siblings or []}
            if 0 < metadata.get("README.md", 0) <= 256_000:
                readme = Path(hf_hub_download(repo, "README.md", revision=ref)).read_text(errors="replace")
                readme = readable_card(readme)
                break
        except Exception:
            continue
    return {"id": repo, "revision": selected_revision, "commit": info.sha,
            "branches": list(dict.fromkeys(branches)), "variants": choices,
            "readme": readme, "gated": bool(info.gated),
            "downloads": info.downloads or 0, "likes": info.likes or 0}


def download(repo: str, commit: str, folder: str, emit) -> object:
    validate_repo_id(repo)
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RegistryError("Select a resolved revision before downloading")
    info = HfApi().model_info(repo, revision=commit, files_metadata=True)
    choices = variants({f.rfilename: f.size or 0 for f in info.siblings or []})
    selected = next((v for v in choices if v["id"] == folder), None)
    if selected is None:
        raise RegistryError("No complete EXL3 checkpoint in the selected folder")
    key = hashlib.sha256(f"{repo}\n{commit}\n{folder}".encode()).hexdigest()[:12]
    label = re.sub(r"[^A-Za-z0-9._-]+", "-", repo.rsplit("/", 1)[-1] + ("-" + folder if folder else ""))[:100]
    name = f"{label}-{key[:8]}"
    root = managed_models_path().resolve()
    destination = root / name
    if name in load_registry() or destination.exists():
        raise RegistryError("This checkpoint is already installed. Import its folder or choose another variant.")
    stage = root / ".downloads" / key
    stage.mkdir(parents=True, exist_ok=True)
    # Keep the lock outside the resumable stage, whose contents are removed on success.
    with (stage.parent / (key + ".lock")).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RegistryError("This variant is already downloading") from error
        return _download_selected(repo, commit, selected, stage, destination, name, emit)


def _download_selected(repo, commit, selected, stage, destination, name, emit):
    retained = sum(p.stat().st_size for p in (stage / "snapshot").rglob("*") if p.is_file())
    if shutil.disk_usage(stage).free < max(0, selected["size_bytes"] - retained) + 512_000_000:
        raise RegistryError("Not enough free disk space for this checkpoint")
    from tqdm.auto import tqdm
    progress_lock = threading.Lock()

    class Progress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            self.reported = 0.0
            super().__init__(*args, **kwargs)
            self.unit = kwargs.get("unit", "it")
            self.desc = kwargs.get("desc", "")
        def update(self, n=1):
            with progress_lock:
                self.n += n
                now = time.monotonic()
                if self.unit == "B" and "Reconstruct" in (self.desc or "") and now - self.reported > 0.15:
                    self.reported = now
                    emit({"type": "progress", "completed": self.n, "total": selected["size_bytes"]})
        def refresh(self, *args, **kwargs):
            pass

    emit({"type": "progress", "completed": 0, "total": selected["size_bytes"]})
    snapshot_download(repo_id=repo, revision=commit, local_dir=stage / "snapshot",
                      allow_patterns=[glob.escape(path) for path in selected["files"].values()],
                      max_workers=4, tqdm_class=Progress)
    assembled = stage / "model"
    assembled.mkdir(exist_ok=True)
    for base, remote in selected["files"].items():
        target = assembled / base
        if target.exists():
            target.unlink()
        os.link(stage / "snapshot" / remote, target)
    inspect_model(name, assembled)
    if destination.exists():
        raise RegistryError("Destination already exists; refusing to overwrite it")
    assembled.rename(destination)
    entry = register_model(name, destination)
    shutil.rmtree(stage, ignore_errors=True)
    return entry
