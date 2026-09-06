"""Persistent per-user registry for local MLXL3 models."""

from __future__ import annotations

import json
import os
import re
import fcntl
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")


class RegistryError(ValueError):
    """Invalid registry state or operation."""


@dataclass(frozen=True)
class ModelEntry:
    name: str
    path: str
    model_type: str
    format: str
    bits: float | None
    size_bytes: int
    modules: int
    added_at: str

    @classmethod
    def from_dict(cls, value: dict) -> ModelEntry:
        return cls(**value)


def validate_model_name(name: str) -> None:
    if not _NAME_RE.fullmatch(name):
        raise RegistryError(
            "model names must start with a letter or digit and contain only "
            "letters, digits, '.', '_', ':', '/', or '-'"
        )


def registry_path() -> Path:
    override = os.environ.get("MLXL3_HOME")
    root = Path(override).expanduser() if override else Path.home() / ".config" / "mlxl3"
    return root / "models.json"


def managed_models_path() -> Path:
    """Return the user-owned directory used by Desktop model downloads."""

    override = os.environ.get("MLXL3_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / "io.mlxl3.desktop" / "Models"


def load_registry() -> dict[str, ModelEntry]:
    path = registry_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("version") != 1 or not isinstance(payload.get("models"), dict):
            raise RegistryError(f"unsupported registry format in {path}")
        return {
            name: ModelEntry.from_dict({"name": name, **metadata})
            for name, metadata in payload["models"].items()
        }
    except (json.JSONDecodeError, TypeError) as error:
        raise RegistryError(f"invalid model registry {path}: {error}") from error


@contextmanager
def registry_lock(path: Path | None = None):
    path = path or registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def save_registry(entries: dict[str, ModelEntry]) -> None:
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    models = {}
    for name, entry in sorted(entries.items()):
        metadata = asdict(entry)
        metadata.pop("name")
        models[name] = metadata
    atomic_json(path, {"version": 1, "models": models})


def inspect_model(name: str, model_path: str | Path) -> ModelEntry:
    from mlxl3.checkpoint import list_exl3_modules, quantization_config, validate_checkpoint_files

    validate_model_name(name)
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise RegistryError(f"model directory does not exist: {path}")
    config_path = path / "config.json"
    if not config_path.exists():
        raise RegistryError(f"missing model config: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RegistryError(f"invalid model config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise RegistryError('model config must be a JSON object')
    quantization = quantization_config(path)
    validate_checkpoint_files(path)
    if not any(path.glob('tokenizer*')) and not any(path.glob('*.model')) and not (path / 'vocab.json').exists():
        raise RegistryError('missing tokenizer files; download the complete checkpoint')
    size_bytes = sum(file.stat().st_size for file in path.iterdir() if file.is_file())
    return ModelEntry(
        name=name,
        path=str(path),
        model_type=str(config.get("model_type", "unknown")),
        format="EXL3",
        bits=float(quantization["bits"]) if quantization.get("bits") is not None else None,
        size_bytes=size_bytes,
        modules=len(list_exl3_modules(path)),
        added_at=datetime.now(UTC).isoformat(),
    )


def register_model(name: str, model_path: str | Path, *, force: bool = False) -> ModelEntry:
    entry = inspect_model(name, model_path)
    with registry_lock():
        entries = load_registry()
        existing = entries.get(name)
        if existing is not None and existing.path != entry.path and not force:
            raise RegistryError(f"model {name!r} already exists; pass --force to replace it")
        entries[name] = entry
        save_registry(entries)
    return entry


def remove_model(name: str, *, expected_path: str | None = None) -> ModelEntry:
    with registry_lock():
        entries = load_registry()
        try:
            removed = entries.pop(name)
        except KeyError as error:
            raise RegistryError(f"unknown model {name!r}") from error
        if expected_path is not None and Path(removed.path).resolve() != Path(expected_path).resolve():
            raise RegistryError("model path changed; refresh the library before removing")
        save_registry(entries)
    return removed


def resolve_model(name_or_path: str) -> tuple[str, Path]:
    candidate = Path(name_or_path).expanduser()
    if candidate.is_dir():
        return candidate.name, candidate.resolve()
    entry = load_registry().get(name_or_path)
    if entry is None:
        raise RegistryError(
            f"unknown model {name_or_path!r}; use 'mlxl3 list' or 'mlxl3 register NAME PATH'"
        )
    path = Path(entry.path)
    if not path.is_dir():
        raise RegistryError(f"registered model path no longer exists: {path}")
    return entry.name, path


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1000.0
    raise AssertionError("unreachable")
