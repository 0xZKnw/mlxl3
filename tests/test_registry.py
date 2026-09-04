from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlxl3.registry import (
    RegistryError,
    load_registry,
    managed_models_path,
    register_model,
    remove_model,
    resolve_model,
)


def _fake_model(path: Path, *, model_type: str = "lfm2_moe") -> Path:
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": model_type}))
    (path / "quantization_config.json").write_text(
        json.dumps(
            {
                "quant_method": "exl3",
                "bits": 3.1,
                "tensor_storage": {"lm_head": {"quant_format": "exl3"}},
            }
        )
    )
    (path / "model.safetensors").write_bytes(b"fixture")
    return path


def test_register_list_resolve_and_remove(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path / "registry"))
    model_path = _fake_model(tmp_path / "model")

    entry = register_model("lfm-test", model_path)
    assert entry.bits == 3.1
    assert entry.modules == 1
    assert load_registry()["lfm-test"] == entry
    assert resolve_model("lfm-test") == ("lfm-test", model_path.resolve())

    removed = remove_model("lfm-test")
    assert removed == entry
    assert load_registry() == {}
    assert model_path.exists()


def test_register_refuses_replacing_a_different_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path / "registry"))
    register_model("same-name", _fake_model(tmp_path / "first"))

    with pytest.raises(RegistryError, match="already exists"):
        register_model("same-name", _fake_model(tmp_path / "second"))


def test_resolve_accepts_a_direct_directory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path / "registry"))
    model_path = _fake_model(tmp_path / "direct")
    assert resolve_model(str(model_path)) == ("direct", model_path.resolve())


def test_managed_models_path_honors_override(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MLXL3_MODELS_DIR", str(tmp_path / "models"))
    assert managed_models_path() == tmp_path / "models"
