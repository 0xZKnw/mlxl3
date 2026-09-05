import json
import inspect
from fnmatch import fnmatch
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlxl3 import hub
from mlxl3.registry import RegistryError, load_registry


def test_variants_are_isolated_and_share_only_metadata():
    files = {"config.json": 10, "tokenizer.json": 20, "run.py": 999,
             "3bpw/quantization_config.json": 30, "3bpw/model.safetensors": 300,
             "4bpw/quantization_config.json": 40, "4bpw/model.safetensors": 400,
             "../../model.safetensors": 999, "/config.json": 99}
    low, high = hub.variants(files)
    assert low["id"] == "3bpw" and low["size_bytes"] == 360
    assert set(low["files"].values()) == {"config.json", "tokenizer.json", "3bpw/quantization_config.json", "3bpw/model.safetensors"}
    assert high["size_bytes"] == 470
    same_folder = {"config.json": 10, "quantization_config_3.0bpw.json": 30,
                   "model-3.0bpw.safetensors": 300, "quantization_config_4.0bpw.json": 40,
                   "model-4.0bpw.safetensors": 400}
    low, high = hub.variants(same_folder)
    assert low["size_bytes"] == 340 and high["size_bytes"] == 450
    assert low["files"]["quantization_config.json"] == "quantization_config_3.0bpw.json"
    assert not any("4.0bpw" in name for name in low["files"].values())
    ambiguous = {"config.json": 10, "quantization_config.json": 30,
                 "model-3.0bpw.safetensors": 300, "model-4.0bpw.safetensors": 400}
    assert hub.variants(ambiguous) == []


def test_main_card_only_chooses_quant_branch(monkeypatch, tmp_path):
    card = tmp_path / "README.md"
    card.write_text("---\nlicense: mit\n---\n# Model\nRead me.")
    class API:
        def list_repo_refs(self, repo):
            return SimpleNamespace(branches=[SimpleNamespace(name=x) for x in ["main", "4.0bpw", "3.0bpw"]], tags=[])
        def model_info(self, repo, revision, **kwargs):
            names = ["README.md"] if revision == "main" else ["README.md", "config.json", "quantization_config.json", "model.safetensors"]
            return SimpleNamespace(sha="a" * 40, siblings=[SimpleNamespace(rfilename=n, size=100) for n in names], gated=False, downloads=2, likes=1)
    monkeypatch.setattr(hub, "HfApi", API)
    monkeypatch.setattr(hub, "hf_hub_download", lambda *a, **kw: str(card))
    detail = hub.details("test/model-exl3")
    assert detail["revision"] == "3.0bpw" and detail["commit"] == "a" * 40
    assert len(detail["variants"]) == 1
    assert detail["readme"].startswith("# Model")
    card = '<div><h2>Model</h2><a href="https://example.com"><img alt="badge" src="x"/></a></div>\n```html\n<div>example</div>\n```\n`<think>` <think>text</think><script>untrusted()</script>'
    cleaned = hub.readable_card(card)
    assert "## Model" in cleaned and "<img" not in cleaned and "untrusted()" not in cleaned
    assert "```html\n<div>example</div>\n```" in cleaned
    assert "`<think>` <think>text</think>" in cleaned


def test_download_pins_commit_filters_files_and_never_overwrites(monkeypatch, tmp_path):
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path / "registry"))
    monkeypatch.setenv("MLXL3_MODELS_DIR", str(tmp_path / "models"))
    config = json.dumps({"model_type": "test"}).encode()
    quant = json.dumps({"quant_method": "exl3", "bits": 3.0, "tensor_storage": {"x": {"quant_format": "exl3"}}}).encode()
    files = {"config.json": config, "tokenizer[shared].json": b"{}", "3bpw/quantization_config.json": quant,
             "3bpw/model.safetensors": b"fixture", "4bpw/quantization_config.json": quant,
             "4bpw/model.safetensors": b"must-not-download", "run.py": b"must-not-run"}
    class API:
        def model_info(self, repo, revision, **kwargs):
            assert revision == "a" * 40
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename=n, size=len(v)) for n, v in files.items()])
    monkeypatch.setattr(hub, "HfApi", API)
    downloads = []
    attempts = 0
    def snapshot(**kwargs):
        nonlocal attempts
        attempts += 1
        assert kwargs["revision"] == "a" * 40
        names = [name for name in files if any(fnmatch(name, pattern) for pattern in kwargs["allow_patterns"])]
        assert "tokenizer[shared].json" in names
        downloads.extend(names)
        with pytest.raises(RegistryError, match="already downloading"):
            hub.download("test/model-exl3", "a" * 40, "3bpw", lambda e: None)
        for name in names:
            target = Path(kwargs["local_dir"]) / name
            if attempts > 1:
                assert target.exists(), "An interrupted download must retain its stage"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files[name])
        if attempts == 1:
            raise RuntimeError("interrupted fixture transfer")
        progress = kwargs["tqdm_class"](total=100, unit="B", desc="Reconstructing")
        from huggingface_hub.utils._xet_progress_reporting import _set_aggregate_rate_postfix
        transfer = kwargs["tqdm_class"](total=100, unit="B", desc="Downloading bytes")
        transfer.update(25)
        _set_aggregate_rate_postfix(transfer)
        progress.update(50)
        _set_aggregate_rate_postfix(progress)
        progress.update(None)
        transfer.close()
        progress.close()
    monkeypatch.setattr(hub, "snapshot_download", snapshot)
    events = []
    with pytest.raises(RuntimeError, match="interrupted"):
        hub.download("test/model-exl3", "a" * 40, "3bpw", events.append)
    assert not load_registry()
    entry = hub.download("test/model-exl3", "a" * 40, "3bpw", events.append)
    assert entry.name in load_registry()
    assert (Path(entry.path) / "model.safetensors").read_bytes() == b"fixture"
    assert not any("4bpw" in name or name == "run.py" for name in downloads)
    assert any(e["completed"] == 50 for e in events)
    with pytest.raises(RegistryError, match="already installed"):
        hub.download("test/model-exl3", "a" * 40, "3bpw", events.append)
    with pytest.raises(RegistryError, match="resolved revision"):
        hub.download("test/model-exl3", "main", "3bpw", events.append)


def test_search_filters_and_deduplicates(monkeypatch):
    signature = inspect.signature(hub.HfApi.list_models)
    class API:
        def list_models(self, **kwargs):
            signature.bind(self, **kwargs)
            assert kwargs["limit"] <= 100
            return [SimpleNamespace(id=n, tags=tags, downloads=3, likes=1, gated=False) for n, tags in
                    [("a/Qwen-exl3", []), ("b/Qwen", ["exl3"]), ("c/Qwen-GGUF", []), ("d/gemma-exl3", [])]]
    monkeypatch.setattr(hub, "HfApi", API)
    assert {m["id"] for m in hub.search("Qwen")} == {"a/Qwen-exl3", "b/Qwen"}


def test_remove_expected_path_protects_registry(monkeypatch, tmp_path):
    from mlxl3 import cli
    from mlxl3.registry import ModelEntry, save_registry
    monkeypatch.setenv("MLXL3_HOME", str(tmp_path))
    entry = ModelEntry("test", str(tmp_path / "model"), "test", "EXL3", 3.0, 1, 0, "")
    save_registry({"test": entry})
    assert cli.main(["remove", "test", "--expected-path", str(tmp_path / "other")]) != 0
    assert "test" in load_registry()
    assert cli.main(["remove", "test", "--expected-path", entry.path]) == 0
    assert not load_registry()
