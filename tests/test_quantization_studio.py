import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "quantization_studio", ROOT / "scripts/quantization_studio.py"
)
studio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(studio)


@pytest.fixture
def source(tmp_path):
    tmp_path = tmp_path / "source"
    tmp_path.mkdir()
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "lfm2", "tie_word_embeddings": True})
    )
    (tmp_path / "tokenizer.json").write_text("{}")
    save_file(
        {
            "model.layers.0.feed_forward.w1.weight": np.zeros((128, 128), np.float16),
            "scalar": np.array(1, np.float32),
        },
        tmp_path / "model.safetensors",
    )
    return tmp_path


@pytest.mark.parametrize(
    "value,expected",
    [
        ("LiquidAI/LFM2.5-2.6B", ("LiquidAI/LFM2.5-2.6B", "main")),
        ("https://huggingface.co/owner/model", ("owner/model", "main")),
        ("https://huggingface.co/owner/model/tree/v1", ("owner/model", "v1")),
        ("https://hf.co/owner/model/resolve/main/model.safetensors", ("owner/model", "main")),
    ],
)
def test_hub_reference(value, expected):
    assert studio.hub_reference(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "https://evil.test/owner/model",
        "http://huggingface.co/owner/model",
        "https://huggingface.co@evil.test/a/b",
        "https://huggingface.co:443/a/b",
        "../model",
        "owner/../file",
        "https://huggingface.co/a/b/tree/..",
        "https://huggingface.co/a/b?token=secret",
        "a/b/c",
        "a/b\0",
    ],
)
def test_hub_reference_rejects_non_model_inputs(value):
    with pytest.raises(ValueError):
        studio.hub_reference(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("bits", True),
        ("bits", math.nan),
        ("bits", math.inf),
        ("bits", 1.9),
        ("bits", 8.1),
        ("rows", 0),
        ("rows", 8193),
        ("rows", 32.0),
        ("sequence", 4097),
        ("head_bits", 1),
        ("excluded", ["../secret"]),
        ("excluded", ["x;rm -rf /"]),
        ("keep_head", "yes"),
    ],
)
def test_settings_fail_closed(field, value):
    with pytest.raises(ValueError):
        studio.settings({field: value})


def test_settings_boundaries_and_holdout(tmp_path):
    for bits in (2, 3.5, 8):
        assert studio.settings({"bits": bits, "rows": 32, "sequence": 4096})["bits"] == bits
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("same corpus")
    b.write_text("same corpus")
    with pytest.raises(ValueError, match="separate"):
        studio.settings({"calibration": str(a), "validation": str(b)})
    b.write_text("held out corpus")
    assert studio.settings({"validation": str(b)})["validation"] == str(b)


def test_probe_headers_and_source_change(source):
    result = studio.inspect_local(source)
    assert (
        result["inference_supported"]
        and result["conversion_supported"]
        and result["header_checked"]
    )
    assert result["parameters"] == 128 * 128 + 1
    assert result["modules"] == ["model.layers.0.feed_forward.w1"]
    assert studio.probe(str(source / "model.safetensors"))["fingerprint"] == result["fingerprint"]
    tokenizer = source / "tokenizer.json"
    tokenizer.write_text('{"changed": true}')
    assert studio.inspect_local(source)["fingerprint"] != result["fingerprint"]
    tokenizer.write_text("{}")
    config = source / "config.json"
    config.write_text(config.read_text() + " ")
    assert studio.inspect_local(source)["fingerprint"] != result["fingerprint"]


def test_inference_and_conversion_are_separate(source):
    (source / "config.json").write_text('{"model_type":"qwen3_5"}')
    result = studio.inspect_local(source)
    assert result["inference_supported"] and not result["conversion_supported"]
    (source / "config.json").write_text('{"model_type":"unknown"}')
    result = studio.inspect_local(source)
    assert not result["inference_supported"] and not result["conversion_supported"]


def test_rejects_partial_mixed_linked_and_quantized_weights(source):
    index = source / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"x":"../secret.safetensors"}}')
    with pytest.raises(ValueError, match="index"):
        studio.inspect_local(source)
    index.unlink()
    other = source / "other.safetensors"
    other.write_bytes((source / "model.safetensors").read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        studio.inspect_local(source)
    other.unlink()
    other.symlink_to(source / "model.safetensors")
    with pytest.raises(ValueError, match="Linked"):
        studio.inspect_local(source)
    other.unlink()
    (source / "quantization_config.json").write_text('{"quant_method":"exl3"}')
    assert not studio.inspect_local(source)["conversion_supported"]


def test_missing_tokenizer_is_visible(source):
    (source / "tokenizer.json").unlink()
    result = studio.inspect_local(source)
    assert not result["conversion_supported"] and "tokenizer" in result["reason"]


def test_worker_protocol_real_probe_and_invalid_input(source):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/quantization_studio.py")],
        input=json.dumps({"operation": "probe", "source": str(source)}),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0
    event = json.loads(result.stdout)
    assert event["type"] == "result" and event["result"]["header_checked"]
    for payload in ("{}", "{broken", "x" * 1_000_001):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/quantization_studio.py")],
            input=payload,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert result.returncode != 0 and json.loads(result.stdout)["type"] == "error"


def test_existing_output_never_modified(source):
    before = (source / "model.safetensors").read_bytes()
    with pytest.raises(ValueError, match="new output"):
        studio.convert(
            {
                "source": studio.inspect_local(source),
                "output": str(source),
                "settings": {},
                "jobs": str(source / "jobs"),
            }
        )
    assert (source / "model.safetensors").read_bytes() == before
    assert not (source / "jobs").exists()


def test_exclusions_are_exact_module_prefixes(monkeypatch):
    from mlxl3_quantizer.options import excluded_projection

    monkeypatch.setenv("MLXL3_QUANT_EXCLUDE_PREFIXES", '["model.layers.1","lm_head"]')
    assert excluded_projection("model.layers.1.mlp.up_proj") and excluded_projection("lm_head")
    assert not excluded_projection("model.layers.10.mlp.up_proj") and not excluded_projection(
        "model.lm_head"
    )
    monkeypatch.setenv("MLXL3_QUANT_EXCLUDE_PREFIXES", "[null]")
    with pytest.raises(ValueError):
        excluded_projection("anything")


def test_source_loader_forces_remote_code_off(source, tmp_path, monkeypatch):
    import importlib
    from types import SimpleNamespace

    import mlx_lm.utils

    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs["tokenizer_config"]["trust_remote_code"])
        raise RuntimeError("loader oracle reached")

    monkeypatch.setattr(mlx_lm.utils, "load", spy)
    monkeypatch.setattr(sys, "argv", ["test"])
    for name in ("MLXL3_QUANT_EXCLUDE_PREFIXES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.setenv(name, "")
    original = importlib.import_module

    def fake_adapter(name, *args):
        if name == "quantize_lfm26":
            return SimpleNamespace(
                main=lambda: mlx_lm.utils.load(
                    str(source), tokenizer_config={"trust_remote_code": True}
                )
            )
        return original(name, *args)

    monkeypatch.setattr(importlib, "import_module", fake_adapter)
    with pytest.raises(RuntimeError, match="oracle reached"):
        studio.convert(
            {
                "source": studio.inspect_local(source),
                "output": str(tmp_path / "new-output"),
                "settings": {},
                "jobs": str(tmp_path / "jobs"),
            }
        )
    assert calls == [False]
    record = json.loads(next((tmp_path / "jobs").glob("*/job.json")).read_text())
    assert record["status"] == "failed" and not (tmp_path / "new-output").exists()


def test_converter_index_accepts_single_file_without_mutating_source(source):
    from ponyexl3.convert.fixtures import SafetensorIndex

    before = {p.name: p.read_bytes() for p in source.iterdir()}
    index = SafetensorIndex(source)
    assert index.weight_map == {
        "model.layers.0.feed_forward.w1.weight": "model.safetensors",
        "scalar": "model.safetensors",
    }
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    save_file({"other": np.zeros((16, 16), np.float16)}, source / "extra.safetensors")
    with pytest.raises(ValueError, match="require a shard index"):
        SafetensorIndex(source)
