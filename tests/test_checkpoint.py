from __future__ import annotations

from pathlib import Path

import pytest

from mlxl3.checkpoint import list_exl3_modules, quantization_config

FIXTURE = Path("models/fixtures/Qwen3.5-0.8B-EXL3-3.0bpw")
pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="local EXL3 fixture not installed")


def test_fixture_is_standard_exl3_3bpw() -> None:
    config = quantization_config(FIXTURE)
    modules = list_exl3_modules(FIXTURE)
    assert config["bits"] == 3.0
    assert len(modules) == 151
    assert "model.language_model.layers.0.mlp.gate_proj" in modules
