from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlx_lm.models.bailing_moe_v3 import ModelArgs

MODEL_PATH = Path("models/source/Ling-3.0-tiny-HF")
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="local Ling model not installed")


def test_ling_config_is_supported_by_current_mlx_lm() -> None:
    config = json.loads((MODEL_PATH / "config.json").read_text())
    args = ModelArgs.from_dict(config)
    assert args.model_type == "bailing_hybrid"
    assert args.num_hidden_layers == 24
    assert args.layer_group_size == 4
    assert args.num_experts == 128
    assert args.num_experts_per_tok == 8
