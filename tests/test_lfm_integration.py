from __future__ import annotations

import json
from pathlib import Path

import pytest
from mlx_lm.models.lfm2_moe import ModelArgs

from mlxl3.checkpoint import list_exl3_modules

MODEL_PATH = Path("models/fixtures/LFM2.5-8B-A1B-EXL3-3.10bpw")
pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(), reason="local LFM fixture not installed")


def test_lfm_checkpoint_shape_and_exl3_inventory() -> None:
    config = json.loads((MODEL_PATH / "config.json").read_text())
    args = ModelArgs.from_dict(config)
    modules = list_exl3_modules(MODEL_PATH)

    assert args.model_type == "lfm2_moe"
    assert args.num_hidden_layers == 24
    assert args.num_experts == 32
    assert args.num_experts_per_tok == 4
    assert len(modules) == 2179
    assert "lm_head" in modules
    assert "model.layers.2.feed_forward.experts.0.w1" in modules
