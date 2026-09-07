import runpy
from pathlib import Path

import pytest


def test_lfm_projection_selection_is_narrow():
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/quantize_lfm26.py"))
    accepts = module["is_lfm_projection"]
    for suffix in (
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "conv.in_proj",
        "conv.out_proj",
    ):
        assert accepts("model.layers.12." + suffix)
    for suffix in (
        "conv.conv",
        "ffn_norm",
        "operator_norm",
        "feed_forward.router",
        "feed_forward.w4",
    ):
        assert not accepts("model.layers.0." + suffix)
    assert not accepts("model.embed_tokens")
    assert not accepts("other.layers.0.conv.in_proj")


def test_export_preserves_runtime_codebook_markers(monkeypatch):
    from types import SimpleNamespace

    import numpy as np

    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/quantize_lfm26.py"))
    pytest.importorskip("ponyexl3.convert")
    from ponyexl3.convert import capture, direct, discovery, driver, measure, reuse

    for owner, name in (
        (direct, "_layer_tensors"),
        (discovery, "_is_exl3_linear_key"),
        (driver, "_sibling_group_signature"),
        (measure, "_sibling_group_signature"),
        (capture, "capture_calibration_activations"),
        (reuse, "enable"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    module["patch_converter"]()
    for mode, multiplier in (("mcg", 0xCBAC1FED), ("mul1", 0x83DCD12D), ("default", None)):
        layer = SimpleNamespace(
            key="model.layers.0.conv.in_proj",
            trellis=np.zeros((8, 8, 64), dtype=np.uint16),
            suh=None,
            svh=None,
            bias=None,
            mcg=mode == "mcg",
            mul1=mode == "mul1",
        )
        tensors = direct._layer_tensors(layer)
        markers = {key.rsplit(".", 1)[-1] for key in tensors if key.endswith((".mcg", ".mul1"))}
        assert markers == ({mode} if multiplier is not None else set())
        if multiplier is not None:
            assert tensors[layer.key + "." + mode].dtype == np.int32
            assert int(tensors[layer.key + "." + mode].view(np.uint32)) == multiplier
