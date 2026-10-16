from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from exl3_moe_calibration import (
    capture_weighted,
    check_coverage,
    expert_capture_params,
    expert_qmap,
    patch_converter,
)

pytest.importorskip("torch")


def fake_linear(key="model.layers.2.feed_forward.experts.0.w3", qmap="expert0"):
    import torch

    return SimpleNamespace(
        key=key,
        qmap=qmap,
        in_features=3,
        init_H_data=lambda _: {
            "H": torch.zeros((3, 3)),
            "first_key": key,
            "count": 0,
            "num_total": 0,
            "finalized": False,
        },
    )


@pytest.mark.parametrize("mode,power", [("routed", 0), ("affinity", 1), ("squared", 2)])
def test_hessian_matches_independent_outer_product_oracle(mode, power):
    import torch

    rng = np.random.default_rng(25)
    values = rng.normal(size=(32, 3)).astype(np.float32)
    weights = rng.uniform(0.01, 1, size=32).astype(np.float32)
    expected = sum(
        float(w) ** power * np.outer(row.astype(np.float64), row.astype(np.float64))
        for row, w in zip(values, weights, strict=True)
    )
    params = {"capture": {}, "mlxl3_moe_mode": mode}
    linear = fake_linear()
    for start in (0, 16):
        p = expert_capture_params(params, torch.from_numpy(weights[start : start + 16, None]))
        capture_weighted(linear, torch.from_numpy(values[start : start + 16]), p)
    result = params["capture"]["expert0"]
    np.testing.assert_allclose(result["H"].numpy(), expected, rtol=2e-6, atol=2e-6)
    assert result["count"] == 32 and result["num_total"] == 96
    assert check_coverage(result["count"], result["affinity_mass"], result["affinity_mass2"], 1) > 1
    assert "mlxl3_expert_weights" not in params
    # A different weight/row association is not the same objective.
    if power:
        bad = sum(
            float(w) ** power * np.outer(row, row)
            for row, w in zip(values, weights[::-1], strict=True)
        )
        assert not np.allclose(expected, bad, rtol=1e-3, atol=1e-3)


def test_gate_up_share_only_their_own_expert_and_inference_params_are_unchanged():
    import torch

    capture = {}
    for expert, values in enumerate(([[1, 0, 0], [0, 2, 0]], [[0, 0, 7], [0, 0, 9]])):
        key = expert_qmap("block", expert, "up")
        assert key == expert_qmap("block", expert, "gate")
        assert key != expert_qmap("block", expert, "down")
        params = {
            "capture": capture,
            "mlxl3_moe_mode": "affinity",
            "mlxl3_expert_weights": torch.tensor([0.2, 0.8]),
        }
        up = fake_linear(f"block.experts.{expert}.w3", key)
        capture_weighted(up, torch.tensor(values), params)
        gate = fake_linear(f"block.experts.{expert}.w1", key)
        capture_weighted(gate, torch.tensor(values), params)
        assert capture[key]["count"] == 2
    assert len(capture) == 2
    assert capture[expert_qmap("block", 0, "up")]["H"][2, 2] == 0
    assert capture[expert_qmap("block", 1, "up")]["H"][2, 2] > 0
    inference = {"cache": object()}
    assert expert_capture_params(inference, torch.ones(2)) is inference
    legacy = {"capture": {}}
    assert expert_capture_params(legacy, torch.ones(2)) is legacy


@pytest.mark.parametrize(
    "values,weights,mode",
    [
        ([[1, 2, 3]], [0], "affinity"),
        ([[1, 2, 3]], [-1], "affinity"),
        ([[1, 2, 3]], [float("nan")], "affinity"),
        ([[1, 2, 3]], [float("inf")], "affinity"),
        ([[1, 2, float("nan")]], [1], "affinity"),
        ([[1, 2, float("inf")]], [1], "affinity"),
        ([[1, 2, 3]], [1, 2], "affinity"),
        ([[1, 2]], [1], "affinity"),
        ([[1, 2, 3]], [[1]], "affinity"),
        ([[1, 2, 3]], [1], "unknown"),
        ([[1, 2, 3]], [1e30], "squared"),
        ([[1, 2, 3]], [1e-40], "squared"),
    ],
)
def test_invalid_affinities_do_not_create_a_hessian(values, weights, mode):
    import torch

    params = {
        "capture": {},
        "mlxl3_moe_mode": mode,
        "mlxl3_expert_weights": torch.tensor(weights),
    }
    with pytest.raises(ValueError):
        capture_weighted(fake_linear(), torch.tensor(values), params)
    assert params["capture"] == {}


@pytest.mark.parametrize(
    "count,mass,mass2,minimum",
    [
        (0, 0, 0, 1),
        (255, 255, 255, 256),
        (256, 1, 1, 256),
        (256, 1, 0, 1),
        (256, float("nan"), 1, 1),
        (256, 1, float("inf"), 1),
        (256, 1, 1, 0),
    ],
)
def test_cold_or_concentrated_experts_fail_closed(count, mass, mass2, minimum):
    with pytest.raises(ValueError):
        check_coverage(count, mass, mass2, minimum)


def test_uniform_coverage_boundary_and_joint_permutation():
    import torch

    assert check_coverage(256, 256, 256, 256) == 256
    rows = torch.tensor([[2, 4, 1], [3, 9, 8], [1, 0, 7]], dtype=torch.float32)
    weights = torch.tensor([0.1, 0.3, 0.9])
    results = []
    for order in ([0, 1, 2], [2, 0, 1]):
        params = {
            "capture": {},
            "mlxl3_moe_mode": "affinity",
            "mlxl3_expert_weights": weights[order],
        }
        capture_weighted(fake_linear(), rows[order], params)
        results.append(params["capture"]["expert0"]["H"])
    torch.testing.assert_close(*results)


def test_converter_patch_is_hash_guarded_and_keeps_each_projection_wired(tmp_path, monkeypatch):
    import exl3_moe_calibration as module

    path = tmp_path / "modules/block_sparse_mlp.py"
    path.parent.mkdir()
    text = (
        "from typing_extensions import override\n"
        "def outer(self, params, w):\n"
        "                        def mlp(exp_i, xc):\n"
        "                            u = self.ups[exp_i].forward(xc, params)\n"
        "                            g = self.gates[exp_i].forward(xc, params)\n"
        "                            a = u * g\n"
        "                            return self.downs[exp_i].forward(a, params)\n"
    )
    path.write_text(text)
    with pytest.raises(ValueError, match="unmodified"):
        patch_converter(tmp_path)
    assert path.read_text() == text
    monkeypatch.setattr(module, "BLOCK_SHA", hashlib.sha256(text.encode()).hexdigest())
    digest = patch_converter(tmp_path)
    changed = path.read_text()
    assert hashlib.sha256(changed.encode()).hexdigest() == digest
    assert changed.count("forward(xc, expert_params)") == 2
    assert "forward(a, expert_params)" in changed
    with pytest.raises(ValueError):
        patch_converter(tmp_path)


@pytest.mark.parametrize("expert,projection", [(-1, "up"), (0, "router")])
def test_invalid_expert_map(expert, projection):
    with pytest.raises(ValueError):
        expert_qmap("block", expert, projection)


def test_all_actual_lfm25_maps_are_isolated_exhaustively():
    maps = set()
    for layer in range(2, 40):
        for expert in range(64):
            block = f"model.layers.{layer}.feed_forward"
            up = expert_qmap(block, expert, "up")
            assert up == expert_qmap(block, expert, "gate")
            down = expert_qmap(block, expert, "down")
            assert up != down and up not in maps and down not in maps
            maps.update((up, down))
    assert len(maps) == 4864


def test_installed_hooks_isolate_experts_keep_inference_and_reject_fallback(monkeypatch, tmp_path):
    import json
    import sys
    from types import ModuleType

    import torch
    from exl3_moe_calibration import install

    class Linear:
        def __init__(self, key):
            self.__dict__.update(vars(fake_linear(key)))

        def capture_H(self, x, params):
            raise AssertionError("Legacy expert capture was selected")

    class BlockSparseMLP:
        def __init__(self):
            self.key = "block"
            self.ups = [Linear(f"block.experts.{i}.w3") for i in range(2)]
            self.gates = [Linear(f"block.experts.{i}.w1") for i in range(2)]
            self.downs = [Linear(f"block.experts.{i}.w2") for i in range(2)]

        def forward(self, x, params):
            if "capture" in params:
                assert params["activate_all_experts"] is False
                for up, gate, down in zip(self.ups, self.gates, self.downs, strict=True):
                    p = expert_capture_params(params, torch.tensor([0.25, 0.75]))
                    for linear in (up, gate, down):
                        linear.capture_H(x, p)
            return x

    quantize = ModuleType("exllamav3.modules.quant.exl3_lib.quantize")
    fallback = False

    def finalize(data, *args):
        data["finalized"] = True
        return (fallback, data["H"])

    quantize.finalize_capture_H = finalize
    names = [
        "exllamav3",
        "exllamav3.modules",
        "exllamav3.modules.block_sparse_mlp",
        "exllamav3.modules.linear",
        "exllamav3.modules.quant",
        "exllamav3.modules.quant.exl3_lib",
    ]
    for name in names:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules[names[2]].BlockSparseMLP = BlockSparseMLP
    sys.modules[names[3]].Linear = Linear
    sys.modules[names[5]].quantize = quantize
    monkeypatch.setenv("MLXL3_MOE_COVERAGE", str(tmp_path / "coverage.jsonl"))
    install("affinity", 1)
    block = BlockSparseMLP()
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    assert block.forward(x, {}) is x
    params = {"capture": {}, "activate_all_experts": True}
    assert block.forward(x, params) is x
    assert params["activate_all_experts"] is True
    assert len(params["capture"]) == 4
    for data in params["capture"].values():
        assert data["count"] == 2
        quantize.finalize_capture_H(data, {}, False)
    assert len((tmp_path / "coverage.jsonl").read_text().splitlines()) == 4
    assert (
        json.loads((tmp_path / "coverage.jsonl").read_text().splitlines()[0])["mode"] == "affinity"
    )
    fallback = True
    with pytest.raises(ValueError, match="fallback"):
        quantize.finalize_capture_H(next(iter(params["capture"].values())), {}, False)
    with pytest.raises(RuntimeError, match="already"):
        install("affinity", 1)
