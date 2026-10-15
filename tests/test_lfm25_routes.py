from __future__ import annotations

import copy
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from diagnose_lfm25_routes import (
    V2_SHARDS,
    WINDOW_HASHES,
    RouteCapture,
    artifact,
    checked_position,
    compare_logits,
    cuda_selected_weights,
    decode_window,
    main,
    route_comparison,
    run,
    validate_plan,
    validate_routes,
)

torch = pytest.importorskip("torch")


def plan():
    return {
        "windows": {name: list(range(1024)) for name in ("window1", "window5")},
        "shards": V2_SHARDS.copy(),
    }


def routes(steps=3, key="layer", offset=0):
    return {
        key: (
            torch.tensor([[offset + n, 10, 20, 30] for n in range(steps)], dtype=torch.long),
            torch.tensor([[0.125, 0.25, 0.125, 0.5]] * steps, dtype=torch.float16),
        )
    }


def module():
    ids = torch.empty((1, 4), dtype=torch.long)
    weights = torch.empty((1, 4), dtype=torch.float16)

    def routing(bsz, cfg, hidden, params):
        assert bsz == 1
        ids.copy_(torch.tensor([[int(hidden.item()), 10, 20, 30]]))
        weights.copy_(torch.tensor([[0.125, 0.25, 0.125, 0.5]], dtype=torch.float16))
        return ids, weights

    return SimpleNamespace(device="cpu", routing_fn=routing, ids=ids, weights=weights)


@pytest.fixture(autouse=True)
def synthetic_windows(monkeypatch):
    import diagnose_lfm25_routes as target

    digest = hashlib.sha256(struct.pack("<1024I", *range(1024))).hexdigest()
    monkeypatch.setattr(target, "WINDOW_HASHES", {"window1": digest, "window5": digest})


def test_historical_window_hashes_remain_frozen():
    assert WINDOW_HASHES == {
        "window1": "bdafc8c2a7c8ef0322c186013d1b99f7bff96c45fc91f47e9125e9c98fd62061",
        "window5": "23b146a0a8b43a1b0495f9fc9bd6882e7ed628be7cd7c7aa4751082dade7f7a4",
    }


def test_frozen_plan_and_positive_index_boundaries():
    assert validate_plan(plan())["windows"]["window1"][0] == 0
    for total in range(1, 25):
        for index in range(total):
            assert checked_position(index, total) == index
        for bad in (-1, total):
            with pytest.raises(ValueError):
                checked_position(bad, total)


@pytest.mark.parametrize(
    "failure",
    [
        "extra",
        "window",
        "short",
        "bool",
        "negative",
        "padding",
        "shard",
        "sha",
        "changed_tokens",
        "changed_shards",
    ],
)
def test_invalid_frozen_plan_is_rejected(failure):
    data = plan()
    if failure == "extra":
        data["extra"] = True
    elif failure == "window":
        data["windows"]["wrong"] = data["windows"].pop("window1")
    elif failure == "short":
        data["windows"]["window1"].pop()
    elif failure in ("bool", "negative", "padding"):
        data["windows"]["window1"][0] = {"bool": True, "negative": -1, "padding": 125017}[failure]
    elif failure == "changed_tokens":
        data["windows"]["window1"][0] = 8
    elif failure == "changed_shards":
        data["shards"][next(iter(data["shards"]))] = "0" * 64
    elif failure == "shard":
        data["shards"]["../wrong.safetensors"] = data["shards"].pop(next(iter(data["shards"])))
    else:
        data["shards"][next(iter(data["shards"]))] = "z" * 64
    with pytest.raises(ValueError):
        validate_plan(data)


@pytest.mark.parametrize(
    "failure",
    [
        "empty",
        "missing",
        "shape",
        "ids_dtype",
        "weights_dtype",
        "nan",
        "inf",
        "zero",
        "negative",
        "range",
        "duplicate",
        "mass",
    ],
)
def test_route_validation_refuses_invalid_or_partial_payload(failure):
    data = routes()
    ids, weights = data["layer"]
    if failure == "empty":
        data = {}
    elif failure == "missing":
        data["other"] = data.pop("layer")
    elif failure == "shape":
        data["layer"] = ids[:2], weights[:2]
    elif failure == "ids_dtype":
        data["layer"] = ids.int(), weights
    elif failure == "weights_dtype":
        data["layer"] = ids, weights.float()
    elif failure in ("nan", "inf", "zero", "negative", "mass"):
        weights[0, 0] = {
            "nan": float("nan"),
            "inf": float("inf"),
            "zero": 0,
            "negative": -1,
            "mass": 1,
        }[failure]
    elif failure == "range":
        ids[0, 0] = 64
    else:
        ids[0, 0] = ids[0, 1]
    with pytest.raises(ValueError):
        validate_routes(data, ("layer",), 3)


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_actual_buffers_are_recorded_or_forced_and_hooks_always_restore(replay, fail):
    layer = module()
    original = layer.routing_fn
    data = routes(offset=4) if replay else None
    capture = RouteCapture({"layer": layer}, 3, data)
    try:
        with capture:
            for n in range(3):
                capture.begin(n)
                ids, weights = layer.routing_fn(1, None, torch.tensor(n), {})
                assert ids.data_ptr() == layer.ids.data_ptr()
                assert weights.data_ptr() == layer.weights.data_ptr()
                assert ids[0, 0].item() == n + (4 if replay else 0)
                capture.finish()
                if fail and n == 1:
                    raise RuntimeError("injected forward interruption")
            result = capture.payload()
            for actual, expected in zip(
                result["layer"], routes(offset=4 if replay else 0)["layer"]
            ):
                assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    except RuntimeError:
        assert fail
        with pytest.raises(ValueError, match="incomplete"):
            capture.payload()
    assert layer.routing_fn is original


@pytest.mark.parametrize(
    "failure",
    ["out_of_range", "skipped", "nested", "missing", "duplicate", "outside", "batch", "dtype"],
)
def test_call_alignment_cannot_silently_capture_wrong_layers_or_tokens(failure):
    layer = module()
    original = layer.routing_fn
    with RouteCapture({"layer": layer}, 3) as capture, pytest.raises(ValueError):
        if failure == "out_of_range":
            capture.begin(3)
        elif failure == "skipped":
            capture.begin(1)
        elif failure == "outside":
            layer.routing_fn(1, None, torch.tensor(0), {})
        else:
            capture.begin(0)
            if failure == "nested":
                capture.begin(0)
            elif failure == "missing":
                capture.finish()
            elif failure == "batch":
                layer.routing_fn(2, None, torch.tensor(0), {})
            elif failure == "dtype":
                layer.routing_fn.__kwdefaults__["original"] = lambda *_: (
                    torch.zeros(1, 4),
                    torch.zeros(1, 4),
                )
                layer.routing_fn(1, None, torch.tensor(0), {})
            else:
                layer.routing_fn(1, None, torch.tensor(0), {})
                layer.routing_fn(1, None, torch.tensor(0), {})
    assert layer.routing_fn is original


def test_ids_only_calls_official_normalizer_and_preserves_weight_allocations(monkeypatch):
    import diagnose_lfm25_routes as target

    layer, seen = module(), []

    def normalize(cfg, hidden, ids, weights):
        seen.append((int(hidden.item()), ids.clone()))
        weights.copy_(torch.tensor([[0.5, 0.125, 0.25, 0.125]], dtype=torch.float16))

    monkeypatch.setattr(target, "cuda_selected_weights", normalize)
    with RouteCapture({"layer": layer}, 3, routes(offset=4), ids_only=True) as capture:
        for n in range(3):
            capture.begin(n)
            ids, weights = layer.routing_fn(1, None, torch.tensor(n), {})
            assert ids[0, 0] == n + 4 and weights[0, 0] == 0.5
            assert weights.data_ptr() == layer.weights.data_ptr()
            capture.finish()
        assert capture.payload()["layer"][1][0, 0] == 0.5
    assert len(seen) == 3


def test_cuda_normalizer_uses_exact_binding_order_without_reimplementing_math(monkeypatch):
    import sys

    cfg = SimpleNamespace(
        **{
            k: object()
            for k in (
                "gate_tensor",
                "router_logits_bsz1",
                "routed_scaling_factor",
                "gate_tensor_t",
                "gate_i8",
                "gate_sb",
            )
        }
    )
    hidden, ids, weights, calls = object(), object(), object(), []
    monkeypatch.setitem(
        sys.modules,
        "exllamav3.ext",
        SimpleNamespace(
            exllamav3_ext=SimpleNamespace(routing_sel_norm=lambda *args: calls.append(args))
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "exllamav3.modules.block_sparse_mlp_routing",
        SimpleNamespace(ROUTING_ACT_SIGMOID=0, _gate_t=lambda actual: calls.append(actual)),
    )
    cuda_selected_weights(cfg, hidden, ids, weights)
    assert calls == [
        cfg,
        (
            hidden,
            cfg.gate_tensor,
            cfg.router_logits_bsz1,
            ids,
            weights,
            cfg.routed_scaling_factor,
            cfg.gate_tensor_t,
            0,
            cfg.gate_i8,
            cfg.gate_sb,
        ),
    ]


def test_route_comparison_checks_sets_order_and_payload():
    data, other = routes(), copy.deepcopy(routes())
    other["layer"][0][0] = other["layer"][0][0].flip(0)
    r = route_comparison(data, other, 3)["layer"]
    assert r["changed_top4_sets"] == 0 and r["same_order"] == 2 and not r["bitexact"]
    other["layer"][0][1, 0] = 5
    assert route_comparison(data, other, 3)["layer"]["changed_top4_sets"] == 1


def test_saved_logit_kl_has_independent_closed_form_and_argmax_oracle(tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.log(np.array([[0.75, 0.25], [0.75, 0.25]], dtype=np.float64)))
    np.save(b, np.log(np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)))
    result = compare_logits(a, b)
    assert result["tokens"] == result["greedy_matches"] == 2
    assert result["kl"] == pytest.approx(0.75 * np.log(1.5) + 0.25 * np.log(0.5), abs=1e-14)
    assert compare_logits(a, a)["kl"] == 0


@pytest.mark.parametrize(
    "bad", [np.empty((0, 2)), np.array([0, 1]), np.array([[np.nan, 0]]), np.zeros((2, 3))]
)
def test_logit_comparison_refuses_empty_changed_or_nonfinite_data(tmp_path, bad):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.zeros((1, 2)))
    np.save(b, bad)
    with pytest.raises(ValueError):
        compare_logits(a, b)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.bfloat16])
def test_production_decode_boundary_saves_routes_logits_and_past_lengths(tmp_path, dtype):
    layer, lengths = module(), []

    def forward(tokens, params):
        lengths.append(params["past_len"])
        assert params["attn_mode"] == "flash_attn" and params["batch_shape"] == (1, 3)
        ids, weights = layer.routing_fn(1, None, tokens.flatten()[0], params)
        return torch.full((1, 1, 128000), float(ids[0, 0]) + float(weights[0, 0]), dtype=dtype)

    path = tmp_path / "source-window1.npy"
    payload, report = decode_window(
        SimpleNamespace(forward=forward), object(), [0, 1, 2], {"layer": layer}, path
    )
    values = np.load(path, allow_pickle=False)
    assert values.shape == (3, 128000) and np.isfinite(values).all()
    assert lengths == [0, 1, 2] and report["routes"]["layers"] == 1
    assert report["compute_dtypes"] == [str(dtype)]
    assert np.array_equal(values[:, 0], np.array([0.125, 1.125, 2.125], dtype=np.float16))
    assert payload["layer"][0][2, 0] == 2
    assert artifact(path)["sha256"] == report["sha256"]
    with pytest.raises(FileExistsError):
        decode_window(SimpleNamespace(forward=forward), object(), [0, 1, 2], {"layer": layer}, path)


@pytest.mark.parametrize("failure", ["interrupt", "shape", "nonfinite"])
def test_forward_error_restores_hooks_and_never_exports_complete_routes(tmp_path, failure):
    layer = module()
    original = layer.routing_fn

    def forward(tokens, params):
        layer.routing_fn(1, None, tokens.flatten()[0], params)
        if failure == "interrupt":
            raise RuntimeError("injected failure")
        if failure == "shape":
            return torch.zeros(128, dtype=torch.float16)
        return torch.full((128000,), float("inf"), dtype=torch.float16)

    path = tmp_path / "bad.npy"
    with pytest.raises((RuntimeError, ValueError)):
        decode_window(SimpleNamespace(forward=forward), object(), [0, 1, 2], {"layer": layer}, path)
    assert layer.routing_fn is original
    assert not path.with_suffix(".routes.safetensors").exists()


def test_production_entrypoint_marks_failure_and_preserves_prior_attempts(tmp_path):
    path, output = tmp_path / "plan.json", tmp_path / "output"
    path.write_text(json.dumps({"invalid": True}))
    with pytest.raises(ValueError):
        main([str(path), str(output)])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and not report["causal_interpretation"]
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        run(plan(), output)
    assert (output / "report.json").read_bytes() == before


def complete_result():
    record = {
        "shape": [1024, 128000],
        "dtype": "float16",
        "bytes": 262144128,
        "sha256": "0" * 64,
        "routes": {"bytes": 100, "layers": 38, "tokens_per_layer": 1024, "top_k": 4},
        "replay_bitexact": True,
    }
    return {
        "status": "complete",
        "ids_only": "not_executed_binding_absent",
        "arms": {
            arm: {name: copy.deepcopy(record) for name in ("window1", "window5")}
            for arm in ("source", "normal", "own_routes", "source_routes")
        },
    }


@pytest.mark.parametrize(
    "failure", [None, "timeout", "error", "incomplete", "invalid_json", "partial_complete"]
)
def test_modal_real_child_boundary_reaps_and_marks_failures(tmp_path, monkeypatch, failure):
    import diagnose_lfm25_routes_modal as target

    root = tmp_path / "diagnostic36-routes"
    commits, commands = [], []
    monkeypatch.setattr(target, "DATA", root)
    monkeypatch.setattr(target, "volume", SimpleNamespace(commit=lambda: commits.append(True)))

    def child(command, log, timeout):
        commands.append(command)
        assert timeout == 900 and command[2] == "/opt/mlxl3/diagnose_lfm25_routes.py"
        assert json.loads(Path(command[3]).read_text()) == plan()
        if failure in ("timeout", "error"):
            raise (
                TimeoutError("child limit")
                if failure == "timeout"
                else RuntimeError("child nonzero")
            )
        root.mkdir()
        if failure == "invalid_json":
            (root / "report.json").write_text("{truncated")
        else:
            data = complete_result()
            if failure == "partial_complete":
                data["arms"].pop("source")
            if failure == "incomplete":
                data["status"] = "running"
            (root / "report.json").write_text(json.dumps(data))

    monkeypatch.setattr(target.base, "run_logged", child)
    if failure:
        with pytest.raises((TimeoutError, RuntimeError, ValueError)):
            target.execute(plan())
        data = json.loads((root / "report.json").read_text())
        assert data["status"] == "failed" and not data["causal_interpretation"]
    else:
        assert target.execute(plan())["status"] == "complete"
    assert commits == [True] and len(commands) == 1
    with pytest.raises(FileExistsError):
        target.execute(plan())
    assert len(commands) == 1


@pytest.mark.parametrize(
    "failure", [None, "own_replay", "ids_control", "wrong_imposition", "not_fused"]
)
def test_full_orchestrator_uses_fresh_caches_and_one_loaded_model(tmp_path, monkeypatch, failure):
    import importlib.metadata
    import sys

    import diagnose_lfm25_routes as target
    import quantize_lfm25_modal as base

    folder = tmp_path / "data"
    source = folder / "hf/hub/models--LiquidAI--LFM2.5-24B-A2B/snapshots" / target.SOURCE_REVISION
    source.mkdir(parents=True)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {str(i): name for i, name in enumerate(target.SOURCE_SHARDS)}})
    )
    live, events, calls = [], [], []

    class Config:
        @staticmethod
        def from_directory(path):
            return SimpleNamespace(
                kind="v2" if path.endswith("/v2/output") else "source",
                infer_params=SimpleNamespace(no_reconstruct=False),
            )

    class Model:
        def __init__(self, config):
            self.config, self.caches = config, []
            self.modules = [
                SimpleNamespace(
                    mlp=SimpleNamespace(
                        key=key,
                        is_quantized=config.kind == "v2",
                        bc=object() if config.kind == "v2" and failure != "not_fused" else None,
                    )
                )
                for key in target.LAYERS
            ]

        @staticmethod
        def from_config(config):
            assert not live
            return Model(config)

        def load(self, **kwargs):
            assert not live and len(self.caches) == (2 if self.config.kind == "source" else 10)
            assert kwargs == {"device": "cuda:0", "max_output_size": 1}
            live.append(self)
            events.append("load-" + self.config.kind)

        def unload(self):
            assert live == [self]
            live.clear()
            events.append("unload-" + self.config.kind)

    def cache(model, **kwargs):
        assert not live and kwargs == {"max_num_tokens": 1024, "max_batch_size": 1}
        result = object()
        model.caches.append(result)
        return result

    def fake_artifact(path):
        sha = target.SOURCE_SHARDS.get(path.name, target.V2_SHARDS.get(path.name, "0" * 64))
        return {"path": str(path), "bytes": 100, "sha256": sha}

    def decode(model, actual_cache, tokens, modules, path, replay, ids_only):
        assert live == [model] and actual_cache in model.caches
        assert len(tokens) == 1024 and tuple(modules) == target.LAYERS
        arm = path.name.split("-")[0]
        calls.append(arm)
        if replay is None:
            offset = 1 if arm == "source" else 2
            payload = {key: routes(steps=1, offset=offset)["layer"] for key in modules}
        else:
            payload = replay
        if failure == "wrong_imposition" and arm == "source_routes":
            payload = {key: routes(steps=1, offset=2)["layer"] for key in modules}
        digest = "normal" if arm in ("normal", "own_routes") else arm
        if failure == "own_replay" and arm == "own_routes":
            digest = "changed"
        return payload, {"path": str(path), "sha256": digest}

    # One-row deterministic tensors keep this orchestration test small; the real trace boundary
    # is exercised separately with actual tensor buffers and full-vocabulary saved logits.
    real_comparison = target.route_comparison
    monkeypatch.setattr(target, "route_comparison", lambda a, b, _: real_comparison(a, b, 1))
    monkeypatch.setattr(target, "decode_window", decode)
    monkeypatch.setattr(target, "artifact", fake_artifact)
    monkeypatch.setattr(
        target,
        "compare_logits",
        lambda *_: {
            "tokens": 1024,
            "kl": 1.0 if failure == "ids_control" else 0.0,
            "greedy_matches": 1024,
        },
    )
    monkeypatch.setattr(base, "check_converter_sources", lambda _: {"pinned": True})
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda _: SimpleNamespace(locate_file=lambda _: tmp_path),
    )
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "test-only")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU fake")
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setitem(
        sys.modules, "exllamav3", SimpleNamespace(Cache=cache, Config=Config, Model=Model)
    )
    monkeypatch.setitem(
        sys.modules,
        "exllamav3.ext",
        SimpleNamespace(exllamav3_ext=SimpleNamespace(routing_sel_norm=lambda *_: None)),
    )
    output = tmp_path / "output"
    if failure in ("own_replay", "wrong_imposition", "not_fused"):
        with pytest.raises(ValueError):
            target.run(plan(), output, folder)
        assert json.loads((output / "report.json").read_text())["status"] == "failed"
    else:
        result = target.run(plan(), output, folder)
        assert result["status"] == "complete" and not result["causal_interpretation"]
        assert result["ids_only"] == ("not_executed_control_failed" if failure else "approved")
        assert ("source_ids" in result["arms"]) == (failure is None)
        assert all(
            result["arms"]["own_routes"][name]["replay_bitexact"] for name in ("window1", "window5")
        )
    assert not live
    assert events == ["load-source", "unload-source", "load-v2", "unload-v2"]
    if failure is None:
        assert len(calls) == 12 and calls[:2] == ["source", "source"]


@pytest.mark.parametrize(
    "failure",
    [
        "arm",
        "window",
        "shape",
        "zero_bytes",
        "layer",
        "replay",
        "ids_nan",
        "ids_tokens",
        "ids_missing",
        "ids_passed",
    ],
)
def test_modal_report_requires_complete_positive_evidence(failure):
    import diagnose_lfm25_routes_modal as target

    data = complete_result()
    record = data["arms"]["own_routes"]["window1"]
    if failure == "arm":
        data["arms"].pop("source")
    elif failure == "window":
        data["arms"]["source"].pop("window1")
    elif failure == "shape":
        record["shape"] = [0, 128000]
    elif failure == "zero_bytes":
        record["bytes"] = 0
    elif failure == "layer":
        record["routes"]["layers"] = 37
    elif failure == "replay":
        record["replay_bitexact"] = False
    else:
        data["ids_only"] = "approved"
        data["arms"]["own_ids"] = copy.deepcopy(data["arms"]["normal"])
        data["arms"]["source_ids"] = copy.deepcopy(data["arms"]["normal"])
        for item in data["arms"]["own_ids"].values():
            item["normal_comparison"] = {"kl": 0.0, "tokens": 1024, "greedy_matches": 1024}
        if failure == "ids_nan":
            data["arms"]["own_ids"]["window1"]["normal_comparison"]["kl"] = float("nan")
        elif failure == "ids_tokens":
            data["arms"]["own_ids"]["window1"]["normal_comparison"]["tokens"] = 0
        elif failure == "ids_missing":
            data["arms"].pop("source_ids")
    if failure == "ids_passed":
        assert target.validate_result(data) is data
    else:
        with pytest.raises(ValueError):
            target.validate_result(data)


def test_route_diagnostic_uses_the_existing_v3_volume_without_creating_another():
    import diagnose_lfm25_routes_modal as target

    assert target.VOLUME == "mlxl3-lfm25-v3-20261015"
