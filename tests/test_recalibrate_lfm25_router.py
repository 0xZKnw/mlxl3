from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import recalibrate_lfm25_router as target

torch = pytest.importorskip("torch")
_tensor_files = importlib.import_module("safetensors.torch")
load_file, save_file = _tensor_files.load_file, _tensor_files.save_file


def captures():
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float16)
    source = torch.tensor([[3, 1], [2, 3], [5, 4], [2, 3]], dtype=torch.bfloat16)
    return x, torch.zeros((4, 2), dtype=torch.float16), source, torch.zeros(2)


def test_scalar_normal_equations_have_hand_computed_answer_and_do_not_change_inputs():
    values = captures()
    before = [value.clone() for value in values]
    delta, stats = target.fit_delta(*values, min_samples=2, top_k=1)
    assert delta.dtype == torch.float64 and delta.shape == (2, 1)
    torch.testing.assert_close(delta[:, 0], torch.tensor([18 / 11, 9 / 11], dtype=torch.float64))
    assert stats["fitted_experts"] == 2 and not stats["quality_validation"]
    assert [row["samples"] for row in stats["per_expert"]] == [2, 2]
    assert [row["ridge_lambda"] for row in stats["per_expert"]] == [0.5, 1.0]
    assert all(torch.equal(a, b) for a, b in zip(values, before))


def test_fit_matches_independent_augmented_least_squares_across_seeded_cases():
    rng = np.random.default_rng(38)
    for _ in range(5):
        x = rng.normal(size=(17, 3))
        zq, zs = rng.normal(size=(2, 17, 4))
        bias = rng.normal(size=4) * 0.2
        delta, stats = target.fit_delta(
            *[torch.tensor(v, dtype=torch.float64) for v in (x, zq, zs, bias)],
            min_samples=2,
            top_k=2,
        )
        for expert in range(4):
            chosen = [
                i
                for i, row in enumerate(zs)
                if expert
                in sorted(range(4), key=lambda e: (-(1 / (1 + math.exp(-row[e])) + bias[e]), e))[:2]
            ]
            assert stats["per_expert"][expert]["samples"] == len(chosen)
            xe, residual = x[chosen], (zs - zq)[chosen, expert]
            if len(chosen) < 2:
                expected = np.zeros(3)
            else:
                ridge = 0.1 * np.square(xe).sum() / len(chosen) / 3
                a = np.concatenate((xe / math.sqrt(len(chosen)), math.sqrt(ridge) * np.eye(3)))
                b = np.concatenate((residual / math.sqrt(len(chosen)), np.zeros(3)))
                expected = np.linalg.lstsq(a, b, rcond=None)[0]
            np.testing.assert_allclose(delta[expert].numpy(), expected, rtol=1e-11, atol=1e-11)


def test_sigmoid_selection_bias_changes_mask_instead_of_ranking_raw_logits():
    x = torch.ones((3, 1), dtype=torch.float16)
    source = torch.tensor([[10, 2]] * 3, dtype=torch.bfloat16)
    delta, stats = target.fit_delta(
        x,
        torch.zeros((3, 2)),
        source,
        torch.tensor([0.0, 1.0]),
        min_samples=3,
        top_k=1,
    )
    assert [row["samples"] for row in stats["per_expert"]] == [0, 3]
    assert delta[0].item() == 0 and delta[1].item() == pytest.approx(2 / 1.1)


def test_equal_selection_scores_break_ties_by_expert_index():
    delta, stats = target.fit_delta(
        torch.ones((2, 1)),
        -torch.ones((2, 2)),
        torch.zeros((2, 2)),
        torch.zeros(2),
        min_samples=2,
        top_k=1,
    )
    assert [row["samples"] for row in stats["per_expert"]] == [2, 0]
    assert delta[0].item() == pytest.approx(1 / 1.1) and delta[1].item() == 0


def test_cold_and_zero_energy_experts_are_explicitly_frozen():
    x, zq, source, bias = captures()
    cold, stats = target.fit_delta(x, zq, source, bias, min_samples=3, top_k=1)
    assert torch.count_nonzero(cold).item() == 0
    assert [r["status"] for r in stats["per_expert"]] == ["cold", "cold"]
    zero, stats = target.fit_delta(x * 0, zq, source, bias, min_samples=2, top_k=1)
    assert torch.count_nonzero(zero).item() == 0
    assert [r["status"] for r in stats["per_expert"]] == ["zero_energy", "zero_energy"]


def test_identity_teacher_and_rank_deficient_inputs_are_well_defined():
    x, zq, source, bias = captures()
    delta, _ = target.fit_delta(x.expand(-1, 3), source, source, bias, min_samples=2, top_k=1)
    assert torch.count_nonzero(delta).item() == 0
    delta, _ = target.fit_delta(x.expand(-1, 3), zq, source, bias, min_samples=2, top_k=1)
    assert torch.isfinite(delta).all() and delta.shape == (2, 3)


@pytest.mark.parametrize(
    "change", ["empty", "rank", "rows", "source", "bias", "integer", "nan", "inf"]
)
def test_bad_capture_shapes_types_and_values_are_rejected(change):
    x, zq, source, bias = captures()
    if change == "empty":
        x, zq, source = x[:0], zq[:0], source[:0]
    elif change == "rank":
        x = x.reshape(-1)
    elif change == "rows":
        zq, source = zq[:2], source[:2]
    elif change == "source":
        source = source[:, :1]
    elif change == "bias":
        bias = bias[:1]
    elif change == "integer":
        x = x.long()
    elif change == "nan":
        source[0, 0] = float("nan")
    else:
        bias[0] = float("inf")
    with pytest.raises(ValueError):
        target.fit_delta(x, zq, source, bias, min_samples=1, top_k=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_samples": 0},
        {"min_samples": True},
        {"top_k": 0},
        {"top_k": 3},
        {"top_k": True},
        {"rho": 0},
        {"rho": -1},
        {"rho": float("nan")},
        {"rho": float("inf")},
    ],
)
def test_fit_configuration_rejects_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        target.fit_delta(*captures(), **({"min_samples": 1, "top_k": 1} | kwargs))


def test_nonfinite_normal_equations_do_not_become_successful_fit():
    x, zq, source, bias = captures()
    with pytest.raises(ValueError, match="H"):
        target.fit_delta(x.double() * 1e200, zq, source, bias, min_samples=2, top_k=1)


def test_alpha_zero_preserves_signed_zero_bytes_and_interpolation_orientation():
    weight = torch.tensor([[0.0, -0.0, 2.0], [3.0, -4.0, 5.0]], dtype=torch.float16)
    delta = torch.ones((2, 3), dtype=torch.float64)
    original = target.tensor_sha(weight)
    control = target.alpha_weight(weight, delta * 1e300, 0)
    assert target.tensor_sha(control) == original and control.data_ptr() != weight.data_ptr()
    for alpha in (0.25, 0.5, 1):
        result = target.alpha_weight(weight, delta, alpha)
        np.testing.assert_array_equal(
            result.numpy(), (weight.numpy().astype(np.float64) + alpha).astype(np.float16)
        )
    assert target.tensor_sha(weight) == original


@pytest.mark.parametrize("alpha", [-1, 2, float("inf"), float("nan"), True, None])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError):
        target.alpha_weight(torch.ones((2, 3), dtype=torch.float16), torch.ones((2, 3)), alpha)


def test_alpha_rejects_shape_dtype_nonfinite_and_f16_overflow():
    weight = torch.ones((2, 3), dtype=torch.float16)
    for w, delta, alpha in (
        (weight.float(), torch.ones((2, 3)), 0),
        (weight, torch.ones((3, 2)), 1),
        (weight, torch.full((2, 3), float("nan")), 0),
        (weight, torch.full((2, 3), 1e10, dtype=torch.float64), 1),
    ):
        with pytest.raises(ValueError):
            target.alpha_weight(w, delta, alpha)


def fixture_checkpoint(root: Path):
    root.mkdir()
    gates = {
        f"model.layers.{i}.feed_forward.gate.weight": torch.zeros((2, 3), dtype=torch.float16)
        for i in (2, 3)
    }
    prefix = "model.layers.2.feed_forward.experts.0.w1"
    a = {
        next(iter(gates)): next(iter(gates.values())),
        prefix + ".trellis": torch.arange(16, dtype=torch.int16).reshape(1, 1, 16),
        prefix + ".mcg": torch.tensor(-877912083, dtype=torch.int32),
    }
    b = {
        list(gates)[1]: list(gates.values())[1],
        prefix + ".suh": torch.ones(16, dtype=torch.float16),
        prefix + ".svh": -torch.ones(16, dtype=torch.float16),
    }
    for i in (2, 3):
        b[f"model.layers.{i}.feed_forward.expert_bias"] = torch.tensor(
            [0.125, -0.125], dtype=torch.float32
        )
    owners = {}
    for index, values in enumerate((a, b), 1):
        name = f"model-{index:05}-of-00002.safetensors"
        save_file(values, root / name, metadata={"format": "pt"})
        owners.update({key: name for key in values})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": owners, "metadata": {"preserved": True}})
    )
    (root / "config.json").write_text(json.dumps({"model_type": "lfm2_moe"}))
    (root / "quantization_config.json").write_text(
        json.dumps(
            {
                "quant_method": "exl3",
                "tensor_storage": {
                    prefix: {
                        "quant_format": "exl3",
                        "stored_tensors": {prefix + ".trellis": {"shape": [1, 1, 16]}},
                    }
                },
            }
        )
    )
    (root / "tokenizer.json").write_text("{}")
    return gates


def directory_sha(root):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()
    }


def test_rewrite_export_is_native_compatible_and_preserves_all_other_tensor_bytes(tmp_path):
    from mlxl3_quantizer.checkpoint import validate_checkpoint_files

    source = tmp_path / "source"
    gates = fixture_checkpoint(source)
    before = directory_sha(source)
    overrides = {key: value + 0.5 for key, value in gates.items()}
    destination = tmp_path / "candidate"
    report = target.export_checkpoint(source, destination, overrides)
    assert report["status"] == "complete" and not report["publication"]
    assert directory_sha(source) == before
    validate_checkpoint_files(destination)
    assert target._owners(destination) == target._owners(source)
    assert set(report["gate_sha256"]) == set(gates)
    assert len(report["non_gate_sha256"]) == 6
    for key, filename in target._owners(source).items():
        a = load_file(source / filename)[key]
        b = load_file(destination / filename)[key]
        assert target.tensor_sha(b) == target.tensor_sha(overrides[key] if key in gates else a)
    for name in (
        "config.json",
        "quantization_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
    ):
        assert (source / name).read_bytes() == (destination / name).read_bytes()
    with pytest.raises(FileExistsError):
        target.export_checkpoint(source, destination, overrides)


@pytest.mark.parametrize(
    "failure",
    [
        "missing_gate",
        "extra_gate",
        "wrong_shape",
        "wrong_dtype",
        "index",
        "duplicate",
        "inside",
        "write",
    ],
)
def test_export_errors_never_publish_or_modify_original(tmp_path, monkeypatch, failure):
    import safetensors.torch

    source = tmp_path / "source"
    overrides = fixture_checkpoint(source)
    destination = tmp_path / "candidate"
    if failure == "missing_gate":
        overrides.pop(next(iter(overrides)))
    elif failure == "extra_gate":
        overrides["model.layers.4.feed_forward.gate.weight"] = torch.ones(
            (2, 3), dtype=torch.float16
        )
    elif failure == "wrong_shape":
        overrides[next(iter(overrides))] = torch.ones((3, 2), dtype=torch.float16)
    elif failure == "wrong_dtype":
        overrides[next(iter(overrides))] = torch.ones((2, 3))
    elif failure == "index":
        (source / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    elif failure == "duplicate":
        save_file(overrides, source / "model-duplicate.safetensors")
    elif failure == "inside":
        destination = source / "new"
    elif failure == "write":
        monkeypatch.setattr(
            safetensors.torch,
            "save_file",
            lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")),
        )
    before = directory_sha(source)
    with pytest.raises((ValueError, OSError)):
        target.export_checkpoint(source, destination, overrides)
    assert directory_sha(source) == before
    if destination.exists():
        assert json.loads((destination / "router-export.json").read_text())["status"] == "failed"


def cli_fixture(tmp_path, monkeypatch):
    source, teacher, quant, output = [
        tmp_path / name for name in ("source", "teacher", "quant", "output")
    ]
    gates = fixture_checkpoint(source)
    teacher.mkdir()
    quant.mkdir()
    teacher_parameters = {}
    for layer in (2, 3):
        key = f"model.layers.{layer}.feed_forward.gate.weight"
        teacher_parameters[key] = gates[key].to(torch.bfloat16)
        teacher_parameters[key.removesuffix("gate.weight") + "expert_bias"] = torch.zeros(2)
        save_file(
            {
                "x": torch.ones((8, 3), dtype=torch.float16),
                "zq": torch.zeros((8, 2), dtype=torch.float16),
            },
            quant / f"layer-{layer}.safetensors",
        )
        save_file(
            {"zsource": torch.tensor([[2.0, 1.0]] * 8, dtype=torch.bfloat16)},
            teacher / f"layer-{layer}.safetensors",
        )
    save_file(gates, quant / "routing-parameters.safetensors")
    save_file(teacher_parameters, teacher / "routing-parameters.safetensors")
    for name, value in (
        ("LAYERS", (2, 3)),
        ("FEATURES", 3),
        ("EXPERTS", 2),
        ("SAMPLES", 8),
        ("MIN_SAMPLES", 2),
        ("TOP_K", 1),
    ):
        monkeypatch.setattr(target, name, value)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return source, teacher, quant, output, gates


def test_cli_runs_actual_fit_and_three_physical_exports_on_disposable_capture_files(
    tmp_path, monkeypatch
):
    source, teacher, quant, output, gates = cli_fixture(tmp_path, monkeypatch)
    report = target.main([str(p) for p in (source, teacher, quant, output)])
    assert report["status"] == "complete" and not report["quality_validation"]
    assert set(report["candidates"]) == set(target.ALPHAS)
    assert json.loads((output / "report.json").read_text()) == report
    for candidate in report["candidates"].values():
        assert Path(candidate["checkpointdir"]).is_dir()
        assert target.file_sha(Path(candidate["gatefile"])) == candidate["gatefile_sha256"]
        assert set(load_file(candidate["gatefile"])) == set(gates)
    before = directory_sha(output)
    with pytest.raises(FileExistsError):
        target.main([str(p) for p in (source, teacher, quant, output)])
    assert directory_sha(output) == before


def test_cli_missing_input_and_interruption_save_explicit_failure(tmp_path, monkeypatch):
    source = tmp_path / "source"
    fixture_checkpoint(source)
    monkeypatch.setattr(target, "LAYERS", (2, 3))
    with pytest.raises(FileNotFoundError):
        target.run(source, tmp_path / "absent", tmp_path / "quant", tmp_path / "failed")
    failed = json.loads((tmp_path / "failed/report.json").read_text())
    assert failed["status"] == "failed" and not failed["publication"]
    monkeypatch.setattr(target, "_owners", lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        target.run(source, tmp_path, tmp_path, tmp_path / "interrupted")
    assert json.loads((tmp_path / "interrupted/report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "failure",
    [
        "stale_gate",
        "teacher_dtype",
        "quant_dtype",
        "extra_capture_key",
        "missing_layer",
        "production_gates",
    ],
)
def test_cli_rejects_stale_parameters_bad_dtypes_and_incomplete_layer_sets(
    tmp_path, monkeypatch, failure
):
    source, teacher, quant, output, gates = cli_fixture(tmp_path, monkeypatch)
    if failure == "stale_gate":
        save_file(
            {key: value + 1 for key, value in gates.items()},
            quant / "routing-parameters.safetensors",
        )
    elif failure in ("teacher_dtype", "quant_dtype", "extra_capture_key"):
        path = (teacher if failure == "teacher_dtype" else quant) / "layer-2.safetensors"
        values = load_file(path)
        if failure == "teacher_dtype":
            values["zsource"] = values["zsource"].float()
        elif failure == "quant_dtype":
            values["x"] = values["x"].float()
        else:
            values["surprise"] = torch.ones(1)
        save_file(values, path)
    elif failure == "missing_layer":
        (quant / "layer-3.safetensors").unlink()
    else:
        monkeypatch.setattr(target, "LAYERS", tuple(range(2, 40)))
    before = directory_sha(source)
    with pytest.raises((ValueError, FileNotFoundError)):
        target.main([str(p) for p in (source, teacher, quant, output)])
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and not report["publication"] and not report["candidates"]
    assert directory_sha(source) == before


def test_export_reads_back_written_bytes_and_rejects_non_gate_corruption(tmp_path, monkeypatch):
    import safetensors.torch

    source = tmp_path / "source"
    overrides = fixture_checkpoint(source)
    save = safetensors.torch.save_file

    def corrupt(tensors, filename, **kwargs):
        changed = dict(tensors)
        key = next(key for key in changed if key.endswith(".trellis"))
        changed[key] = changed[key] + 1
        save(changed, filename, **kwargs)

    monkeypatch.setattr(safetensors.torch, "save_file", corrupt)
    with pytest.raises(ValueError, match="tensor bytes"):
        target.export_checkpoint(source, tmp_path / "bad", overrides)
    assert json.loads((tmp_path / "bad/router-export.json").read_text())["status"] == "failed"


def test_export_partial_second_shard_failure_preserves_first_shard_and_failure_report(
    tmp_path, monkeypatch
):
    import safetensors.torch

    source = tmp_path / "source"
    gates = fixture_checkpoint(source)
    save = safetensors.torch.save_file
    written = []

    def fail_second(tensors, filename, **kwargs):
        if written:
            raise OSError("second shard interrupted")
        save(tensors, filename, **kwargs)
        written.append(Path(filename))

    monkeypatch.setattr(safetensors.torch, "save_file", fail_second)
    with pytest.raises(OSError, match="second shard"):
        target.export_checkpoint(source, tmp_path / "partial", gates)
    report = json.loads((tmp_path / "partial/router-export.json").read_text())
    assert report["status"] == "failed" and len(report["shards"]) == 1
    assert written[0].is_file()
    with safe_open(written[0], framework="pt") as handle:
        assert len(handle.keys()) > 0


def test_pure_guards_cover_boundary_and_invalid_counts():
    assert target.enough_samples(2047, 2048) is False
    assert target.enough_samples(2048, 2048) is True
    assert target.checked_alpha(0) == 0.0 and target.checked_alpha(1) == 1.0
    for count, minimum in ((-1, 1), (0, 0), (True, 1), (1, None)):
        with pytest.raises(ValueError):
            target.enough_samples(count, minimum)


def test_nonfinite_derived_fit_statistics_are_rejected_before_json_reporting():
    with pytest.raises(ValueError, match="nonfinite fit statistics"):
        target.fit_delta(
            torch.ones((3, 1), dtype=torch.float64),
            torch.zeros((3, 1), dtype=torch.float64),
            torch.full((3, 1), 1e200, dtype=torch.float64),
            torch.zeros(1),
            min_samples=2,
            top_k=1,
        )


def test_scalar_mcg_hash_matches_independent_numpy_payload_bytes():
    value = -877912083
    scalar = torch.tensor(value, dtype=torch.int32)
    expected = hashlib.sha256(np.asarray(value, dtype="<i4").tobytes()).hexdigest()
    assert scalar.ndim == 0 and target.tensor_sha(scalar) == expected
