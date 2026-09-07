"""CPU-only gates for optional proxy diagnostics in the reference converter."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "references/PonyExl3"))
pytest.importorskip("ponyexl3.convert")
from ponyexl3.convert import direct, hessian, measure, metal_search, mlx_trellis, reuse
from ponyexl3.convert.direct import LayerQuantizationBasis
from ponyexl3.ref.codebook import CodebookMode
from ponyexl3.ref.layer import EXL3Layer
from ponyexl3.ref.trellis import pack_trellis


def _cpu_fixture(monkeypatch):
    rng = np.random.default_rng(47)
    weight = rng.normal(size=(128, 128)).astype(np.float32) / 10
    acts = rng.normal(size=(32, 128)).astype(np.float32)
    layer = EXL3Layer("m", 128, 128, 4, np.zeros((8, 8, 64), dtype=np.uint16))
    fixture = SimpleNamespace(
        source=None,
        oracle=SimpleNamespace(layer=layer, cb=CodebookMode.DEFAULT),
        activations=acts,
    )
    monkeypatch.setattr(hessian, "build_layer_fixture", lambda *a, **kw: fixture)
    basis = LayerQuantizationBasis(weight, weight, None, None, {})
    monkeypatch.setattr(hessian, "prepare_layer_quantization_basis", lambda *a, **kw: basis)
    monkeypatch.setattr(hessian, "oracle_comparison_weights", lambda *a, **kw: (weight, weight))
    monkeypatch.setattr(measure, "_release_gpu_cache", lambda: None)

    def cpu_search(rows, *, k, **kwargs):
        # Stand in only for trellis search; run the actual CPU LDLQ and metrics.
        packed = np.zeros((rows.shape[0] // 16, rows.shape[1] // 16, 16 * k), np.uint16)
        reconstructed = np.round(rows * (1 << k)) / (1 << k)
        return packed, None, reconstructed.astype(np.float32)

    monkeypatch.setattr(hessian, "quantize_inner_matrix_direct", cpu_search)
    proxy_calls = []
    original_proxy = hessian.hessian_proxy_stats

    def counted_proxy(*args, **kwargs):
        proxy_calls.append(True)
        return original_proxy(*args, **kwargs)

    monkeypatch.setattr(hessian, "hessian_proxy_stats", counted_proxy)
    return acts, proxy_calls


def test_output_metrics_are_exact_and_cache_upgrades_missing_proxy(monkeypatch):
    acts, calls = _cpu_fixture(monkeypatch)
    kwargs = {"search_backend": "cpu", "compare_oracle": False, "calibration_activations": acts}
    slim = hessian.ldlq_quantize_layer("source", "plan", "m", compute_proxy=False, **kwargs)
    assert not calls
    full = hessian.ldlq_quantize_layer("source", "plan", "m", **kwargs)
    assert len(calls) == 1
    for name in ("source_output", "converted_output"):
        assert np.array_equal(getattr(slim, name), getattr(full, name))
    assert np.array_equal(slim.layer.trellis, full.layer.trellis)
    assert slim.layer.suh is full.layer.suh is None
    assert slim.layer.svh is full.layer.svh is None
    assert slim.stats == {k: v for k, v in full.stats.items() if not k.startswith("hessian_proxy_")}
    report = hessian.ldlq_layer_summary(slim)["stats"]
    assert all(report.get(name) is None for name in full.stats if name.startswith("hessian_proxy_"))

    reuse.enable()
    try:
        hessian.ldlq_quantize_layer("source", "plan", "m", fast_metrics=True, **kwargs)
        measured = hessian.ldlq_quantize_layer("source", "plan", "m", compute_proxy=False, **kwargs)
        assert measured.stats == slim.stats  # Cached fast metrics must not supply a NaN score.
        hessian.ldlq_quantize_layer("source", "plan", "m", fast_metrics=True, **kwargs)
        assert len(calls) == 1  # Emission reuses the layer even without a proxy.
        upgraded = hessian.ldlq_quantize_layer("source", "plan", "m", **kwargs)
        assert len(calls) == 2  # A full-diagnostics request cannot consume a missing proxy.
        cached = hessian.ldlq_quantize_layer("source", "plan", "m", **kwargs)
        assert len(calls) == 2
        assert cached.stats == upgraded.stats == full.stats
        kwargs["compare_oracle"] = True
        compared = hessian.ldlq_quantize_layer("source", "plan", "m", compute_proxy=False, **kwargs)
        assert len(calls) == 4  # Oracle ratios require converted and oracle proxies.
        assert compared.stats["hessian_proxy_rel_rms"] == full.stats["hessian_proxy_rel_rms"]
        cached = hessian.ldlq_quantize_layer("source", "plan", "m", **kwargs)
        assert cached.stats == compared.stats
        assert len(calls) == 4
        assert reuse.stats() == {
            "hits": 3,
            "misses": 4,
            "entries": 1,
            "bytes": full.layer.trellis.nbytes,
        }
    finally:
        reuse.disable()


def test_measurement_score_plan_and_resume_keep_required_diagnostics(monkeypatch, tmp_path):
    acts, calls = _cpu_fixture(monkeypatch)
    kwargs = {
        "candidate_bits": [2, 3],
        "search_backend": "cpu",
        "max_workers": 1,
        "calibration_activations": acts,
    }
    checkpoint = tmp_path / "measure.json"
    slim = measure.measure_ldlq_candidates(
        "source", "plan", ["m"], checkpoint_path=checkpoint, **kwargs
    )
    assert not calls
    for row in slim["records"]:
        assert row["summary"]["stats"]["hessian_proxy_rel_rms"] is None
    with pytest.raises(ValueError, match="was not computed"):
        measure.optimize_measurement_plan(
            slim, target_bpw=2.5, score_metric="hessian_proxy_rel_rms"
        )
    full = measure.measure_ldlq_candidates("source", "plan", ["m"], full_metrics=True, **kwargs)
    assert len(calls) == 2
    assert [r["score"] for r in slim["records"]] == [r["score"] for r in full["records"]]
    assert (
        measure.optimize_measurement_plan(slim, target_bpw=2.5)["bit_plan"]
        == (measure.optimize_measurement_plan(full, target_bpw=2.5)["bit_plan"])
    )
    resumed = measure.measure_ldlq_candidates(
        "source",
        "plan",
        ["m"],
        checkpoint_path=checkpoint,
        resume=True,
        score_metric="hessian_proxy_rel_rms",
        **kwargs,
    )
    assert len(calls) == 4  # Output-only checkpoint cannot satisfy a proxy score request.
    for row in resumed["records"]:
        assert np.isfinite(row["score"])
        assert row["score"] == row["summary"]["stats"]["hessian_proxy_rel_rms"]
    outputs = measure.measure_ldlq_candidates(
        "source", "plan", ["m"], checkpoint_path=checkpoint, resume=True, **kwargs
    )
    assert len(calls) == 4  # Present diagnostics are reused, with the selected score refreshed.
    assert [r["score"] for r in outputs["records"]] == [r["score"] for r in slim["records"]]
    assert outputs["best_by_module"] == slim["best_by_module"]
    oracle = measure.measure_ldlq_candidates(
        "source",
        "plan",
        ["m"],
        checkpoint_path=checkpoint,
        resume=True,
        compare_oracle=True,
        score_metric="hessian_proxy_rel_rms_over_oracle",
        **kwargs,
    )
    assert len(calls) == 8
    assert all(r["summary"]["stats"]["oracle_metrics"] for r in oracle["records"])


def test_legacy_same_metric_score_fallback_does_not_relabel_other_metrics():
    record = {
        "module": "m",
        "k": 4,
        "score_metric": "output_rel_rms",
        "score": 0.125,
        "summary": {"shape": [128, 128], "stats": {}},
    }
    legacy = {"score_metric": "output_rel_rms", "records": [record]}
    assert measure.optimize_measurement_plan(legacy, target_bpw=4)["objective"] == 0.125
    with pytest.raises(ValueError, match="was not computed"):
        measure.optimize_measurement_plan(legacy, target_bpw=4, score_metric="public_rel_rms")


def test_ldl_owned_cholesky_buffer_is_exact_and_does_not_alias_input():
    rng = np.random.default_rng(19)
    for n in (16, 32, 128, 256):
        x = rng.normal(size=(n, n)).astype(np.float32)
        for order in ("C", "F"):
            h = np.array(x @ x.T + np.eye(n, dtype=np.float32), order=order)
            before = h.copy()
            reference = hessian.block_ldl(h, copy_cholesky=True)
            result = hessian.block_ldl(h)
            assert np.array_equal(result.l, reference.l)
            assert np.array_equal(result.hessian, reference.hessian)
            assert result.retries == reference.retries
            assert np.array_equal(h, before)
            assert not np.shares_memory(result.l, h)
            assert not np.shares_memory(result.l, result.hessian)
            result.l.fill(0)
            assert np.array_equal(h, before)
            assert np.array_equal(result.hessian, reference.hessian)
    h = np.eye(32, dtype=np.float32)
    h[0, 1] = h[1, 0] = 1.1
    reference = hessian.block_ldl(h, copy_cholesky=True)
    result = hessian.block_ldl(h)
    assert result.retries == reference.retries > 0
    assert np.array_equal(result.l, reference.l)
    assert np.array_equal(result.hessian, reference.hessian)


def test_zero_shrinkage_reuses_identical_diagnostic(monkeypatch):
    original = hessian.hessian_offdiag_rel
    calls = []

    def counted(matrix):
        calls.append(True)
        return original(matrix)

    monkeypatch.setattr(hessian, "hessian_offdiag_rel", counted)
    for acts in (np.zeros((8, 16), np.float32), np.arange(128, dtype=np.float32).reshape(8, 16)):
        for shrinkage in (0.0, -0.0, 0.1):
            calls.clear()
            prepared, stats = hessian._prepare_activation_hessian(
                acts, sigma_reg=0.025, hessian_shrinkage=shrinkage
            )
            assert len(calls) == (1 if shrinkage == 0 else 2)
            raw = hessian.capture_hessian(acts)
            shrunk = hessian.apply_hessian_shrinkage(raw, shrinkage=shrinkage)
            assert stats["hessian_offdiag_rel_unshrunk"] == original(raw)
            assert stats["hessian_offdiag_rel"] == original(shrunk)
            assert np.array_equal(prepared.hessian, hessian.prepare_hessian_for_ldl(shrunk).hessian)


def test_gss_reconstruction_only_preserves_every_score_and_scale(monkeypatch):
    # Exercise the Metal plumbing with NumPy stand-ins; no GPU is initialized.
    mx = SimpleNamespace(eval=lambda *args: None)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.setitem(sys.modules, "mlx.core", mx)

    def search(tiles, *, k, cb):
        decoded = (np.round(tiles * (1 << k)) / (1 << k)).astype(np.float32)
        return decoded, np.zeros_like(tiles, dtype=np.uint16)

    def cpu_search(tile, *, k, cb, **kwargs):
        decoded, states = search(tile[None], k=k, cb=cb)
        return states[0], decoded[0]

    monkeypatch.setattr(metal_search, "quantize_tiles_mlx", search)
    monkeypatch.setattr(metal_search, "quantize_tiles_mlx_np", search)
    monkeypatch.setattr(direct, "quantize_tile_reference", cpu_search)
    pack_calls = []

    def packed(states, k):
        pack_calls.append(True)
        return pack_trellis(states, k)

    monkeypatch.setattr(mlx_trellis, "pack_trellis_mlx", packed)
    monkeypatch.setattr(direct, "pack_trellis", packed)
    rng = np.random.default_rng(37)
    weight = rng.normal(size=(128, 128)).astype(np.float32)
    monkeypatch.setattr(direct, "read_source_public_matrix", lambda *args: weight)
    quantize = direct.quantize_inner_matrix_direct

    def with_pack(*args, **kwargs):
        kwargs["return_packed"] = True
        return quantize(*args, **kwargs)

    runs = []
    gss = direct.g_scale_gss

    def observed_gss(score):
        values = []

        def observed(scale):
            value = score(scale)
            values.append((scale, value))
            return value

        result = gss(observed)
        runs.append((values, result))
        return result

    monkeypatch.setattr(direct, "g_scale_gss", observed_gss)
    layer = EXL3Layer("m", 128, 128, 4, np.zeros((8, 8, 64), dtype=np.uint16))
    for k in range(2, 9):
        for backend in ("cpu", "metal"):
            kwargs = {"k": k, "cb": CodebookMode.MCG, "search_backend": backend}
            baseline = quantize(weight[:16, :32], return_states=False, **kwargs)
            assert baseline[0] is not None
            pack_calls.clear()
            slim = quantize(weight[:16, :32], return_states=False, return_packed=False, **kwargs)
            assert slim[0] is slim[1] is None
            assert not pack_calls
            assert np.array_equal(slim[2], baseline[2])
            # Explicit roundtrip verification still packs even if the caller discards it.
            checked = quantize(
                weight[:16, :32], return_packed=False, verify_roundtrip=True, **kwargs
            )
            assert checked[0] is None and checked[1] is not None and pack_calls
            assert np.array_equal(checked[2], baseline[2])
        bases = []
        runs.clear()
        for function in (with_pack, quantize):
            monkeypatch.setattr(direct, "quantize_inner_matrix_direct", function)
            pack_calls.clear()
            bases.append(
                direct.prepare_layer_quantization_basis(
                    "source",
                    None,
                    layer,
                    CodebookMode.MCG,
                    scale_mode="computed",
                    search_backend="metal",
                    quant_bits=k,
                    g_scale_width=1,
                )
            )
            assert bool(pack_calls) == (function is with_pack)
        assert runs[0] == runs[1]
        for name in ("source_public", "target_inner", "suh", "svh"):
            assert np.array_equal(getattr(bases[0], name), getattr(bases[1], name))

    # Packing never validated or repaired nonfinite decoded values: preserve their bits.
    nonfinite = weight[:16, :32].copy()
    nonfinite[0, :3] = (np.nan, np.inf, -np.inf)
    for backend in ("cpu", "metal"):
        kwargs = {"k": 4, "cb": CodebookMode.MCG, "search_backend": backend}
        baseline = quantize(nonfinite, return_states=False, **kwargs)
        slim = quantize(nonfinite, return_states=False, return_packed=False, **kwargs)
        assert slim[2].tobytes() == baseline[2].tobytes()
        # Opting out of the returned packed tensor must not suppress a failed check.
        with monkeypatch.context() as check:
            check.setattr(direct, "unpack_trellis", lambda packed, k: np.ones((1,), np.uint16))
            with pytest.raises(AssertionError, match="non-round-trippable"):
                quantize(weight[:16, :32], return_packed=False, verify_roundtrip=True, **kwargs)
