"""Maintained converter boundaries with frozen independent Metal results."""

import importlib.abc
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from mlxl3_quantizer import native

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "apps/MLXL3Studio/Vendor/PonyExl3"
CORPUS = ROOT / "tests/fixtures/native-quantizer"
sys.path.insert(0, str(VENDOR))


@pytest.fixture(autouse=True)
def no_mlx_import(monkeypatch):
    attempted = []

    class RejectMLX(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname.split(".")[0] in {"mlx", "mlx_lm"}:
                attempted.append(fullname)
                raise ImportError("MLX is forbidden on the native conversion path")

    monkeypatch.setattr(sys, "meta_path", [RejectMLX(), *sys.meta_path])
    yield
    assert not attempted, attempted


def scatter_kernel_tiles(tiles, rows, cols):
    """Independent lane/fragment placement from the documented TC 16x16 format."""
    out = np.empty((rows, cols), dtype=tiles.dtype)
    for tile_index, tile in enumerate(tiles):
        row_base = (tile_index // (cols // 16)) * 16
        col_base = (tile_index % (cols // 16)) * 16
        for lane in range(32):
            for fragment in range(8):
                r = 2 * (lane % 4) + fragment % 4 // 2 * 8 + fragment % 2
                c = lane // 4 + fragment // 4 * 8
                out[row_base + r, col_base + c] = tile[lane * 8 + fragment]
    return out


@pytest.mark.parametrize("k,cb", [(2, 0), (4, 1), (8, 2)])
def test_matrix_layout_packing_and_output_flags_use_independent_fixture(monkeypatch, k, cb):
    from ponyexl3.convert.direct import quantize_inner_matrix_direct

    tiles = np.fromfile(CORPUS / "tiles.f32", "<f4").reshape(4, 256)
    decoded = np.fromfile(CORPUS / f"oracle-{k}-{cb}-decoded.f32", "<f4").reshape(4, 256)
    states = np.fromfile(CORPUS / f"oracle-{k}-{cb}-states.u16", "<u2").reshape(4, 256)
    packed = np.fromfile(CORPUS / f"oracle-{k}-{cb}-packed.u16", "<u2").reshape(2, 2, 16 * k)
    calls = []

    def search(actual, actual_k, actual_cb=0, **kwargs):
        np.testing.assert_array_equal(actual, tiles)
        assert (actual_k, actual_cb) == (k, cb)
        kwargs["cancel_check"]()
        calls.append(1)
        return decoded, states

    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", "/explicit/native/engine")
    monkeypatch.setattr(native, "quantize_tiles_native", search)
    matrix = scatter_kernel_tiles(tiles, 32, 32)
    for with_states, with_packed in [(True, True), (False, True), (False, False)]:
        actual_packed, actual_states, reconstructed = quantize_inner_matrix_direct(
            matrix,
            k=k,
            cb=cb,
            search_backend="native",
            verify_roundtrip=True,
            return_states=with_states,
            return_packed=with_packed,
        )
        np.testing.assert_array_equal(reconstructed, scatter_kernel_tiles(decoded, 32, 32))
        if with_packed:
            np.testing.assert_array_equal(actual_packed, packed)
        else:
            assert actual_packed is None
        if with_states:
            np.testing.assert_array_equal(actual_states, states.reshape(2, 2, 256))
        else:
            assert actual_states is None
    assert len(calls) == 3


def test_native_converter_batches_and_propagates_shared_cancellation(monkeypatch):
    from ponyexl3.convert import cancel

    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", "/native")
    monkeypatch.setattr(native, "_CONVERTER_TILE_BATCH", 2)
    values = np.arange(5 * 256, dtype=np.float32).reshape(5, 256)
    calls = []

    def search(block, k, cb, **kwargs):
        kwargs["cancel_check"]()
        calls.append(block.shape[0])
        return block.copy(), np.ones(block.shape, dtype=np.uint16)

    monkeypatch.setattr(native, "quantize_tiles_native", search)
    decoded, states = native.converter_tiles(values, 4)
    assert calls == [2, 2, 1]
    np.testing.assert_array_equal(decoded, values)
    assert states.shape == values.shape and states.min() == 1
    try:
        cancel.request()
        with pytest.raises(cancel.ConversionCancelled):
            native.converter_tiles(values, 4)
    finally:
        cancel.clear()
    assert calls == [2, 2, 1]


def test_native_converter_requires_explicit_engine(monkeypatch):
    monkeypatch.delenv("MLXL3_NATIVE_ENGINE", raising=False)
    with pytest.raises(RuntimeError, match="MLXL3_NATIVE_ENGINE"):
        native.converter_tiles(np.zeros((1, 256)), 4)


def test_native_basis_and_activation_policy_do_not_import_mlx():
    from types import SimpleNamespace

    from ponyexl3.convert.direct import _oracle_or_identity_basis
    from ponyexl3.convert.hessian import (
        _public_activations_to_inner_for_backend,
        public_activations_to_inner,
    )

    from mlxl3_quantizer.transforms import hadamard128, public_matrix_to_inner_native

    rng = np.random.default_rng(75)
    source = rng.normal(size=(128, 256)).astype(np.float32)
    su = rng.uniform(0.25, 2, 128).astype(np.float16)
    sv = rng.uniform(0.25, 2, 256).astype(np.float16)
    layer = SimpleNamespace(suh=su, svh=sv, in_features=128, out_features=256)
    actual = _oracle_or_identity_basis(
        source, layer, scale_mode="oracle_safe", search_backend="native"
    )
    np.testing.assert_array_equal(
        actual.target_inner, public_matrix_to_inner_native(source, suh=su, svh=sv)
    )
    assert actual.stats["basis_native_hadamard"] and not actual.stats["basis_mlx_hadamard"]
    for rows in [191, 192]:
        x = rng.normal(size=(rows, 128)).astype(np.float16)
        actual, mlx_used = _public_activations_to_inner_for_backend(x, su, search_backend="native")
        expected = (
            hadamard128(x.astype(np.float32) * su.astype(np.float32))
            if rows >= 192
            else public_activations_to_inner(x, su)
        )
        np.testing.assert_array_equal(actual, expected)
        assert not mlx_used


def test_non_mlx_backends_never_clear_an_mlx_cache():
    from ponyexl3.convert import driver, measure

    for backend in ["native", "cpu"]:
        driver._release_gpu_cache(backend)
        measure._release_gpu_cache(backend)


# This executable replays frozen independent search results. It checks the real
# process/file protocol and the converter CLI; physical search is tested separately.
REPLAY = """#!/usr/bin/env python3
import argparse,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('cmd');p.add_argument('input');p.add_argument('output');p.add_argument('--k',type=int);p.add_argument('--codebook',type=int);p.add_argument('--max-scratch-bytes',type=int);a=p.parse_args()
root=Path(os.environ['FROZEN_TILES']);x=np.fromfile(a.input,'<f4').reshape(-1,256);count=len(x)
ref=np.fromfile(root/'tiles.f32','<f4').reshape(4,256)
assert np.array_equal(x,np.tile(ref,(count//4,1)))
assert a.cmd=='quantize-tiles' and count%4==0
out=Path(a.output);out.mkdir()
for name in ['decoded.f32','states.u16','packed.u16']:
 raw=(root/f'oracle-{a.k}-{a.codebook}-{name}').read_bytes();(out/name).write_bytes(raw*(count//4))
r={'schema':1,'status':'complete','tiles':count,'values':x.size,'k':a.k,'codebook':a.codebook,'max_scratch_bytes':a.max_scratch_bytes,'launch_tiles':min(count,a.max_scratch_bytes//(516*(1<<(16-a.k))))}
(out/'report.json').write_text(json.dumps(r));print(json.dumps(r))
"""


def test_advanced_converter_source_plan_to_checkpoint_without_mlx(tmp_path, monkeypatch):
    from ponyexl3.ref.loader import load_exl3_layer

    key = "model.layers.0.mlp.gate_proj"
    source, output = tmp_path / "source", tmp_path / "output"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "llama", "hidden_size": 128, "num_hidden_layers": 1})
    )
    tiles = np.tile(np.fromfile(CORPUS / "tiles.f32", "<f4").reshape(4, 256), (16, 1))
    save_file(
        {key + ".weight": scatter_kernel_tiles(tiles, 128, 128).T.copy()},
        source / "model.safetensors",
    )
    engine = tmp_path / "engine"
    engine.write_text(REPLAY.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    engine.chmod(0o755)
    # The separate process must also reject even an attempted MLX fallback.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys,importlib.abc\nfrom pathlib import Path\n"
        "class Reject(importlib.abc.MetaPathFinder):\n"
        " def find_spec(self,fullname,path,target=None):\n"
        "  if fullname.split('.')[0] in {'mlx','mlx_lm'}:\n"
        f"   Path({str(tmp_path / 'forbidden')!r}).write_text(fullname)\n"
        "   raise ImportError('forbidden MLX import')\n"
        "sys.meta_path.insert(0,Reject())\n"
    )
    env = {
        **os.environ,
        "MLXL3_NATIVE_ENGINE": str(engine),
        "FROZEN_TILES": str(CORPUS),
        "PYTHONPATH": os.pathsep.join([str(tmp_path), str(ROOT / "src"), str(VENDOR)]),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ponyexl3.cli.convert_advanced",
            "--in-dir",
            str(source),
            "--out-dir",
            str(output),
            "--direct-layer",
            "--only-module",
            key,
            "--scale-mode",
            "identity",
            "--search-backend",
            "native",
            "--codebook",
            "mcg",
            "--json",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "forbidden").exists()
    report = json.loads(result.stdout)
    assert report["search_backend"] == "native"
    layer = load_exl3_layer(output, key)
    expected = np.tile(
        np.fromfile(CORPUS / "oracle-4-1-packed.u16", "<u2").reshape(4, 64), (16, 1)
    ).reshape(8, 8, 64)
    np.testing.assert_array_equal(layer.trellis, expected)
    assert any(key + ".trellis" in load_file(file) for file in output.glob("*.safetensors"))


@pytest.mark.parametrize("fault", ["missing", "empty", "shape", "nonfinite", "different"])
def test_physical_driver_rejects_nontrivial_checkpoint_failures(tmp_path, fault):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "physical_host", ROOT / "tests/physical_native_converter_host.py"
    )
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    key = "model.layers.0.mlp.gate_proj.trellis"
    expected = np.arange(4096, dtype=np.uint16).reshape(8, 8, 64)
    for directory in [left, right]:
        (directory / "quantization_config.json").write_text(json.dumps({"quant_method": "exl3"}))
    save_file({key: expected}, left / "weights.safetensors")
    if fault != "missing":
        actual = expected.copy()
        if fault == "empty":
            actual = np.empty((0, 8, 64), dtype=np.uint16)
        elif fault == "shape":
            actual = actual.reshape(4, 8, 128)
        elif fault == "nonfinite":
            actual = actual.astype(np.float32)
            actual[0, 0, 0] = np.nan
        elif fault == "different":
            actual[0, 0, 0] = 5
        save_file({key: actual}, right / "weights.safetensors")
    with pytest.raises(ValueError):
        driver.compare_checkpoints(left, right)


def test_physical_driver_failure_never_publishes_parity(tmp_path):
    engine = tmp_path / "broken-engine"
    engine.write_text("#!/bin/sh\nexit 3\n")
    engine.chmod(0o755)
    output = tmp_path / "result"
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/physical_native_converter_host.py"),
            "--engine",
            str(engine),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert process.returncode != 0
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and report["parity"] is False
    assert report["cases"] == [] and "error" in report


def test_converter_request_never_spans_more_than_one_scratch_launch(monkeypatch):
    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", "/native")
    calls = []

    def search(block, k, cb, **kwargs):
        calls.append(block.shape[0])
        assert block.shape[0] * 516 * (1 << (16 - k)) <= 256 * 1024 * 1024
        return block.copy(), np.zeros(block.shape, dtype=np.uint16)

    monkeypatch.setattr(native, "quantize_tiles_native", search)
    native.converter_tiles(np.zeros((70, 256)), 2)
    assert calls == [31, 31, 8]


@pytest.mark.parametrize(
    "failure", [ValueError("partial output"), FileNotFoundError("missing engine")]
)
def test_native_service_failures_are_fatal_not_unsupported_modules(monkeypatch, failure):
    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", "/native")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(native, "quantize_tiles_native", fail)
    with pytest.raises(RuntimeError, match="native converter service failed"):
        native.converter_tiles(np.zeros((1, 256)), 4)


def test_unsupported_native_k_cannot_be_relabelled_as_a_skipped_module(monkeypatch):
    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", "/native")
    with pytest.raises(RuntimeError, match="configuration"):
        native.converter_tiles(np.zeros((1, 256)), 1)


def test_unqualified_native_ldlq_stops_before_any_service_call(monkeypatch):
    from ponyexl3.convert.hessian import ldlq_inner_matrix, ldlq_quantize_layer

    def forbidden(*args, **kwargs):
        pytest.fail("unqualified LDLQ must not invoke a GPU service")

    monkeypatch.setattr(native, "converter_tiles", forbidden)
    with pytest.raises(RuntimeError, match="FP32 compensation"):
        ldlq_inner_matrix(np.zeros((16, 16)), np.eye(16), k=4, cb=1, search_backend="native")
    with pytest.raises(RuntimeError, match="FP32 compensation"):
        ldlq_quantize_layer("missing", "missing", "missing", search_backend="native")
