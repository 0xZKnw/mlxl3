"""FP32 converter product service boundary; physical GEMM parity is separate."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlxl3_quantizer.feedback import matmul_native, matrix_values

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/MLXL3Studio/Vendor/PonyExl3"))

FAKE = """#!/usr/bin/env python3
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('left');p.add_argument('right');p.add_argument('output');p.add_argument('--m',type=int);p.add_argument('--n',type=int);p.add_argument('--k',type=int);p.add_argument('--full-fp32',action='store_true');a=p.parse_args()
fault=os.environ.get('MATRIX_FAULT','')
if fault=='silent':time.sleep(30)
if fault=='exit':sys.exit(3)
if fault=='invalid':print('not JSON');sys.exit(0)
x=np.fromfile(a.left,'<f4').reshape(a.m,a.k);y=np.fromfile(a.right,'<f4').reshape(a.k,a.n)
out=Path(a.output);out.mkdir()
(out/'output.f32').write_bytes(np.asarray(x@y,dtype='<f4').tobytes())
r={'schema':1,'status':'complete','m':a.m,'n':a.n,'k':a.k,'values':a.m*a.n,'arithmetic':'simdgroup-fp32' if a.full_fp32 else 'nax-tf32'}
if fault=='foreign':r['m']+=8
if fault=='bool':r['schema']=True
if fault=='partial':(out/'output.f32').write_bytes(b'')
if fault=='nonfinite':(out/'output.f32').write_bytes(bytes.fromhex('0000c07f')+bytes((a.m*a.n-1)*4))
if fault=='symlink':(out/'output.f32').unlink();(out/'output.f32').symlink_to(a.left)
if fault=='extra':(out/'unexpected').write_bytes(b'x')
if fault!='missing-report':(out/'report.json').write_text(json.dumps(r))
if fault=='directory-symlink':
    moved=out.with_name('foreign-result');out.rename(moved);out.symlink_to(moved,target_is_directory=True)
if fault=='stdout-flood':sys.stdout.write('x'*100000)
if fault=='stderr-flood':sys.stderr.write('x'*100000)
print(json.dumps(r))
"""


@pytest.fixture
def engine(tmp_path):
    path = tmp_path / "engine"
    path.write_text(FAKE.replace("#!/usr/bin/env python3", f"#!{sys.executable}"))
    path.chmod(0o755)
    return path


def test_real_file_protocol_preserves_strided_matrix_order_and_fp32(engine):
    left = np.arange(128, dtype=np.float32).reshape(16, 8)[:, ::-1]
    right = np.ones((8, 24), dtype=np.float32)
    actual = matmul_native(left, right, engine=engine)
    # Integer sums in this range are exact, independent of BLAS reduction order.
    expected = np.repeat(np.sum(left.astype(np.float64), axis=1, keepdims=True), 24, axis=1)
    assert actual.shape == (16, 24) and actual.dtype == np.dtype("float32")
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "fault",
    [
        "exit",
        "invalid",
        "foreign",
        "bool",
        "partial",
        "nonfinite",
        "symlink",
        "directory-symlink",
        "extra",
        "missing-report",
        "stdout-flood",
        "stderr-flood",
    ],
)
def test_native_matrix_never_accepts_bad_or_incomplete_child_results(engine, monkeypatch, fault):
    monkeypatch.setenv("MATRIX_FAULT", fault)
    with pytest.raises((RuntimeError, ValueError)):
        matmul_native(np.ones((8, 8)), np.ones((8, 8)), engine=engine, timeout=3)


def test_matrix_timeout_and_cancellation_reap_actual_child_pid(engine, monkeypatch):
    from mlxl3_quantizer import process as native_process

    monkeypatch.setenv("MATRIX_FAULT", "silent")
    original = subprocess.Popen
    launched = []

    def remember(*args, **kwargs):
        process = original(*args, **kwargs)
        launched.append(process.pid)
        return process

    monkeypatch.setattr(native_process.subprocess, "Popen", remember)
    with pytest.raises(TimeoutError):
        matmul_native(np.ones((8, 8)), np.ones((8, 8)), engine=engine, timeout=0.15)
    assert len(launched) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(launched[0], 0)

    class Cancelled(Exception):
        pass

    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise Cancelled

    with pytest.raises(Cancelled):
        matmul_native(np.ones((8, 8)), np.ones((8, 8)), engine=engine, cancel_check=cancel)
    assert len(launched) == 2
    with pytest.raises(ProcessLookupError):
        os.kill(launched[1], 0)


@pytest.mark.parametrize(
    "left,right",
    [
        ([], []),
        (np.ones((0, 8)), np.ones((8, 8))),
        (np.ones(8), np.ones((8, 8))),
        (np.ones((8, 8)), np.ones((9, 8))),
        (np.full((8, 8), np.nan), np.ones((8, 8))),
        (np.ones((8, 8)), np.full((8, 8), np.inf)),
    ],
)
def test_invalid_matrices_fail_before_spawning(engine, monkeypatch, left, right):
    from mlxl3_quantizer import process as native_process

    def forbidden(*args, **kwargs):
        pytest.fail("invalid matrix must not start a child")

    monkeypatch.setattr(native_process.subprocess, "Popen", forbidden)
    with pytest.raises(ValueError):
        matmul_native(left, right, engine=engine)


def test_dimensions_reject_empty_and_budget_overflow_without_allocations():
    assert matrix_values(96, 128, 32) == 12288
    assert matrix_values(8, 8, 8388608) == 64
    for shape in [(0, 8, 8), (8, 0, 8), (8, 8, 0), (8, 8, 8388616), (2**32, 8, 8)]:
        with pytest.raises(ValueError):
            matrix_values(*shape)


def test_explicit_full_precision_mode_is_passed_to_the_real_child(engine):
    values = np.ones((3, 8), dtype=np.float32)
    actual = matmul_native(values, values.T, engine=engine, full_fp32=True)
    np.testing.assert_array_equal(actual, np.full((3, 3), 8, dtype=np.float32))


def test_wide_products_split_rows_and_columns_without_splitting_k(monkeypatch):
    from mlxl3_quantizer import feedback

    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", "/native")
    monkeypatch.setattr(feedback, "_CONVERTER_MATRIX_VALUES", 1024)
    left = (np.arange(32 * 16).reshape(32, 16) % 17).astype(np.float32)
    right = (np.arange(16 * 128).reshape(16, 128) % 11).astype(np.float32)
    calls = []

    def product(a, b, **kwargs):
        assert a.shape[1] == b.shape[0] == 16
        assert max(a.size, b.size, a.shape[0] * b.shape[1]) <= 1024
        calls.append((a.shape, b.shape))
        return a @ b

    monkeypatch.setattr(feedback, "matmul_native", product)
    actual = feedback.converter_matmul(left, right)
    np.testing.assert_array_equal(
        actual, (left.astype(np.float64) @ right.astype(np.float64)).astype(np.float32)
    )
    assert calls == [((16, 16), (16, 64))] * 4


def test_head_dimensions_fit_budget_without_allocating_the_complete_head():
    from mlxl3_quantizer.feedback import matrix_blocks

    rows, cols = matrix_blocks(2432, 248320, 128, 67108864)
    assert (rows, cols) == (256, 248320)
    assert max(rows * 128, 128 * cols, rows * cols) <= 67108864


def test_later_native_chunk_failure_never_returns_a_partially_filled_product(monkeypatch):
    from mlxl3_quantizer import feedback

    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", "/native")
    monkeypatch.setattr(feedback, "_CONVERTER_MATRIX_VALUES", 1024)
    calls = 0

    def product(a, b, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("truncated later product")
        return a @ b

    monkeypatch.setattr(feedback, "matmul_native", product)
    with pytest.raises(RuntimeError, match="truncated later product"):
        feedback.converter_matmul(np.ones((32, 16)), np.ones((16, 128)))
    assert calls == 2


def test_matrix_blocks_small_budgets_keep_every_extent_positive_and_bounded():
    from mlxl3_quantizer.feedback import matrix_blocks

    for m in [1, 3, 16, 17, 32]:
        for n in [1, 3, 32, 33, 128]:
            for k in [1, 3, 8, 16]:
                for limit in [1, 3, 8, 16, 32, 256]:
                    if k > limit:
                        with pytest.raises(ValueError):
                            matrix_blocks(m, n, k, limit)
                        continue
                    rows, cols = matrix_blocks(m, n, k, limit)
                    assert 0 < rows <= m and 0 < cols <= n
                    assert max(rows * k, k * cols, rows * cols) <= limit


def test_frozen_real_feedback_corpus_has_exact_hashes_shapes_and_detects_wrong_precision():
    import hashlib
    import json

    root = Path(__file__).parent / "fixtures/native-feedback"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["schema"] == 1
    assert len(manifest["products"]) == 7 and len(manifest["searches"]) == 8
    words, wrong_full_precision = 0, 0
    for kind in ["products", "searches"]:
        for case in manifest[kind]:
            path = root / case["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == case["sha256"]
            with np.load(path, allow_pickle=False) as arrays:
                assert set(arrays) == set(case["arrays"])
                for key, descriptor in case["arrays"].items():
                    value = arrays[key]
                    assert list(value.shape) == descriptor["shape"]
                    assert str(value.dtype) == descriptor["dtype"]
                    assert value.size > 0 and np.isfinite(value).all()
                if kind == "products":
                    left, right, expected = arrays["left"], arrays["right"], arrays["output"]
                    assert left.shape[1] == right.shape[0]
                    assert expected.shape == (left.shape[0], right.shape[1])
                    full = left @ right
                    wrong_full_precision += int(
                        np.count_nonzero(full.view(np.uint32) != expected.view(np.uint32))
                    )
                    words += expected.size
    assert words == 32768
    assert wrong_full_precision > 0  # The rejected full-FP32 prototype cannot satisfy this oracle.
