"""Native conversion process/data boundaries, independent of a physical GPU."""

import os
import subprocess

import numpy as np
import pytest

from mlxl3_quantizer.native import launch_tiles, quantize_tiles_native

FAKE = """#!/usr/bin/env python3
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('input');p.add_argument('output');p.add_argument('--k',type=int);p.add_argument('--codebook',type=int);p.add_argument('--max-scratch-bytes',type=int);a=p.parse_args()
fault=os.environ.get('TILE_FAULT','')
if os.environ.get('TILE_PID'):Path(os.environ['TILE_PID']).write_text(str(os.getpid()))
if fault=='silent':time.sleep(30)
if fault=='flood':sys.stdout.write('x'*100000);sys.stdout.flush();time.sleep(30)
if fault=='exit':sys.exit(3)
if fault=='invalid':print('invalid JSON');sys.exit(0)
raw=Path(a.input).read_bytes();n=len(raw)//4;tiles=n//256
out=Path(a.output);out.mkdir()
(out/'decoded.f32').write_bytes(raw)
(out/'states.u16').write_bytes(bytes(n*2))
(out/'packed.u16').write_bytes(bytes(tiles*16*a.k*2))
r={'schema':1,'status':'complete','tiles':tiles,'values':n,'k':a.k,'codebook':a.codebook,'launch_tiles':min(tiles,a.max_scratch_bytes//(516*(1<<(16-a.k)))),'max_scratch_bytes':a.max_scratch_bytes}
if fault=='foreign':r['codebook']=99
if fault=='launch':r['launch_tiles']=0
if fault=='empty':r['values']=0
if fault=='partial':(out/'states.u16').write_bytes(b'')
if fault=='nonfinite':(out/'decoded.f32').write_bytes(bytes.fromhex('0000c07f')+raw[4:])
if fault=='symlink':(out/'decoded.f32').unlink();(out/'decoded.f32').symlink_to(a.input)
if fault=='extra':(out/'unexpected').write_bytes(b'x')
if fault!='missing-report':(out/'report.json').write_text(json.dumps(r))
if fault=='directory-symlink':
    moved=out.with_name('foreign-result');out.rename(moved);out.symlink_to(moved,target_is_directory=True)
print(json.dumps(r))
"""


@pytest.fixture
def engine(tmp_path):
    file = tmp_path / "engine"
    file.write_text(FAKE)
    file.chmod(0o755)
    return file


def test_public_native_tile_entry_preserves_binary_order_shape_and_types(engine):
    values = np.arange(768, dtype=np.float32).reshape(3, 256)[:, ::-1]
    decoded, states = quantize_tiles_native(values, 4, engine=engine)
    assert decoded.dtype == np.dtype("float32") and states.dtype == np.dtype("uint16")
    np.testing.assert_array_equal(decoded, values)
    np.testing.assert_array_equal(states, np.zeros((3, 256), dtype=np.uint16))
    decoded, _ = quantize_tiles_native(values[0], 4, engine=engine)
    assert decoded.shape == (1, 256)


@pytest.mark.parametrize(
    "fault",
    [
        "exit",
        "invalid",
        "foreign",
        "launch",
        "empty",
        "partial",
        "nonfinite",
        "symlink",
        "directory-symlink",
        "extra",
        "missing-report",
        "flood",
    ],
)
def test_native_failures_never_return_a_partial_conversion(engine, monkeypatch, fault):
    monkeypatch.setenv("TILE_FAULT", fault)
    with pytest.raises((ValueError, RuntimeError)):
        quantize_tiles_native(np.zeros((1, 256)), 4, engine=engine, timeout=2)


def test_silent_child_is_reaped_before_timeout_returns(engine, tmp_path, monkeypatch):
    from mlxl3_quantizer import process as native_process

    launched = []
    original = subprocess.Popen

    def remember_child(*args, **kwargs):
        process = original(*args, **kwargs)
        launched.append(process.pid)
        return process

    monkeypatch.setattr(native_process.subprocess, "Popen", remember_child)
    monkeypatch.setenv("TILE_FAULT", "silent")
    with pytest.raises(TimeoutError):
        quantize_tiles_native(np.zeros((1, 256)), 4, engine=engine, timeout=0.15)
    assert len(launched) == 1
    with pytest.raises(ProcessLookupError):
        os.kill(launched[0], 0)


def test_existing_converter_cancellation_exception_reaps_native_child(
    engine, tmp_path, monkeypatch
):
    marker = tmp_path / "pid"
    monkeypatch.setenv("TILE_FAULT", "silent")
    monkeypatch.setenv("TILE_PID", str(marker))

    class Cancelled(Exception):
        pass

    def check_cancel():
        if marker.exists():
            raise Cancelled

    with pytest.raises(Cancelled):
        quantize_tiles_native(np.zeros((1, 256)), 4, engine=engine, cancel_check=check_cancel)
    with pytest.raises(ProcessLookupError):
        os.kill(int(marker.read_text()), 0)


@pytest.mark.parametrize(
    "values", [[], np.zeros(255), np.zeros((2, 128)), [np.inf] * 256, [np.nan] * 256]
)
def test_invalid_tile_arrays_do_not_start_a_native_process(engine, values, monkeypatch):
    monkeypatch.setenv("TILE_FAULT", "silent")
    with pytest.raises(ValueError):
        quantize_tiles_native(values, 4, engine=engine)


def test_launch_scratch_matches_independent_byte_counts_for_all_supported_k():
    for k in range(2, 9):
        edges = 2 ** (16 - k)
        memory = 256 * edges * 2 + 2 * edges * 2
        for cb in range(3):
            assert launch_tiles(768, k, cb, memory * 2) == 2
            with pytest.raises(ValueError):
                launch_tiles(256, k, cb, memory - 1)
