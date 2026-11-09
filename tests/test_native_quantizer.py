"""Native conversion process/data boundaries, independent of a physical GPU."""

import os
import subprocess
from pathlib import Path

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


WORKER = """#!/usr/bin/env python3
import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('command');p.add_argument('root');p.add_argument('--k',type=int);p.add_argument('--codebook',type=int);p.add_argument('--max-scratch-bytes',type=int);a=p.parse_args()
if a.command != 'quantize-worker': sys.exit(9)
fault=os.environ.get('TILE_FAULT','')
if os.environ.get('TILE_PID'):Path(os.environ['TILE_PID']).write_text(str(os.getpid()))
for line in sys.stdin:
    i=int(line);root=Path(a.root)/str(i)
    if fault=='silent' or (fault=='second-silent' and i==2):time.sleep(30)
    if fault=='exit':sys.exit(3)
    if fault=='invalid':print('oops',flush=True);continue
    if fault in ('flood','stderr-flood'):
        stream=sys.stderr if fault=='stderr-flood' else sys.stdout
        stream.write('x'*100000);stream.flush();time.sleep(30)
    raw=(root/'tiles.f32').read_bytes();n=len(raw)//4;tiles=n//256
    stage=root/'stage';stage.mkdir()
    (stage/'decoded.f32').write_bytes(raw)
    (stage/'states.u16').write_bytes(bytes(n*2))
    (stage/'packed.u16').write_bytes(bytes(tiles*16*a.k*2))
    r={'schema':1,'status':'complete','tiles':tiles,'values':n,'k':a.k,'codebook':a.codebook,'launch_tiles':min(tiles,a.max_scratch_bytes//(516*(1<<(16-a.k)))),'max_scratch_bytes':a.max_scratch_bytes}
    (stage/'report.json').write_text(json.dumps(r))
    if fault=='partial':(stage/'states.u16').write_bytes(b'')
    stage.rename(root/'result')
    print(json.dumps({'id':i+1 if fault=='foreign' else i,'report':r}),flush=True)
"""


@pytest.fixture
def worker_engine(tmp_path):
    path = tmp_path / "worker"
    path.write_text(WORKER)
    path.chmod(0o755)
    return path


@pytest.mark.parametrize("k", range(2, 9))
@pytest.mark.parametrize("cb", range(3))
def test_owned_worker_reuses_one_child_across_bounded_batches(worker_engine, monkeypatch, k, cb):
    import sys
    from pathlib import Path

    from mlxl3_quantizer import native, process

    vendor = Path(__file__).resolve().parents[1] / "apps/MLXL3Studio/Vendor/PonyExl3"
    monkeypatch.syspath_prepend(str(vendor))
    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", str(worker_engine))
    monkeypatch.setenv("MLXL3_NATIVE_CONVERTER_WORKER", "1")
    monkeypatch.setattr(native, "_CONVERTER_TILE_BATCH", 2)
    children = []
    original = subprocess.Popen

    def launch(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(process.subprocess, "Popen", launch)
    values = np.arange(5 * 256, dtype=np.float32).reshape(5, 256)[:, ::-1]
    decoded, states = native.converter_tiles(values, k, cb)
    np.testing.assert_array_equal(decoded, values)
    assert states.shape == values.shape and not states.any()
    assert len(children) == 1 and children[0].poll() is not None
    assert "mlx.core" not in sys.modules


@pytest.mark.parametrize(
    "fault", ["exit", "invalid", "foreign", "partial", "flood", "stderr-flood", "silent"]
)
def test_worker_failure_reaps_before_root_cleanup(worker_engine, monkeypatch, tmp_path, fault):
    from mlxl3_quantizer.process import TileWorker

    monkeypatch.setenv("TILE_FAULT", fault)
    root = tmp_path / "requests"
    root.mkdir()
    with (
        pytest.raises((ValueError, RuntimeError, TimeoutError)),
        TileWorker(
            [
                str(worker_engine),
                "quantize-worker",
                str(root),
                "--k",
                "4",
                "--codebook",
                "0",
                "--max-scratch-bytes",
                str(1 << 28),
            ]
        ) as worker,
    ):
        quantize_tiles_native(
            np.zeros((1, 256)), 4, engine=worker_engine, timeout=0.2, _worker=(worker, root)
        )
    assert worker.process.poll() is not None
    assert list(root.iterdir()) == []


def test_worker_cancellation_after_success_does_not_publish_partial_second_request(
    worker_engine, monkeypatch, tmp_path
):
    from mlxl3_quantizer.process import TileWorker

    monkeypatch.setenv("TILE_FAULT", "second-silent")

    class Cancelled(Exception):
        pass

    root = tmp_path / "requests"
    root.mkdir()
    with (
        pytest.raises(Cancelled),
        TileWorker(
            [
                str(worker_engine),
                "quantize-worker",
                str(root),
                "--k",
                "4",
                "--codebook",
                "0",
                "--max-scratch-bytes",
                str(1 << 28),
            ]
        ) as worker,
    ):
        quantize_tiles_native(np.zeros((1, 256)), 4, engine=worker_engine, _worker=(worker, root))

        def cancel():
            if worker.request_id == 2:
                raise Cancelled

        quantize_tiles_native(
            np.ones((1, 256)),
            4,
            engine=worker_engine,
            cancel_check=cancel,
            _worker=(worker, root),
        )
    assert worker.process.poll() is not None and list(root.iterdir()) == []


@pytest.mark.skipif(
    not os.environ.get("MLXL3_PHYSICAL_WORKER_ENGINE"),
    reason="explicit physical Apple GPU worker engine required",
)
def test_physical_worker_all_codebooks_match_independent_frozen_corpus(tmp_path):
    """Explicit hardware gate, not exercised by CPU/fake-worker success."""
    from mlxl3_quantizer.process import TileWorker

    engine = os.environ["MLXL3_PHYSICAL_WORKER_ENGINE"]
    corpus = Path(__file__).parent / "fixtures/native-quantizer"
    values = np.fromfile(corpus / "tiles.f32", "<f4").reshape(-1, 256)
    for k in range(2, 9):
        for cb in range(3):
            root = tmp_path / f"{k}-{cb}"
            root.mkdir()
            expected = np.fromfile(corpus / f"oracle-{k}-{cb}-decoded.f32", "<f4").reshape(
                values.shape
            )
            indices = np.fromfile(corpus / f"oracle-{k}-{cb}-states.u16", "<u2").reshape(
                values.shape
            )
            with TileWorker(
                [
                    engine,
                    "quantize-worker",
                    str(root),
                    "--k",
                    str(k),
                    "--codebook",
                    str(cb),
                    "--max-scratch-bytes",
                    str(1 << 28),
                ]
            ) as worker:
                for _ in range(2):
                    decoded, states = quantize_tiles_native(
                        values, k, cb, engine=engine, _worker=(worker, root)
                    )
                    np.testing.assert_array_equal(decoded.view(np.uint32), expected.view(np.uint32))
                    np.testing.assert_array_equal(states, indices)


def test_waiting_converter_lock_remains_cancellable():
    from mlxl3_quantizer import native

    class Cancelled(Exception):
        pass

    def cancel():
        raise Cancelled

    with native._CONVERTER_LOCK, pytest.raises(Cancelled), native._converter_lock(cancel):
        pytest.fail("contended lock was acquired")


def test_worker_startup_failure_is_fatal_and_next_call_recovers(worker_engine, monkeypatch):
    from mlxl3_quantizer import native

    vendor = Path(__file__).resolve().parents[1] / "apps/MLXL3Studio/Vendor/PonyExl3"
    monkeypatch.syspath_prepend(str(vendor))
    monkeypatch.setenv("MLXL3_NATIVE_CONVERTER_WORKER", "1")
    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", str(worker_engine.with_name("missing")))
    with pytest.raises(RuntimeError, match="native converter service failed"):
        native.converter_tiles(np.zeros((1, 256)), 4)
    monkeypatch.setenv("MLXL3_NATIVE_ENGINE", str(worker_engine))
    worker_engine.chmod(0o644)
    with pytest.raises(RuntimeError, match="worker launch failed"):
        native.converter_tiles(np.zeros((1, 256)), 4)
    worker_engine.chmod(0o755)
    actual, _ = native.converter_tiles(np.ones((1, 256)), 4)
    assert np.all(actual == 1)
