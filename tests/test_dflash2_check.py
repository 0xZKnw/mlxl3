"""Exercise the production checker, including interrupted and dishonest bridges."""

import copy
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_dflash2 import equal, run


def completion():
    return {
        "token_hash": "0123456789abcdef",
        "cache_context": "nonempty-history",
        "stats": {
            "generated_tokens": 17,
            "decode_seconds": 0.1,
            "ttft_seconds": 0.1,
            "elapsed_seconds": 0.2,
        },
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("token_hash", "fedcba9876543210"),
        ("token_hash", ""),
        ("cache_context", "different"),
        ("cache_context", ""),
        ("generated_tokens", 0),
        ("generated_tokens", True),
        ("generated_tokens", 18),
        ("decode_seconds", float("nan")),
        ("ttft_seconds", float("inf")),
        ("elapsed_seconds", -1),
        ("elapsed_seconds", True),
    ],
)
def test_parity_requires_nonempty_finite_equal_ids_and_history(field, value):
    reference = completion()
    equal(reference, copy.deepcopy(reference))
    candidate = copy.deepcopy(reference)
    target = candidate if field in candidate else candidate["stats"]
    target[field] = value
    with pytest.raises(AssertionError):
        equal(reference, candidate)


def write_bridge(path, behavior):
    path.write_text(
        f"#!{sys.executable}\n"
        + f"behavior = {behavior!r}\n"
        + """
import io, json, os, signal, sys, time
from pathlib import Path
Path(__file__).with_suffix('.pid').write_text(str(os.getpid()))
inject_signal = False
if behavior == 'cancel_during_write':
    original_raw = sys.stdout.buffer.raw
    class SignalRaw(io.RawIOBase):
        def writable(self): return True
        def write(self, value):
            global inject_signal
            if inject_signal:
                inject_signal = False
                os.kill(os.getpid(), signal.SIGUSR1)
            return original_raw.write(value)
    sys.stdout = io.TextIOWrapper(io.BufferedWriter(SignalRaw()))
def emit(kind, **fields):
    global inject_signal
    inject_signal = behavior == 'cancel_during_write' and kind == 'context_usage'
    print(json.dumps(dict(type=kind, **fields)), flush=True)
if behavior == 'silent': time.sleep(60)
if behavior == 'invalid': print('not JSON', flush=True); time.sleep(60)
if behavior == 'eof': sys.exit(0)
emit('ready', dflash_supported=behavior != 'unsupported', dflash_tune_supported=True)
cancelled = False
rid = None
def cancel(*_):
    global cancelled
    cancelled = True
signal.signal(signal.SIGUSR1, cancel)
for line in sys.stdin:
    request = json.loads(line)
    rid = request['request_id']
    if request['type'] == 'set_dflash':
        if request['dflash_draft_path'].endswith(('missing', 'foreign')):
            emit('error', request_id=rid, message='missing draft')
        else: emit('dflash_status', request_id=rid, dflash_active=request['enabled'])
        continue
    if request['dflash2'] and (request['dflash_mode'] not in (1, 2, 3)
            or request.get('mtp') or request['repetition_penalty'] != 1):
        emit('error', request_id=rid, message='incompatible request')
        continue
    if request['max_tokens'] == 256:
        cancelled = False
        emit('context_usage', request_id=rid)
        time.sleep(.05)
        if not cancelled: emit('delta', request_id=rid, text='token')
        time.sleep(.1)
        if cancelled:
            emit('cancelled', request_id=rid)
            continue
    if behavior == 'request_silent': time.sleep(60)
    stats = dict(generated_tokens=request['max_tokens'], decode_seconds=.1,
                 ttft_seconds=.1, elapsed_seconds=.2,
                 cached_prompt_tokens=256 if request.get('reuse_prompt_cache')
                     and not request.get('memory_saver')
                     and request['conversation_id'] == 'dflash2-check' else 0)
    if behavior == 'empty': stats['generated_tokens'] = 0
    if behavior == 'nonfinite': stats['elapsed_seconds'] = float('nan')
    token_hash = 'fedcba9876543210' if behavior == 'divergent' and request['dflash2'] else '0123456789abcdef'
    emit('complete', request_id=rid, token_hash=token_hash,
         cache_context='nonempty-history', stats=stats)
"""
    )
    path.chmod(0o755)


@pytest.mark.parametrize(
    "behavior",
    [
        "valid",
        "cancel_during_write",
        "divergent",
        "empty",
        "nonfinite",
        "unsupported",
        "silent",
        "request_silent",
        "invalid",
        "eof",
    ],
)
def test_checker_bounds_failures_preserves_report_and_reaps_child(tmp_path, behavior):
    engine, output = tmp_path / "engine", tmp_path / "report.json"
    write_bridge(engine, behavior)
    if behavior in ("valid", "cancel_during_write"):
        result = run(
            str(engine),
            "model",
            "draft",
            timeout=5,
            tune=False,
            output=output,
            foreign_draft="foreign",
        )
        assert result["status"] == "complete" and result["parity"] is True
        assert result["checks"] == 43
    else:
        with pytest.raises((AssertionError, RuntimeError, TimeoutError)):
            run(str(engine), "model", "draft", timeout=0.5, tune=False, output=output)
        result = json.loads(output.read_text())
        assert result["status"] == "failed" and result["parity"] is None
        assert result["error"]
    pid = int(engine.with_suffix(".pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
