"""The real MTP checker must reject broken responses and clean up its bridge."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "mtp_bridge", Path(__file__).parents[1] / "scripts/check-mtp-bridge.py"
)
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


@pytest.mark.parametrize(
    "fault",
    [
        "ok",
        "invalid",
        "malformed",
        "silent",
        "endless",
        "crash",
        "mode",
        "empty",
        "hash",
        "count",
        "divergence",
        "accept-invalid",
        "cancel-corruption",
        "configure",
    ],
)
@pytest.mark.parametrize("kind", ["mtp", "dflash"])
def test_bridge_boundaries_and_cleanup(tmp_path, capsys, monkeypatch, fault, kind):
    engine = tmp_path / "engine"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        + f"fault = {fault!r}\nkind = {kind!r}\n"
        + """
import json, signal, sys, time
cancelled = False
corrupt = False
def cancel(*args):
    global cancelled
    cancelled = True
signal.signal(signal.SIGUSR1, cancel)
def emit(**event): print(json.dumps(event), flush=True)
emit(type='ready', mtp_max_depth=1, **{f'{kind}_configure_supported': fault in ('ok', 'configure')})
seen = set()
for line in sys.stdin:
    r = json.loads(line)
    if r['type'] == 'shutdown': break
    cancelled = False
    if fault == 'silent': time.sleep(5)
    if fault == 'endless':
        while True:
            emit(type='heartbeat')
            time.sleep(.01)
    if fault == 'crash': sys.exit(3)
    if fault == 'invalid': print('null', flush=True); continue
    if fault == 'malformed': print('{bad', flush=True); continue
    rid = r['request_id']
    if rid in seen:
        emit(type='error', request_id=rid, message='duplicate request ID')
        continue
    seen.add(rid)
    if r['type'] == f'set_{kind}':
        emit(type=f'{kind}_status', request_id=rid, **{f'{kind}_active': not r['enabled'] if fault == 'configure' else r['enabled']})
        continue
    path_field = 'mtp_head_path' if kind == 'mtp' else 'dflash_draft_path'
    requested = 'mtp' if kind == 'mtp' else 'dflash2'
    invalid = (not r[path_field] or r[path_field] == '/missing-head' or
               r.get('dflash2' if kind == 'mtp' else 'mtp') or
               (r.get('mtp_depth', 1) != 1 if kind == 'mtp' else
                r.get('dflash_mode', 1) not in (1,2,3) or r['repetition_penalty'] != 1))
    if invalid and fault != 'accept-invalid':
        emit(type='error', request_id=rid, message='rejected')
        continue
    active = r[requested] and r['repetition_penalty'] == 1
    if fault != 'mode': emit(type='generation_mode', request_id=rid, **{f'{kind}_active': active})
    emit(type='context_usage', request_id=rid, used_tokens=3)
    if not cancelled:
        emit(type='delta', request_id=rid, text='' if fault == 'empty' else 'answer')
    if r['max_tokens'] == 256:
        deadline = time.monotonic() + 1
        while not cancelled and time.monotonic() < deadline:
            time.sleep(.005)
    if cancelled:
        corrupt = True
        emit(type='cancelled', request_id=rid)
        continue
    emit(type='complete', request_id=rid, token_hash='' if fault == 'hash' else 'different' if (fault == 'divergence' and active) or (fault == 'cancel-corruption' and corrupt) else 'abc', cache_context='' if fault == 'empty' else 'answer', stats={'generated_tokens': 0 if fault == 'count' else r['max_tokens']})
"""
    )
    engine.chmod(0o755)
    processes = []
    original = bridge.subprocess.Popen

    def tracked(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(bridge.subprocess, "Popen", tracked)
    if fault == "ok":
        bridge.run(str(engine), "model", "model", [1, 2, 3], "prompt", timeout=2, mode=kind)
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            bridge.run(str(engine), "model", "model", [1, 2], "prompt", timeout=0.2, mode=kind)
    events = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert events[-1]["status"] == ("validated" if fault == "ok" else "failed")
    assert not any(event.get("status") == "validated" for event in events[:-1])
    assert len(processes) == 1 and processes[0].poll() is not None
    assert processes[0].stdin.closed and processes[0].stdout.closed


@pytest.mark.parametrize(
    "budgets,context,timeout",
    [([], 4, 1), ([0], 4, 1), ([True], 4, 1), ([1], 0, 1), ([1], 4, 0), ([1], 4, float("nan"))],
)
def test_invalid_protocol(budgets, context, timeout):
    with pytest.raises(ValueError):
        bridge.run("missing", "model", "head", budgets, "prompt", context=context, timeout=timeout)


def test_unknown_speculative_backend_fails_before_starting_a_process():
    with pytest.raises(ValueError):
        bridge.run("missing", "model", "head", [1], "prompt", mode="other")
