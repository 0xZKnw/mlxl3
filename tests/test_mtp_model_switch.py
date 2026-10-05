"""Exercise the MTP lifecycle verifier's real CLI, failure reports and cleanup."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check-mtp-model-switch.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("mtp_switch", SCRIPT)
switch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(switch)

FAKE = r"""#!/usr/bin/env python3
import json, os, sys, time
case = os.environ['MTP_SWITCH_CASE']
with open(os.environ['MTP_SWITCH_PIDS'], 'a') as f: f.write(str(os.getpid()) + '\n')
def emit(kind, **kw): print(json.dumps(dict(type=kind, **kw)), flush=True)
if sys.argv[1] == 'mtp-head':
    if case == 'download_error': sys.exit(2)
    target = sys.argv[sys.argv.index('--target')+1]
    emit('installed', path='/heads/same' if case == 'same_heads' else '/heads/'+target)
    sys.exit(0)
model = sys.argv[sys.argv.index('bridge')+1]
active = 100
if case == 'silent': time.sleep(10)
if case == 'invalid_json': print('not-json', flush=True); time.sleep(10)
emit('ready', mtp_auto_download_supported=True, mtp_configure_supported=case != 'missing_caps')
for line in sys.stdin:
    r = json.loads(line)
    if case == 'stream':
        while True:
            emit('progress', request_id=r['request_id']); time.sleep(0.001)
    if r['type'] == 'set_mtp':
        wrong = r['enabled'] and r['mtp_head_path'] != '/heads/'+model
        if wrong:
            active = 100 if case != 'retained_failure' else active
            emit('error', request_id=r['request_id'], mtp_active=False, memory={'mlx_active_bytes':active})
        else:
            previous = active
            active = 300 if r['enabled'] else 100
            if case == 'reuse_growth' and previous == 300 and r['enabled']: active = 400
            if case == 'retained_off' and not r['enabled'] and previous == 300: active = 300
            kw = {} if case == 'missing_memory' and r['enabled'] else {'memory':{'mlx_active_bytes':active}}
            emit('mtp_status', request_id=r['request_id'], mtp_active=r['enabled'], mtp_head_path=r['mtp_head_path'], **kw)
    else:
        if case == 'cancelled': emit('cancelled', request_id=r['request_id']); continue
        count = min(2,r['max_tokens']) if case != 'empty' else 0
        h = 'diverged' if case == 'mismatch' and r['mtp_depth'] == 2 else 'hash'+str(count)
        emit('complete', request_id=r['request_id'], token_hash=h, cache_context='history', stats={'generated_tokens':count})
"""


@pytest.mark.parametrize(
    "case",
    [
        "success",
        "download_error",
        "same_heads",
        "missing_caps",
        "silent",
        "stream",
        "invalid_json",
        "retained_failure",
        "retained_off",
        "reuse_growth",
        "missing_memory",
        "cancelled",
        "empty",
        "mismatch",
    ],
)
def test_production_verifier_reports_failures_and_reaps_processes(tmp_path, case):
    fake = tmp_path / "engine"
    fake.write_text(FAKE)
    fake.chmod(0o755)
    pids = tmp_path / "pids"
    output = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(fake),
            "dense",
            "moe",
            "--timeout",
            "0.4",
            "--output",
            str(output),
        ],
        env={**os.environ, "MTP_SWITCH_CASE": case, "MTP_SWITCH_PIDS": str(pids)},
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    report = json.loads(output.read_text())
    assert (result.returncode == 0) == (case == "success"), result.stderr
    assert report["status"] == ("passed" if case == "success" else "failed")
    assert report["parity"] is (case == "success")
    if case == "success":
        assert [item["model"] for item in report["models"]] == ["dense", "moe", "dense"]
        assert all(len(item["cases"]) == 12 for item in report["models"])
    for pid in map(int, pids.read_text().splitlines()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("count", [None, False, True, -1, 0, 1, 3, 17, 2**63, 1.5])
@pytest.mark.parametrize("budget", [-1, 0, 1, 3, 17, 2**64])
def test_parity_requires_real_nonempty_bounded_completions(count, budget):
    complete = {
        "token_hash": "hash",
        "cache_context": "history",
        "stats": {"generated_tokens": count},
    }
    assert switch.matching_completions(complete, complete, budget) is (
        type(count) is int and 0 < count <= budget
    )
    for changed in [
        {"token_hash": "different"},
        {"token_hash": ""},
        {"cache_context": "different"},
        {"cache_context": ""},
        {"stats": None},
        {"stats": {}},
        {"stats": {"generated_tokens": -1}},
    ]:
        assert not switch.matching_completions(complete, {**complete, **changed}, budget)
