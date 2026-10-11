"""Exercise the real bounded benchmark entry with disposable bridge processes."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "flash_benchmark", Path(__file__).parents[1] / "scripts/benchmark-flash-bridge.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize(
    "fault",
    ["ok", "mode", "hash", "empty", "timing", "short", "drift", "invalid", "silent", "crash"],
)
def test_campaign(tmp_path, fault):
    engine = tmp_path / "bridge"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        + f"fault = {fault!r}\n"
        + """
import json, sys, time
print(json.dumps({"type": "ready"}), flush=True)
counter = 0
for line in sys.stdin:
    r = json.loads(line)
    if r['type'] == 'shutdown': break
    counter += 1
    if fault == 'silent': time.sleep(30)
    if fault == 'crash': sys.exit(3)
    if fault == 'invalid': print('null', flush=True); continue
    if fault != 'mode': print(json.dumps({'type': 'generation_mode', 'request_id': r['request_id'], 'mtp_active': False, 'dflash_active': False}), flush=True)
    e = {'type': 'complete', 'request_id': r['request_id'], 'token_hash': 'abc', 'cache_context': 'answer',
         'stats': {'generated_tokens': 2, 'decode_tokens': 1, 'decode_seconds': .1, 'decode_tps': 10, 'ttft_seconds': .1, 'prefill_tps': 100}}
    if fault == 'hash': del e['token_hash']
    if fault == 'empty': e['cache_context'] = ''
    if fault == 'timing': e['stats']['decode_tps'] = float('nan')
    if fault == 'short': e['stats']['generated_tokens'] = 1
    if fault == 'drift' and counter > 2: e['token_hash'] = 'different'
    print(json.dumps(e), flush=True)
"""
    )
    engine.chmod(0o755)
    output = tmp_path / "results"
    if fault == "ok":
        result = bench.run(engine, tmp_path, output, tokens=2, repeats=1, timeout=2)
        assert result["status"] == "validated"
        assert len(result["runs"]) == len(result["warmup"]) == 2
        assert result["median_decode_tps"] == {"MAIN": 10, "HOLD": 10}
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            bench.run(engine, tmp_path, output, tokens=2, repeats=1, timeout=0.5)
        result = json.loads((output / "report.json").read_text())
        assert result["status"] == "failed" and result["error"]
        assert "median_decode_tps" not in result


@pytest.mark.parametrize(
    "tokens,repeats,timeout", [(1, 1, 1), (2, 0, 1), (2, 1, 0), (2, 1, float("nan"))]
)
def test_invalid_protocol(tmp_path, tokens, repeats, timeout):
    with pytest.raises(ValueError):
        bench.run(tmp_path, tmp_path, tmp_path, tokens=tokens, repeats=repeats, timeout=timeout)
