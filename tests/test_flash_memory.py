"""Disposable processes exercise the actual RAM campaign and its failure states."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "flash_memory", Path(__file__).parents[1] / "benchmarks/benchmark_flash_memory.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize("size", [4096, 16384, 65536])
def test_vm_units_and_invalid_counters(size):
    raw = f"Mach Virtual Memory Statistics: (page size of {size} bytes)\n"
    raw += "\n".join(f"{key}: {i}." for i, key in enumerate(bench.VM_KEYS))
    assert list(bench.parse_vm(raw).values()) == [0, size, 2 * size, 3 * size, 4 * size]
    for broken in (
        "",
        raw.replace(str(size), "0", 1),
        raw.replace("Swapins:", "Wrong:"),
        raw + "\nSwapins: 3.",
        raw.replace("Swapins: 3.", "Swapins: -3."),
        raw.replace("Swapins: 3.", "Swapins: nan."),
    ):
        with pytest.raises(ValueError):
            bench.parse_vm(broken)


@pytest.mark.parametrize(
    "fault",
    [
        "ok",
        "silent",
        "crash",
        "invalid",
        "mode",
        "short",
        "empty",
        "drift",
        "memory",
        "identity",
        "sample",
        "interrupt",
        "exit",
    ],
)
def test_campaign(tmp_path, fault):
    engine = tmp_path / "engine"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        + f"fault = {fault!r}\n"
        + """
import json, sys, time
mem = dict(mlx_active_bytes=100, mlx_peak_bytes=120, mlx_cache_bytes=0, process_footprint_bytes=130)
print('{"type":"ready"}', flush=True)
for line in sys.stdin:
    r = json.loads(line)
    if fault == 'silent': time.sleep(30)
    if fault == 'crash': sys.exit(3)
    if fault == 'invalid': print('null', flush=True); continue
    if r['type'] == 'ping':
        print(json.dumps(dict(type='pong', request_id=r['request_id'], memory=mem)), flush=True)
        continue
    if fault != 'mode': print(json.dumps(dict(type='generation_mode', request_id=r['request_id'], mtp_active=r['mtp'], dflash_active=False)), flush=True)
    stats = dict(generated_tokens=2, decode_tokens=1, decode_seconds=.1, decode_tps=10, ttft_seconds=.1, prefill_tps=100, memory=mem)
    if r['mtp']: stats.update(mtp_proposed_tokens=0, mtp_accepted_tokens=0, mtp_blocks=0)
    if fault == 'short': stats['generated_tokens'] = 1
    if fault == 'memory': stats['memory'] = {}
    e = dict(type='complete', request_id=r['request_id'], token_hash='abc', cache_context='answer', stats=stats)
    if fault == 'identity': e['request_id'] = 'wrong'
    if fault == 'empty': e['cache_context'] = ''
    if fault == 'drift' and r['mtp']: e['token_hash'] = 'different'
    print(json.dumps(e), flush=True)
if fault == 'exit': sys.exit(3)
"""
    )
    engine.chmod(0o755)
    output = tmp_path / "results"
    calls = 0

    def sample():
        nonlocal calls
        calls += 1
        if fault == "interrupt" and calls == 3:
            raise KeyboardInterrupt()
        return {"bytes": {} if fault == "sample" and calls > 1 else dict.fromkeys(bench.VM_KEYS, 0)}

    if fault == "ok":
        report = bench.run(engine, tmp_path, output, tokens=2, timeout=2, sample=sample)
        assert report["status"] == "validated" and report["parity"] and report["reaped"]
        assert len(report["runs"]) == 4
        assert len(report["samples"]) >= 6
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError, KeyboardInterrupt)):
            bench.run(engine, tmp_path, output, tokens=2, timeout=0.3, sample=sample)
        report = json.loads((output / "report.json").read_text())
        assert report["status"] == "failed" and not report["parity"] and report["reaped"]
        assert report["error"]


@pytest.mark.parametrize(
    "options",
    [
        {"tokens": 1},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"cache_gib": 0.5},
        {"cache_gib": 13},
        {"cache_gib": float("inf")},
    ],
)
def test_invalid_protocol(tmp_path, options):
    with pytest.raises(ValueError):
        bench.run(tmp_path, tmp_path, tmp_path, **options)
