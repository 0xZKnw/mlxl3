"""Exercise the real bounded benchmark entry with disposable bridge processes."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "flash_benchmark", Path(__file__).parents[1] / "scripts/benchmark-flash-bridge.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


@pytest.mark.parametrize(
    "fault",
    [
        "ok",
        "auto_cache",
        "fixed_cache",
        "one_prompt",
        "mode",
        "hash",
        "empty",
        "timing",
        "short",
        "drift",
        "invalid",
        "silent",
        "crash",
        "compare_drift",
        "counter_type",
        "counter_range",
        "negative_counter",
        "missing_counter",
    ],
)
def test_campaign(tmp_path, fault, monkeypatch):
    monkeypatch.setenv("MLXL3_SSD_CACHE_GIB", "12")
    engine = tmp_path / "bridge"
    engine.write_text(
        "#!/usr/bin/env python3\n"
        + f"fault = {fault!r}\n"
        + """
import json, os, sys, time
print(json.dumps({"type": "ready", "expert_cache": os.environ.get('MLXL3_SSD_CACHE_GIB')}), flush=True)
counter = 0
for line in sys.stdin:
    r = json.loads(line)
    if r['type'] == 'shutdown': break
    counter += 1
    if fault == 'silent': time.sleep(30)
    if fault == 'crash': sys.exit(3)
    if fault == 'invalid': print('null', flush=True); continue
    if fault != 'mode': print(json.dumps({'type': 'generation_mode', 'request_id': r['request_id'], 'mtp_active': r['mtp'], 'dflash_active': False}), flush=True)
    tokens = r['max_tokens']
    e = {'type': 'complete', 'request_id': r['request_id'], 'token_hash': 'abc', 'cache_context': 'answer',
         'stats': {'generated_tokens': tokens, 'decode_tokens': tokens - 1, 'decode_seconds': (tokens - 1) / 10, 'decode_tps': 10, 'ttft_seconds': .1, 'prefill_tps': 100}}
    if fault == 'hash': del e['token_hash']
    if fault == 'empty': e['cache_context'] = ''
    if fault == 'timing': e['stats']['decode_tps'] = float('nan')
    if fault == 'short': e['stats']['generated_tokens'] = 1
    if fault == 'drift' and counter > 2: e['token_hash'] = 'different'
    if r['mtp']:
        e['stats'].update(mtp_proposed_tokens=0, mtp_accepted_tokens=0, mtp_blocks=0)
        if fault == 'compare_drift': e['token_hash'] = 'different'
        if fault == 'counter_type': e['stats']['mtp_blocks'] = True
        if fault == 'counter_range': e['stats']['mtp_accepted_tokens'] = 1
        if fault == 'negative_counter': e['stats']['mtp_blocks'] = -1
        if fault == 'missing_counter': del e['stats']['mtp_blocks']
    print(json.dumps(e), flush=True)
"""
    )
    engine.chmod(0o755)
    output = tmp_path / "results"
    if fault in ("ok", "auto_cache", "fixed_cache", "one_prompt"):
        automatic = fault == "auto_cache"
        budget = 4.75 if fault == "fixed_cache" else 8
        result = bench.run(
            engine,
            tmp_path,
            output,
            tokens=2,
            repeats=1,
            timeout=2,
            auto_cache=automatic,
            cache_gib=budget,
        )
        assert result["status"] == "validated"
        assert result["runtime"]["expert_cache"] == (None if automatic else str(budget))
        assert result["expert_cache_gib"] == (None if automatic else budget)
        assert len(result["runs"]) == len(result["warmup"]) == 2
        assert result["median_decode_tps"] == {"MAIN": 10, "HOLD": 10}
        if fault == "one_prompt":
            selected = bench.run(
                engine,
                tmp_path,
                tmp_path / "selected",
                tokens=2,
                repeats=1,
                timeout=2,
                prompt="HOLD",
                mtp=True,
            )
            assert selected["status"] == "validated"
            assert selected["prompts"] == {"HOLD": bench.PROMPTS["HOLD"]}
            assert selected["median_decode_tps"] == {"HOLD": 10}
            assert len(selected["runs"]) == len(selected["warmup"]) == 1
            assert selected["runs"][0]["mode"] == "mtp"
            subprocess.run(
                [
                    sys.executable,
                    spec.origin,
                    str(engine),
                    str(tmp_path),
                    str(tmp_path / "selected-cli"),
                    "--prompt",
                    "MAIN",
                    "--repeats",
                    "1",
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
            cli = json.loads((tmp_path / "selected-cli/report.json").read_text())
            assert cli["status"] == "validated" and cli["prompts"] == {
                "MAIN": bench.PROMPTS["MAIN"]
            }
            assert len(cli["runs"]) == len(cli["warmup"]) == 1
        comparison = bench.run(
            engine, tmp_path, tmp_path / "comparison", tokens=2, repeats=2, timeout=2, compare=True
        )
        assert len(comparison["runs"]) == 8 and len(comparison["warmup"]) == 4
        assert comparison["median_decode_tps"] == {
            "baseline": {"MAIN": 10, "HOLD": 10},
            "mtp": {"MAIN": 10, "HOLD": 10},
        }
        assert [r["mode"] for r in comparison["runs"]] == [
            "mtp",
            "baseline",
            "mtp",
            "baseline",
            "baseline",
            "mtp",
            "baseline",
            "mtp",
        ]
        if automatic or fault == "fixed_cache":
            cli_output = tmp_path / "cli"
            subprocess.run(
                [
                    sys.executable,
                    spec.origin,
                    str(engine),
                    str(tmp_path),
                    str(cli_output),
                    "--compare",
                    *(["--auto-cache"] if automatic else ["--cache-gib", str(budget)]),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            cli = json.loads((cli_output / "report.json").read_text())
            assert cli["status"] == "validated" and len(cli["runs"]) == 12
            assert cli["runtime"]["expert_cache"] == (None if automatic else str(budget))
            assert cli["expert_cache_gib"] == (None if automatic else budget)
    else:
        with pytest.raises((ValueError, RuntimeError, TimeoutError)):
            bench.run(
                engine,
                tmp_path,
                output,
                tokens=2,
                repeats=1,
                timeout=0.5,
                compare=fault.startswith(("compare", "counter", "negative", "missing")),
            )
        result = json.loads((output / "report.json").read_text())
        assert result["status"] == "failed" and result["error"]
        assert "median_decode_tps" not in result


@pytest.mark.parametrize(
    "tokens,repeats,timeout", [(1, 1, 1), (2, 0, 1), (2, 1, 0), (2, 1, float("nan"))]
)
def test_invalid_protocol(tmp_path, tokens, repeats, timeout):
    with pytest.raises(ValueError):
        bench.run(tmp_path, tmp_path, tmp_path, tokens=tokens, repeats=repeats, timeout=timeout)


@pytest.mark.parametrize("budget", [0, 0.59, 12.01, float("inf"), float("nan")])
def test_invalid_cache_budget(tmp_path, budget):
    with pytest.raises(ValueError, match="expert cache"):
        bench.run(tmp_path, tmp_path, tmp_path, cache_gib=budget)


@pytest.mark.parametrize("prompt", ["", "missing", "main"])
def test_invalid_prompt_selection(tmp_path, prompt):
    with pytest.raises(ValueError, match="unknown benchmark prompt"):
        bench.run(tmp_path, tmp_path, tmp_path, prompt=prompt)


def test_mutually_exclusive_cache_modes(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            spec.origin,
            *([str(tmp_path)] * 3),
            "--auto-cache",
            "--cache-gib",
            "4.75",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 2 and "not allowed" in result.stderr
