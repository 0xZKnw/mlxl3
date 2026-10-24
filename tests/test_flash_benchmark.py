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
        "df_ok",
        "df_mode",
        "df_hash",
        "df_empty",
        "df_timing",
        "df_short",
        "df_drift",
        "df_invalid",
        "df_silent",
        "df_crash",
        "df_counter_range",
        "df_counter_type",
        "df_missing_counter",
        "late_status",
        "late_error",
        "late_invalid",
        "late_null",
        "late_missing_type",
        "late_bad_type",
        "late_empty_type",
        "late_array",
        "late_nonzero",
        "eof",
        "error",
        "malformed",
        "df_late_error",
        "df_late_invalid",
        "df_late_nonzero",
        "compare_late_error",
        "compare_late_invalid",
        "compare_late_nonzero",
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
fault = fault.removeprefix('df_')
if fault.startswith('compare_late_'): fault = fault.removeprefix('compare_')
print(json.dumps({"type": "ready", "expert_cache": os.environ.get('MLXL3_SSD_CACHE_GIB')}), flush=True)
counter = 0
for line in sys.stdin:
    r = json.loads(line)
    if r['type'] == 'shutdown':
        if fault == 'late_error': print('{"type":"error","message":"late SSD read failure"}', flush=True)
        if fault == 'late_invalid': print('invalid json', flush=True)
        if fault == 'late_null': print('null', flush=True)
        if fault == 'late_missing_type': print('{}', flush=True)
        if fault == 'late_bad_type': print('{"type":null}', flush=True)
        if fault == 'late_empty_type': print('{"type":""}', flush=True)
        if fault == 'late_array': print('[]', flush=True)
        if fault == 'late_status': print('{"type":"status","message":"closed"}', flush=True)
        sys.exit(7 if fault == 'late_nonzero' else 0)
    counter += 1
    if fault == 'silent': time.sleep(30)
    if fault == 'crash': sys.exit(3)
    if fault == 'eof': sys.exit(0)
    if fault == 'error': print('{"type":"error","message":"read failed"}', flush=True); continue
    if fault == 'malformed': print('invalid json', flush=True); continue
    if fault == 'invalid': print('null', flush=True); continue
    if fault != 'mode': print(json.dumps({'type': 'generation_mode', 'request_id': r['request_id'], 'mtp_active': r['mtp'], 'dflash_active': r.get('dflash2',False)}), flush=True)
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
    if r.get('dflash2'):
        assert os.path.isdir(r['dflash_draft_path']) and r['dflash_mode'] == 1 and not r['mtp']
        e['stats'].update(dflash_proposed_tokens=7, dflash_accepted_tokens=6, dflash_blocks=1)
        if fault == 'counter_range': e['stats']['dflash_accepted_tokens'] = 8
        if fault == 'counter_type': e['stats']['dflash_blocks'] = True
        if fault == 'missing_counter': del e['stats']['dflash_blocks']
    print(json.dumps(e), flush=True)
"""
    )
    engine.chmod(0o755)
    output = tmp_path / "results"
    if fault.startswith("df_"):
        if fault == "df_ok":
            result = bench.run(
                engine, tmp_path, output, tokens=8, repeats=2, timeout=2, dflash_draft=tmp_path
            )
            assert result["status"] == "validated" and result["mode"] == "dflash2"
            assert len(result["runs"]) == 4 and len(result["warmup"]) == 2
            assert result["median_decode_tps"] == {"MAIN": 10, "HOLD": 10}
            assert all(r["mode"] == "dflash2" for r in result["runs"])
            assert all(
                r["generation_mode"]["dflash_active"] is True
                for r in result["runs"] + result["warmup"]
            )
            assert result["reaped"] and result["reader_joined"]
        else:
            with pytest.raises((ValueError, RuntimeError, TimeoutError)):
                bench.run(
                    engine,
                    tmp_path,
                    output,
                    tokens=8,
                    repeats=2,
                    timeout=0.5,
                    dflash_draft=tmp_path,
                )
            result = json.loads((output / "report.json").read_text())
            assert (
                result["status"] == "failed"
                and result["error"]
                and "median_decode_tps" not in result
            )
            assert result["reaped"] and result["reader_joined"]
        return
    if fault in ("ok", "auto_cache", "fixed_cache", "one_prompt", "late_status"):
        if fault == "ok":
            original_validate = bench.validate

            def legacy_validate(event, mode, mtp, budget):
                return original_validate(event, mode, mtp, budget)

            monkeypatch.setattr(bench, "validate", legacy_validate)
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
        assert all(
            r["generation_mode"]["dflash_active"] is False
            for r in result["runs"] + result["warmup"]
        )
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
        assert result["reaped"] and result["reader_joined"]


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


@pytest.mark.parametrize(
    "options", [{"mtp": True}, {"compare": True}, {"dflash_mode": 0}, {"dflash_mode": True}]
)
def test_dflash_rejects_invalid_modes_before_starting_a_child(tmp_path, options):
    with pytest.raises(ValueError):
        bench.run(tmp_path, tmp_path, tmp_path, dflash_draft=tmp_path, **options)
    assert not (tmp_path / "report.json").exists()
