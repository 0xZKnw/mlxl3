"""The pool speed driver must reject divergence before recording validation."""

import copy
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = runpy.run_path(str(ROOT / "docs/measurements/issue58-20261029/pool-speed-driver.py"))
GOLD = json.loads((ROOT / "docs/measurements/sushi-20261027/gold-AR-MAIN.json").read_text())


def event(budget):
    return {
        "token_hash": "4b46ba2824c9f47c" if budget == 17 else GOLD["token_hash"],
        "cache_context": GOLD["cache_context"][:30] if budget == 17 else GOLD["cache_context"],
        "stats": {
            "memory": {
                "process_footprint_bytes": 5_000_000_000,
                "process_lifetime_peak_bytes": 6_000_000_000,
            }
        },
    }


@pytest.mark.parametrize("budget", [17, 128])
def test_external_oracle_accepts_nonempty_complete_outputs(budget):
    # The 128-token hash was recorded by a separate native generation.
    assert DRIVER["token_hash"](GOLD["tokens"]) == "f6b5f974c6db2e7a"
    DRIVER["check_golden"](event(budget), GOLD, budget)


@pytest.mark.parametrize("fault", range(8))
def test_external_oracle_rejects_divergence_and_invalid_memory(fault):
    value, golden = event(128), copy.deepcopy(GOLD)
    if fault == 0:
        value["token_hash"] = "wrong"
    elif fault == 1:
        value["cache_context"] = ""
    elif fault == 2:
        value["cache_context"] = golden["cache_context"][:30]
    elif fault == 3:
        golden["tokens"] = []
    elif fault == 4:
        golden["tokens"][0] = True
    elif fault == 5:
        golden["token_hash"] = "wrong"
    else:
        value["stats"]["memory"]["process_footprint_bytes"] = (0, 12_000_000_000)[fault - 6]
    with pytest.raises(AssertionError):
        DRIVER["check_golden"](value, golden, 128)


@pytest.mark.parametrize("diverge", [False, True])
def test_driver_reaches_real_runner_and_records_failure_without_validation(tmp_path, diverge):
    engine = tmp_path / "bridge"
    complete = event(17)
    complete["stats"].update(
        generated_tokens=17,
        decode_tokens=16,
        decode_tps=16.0,
        decode_seconds=1.0,
        ttft_seconds=1.0,
        prefill_tps=80.0,
    )
    if diverge:
        complete["token_hash"] = "wrong"
    engine.write_text(
        "#!/usr/bin/env python3\nimport json,sys\n"
        + f"complete={complete!r}\n"
        + "print(json.dumps({'type':'ready'}),flush=True)\n"
        + "for line in sys.stdin:\n"
        + " r=json.loads(line)\n"
        + " if r['type']=='shutdown': break\n"
        + " rid=r['request_id']\n"
        + " print(json.dumps({'type':'generation_mode','request_id':rid,'mtp_active':False,'dflash_active':False}),flush=True)\n"
        + " print(json.dumps(dict(complete,type='complete',request_id=rid)),flush=True)\n"
    )
    engine.chmod(0o755)
    config = tmp_path / "config.json"
    output = tmp_path / "output"
    config.write_text(
        json.dumps(
            {
                "engine": str(engine),
                "model": str(tmp_path),
                "output": str(output),
                "golden": str(ROOT / "docs/measurements/sushi-20261027/gold-AR-MAIN.json"),
                "tokens": 17,
                "repeats": 1,
                "prompt": "MAIN",
            }
        )
    )
    if diverge:
        with pytest.raises(AssertionError, match="external token oracle"):
            DRIVER["main"](config)
    else:
        DRIVER["main"](config)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == ("failed" if diverge else "validated")
    assert report["reaped"] and report["reader_joined"]
    assert len(report["runs"]) == (0 if diverge else 1)
    saved = (output / "report.json").read_bytes()
    with pytest.raises(AssertionError, match="preserve previous results"):
        DRIVER["main"](config)
    assert (output / "report.json").read_bytes() == saved
