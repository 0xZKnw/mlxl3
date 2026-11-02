"""Frozen pool campaign: reuse the bridge runner with an external token oracle."""

import json
import runpy
import sys
from pathlib import Path


def token_hash(tokens):
    assert tokens and all(type(t) is int and 0 <= t < 248320 for t in tokens)
    value = 0xCBF29CE484222325
    for token in tokens:
        for byte in token.to_bytes(4, "little"):
            value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    return f"{value:016x}"


def check_golden(event, golden, budget):
    assert type(budget) is int and budget in (17, 128)
    assert len(golden["tokens"]) == 128
    assert token_hash(golden["tokens"]) == golden["token_hash"], "invalid golden"
    assert event["token_hash"] == token_hash(golden["tokens"][:budget]), "external token oracle"
    text = event["cache_context"]
    assert isinstance(text, str) and text and golden["cache_context"].startswith(text)
    if budget == 128:
        assert text == golden["cache_context"], "external text oracle"
    memory = event["stats"]["memory"]
    for key in ("process_footprint_bytes", "process_lifetime_peak_bytes"):
        assert type(memory[key]) is int and 0 < memory[key] < 12_000_000_000, key


def main(config_path):
    config = json.loads(config_path.read_text())
    output = Path(config["output"])
    assert not (output / "report.json").exists(), "preserve previous results"
    budget = config["tokens"]
    assert type(budget) is int and budget in (17, 128)
    assert config["repeats"] == (1 if budget == 17 else 3)
    golden = json.loads(Path(config["golden"]).read_text())
    runner = runpy.run_path("scripts/benchmark-flash-bridge.py")
    validate = runner["validate"]

    def checked(event, mode, mtp, budget, *args):
        validate(event, mode, mtp, budget, *args)
        check_golden(event, golden, budget)

    runner["run"].__globals__["validate"] = checked
    report = runner["run"](
        Path(config["engine"]),
        Path(config["model"]),
        output,
        mtp=False,
        tokens=budget,
        repeats=config["repeats"],
        cache_gib=3,
        timeout=180,
        prompt=config["prompt"],
    )
    assert report["status"] == "validated" and report["reaped"] and report["reader_joined"]
    assert len(report["warmup"]) == 1 and len(report["runs"]) == config["repeats"]
    print("POOL_BRIDGE_VALIDATED", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
