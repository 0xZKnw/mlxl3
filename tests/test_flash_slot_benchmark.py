"""Trial46 qualification cannot accept drift, partial matrices or divergence."""

import copy
import json
import math
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT / "docs/measurements/flash-next-ssd-impl/opt-46-slot-nt2/speed-final/speed-supervisor.txt"
)
API = runpy.run_path(str(SOURCE))


def report(mode="baseline"):
    rows = []
    for repeat in range(3):
        rows.append(
            {
                "mode": mode,
                "prompt": "MAIN",
                "repeat": repeat,
                "token_hash": "1234567890abcdef",
                "text": "known prefix continued",
                "stats": {
                    "generated_tokens": 128,
                    "decode_tokens": 127,
                    "decode_tps": 10.0,
                    "decode_seconds": 12.7,
                    "ttft_seconds": 1.0,
                    "prefill_tps": 1.0,
                    "memory": {"process_lifetime_peak_bytes": 8 << 30},
                    "mtp_proposed_tokens": 70,
                    "mtp_accepted_tokens": 57,
                    "mtp_blocks": 70,
                },
            }
        )
    return {
        "status": "validated",
        "tokens": 128,
        "repeats": 2,
        "engine": "/engine",
        "model": "/model",
        "binary_sha256": "1" * 64,
        "expert_cache_gib": 3,
        "warmup": rows[:1],
        "runs": rows[1:],
    }


def item(mode="baseline"):
    return {
        "stage": 0,
        "mode": mode,
        "prompt": "MAIN",
        "label": "H",
        "engine": "/engine",
        "model": "/model",
        "binary_sha256": "1" * 64,
    }


@pytest.mark.parametrize("mode", ["baseline", "mtp"])
def test_real_report_validation_and_statistics(mode, monkeypatch):
    monkeypatch.chdir(ROOT)
    data = report(mode)
    before = copy.deepcopy(data)
    result = API["check_report"](data, item(mode), {"text": "known prefix"})
    assert data == before and result["tps"] == 10 and result["peak_bytes"] == 8 << 30
    assert result["mtp_counts"] == ([70, 57, 70] if mode == "mtp" else None)


@pytest.mark.parametrize(
    "bad",
    [
        "partial",
        "empty",
        "warmup",
        "repeat",
        "hash",
        "text",
        "prefix",
        "binary",
        "engine",
        "model",
        "memory",
        "nan",
        "timing",
        "drift",
        "warm_drift",
        "counts",
    ],
)
def test_report_rejects_invalid_or_incomplete_evidence(bad, monkeypatch):
    monkeypatch.chdir(ROOT)
    d = report("mtp")
    oracle = {"text": "known prefix"}
    if bad == "partial":
        d["status"] = "incomplete"
    elif bad == "empty":
        d["runs"] = []
    elif bad == "warmup":
        d["warmup"] = []
    elif bad == "repeat":
        d["runs"][1]["repeat"] = 1
    elif bad == "hash":
        d["runs"][0]["token_hash"] = "foo"
    elif bad == "text":
        d["runs"][1]["text"] += " divergence"
    elif bad == "prefix":
        oracle["text"] = ""
    elif bad == "binary":
        d["binary_sha256"] = "2" * 64
    elif bad in ("engine", "model"):
        d[bad] = "/wrong"
    elif bad == "memory":
        d["runs"][0]["stats"]["memory"]["process_lifetime_peak_bytes"] = (10 << 30) + 1
    elif bad == "nan":
        d["runs"][0]["stats"]["decode_tps"] = math.nan
    elif bad == "timing":
        d["runs"][0]["stats"]["decode_seconds"] = 1.0
    elif bad in ("drift", "warm_drift"):
        row = d["warmup"][0] if bad == "warm_drift" else d["runs"][1]
        row["stats"]["decode_tps"] = 7.0 if bad == "warm_drift" else 10.6
        row["stats"]["decode_seconds"] = 127 / row["stats"]["decode_tps"]
    else:
        d["runs"][1]["stats"]["mtp_accepted_tokens"] = 56
    with pytest.raises((AssertionError, ValueError)):
        API["check_report"](d, item("mtp"), oracle)


def records(stages):
    return [
        {
            "stage": s,
            "mode": m,
            "prompt": p,
            "label": label,
            "status": "validated",
            "swapouts_bytes": 0,
            "tps": 10.3 if label == "C" else 10.0,
            "peak_bytes": 8 << 30,
            "token_hash": "1234567890abcdef",
            "text": p,
            "mtp_counts": [70, 57, 70] if m == "mtp" else None,
        }
        for s in range(stages)
        for m, p in API["CONTEXTS"]
        for label in ("H", "C")
    ]


@pytest.mark.parametrize("stages", [1, 2])
def test_complete_matrices_and_memory_boundary(stages):
    data = records(stages)
    for r in data:
        if r["label"] == "C":
            r["peak_bytes"] += 256 << 20
    result = API["qualification"](data, stages)
    assert result["eligible"] and len(result["pairs"]) == 4 * stages
    data[-1]["peak_bytes"] += 1
    assert not API["qualification"](data, stages)["eligible"]


@pytest.mark.parametrize(
    "bad",
    [
        "empty",
        "zero_stage",
        "missing",
        "duplicate",
        "status",
        "swap",
        "nan",
        "peak",
        "text",
        "counts",
    ],
)
def test_incomplete_divergent_or_invalid_matrices_cannot_qualify(bad):
    data = records(2)
    if bad == "empty":
        data = []
    elif bad == "zero_stage":
        with pytest.raises(AssertionError):
            API["qualification"]([], 0)
        return
    elif bad == "missing":
        data.pop()
    elif bad == "duplicate":
        data[-1] = data[0]
    elif bad == "status":
        data[-1]["status"] = "running"
    elif bad == "swap":
        data[-1]["swapouts_bytes"] = 16384
    elif bad == "nan":
        data[-1]["tps"] = math.nan
    elif bad == "peak":
        data[-1]["peak_bytes"] = True
    elif bad == "text":
        data[-1]["text"] = "diverged"
    else:
        data[-1]["mtp_counts"] = [70, 56, 70]
    with pytest.raises(AssertionError):
        API["qualification"](data, 2)


@pytest.mark.parametrize("bad", ["no_gain", "confirmation_drift", "control_drift"])
def test_failed_gain_or_drift_is_rejected(bad):
    data = records(2)
    for r in data:
        if r["stage"] == 1 and r["label"] == "C":
            r["tps"] = 10.0 if bad == "no_gain" else 10.7
        if bad == "control_drift" and r["stage"] == 1:
            r["tps"] *= 1.06
    assert not API["qualification"](data, 2)["eligible"]


def test_aborted_child_keeps_failure_and_refuses_overwrite(tmp_path, monkeypatch):
    main = runpy.run_path(str(SOURCE))["main"]
    monkeypatch.chdir(tmp_path)
    main.__globals__["__file__"] = str(tmp_path / "speed-supervisor.txt")
    (tmp_path / "speed-matrix.json").write_text(
        json.dumps([{**r, "plan": "arm.json"} for r in records(1)])
    )
    (tmp_path / "source").write_text("pinned")
    import hashlib

    (tmp_path / "speed-identity.json").write_text(
        json.dumps({str(tmp_path / "source"): hashlib.sha256(b"pinned").hexdigest()})
    )
    (tmp_path / "quality-plan-result.json").write_text(
        json.dumps(
            {
                "status": "completed_expected_outcomes_not_speed",
                "commands": [
                    {"status": "completed_expected_outcome", "reaped": True, "swapouts_bytes": 0}
                ]
                * 13,
            }
        )
    )

    def interrupt(path):
        raise TimeoutError("bounded child failed")

    monkeypatch.setattr(runpy, "run_path", lambda _: {"main": interrupt})
    with pytest.raises(TimeoutError):
        main(0)
    output = tmp_path / "speed-result-0.json"
    saved = output.read_bytes()
    assert (
        json.loads(saved)["status"] == "failed_or_interrupted"
        and json.loads(saved)["confirmed_gain"] is False
    )
    with pytest.raises(AssertionError, match="overwrite"):
        main(0)
    assert output.read_bytes() == saved


@pytest.mark.parametrize(
    "outcome", ["gain", "reject", "diverge", "missing_report", "partial_report"]
)
def test_serialized_campaign_only_confirms_complete_consistent_measurements(
    tmp_path, monkeypatch, outcome
):
    import hashlib

    main = runpy.run_path(str(SOURCE))["main"]
    main.__globals__["__file__"] = str(tmp_path / "speed-supervisor.txt")
    monkeypatch.chdir(tmp_path)
    matrix = []
    for r in records(2):
        name = f"{r['stage']}-{r['mode']}-{r['prompt']}-{r['label']}"
        matrix.append(
            {
                **item(r["mode"]),
                **{k: r[k] for k in ("stage", "mode", "prompt", "label")},
                "plan": name + ".json",
                "report": name + "/report.json",
                "oracle64": str(tmp_path / "oracle.json"),
            }
        )
    (tmp_path / "speed-matrix.json").write_text(json.dumps(matrix))
    (tmp_path / "oracle.json").write_text(json.dumps({"runs": [{"text": "known prefix"}]}))
    (tmp_path / "source").write_bytes(b"pinned")
    (tmp_path / "speed-identity.json").write_text(
        json.dumps({str(tmp_path / "source"): hashlib.sha256(b"pinned").hexdigest()})
    )
    (tmp_path / "quality-plan-result.json").write_text(
        json.dumps(
            {
                "status": "completed_expected_outcomes_not_speed",
                "commands": [
                    {"status": "completed_expected_outcome", "reaped": True, "swapouts_bytes": 0}
                ]
                * 13,
            }
        )
    )
    calls = []

    def supervise(path):
        cell = next(i for i in matrix if i["plan"] == path.name)
        calls.append(path.name)
        path.with_name(path.stem + "-result.json").write_text(
            json.dumps(
                {
                    "commands": [
                        {
                            "status": "completed_expected_outcome",
                            "reaped": True,
                            "swapouts_bytes": 0,
                        }
                    ]
                }
            )
        )
        data = report(cell["mode"])
        for r in data["warmup"] + data["runs"]:
            r["prompt"] = cell["prompt"]
            tps = 10.3 if cell["label"] == "C" and outcome != "reject" else 10.0
            r["stats"].update(decode_tps=tps, decode_seconds=127 / tps)
            if outcome == "diverge" and cell["label"] == "C":
                r["text"] += " changed"
        if outcome == "partial_report":
            data["status"] = "incomplete"
        if outcome != "missing_report":
            dest = tmp_path / cell["report"]
            dest.parent.mkdir()
            dest.write_text(json.dumps(data))

    original = runpy.run_path
    monkeypatch.setattr(
        runpy,
        "run_path",
        lambda path: (
            {"main": supervise}
            if str(path).endswith("run-supervisor.txt")
            else original(str(ROOT / path))
        ),
    )
    if outcome in ("gain", "reject"):
        main(0)
        first = json.loads((tmp_path / "speed-result-0.json").read_text())
        assert not first["confirmed_gain"] and len(calls) == 8
        if outcome == "gain":
            assert first["status"] == "screen_eligible"
            main(1)
            final = json.loads((tmp_path / "speed-result-1.json").read_text())
            assert (
                final["status"] == "validated_gain" and final["confirmed_gain"] and len(calls) == 16
            )
        else:
            assert first["status"] == "rejected_screen"
            with pytest.raises(AssertionError):
                main(1)
            assert len(calls) == 8
    else:
        with pytest.raises((AssertionError, FileNotFoundError)):
            main(0)
        final = json.loads((tmp_path / "speed-result-0.json").read_text())
        assert (
            final["status"] == "failed_or_interrupted"
            and not final["confirmed_gain"]
            and len(calls) <= 2
        )


@pytest.mark.parametrize("mode", ["baseline", "mtp"])
def test_bridge_driver_selects_the_registered_budget_and_options(monkeypatch, mode):
    import sys

    calls = []
    original = runpy.run_path
    monkeypatch.setattr(sys, "argv", ["driver", "an engine", "a model", "an output", mode, "MAIN"])
    monkeypatch.setattr(
        runpy, "run_path", lambda path: {"run": lambda *args, **kw: calls.append((args, kw))}
    )
    driver = SOURCE.with_name("bridge-driver.txt")
    original(str(driver))
    assert calls == [
        (
            (Path("an engine"), Path("a model"), Path("an output")),
            {
                "mtp": mode == "mtp",
                "tokens": 128,
                "repeats": 2,
                "timeout": 120,
                "cache_gib": 3,
                "prompt": "MAIN",
            },
        )
    ]
