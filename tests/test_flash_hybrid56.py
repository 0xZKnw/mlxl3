"""Fresh-process Flash56 reports cannot qualify partial, divergent or swapped runs."""

import hashlib
import json
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/measurements/flash-next-ssd-impl/opt-56-hybrid-hot/bench.py"
API = runpy.run_path(str(SOURCE))
FIXTURE = runpy.run_path(str(ROOT / "tests/test_flash_larger_cache.py"))["fixture"]


def records(mode="baseline"):
    return [
        {
            "mode": mode,
            "prompt": p,
            "index": i,
            "label": label,
            "engine": "/" + label,
            "binary_sha256": ("a" if label == "H" else "b") * 64,
            "gpu_hot": int(label == "C"),
            "cache_gib": 3,
            "status": "validated",
            "tps": 10.2 if label == "C" else 10.0,
            "peak_bytes": 12_000_000_000,
            "swapouts_bytes": 0,
            "token_hash": "1234567890abcdef",
            "text": "known prefix continued",
            "mtp_counts": [70, 57, 70] if mode == "mtp" else [None] * 3,
        }
        for p in ("MAIN", "HOLD")
        for i, label in enumerate("HCCH")
    ]


@pytest.mark.parametrize("mode", ["baseline", "mtp"])
@pytest.mark.parametrize("case", ["eligible", "nogain", "drift", "spread"])
def test_qualification_requires_both_pairs_on_both_prompts(mode, case):
    data = records(mode)
    if case == "nogain":
        data[1]["tps"] = 10.0
    elif case == "drift":
        data[0]["tps"] = 9.4
    elif case == "spread":
        data[1]["tps"] = 10.6
    result = API["qualify"](data, mode)
    assert (
        result["status"]
        == {
            "eligible": "validated_gain",
            "nogain": "rejected_gain",
            "drift": "inconclusive_drift",
            "spread": "inconclusive_drift",
        }[case]
    )
    assert result["confirmed_gain"] == (case == "eligible")


@pytest.mark.parametrize(
    "fault",
    [
        "partial",
        "empty",
        "swap",
        "ram",
        "nan",
        "zero",
        "flag",
        "bool_flag",
        "cache",
        "hash",
        "text",
        "counts",
        "binary",
        "mode",
    ],
)
def test_qualification_refuses_invalid_evidence(fault):
    data = records("mtp")
    if fault == "empty":
        data = []
    elif fault == "partial":
        data = data[:-1]
    elif fault == "swap":
        data[1]["swapouts_bytes"] = 16384
    elif fault == "ram":
        data[1]["peak_bytes"] += 1
    elif fault == "nan":
        data[1]["tps"] = float("nan")
    elif fault == "zero":
        data[1]["tps"] = 0
    elif fault == "flag":
        data[1]["gpu_hot"] = 0
    elif fault == "bool_flag":
        data[1]["gpu_hot"] = True
    elif fault == "cache":
        data[1]["cache_gib"] = 6
    elif fault == "hash":
        data[1]["token_hash"] = "wrong"
    elif fault == "text":
        data[1]["text"] += "wrong"
    elif fault == "counts":
        data[1]["mtp_counts"][1] += 1
    elif fault == "binary":
        data[1]["binary_sha256"] = data[0]["binary_sha256"]
    elif fault == "mode":
        data[1]["mode"] = "baseline"
    with pytest.raises(AssertionError):
        API["qualify"](data, "mtp")


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "timeout",
        "empty_report",
        "divergence",
        "unreaped",
        "swap",
        "identity",
        "flag",
        "command",
        "reportpath",
    ],
)
@pytest.mark.parametrize("mode", ["baseline", "mtp"])
def test_real_campaign_entry_records_failures_and_preserves_old_results(
    tmp_path, monkeypatch, fault, mode
):
    api = runpy.run_path(str(SOURCE))
    glob = api["main"].__globals__
    monkeypatch.setitem(glob, "BASE", tmp_path)
    monkeypatch.setitem(glob, "JOURNAL", tmp_path / "opti.md")
    matrix, identity = [], {}
    for label in "HC":
        engine = tmp_path / label
        engine.write_bytes(label.encode())
        identity[str(engine)] = hashlib.sha256(engine.read_bytes()).hexdigest()
    for row in records(mode):
        item = {k: row[k] for k in ("mode", "prompt", "index", "label", "gpu_hot", "cache_gib")}
        dest = tmp_path / f"{item['prompt']}-{item['index']}"
        dest.mkdir()
        engine = str(tmp_path / item["label"])
        item.update(
            engine=engine,
            binary_sha256=identity[engine],
            model="/model",
            plan=str(dest / "run.json"),
            report=str(dest / "report.json"),
            golden=str(tmp_path / "golden.json"),
        )
        Path(item["plan"]).write_text(
            json.dumps(
                [
                    {
                        "name": dest.name,
                        "env": {"MLXL3_SSD_GPU_HOT": str(item["gpu_hot"])},
                        "command": [
                            "python",
                            str(SOURCE),
                            "generate",
                            engine,
                            "/model",
                            str(dest),
                            mode,
                            item["prompt"],
                            item["golden"],
                        ],
                    }
                ]
            )
        )
        matrix.append(item)
    (tmp_path / f"{mode}-matrix.json").write_text(json.dumps(matrix))
    (tmp_path / "quality-summary.json").write_text(
        json.dumps({"status": "validated_quality_not_speed", "cache_gib": 3})
    )
    (tmp_path / "speed-identity.json").write_text(json.dumps(identity))
    (tmp_path / "campaign-baseline.json").write_text(
        json.dumps({"original_vm_baseline": {"Swapins": 0, "Swapouts": 0}})
    )
    (tmp_path / "golden.json").write_text(
        json.dumps({"token_hash": "1234567890abcdef", "cache_context": "known prefix continued"})
    )
    if fault == "identity":
        (tmp_path / "H").write_bytes(b"changed")
    if fault == "flag":
        plan = Path(matrix[0]["plan"])
        data = json.loads(plan.read_text())
        data[0]["env"]["MLXL3_SSD_GPU_HOT"] = "1"
        plan.write_text(json.dumps(data))
    if fault == "command":
        plan = Path(matrix[0]["plan"])
        data = json.loads(plan.read_text())
        data[0]["command"][5] = "/wrong-report-directory"
        plan.write_text(json.dumps(data))
    if fault == "reportpath":
        matrix[0]["report"] = str(tmp_path / "wrong.json")
        (tmp_path / f"{mode}-matrix.json").write_text(json.dumps(matrix))

    def sample(plan, baseline):
        assert baseline == {"Swapins": 0, "Swapouts": 0}
        item = next(i for i in matrix if Path(i["plan"]) == plan)
        process = {
            "status": "completed_expected_outcome",
            "reaped": fault != "unreaped",
            "swapouts_bytes": 16384 if fault == "swap" else 0,
        }
        if fault == "timeout":
            process.update(status="failed_or_interrupted", exit_code=124)
        plan.with_name("run-result.json").write_text(json.dumps({"commands": [process]}))
        if fault == "timeout":
            raise TimeoutError("joined fake child")
        data, _, _ = FIXTURE(mode)
        data.update(engine=item["engine"], binary_sha256=item["binary_sha256"], expert_cache_gib=3)
        for run in data["warmup"] + data["runs"]:
            run["prompt"] = item["prompt"]
            tps = 10.2 if item["label"] == "C" else 10.0
            run["stats"].update(decode_tps=tps, decode_seconds=127 / tps)
        if fault == "empty_report":
            data["runs"] = []
        if fault == "divergence":
            data["runs"][0]["text"] += "wrong"
        Path(item["report"]).write_text(json.dumps(data))
        plan.with_name("stderr.log").write_text(
            "\n".join(
                f"Flash Next: loading layer {i}/48, 53 exact expert slots" for i in range(1, 49)
            )
        )

    monkeypatch.setitem(glob, "PROFILE", {"run_sampler": sample})
    if fault:
        with pytest.raises((AssertionError, TimeoutError)):
            api["main"](mode)
    else:
        api["main"](mode)
    result = tmp_path / f"{mode}-result.json"
    data = json.loads(result.read_text())
    assert data["status"] == ("failed_or_interrupted" if fault else "validated_gain")
    assert data["confirmed_gain"] == (fault is None)
    if fault and data["records"]:
        assert data["records"][-1]["status"] == "failed_or_interrupted"
        if fault == "timeout":
            assert data["records"][-1]["process"]["reaped"]
    before = result.read_bytes()
    with pytest.raises(AssertionError):
        api["main"](mode)
    assert result.read_bytes() == before


@pytest.mark.parametrize("mode", ["baseline", "mtp"])
def test_generate_cli_passes_the_real_report_directory_and_fixed_cache(mode, tmp_path, monkeypatch):
    calls = []
    real_load = runpy.run_path
    old = SOURCE.parent.parent / "opt-50-larger-cache/bench.txt"

    def load(path, *args, **kwargs):
        if Path(path).resolve() == old.resolve():
            return {"generate": lambda *args: calls.append(args)}
        return real_load(path, *args, **kwargs)

    dest, golden = str(tmp_path / "output"), str(tmp_path / "golden.json")
    monkeypatch.setattr(runpy, "run_path", load)
    monkeypatch.setattr(
        sys, "argv", [str(SOURCE), "generate", "/engine", "/model", dest, mode, "HOLD", golden]
    )
    real_load(str(SOURCE), run_name="__main__")
    assert calls == [("/engine", "/model", dest, mode, "HOLD", 3, golden)]
