"""77B accounting tests: disposable reports/CLI, no model or GPU."""

import copy
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_flash_process_memory.py"
spec = importlib.util.spec_from_file_location("flash_process_memory", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

FIELDS = (
    "process_id",
    "process_start_abstime",
    "process_footprint_bytes",
    "process_compressed_logical_bytes",
    "process_lifetime_peak_bytes",
    "process_compressed_peak_logical_bytes",
    "process_compressed_lifetime_logical_bytes",
    "process_decompressions",
    "process_ledger_swapins_logical_bytes",
)
MONOTONE = FIELDS[4:]
U64_MAX = 18_446_744_073_709_551_615


def memory(**changes):
    result = dict.fromkeys(FIELDS, 0)
    result.update(
        process_id=123,
        process_start_abstime=456,
        process_footprint_bytes=7_800_000_000,
        process_lifetime_peak_bytes=8_000_000_000,
    )
    result.update(changes)
    return result


def checker(*values):
    samples = values or (memory(), memory())
    return {
        "status": "complete",
        "parity": True,
        "ready": {"type": "ready", "memory": samples[0]},
        "events": [
            {"type": "ready", "memory": samples[0]},
            {"type": "error", "message": "expected invalid-mode error"},
            *(
                {"type": "complete", "request_id": str(i), "stats": {"memory": value}}
                for i, value in enumerate(samples[1:])
            ),
        ],
    }


def assert_failed(document, message=None):
    with pytest.raises(analysis.AnalysisError, match=message) as error:
        analysis.analyze(document)
    report = error.value.report
    assert report["status"] == "failed" and report["qualification"] is None
    return report


def test_checker_records_all_samples_and_deltas_without_claiming_model_parity():
    first = memory(process_compressed_logical_bytes=10, process_compressed_peak_logical_bytes=10)
    second = memory(
        process_footprint_bytes=7_700_000_000,
        process_compressed_peak_logical_bytes=10,
        process_compressed_lifetime_logical_bytes=20,
        process_decompressions=4,
    )
    last = dict(second, process_compressed_lifetime_logical_bytes=25, process_decompressions=5)
    document = checker(first, second, last)
    document["global_vm"] = {"Swapouts": 9876543210}
    original = copy.deepcopy(document)
    result = analysis.analyze(document)
    assert document == original
    assert result["status"] == "complete"
    assert result["qualification"] == "observed_process_pressure_within_budget"
    assert "parity" not in result
    assert len(result["samples"]) == 3
    delta = result["final_deltas_since_ready"]
    assert delta["process_footprint_bytes"] == -100_000_000
    assert delta["process_compressed_logical_bytes"] == -10
    assert delta["process_compressed_lifetime_logical_bytes"] == 25
    assert delta["process_decompressions"] == 5
    assert delta["process_ledger_swapins_logical_bytes"] == 0
    assert result["maxima"]["process_compressed_logical_bytes"] == 10
    assert result["startup_already_affected"] is False


def test_benchmark_runtime_warmup_and_every_run_are_inspected_in_order():
    document = {
        "status": "validated",
        "reaped": True,
        "reader_joined": True,
        "runtime": {"memory": memory()},
        "warmup": [{"stats": {"memory": memory(process_decompressions=1)}}],
        "runs": [
            {"stats": {"memory": memory(process_decompressions=2)}},
            {"stats": {"memory": memory(process_decompressions=3)}},
        ],
    }
    result = analysis.analyze(document)
    assert [s["label"] for s in result["samples"]] == ["ready", "warmup[0]", "runs[0]", "runs[1]"]
    assert result["final_deltas_since_ready"]["process_decompressions"] == 3
    document["runs"][0]["stats"]["memory"]["process_start_abstime"] += 1
    assert_failed(document, "identity changed")


def test_zeros_and_full_u64_cumulative_baselines_are_not_missing():
    zero = dict.fromkeys(FIELDS, 0)
    zero.update(process_id=1, process_start_abstime=1)
    assert analysis.analyze(checker(zero, zero))["startup_already_affected"] is False
    large = memory(
        process_id=U64_MAX,
        process_start_abstime=U64_MAX,
        process_compressed_lifetime_logical_bytes=U64_MAX,
        process_ledger_swapins_logical_bytes=U64_MAX,
        process_decompressions=(1 << 31) - 2,
    )
    result = analysis.analyze(checker(large, large))
    assert result["startup_already_affected"] is True
    assert result["startup_ledger_swapins_logical_bytes"] == U64_MAX
    assert result["final_deltas_since_ready"]["process_ledger_swapins_logical_bytes"] == 0
    assert result["final_deltas_since_ready"]["process_compressed_lifetime_logical_bytes"] == 0


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize(
    "value", [None, True, False, -1, U64_MAX + 1, 0.0, "0", [], {}, float("nan"), float("inf")]
)
def test_required_counters_are_strict_u64_and_unknown_is_not_zero(field, value):
    document = checker(memory(), memory(**{field: value}))
    report = assert_failed(document, field)
    assert len(report["samples"]) == 2
    assert report["samples"][-1]["memory"][field] is None
    assert report["samples"][-1]["deltas_since_ready"] is None
    assert report["final_deltas_since_ready"] is None


@pytest.mark.parametrize("field", FIELDS)
def test_missing_counters_preserve_a_partial_unknown_sample(field):
    document = checker()
    del document["events"][-1]["stats"]["memory"][field]
    report = assert_failed(document, field)
    assert report["samples"][-1]["memory"][field] is None
    assert report["samples"][-1]["deltas_since_ready"] is None


@pytest.mark.parametrize("field", ("process_id", "process_start_abstime"))
def test_identity_changes_are_rejected(field):
    assert_failed(checker(memory(), memory(**{field: 999})), "identity changed")
    assert_failed(checker(memory(**{field: 0}), memory()), "nonpositive")


@pytest.mark.parametrize("field", MONOTONE)
def test_each_cumulative_counter_cannot_reverse(field):
    first = memory(**{field: 10 if field != "process_lifetime_peak_bytes" else 8_000_000_000})
    last = dict(first)
    last[field] -= 1
    assert_failed(checker(first, last), "reversed")


@pytest.mark.parametrize("value", [(1 << 31) - 1, 1 << 31, U64_MAX])
def test_decompression_saturation_cannot_masquerade_as_no_activity(value):
    report = assert_failed(checker(memory(process_decompressions=value), memory()), "saturated")
    assert report["samples"][0]["deltas_since_ready"] is None


@pytest.mark.parametrize(
    "field,limit",
    [
        ("process_lifetime_peak_bytes", 12_000_000_000),
        ("process_compressed_logical_bytes", 512 * 1024 * 1024),
        ("process_compressed_peak_logical_bytes", 512 * 1024 * 1024),
    ],
)
def test_absolute_caps_accept_equality_and_reject_one_more(field, limit):
    for extra in [0, 1]:
        values = memory(**{field: limit + extra})
        if field == "process_compressed_logical_bytes":
            values["process_compressed_peak_logical_bytes"] = limit + extra
        document = checker(values, values)
        if extra:
            report = assert_failed(document, "exceeds budget")
            assert report["samples"][0]["memory"][field] == limit + 1
        else:
            assert analysis.analyze(document)["maxima"][field] == limit


def test_lifetime_delta_budget_is_from_ready_and_inclusive():
    first = memory(process_compressed_lifetime_logical_bytes=7 << 30)
    last = memory(process_compressed_lifetime_logical_bytes=8 << 30)
    assert (
        analysis.analyze(checker(first, last))["final_deltas_since_ready"][
            "process_compressed_lifetime_logical_bytes"
        ]
        == 1 << 30
    )
    last["process_compressed_lifetime_logical_bytes"] += 1
    report = assert_failed(checker(first, last), "lifetime delta exceeds")
    assert (
        report["final_deltas_since_ready"]["process_compressed_lifetime_logical_bytes"]
        == (1 << 30) + 1
    )


def test_logical_swapins_before_ready_are_reported_but_new_consumption_fails():
    first = memory(process_ledger_swapins_logical_bytes=17)
    assert analysis.analyze(checker(first, first))["startup_already_affected"] is True
    last = memory(process_ledger_swapins_logical_bytes=18)
    report = assert_failed(checker(first, last), "logical swapin delta exceeds")
    assert report["final_deltas_since_ready"]["process_ledger_swapins_logical_bytes"] == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "failed"},
        {"status": "running"},
        {"status": "incomplete"},
        {"parity": False},
        {"parity": None},
        {"parity": 1},
        {"reaped": False},
        {"reaped": None},
        {"reaped": 1},
        {"reader_joined": False},
        {"sampler_joined": False},
        {"active_request": {"status": "running"}},
    ],
)
def test_incomplete_or_unjoined_sources_never_qualify_but_keep_observations(changes):
    document = checker()
    document.update(changes)
    report = assert_failed(document, "source")
    assert len(report["samples"]) == 2
    assert report["final_deltas_since_ready"]["process_ledger_swapins_logical_bytes"] == 0


@pytest.mark.parametrize(
    "document",
    [
        None,
        [],
        {},
        {"status": "complete", "events": []},
        {"status": "complete", "ready": {"memory": memory()}, "events": []},
    ],
)
def test_no_empty_or_unrecognized_input_can_qualify(document):
    assert_failed(document)


def test_malformed_middle_sample_and_empty_benchmark_are_not_skipped():
    document = checker(memory(), memory(), memory())
    document["events"][-2]["stats"] = None
    result = assert_failed(document, "missing memory")
    assert len(result["samples"]) == 2
    assert result["samples"][-1]["memory"]["process_id"] is None
    assert result["final_deltas_since_ready"] is None
    for field in ["warmup", "runs"]:
        benchmark = {
            "status": "validated",
            "runtime": {"memory": memory()},
            "warmup": [{"stats": {"memory": memory()}}],
            "runs": [{"stats": {"memory": memory()}}],
        }
        benchmark[field] = []
        assert_failed(benchmark, "nonempty")


@pytest.mark.parametrize(
    "changes",
    [
        {"process_lifetime_peak_bytes": 1},
        {"process_compressed_logical_bytes": 1, "process_compressed_peak_logical_bytes": 0},
    ],
)
def test_inconsistent_peaks_are_rejected(changes):
    assert_failed(checker(memory(**changes), memory()), "peak below")


def cli(source, output):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(source), str(output)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )


def test_real_cli_saves_provenance_and_refuses_overwrite(tmp_path):
    source, output = tmp_path / "input.json", tmp_path / "analysis.json"
    source.write_text(json.dumps(checker()))
    done = cli(source, output)
    assert done.returncode == 0, done.stderr
    report = json.loads(output.read_text())
    assert (
        report["status"] == "complete"
        and report["qualification"] == "observed_process_pressure_within_budget"
    )
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    original = output.read_bytes()
    assert cli(source, output).returncode != 0
    assert output.read_bytes() == original


@pytest.mark.parametrize("fault", ["json", "incomplete", "counter", "missing_input"])
def test_real_cli_saves_failures_and_returns_nonzero(tmp_path, fault):
    source, output = tmp_path / "input.json", tmp_path / "analysis.json"
    document = checker()
    if fault == "incomplete":
        document["status"] = "failed"
    if fault == "counter":
        document["events"][-1]["stats"]["memory"]["process_compressed_logical_bytes"] = None
    if fault != "missing_input":
        source.write_text("not JSON" if fault == "json" else json.dumps(document))
    done = cli(source, output)
    assert done.returncode != 0
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["qualification"] is None
    assert report["error"]
    if fault == "counter":
        assert report["samples"][-1]["memory"]["process_compressed_logical_bytes"] is None
    if fault == "incomplete":
        assert len(report["samples"]) == 2


@pytest.mark.skipif(os.name != "posix", reason="FIFO/signal test requires POSIX")
@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGKILL])
def test_actual_cli_interruption_saves_failure_and_child_is_joined(tmp_path, stop_signal):
    source, output = tmp_path / "blocked-input", tmp_path / "interrupted.json"
    os.mkfifo(source)
    child = subprocess.Popen(
        [sys.executable, str(SCRIPT), str(source), str(output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 3
        initialized = False
        while child.poll() is None and time.monotonic() < deadline:
            try:
                initialized = json.loads(output.read_text())["error"] == "analysis did not finish"
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            if initialized:
                break
            time.sleep(0.01)
        assert initialized and child.poll() is None
        child.send_signal(stop_signal)
        assert child.wait(timeout=3) != 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)
    report = json.loads(output.read_text())
    assert report["status"] == "failed" and report["qualification"] is None
    assert (
        ("InterruptedError" in report["error"])
        if stop_signal == signal.SIGTERM
        else (report["error"] == "analysis did not finish")
    )
    assert report["source_sha256"] is None


def test_observed_large_compression_is_unqualified_even_with_zero_logical_swapin_delta():
    # Rounded observed77B magnitudes, keeping peaks/gauges coherent. Synthetic
    # accounting fixture, not a replacement for the raw physical benchmark.
    first = memory(
        process_compressed_logical_bytes=3_900_000_000,
        process_compressed_peak_logical_bytes=3_983_000_000,
        process_compressed_lifetime_logical_bytes=5_000_000_000,
    )
    last = memory(
        process_compressed_peak_logical_bytes=3_983_000_000,
        process_compressed_lifetime_logical_bytes=7_384_000_000,
    )
    report = assert_failed(checker(first, last), "exceeds budget")
    assert len(report["samples"]) == 2
    assert report["maxima"]["process_compressed_peak_logical_bytes"] == 3_983_000_000
    assert (
        report["final_deltas_since_ready"]["process_compressed_lifetime_logical_bytes"]
        == 2_384_000_000
    )
    assert report["final_deltas_since_ready"]["process_ledger_swapins_logical_bytes"] == 0
    violations = {v["metric"] for v in report["budget_violations"]}
    assert violations == {
        "process_compressed_logical_bytes",
        "process_compressed_peak_logical_bytes",
        "compressed_lifetime_delta_bytes",
    }


@pytest.mark.parametrize("value", [(1 << bits) - 1 for bits in range(65)])
def test_u64_reader_accepts_finite_bit_boundaries(value):
    assert analysis.u64(value) == value


def test_invalid_event_object_is_not_silently_skipped():
    document = checker()
    document["events"].insert(2, None)
    report = assert_failed(document, "invalid event at 2")
    assert len(report["samples"]) == 1
    assert report["qualification"] is None
