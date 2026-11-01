"""Bounded six-phase diagnostic capture; trace coverage requires later inspection."""

import hashlib
import json
import math
import os
import re
import runpy
import selectors
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PHASES = ("warmup", "control_before", "trace_1", "trace_2", "trace_3", "control_after")
SELECTOR = "qwen4::profile::flash_detailed_profile_preserves_full_outputs_and_states"
SUPERVISOR = runpy.run_path(str(ROOT / "docs/measurements/slotstream-20261026/supervisor.py"))
PROFILE = runpy.run_path(str(ROOT / "scripts/profile_flash.py"))
PARSE_VM = runpy.run_path(str(ROOT / "benchmarks/benchmark_flash_memory.py"))["parse_vm"]
STOP = SUPERVISOR["stop"]
QUIET = SUPERVISOR["check_quiet"]
TRACER_JOIN_SECONDS = 10
BASELINE_ENV = {
    "MLXL3_ALLOCATOR_CACHE_MIB": "64",
    "MLXL3_PROFILE_KIND": "natural",
    "MLXL3_SSD_GPU_HOT": "1",
    "MLXL3_SSD_GPU_HOT_WIDE": "1",
    "MLXL3_FLASH_MULTI_TOKEN": "1",
    "MLXL3_SSD_UNIQUE_COHORT": "1",
    "MLXL3_SSD_OVERLAP": "1",
    "MLXL3_SSD_SHARED_EXL3": "0",
    "MLXL3_SSD_CACHE_POLICY": "auto",
    "MLX_MAX_OPS_PER_BUFFER": "40",
    "MLX_MAX_MB_PER_BUFFER": "40",
    "MLXL3_SSD_CACHE_GIB": "3",
    "MLXL3_SSD_READERS": "4",
    "MLXL3_SSD_DYNAMIC_READERS": "0",
    "MLXL3_SSD_WEIGHTED_EPILOGUE": "0",
    "MLXL3_FLASH_GDN_EPILOGUE": "0",
    "MLXL3_FLASH_SWIGLU_DOWN": "0",
    "MLXL3_SSD_LOOKAHEAD_DEPTH": "0",
}


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tracer_command(kind, pid, output, options):
    command = [
        "/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace",
        "record",
        "--template",
        "Time Profiler",
    ]
    if kind == "combined":
        command += [
            "--instrument",
            "GPU",
            "--instrument",
            "Thread Activity",
            "--instrument",
            "System Call Trace",
        ]
    return command + [
        "--recording-options",
        str(options),
        "--attach",
        str(pid),
        "--output",
        str(output),
        "--window",
        "8s",
        "--time-limit",
        "45s",
    ]


def guard(record, baseline, minimum_available, pids, admission=False):
    """Reuse strict byte VM parsing and quiet/process-group policy; never reanchor."""
    vm = PARSE_VM(subprocess.check_output(["vm_stat"], text=True, timeout=5))
    record["last_vm_bytes"] = vm
    for key, limit in (("Swapouts", 0), ("Swapins", 256 << 20)):
        delta = vm[key] - baseline[key]
        field = f"maximum_live_{key.lower()}_bytes"
        record[field] = max(record.get(field, 0), delta)
        assert type(vm[key]) is int and 0 <= delta <= limit, key
    power = subprocess.check_output(["pmset", "-g", "batt"], text=True, timeout=5)
    assert power.splitlines()[0] == "Now drawing from 'AC Power'", "AC power required"
    QUIET(record, os.getpid())
    if admission:
        available = vm["Pages free"] + vm["File-backed pages"]
        record["available_estimate_bytes"] = available
        assert available >= minimum_available, "load headroom estimate"
    for pid in pids:
        raw = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, timeout=5)
        if raw.strip():
            rss = int(raw.strip()) * 1024
            record["maximum_sampled_process_rss_bytes"] = max(
                record.get("maximum_sampled_process_rss_bytes", 0), rss
            )
            assert 0 < rss <= 10 << 30, "RSS guard"


def validate_plan(plan, identity):
    assert plan["command"][1:] == [
        SELECTOR,
        "--exact",
        "--ignored",
        "--nocapture",
        "--test-threads=1",
    ]
    assert plan["mode"] in ("AR", "MTP", "DF") and plan["prompt"] in ("MAIN", "HOLD")
    assert plan["captures"] in (["combined"], ["combined", "cpu"])
    assert Path(plan["recording_options"]).is_absolute()
    assert sha(plan["recording_options"]) == plan["recording_options_sha256"]
    assert json.loads(Path(plan["recording_options"]).read_text()) == {
        "Time Profiler": {
            "contextSwitchSampling": True,
            "highFrequencySampling": True,
            "recordKernelStacks": True,
            "recordWaitingThreads": False,
        }
    }
    assert plan["require_quiet"] is True
    for field, lower, upper in (("timeout_seconds", 0, 900), ("silence_seconds", 0, 300)):
        value = plan[field]
        assert type(value) in (int, float) and math.isfinite(value) and lower < value <= upper
    assert (
        type(plan["minimum_available_bytes"]) is int
        and plan["minimum_available_bytes"] >= 10_500_000_000
    )
    env = plan["env"]
    assert all(env.get(key) == value for key, value in BASELINE_ENV.items()), "baseline flags"
    assert env["MLXL3_PROFILE_OUTPUT"] == plan["profile_directory"]
    assert env["MLXL3_FLASH_PROFILE_PROMPT"] == plan["prompt"]
    assert env["MLXL3_FLASH_PROFILE_MTP"] == {"AR": "0", "MTP": "1", "DF": "DFLASH2"}[plan["mode"]]
    assert all(Path(plan[key]).is_absolute() for key in ("profile_directory", "golden"))
    for name, digest in identity["sources"].items():
        assert sha(ROOT / name) == digest, name
    binaries = {value["path"]: value["sha256"] for value in identity["binaries"].values()}
    binary = plan["command"][0]
    assert binary in binaries and sha(binary) == binaries[binary], "binary identity"
    assert sha(plan["golden"]) == plan["golden_sha256"], "golden identity"


def validate_phase(marker, index, directory, plan, golden):
    assert index < len(PHASES) and marker["phase"] == PHASES[index], "phase order"
    if index == 0:
        started = json.loads((directory / "started.json").read_text())
        assert started["status"] == "incomplete" and started["kind"] == "natural"
        assert started["mtp"] is (plan["mode"] == "MTP") and started["prompt"] == plan["prompt"]
        assert type(started["budget"]) is int and started["budget"] == 128
    phase = marker["phase"]
    summary = json.loads((directory / f"{phase}-summary.json").read_text())
    assert summary["phase"] == phase
    assert summary["mtp"] is (plan["mode"] == "MTP") and summary["dflash"] is (plan["mode"] == "DF")
    assert summary["prompt"] == plan["prompt"] and summary["fenced"] is False
    traced = phase.startswith("trace_")
    assert summary["traced"] is traced
    PROFILE["validate_summary"](summary, "natural", golden)
    for part in ("prefill", "decode"):
        count = summary[f"{part}_event_count"]
        assert type(count) is int and (count > 0 if traced else count == 0)
        if traced:
            path = directory / f"{phase}-{part}.json"
            assert path.is_file() and path.stat().st_size > 0, "missing closed native trace"
    assert all(
        type(row["target_rows"]) is int and 0 <= row["target_rows"] <= 8 for row in summary["steps"]
    )
    for field in ("diagnostic_tps", "decode_seconds"):
        assert type(marker[field]) in (int, float) and math.isfinite(marker[field])
        assert marker[field] == summary[field], "marker/closed summary mismatch"
    assert (
        type(marker["epoch"]) in (int, float)
        and math.isfinite(marker["epoch"])
        and marker["epoch"] > 0
    )
    assert type(marker["events"]) is int
    assert marker["events"] == summary["prefill_event_count"] + summary["decode_event_count"]
    assert (marker["events"] > 0) is traced
    return summary


def run(plan_path, anchor_path, identity_path):
    """Run one immutable plan; SIGINT/exception still joins every owned process."""
    result_path = plan_path.with_name(plan_path.stem + "-result.json")
    with result_path.open("x") as stream:
        stream.write('{"status":"incomplete","parity":false}\n')
    result = {
        "status": "incomplete",
        "parity": False,
        "phases": [],
        "captures": [],
        "plan_sha256": sha(plan_path),
        "source_sha256": sha(__file__),
        "trace_coverage": "pending_external_TOC_and_time_info_inspection",
        "limits": "rolling 8s/time limit45s; no CPU/GPU timing or covered-transition claim",
    }
    target = tracer = None
    streams = []
    selector = selectors.DefaultSelector()

    def save():
        result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    def check(admission=False):
        pids = [
            child.pid for child in (target, tracer) if child is not None and child.poll() is None
        ]
        guard(result, baseline, plan["minimum_available_bytes"], pids, admission)

    def start_capture(index):
        nonlocal tracer
        path = plan_path.parent / f"{plan_path.stem}-capture-{index + 1}.trace"
        assert not path.exists(), "refusing to overwrite trace"
        command = tracer_command(
            plan["captures"][index], target.pid, path, plan["recording_options"]
        )
        log = path.with_suffix(".log").open("xb")
        streams.append(log)
        entry = {
            "command": command,
            "output": str(path),
            "status": "incomplete",
            "reaped": False,
            "started_epoch": time.time(),
        }
        result["captures"].append(entry)
        tracer = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=environment
        )
        entry["pid"] = tracer.pid
        save()

    def finish_capture():
        nonlocal tracer
        entry = result["captures"][-1]
        assert tracer.poll() is None, "tracer exited before stop marker"
        os.killpg(tracer.pid, signal.SIGINT)
        entry["stop_requested_epoch"] = time.time()
        end = time.monotonic() + TRACER_JOIN_SECONDS
        while tracer.poll() is None:
            assert time.monotonic() < end, "tracer join timeout"
            check()
            time.sleep(0.05)
        entry["exit"] = tracer.wait()
        entry["reaped"] = True
        assert entry["exit"] == 0, "tracer failed"
        path = Path(entry["output"])
        assert path.is_dir() and any(
            p.is_file() and p.stat().st_size > 0 for p in path.rglob("*")
        ), "empty trace"
        entry["status"] = "recorded_coverage_pending_inspection"
        tracer = None
        save()

    try:
        plan = json.loads(plan_path.read_text())
        identity = json.loads(identity_path.read_text())
        result["identity_sha256"] = sha(identity_path)
        validate_plan(plan, identity)
        anchor = json.loads(anchor_path.read_text())
        baseline = {key: anchor["vm_bytes"][key] for key in ("Swapouts", "Swapins")}
        assert all(type(value) is int and value >= 0 for value in baseline.values())
        result["original_vm_baseline"] = baseline
        golden = json.loads(Path(plan["golden"]).read_text())
        directory = Path(plan["profile_directory"])
        directory.mkdir(exist_ok=False)
        environment = {**os.environ, **plan["env"]}
        check(admission=True)
        log = plan_path.with_suffix(".log").open("xb")
        streams.append(log)
        target = subprocess.Popen(
            plan["command"],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        result["target_pid"] = target.pid
        os.set_blocking(target.stdout.fileno(), False)
        selector.register(target.stdout, selectors.EVENT_READ)
        started = last_output = last_guard = time.monotonic()
        pending = b""
        footer = 0
        summaries = []
        eof = False
        while not eof:
            now = time.monotonic()
            assert now - started < plan["timeout_seconds"], "target timeout"
            assert now - last_output < plan["silence_seconds"], "target silence timeout"
            if now - last_guard >= 2:
                check()
                last_guard = now
            if tracer is not None:
                assert tracer.poll() is None, "tracer exited before stop marker"
            for key, _ in selector.select(timeout=0.05):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    eof = True
                    break
                log.write(chunk)
                log.flush()
                last_output = time.monotonic()
                pending += chunk
                assert len(pending) <= 1 << 20, "unterminated/oversized stdout line"
                while b"\n" in pending:
                    raw, pending = pending.split(b"\n", 1)
                    line = raw.decode("utf-8")
                    if "FLASH_DETAILED_PHASE" in line:
                        prefix, payload = line.split("FLASH_DETAILED_PHASE ", 1)
                        assert not prefix.strip(), "unexpected phase marker prefix"
                        marker = json.loads(payload)
                        summary = validate_phase(marker, len(summaries), directory, plan, golden)
                        if summaries:
                            assert summary["tokens"] == summaries[0]["tokens"], (
                                "phase token divergence"
                            )
                        summaries.append(summary)
                        result["phases"].append(marker)
                        phase = marker["phase"]
                        if phase == "control_before":
                            start_capture(0)
                        elif phase == "trace_1":
                            finish_capture()
                            if len(plan["captures"]) == 2:
                                start_capture(1)
                        elif phase == "trace_2" and tracer is not None:
                            finish_capture()
                        save()
                    if line.startswith("test result:"):
                        assert re.search(r"\bok\. 1 passed; 0 failed; 0 ignored;", line), (
                            "native test count/status"
                        )
                        footer += 1
        assert not pending.strip(), "unterminated stdout"
        result["target_exit"] = target.wait(timeout=5)
        result["target_reaped"] = True
        assert result["target_exit"] == 0, "native exit"
        assert footer == 1 and len(summaries) == 6, "missing native completion/phases"
        assert tracer is None and len(result["captures"]) == len(plan["captures"])
        complete = json.loads((directory / "complete.json").read_text())
        final = json.loads((directory / "final-snapshot.json").read_text())
        memory = final["memory"]
        assert (
            0
            < memory["process_footprint_bytes"]
            <= memory["process_lifetime_peak_bytes"]
            <= 12_000_000_000
        )
        PROFILE["vm_counts"](final["vm_stat"])
        assert complete["kind"] == "natural" and complete["summaries"] == summaries, (
            "complete/closed summary mismatch"
        )
        assert complete["mtp"] is (plan["mode"] == "MTP") and complete["dflash"] is (
            plan["mode"] == "DF"
        )
        assert complete["prompt"] == plan["prompt"]
        analysis = PROFILE["analyze"](directory, "natural", golden, summaries[0]["tokens"])
        assert analysis["status"] == "validated_diagnostic_external_golden_full_internal_parity"
        check()
        validate_plan(plan, identity)
        result.update(status="native_validated_captures_pending_inspection", parity=True)
    except BaseException as error:
        result.update(status="failed_or_interrupted", parity=False, error=repr(error))
        raise
    finally:
        cleanup = []
        for label, child in (("tracer", tracer), ("target", target)):
            if child is not None:
                try:
                    STOP(child)
                    result[label + "_reaped"] = child.poll() is not None
                    if label == "tracer":
                        result["captures"][-1]["reaped"] = child.poll() is not None
                except BaseException as error:  # noqa: BLE001 -- join the other child even on interruption
                    cleanup.append(repr(error))
        if cleanup:
            result.update(status="failed_or_interrupted", parity=False, cleanup_errors=cleanup)
        selector.close()
        if target is not None and target.stdout is not None:
            target.stdout.close()
        for stream in streams:
            stream.close()
        save()
        if cleanup:
            raise RuntimeError(f"process cleanup failed: {cleanup}")
    return result


if __name__ == "__main__":
    run(*(Path(value) for value in sys.argv[1:]))
