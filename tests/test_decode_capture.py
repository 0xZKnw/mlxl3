"""Exercise the real controller with disposable CPU children, never xctrace/GPU."""

import copy
import hashlib
import json
import os
import runpy
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/measurements/decode-profile-20261031/capture.py"
PHASES = ("warmup", "control_before", "trace_1", "trace_2", "trace_3", "control_after")
NATIVE = r"""
import json, os, pathlib, sys, time
data = json.loads(pathlib.Path(os.environ["FIXTURE"]).read_text())
out = pathlib.Path(os.environ["MLXL3_PROFILE_OUTPUT"])
fault = os.environ.get("FAULT", "")
if fault == "silence":
    time.sleep(10)
if fault == "native_exit":
    sys.exit(7)
print("running 1 test", flush=True)
if fault != "missing_started":
    (out / "started.json").write_text(json.dumps({"status":"incomplete", "kind":"natural", "mtp":False, "prompt":"MAIN", "budget":128, "epoch":1.0}))
for index, summary in enumerate(data["summaries"]):
    time.sleep(0.12)
    phase = summary["phase"]
    if phase == "trace_1":
        # Wait for the disposable tracer's handler, independent of Python startup speed.
        ready = pathlib.Path(os.environ["TRACER_READY_PATH"])
        deadline = time.monotonic() + 2
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
    if fault == "missing_phase" and index == 2:
        continue
    if fault != "missing_file" or index != 1:
        (out / (phase + "-summary.json")).write_text(json.dumps(summary))
    if summary["traced"]:
        for part in ("prefill", "decode"):
            (out / (phase + "-" + part + ".json")).write_text(json.dumps(data["events"]))
    marker = {"phase": phase, "diagnostic_tps": summary["diagnostic_tps"],
              "decode_seconds": summary["decode_seconds"], "events": summary["prefill_event_count"] + summary["decode_event_count"], "epoch": 1.0}
    if fault == "bad_marker" and index == 1:
        print("FLASH_DETAILED_PHASE {", flush=True)
    else:
        print("FLASH_DETAILED_PHASE " + json.dumps(marker), flush=True)
    if fault == "duplicate" and index == 1:
        print("FLASH_DETAILED_PHASE " + json.dumps(marker), flush=True)
    if fault == "hanging_target" and index == 1:
        time.sleep(10)
complete = {"status": "complete_diagnostic_internal_full_parity_external_golden_pending",
            "profile_schema_version": 1, "kind": "natural", "mtp": False,
            "dflash": False, "prompt": "MAIN", "load_seconds": 0.1, "summaries": data["summaries"]}
if fault != "missing_complete":
    (out / "complete.json").write_text(json.dumps(complete))
if fault != "missing_final":
    (out / "final-snapshot.json").write_text(json.dumps(data["summaries"][-1]["after"]))
print("test result: ok. " + ("0 passed; 0 failed; 1 ignored;" if fault == "zero_tests" else "1 passed; 0 failed; 0 ignored;") + " 0 measured; 0 filtered out", flush=True)
if fault == "late_exit":
    sys.exit(3)
"""
TRACER = r"""
import pathlib, signal, sys, time
out, fault = pathlib.Path(sys.argv[1]), sys.argv[2]
if fault == "startup_delay":
    time.sleep(0.4)
if fault == "early":
    sys.exit(4)
if fault in ("ignore_term", "ignore_all"):
    signal.signal(signal.SIGTERM, lambda sig, frame: out.with_suffix(".terminating").write_text("TERM ignored"))
def done(sig, frame):
    if fault in ("hang", "ignore_all"):
        out.with_suffix(".stopping").write_text("SIGINT received")
        return
    if fault == "delayed":
        time.sleep(0.25)
    out.mkdir()
    if fault != "empty":
        (out / "record.data").write_bytes(b"synthetic trace; coverage unqualified")
    sys.exit(5 if fault == "exit" else 0)
signal.signal(signal.SIGINT, done)
out.with_suffix(".ready").write_text("signal handlers installed")
while True:
    time.sleep(0.05)
"""


def fixture_data():
    api = runpy.run_path(str(ROOT / "tests/test_flash_profile.py"))
    summary = api["summary_fixture"]()
    tokens = list(range(128))
    value = 0xCBF29CE484222325
    for token in tokens:
        for byte in token.to_bytes(4, "little"):
            value = ((value ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
    events = [{"name": "decode", "parent": None, "start_ns": 0, "end_ns": 1000, "data": None}]
    for layer in range(48):
        start = 10 * layer
        parent = len(events)
        events.append(
            {
                "name": f"layer_{layer}",
                "parent": 0,
                "start_ns": start,
                "end_ns": start + 9,
                "data": None,
            }
        )
        events.append(
            {
                "name": "moe_forward_rows",
                "parent": parent,
                "start_ns": start + 1,
                "end_ns": start + 8,
                "data": {
                    "rows": 1,
                    "capacity": 53,
                    "ids": list(range(10)),
                    "slots_hit": [[i, True] for i in range(10)],
                    "hot_rows": 10,
                    "cold_rows": 0,
                },
            }
        )
    summary.update(
        tokens=tokens,
        token_hash=f"{value:016x}",
        mtp=False,
        dflash=False,
        full_draft_states_byte_exact=None,
        dflash_stats=None,
        dflash_profile=None,
        prompt="MAIN",
        fenced=False,
        after_counts=[[20, 0, 0]] * 48,
        diagnostic_tps=63.5,
        steps=[
            {"step": i, "token": tokens[i], "seconds": 0.01, "target_rows": 1}
            for i in range(1, 128)
        ],
    )
    summaries = []
    for phase in PHASES:
        traced = phase.startswith("trace_")
        summaries.append(
            {
                **copy.deepcopy(summary),
                "phase": phase,
                "traced": traced,
                "test_stage_retention": phase == "warmup",
                "test_stage_count": 1 if phase == "warmup" else 0,
                "prefill_event_count": len(events) if traced else 0,
                "decode_event_count": len(events) if traced else 0,
            }
        )
    return {"summaries": summaries, "events": events}, {
        "cache_context": summary["text"],
        "token_hash": summary["token_hash"],
    }


def select_capture(plan, kind):
    plan["captures"] = [kind]
    fine = kind == "cpu"
    path = Path(plan["recording_options"])
    path.write_text(
        json.dumps(
            {
                "Time Profiler": {
                    "contextSwitchSampling": fine,
                    "highFrequencySampling": fine,
                    "recordKernelStacks": fine,
                    "recordWaitingThreads": False,
                }
            }
        )
    )
    plan["recording_options_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    api = runpy.run_path(str(SOURCE))
    namespace = api["run"].__globals__
    monkeypatch.setitem(namespace, "guard", lambda *args, **kwargs: None)
    data, golden = fixture_data()
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps(data))
    golden_path = tmp_path / "golden.json"
    golden_path.write_text(json.dumps(golden))
    options = tmp_path / "options.json"
    options.write_text(
        json.dumps(
            {
                "Time Profiler": {
                    "contextSwitchSampling": False,
                    "highFrequencySampling": False,
                    "recordKernelStacks": False,
                    "recordWaitingThreads": False,
                }
            }
        )
    )
    native = tmp_path / "fixture-child"
    native.write_text(f"#!{sys.executable}\n" + NATIVE)
    native.chmod(0o700)
    tracer = tmp_path / "fixture-tracer.py"
    tracer.write_text(TRACER)
    profile = tmp_path / "profile"
    plan = {
        "command": [
            str(native),
            api["SELECTOR"],
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ],
        "env": {
            **api["BASELINE_ENV"],
            "MLXL3_PROFILE_OUTPUT": str(profile),
            "MLXL3_FLASH_PROFILE_PROMPT": "MAIN",
            "MLXL3_FLASH_PROFILE_MTP": "0",
            "FIXTURE": str(fixture),
            "TRACER_READY_PATH": str(tmp_path / "plan-capture-1.ready"),
        },
        "profile_directory": str(profile),
        "golden": str(golden_path),
        "golden_sha256": api["sha"](golden_path),
        "mode": "AR",
        "prompt": "MAIN",
        "captures": ["combined"],
        "recording_options": str(options),
        "recording_options_sha256": api["sha"](options),
        "require_quiet": True,
        "minimum_available_bytes": 10_500_000_000,
        "timeout_seconds": 8,
        "silence_seconds": 2,
    }
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "sources": {},
                "binaries": {"native": {"path": str(native), "sha256": api["sha"](native)}},
            }
        )
    )
    anchor = tmp_path / "anchor.json"
    anchor.write_text('{"vm_bytes":{"Swapins":0,"Swapouts":0}}')
    path = tmp_path / "plan.json"
    tracer_fault = [""]
    monkeypatch.setitem(
        namespace,
        "tracer_command",
        lambda kind, pid, out, options: [sys.executable, str(tracer), str(out), tracer_fault[0]],
    )

    def execute():
        path.write_text(json.dumps(plan))
        fixture.write_text(json.dumps(data))
        previous = tuple(signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM))
        try:
            return api["run"](path, anchor, identity)
        finally:
            assert (
                tuple(signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)) == previous
            ), "signal handlers not restored"

    return execute, plan, data, path, tracer_fault, api, namespace


def assert_failed_and_joined(path):
    result = json.loads(path.with_name("plan-result.json").read_text())
    assert result["status"] == "failed_or_interrupted" and result["parity"] is False
    if "target_pid" in result:
        assert result["target_reaped"] is True
        with pytest.raises(ProcessLookupError):
            os.kill(result["target_pid"], 0)
    for capture in result["captures"]:
        assert capture["reaped"] is True
        with pytest.raises(ProcessLookupError):
            os.kill(capture["pid"], 0)
    return result


@pytest.mark.parametrize("kind", ["combined", "cpu"])
def test_six_phases_complete_and_all_owned_processes_joined(setup, kind):
    execute, plan, _, path, _, _, _ = setup
    select_capture(plan, kind)
    result = execute()
    assert (
        result["status"] == "native_validated_captures_pending_inspection"
        and result["parity"] is True
    )
    assert [phase["phase"] for phase in result["phases"]] == list(PHASES)
    assert len(result["captures"]) == 1 and all(c["reaped"] for c in result["captures"])
    assert all(c["command"][0] == sys.executable for c in result["captures"])
    assert result["trace_coverage"].startswith("pending_") and result["target_reaped"]
    assert (Path(plan["profile_directory"]) / "analysis.json").is_file()
    original = path.with_name("plan-result.json").read_bytes()
    with pytest.raises(FileExistsError):
        execute()
    assert path.with_name("plan-result.json").read_bytes() == original


def test_tracer_startup_does_not_race_the_first_trace_stop(setup):
    execute, _, _, _, tracer_fault, _, _ = setup
    tracer_fault[0] = "startup_delay"
    result = execute()
    assert result["parity"] is True
    assert result["target_reaped"] and all(c["reaped"] for c in result["captures"])


@pytest.mark.parametrize(
    "fault",
    [
        "silence",
        "native_exit",
        "missing_phase",
        "missing_started",
        "missing_final",
        "missing_file",
        "bad_marker",
        "duplicate",
        "missing_complete",
        "zero_tests",
        "late_exit",
        "hanging_target",
    ],
)
def test_native_failures_remain_failed_and_join_every_child(setup, fault):
    execute, plan, _, path, _, _, _ = setup
    plan["env"]["FAULT"] = fault
    if fault in ("silence", "hanging_target"):
        plan["silence_seconds"] = 0.35
    with pytest.raises((AssertionError, FileNotFoundError, ValueError)):
        execute()
    assert_failed_and_joined(path)


@pytest.mark.parametrize("fault", ["early", "empty", "exit"])
def test_tracer_failure_never_validates_native_parity(setup, fault):
    execute, _, _, path, tracer_fault, _, _ = setup
    tracer_fault[0] = fault
    with pytest.raises(AssertionError):
        execute()
    assert_failed_and_joined(path)


@pytest.mark.parametrize(
    "fault", ["hash", "count", "nonfinite", "mode", "tokens", "events", "fenced"]
)
def test_closed_summary_corruption_is_refused_before_attach(setup, fault):
    execute, _, data, path, _, _, _ = setup
    summary = data["summaries"][1]
    if fault == "hash":
        summary["token_hash"] = "0" * 16
    elif fault == "count":
        summary["logit_count"] = 0
    elif fault == "nonfinite":
        summary["decode_seconds"] = float("nan")
    elif fault == "mode":
        summary["mtp"] = True
    elif fault == "tokens":
        summary["tokens"][0] = 248320
    elif fault == "events":
        summary["decode_event_count"] = 1
    else:
        summary["fenced"] = True
    with pytest.raises((AssertionError, ValueError)):
        execute()
    assert not assert_failed_and_joined(path)["captures"]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize(
    "fault",
    [
        "missing_retention",
        "missing_count",
        "integer_retention",
        "null_retention",
        "string_retention",
        "boolean_count",
        "float_count",
        "string_count",
        "null_count",
        "negative_count",
        "wrong_policy",
        "wrong_count",
    ],
)
def test_stage_contract_is_strict_before_golden_and_attach(setup, monkeypatch, index, fault):
    execute, _, data, path, _, _, namespace = setup
    summary = data["summaries"][index]
    if fault == "missing_retention":
        del summary["test_stage_retention"]
    elif fault == "missing_count":
        del summary["test_stage_count"]
    elif fault == "integer_retention":
        summary["test_stage_retention"] = int(summary["test_stage_retention"])
    elif fault == "null_retention":
        summary["test_stage_retention"] = None
    elif fault == "string_retention":
        summary["test_stage_retention"] = str(summary["test_stage_retention"])
    elif fault == "boolean_count":
        summary["test_stage_count"] = bool(summary["test_stage_count"])
    elif fault == "float_count":
        summary["test_stage_count"] = float(summary["test_stage_count"])
    elif fault == "string_count":
        summary["test_stage_count"] = str(summary["test_stage_count"])
    elif fault == "null_count":
        summary["test_stage_count"] = None
    elif fault == "negative_count":
        summary["test_stage_count"] = -1
    elif fault == "wrong_policy":
        summary["test_stage_retention"] = index != 0
    else:
        summary["test_stage_count"] = 0 if index == 0 else 1
    visited = []
    original = namespace["PROFILE"]["validate_summary"]

    def checked(*args):
        visited.append(args[0]["phase"])
        return original(*args)

    monkeypatch.setitem(namespace["PROFILE"], "validate_summary", checked)
    with pytest.raises((AssertionError, KeyError)):
        execute()
    assert visited == list(PHASES[:index])
    assert not assert_failed_and_joined(path)["captures"]


@pytest.mark.parametrize("index", [2, 3, 4, 5])
def test_residual_stages_in_later_phases_never_validate_complete_run(setup, index):
    execute, _, data, path, _, _, _ = setup
    data["summaries"][index]["test_stage_count"] = 1
    with pytest.raises(AssertionError):
        execute()
    assert_failed_and_joined(path)


def test_exception_after_attach_cleans_target_and_tracer(setup, monkeypatch):
    execute, _, _, path, _, api, namespace = setup
    original = api["validate_phase"]

    def interrupted(marker, index, *args):
        if index == 2:
            raise KeyboardInterrupt("injected interruption")
        return original(marker, index, *args)

    monkeypatch.setitem(namespace, "validate_phase", interrupted)
    with pytest.raises(KeyboardInterrupt):
        execute()
    assert_failed_and_joined(path)


@pytest.mark.parametrize("scenario", ["term", "term_join", "int_int", "term_int"])
def test_real_sigterm_joins_both_process_groups_and_restores_handler(setup, scenario):
    _, plan, data, path, _, _, _ = setup
    during_join = scenario == "term_join"
    twice = scenario in ("int_int", "term_int")
    plan["env"]["FAULT"] = "" if during_join else "hanging_target"
    path.write_text(json.dumps(plan))
    Path(plan["env"]["FIXTURE"]).write_text(json.dumps(data))
    wrapper = path.parent / "controller-fixture.py"
    restored = path.parent / "handler-restored.json"
    wrapper.write_text(
        "import json, pathlib, runpy, signal, sys\n"
        "source, plan, anchor, identity, tracer, fault, restored = sys.argv[1:]\n"
        "api = runpy.run_path(source)\n"
        "ns = api['run'].__globals__\n"
        "ns['guard'] = lambda *args, **kwargs: None\n"
        "ns['tracer_command'] = lambda kind, pid, out, opts: [sys.executable, tracer, str(out), fault]\n"
        "previous = tuple(signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM))\n"
        "try:\n"
        "    api['run'](pathlib.Path(plan), pathlib.Path(anchor), pathlib.Path(identity))\n"
        "finally:\n"
        "    pathlib.Path(restored).write_text(json.dumps(tuple(signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)) == previous))\n"
    )
    owned = []
    with (path.parent / "controller-fixture.log").open("wb") as log:
        controller = subprocess.Popen(
            [
                sys.executable,
                str(wrapper),
                str(SOURCE),
                str(path),
                str(path.parent / "anchor.json"),
                str(path.parent / "identity.json"),
                str(path.parent / "fixture-tracer.py"),
                "hang" if during_join else ("ignore_term" if twice else ""),
                str(restored),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 5
            result_path = path.with_name("plan-result.json")
            stopping = path.parent / "plan-capture-1.stopping"
            ready = path.parent / "plan-capture-1.ready"
            while time.monotonic() < deadline:
                if result_path.exists():
                    try:
                        receipt = json.loads(result_path.read_text())
                    except json.JSONDecodeError:
                        receipt = {}
                    if receipt.get("captures"):
                        owned = [receipt["target_pid"], receipt["captures"][0]["pid"]]
                        if (not during_join or stopping.exists()) and (not twice or ready.exists()):
                            break
                assert controller.poll() is None, "controller exited before signal"
                time.sleep(0.01)
            assert len(owned) == 2 and (not during_join or stopping.exists()), "children not ready"
            for pid in owned:
                os.killpg(pid, 0)
            controller.send_signal(signal.SIGINT if scenario == "int_int" else signal.SIGTERM)
            if twice:
                terminating = path.parent / "plan-capture-1.terminating"
                deadline = time.monotonic() + 3
                while not terminating.exists() and time.monotonic() < deadline:
                    assert controller.poll() is None, "controller exited before second signal"
                    time.sleep(0.01)
                assert terminating.exists(), "tracer did not enter bounded cleanup wait"
                controller.send_signal(signal.SIGINT)
            assert controller.wait(timeout=8) != 0
            assert_failed_and_joined(path)
            assert json.loads(restored.read_text()) is True
            for pid in owned:
                with pytest.raises(ProcessLookupError):
                    os.killpg(pid, 0)
        finally:
            # Also clean the deliberately broken pre-fix reproduction/mutation.
            for pid in owned + [controller.pid]:
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            controller.wait(timeout=5)


@pytest.mark.parametrize("identity_refused", [False, True])
def test_custom_sigterm_handler_is_restored_on_success_and_prelaunch_refusal(
    setup, identity_refused
):
    execute, _, _, path, _, _, _ = setup
    previous = tuple(signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM))

    def custom(signum, frame):
        raise AssertionError("unexpected signal during handler-restoration test")

    signal.signal(signal.SIGTERM, custom)
    signal.signal(signal.SIGINT, custom)
    try:
        if identity_refused:
            identity_path = path.parent / "identity.json"
            identity = json.loads(identity_path.read_text())
            identity["binaries"]["native"]["sha256"] = "0" * 64
            identity_path.write_text(json.dumps(identity))
            with pytest.raises(AssertionError, match="binary identity"):
                execute()
            assert "target_pid" not in assert_failed_and_joined(path)
        else:
            assert execute()["parity"] is True
        assert signal.getsignal(signal.SIGTERM) is custom
        assert signal.getsignal(signal.SIGINT) is custom
    finally:
        for sig, handler in zip((signal.SIGINT, signal.SIGTERM), previous, strict=True):
            signal.signal(sig, handler)


def test_tracer_ignoring_sigint_is_bounded_and_joined(setup, monkeypatch):
    execute, _, _, path, tracer_fault, _, namespace = setup
    tracer_fault[0] = "ignore_all"
    monkeypatch.setitem(namespace, "TRACER_JOIN_SECONDS", 0.2)
    original_stop = namespace["STOP"]
    exits = []

    def stopped(child):
        original_stop(child)
        exits.append(child.returncode)

    monkeypatch.setitem(namespace, "STOP", stopped)
    with pytest.raises(AssertionError, match="tracer join timeout"):
        execute()
    assert_failed_and_joined(path)
    assert exits[0] == -signal.SIGKILL


@pytest.mark.parametrize("join_seconds", [0.1, 0.8])
def test_graceful_delayed_flush_obeys_join_deadline_without_waiting_sixty_seconds(
    setup, monkeypatch, join_seconds
):
    execute, plan, _, path, tracer_fault, _, namespace = setup
    assert namespace["TRACER_JOIN_SECONDS"] == 60
    select_capture(plan, "cpu")
    tracer_fault[0] = "delayed"
    monkeypatch.setitem(namespace, "TRACER_JOIN_SECONDS", join_seconds)
    if join_seconds < 0.25:
        with pytest.raises(AssertionError, match="tracer join timeout"):
            execute()
        assert_failed_and_joined(path)
    else:
        result = execute()
        assert result["parity"] is True and len(result["phases"]) == 6
        assert result["captures"][0]["exit"] == 0 and result["captures"][0]["reaped"] is True


def test_target_total_timeout_is_bounded_and_joins_active_tracer(setup):
    execute, plan, _, path, _, _, _ = setup
    plan["env"]["FAULT"] = "hanging_target"
    plan["timeout_seconds"] = 0.5
    with pytest.raises(AssertionError, match="target timeout"):
        execute()
    assert_failed_and_joined(path)


def test_guard_failure_after_attach_is_durable_and_joins_both(setup, monkeypatch):
    execute, _, _, path, _, _, namespace = setup

    def failed_guard(record, baseline, minimum, pids, admission=False):
        if len(pids) == 2:
            raise RuntimeError("injected VM guard failure")

    monkeypatch.setitem(namespace, "guard", failed_guard)
    with pytest.raises(RuntimeError, match="VM guard failure"):
        execute()
    assert_failed_and_joined(path)


@pytest.mark.parametrize("wait_failure", [None, "timeout", "error"])
def test_cleanup_permission_error_requires_bounded_join_and_continues_other_child(
    setup, monkeypatch, wait_failure
):
    execute, _, data, path, _, _, namespace = setup
    data["summaries"][2]["logit_count"] = 0
    original_stop = namespace["STOP"]
    owned = []
    attempts = []

    def denied_stop(child):
        original_wait = child.wait
        owned.append((child, original_wait))

        def waited(timeout):
            attempts.append((child.pid, timeout))
            if wait_failure == "timeout":
                raise subprocess.TimeoutExpired(child.args, timeout)
            if wait_failure == "error":
                raise OSError("injected join failure")
            return original_wait(timeout=timeout)

        monkeypatch.setattr(child, "wait", waited)
        if wait_failure is None:
            child.terminate()
        raise PermissionError(1, "injected exit-time EPERM")

    monkeypatch.setitem(namespace, "STOP", denied_stop)
    try:
        if wait_failure is None:
            with pytest.raises(AssertionError):
                execute()
            receipt = assert_failed_and_joined(path)
            assert not receipt.get("cleanup_errors")
        else:
            with pytest.raises(RuntimeError, match="process cleanup failed"):
                execute()
            receipt = json.loads(path.with_name("plan-result.json").read_text())
            assert receipt["status"] == "failed_or_interrupted" and receipt["parity"] is False
            assert len(receipt["cleanup_errors"]) == 2
            assert not receipt["captures"][0]["reaped"]
            assert not receipt.get("target_reaped", False)
        assert len(owned) == 2 and attempts == [(child.pid, 5) for child, _ in owned]
    finally:
        for child, original_wait in owned:
            child.wait = original_wait
            original_stop(child)
            with pytest.raises(ProcessLookupError):
                os.killpg(child.pid, 0)


def test_capture_command_contains_preregistered_instruments_and_options():
    api = runpy.run_path(str(SOURCE))
    for kind in ("combined", "cpu"):
        command = api["tracer_command"](kind, 123, Path("/fresh.trace"), Path("/options.json"))
        assert command[-4:] == ["--window", "8s", "--time-limit", "45s"]
        assert command[command.index("--attach") + 1] == "123"
        assert command[command.index("--recording-options") + 1] == "/options.json"
        assert command.count("--instrument") == (3 if kind == "combined" else 0)


def test_guard_records_swap_failure_before_rejection(monkeypatch):
    api = runpy.run_path(str(SOURCE))
    namespace = api["guard"].__globals__
    monkeypatch.setitem(namespace, "PARSE_VM", lambda raw: {"Swapins": 0, "Swapouts": 16384})
    monkeypatch.setattr(namespace["subprocess"], "check_output", lambda *args, **kwargs: "unused")
    record = {}
    with pytest.raises(AssertionError, match="Swapouts"):
        api["guard"](record, {"Swapins": 0, "Swapouts": 0}, 10_500_000_000, [])
    assert record["maximum_live_swapouts_bytes"] == 16384


def test_plan_refuses_lowered_admission_before_launch(setup):
    execute, plan, _, path, _, _, _ = setup
    plan["minimum_available_bytes"] -= 1
    with pytest.raises(AssertionError):
        execute()
    result = assert_failed_and_joined(path)
    assert "target_pid" not in result


def test_plan_refuses_two_captures_before_launch(setup):
    execute, plan, _, path, _, _, _ = setup
    plan["captures"] = ["combined", "cpu"]
    with pytest.raises(AssertionError, match="one capture per process"):
        execute()
    assert "target_pid" not in assert_failed_and_joined(path)


@pytest.mark.parametrize("kind", ["combined", "cpu"])
@pytest.mark.parametrize(
    "fault",
    [
        "contextSwitchSampling",
        "highFrequencySampling",
        "recordKernelStacks",
        "recordWaitingThreads",
        "missing",
        "extra",
        "integer",
        "outer",
    ],
)
def test_capture_options_are_exact_and_kind_specific_before_launch(setup, kind, fault):
    execute, plan, _, path, _, _, _ = setup
    select_capture(plan, kind)
    options_path = Path(plan["recording_options"])
    options = json.loads(options_path.read_text())
    values = options["Time Profiler"]
    if fault in values:
        values[fault] = not values[fault]
    elif fault == "missing":
        del values["recordKernelStacks"]
    elif fault == "extra":
        values["extra"] = False
    elif fault == "integer":
        values["highFrequencySampling"] = int(values["highFrequencySampling"])
    else:
        options["extra"] = {}
    options_path.write_text(json.dumps(options))
    plan["recording_options_sha256"] = hashlib.sha256(options_path.read_bytes()).hexdigest()
    with pytest.raises(AssertionError, match="recording options"):
        execute()
    assert "target_pid" not in assert_failed_and_joined(path)


@pytest.mark.parametrize(
    "raw,code,error,accepted",
    [
        ("Z 0\n", 0, "", True),
        ("Z+ 0\n", 0, "", True),
        pytest.param("?Es 0\n", 0, "", True, id="darwin-exiting-zero"),
        ("E 0\n", 0, "", True),
        ("?Es 1\n", 0, "", True),
        ("?Es -1\n", 0, "", False),
        ("", 1, "", True),
        ("", 0, "", True),
        ("S 1\n", 0, "", True),
        ("R 10485760\n", 0, "", True),
        ("R 0\n", 0, "", False),
        ("S 0\n", 0, "", False),
        ("?s 0\n", 0, "", False),
        ("S 10485761\n", 0, "", False),
        ("S -1\n", 0, "", False),
        ("S 1\nR 2\n", 0, "", False),
        ("", 1, "denied", False),
        ("", 2, "", False),
        ("S 1\n", 1, "", False),
    ],
)
def test_rss_guard_ignores_only_terminal_or_missing_processes(
    monkeypatch, raw, code, error, accepted
):
    api = runpy.run_path(str(SOURCE))
    namespace = api["guard"].__globals__
    vm = {"Swapins": 0, "Swapouts": 0, "Pages free": 11_000_000_000, "File-backed pages": 0}
    monkeypatch.setitem(namespace, "PARSE_VM", lambda raw: vm)
    monkeypatch.setitem(namespace, "QUIET", lambda *args: None)
    monkeypatch.setattr(
        namespace["subprocess"],
        "check_output",
        lambda command, **kwargs: (
            "Now drawing from 'AC Power'\n" if command[0] == "pmset" else "vm"
        ),
    )

    def sample(command, **kwargs):
        assert command == ["ps", "-o", "stat=,rss=", "-p", "123"]
        assert kwargs == {"text": True, "capture_output": True, "timeout": 5, "check": False}
        return subprocess.CompletedProcess(command, code, raw, error)

    monkeypatch.setattr(namespace["subprocess"], "run", sample)
    record = {}
    if accepted:
        api["guard"](record, {"Swapins": 0, "Swapouts": 0}, 10_500_000_000, [123], True)
        assert record["last_owned_process_rss_samples"][0]["pid"] == 123
        assert record.get("maximum_sampled_process_rss_bytes", 0) == (
            int(raw.split()[1]) * 1024
            if raw and not raw.startswith("Z") and "E" not in raw.split()[0]
            else 0
        )
    else:
        with pytest.raises(AssertionError):
            api["guard"](record, {"Swapins": 0, "Swapouts": 0}, 10_500_000_000, [123], True)
