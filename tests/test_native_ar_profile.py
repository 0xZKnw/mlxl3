"""Exercise the real controller with disposable CLI/trace processes, never Metal."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("native_ar_profile", SCRIPTS / "profile_native_ar.py")
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)
sys.path.pop(0)

CLI = r"""
import json,sys,time
from pathlib import Path
mode=sys.argv[1]
if mode=='silent': time.sleep(30)
args=sys.argv[2:]
model=args[args.index('bridge')+1]
loading=dict(type='loading',model='wrong' if mode=='loading-wrong' else model)
if mode not in ('loading-missing','loading-late'): print(json.dumps(loading),flush=True)
if mode=='loading-duplicate': print(json.dumps(loading),flush=True)
ready=dict(type='ready',backend='direct-metal',runtime_profile='release',context_limit=4096,
           runtime_executable=str(Path(sys.executable).resolve()),model_path=model,runtime_commit='fake-source')
if mode=='backend': ready['backend']='mlx'
if mode!='ready-late': print(json.dumps(ready),flush=True)
if mode=='ready-duplicate': print(json.dumps(ready),flush=True)
if mode=='loading-late': print(json.dumps(loading),flush=True)
for line in sys.stdin:
 request=json.loads(line)
 if request['type']=='shutdown': break
 identifier=request['request_id']
 print(json.dumps(dict(type='generation_mode',request_id=identifier,mtp_active=False,dflash_active=False)),flush=True)
 if mode=='ready-late' and identifier=='native-ar-warmup': print(json.dumps(ready),flush=True)
 budget=request['max_tokens']
 stats=dict(generated_tokens=budget,decode_tokens=budget-1,cached_prompt_tokens=0,context_limit=4096,
            decode_tps=10.,decode_seconds=(budget-1)/10.,elapsed_seconds=3.2,ttft_seconds=.1,prefill_tps=1.)
 if mode=='nonfinite': stats['decode_tps']=float('nan')
 if mode=='counts': stats['decode_tokens']=30
 event=dict(type='complete',request_id=identifier,stats=stats,cache_context='golden fixture',token_hash='0123456789abcdef')
 if mode=='diverge': event['cache_context']='wrong fixture'
 if mode=='missing' and identifier=='native-ar-measured': continue
 if mode=='truncated' and identifier=='native-ar-measured':
  sys.stdout.write(json.dumps(event));sys.stdout.flush();break
 print(json.dumps(event),flush=True)
 if mode=='duplicate': print(json.dumps(event),flush=True)
if mode=='invalid': print('{',flush=True)
if mode=='exit': raise SystemExit(7)
"""

TOOL = r"""
import subprocess,sys
from pathlib import Path
args=sys.argv[1:]
trace=Path(args[args.index('--output')+1])
source=Path(args[args.index('--target-stdin')+1])
output=Path(args[args.index('--target-stdout')+1])
command=args[args.index('--')+1:]
MODE='success'
if MODE=='denied':
 print('Permission denied by fake trace service',file=sys.stderr);raise SystemExit(9)
with source.open('rb') as stdin,output.open('wb') as stdout:
 if MODE in ('detach','dangling'):
  child=subprocess.Popen(command,stdin=stdin,stdout=stdout,start_new_session=True)
  (trace.parent/'detached.pid').write_text(str(child.pid))
  if MODE=='detach':raise SystemExit(8)
  import time;time.sleep(.1)
  trace.mkdir();(trace/'fake.data').write_text('fake');raise SystemExit(0)
 result=subprocess.run(command,stdin=stdin,stdout=stdout,check=False)
trace.mkdir()
if MODE!='empty': (trace/'fake.data').write_text('synthetic trace, not coverage')
raise SystemExit(result.returncode)
"""


class NativeArProfileTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="ar31-controller-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.engine = Path(sys.executable).resolve()
        if sys.platform == "darwin":
            # Python's macOS launcher re-execs Python.app with a different argv[0].
            self.engine = Path(
                subprocess.check_output(
                    ["/bin/ps", "-p", str(os.getpid()), "-o", "comm="], text=True
                ).strip()
            ).resolve()
        self.fake = self.root / "fake_cli.py"
        # Rust current_exe reports the actual binary, unlike Python's launcher alias.
        self.fake.write_text(
            CLI.replace("str(Path(sys.executable).resolve())", repr(str(self.engine)))
        )
        self.model = self.root / "model"
        self.model.mkdir()
        metadata = {}
        for name in (
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            path = self.model / name
            path.write_text("{}")
            metadata[str(path)] = PROFILE.sha(path)
        self.golden = self.root / "golden.json"
        self.golden.write_text(
            json.dumps({"runs": [{"text": "golden fixture", "token_hash": "0123456789abcdef"}]})
        )
        self.options = self.root / "options.json"
        self.options.write_text('{"Time Profiler":{"recordWaitingThreads":true}}')
        self.plan = {
            "engine": {"path": str(self.engine), "sha256": PROFILE.sha(self.engine)},
            "model": {"path": str(self.model), "metadata": metadata},
            "golden": {"path": str(self.golden), "sha256": PROFILE.sha(self.golden)},
            "source_files": {
                str(p): PROFILE.sha(p)
                for p in (
                    SCRIPTS / "profile_native_ar.py",
                    SCRIPTS / "benchmark-flash-bridge.py",
                    SCRIPTS / "native_json_process.py",
                    self.fake,
                )
            },
            "recording_options": {"path": str(self.options), "sha256": PROFILE.sha(self.options)},
        }

    def execute(self, behavior="success", mode="control", tool="success", timeout=2.0):
        self.plan["engine"]["arguments"] = [str(self.fake), behavior]
        trace_tool = self.root / "fake_xctrace"
        trace_tool.write_text(
            f"#!{sys.executable}\n" + TOOL.replace("MODE='success'", f"MODE={tool!r}")
        )
        trace_tool.chmod(0o755)
        self.plan["xctrace"] = {"path": str(trace_tool), "sha256": PROFILE.sha(trace_tool)}
        self.plan_path = self.root / f"plan-{behavior}-{mode}-{tool}.json"
        self.plan_path.write_text(json.dumps(self.plan))
        self.out = self.root / f"out-{behavior}-{mode}-{tool}"
        return PROFILE.run(self.plan_path, self.out, mode, timeout)

    def failed(self, **kwargs):
        with self.assertRaises((ValueError, RuntimeError, TimeoutError, OSError)):
            self.execute(**kwargs)
        report = json.loads((self.out / "report.json").read_text())
        self.assertFalse(report["parity"])
        self.assertTrue(report["all_children_joined"], report)
        return report

    def test_real_control_and_trace_keep_distinct_validation_states(self):
        control = self.execute()
        self.assertEqual(control["status"], "validated_control")
        self.assertTrue(control["parity"])
        self.assertEqual(len(control["evidence"]["complete"]), 2)
        trace = self.execute(mode="trace")
        self.assertEqual(trace["status"], "captured_pending_export")
        self.assertTrue(trace["numeric_parity"])
        self.assertFalse(trace["parity"])
        self.assertFalse(trace["coverage_validated"])
        command = trace["trace_command"]
        self.assertEqual(command[1], "record")
        self.assertEqual(command[command.index("--time-limit") + 1], "180s")
        self.assertNotIn("--window", command)
        self.assertIsNone(trace["recording_window_seconds"])
        self.assertIn("--no-prompt", command)
        self.assertEqual(command[command.index("--recording-options") + 1], str(self.options))
        self.assertEqual(
            {command[i + 1] for i, v in enumerate(command) if v == "--instrument"},
            set(PROFILE.INSTRUMENTS),
        )

    def test_missing_duplicate_invalid_nonfinite_truncated_wrong_backend_and_counts(self):
        for behavior in (
            "missing",
            "duplicate",
            "invalid",
            "nonfinite",
            "truncated",
            "backend",
            "counts",
            "diverge",
            "exit",
        ):
            with self.subTest(behavior=behavior):
                self.assertFalse(self.failed(behavior=behavior)["numeric_parity"])

    def test_loading_ready_preamble_must_match_real_production_order(self):
        for behavior in (
            "loading-missing",
            "loading-duplicate",
            "loading-late",
            "loading-wrong",
            "ready-duplicate",
            "ready-late",
        ):
            with self.subTest(behavior=behavior):
                self.assertFalse(self.failed(behavior=behavior)["numeric_parity"])

    def test_silence_is_bounded_and_reaped(self):
        start = time.monotonic()
        self.failed(behavior="silent", timeout=0.6)
        self.assertLess(time.monotonic() - start, 2.0)

    def test_128_budget_uses_actual_request_and_checks_complete_counts(self):
        self.plan["tokens"] = 128
        report = self.execute()
        self.assertEqual(report["generated_tokens"], 128)
        self.assertEqual(report["evidence"]["complete"][1]["stats"]["decode_tokens"], 127)
        self.failed(behavior="counts")

    def test_unsupported_token_budgets_refuse_before_launch(self):
        for value in (0, 31, 33, 127, 129, True, 128.0, "128", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PROFILE.token_budget(value)
        for value in (32, 128):
            self.assertEqual(PROFILE.requests(value)[0]["max_tokens"], value)

    def test_single_128_request_still_checks_a_nonempty_complete_generation(self):
        self.plan.update(tokens=128, warmup=False)
        report = self.execute()
        self.assertEqual(len(report["evidence"]["complete"]), 1)
        self.assertEqual(report["evidence"]["complete"][0]["stats"]["generated_tokens"], 128)
        self.assertEqual(PROFILE.requests(128, warmup=False)[0]["request_id"], PROFILE.IDS[1])
        self.assertEqual(len(PROFILE.requests(128, warmup=False)), 2)
        self.failed(behavior="missing")
        self.failed(behavior="counts")

    def test_warmup_requires_a_real_boolean(self):
        for value in (0, 1, "false", None, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PROFILE.requests(128, warmup=value)

    def test_cpu_template_is_selected_and_invalid_template_refuses(self):
        self.plan.update(template="Time Profiler", instruments=[])
        report = self.execute(mode="trace")
        command = report["trace_command"]
        self.assertEqual(command[command.index("--template") + 1], "Time Profiler")
        self.plan["template"] = "unrecognized"
        self.failed(mode="trace", tool="empty")

    def test_sigterm_main_writes_failure_and_reaps_silent_target(self):
        self.plan["engine"]["arguments"] = [str(self.fake), "silent"]
        plan = self.root / "signal-plan.json"
        plan.write_text(json.dumps(self.plan))
        out = self.root / "signal-out"
        controller = subprocess.Popen(
            [
                sys.executable,
                PROFILE.__file__,
                "--plan",
                str(plan),
                "--out",
                str(out),
                "--mode",
                "control",
                "--timeout",
                "5",
                "--run-after-root-grant",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                report_path = out / "report.json"
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    if "pid" in report:
                        break
                time.sleep(0.01)
            else:
                self.fail("controller did not launch disposable target")
            controller.send_signal(signal.SIGTERM)
            self.assertNotEqual(controller.wait(timeout=3), 0)
            report = json.loads(report_path.read_text())
            self.assertFalse(report["parity"])
            self.assertTrue(report["all_children_joined"], report)
            with self.assertRaises(ProcessLookupError):
                os.kill(report["pid"], 0)
        finally:
            if controller.poll() is None:
                controller.kill()
            controller.wait(timeout=3)

    def test_denial_and_empty_trace_do_not_validate(self):
        self.assertEqual(self.failed(mode="trace", tool="denied")["status"], "permission_denied")
        report = self.failed(mode="trace", tool="empty")
        self.assertTrue(report["numeric_parity"])
        self.assertEqual(report["capture_status"], "incomplete")

    def test_detached_exact_target_is_killed_and_unrelated_witness_survives(self):
        witness = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True
        )
        try:
            for tool in ("detach", "dangling"):
                with self.subTest(tool=tool):
                    self.failed(behavior="silent", mode="trace", tool=tool)
                    self.assertIsNone(witness.poll())
                    pid = int((self.out / "detached.pid").read_text())
                    row = PROFILE.process_rows().get(pid)
                    self.assertTrue(row is None or str(self.fake) not in row[2])
        finally:
            witness.terminate()
            witness.wait(timeout=2)
            target = self.out / "detached.pid"
            if target.exists():
                pid = int(target.read_text())
                row = PROFILE.process_rows().get(pid)
                if (
                    row is not None
                    and str(self.fake) in row[2]
                    and str(self.out / "registry.json") in row[2]
                ):
                    os.kill(pid, 9)

    def test_file_fingerprint_mutations_refuse_before_launch(self):
        self.plan["source_files"][str(self.fake)] = "0" * 64
        report = self.failed()
        self.assertNotIn("pid", report)

    def test_recording_options_missing_or_changed_refuse_before_launch(self):
        self.plan["recording_options"]["sha256"] = "0" * 64
        report = self.failed(mode="trace")
        self.assertNotIn("pid", report)
        self.plan["recording_options"]["sha256"] = PROFILE.sha(self.options)
        self.options.unlink()
        report = self.failed(mode="trace", tool="empty")
        self.assertNotIn("pid", report)

    def test_empty_instrument_list_keeps_only_metal_system_trace(self):
        self.plan["instruments"] = []
        report = self.execute(mode="trace")
        self.assertEqual(report["instruments"], [])
        command = report["trace_command"]
        self.assertNotIn("--instrument", command)
        self.assertEqual(command[command.index("--template") + 1], "Metal System Trace")
        self.assertEqual(report["status"], "captured_pending_export")
        self.assertFalse(report["parity"])

    def test_invalid_instrument_metadata_is_rejected_before_launch(self):
        for i, value in enumerate(
            (None, {}, "Thread Activity", [1], ["unknown"], ["Thread Activity", "Thread Activity"])
        ):
            with self.subTest(value=value):
                self.plan["instruments"] = value
                report = self.failed(mode="trace", tool=f"invalid-instruments-{i}")
                self.assertNotIn("pid", report)

    def test_recording_window_integer_boundaries_reach_trace_command_and_report(self):
        for seconds in (1, 10, 30):
            with self.subTest(seconds=seconds):
                self.plan["recording_window_seconds"] = seconds
                report = self.execute(mode="trace", tool=f"window-{seconds}")
                command = report["trace_command"]
                self.assertIn("--window", command)
                self.assertEqual(command[command.index("--window") + 1], f"{seconds}s")
                self.assertEqual(report["recording_window_seconds"], seconds)
                self.assertTrue(report["numeric_parity"])
                self.assertFalse(report["coverage_validated"])

    def test_invalid_recording_windows_refuse_before_launch(self):
        for i, seconds in enumerate((None, False, True, 0, -1, 1.0, 1.5, 31, "10", [], {})):
            with self.subTest(seconds=seconds):
                self.plan["recording_window_seconds"] = seconds
                report = self.failed(mode="trace", tool=f"invalid-window-{i}")
                self.assertNotIn("pid", report)

    def test_outer_timeout_accepts_300_and_rejects_outside_finite_domain(self):
        report = self.execute(timeout=300)
        self.assertTrue(report["parity"])
        for i, timeout in enumerate((0, -1, 300.01, float("inf"), -float("inf"), float("nan"))):
            with self.subTest(timeout=timeout):
                report = self.failed(timeout=timeout, tool=f"invalid-timeout-{i}")
                self.assertNotIn("pid", report)

    def test_cleanup_receives_fresh_budget_after_expired_run_deadline(self):
        with (
            patch.object(PROFILE.time, "monotonic", side_effect=[100.0, 200.0]),
            patch.object(PROFILE, "checked_files", side_effect=ValueError("slow preflight")),
            patch.object(PROFILE, "cleanup", return_value=True) as cleanup,
        ):
            self.failed(timeout=0.6)
        self.assertEqual(cleanup.call_args.args[2], 201.0)

    def test_cleanup_reaps_direct_child_when_process_discovery_fails(self):
        target = "/engine --registry /unique/registry.json bridge /model"
        state = {"alive": True, "waits": 0}

        def wait(**_):
            state["waits"] += 1
            state["alive"] = False
            return 0

        child = SimpleNamespace(pid=123, wait=wait, poll=lambda: None if state["alive"] else 0)
        with (
            patch.object(PROFILE, "process_rows", side_effect=OSError("ps unavailable")),
            patch.object(PROFILE.os, "killpg") as group,
        ):
            with self.assertRaises(OSError):
                PROFILE.cleanup(child, target, time.monotonic() + 1)
            self.assertGreaterEqual(state["waits"], 1)
            group.assert_called_once_with(123, signal.SIGTERM)

    def test_cleanup_snapshot_timeout_respects_remaining_budget(self):
        target = "/engine --registry /unique/registry.json bridge /model"
        child = SimpleNamespace(pid=123, wait=lambda **_: 0, poll=lambda: 0)
        with patch.object(PROFILE, "process_rows", return_value={}) as rows:
            self.assertTrue(PROFILE.cleanup(child, target, time.monotonic() + 0.05))
            self.assertGreaterEqual(rows.call_count, 3)
            for call in rows.call_args_list:
                self.assertGreater(call.kwargs["timeout"], 0)
                self.assertLessEqual(call.kwargs["timeout"], 0.05)

    def test_cleanup_group_permission_race_still_reaps_child(self):
        target = "/engine --registry /unique/registry.json bridge /model"
        state = {"alive": True, "waits": 0, "signals": []}

        def wait(**_):
            state["waits"] += 1
            state["alive"] = False
            return 0

        child = SimpleNamespace(
            pid=123,
            poll=lambda: None if state["alive"] else 0,
            wait=wait,
            send_signal=state["signals"].append,
        )
        with (
            patch.object(PROFILE, "process_rows", return_value={}),
            patch.object(PROFILE.os, "killpg", side_effect=PermissionError("exiting group")),
        ):
            self.assertTrue(PROFILE.cleanup(child, target, time.monotonic() + 1))
        self.assertEqual(state["signals"], [signal.SIGTERM])
        self.assertGreaterEqual(state["waits"], 1)

    def test_cleanup_permission_failure_waits_and_never_claims_success(self):
        target = "/engine --registry /unique/registry.json bridge /model"
        waited = []
        child = SimpleNamespace(
            pid=123,
            poll=lambda: None,
            wait=lambda **kw: waited.append(kw),
            send_signal=lambda _: (_ for _ in ()).throw(PermissionError("live child denied")),
        )
        with (
            patch.object(PROFILE, "process_rows", return_value={}),
            patch.object(PROFILE.os, "killpg", side_effect=PermissionError("denied")),
            self.assertRaises(PermissionError),
        ):
            PROFILE.cleanup(child, target, time.monotonic() + 1)
        self.assertEqual(len(waited), 1)

    def test_cleanup_revalidates_reused_pid_and_never_kills_another_group(self):
        target = "/engine --registry /unique/registry.json bridge /model --context-length 4096"
        original = {456: (1, 456, target)}
        reused = {456: (1, 456, "/unrelated witness")}
        child = SimpleNamespace(pid=123, wait=lambda **_: 0, poll=lambda: 0)
        with (
            patch.object(PROFILE, "process_rows", side_effect=[original, reused, reused, reused]),
            patch.object(PROFILE.os, "kill") as kill,
            patch.object(PROFILE.os, "killpg") as group,
        ):
            self.assertTrue(PROFILE.cleanup(child, target, time.monotonic() + 1))
            kill.assert_not_called()
            self.assertTrue(all(call.args[0] == 123 for call in group.call_args_list))

    def test_cleanup_never_signals_recycled_recorder_process_group(self):
        target = "/engine --registry /unique/registry.json bridge /model --context-length 4096"
        witness = {123: (1, 123, "/unrelated witness")}
        child = SimpleNamespace(pid=123, wait=lambda **_: 0, poll=lambda: 0)
        with (
            patch.object(PROFILE, "process_rows", return_value=witness),
            patch.object(PROFILE.os, "kill") as kill,
            patch.object(PROFILE.os, "killpg") as group,
        ):
            PROFILE.cleanup(child, target, time.monotonic() + 0.05)
            group.assert_not_called()
            kill.assert_not_called()

    def test_unrecognized_executable_with_unique_registry_is_not_claimed_joined(self):
        target = "/engine --registry /unique/registry.json bridge /model --context-length 4096"
        rows = {456: (1, 456, target.replace("/engine", "/different"))}
        child = SimpleNamespace(pid=123, wait=lambda **_: 0, poll=lambda: 0)
        with (
            patch.object(PROFILE, "process_rows", return_value=rows),
            patch.object(PROFILE.os, "kill") as kill,
            patch.object(PROFILE.os, "killpg"),
        ):
            self.assertFalse(PROFILE.cleanup(child, target, time.monotonic() + 0.05))
            kill.assert_not_called()

    def test_strict_json_and_environment_contract(self):
        for text in ('{"a":1,"a":2}', '{"x":NaN}', '{"x":1e999}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                PROFILE.strict(text)
        with patch.dict(
            os.environ, {"MLXL3_NATIVE_DEVICE_GREEDY": "1", "DYLD_INSERT_LIBRARIES": "bad"}
        ):
            env = PROFILE.environment()
        self.assertEqual(len(PROFILE.ACCEPTED), 13)
        self.assertTrue(all(env[name] == "1" for name in PROFILE.ACCEPTED))
        self.assertTrue(all(env[name] == "0" for name in PROFILE.EXPERIMENTS))
        self.assertNotIn("DYLD_INSERT_LIBRARIES", env)
        self.assertEqual([r["request_id"] for r in PROFILE.requests()[:-1]], list(PROFILE.IDS))


if __name__ == "__main__":
    unittest.main()
