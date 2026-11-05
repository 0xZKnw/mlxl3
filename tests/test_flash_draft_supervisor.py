"""Exercise the archived supervisors' actual cleanup with disposable CPU children."""

import ast
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head"
SOURCES = [
    BASE / name / "quality-supervisor.txt"
    for name in ("drafts", "drafts3-quality", "drafts3-3g-quality", "drafts4-quality")
]


def cleanup_source(path):
    tree = ast.parse(path.read_text())
    loop = next(
        n for n in ast.walk(tree) if isinstance(n, ast.For) and ast.unparse(n.target) == "prev"
    )
    initializer = next(
        n
        for n in loop.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "r" for t in n.targets)
    )
    cleanup = next(n.finalbody for n in loop.body if isinstance(n, ast.Try))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reap_child"]
    return (
        compile(ast.Module(body=[initializer], type_ignores=[]), str(path), "exec"),
        compile(ast.Module(body=functions + cleanup, type_ignores=[]), str(path), "exec"),
    )


class DraftSupervisorCleanupTest(unittest.TestCase):
    def run_cleanup(self, source, child, signal_group, folder, status="failed"):
        context = {
            "child": child,
            "p": folder,
            "w": folder,
            "name": "fixture",
            "cmd": ["fixture"],
            "limit": 1,
            "prev": {"env": {}, "expected_tests": 1},
            "snapshot": dict,
            "t": time.monotonic(),
            "time": time,
            "json": json,
            "signal": signal,
            "subprocess": subprocess,
            "os": types.SimpleNamespace(killpg=signal_group),
        }
        initializer, cleanup = cleanup_source(source)
        exec(initializer, context)  # noqa: S102 -- exercise reviewed production AST
        record = context["r"]
        self.assertIs(record["reaped"], False)
        record.update(status=status, error="original failure")
        context["records"] = [record]
        with contextlib.redirect_stdout(io.StringIO()):
            exec(cleanup, context)  # noqa: S102 -- exercise reviewed production AST
        return record

    def test_child_exit_between_poll_and_group_signal_is_reaped(self):
        for source in SOURCES:
            with self.subTest(source=source.parent.name), tempfile.TemporaryDirectory() as tmp:
                child = subprocess.Popen(
                    [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
                )

                def vanished(pid, sig, child=child):
                    os.killpg(pid, sig)
                    child.wait(timeout=2)
                    raise ProcessLookupError("child exited before group signal")

                try:
                    record = self.run_cleanup(source, child, vanished, Path(tmp))
                    self.assertIs(record["reaped"], True)
                    self.assertEqual(record["status"], "failed")
                    self.assertEqual(record["error"], "original failure")
                    self.assertIsNotNone(child.returncode)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=2)

    def test_cleanup_error_publishes_explicit_false_without_passing(self):
        for source in SOURCES:
            with self.subTest(source=source.parent.name), tempfile.TemporaryDirectory() as tmp:
                child = types.SimpleNamespace(pid=123, poll=lambda: None)

                def denied(pid, sig):
                    raise PermissionError("fixture cannot signal")

                with self.assertRaises(PermissionError):
                    self.run_cleanup(source, child, denied, Path(tmp), status="passed")
                report = json.loads((Path(tmp) / "quality.json").read_text())[0]
                self.assertIs(report["reaped"], False)
                self.assertEqual(report["status"], "failed")
                self.assertIn("PermissionError", report["cleanup_error"])

    def test_already_exited_child_and_no_child_remain_truthful(self):
        for source in SOURCES:
            with self.subTest(source=source.parent.name), tempfile.TemporaryDirectory() as tmp:
                for child in [None, types.SimpleNamespace(poll=lambda: 0, wait=lambda timeout: 0)]:
                    record = self.run_cleanup(
                        source,
                        child,
                        lambda *_: self.fail("unexpected signal"),
                        Path(tmp),
                        status="passed",
                    )
                    self.assertIs(record["reaped"], True)
                    self.assertEqual(record["status"], "passed")

    def test_unresponsive_child_has_two_bounded_waits_and_never_passes(self):
        class Silent:
            pid = 123

            def poll(self):
                return None

            def wait(self, timeout):
                self.waits.append(timeout)
                raise subprocess.TimeoutExpired("fixture", timeout)

        for source in SOURCES:
            with self.subTest(source=source.parent.name), tempfile.TemporaryDirectory() as tmp:
                child = Silent()
                child.waits = []
                with self.assertRaises(TimeoutError):
                    self.run_cleanup(source, child, lambda *_: None, Path(tmp), status="passed")
                self.assertEqual(child.waits, [5, 5])
                report = json.loads((Path(tmp) / "quality.json").read_text())[0]
                self.assertIs(report["reaped"], False)
                self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()
