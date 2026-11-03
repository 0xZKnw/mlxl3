"""Check resident NT2 evidence validation and bounded process failure paths."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import physical_flash_nt2 as driver


class Nt2EvidenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_numerical_checker_detects_empty_partial_nonfinite_and_divergent(self):
        self.assertEqual(driver.words([0x3C00, 0x4000], [0x3C00, 0x4000]), 2)
        for a, b in [
            ([], []),
            ([0x3C00], [0x3C00, 0]),
            ([0x7C00], [0x7C00]),
            ([0], [0]),
            ([0x4000], [0x3C00]),
            ([1.0], [1]),
            ([65536], [1]),
        ]:
            with self.subTest(a=a, b=b), self.assertRaises(AssertionError):
                driver.words(a, b)

    def invoke(self, mutation=None):
        cases = [{"generic": False, "candidate_nt": 2, "control_nt": 1, "words": 2}]
        ref = [[0x3C00, 0x4000]]
        args = SimpleNamespace(model=self.root, native=Path("native"), stock=Path("stock"))

        def child(command, env, log):
            if command[0] == "stock":
                (self.root / "stock.json").write_text(json.dumps(ref))
                return ""
            if env["MLXL3_NATIVE_FLASH_NT2"] == "true":
                (self.root / "invalid-flag.json").write_text(
                    json.dumps({"rejected": True, "fc_hidden_nt": 1, "generic_unchanged": True})
                )
                return ""
            enabled = int(env["MLXL3_NATIVE_FLASH_NT2"])
            record = {"case": 0, "nt": 2 if enabled else 1, "nt2_submissions": enabled, "words": 2}
            result = {"parity": True, "unequal": 0, "cases": [record], "nt2_submissions": enabled}
            if mutation and enabled:
                mutation(result)
            (self.root / f"native-{enabled}.json").write_text(json.dumps(result))
            np.array(ref[0], dtype="<u2").tofile(self.root / f"native-{enabled}-0.bin")
            return "dyld[1]: /System/Library/Frameworks/Metal.framework/Metal"

        report = {"status": "started", "parity": False}
        with (
            patch.object(driver, "fixtures", return_value=cases),
            patch.object(driver, "process", side_effect=child),
        ):
            driver.verify(args, self.root, report)
        return report

    def test_actual_path_and_complete_corpus_required(self):
        self.assertEqual(self.invoke()["compared"], 4)
        mutations = [
            lambda r: r.update(parity=False),
            lambda r: r.update(cases=[]),
            lambda r: r["cases"][0].update(nt=1),
            lambda r: r["cases"][0].update(nt2_submissions=0),
            lambda r: r.update(nt2_submissions=0),
            lambda r: r["cases"][0].update(words=1),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(AssertionError):
                self.invoke(mutation)

    def test_process_timeout_and_nonzero_exit_are_retained(self):
        log = self.root / "child.log"
        with (
            patch.object(
                driver.subprocess,
                "run",
                side_effect=subprocess.TimeoutExpired("child", 240, "partial", "error"),
            ),
            self.assertRaises(subprocess.TimeoutExpired),
        ):
            driver.process(["child"], {}, log)
        self.assertIn("partial", log.read_text())
        failed = SimpleNamespace(returncode=1, stdout="1 passed; 0 failed", stderr="bad exit")
        with (
            patch.object(driver.subprocess, "run", return_value=failed),
            self.assertRaises(AssertionError),
        ):
            driver.process(["child"], {}, log)
        self.assertIn("bad exit", log.read_text())

    def test_report_failure_never_overwrites_or_retains_parity(self):
        native, stock, report = (self.root / s for s in ("native", "stock", "report.json"))
        native.write_bytes(b"native")
        stock.write_bytes(b"stock")
        argv = ["nt2", str(native), str(stock), str(self.root), str(report)]

        def interrupted(args, root, state):
            state.update(parity=True)
            raise KeyboardInterrupt

        with (
            patch.object(sys, "argv", argv),
            patch.object(driver, "verify", side_effect=interrupted),
        ):
            with self.assertRaises(KeyboardInterrupt):
                driver.main()
            saved = report.read_bytes()
            self.assertEqual(json.loads(saved)["status"], "failed")
            self.assertFalse(json.loads(saved)["parity"])
            with self.assertRaises(FileExistsError):
                driver.main()
            self.assertEqual(saved, report.read_bytes())


if __name__ == "__main__":
    unittest.main()
