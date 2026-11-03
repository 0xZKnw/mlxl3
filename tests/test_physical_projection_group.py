"""Validate the ordinary-group numerical/report checker without a GPU."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import physical_projection_group as driver


class GroupOracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def invoke(self, mutation=None):
        spec = {
            "native": {"path": str(self.root), "prefixes": ["p0", "p1"], "x": [0] * 128},
            "stock": {},
            "widths": [128, 256],
        }
        ref = {"data": {"grouped": True, "outputs": [[0x3C00] * 128, [0x4000] * 256]}}
        result = copy.deepcopy(ref)
        if mutation:
            mutation(result)
        replies = [{"error": "expected"}] * 5 + [result, result]
        args = SimpleNamespace(qwen=None, lfm=None, previous=None, cli="native", stock="stock")
        report = {"status": "started"}
        with (
            patch.object(driver, "synthetic", return_value=[spec]),
            patch.object(driver, "run", side_effect=[([ref], 0), (replies, 0)]),
        ):
            driver.verify(args, self.root, report)
        return report

    def test_nonempty_unequal_widths_pass(self):
        report = self.invoke()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["compared"], 384)
        self.assertEqual(report["unequal"], 0)

    def test_missing_partial_nonfinite_divergent_and_wrong_group_mode_fail(self):
        def missing(r):
            del r["data"]

        def partial(r):
            r["data"]["outputs"][0].pop()

        def nonfinite(r):
            r["data"]["outputs"][0][0] = 0x7C00

        def unequal(r):
            r["data"]["outputs"][1][0] = 0

        def mode(r):
            r["data"]["grouped"] = False

        for change in [missing, partial, nonfinite, unequal, mode]:
            with (
                self.subTest(change=change.__name__),
                self.assertRaises((AssertionError, KeyError)),
            ):
                self.invoke(change)

    def test_failed_report_survives_interruption_and_refuses_overwrite(self):
        cli, stock, report = (self.root / name for name in ("native", "stock", "report.json"))
        cli.write_bytes(b"native fixture")
        stock.write_bytes(b"stock fixture")
        args = ["physical_projection_group", str(cli), str(stock), str(report)]
        with patch.object(sys, "argv", args):
            with (
                patch.object(driver, "verify", side_effect=KeyboardInterrupt),
                self.assertRaises(KeyboardInterrupt),
            ):
                driver.main()
            failed = report.read_bytes()
            self.assertEqual(json.loads(failed)["status"], "failed")
            with self.assertRaises(FileExistsError):
                driver.main()
            self.assertEqual(report.read_bytes(), failed)


if __name__ == "__main__":
    unittest.main()
