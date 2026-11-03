"""Failure handling for the physical Qwen oracle, without Metal or model weights."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import physical_qwen_gdn as driver


class QwenOracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        config = {
            "hidden_size": 128,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 4,
            "linear_conv_kernel_dim": 2,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        }
        (self.root / "config.json").write_text(json.dumps({"text_config": config}))
        self.refs = [
            {
                "data": {
                    "output": [0x3C00] * (n * 128),
                    "conv_state": [0x3C00] * 768,
                    "state": [0.25] * (4 * 128 * 128),
                }
            }
            for _layer in (0, 36)
            for n in (1, 2, 4)
        ]

    def invoke(self, mutation=None):
        native = copy.deepcopy(self.refs)
        if mutation:
            mutation(native)
        replies = [{"error": "expected rejection"}] * 3 + native + [copy.deepcopy(native[-1])]
        with patch.object(driver, "run", side_effect=[(self.refs, []), (replies, [])]):
            report = {"status": "started"}
            driver.verify(Path("native"), Path("stock"), self.root, report)
        return report

    def test_nonempty_finite_equal_cases_pass(self):
        report = self.invoke()
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["mismatches"], 0)
        self.assertEqual(len(report["cases"]), 6)
        self.assertGreater(report["compared"], 0)

    def test_malformed_incomplete_nonfinite_and_divergent_responses_fail(self):
        def missing_data(replies):
            replies[0] = {"error": "unexpected oracle failure"}

        def missing_row(replies):
            replies[0]["data"]["output"].pop()

        def nonfinite(replies):
            replies[0]["data"]["state"][0] = float("nan")

        def divergence(replies):
            replies[0]["data"]["state"][0] = 0.5

        def invalid_half(replies):
            replies[0]["data"]["output"][0] = -1

        for mutation in (missing_data, missing_row, nonfinite, divergence, invalid_half):
            with self.subTest(mutation=mutation.__name__), self.assertRaises(AssertionError):
                self.invoke(mutation)

    def test_report_records_interruption_and_refuses_overwrite(self):
        native, stock, report = (self.root / name for name in ("native", "stock", "report.json"))
        native.write_bytes(b"native fixture")
        stock.write_bytes(b"stock fixture")
        arguments = ["physical_qwen_gdn", str(native), str(stock), str(self.root), str(report)]
        with patch.object(sys, "argv", arguments):
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
