"""Exercise the routed expert oracle's reports and adversarial replies on CPU."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import physical_mapped_expert as driver
from safetensors.numpy import load_file


class MappedOracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_synthetic_matrix_contains_all_supported_bits_codebooks_and_duplicate_routes(self):
        cases = driver.synthetic(self.root)
        self.assertEqual(len(cases), 22)
        self.assertEqual(len({p["path"] for p in cases}), 21)
        self.assertTrue(any(len(set(p["selected"])) < len(p["selected"]) for p in cases))
        tensors = load_file(Path(cases[0]["path"]) / "model.safetensors")
        self.assertEqual(len(tensors), 36)
        self.assertEqual(tensors["experts.0.gate_proj.suh"].dtype, np.float16)
        self.assertEqual(tensors["experts.0.gate_proj.trellis"].dtype, np.uint16)
        for p in cases:
            self.assertEqual(len(p["selected"]), len(p["beta"]))
            self.assertGreater(len(p["x"]), 0)

    def invoke(self, values):
        packet = {
            "path": str(self.root),
            "prefix": "experts",
            "key_heads": 1,
            "selected": [0],
            "beta": [0x3C00],
            "x": [0] * 4,
        }
        refs = [{"data": [0x3C00] * 4}]
        replies = [{"error": "expected rejection"}] * 6 + [{"data": values}] * 2
        args = SimpleNamespace(qwen=None, lfm=None, cli="native", stock="stock")
        report = {"status": "started"}
        with (
            patch.object(driver, "synthetic", return_value=[packet]),
            patch.object(driver, "run", side_effect=[(refs, 0), (copy.deepcopy(replies), 0)]),
        ):
            driver.verify(args, self.root, report)
        return report

    def test_complete_equal_responses_pass(self):
        report = self.invoke([0x3C00] * 4)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["compared"], 4)
        self.assertEqual(report["unequal"], 0)

    def test_empty_partial_invalid_nonfinite_and_divergent_values_fail(self):
        for values in ([], [0x3C00], [-1] * 4, [0x7C00] * 4, [0] * 4):
            with self.subTest(values=values), self.assertRaises(AssertionError):
                self.invoke(values)

    def test_interruption_is_failed_and_existing_reports_are_immutable(self):
        native, stock, report = (self.root / name for name in ("native", "stock", "report.json"))
        native.write_bytes(b"native fixture")
        stock.write_bytes(b"stock fixture")
        argv = ["physical_mapped_expert", str(native), str(stock), str(report)]
        with patch.object(sys, "argv", argv):
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
