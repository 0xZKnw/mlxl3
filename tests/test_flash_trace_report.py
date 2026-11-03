import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import physical_flash_trace
from physical_flash_trace import state_digest


class FlashTraceTest(unittest.TestCase):
    def test_state_words_reject_malformed_nonfinite_and_detect_raw_changes(self):
        data = struct.pack("<2f", -0.0, 1.5)
        assert state_digest(data, 2) == hashlib.sha256(data).hexdigest()
        assert state_digest(data, 2) != state_digest(struct.pack("<2f", 0.0, 1.5), 2)
        for raw, count in [(b"", 0), (data[:-1], 2), (data, 1), (b"", -1)]:
            with self.assertRaises(ValueError):
                state_digest(raw, count)
        for x in [float("nan"), float("-inf"), float("inf")]:
            with self.assertRaises(ValueError):
                state_digest(struct.pack("<f", x), 1)

    def test_divergence_persists_a_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {"text_config": {"layer_types": ["linear_attention", "full_attention"] * 24}}
                )
            )
            binary = root / "cli"
            binary.write_bytes(b"test")
            report = root / "report"
            args = [
                "test",
                "--cli",
                str(binary),
                "--stock",
                str(binary),
                "--model",
                str(root),
                "--report",
                str(report),
            ]
            with (
                patch.object(sys, "argv", args),
                patch.object(
                    physical_flash_trace,
                    "observe",
                    side_effect=[({"tensors": ["a", "b"]}, 3), ({"tensors": ["b", "a"]}, 2)],
                ),
                self.assertRaisesRegex(ValueError, "differ"),
            ):
                physical_flash_trace.main()
            result = json.loads(report.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertNotIn("batch", result)
