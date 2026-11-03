"""The physical parity driver must refuse empty, incomplete and malformed CLI generations."""

import unittest

from physical_flash_run import scored


class RunReport(unittest.TestCase):
    def test_only_a_completed_turn_returns_text_and_count(self):
        text = "\n— Answer —\n4"
        footer = "\n[prefill 1.2 tok/s · decode 0.0 tok/s · TTFT 824 ms · 1 tokens]\n"
        self.assertEqual(scored(text + footer), (text, 1))
        for bad in [
            text,
            footer,
            text + footer[:-1],
            text + footer + "crash",
            text + footer.replace("1.2", ".."),
            text + footer.replace("1 tokens", "0 tokens"),
        ]:
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                scored(bad)
