import hashlib
import struct
import unittest

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
