"""Deterministic policy/oracle and failure tests; no model, SSD payload or GPU."""

import importlib.util
import json
import math
import random
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_flash_cycle_cache.py"
SPEC = importlib.util.spec_from_file_location("cycle_replay", SCRIPT)
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class FullScanPool(REPLAY.Pool):
    """Independent full-slot candidate enumeration instead of cached per-layer heads."""

    def _head(self, layer, protected, blocked, cyclic):
        slots = []
        for slot, key in enumerate(self.keys):
            if key is not None and key[0] == layer and key not in protected and slot not in blocked:
                count = self.weights[layer][key[1]] if cyclic else self.counts[layer][key[1]]
                slots.append((count, self.ages[slot], slot))
        return min(slots)[2] if slots else -1


class CycleCacheTests(unittest.TestCase):
    def test_native_physical_slot_order_and_wrong_observation_are_not_vacuous(self):
        observation = {"ids": [5, 1, 3], "slots_hit": [[2, False], [0, False], [1, False]]}
        cache, counts = REPLAY.Pool(1, 8, 3), [0, 0, 0]
        self.assertEqual(REPLAY.observe(cache, 0, observation, counts, False, True), 0)
        self.assertEqual(counts, [0, 3, 3 * REPLAY.EXPERT_BYTES])
        bad = {"ids": [5, 1, 3], "slots_hit": [[1, True], [0, True], [2, True]]}
        with self.assertRaisesRegex(ValueError, "physical slot/hit"):
            REPLAY.observe(cache, 0, bad, counts, True, True)

    def test_cycle_distance_zero_probability_and_known_ema_steps(self):
        self.assertEqual(REPLAY.far_key(48, 0, 0, 1), 48)
        self.assertEqual(REPLAY.far_key(48, 0, 1, 1), 1)
        self.assertEqual(REPLAY.far_key(48, 47, 0, 1), 1)
        self.assertEqual(REPLAY.far_key(48, 0, 1, 0.5), 49)
        self.assertTrue(math.isinf(REPLAY.far_key(48, 0, 1, 0)))
        pool = REPLAY.Pool(2, 4, 3, cycle=True)
        pool.reserve(0, [0], False)
        self.assertEqual(pool.weights[0], [0] * 4)  # prefill must not observe EMA
        pool.reserve(0, [0], True)
        self.assertAlmostEqual(pool.weights[0][0] * pool.scale[0], 1 / 64)
        pool.reserve(0, [1], True)
        self.assertAlmostEqual(pool.weights[0][0] * pool.scale[0], 63 / 4096)
        self.assertAlmostEqual(pool.weights[0][1] * pool.scale[0], 1 / 64)
        self.assertEqual(pool.cycle_observations, 2)

    def test_cached_heads_match_full_scan_through_mode_pins_and_phase_changes(self):
        rng = random.Random(138)
        for cyclic in (False, True):
            optimized = REPLAY.Pool(3, 8, 8, cycle=cyclic)
            reference = FullScanPool(3, 8, 8, cycle=cyclic)
            for visit in range(360):
                layer = visit % 3
                ids = rng.sample(range(8), 2)
                decode = visit % 60 >= 15
                leased = {0} if visit % 13 == 0 else set()
                pending = {1} if visit % 17 == 0 else set()
                actual = optimized.reserve(layer, ids, decode, leased, pending)
                expected = reference.reserve(layer, ids, decode, leased, pending)
                self.assertEqual(actual, expected, (cyclic, visit))
                self.assertEqual(optimized.keys, reference.keys)
                self.assertEqual(optimized.generations, reference.generations)
                self.assertEqual(sum(map(len, optimized.resident)), len(optimized.where))
                self.assertLessEqual(len(optimized.where), 8)

    def test_current_cohort_pending_and_leased_slots_cannot_be_evicted(self):
        pool = REPLAY.Pool(2, 8, 4, cycle=True)
        pool.reserve(0, [0, 1], True)
        pool.reserve(1, [0, 1], True)
        original = pool.keys.copy()
        slots = pool.reserve(0, [0, 2], True, leased={1}, pending={2})
        self.assertEqual(slots[0], (0, True))
        self.assertEqual(slots[1], (3, False))
        self.assertEqual(pool.keys[1:3], original[1:3])
        self.assertEqual(pool.generations[3], 2)
        with self.assertRaisesRegex(ValueError, "protected"):
            pool.reserve(1, [3, 4], True, leased={0, 1, 2})

    def test_temporarily_leased_empty_slots_remain_reusable(self):
        pool = REPLAY.Pool(1, 4, 2)
        self.assertEqual(pool.reserve(0, [1], False, leased={0}), [(1, False)])
        self.assertEqual(pool.reserve(0, [2], False), [(0, False)])

    def test_invalid_geometry_ids_probabilities_and_duplicate_json_refuse(self):
        for dimensions in ((0, 8, 4), (2, False, 4), (2, 8, -1)):
            with self.assertRaises(ValueError):
                REPLAY.Pool(*dimensions)
        pool = REPLAY.Pool(2, 8, 4)
        for layer, ids in ((2, [0]), (0, []), (0, [True]), (0, [8]), (0, list(range(5)))):
            with self.assertRaises(ValueError):
                pool.reserve(layer, ids, True)
        for p in (float("nan"), float("inf"), -0.1, 1.01):
            with self.assertRaises(ValueError):
                REPLAY.far_key(2, 0, 1, p)
        with self.assertRaises(ValueError):
            json.loads('{"status":1,"status":2}', object_pairs_hook=REPLAY.unique_object)

    def test_replay_deadline_rejects_a_nonempty_cohort_before_scoring(self):
        case = {"phases": [{"prefill": [{"ids": [0], "slots_hit": [[0, False]]}]}]}
        with self.assertRaisesRegex(TimeoutError, "deadline"):
            REPLAY.replay(case, "physical_per_layer_LFU53", 0)

    def test_failed_input_preserves_explicit_failure_and_previous_output(self):
        with tempfile.TemporaryDirectory(prefix="ar33-cycle-") as temporary:
            root = Path(temporary)
            source, out = root / "capture.json", root / "report.json"
            source.write_text("{}")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                REPLAY.run(source, out)
            self.assertEqual(json.loads(out.read_text())["status"], "failed")
            previous = out.read_bytes()
            with self.assertRaises(FileExistsError):
                REPLAY.run(source, out)
            self.assertEqual(out.read_bytes(), previous)
            with self.assertRaises(ValueError):
                REPLAY.validate_capture({"status": "complete_ar_routes"})


if __name__ == "__main__":
    unittest.main()
