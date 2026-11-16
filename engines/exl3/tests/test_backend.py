import json
import tempfile
import unittest
from pathlib import Path

from mlxl3_exl3.backend import (DEFAULT_CONTEXT, MAX_CONTEXT, PAGE_SIZE, BackendError,
                                FakeBackend, is_exl3_checkpoint, new_token_budget, resolve_context)
from mlxl3_exl3.protocol import parse_request


class ContextTests(unittest.TestCase):
    def test_automatic_context(self):
        self.assertEqual(resolve_context(0, 131072), DEFAULT_CONTEXT)
        self.assertEqual(resolve_context(0, 4096), 4096)
        self.assertEqual(resolve_context(0, 1000), 1024)  # whole pages
        self.assertEqual(resolve_context(0, 10), PAGE_SIZE)

    def test_explicit_context_rounds_up_to_pages(self):
        self.assertEqual(resolve_context(1, 8192), PAGE_SIZE)
        self.assertEqual(resolve_context(257, 8192), 512)
        self.assertEqual(resolve_context(8192, 8192), 8192)
        self.assertEqual(resolve_context(MAX_CONTEXT, 10 * MAX_CONTEXT), MAX_CONTEXT)

    def test_exhaustive_small_domain(self):
        # Every requested size up to 4 pages against several limits: result is a
        # whole number of pages, covers the request, and exceeds it by < 1 page.
        for limit in (PAGE_SIZE, 3 * PAGE_SIZE + 7, 4 * PAGE_SIZE):
            for requested in range(1, limit + 1):
                value = resolve_context(requested, limit)
                self.assertEqual(value % PAGE_SIZE, 0)
                self.assertGreaterEqual(value, requested)
                self.assertLess(value - requested, PAGE_SIZE)

    def test_invalid_context(self):
        with self.assertRaises(BackendError):
            resolve_context(-1, 4096)
        with self.assertRaises(BackendError):
            resolve_context(4097, 4096)
        with self.assertRaises(BackendError):
            resolve_context(MAX_CONTEXT + 1, 10 * MAX_CONTEXT)

    def test_token_budget(self):
        self.assertEqual(new_token_budget(-1, 100, 1024), 924)
        self.assertEqual(new_token_budget(10, 100, 1024), 10)
        self.assertEqual(new_token_budget(5000, 100, 1024), 924)
        for prompt in (1024, 2000):
            with self.assertRaises(BackendError):
                new_token_budget(-1, prompt, 1024)


class CheckpointTests(unittest.TestCase):
    def check(self, payload):
        with tempfile.TemporaryDirectory() as folder:
            if payload is not None:
                Path(folder, "config.json").write_text(payload, encoding="utf-8")
            return is_exl3_checkpoint(folder)

    def test_detection(self):
        self.assertTrue(self.check(json.dumps({"quantization_config": {"quant_method": "EXL3"}})))
        self.assertFalse(self.check(json.dumps({"quantization_config": {"quant_method": "gptq"}})))
        self.assertFalse(self.check(json.dumps({"quantization_config": "exl3"})))
        self.assertFalse(self.check("{"))
        self.assertFalse(self.check(None))


class FakeBackendTests(unittest.TestCase):
    def test_budget_truncates(self):
        backend = FakeBackend("unused", 0)
        request = parse_request(json.dumps({"type": "generate", "request_id": "r", "max_tokens": 2,
                                            "messages": [{"role": "user", "content": "a b c"}]}))
        items = list(backend.generate(request, lambda: False))
        self.assertEqual([i["text"] for i in items if "text" in i], ["a", " b"])
        self.assertEqual(items[-1]["stats"]["stop_reason"], "max_new_tokens")


if __name__ == "__main__":
    unittest.main()
