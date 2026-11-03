import json
import tempfile
import unittest
from pathlib import Path

import build_mutants as runner


def cargo_output(*, success=True, paths=("/tmp/disposable-native-tests",)):
    rows = [
        {
            "reason": "compiler-artifact",
            "target": {"name": "mlxl3_native", "kind": ["lib"]},
            "profile": {"test": True},
            "executable": p,
        }
        for p in paths
    ]
    rows.append({"reason": "build-finished", "success": success})
    return "\n".join(json.dumps(r) for r in rows)


class CargoAndRestoreTests(unittest.TestCase):
    def test_one_actual_lib_test_artifact(self):
        self.assertEqual(runner.artifact(cargo_output()), Path("/tmp/disposable-native-tests"))

    def test_repeat_artifact_message_for_same_file_is_not_ambiguous(self):
        self.assertEqual(runner.artifact(cargo_output(paths=("/tmp/a", "/tmp/a"))), Path("/tmp/a"))

    def test_failed_incomplete_empty_malformed_ambiguous_outputs(self):
        invalid = [
            "",
            "{}",
            "not JSON",
            cargo_output(success=False),
            cargo_output(paths=()),
            cargo_output(paths=("/tmp/a", "/tmp/b")),
            cargo_output() + "\n" + json.dumps({"reason": "build-finished", "success": True}),
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runner.artifact(raw)

    def test_wrong_target_or_non_test_binary_is_rejected(self):
        for field in ("target", "profile"):
            rows = [json.loads(s) for s in cargo_output().splitlines()]
            rows[0][field] = (
                {"name": "other", "kind": ["bin"]} if field == "target" else {"test": False}
            )
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.artifact("\n".join(json.dumps(r) for r in rows))

    def test_restore_is_byte_exact_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "source.rs"
            p.write_bytes(b"mutant\n")
            h = runner.digest(p)
            runner.restore(p, b"normal\r\n", h)
            self.assertEqual(p.read_bytes(), b"normal\r\n")
            runner.restore(p, b"normal\r\n", h)
            self.assertEqual(p.read_bytes(), b"normal\r\n")

    def test_unexpected_external_edit_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "source.rs"
            p.write_bytes(b"mutant")
            h = runner.digest(p)
            p.write_bytes(b"external edit")
            with self.assertRaises(RuntimeError):
                runner.restore(p, b"normal", h)
            self.assertEqual(p.read_bytes(), b"external edit")


if __name__ == "__main__":
    unittest.main()
