import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.check_native_coverage import BASELINE, differences, python_calls

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_native_coverage.py"


class NativeCoverageTest(unittest.TestCase):
    def test_guard_rejects_growth_and_removal_until_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init", "-q", tmp], check=True)
            (root / "native/src").mkdir(parents=True)
            (root / "native/src/array.rs").write_text("pub fn eval(&self) {}")
            (root / "src").mkdir()
            source = root / "src/converter.py"
            source.write_text("from mlx.core import eval as sync\nsync(1)\n")
            baseline = root / BASELINE
            baseline.parent.mkdir(parents=True)
            cmd = [sys.executable, str(SCRIPT), "--root", tmp]
            baseline.write_bytes(subprocess.check_output([*cmd, "--inventory"]))
            self.assertEqual(
                json.loads(baseline.read_bytes())["src/converter.py"]["call mlx.core.eval"], 1
            )
            subprocess.run(cmd, check=True, capture_output=True)
            source.write_text(source.read_text() + "sync(2)\n")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("mlx.core.eval 1 -> 2", result.stderr)
            source.unlink()
            (root / "native/src/new.rs").write_text("fn run(a: Array) { a.eval(); }")
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertIn("new.rs: .eval 0 -> 1", result.stderr)
            self.assertIn("converter.py: call mlx.core.eval 1 -> 0", result.stderr)

    def test_all_registered_model_types_and_bridge_functions_are_accounted_for(self):
        root = SCRIPT.parents[1]
        registry = (root / "docs/native-coverage/README.md").read_text()
        main = (root / "native/src/main.rs").read_text().split("impl NativeChatModel {", 1)[1]
        dispatch = main.split("match checkpoint.model_type.as_str() {", 1)[1].split("other =>", 1)[
            0
        ]
        models = set(re.findall(r'"([a-z0-9_]+)"', dispatch))
        rows = set(re.findall(r"^\| \x60([a-z0-9_]+)\x60 \|", registry, flags=re.MULTILINE))
        self.assertEqual(models, rows)
        self.assertIn("qwen4_exp", models)
        rust = (root / "native/src/array.rs").read_text()
        cpp = (root / "native/ffi/mlx_bridge.cpp").read_text()
        declared = set(re.findall(r"\bfn (mlxl3_\w+)\s*\(", rust))
        defined = set(re.findall(r"^(?:[\w:*]+\s+)+(mlxl3_\w+)\s*\(", cpp, flags=re.MULTILINE))
        self.assertIn("mlxl3_flash_prepare128", declared)
        self.assertEqual(declared, defined)

    def test_aliases_are_canonical_and_distinct_modules_stay_distinct(self):
        self.assertEqual(
            python_calls(
                "import mlx.core as a\nfrom mlx_lm import load as b\na.fast.rms_norm(x)\nb()\n"
            ),
            {"call mlx.core.fast.rms_norm": 1, "call mlx_lm.load": 1},
        )
        self.assertFalse(python_calls("import math as mx\nmx.sqrt(4)"))
        with self.assertRaises(SyntaxError):
            python_calls("import (")

    def test_empty_file_and_repeated_call_boundaries(self):
        choices = [{}, {"a": {}}, {"a": {"b": 1}}, {"a": {"b": 2}}, {"c": {"b": 1}}]
        for actual in choices:
            for expected in choices:
                self.assertEqual(not differences(expected, actual), expected == actual)
