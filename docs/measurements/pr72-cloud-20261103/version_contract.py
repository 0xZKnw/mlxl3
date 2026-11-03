"""Bounded source-level contract; calls the real helper without rewriting it."""

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/copy-test-runtime.py"
SPEC = importlib.util.spec_from_file_location("pr72_copy_runtime", SCRIPT)
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


def version_result(value: str) -> tuple[int, int, int]:
    """
    pre: len(value) <= 32
    post: len(__return__) == 3 and all(part >= 0 for part in __return__)
    raises: ValueError
    """
    return helper.version(value)
