"""Symbolic entry points exercise the actual profile guards without GPU or writes."""

import importlib.util
from pathlib import Path

PATH = (
    Path(__file__).parents[1]
    / "docs/measurements/quant-lfm25-research-20261013/profile-baseline.py"
)
spec = importlib.util.spec_from_file_location("lfm25_profile", PATH)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


def rejects_empty_profile(rows: int) -> bool:
    """
    pre: 1 <= rows <= 2048
    post: _
    """
    try:
        profile.fingerprints([], {"m": (rows, rows)})
    except ValueError:
        return True
    return False


def rejects_nonpositive_repeats(repeats: int) -> bool:
    """
    pre: repeats <= 0
    post: _
    """
    try:
        profile.run("unused-report.json", repeats)
    except ValueError:
        return True
    return False


def rejects_false_complete_report(repeats: int) -> bool:
    """
    pre: repeats >= 1
    post: _
    """
    try:
        profile.validate_report({"status": "complete", "parity": True, "runs": [{}] * 8}, repeats)
    except ValueError:
        return True
    return False
