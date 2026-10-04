"""CPU-only negative check: a benchmark must not time an unacknowledged mode."""
from pathlib import Path
import subprocess
import sys


def test_rejects_missing_runtime_mode():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts/smoke-dflash-bridge.py"),
         str(root / "tests/fake-desktop-engine.py"), "fixture", "draft", "--tokens", "1"],
        cwd=root, capture_output=True, text=True, timeout=15,
    )
    assert result.returncode != 0
    assert "engine mode mismatch" in result.stderr
