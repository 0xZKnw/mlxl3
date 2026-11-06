"""A missing explicit Metal library must fail before GPU initialization or fallback."""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_explicit_metallib_path_cannot_silently_use_build_machine_library(tmp_path, kind):
    engine = os.environ.get("MLXL3_TEST_MLX_ENGINE")
    if not engine:
        pytest.skip("requires the compiled MLX helper; no model or physical GPU required")
    path = tmp_path / "mlx.metallib"
    if kind == "directory":
        path.mkdir()
    result = subprocess.run(
        [str(Path(engine).resolve()), "codec"],
        input=json.dumps({"op": "mlx-router", "data": [0], "cols": 1, "top_k": 1}) + "\n",
        capture_output=True,
        text=True,
        timeout=10,
        env=dict(os.environ, MLXL3_METALLIB_PATH=str(path)),
        check=False,
    )
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert len(rows) == 1
    assert "data" not in rows[0]
    assert "MLX metallib missing: " + str(path) in rows[0]["error"]
