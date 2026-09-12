"""Compare the native full Qwen model with a saved Python/MLX oracle."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


binary, model, reference = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
request = {"op": "mlx-qwen-model", "path": str(model), "token": 1}
process = subprocess.run([str(binary), "codec"], input=json.dumps(request) + "\n",
    capture_output=True, text=True, timeout=600)
assert process.returncode == 0, process.stderr
response = json.loads(process.stdout)
assert "error" not in response, response
actual = np.asarray(response["data"][0], dtype=np.uint16)
expected = np.load(reference).ravel()
np.testing.assert_array_equal(actual, expected)
print(json.dumps({"status": "passed", "model": model.name, "token": 1,
    "logits": int(actual.size), "comparison": "exact float16 bit patterns",
    "performance": "not measured"}))
