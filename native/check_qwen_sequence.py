"""Compare three stateful native Qwen steps with a saved Python oracle."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np


binary, model, reference = map(Path, sys.argv[1:4])
request = {"op": "mlx-qwen-model", "path": str(model), "tokens": [1, 2, 3]}
process = subprocess.run(
    [str(binary), "codec"], input=json.dumps(request) + "\n",
    capture_output=True, text=True, timeout=900,
)
assert process.returncode == 0, process.stderr
actual = np.asarray(json.loads(process.stdout)["data"], dtype=np.uint16)
expected = np.load(reference).reshape(actual.shape)
np.testing.assert_array_equal(actual, expected)
print(json.dumps({"status": "passed", "tokens": 3,
    "logits_per_token": int(actual.shape[-1]), "comparison": "exact"}))
