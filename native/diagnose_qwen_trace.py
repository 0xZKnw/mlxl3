"""Report the first exact mismatch against a saved Python layer trace."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np


binary, model, reference = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
request = {"op": "mlx-qwen-trace", "path": str(model), "token": 1}
process = subprocess.run([str(binary), "codec"], input=json.dumps(request) + "\n",
    capture_output=True, text=True, timeout=600)
assert process.returncode == 0, process.stderr
response = json.loads(process.stdout)
assert "error" not in response, response
actual = response["data"]
expected = np.load(reference)
embedding = np.asarray(actual["layers"][0], dtype=np.uint16)
if not np.array_equal(embedding, expected["embedding"].ravel()):
    mismatch = np.flatnonzero(embedding != expected["embedding"].ravel())
    print(json.dumps({"status": "mismatch", "tensor": "embedding",
        "count": int(mismatch.size), "first": int(mismatch[0]),
        "actual": int(embedding[mismatch[0]]),
        "expected": int(expected["embedding"].ravel()[mismatch[0]])}))
    raise SystemExit(1)
for index, layer in enumerate(actual["layers"][1:-1]):
    key = f"layer_{index}"
    got = np.asarray(layer, dtype=np.uint16)
    want = expected[key].ravel()
    if not np.array_equal(got, want):
        mismatch = np.flatnonzero(got != want)
        print(json.dumps({"status": "mismatch", "tensor": key,
            "count": int(mismatch.size), "first": int(mismatch[0]),
            "actual": int(got[mismatch[0]]), "expected": int(want[mismatch[0]])}))
        raise SystemExit(1)
norm = np.asarray(actual["layers"][-1], dtype=np.uint16)
if not np.array_equal(norm, expected["norm"].ravel()):
    print(json.dumps({"status": "mismatch", "tensor": "norm"}))
    raise SystemExit(1)
logits = np.asarray(actual["logits"], dtype=np.uint16)
np.testing.assert_array_equal(logits, expected["logits"].ravel())
print(json.dumps({"status": "passed", "layers": len(actual["layers"]) - 2}))
