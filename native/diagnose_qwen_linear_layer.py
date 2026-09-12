"""Compare one native linear Qwen layer against a saved production trace."""
from pathlib import Path
import json
import subprocess
import sys

import numpy as np


binary, model, reference = map(Path, sys.argv[1:4])
layer = int(sys.argv[4])
trace = np.load(reference)
x = trace["embedding"] if layer == 0 else trace[f"layer_{layer - 1}"]
request = {
    "op": "mlx-qwen-layer", "path": str(model), "layer": layer,
    "cols": 2048, "key_heads": 16, "value_heads": 32, "value_dims": 128,
    "conv_length": 4, "top_k": 8, "x": x.view(np.uint16).ravel().tolist(),
}
process = subprocess.run(
    [str(binary), "codec"], input=json.dumps(request) + "\n",
    capture_output=True, text=True, timeout=180,
)
assert process.returncode == 0, process.stderr
data = json.loads(process.stdout)["data"]
actual = np.asarray(data["output"], dtype=np.uint16)
expected = trace[f"layer_{layer}"].view(np.uint16).ravel()
for name, values in zip(
    ("input_norm", "qkv", "z", "a", "b", "conv", "q", "k", "v",
     "beta", "softplus", "g", "gdn_out", "gdn_norm", "gated", "attention",
     "residual", "post_norm", "mlp", "output"),
    data.get("trace", []),
):
    got = np.asarray(values, dtype=np.uint16)
    want = trace[f"trace_{name}"].view(np.uint16).ravel()
    mismatch = np.flatnonzero(got != want)
    if mismatch.size:
        print(json.dumps({"status": "mismatch", "tensor": name,
            "layer": layer, "count": int(mismatch.size),
            "first": int(mismatch[0])}))
        raise SystemExit(1)
mismatch = np.flatnonzero(actual != expected)
print(json.dumps({"status": "passed" if mismatch.size == 0 else "mismatch",
    "layer": layer, "count": int(mismatch.size),
    "first": None if mismatch.size == 0 else int(mismatch[0])}))
raise SystemExit(bool(mismatch.size))
