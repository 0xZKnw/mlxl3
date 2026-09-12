"""Write exact Python logits for a short imposed Qwen token sequence."""
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

from mlxl3.checkpoint import load_exl3_model


model, _, _ = load_exl3_model(Path(sys.argv[1]), lazy=False)
cache = model.make_cache()
outputs = []
for token in (1, 2, 3):
    logits = model(mx.array([[token]], dtype=mx.uint32), cache=cache)
    states = []
    for item in cache:
        values = item.cache if hasattr(item, "cache") else item.state
        states.extend(value for value in values if value is not None)
    mx.eval(logits, *states)
    outputs.append(np.asarray(logits).view(np.uint16))
np.save(sys.argv[2], np.stack(outputs))
print({"status": "written", "tokens": 3})
