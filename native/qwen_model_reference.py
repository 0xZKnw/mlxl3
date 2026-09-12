"""Write one full-model Python/MLX reference output, then exit to release RAM."""
from pathlib import Path
import sys

import mlx.core as mx
import numpy as np

from mlxl3.checkpoint import load_exl3_model


model, _, _ = load_exl3_model(Path(sys.argv[1]), lazy=False)
logits = model(mx.array([[1]], dtype=mx.uint32))
mx.eval(logits)
np.save(sys.argv[2], np.asarray(logits).view(np.uint16))
print({"status": "written", "shape": logits.shape, "dtype": str(logits.dtype)})
