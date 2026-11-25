"""Symbolic contracts for the candidate's Python control flow with a model allocator."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Optional path to a disposable mutated worker. Normally test the repository's
# package source, independent of the shell's working directory.
default = Path(__file__).resolve().parents[3] / "apps/MLXL3Studio/Resources/Image21/render.py"
path = Path(os.environ.get("IMAGE21_PROOF_WORKER", default))
spec = importlib.util.spec_from_file_location("decode_candidate", path)
worker = importlib.util.module_from_spec(spec)
core = SimpleNamespace()
with patch.dict(
    sys.modules,
    {
        "mlx": SimpleNamespace(core=core, nn=SimpleNamespace(Module=object)),
        "mlx.core": core,
        "mlx.utils": SimpleNamespace(tree_flatten=None),
    },
):
    spec.loader.exec_module(worker)


class ModelAllocator:
    def __init__(self, previous):
        self.limit = previous

    def set_cache_limit(self, limit):
        previous, self.limit = self.limit, limit
        return previous

    def isfinite(self, tensor):
        return True

    def all(self, condition):
        return SimpleNamespace(item=lambda: condition)


def check_cache_lifetime(previous: int, fail: bool) -> tuple[int, int, bool]:
    """
    post: __return__ == (previous, 536870912, fail)
    """
    allocator = ModelAllocator(previous)
    tensor = SimpleNamespace(size=1)
    observed = -1

    def decode(value):
        nonlocal observed
        observed = allocator.limit
        if fail:
            raise ValueError("test exception")
        return value

    saved = worker.mx
    worker.mx = allocator
    try:
        failed = False
        try:
            worker.capped_decode(decode, tensor)
        except ValueError:
            failed = True
        return allocator.limit, observed, failed
    finally:
        worker.mx = saved
