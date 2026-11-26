"""Symbolic contracts for the Studio worker's dimensions and modeled allocator."""

import argparse
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


def check_size(value: str) -> tuple[int, int]:
    """
    post: __return__ in ((0, 0), (512, 512), (768, 768), (1024, 1024), (2048, 2048))
    post: __return__ == (0, 0) or value in (str(__return__[0]), "%dx%d" % __return__)
    post: __return__ != (0, 0) or value not in ("512", "768", "1024", "2048", "512x512", "768x768", "1024x1024", "2048x2048")
    """
    try:
        return worker.image_dimensions(value)
    except argparse.ArgumentTypeError:
        return (0, 0)


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
