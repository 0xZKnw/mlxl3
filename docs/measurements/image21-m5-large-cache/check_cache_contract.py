"""CrossHair harness for the worker's real main and GPU check, through entry to build.

Only parsed, supported CLI options are supplied. OS and MLX report a usable GPU.
No inference, allocator, OS query, or model filesystem access runs during exploration.
Set IMAGE21_PROOF_WORKER to the baseline render.py for a negative control.
"""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def load_worker():
    default = Path(__file__).resolve().parents[3] / "apps/MLXL3Studio/Resources/Image21/render.py"
    path = Path(os.environ.get("IMAGE21_PROOF_WORKER", default))
    spec = importlib.util.spec_from_file_location("image21_worker_under_check", path)
    module = importlib.util.module_from_spec(spec)
    core = SimpleNamespace()
    modules = {
        "mlx": SimpleNamespace(core=core, nn=SimpleNamespace(Module=object)),
        "mlx.core": core,
        "mlx.utils": SimpleNamespace(tree_flatten=None),
    }
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


WORKER = load_worker()
# The callback is imported, but the worker will stop before constructing it.
sys.modules["mflux.callbacks.instances.memory_saver"] = SimpleNamespace(MemorySaver=object)
WORKER.platform = SimpleNamespace(mac_ver=lambda: ("26.3", (), ""))
WORKER.os = SimpleNamespace(environ={})
WORKER.print = lambda *args, **kwargs: None


class EnteredBuild(Exception):
    pass


def stop_before_loading(_folder):
    raise EnteredBuild()


WORKER.build = stop_before_loading


def cache_before_build(
    size: int, memory_bytes: int, base: bool, memory_known: bool, low_memory: bool
) -> bool:
    """pre: size in (512, 768, 1024)
    post: __return__
    """
    limits = []
    device = {"device_name": "Apple M5" if base else "Apple M5 Pro"}
    if memory_known:
        device["memory_size"] = memory_bytes
    WORKER.mx = SimpleNamespace(
        device_info=lambda: device,
        metal=SimpleNamespace(is_available=lambda: True),
        set_cache_limit=limits.append,
    )
    parsed = SimpleNamespace(
        size=size, low_memory=low_memory, seed=371, prompt="oak", model="fixture"
    )
    parser = SimpleNamespace(add_argument=lambda *args, **kwargs: None, parse_args=lambda: parsed)
    WORKER.argparse = SimpleNamespace(ArgumentParser=lambda: parser)
    try:
        WORKER.main()
    except EnteredBuild:
        # Acceptance table includes the historical all-M5 rule at the smallest size.
        allowed_sizes = (512,)
        if base and memory_known and memory_bytes == 25769803776:
            allowed_sizes = (512, 768, 1024) if low_memory else (512, 768)
        expected = [536870912] if size in allowed_sizes else []
        return limits == expected
    return False  # Without reaching build, an empty call list cannot count as a pass.
