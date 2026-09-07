"""A/B converter benchmark: exact outputs, wall time and active Metal peak.

Default: search only. --layers additionally converts three existing LFM layers,
using the completed run's calibration. Never writes or replaces model weights.
"""

import argparse
import gc
import hashlib
import inspect
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references/PonyExl3"))
from ponyexl3.convert import hessian, metal_search, timing

from mlxl3.quantize import quantize_tiles_mlx


def fingerprint(*arrays):
    return [hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in arrays]


def measure_pair(name, funcs, repeats):
    durations, peaks, hashes = [[], []], [[], []], [None, None]
    for run in range(repeats):
        for index in [0, 1] if run % 2 == 0 else [1, 0]:
            gc.collect()
            mx.synchronize()
            mx.clear_cache()
            mx.reset_peak_memory()
            start = time.perf_counter()
            arrays = funcs[index]()
            mx.eval(*(a for a in arrays if isinstance(a, mx.array)))
            durations[index].append(time.perf_counter() - start)
            peaks[index].append(mx.get_peak_memory())
            hashes[index] = fingerprint(*arrays)
            del arrays
        assert hashes[0] == hashes[1], f"Output bits changed: {name}"
        print(
            json.dumps(
                {
                    "case": name,
                    "run": run,
                    "baseline_s": durations[0][-1],
                    "optimized_s": durations[1][-1],
                    "exact": True,
                }
            ),
            flush=True,
        )
    before, after = map(statistics.median, durations)
    print(
        json.dumps(
            {
                "case": name,
                "baseline_s": before,
                "optimized_s": after,
                "speedup": before / after,
                "active_metal_peak_bytes": list(map(max, peaks)),
                "exact": True,
                "sha256": hashes[0],
            }
        ),
        flush=True,
    )


def search_bench(repeats):
    original = metal_search.quantize_tiles_mlx
    for k in range(2, 9):
        for n in (128, 672):
            x = mx.array(np.random.default_rng(42).normal(size=(n, 256)).astype(np.float32))
            # Warm both compiled paths outside the timed region.
            for func in (original, quantize_tiles_mlx):
                mx.eval(*func(x, k, 1))
            measure_pair(
                f"search_k{k}_tiles{n}",
                [partial(original, x, k, 1), partial(quantize_tiles_mlx, x, k, 1)],
                repeats,
            )


def layer_bench(repeats, bits=4):
    from safetensors import safe_open

    # Recover the reference allocator behaviour without mutating source files.
    optimized_ldlq = hessian._ldlq_inner_matrix_mlx
    source = inspect.getsource(optimized_ldlq).replace(
        "        if mx.get_cache_memory() > 512 * 1024 * 1024:\n            mx.clear_cache()",
        "        mx.clear_cache()",
    )
    namespace = vars(hessian).copy()
    exec(compile(source, "<reference LDLQ cache>", "exec"), namespace)  # noqa: S102 - local trusted source
    original_ldlq = namespace[optimized_ldlq.__name__]
    original_search = metal_search.quantize_tiles_mlx
    for key in (
        "model.layers.0.conv.out_proj",
        "model.layers.0.feed_forward.w1",
        "model.layers.0.feed_forward.w2",
    ):
        with safe_open(
            ROOT / "build/lfm26-quants/4bpw/calibration.safetensors", framework="numpy"
        ) as f:
            acts = f.get_tensor(key)

        def convert(optimized, key=key, acts=acts):
            metal_search.quantize_tiles_mlx = quantize_tiles_mlx if optimized else original_search
            hessian._ldlq_inner_matrix_mlx = optimized_ldlq if optimized else original_ldlq
            try:
                result = hessian.ldlq_quantize_layer(
                    ROOT / "models/source/LFM2.5-2.6B",
                    ROOT / "build/lfm26-quants/4bpw/source_quant_plan",
                    key,
                    scale_mode="computed",
                    calibration_activations=acts,
                    compare_oracle=False,
                    quant_bits=bits,
                )
                return (
                    result.layer.trellis,
                    result.layer.suh,
                    result.layer.svh,
                    result.converted_output,
                )
            finally:
                metal_search.quantize_tiles_mlx = original_search
                hessian._ldlq_inner_matrix_mlx = optimized_ldlq

        measure_pair(f"{key}_k{bits}", [partial(convert, False), partial(convert, True)], repeats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--bits", type=int, default=4, choices=range(2, 9), help="Bits for full-layer comparisons"
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    timing.disable()
    print(json.dumps({"device": mx.device_info(), "mlx_version": mx.__version__}), flush=True)
    if args.layers:
        layer_bench(args.repeats, args.bits)
    else:
        search_bench(args.repeats)
