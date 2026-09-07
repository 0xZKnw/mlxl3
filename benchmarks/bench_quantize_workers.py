"""End-to-end measurement-stage A/B using existing bounded worker scheduler."""

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "references/PonyExl3"))
from ponyexl3.convert import hessian, measure, metal_search, timing

from mlxl3 import quantize


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument(
        "--previous-search", action="store_true", help="Disable the new paired/ordered transitions"
    )
    p.add_argument("--profile", action="store_true")
    p.add_argument(
        "--compare-original",
        action="store_true",
        help="First worker entry uses original search/cache; others use MLXL3",
    )
    p.add_argument(
        "--compare-previous",
        action="store_true",
        help="First worker entry uses the previous MLXL3 search, retaining its cache/batching fixes",
    )
    p.add_argument("--bits", type=int, default=4, choices=range(2, 9))
    p.add_argument(
        "--baseline-module",
        type=Path,
        help="Trusted local quantize.py snapshot for the first worker entry",
    )
    args = p.parse_args()
    if args.repeats < 1 or any(w < 1 for w in args.workers):
        p.error("repeats and workers must be positive")
    if (args.compare_original or args.compare_previous) and len(args.workers) < 2:
        p.error("comparisons need at least two worker entries")
    if args.compare_original and args.compare_previous:
        p.error("choose one baseline")
    if (args.compare_previous or args.previous_search) and args.bits != 4:
        p.error("the pre-LUT transition baseline is K=4 only; use --baseline-module for other K")
    baseline_search = None
    if args.baseline_module:
        if (
            args.compare_original
            or args.compare_previous
            or args.previous_search
            or len(args.workers) < 2
        ):
            p.error("--baseline-module requires two worker entries and no other baseline option")
        spec = importlib.util.spec_from_file_location(
            "quantize_baseline", args.baseline_module.resolve()
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        baseline_search = module.quantize_tiles_mlx
    keys = [
        "model.layers.0.conv.out_proj",
        "model.layers.0.feed_forward.w1",
        "model.layers.0.feed_forward.w2",
        "model.layers.0.conv.in_proj",
    ]
    with safe_open(
        ROOT / "build/lfm26-quants/4bpw/calibration.safetensors", framework="numpy"
    ) as f:
        acts = {key: f.get_tensor(key) for key in keys}
    original_search = metal_search.quantize_tiles_mlx
    optimized_ldlq = hessian._ldlq_inner_matrix_mlx
    original_ldlq = optimized_ldlq
    if args.compare_original:
        old = "        if mx.get_cache_memory() > 512 * 1024 * 1024:\n            mx.clear_cache()"
        source = inspect.getsource(optimized_ldlq)
        assert source.count(old) == 1, "Expected bounded-cache baseline"
        namespace = vars(hessian).copy()
        exec(  # noqa: S102 - reconstruct trusted local reference for A/B only
            compile(source.replace(old, "        mx.clear_cache()"), "<reference cache>", "exec"),
            namespace,
        )
        original_ldlq = namespace[optimized_ldlq.__name__]
    optimized_search = quantize.quantize_tiles_mlx
    grouped_kernel, ordered_kernel = quantize._paired_kernel, quantize._ordered_kernel
    fingerprints = {}
    profiles = {}
    original = measure.ldlq_quantize_layer

    def tracked(*a, **kw):
        if args.profile:
            timing.enable()
            timing.reset()
        result = original(*a, **kw)
        if args.profile:
            profiles[result.module_key] = timing.report()
        fingerprints[result.module_key] = [
            hashlib.sha256(np.asarray(t).tobytes()).hexdigest()
            for t in (
                result.layer.trellis,
                result.layer.suh,
                result.layer.svh,
                result.converted_output,
            )
        ]
        return result

    measure.ldlq_quantize_layer = tracked
    baseline = None
    for repeat in range(args.repeats):
        cases = list(enumerate(args.workers))
        for index, workers in cases if repeat % 2 == 0 else cases[::-1]:
            original_pipeline = args.compare_original and index == 0
            previous_search = args.previous_search or (args.compare_previous and index == 0)
            quantize._paired_kernel = (
                (lambda k, cb: (metal_search._kernel(k, cb), ()))
                if previous_search
                else grouped_kernel
            )
            quantize._ordered_kernel = (
                (lambda k, cb: (metal_search._kernel(k, cb), ()))
                if previous_search
                else ordered_kernel
            )
            metal_search.quantize_tiles_mlx = (
                original_search if original_pipeline else optimized_search
            )
            if baseline_search and index == 0:
                metal_search.quantize_tiles_mlx = baseline_search
            hessian._ldlq_inner_matrix_mlx = original_ldlq if original_pipeline else optimized_ldlq
            fingerprints.clear()
            profiles.clear()
            mx.clear_cache()
            mx.reset_peak_memory()
            start = time.perf_counter()
            measure.measure_ldlq_candidates(
                ROOT / "models/source/LFM2.5-2.6B",
                ROOT / "build/lfm26-quants/4bpw/source_quant_plan",
                keys,
                candidate_bits=[args.bits],
                scale_mode="computed",
                calibration_activations_by_module=acts,
                max_workers=workers,
            )
            elapsed = time.perf_counter() - start
            if baseline is None:
                baseline = dict(fingerprints)
            assert fingerprints == baseline
            print(
                json.dumps(
                    {
                        "repeat": repeat,
                        "workers": workers,
                        "original_pipeline": original_pipeline,
                        "previous_search": previous_search,
                        "baseline_module": str(args.baseline_module)
                        if baseline_search and index == 0
                        else None,
                        "bits": args.bits,
                        "seconds": elapsed,
                        "peak_metal": mx.get_peak_memory(),
                        "exact": True,
                        "fingerprints": fingerprints,
                        "profiles": profiles,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
