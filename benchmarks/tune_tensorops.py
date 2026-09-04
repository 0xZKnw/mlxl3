"""Offline actual-weight TensorOps sweep with independent held-out inputs.

Writes a device/version-bound opt-in profile. Different tile geometry remains
Q1 even if sampled outputs are bit-exact. Does not install/activate the profile.
"""
import argparse
import itertools
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlxl3.cli import _load_model
from mlxl3.kernels.qmv import _qmm_tensor_kernel, _scaled_hadamard_input
from mlxl3.linear import EXL3Linear
from mlxl3.tuning import shape_key, signature, validate_tile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rows', type=int, default=64)
    parser.add_argument('--limit', type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    model, *_ = _load_model(args.model)
    entries, evidence, seen = {}, [], set()
    for name, module in model.named_modules():
        if not isinstance(module, EXL3Linear) or module.output_dims > 16384:
            continue
        d, n, k, mode = module.input_dims, module.output_dims, module.bits, module.mode
        key = shape_key(args.rows, d, n, k, mode)
        if key in seen:
            continue
        seen.add(key)
        # Fixed input seed is independent of the heldout seed below.
        mx.random.seed(124)
        x = _scaled_hadamard_input(mx.random.normal((args.rows, d)).astype(mx.float16), module.suh)
        mx.random.seed(593)
        heldout = _scaled_hadamard_input(mx.random.normal((args.rows, d)).astype(mx.float16), module.suh)
        mx.eval(x, heldout)

        def run(tile, values=x, *, k=k, mode=mode, d=d, n=n, module=module):
            bm, bn, bk = tile
            padded = ((args.rows + bm - 1) // bm) * bm
            if padded != args.rows:
                values = mx.pad(values, [(0, padded - args.rows), (0, 0)])
            out = _qmm_tensor_kernel(k, int(mode), d, n, bm, bn, bk)(
                inputs=[values, module.trellis.reshape(-1).view(mx.uint32)],
                grid=((n // bn) * 32, padded // bm, 1), threadgroup=(32, 1, 1),
                output_shapes=[(padded, n)], output_dtypes=[mx.float16],
            )[0][:args.rows]
            mx.eval(out)
            return out

        baseline = (32, 32, 16)
        reference = np.asarray(run(baseline)).copy()
        reference_heldout = np.asarray(run(baseline, heldout)).copy()
        best, best_gain = baseline, 0.0
        for tile in itertools.product((8, 16, 32), (16, 32, 64), (16, 32, 64)):
            if not validate_tile(tile, d, n):
                continue
            result = np.asarray(run(tile))
            check = np.asarray(run(tile, heldout))
            exact = result.tobytes() == reference.tobytes() and check.tobytes() == reference_heldout.tobytes()
            paired = []
            for pair in range(7):
                times = {}
                for candidate in ((False, True) if pair % 2 == 0 else (True, False)):
                    start = time.perf_counter()
                    for _ in range(5):
                        run(tile if candidate else baseline)
                    times[str(candidate)] = (time.perf_counter() - start) / 5
                paired.append(times)
            gain = statistics.median([r['False'] / r['True'] - 1 for r in paired])
            evidence.append({'module': name, 'shape': key, 'tile': tile,
                             'sample_exact': exact, 'max_abs_error': float(np.max(np.abs(result.astype(float) - reference))),
                             'paired_seconds': paired, 'median_gain': gain})
            if exact and gain > max(.05, best_gain):
                best, best_gain = tile, gain
        entries[key] = {'tile': best, 'heldout_checked': True, 'sample_exact': True,
                        'contract': 'Q1: samples are not an all-input proof', 'gain': best_gain}
        print(json.dumps({'shape': key, **entries[key]}), flush=True)
        if len(seen) >= args.limit:
            break
    with args.output.open('x') as handle:
        json.dump({'schema': 1, 'signature': signature(), 'entries': entries,
                   'evidence': evidence, 'model': str(args.model)}, handle, indent=2)


if __name__ == '__main__':
    main()
