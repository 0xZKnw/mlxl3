"""Shape sweep for dense MUL1 QMV; timings are not full-model throughput."""
import json
import os
import statistics
import time

import mlx.core as mx
import mlxl3.kernels.qmv as qmv
from kernel_bench import Case, _arrays
from mlxl3.codec.codebook import CodebookMode


def main():
    os.environ['MLXL3_MUL1_SWAR'] = '0'
    header = qmv.specialized_codebook_header
    split = qmv._split_count
    tile = qmv._qmv_tiles_per_group
    groups = qmv._qmv_simdgroups
    old = '''    uint sum = 0x6400u;
    sum += bits & 0xffu;
    sum += (bits >> 8) & 0xffu;
    sum += (bits >> 16) & 0xffu;
    sum += (bits >> 24) & 0xffu;'''
    new = '''    uint pairs = (bits & 0x00ff00ffu) + ((bits >> 8) & 0x00ff00ffu);
    uint sum = 0x6400u + (pairs & 0xffffu) + (pairs >> 16);'''
    for ni, no, k in [(5120, 17408, 3), (17408, 5120, 3), (5120, 17408, 2), (5120, 248320, 6)]:
        x, t, su, sv = _arrays(Case(ni, no, k, CodebookMode.MUL1))
        reference = None
        for swar, nt, sg in [(False, 0, 0)] + [(s, n, g) for s in (False, True) for n in (2, 4, 8) for g in (4, 8)]:
            qmv.specialized_codebook_header = lambda mode: header(mode).replace(old, new) if swar else header(mode)
            qmv._qmv_tile_kernel.cache_clear()
            qmv._split_count = split
            qmv._qmv_tiles_per_group = (lambda *a: nt) if nt else tile
            qmv._qmv_simdgroups = (lambda *a: sg) if sg else groups
            operation = mx.compile(lambda x: qmv.qmv_exl3(x, t, su, sv, k, CodebookMode.MUL1))
            for _ in range(3):
                out = operation(x)
                mx.eval(out)
            if reference is None:
                reference = out
            delta = float(mx.max(mx.abs(out.astype(mx.float32)-reference.astype(mx.float32))).item())
            samples = []
            for _ in range(9):
                start = time.perf_counter()
                mx.eval(operation(x))
                samples.append((time.perf_counter()-start)*1000)
            print(json.dumps(dict(shape=[ni,no,k], swar=swar, nt=nt, sg=sg,
                                  ms=statistics.median(samples), max_abs=delta)), flush=True)
        del x, t, su, sv, operation, reference, out
        mx.clear_cache()


if __name__ == '__main__':
    main()
