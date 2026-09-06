"""Exact N-major tile experiment, measured against alternating original kernels."""
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
    factory = mx.fast.metal_kernel
    def nmajor(**kw):
        old = '(tile_k * uint(TILES_N) + tile_n + output_tile)'
        assert old in kw['source']
        kw['source'] = kw['source'].replace(old, '((tile_n + output_tile) * uint(TILES_K) + tile_k)')
        kw['name'] += '_nmajor'
        return factory(**kw)
    tile = qmv._qmv_tiles_per_group
    for ni, no, k in [(5120,17408,3), (17408,5120,3), (5120,34816,2), (5120,248320,6)]:
        x, t, su, sv = _arrays(Case(ni,no,k,CodebookMode.MUL1))
        nt = mx.contiguous(t.transpose(1,0,2)).reshape(t.shape)
        mx.eval(nt)
        operations = {}
        reference = qmv.qmv_exl3(x,t,su,sv,k,CodebookMode.MUL1)
        mx.eval(reference)
        for layout in ('original','nmajor'):
            for tiles in (2,4,8):
                mx.fast.metal_kernel = nmajor if layout == 'nmajor' else factory
                qmv._qmv_tile_kernel.cache_clear()
                qmv._qmv_tiles_per_group = lambda *args, tiles=tiles: tiles
                data = nt if layout == 'nmajor' else t
                op = mx.compile(lambda x, data=data: qmv.qmv_exl3(x,data,su,sv,k,CodebookMode.MUL1))
                out = op(x)
                mx.eval(out)
                assert mx.array_equal(out,reference).item(), (layout,tiles)
                operations[layout,tiles] = op
        samples = {key: [] for key in operations}
        for pair in range(5):
            for key, op in (list(operations.items()) if pair % 2 else list(operations.items())[::-1]):
                for _ in range(3): mx.eval(op(x))
                start = time.perf_counter()
                for _ in range(10): mx.eval(op(x))
                samples[key].append((time.perf_counter()-start)*100)
        print(json.dumps({'shape':[ni,no,k], 'results':[{ 'layout':key[0], 'nt':key[1], 'ms':statistics.median(values)} for key,values in samples.items()]}),flush=True)
        mx.fast.metal_kernel = factory
        qmv._qmv_tiles_per_group = tile
        qmv._qmv_tile_kernel.cache_clear()
        del operations, t, nt, x, su, sv, op, data, reference, out
        mx.clear_cache()


if __name__ == '__main__': main()
