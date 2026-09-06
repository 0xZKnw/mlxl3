"""Isolated Qwen dense experiments; no runtime defaults changed by this script."""
import argparse
import json
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxl3.codec.codebook import decode_codewords, CodebookMode
from mlxl3.cli import _load_model, _stream_response, GenerationSession


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--candidate', choices=['base', 'optimized', 'tuned', 'tensor', 'packed', 'halfdot', 'lut', 'swar', 'swar16', 'shuffle', 'unroll2', 'unroll4', 'unroll8', 'k3', 'nt1', 'sg2', 'sg4', 'fast'], default='base')
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--tile', nargs=3, type=int, default=[32,32,16])
    args = parser.parse_args()
    os.environ['MLXL3_MUL1_SWAR'] = '1' if args.candidate in ('optimized', 'tuned', 'tensor') else '0'
    os.environ['MLXL3_DENSE_PREFILL_M64'] = '1' if args.candidate == 'tuned' else '0'
    original = mx.fast.metal_kernel
    lut = mx.array(decode_codewords(np.arange(65536, dtype=np.uint16), CodebookMode.MUL1))
    mx.eval(lut)

    def factory(**kwargs):
        header = kwargs.get('header', '')
        if '0x83DCD12D' not in header or 'special' in kwargs['name']:
            return original(**kwargs)
        if args.candidate == 'lut':
            kwargs['header'] += '\n#define mlxl3_decode_codeword(x, cb) float(codebook[(x) & 0xffffu])\n'
            kwargs['input_names'] = kwargs['input_names'] + ['codebook']
        elif args.candidate in ('swar', 'swar16', 'shuffle', 'packed') or args.candidate.startswith('unroll'):
            start = header.index('    uint sum = 0x6400u;')
            end = header.index('    half value', start)
            kwargs['header'] = header[:start] + '''    uint pairs = (bits & 0x00ff00ffu) + ((bits >> 8) & 0x00ff00ffu);
    uint sum = 0x6400u + (pairs & 0xffffu) + (pairs >> 16);
''' + header[end:]
            if args.candidate == 'swar16':
                kwargs['header'] = kwargs['header'].replace('uint sum = 0x6400u + (pairs & 0xffffu) + (pairs >> 16);', 'ushort sum = ushort(0x6400u) + ushort(pairs) + ushort(pairs >> 16);')
            if args.candidate == 'packed' and 'qmv_' in kwargs['name']:
                kwargs['header'] += '''
inline float4 mlxl3_decode_four(uint4 x) {
    uint4 bits = (x & 0xffffu) * 0x83DCD12Du;
    uint4 pairs = (bits & 0x00ff00ffu) + ((bits >> 8) & 0x00ff00ffu);
    ushort4 sum = ushort4(0x6400u + (pairs & 0xffffu) + (pairs >> 16));
    half4 values = as_type<half4>(sum);
    return float4(values * half4(as_type<half>(ushort(0x1EEEu))) + half4(as_type<half>(ushort(0xC931u))));
}
'''
                source = kwargs['source'].replace('uint j = group * 4u;', 'uint j = group * 4u;\n float4 decoded = mlxl3_decode_four(uint4(cw0,cw1,cw2,cw3));')
                for i, component in enumerate('xyzw'):
                    source = source.replace(f'float(mlxl3_decode_codeword(cw{i}, CB))', 'decoded.' + component)
                kwargs['source'] = source
            if args.candidate == 'shuffle' and 'qmv_' in kwargs['name'] and re.search(r'_k[234]_', kwargs['name']):
                source = kwargs['source']
                source = source.replace('* uint(PACKED_U32);', '* uint(PACKED_U32);\n uint word_lane = words[lane % uint(PACKED_U32)];')
                for part in (0, 1):
                    source = source.replace(f'words[word{part}[group]]', f'simd_shuffle(word_lane, ushort(word{part}[group]))')
                kwargs['source'] = source
            if args.candidate.startswith('unroll') and 'qmv_' in kwargs['name']:
                kwargs['source'] = kwargs['source'].replace('            for (\n                uint tile_k', '#pragma unroll ' + args.candidate[-1] + '\n            for (\n                uint tile_k')
        elif args.candidate == 'fast':
            kwargs['compile_options'] = {'math_mode': 'fast'}
        elif args.candidate == 'halfdot':
            start = header.index('    uint sum = 0x6400u;')
            end = header.index('    half inv', start)
            kwargs['header'] = header[:start] + '''    half4 bytes = half4(as_type<uchar4>(bits));
    half value = half(1024) + dot(bytes, half4(1));
''' + header[end:]
        kwargs['name'] += '_' + args.candidate
        kernel = original(**kwargs)
        def call(**kw):
            if args.candidate == 'lut':
                kw['inputs'] = kw['inputs'] + [lut]
            return kernel(**kw)
        return call

    if args.candidate == 'halfdot':
        from mlxl3.kernels.common import specialized_codebook_header
        check = factory(name='mlxl3_mul1_check', input_names=['words'], output_names=['values'],
                        header=specialized_codebook_header(2),
                        source='uint i=thread_position_in_grid.x; values[i]=half(mlxl3_decode_codeword(words[i],2));',
                        compile_options={'math_mode': 'safe'})
        values = check(inputs=[mx.arange(65536,dtype=mx.uint32)], grid=(65536,1,1),
                       threadgroup=(256,1,1), output_shapes=[(65536,)], output_dtypes=[mx.float16])[0]
        assert mx.array_equal(values, lut).item(), 'MUL1 codebook changed'

    if args.candidate == 'tensor':
        import mlxl3.kernels.qmv as qmv
        qmv.tensor_tile = lambda *a: tuple(args.tile)
    elif args.candidate in ('nt1', 'sg2', 'sg4'):
        import mlxl3.kernels.qmv as qmv
        if args.candidate == 'nt1':
            qmv._qmv_tiles_per_group = lambda *a: 1
        else:
            qmv._qmv_simdgroups = lambda *a: int(args.candidate[-1])
    elif args.candidate == 'k3':
        import mlxl3.kernels.qmv as qmv
        mapped = qmv._qmv_mapped_tile_kernel
        def mapped_k3(*a, **kw):
            if a[0] == 3 and len(a) < 7:
                kw['k3_window_decode'] = True
            return mapped(*a, **kw)
        qmv._qmv_mapped_tile_kernel = mapped_k3
    elif args.candidate not in ('base', 'optimized', 'tuned'):
        mx.fast.metal_kernel = factory
    started = time.perf_counter()
    model, tokenizer, *_ = _load_model(args.model)
    print(json.dumps({'load_s': time.perf_counter()-started, 'active_bytes': mx.get_active_memory()}), flush=True)
    messages = [{'role': 'user', 'content': 'Explique pourquoi le ciel est bleu en plusieurs paragraphes. '}]
    for run in range(args.runs):
        mx.reset_peak_memory()
        text, stats = _stream_response(model, tokenizer, messages, max_tokens=args.tokens,
                                      temperature=0, top_k=0, repetition_penalty=1.05,
                                      on_text=lambda _: None, session=GenerationSession())
        print(json.dumps({'candidate': args.candidate, 'run': run, **asdict(stats), 'text': text}), flush=True)


if __name__ == '__main__':
    main()
