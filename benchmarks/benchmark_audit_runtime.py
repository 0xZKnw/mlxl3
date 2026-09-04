"""Reproducible runtime audit; stream responses without retaining logprobs.

Run with --model for a cold/warm/cache-reuse smoke benchmark, not a paired
end-to-end speedup claim. The paired routing microbenchmark runs by default.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from mlxl3.cli import GenerationSession, _load_model, _stream_response
from mlxl3.kernels.routing import inverse_permutation


def command(*args):
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()


def emit(value):
    print(json.dumps(value), flush=True)


def manifest(model_path):
    metadata = {}
    if model_path:
        for path in sorted(model_path.glob('*.json')):
            metadata[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        'kind': 'manifest', 'git': command('git', 'rev-parse', 'HEAD'),
        'dirty': command('git', 'status', '--porcelain'),
        'diff_sha256': hashlib.sha256(command('git', 'diff', 'HEAD').encode()).hexdigest(),
        'mlx': importlib.metadata.version('mlx'),
        'mlx_lm': importlib.metadata.version('mlx-lm'),
        'mlx_lm_origin': importlib.metadata.distribution('mlx-lm').read_text('direct_url.json'),
        'os': platform.platform(), 'os_build': command('sw_vers', '-buildVersion'),
        'sdk': command('xcrun', '--show-sdk-version'),
        'device': mx.device_info(),
        'flags': {k: v for k, v in os.environ.items() if k.startswith(('MLXL3_', 'MLX_'))},
        'model': str(model_path) if model_path else None,
        'metadata_sha256': metadata, 'weights_hashed': False,
        'gpu_time': None, 'thermal_state': 'unmeasured',
        'contract': 'Q0 addressing/lifetime; no new quantization or sampling change',
    }


def routing():
    for size in (512, 4096, 16384):
        mx.random.seed(size)
        order = mx.argsort(mx.random.randint(0, 256, (size,)))
        mx.eval(order)
        functions = {'argsort': mx.argsort, 'scatter': inverse_permutation}
        for fn in functions.values():
            for _ in range(5):
                mx.eval(fn(order))
        assert inverse_permutation(order).tolist() == mx.argsort(order).tolist()
        gains = []
        for pair in range(9):
            times = {}
            for name in (('argsort', 'scatter') if pair % 2 == 0 else ('scatter', 'argsort')):
                start = time.perf_counter()
                for _ in range(50):
                    mx.eval(functions[name](order))
                times[name] = (time.perf_counter() - start) / 50
            gains.append((times['argsort'] / times['scatter'] - 1) * 100)
            emit({'kind': 'routing_pair', 'slots': size, 'pair': pair, 'wall_seconds': times})
        emit({'kind': 'routing_summary', 'slots': size,
              'median_speedup_percent': statistics.median(gains),
              'scope': 'synchronized isolated operation, not model decode or DRAM bandwidth'})


def generation(path, count, repeats):
    model, tokenizer, *_ = _load_model(path)
    session = GenerationSession()
    messages = [{'role': 'user', 'content': 'Explique pourquoi le ciel est bleu en plusieurs paragraphes. ' * repeats}]
    for label, current_session in [('cold', session), ('warm_uncached', GenerationSession()), ('cached_prefix', session)]:
        mx.reset_peak_memory()
        text, stats = _stream_response(
            model, tokenizer, messages, max_tokens=count, temperature=0,
            top_k=0, repetition_penalty=1.05, on_text=lambda text: None,
            session=current_session,
        )
        emit({'kind': 'generation', 'scenario': label, **asdict(stats),
              'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
              'active_bytes': mx.get_active_memory(), 'allocator_cache_bytes': mx.get_cache_memory(),
              'session_estimated_bytes': current_session.nbytes(),
              'logprob_vectors_retained': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--prompt-repeats', type=int, default=1)
    parser.add_argument('--reference-routing', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output:
        # Exclusive creation protects previous raw benchmark evidence.
        sys.stdout = args.output.open('x')
    emit(manifest(args.model))
    emit({'kind': 'arguments', 'reference_routing': args.reference_routing,
          'tokens': args.tokens, 'prompt_repeats': args.prompt_repeats})
    routing()
    if args.reference_routing:
        from mlxl3 import moe
        moe.inverse_permutation = mx.argsort
    if args.model:
        generation(args.model, args.tokens, args.prompt_repeats)
