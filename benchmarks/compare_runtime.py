"""Alternating-process runtime A/B with raw rows and paired bootstrap intervals."""
import argparse
import hashlib
import json
import os
import random
import resource
import statistics
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def worker(args):
    import mlx.core as mx

    from mlxl3.cli import GenerationSession, _load_model, _stream_response
    mx.random.seed(781)
    model, tokenizer, *_ = _load_model(args.model)
    messages = [{'role': 'user', 'content': 'Explique en détail pourquoi le ciel est bleu. ' * args.repeat}]
    session = GenerationSession()
    results = []
    for label, cache in [('warmup', GenerationSession()), ('uncached', session), ('cached', session)]:
        mx.reset_peak_memory()
        text, stats = _stream_response(model, tokenizer, messages, max_tokens=32 if label == 'warmup' else args.tokens,
                                      temperature=0, top_k=0, repetition_penalty=1.05,
                                      session=cache, on_text=lambda text: None)
        if label != 'warmup':
            results.append({'scenario': label, **asdict(stats),
                            'text_hash': hashlib.sha256(text.encode()).hexdigest(),
                            'active_bytes': mx.get_active_memory(), 'allocator_cache_bytes': mx.get_cache_memory(),
                            'session_estimated_bytes': cache.nbytes(),
                            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(json.dumps(results))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=128)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--baseline-env', action='append', default=[])
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    from benchmark_audit_runtime import manifest
    with args.output.open('x') as handle:
        handle.write(json.dumps(manifest(args.model)) + '\n')
        handle.write(json.dumps({'kind': 'arguments', 'baseline': args.baseline_env,
                                 'candidate': args.candidate_env, 'pairs': args.pairs,
                                 'tokens': args.tokens, 'repeat': args.repeat,
                                 'protocol': 2, 'warmup': 'same session/prefix path, separate fresh cache, 32 decode tokens'}) + '\n')
        paired = {'uncached': [], 'cached': []}
        for pair in range(args.pairs):
            batch = {}
            for mode in (['baseline', 'candidate'] if pair % 2 == 0 else ['candidate', 'baseline']):
                env = dict(os.environ)
                for flag in getattr(args, mode + '_env'):
                    key, value = flag.split('=', 1)
                    if not key.startswith(('MLXL3_', 'MLX_')):
                        raise ValueError('only experiment flags accepted')
                    env[key] = value
                result = subprocess.run([sys.executable, __file__, str(args.model), '--worker',
                                         '--output', str(args.output), '--tokens', str(args.tokens),
                                         '--repeat', str(args.repeat)], env=env, text=True,
                                        capture_output=True, check=True)
                rows = json.loads(result.stdout)
                batch[mode] = {r['scenario']: r for r in rows}
                for row in rows:
                    handle.write(json.dumps({'kind': 'run', 'pair': pair, 'mode': mode, **row}) + '\n')
                handle.flush()
            for scenario, scenario_pairs in paired.items():
                a, b = batch['baseline'][scenario], batch['candidate'][scenario]
                scenario_pairs.append((a, b))
            print(f'pair {pair + 1}/{args.pairs} complete', flush=True)
        rng = random.Random(731)
        for scenario, pairs in paired.items():
            same_text = all(a['text_hash'] == b['text_hash'] for a, b in pairs)
            for metric in ('decode_tps', 'prefill_tps', 'ttft_seconds', 'peak_memory_gb'):
                higher_better = metric.endswith('tps')
                gains = [100 * ((b[metric] / a[metric] if higher_better else a[metric] / b[metric]) - 1)
                         for a, b in pairs if a[metric] > 0 and b[metric] > 0]
                if not gains:
                    continue
                samples = sorted(statistics.median(rng.choices(gains, k=len(gains))) for _ in range(2000))
                summary = {'kind': 'summary', 'scenario': scenario, 'metric': metric,
                           'median_gain_percent': statistics.median(gains), 'paired_gains': gains,
                           'bootstrap_95_percent': [samples[50], samples[1949]],
                           'texts_equal': same_text, 'thermal_state': 'uncontrolled'}
                handle.write(json.dumps(summary) + '\n')
                print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
