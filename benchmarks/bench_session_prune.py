"""Real two-turn cache-budget A/B; hashes run outside request timings.

Uses one resident model, one conversation, alternating old-reset/new-retain
pairs and an unpruned-cache correctness control. Cold prefill uses a different
batch partition: require exact cache state against the warm control, not cold.
Creates a new JSONL file; never overwrites.
"""
import argparse
import gc
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('models/LFM2.5-2.6B-EXL3-4bpw'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--prompt-repeats', type=int, default=128)
    parser.add_argument('--first-tokens', type=int, default=512)
    parser.add_argument('--second-tokens', type=int, default=32)
    args = parser.parse_args()
    if min(args.pairs, args.prompt_repeats, args.first_tokens, args.second_tokens) < 1:
        parser.error('counts must be positive')
    import mlx.core as mx
    import numpy as np
    from benchmark_audit_runtime import command, manifest

    from mlxl3 import cli

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', buffering=1) as output:
        def emit(row):
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

        emit(manifest(args.model))
        emit({'kind': 'arguments', **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'power': command('pmset', '-g', 'batt'), 'warmup': 'one excluded old/new pair',
              'scope': 'Python streaming request; no network or native UI; hashes outside timing'})
        model, tokenizer, *_ = cli._load_model(args.model)
        note = 'Le projet Atlas est un carnet public. Son identifiant est ORION-73. Ces notes restent inchangées.\n'
        first_messages = [{'role': 'user', 'content': 'Lis ces notes et garde leur identifiant en mémoire.\n'
                           + note * args.prompt_repeats + '\nRéponds uniquement : Compris.'}]
        question = "Quel est l’identifiant du projet Atlas ? Réponds uniquement avec cet identifiant."
        options = {'temperature': 0, 'top_k': 0, 'repetition_penalty': 1.05, 'on_text': lambda _: None}

        def digest(value):
            if isinstance(value, mx.array):
                raw = np.asarray(mx.contiguous(value).view(mx.uint8)).tobytes()
                return [str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()]
            if isinstance(value, (list, tuple)):
                return [digest(v) for v in value]
            if isinstance(value, dict):
                return {str(k): digest(v) for k, v in value.items()}
            return value

        def cache_hashes(cache):
            return [hashlib.sha256(json.dumps(digest((c.state, c.meta_state)), sort_keys=True).encode()).hexdigest()
                    for c in cache]

        def run(pair, mode):
            warmup = pair < 0
            pool = cli.GenerationSessionPool()
            session = pool.acquire('active')
            try:
                mx.random.seed(7309)
                text, first_stats = cli._stream_response(
                    model, tokenizer, first_messages, session=session, max_tokens=args.first_tokens, **options)
                emit({'kind': 'first_turn', 'pair': pair, 'mode': mode, 'warmup': warmup,
                      'text': text, **asdict(first_stats)})
                if not session.exact_tokens or session.exact_tokens[-1] not in tokenizer.eos_token_ids:
                    raise RuntimeError('first turn did not complete; increase --first-tokens')
                second_messages = first_messages + [
                    {'role': 'assistant', 'content': cli._cache_context(text)},
                    {'role': 'user', 'content': question},
                ]
                stable_prompt = cli._render_generation_prompt(tokenizer, second_messages, None, add_generation_prompt=False)
                add_special = tokenizer.bos_token is None or not stable_prompt.startswith(tokenizer.bos_token)
                stable = tokenizer.encode(stable_prompt, add_special_tokens=add_special)
                if stable[:len(session.exact_tokens)] != session.exact_tokens:
                    raise RuntimeError('completed turn is not an exact prefix of the next chat template')
                first_cache = cache_hashes(session.exact_cache)
                exact_bytes = cli._cache_nbytes(session.exact_cache)
                seen = set()
                paired_bytes = cli._cache_nbytes(session.prompt_cache, seen) + cli._cache_nbytes(session.exact_cache, seen)
                pool.budget_bytes = (exact_bytes + paired_bytes) // 2
                if not 0 < exact_bytes <= pool.budget_bytes < paired_bytes:
                    raise RuntimeError('cache layout cannot distinguish the two policies')
                before = session.nbytes()
                started = time.perf_counter()
                if mode == 'old':
                    # Original policy, exactly equivalent for this one-session pool.
                    if session.nbytes() > pool.budget_bytes:
                        session.reset()
                elif mode == 'new':
                    pool.prune('active')
                prune_seconds = time.perf_counter() - started
                after = session.nbytes()
                assert mode == 'unpruned' or after <= pool.budget_bytes
                gc.collect()
                mx.clear_cache()
                mx.reset_peak_memory()
                text, stats = cli._stream_response(
                    model, tokenizer, second_messages, session=session, max_tokens=args.second_tokens, **options)
                row = {'kind': 'run', 'pair': pair, 'mode': mode, 'warmup': warmup,
                       'budget_bytes': pool.budget_bytes, 'exact_bytes': exact_bytes,
                       'prompt_plus_exact_bytes': paired_bytes, 'before_prune_bytes': before,
                       'after_prune_bytes': after, 'prune_seconds': prune_seconds,
                       'first_cache_sha256': first_cache, 'stable_prompt_sha256': hashlib.sha256(stable_prompt.encode()).hexdigest(),
                       'text': text, 'generated_token_ids': session.exact_tokens[stats.prompt_tokens:],
                       'cache_sha256': cache_hashes(session.exact_cache), **asdict(stats)}
                emit(row)
                return row
            finally:
                session.reset()
                gc.collect()
                mx.clear_cache()

        pairs = []
        for pair in range(-1, args.pairs):
            modes = ('old', 'new', 'unpruned') if pair % 2 == 0 else ('unpruned', 'new', 'old')
            results = {mode: run(pair, mode) for mode in modes}
            old, new = results['old'], results['new']
            equality = {field: old[field] == new[field] for field in (
                'first_cache_sha256', 'stable_prompt_sha256', 'text', 'generated_token_ids')}
            equality['unpruned_control'] = all(
                new[field] == results['unpruned'][field] for field in (
                    'first_cache_sha256', 'text', 'generated_token_ids', 'cache_sha256'))
            exact = all(equality.values())
            emit({'kind': 'pair', 'pair': pair, 'warmup': pair < 0, 'exact': exact, 'equality': equality,
                  'cold_cache_bit_exact': old['cache_sha256'] == new['cache_sha256'],
                  'old_ttft_seconds': old['ttft_seconds'], 'new_ttft_seconds': new['ttft_seconds'],
                  'old_evaluated_tokens': old['evaluated_prompt_tokens'], 'new_evaluated_tokens': new['evaluated_prompt_tokens']})
            print(f'pair {pair}: exact={exact}, TTFT {old["ttft_seconds"]:.3f}s → {new["ttft_seconds"]:.3f}s', flush=True)
            if not exact:
                raise SystemExit('output/cache mismatch; raw results retained, no exactness claim')
            if pair >= 0:
                pairs.append((old, new))
        emit({'kind': 'summary', 'pairs': len(pairs), 'exact': True,
              'old_ttft_p50_seconds': statistics.median(a['ttft_seconds'] for a, _ in pairs),
              'new_ttft_p50_seconds': statistics.median(b['ttft_seconds'] for _, b in pairs),
              'median_ttft_gain_percent': statistics.median((a['ttft_seconds'] / b['ttft_seconds'] - 1) * 100 for a, b in pairs)})


if __name__ == '__main__':
    main()
