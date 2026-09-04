"""Compare logits and complete cache state on identical forced continuations.

Diagnostics only: synchronization and CPU hashing invalidate performance
measurements. Each configuration runs in its own process, never two resident
models. Output directories must be new. Non-exact candidates exit nonzero.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def worker(args):
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache

    from mlxl3.cli import GenerationSession, _load_model

    mx.random.seed(7304)
    model, tokenizer, *_ = _load_model(args.model)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': args.prompt * args.repeat}],
        tokenize=False, add_generation_prompt=True,
    )
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if args.prompt_tokens:
        tokens = (tokens * ((args.prompt_tokens + len(tokens) - 1) // len(tokens)))[:args.prompt_tokens]
    cache = make_prompt_cache(model)
    reference_path = args.output / 'reference.json'
    reference = json.loads(reference_path.read_text()) if args.worker == 'candidate' else None

    def digest(value):
        if isinstance(value, mx.array):
            raw = np.asarray(mx.contiguous(value).view(mx.uint8)).tobytes()
            return {'shape': value.shape, 'dtype': str(value.dtype),
                    'sha256': hashlib.sha256(raw).hexdigest()}
        if isinstance(value, (tuple, list)):
            return [digest(v) for v in value]
        if isinstance(value, dict):
            return {str(k): digest(v) for k, v in value.items()}
        return value

    rows = []
    held_snapshots = []
    for step in range(args.steps):
        if step and args.fork_every and step % args.fork_every == 0:
            held_snapshots.append((cache, json.loads(json.dumps([digest((c.state, c.meta_state)) for c in cache]))))
            cache = GenerationSession._clone_cache(cache)
        logits = model(mx.array(tokens)[None], cache=cache)[:, -1, :]
        mx.eval(logits, [c.state for c in cache])
        host = np.asarray(logits.astype(mx.float32))
        state = json.loads(json.dumps([digest((c.state, c.meta_state)) for c in cache]))
        chosen = int(mx.argmax(logits, axis=-1).item())
        path = args.output / f'logits-{step:04d}.npy'
        finite = bool(np.all(np.isfinite(host)))
        row = {'step': step, 'argmax': chosen, 'cache': state,
               'logits_dtype': str(logits.dtype), 'finite': finite}
        if reference is None:
            np.save(path, host, allow_pickle=False)
            row['forced_next_token'] = chosen
        else:
            expected = np.load(path, allow_pickle=False)
            row.update(
                logits_bit_exact=(finite and bool(np.all(np.isfinite(expected)))
                                  and str(logits.dtype) == reference['rows'][step]['logits_dtype']
                                  and host.tobytes() == expected.tobytes()),
                max_abs_error=(float(np.max(np.abs(host - expected)))
                               if finite and np.all(np.isfinite(expected)) else None),
                cache_bit_exact=state == reference['rows'][step]['cache'],
                forced_next_token=reference['rows'][step]['forced_next_token'],
                argmax_equal=chosen == reference['rows'][step]['argmax'],
            )
        rows.append(row)
        tokens = [row['forced_next_token']]
        # Keep only hashes/metadata, never GPU output/state histories.
    for snapshot, expected in held_snapshots:
        assert json.loads(json.dumps([digest((c.state, c.meta_state)) for c in snapshot])) == expected, 'fork corrupted retained state'
    result = {'model': str(args.model), 'rows': rows, 'steps': args.steps,
              'flags': {k: v for k, v in os.environ.items() if k.startswith(('MLX_', 'MLXL3_'))},
              'scope': 'logits + all cache arrays/meta; not every intermediate activation'}
    if reference is not None:
        result['exact'] = all(r['logits_bit_exact'] and r['cache_bit_exact'] for r in rows)
    (args.output / f'{args.worker}.json').write_text(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--prompt-tokens', type=int, default=0)
    parser.add_argument('--fork-every', type=int, default=0)
    parser.add_argument('--prompt', default='Explique pourquoi le ciel est bleu. ')
    parser.add_argument('--baseline-env', action='append', default=[])
    parser.add_argument('--candidate-env', action='append', default=[])
    parser.add_argument('--worker', choices=['reference', 'candidate'])
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    args.output.mkdir(parents=True, exist_ok=False)
    for mode, flags in [('reference', args.baseline_env), ('candidate', args.candidate_env)]:
        env = dict(os.environ)
        for flag in flags:
            key, value = flag.split('=', 1)
            if not key.startswith(('MLX_', 'MLXL3_')):
                raise ValueError('only MLX_/MLXL3_ experiment flags accepted')
            env[key] = value
        subprocess.run([
            sys.executable, __file__, str(args.model), '--output', str(args.output),
            '--steps', str(args.steps), '--repeat', str(args.repeat), '--prompt', args.prompt,
            '--prompt-tokens', str(args.prompt_tokens),
            '--fork-every', str(args.fork_every),
            '--worker', mode,
        ], env=env, check=True)
    result = json.loads((args.output / 'candidate.json').read_text())
    errors = [r['max_abs_error'] for r in result['rows']]
    print(json.dumps({'exact': result['exact'], 'steps': args.steps,
                      'finite': all(r['finite'] for r in result['rows']),
                      'max_abs_error': max(errors) if all(e is not None for e in errors) else None}))
    if not result['exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
