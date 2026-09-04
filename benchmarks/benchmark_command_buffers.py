"""Isolated-process sweep: MLX reads command-buffer limits at initialization."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def worker(args):
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from mlxl3.cli import _load_model

    model, tokenizer, *_ = _load_model(args.model)
    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": "Explique en détail pourquoi le ciel est bleu et le soleil rouge au coucher.",
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
    )
    list(stream_generate(model, tokenizer, prompt, max_tokens=16, sampler=make_sampler(temp=0)))
    rows = []
    for _ in range(2):
        responses = list(
            stream_generate(
                model, tokenizer, prompt, max_tokens=args.tokens, sampler=make_sampler(temp=0)
            )
        )
        last = responses[-1]
        digest = hashlib.sha256(json.dumps([r.token for r in responses]).encode()).hexdigest()
        rows.append(
            {
                "tps": last.generation_tps,
                "prefill_tps": last.prompt_tps,
                "tokens": len(responses),
                "digest": digest,
            }
        )
    print(
        json.dumps(
            {
                "runs": rows,
                "median_tps": statistics.median(r["tps"] for r in rows),
                "peak_gb": mx.get_peak_memory() / 1e9,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--tokens", type=int, default=96)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    configs = [(40, 40), (128, 256), (512, 1024), (10, 40)]
    baseline = None
    for round_id in range(2):
        for ops, mb in configs if round_id == 0 else configs[::-1]:
            env = dict(os.environ, MLX_MAX_OPS_PER_BUFFER=str(ops), MLX_MAX_MB_PER_BUFFER=str(mb))
            result = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    str(args.model),
                    "--worker",
                    "--tokens",
                    str(args.tokens),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
                check=True,
            )
            row = json.loads(result.stdout.strip().splitlines()[-1])
            hashes = [r["digest"] for r in row["runs"]]
            if baseline is None:
                baseline = hashes[0]
            row.update(
                round=round_id, ops=ops, mb=mb, tokens_equal=all(h == baseline for h in hashes)
            )
            print(json.dumps(row), flush=True)
            assert row["tokens_equal"], "generation divergence"


if __name__ == "__main__":
    main()
