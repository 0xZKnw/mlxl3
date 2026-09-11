"""OPT-2026-09-10-07/08/09: isolated sampler experiments, never enabled in production."""
import argparse
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from functools import lru_cache, partial
from pathlib import Path

import mlx.core as mx
from mlx_lm import sample_utils

REFERENCE = sample_utils.apply_top_k
REFERENCE_PENALTY = sample_utils.make_repetition_penalty


@lru_cache(maxsize=8)
def compiled_penalty(penalty, context_size=20):
    original = REFERENCE_PENALTY(penalty, context_size)
    compiled = mx.compile(original)

    def process(tokens, logits):
        if not len(tokens):
            return logits
        return compiled(tokens[-context_size:], logits)

    return process


def penalty_check_micro():
    count = 0
    for dtype in (mx.float16, mx.bfloat16, mx.float32):
        for vocab in (257, 65536, 248320):
            for length in (0, 1, 19, 20, 21, 2048):
                tokens = mx.arange(length, dtype=mx.int32) % 5
                for penalty in (0.8, 1.0, 1.05, 2.0):
                    x = mx.random.normal((1, vocab)).astype(dtype)
                    x[:, :4] = mx.array([float("nan"), float("inf"), -float("inf"), -0.0], dtype)
                    a = REFERENCE_PENALTY(penalty)(tokens, mx.array(x))
                    b = compiled_penalty(penalty)(tokens, mx.array(x))
                    uint = mx.uint32 if dtype == mx.float32 else mx.uint16
                    if not mx.array_equal(a.view(uint), b.view(uint)).item():
                        different = a.view(uint) != b.view(uint)
                        finite = mx.isfinite(a) & mx.isfinite(b)
                        print(json.dumps({"kind": "parity_failure", "dtype": str(dtype),
                                          "vocab": vocab, "length": length, "penalty": penalty,
                                          "finite_mismatches": int(mx.sum(different & finite).item()),
                                          "max_finite_abs": float(mx.max(mx.where(finite, mx.abs(a-b), 0)).item()),
                                          "reference_head": a[:, :6].tolist(),
                                          "candidate_head": b[:, :6].tolist()}), flush=True)
                        raise AssertionError((dtype, vocab, length, penalty))
                    count += 1
    print(json.dumps({"kind": "correctness", "cases": count, "bit_exact": True}), flush=True)
    for vocab in (65536, 248320):
        x = mx.random.normal((1, vocab))
        tokens = mx.arange(20, dtype=mx.int32)
        functions = (REFERENCE_PENALTY(1.05), compiled_penalty(1.05))
        for fn in functions:
            mx.eval(fn(tokens, mx.array(x)))
        for pair in range(5):
            for candidate in ((False, True) if pair % 2 == 0 else (True, False)):
                started = time.perf_counter()
                for _ in range(30):
                    mx.eval(functions[candidate](tokens, mx.array(x)))
                print(json.dumps({"kind": "micro", "vocab": vocab, "pair": pair,
                                  "candidate": candidate,
                                  "wall_ms": (time.perf_counter() - started) * 1000 / 30}), flush=True)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def retained_top_k(logprobs, top_k):
    vocab = logprobs.shape[-1]
    if not isinstance(top_k, int) or not 0 < top_k < vocab:
        raise ValueError("top_k outside vocabulary")
    kept = mx.argpartition(-logprobs, kth=top_k - 1, axis=-1)[..., :top_k]
    values = mx.take_along_axis(logprobs, kept, axis=-1)
    return mx.put_along_axis(mx.full_like(logprobs, -float("inf")), kept, values, axis=-1)


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def hierarchical_top_k(logprobs, top_k):
    vocab = logprobs.shape[-1]
    block = 1024
    if not isinstance(top_k, int) or not 0 < top_k < vocab:
        raise ValueError("top_k outside vocabulary")
    if vocab <= block or top_k >= block:
        return REFERENCE(logprobs, top_k)
    pad = (-vocab) % block
    x = mx.pad(logprobs, [(0, 0)] * (logprobs.ndim - 1) + [(0, pad)],
               constant_values=float("nan")) if pad else logprobs
    chunks = x.reshape(*logprobs.shape[:-1], -1, block)
    local = mx.argpartition(-chunks, kth=top_k - 1, axis=-1)[..., :top_k]
    values = mx.take_along_axis(chunks, local, axis=-1).reshape(*logprobs.shape[:-1], -1)
    indices = (local + mx.arange(chunks.shape[-2], dtype=local.dtype)[:, None] * block)
    indices = indices.reshape(*logprobs.shape[:-1], -1)
    final = mx.argpartition(-values, kth=top_k - 1, axis=-1)[..., :top_k]
    kept = mx.take_along_axis(indices, final, axis=-1)
    return mx.put_along_axis(mx.full_like(logprobs, -float("inf")), kept,
                             mx.take_along_axis(logprobs, kept, axis=-1), axis=-1)


candidate_fn = retained_top_k


def check():
    count = 0
    for dtype in (mx.float16, mx.bfloat16, mx.float32):
        for vocab in (7, 257, 1023, 1024, 1025, 2049, 65536, 248320, 262144):
            for shape in ((vocab,), (2, vocab)):
                for case in ("random", "ties", "nonfinite", "allnan", "zeros"):
                    mx.random.seed(712)
                    x = mx.random.normal(shape).astype(dtype)
                    if case == "ties":
                        x = mx.round(x)
                    elif case == "nonfinite":
                        x[..., :4] = mx.array([float("nan"), float("inf"), -float("inf"), -0.0], dtype)
                    elif case == "allnan":
                        x = mx.full(shape, float("nan"), dtype)
                    elif case == "zeros":
                        x = mx.zeros(shape, dtype)
                    for k in sorted({1, min(40, vocab - 1), min(80, vocab - 1), vocab - 1}):
                        a, b = REFERENCE(x, k), candidate_fn(x, k)
                        uint = mx.uint32 if dtype == mx.float32 else mx.uint16
                        assert mx.array_equal(a.view(uint), b.view(uint)).item(), (dtype, shape, case, k)
                        if case not in ("nonfinite", "allnan"):
                            for value in (a, b):
                                mx.random.seed(512)
                                token = sample_utils.categorical_sampling(value, 0.7)
                                mx.eval(token)
                                if value is a:
                                    expected = token
                                else:
                                    assert mx.array_equal(expected, token).item()
                        count += 1
    print(json.dumps({"kind": "correctness", "cases": count, "bit_exact": True}), flush=True)


def micro():
    for vocab in (65536, 248320, 262144):
        x = mx.random.normal((1, vocab))
        mx.eval(x)
        for fn in (REFERENCE, candidate_fn):
            for _ in range(5):
                mx.eval(fn(x, 80))
        for pair in range(5):
            for candidate in ((False, True) if pair % 2 == 0 else (True, False)):
                fn = candidate_fn if candidate else REFERENCE
                started = time.perf_counter()
                for _ in range(30):
                    mx.eval(fn(x, 80))
                ms = (time.perf_counter() - started) * 1000 / 30
                print(json.dumps({"kind": "micro", "vocab": vocab, "pair": pair,
                                  "candidate": candidate, "wall_ms": ms}), flush=True)


def model_bench(path):
    from mlxl3.cli import GenerationSession, _load_model, _stream_response
    model, tokenizer, *_ = _load_model(path)
    messages = [{"role": "user", "content": "Explique en détail pourquoi le ciel est bleu."}]
    pairs = []
    try:
        for pair in range(-1, 3):
            rows = {}
            for candidate in ((False, True) if pair % 2 == 0 else (True, False)):
                if args.penalty:
                    sample_utils.make_repetition_penalty = compiled_penalty if candidate else REFERENCE_PENALTY
                else:
                    sample_utils.apply_top_k = candidate_fn if candidate else REFERENCE
                mx.synchronize()
                mx.random.seed(811 + pair)
                mx.reset_peak_memory()
                text, stats = _stream_response(
                    model, tokenizer, messages, max_tokens=32 if pair < 0 else 128,
                    temperature=args.temperature, top_k=args.top_k, repetition_penalty=1.05,
                    session=GenerationSession(), on_text=lambda _: None)
                rows[candidate] = (text, stats)
                print(json.dumps({"kind": "warmup" if pair < 0 else "model", "pair": pair,
                                  "model": str(path), "candidate": candidate,
                                  "text_hash": hashlib.sha256(text.encode()).hexdigest(),
                                  **asdict(stats)}), flush=True)
            assert rows[False][0] == rows[True][0], "sampled generation changed"
            if pair >= 0:
                pairs.append(100 * (rows[True][1].decode_tps / rows[False][1].decode_tps - 1))
        print(json.dumps({"kind": "summary", "decode_paired_percent": pairs,
                          "median_percent": statistics.median(pairs), "texts_equal": True}), flush=True)
    finally:
        sample_utils.apply_top_k = REFERENCE
        sample_utils.make_repetition_penalty = REFERENCE_PENALTY
        mx.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--hierarchical", action="store_true")
    parser.add_argument("--penalty", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-k", type=int, default=80)
    args = parser.parse_args()
    if args.penalty and args.hierarchical:
        parser.error("test one change at a time")
    from benchmark_audit_runtime import manifest
    record = manifest(args.model)
    record["contract"] = "Exact sampler comparison; candidate rejected on numerical mismatch"
    record["arguments"] = {key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()}
    print(json.dumps(record), flush=True)
    if args.hierarchical:
        candidate_fn = hierarchical_top_k
    if args.model:
        model_bench(args.model)
    elif args.penalty:
        penalty_check_micro()
    else:
        check()
        micro()
