"""Paired Gemma EXL3 decode benchmark with identical tokens and fresh caches."""
import argparse
import json
import statistics
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_tokenizer
from mlxl3.checkpoint import load_exl3_model
import mlxl3.moe as moe
import mlxl3.recurrent as recurrent
import mlxl3.linear as linear


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    args = parser.parse_args()
    model, _, _ = load_exl3_model(args.model, lazy=False)
    tokenizer = load_tokenizer(args.model)
    prompt = tokenizer.apply_chat_template([{'role':'user','content':'Explique pourquoi le ciel est bleu.'}], tokenize=True, add_generation_prompt=True)
    prefix = mx.array([prompt])
    continuation = tokenizer.encode(' La lumière du Soleil traverse notre atmosphère et se diffuse dans toutes les directions. '*20, add_special_tokens=False)

    def run(enabled, count=128):
        moe._USE_FUSED_MOE_GLU_PREP = enabled
        recurrent._USE_COMPILED_RECURRENT_LAYERS = enabled
        linear._USE_PADDED_GROUPS = enabled
        cache = make_prompt_cache(model)
        mx.eval(model(prefix, cache=cache))
        outputs = []
        start = time.perf_counter()
        for index in range(count):
            token = mx.array([[continuation[index]]])
            logits = model(token, cache=cache)
            mx.eval(logits)
            outputs.append(logits)
        elapsed = time.perf_counter() - start
        return count / elapsed, outputs

    run(False, 4)
    run(True, 4)
    results = {False: [], True: []}
    max_error = 0.0
    agreements = []
    for pair in range(6):
        outputs = {}
        for enabled in ((False, True) if pair % 2 == 0 else (True, False)):
            speed, outputs[enabled] = run(enabled)
            results[enabled].append(speed)
        for a, b in zip(outputs[False], outputs[True]):
            assert bool(mx.all(mx.isfinite(a))) and bool(mx.all(mx.isfinite(b))), 'nonfinite logits'
            max_error = max(max_error, float(mx.max(mx.abs(a-b))))
            agreements.append(int(mx.argmax(a)) == int(mx.argmax(b)))
        print(json.dumps({'pair': pair, 'baseline': results[False][-1],
                          'optimized': results[True][-1]}), flush=True)
    print(json.dumps({'baseline_median': statistics.median(results[False]),
                      'optimized_median': statistics.median(results[True]),
                      'paired_gain_percent': statistics.median([(b/a-1)*100 for a,b in zip(results[False],results[True])]),
                      'max_logit_absolute_error': max_error,
                      'argmax_agreement': sum(agreements)/len(agreements),
                      'peak_gb': mx.get_peak_memory()/1e9}), flush=True)


if __name__ == '__main__':
    main()
