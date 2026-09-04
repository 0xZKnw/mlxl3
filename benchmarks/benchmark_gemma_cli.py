"""Paired real streaming generation; toggle only scaled Hadamard kernels."""
import argparse
import json
import statistics
from pathlib import Path

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler
from mlxl3.cli import _load_model
import mlxl3.kernels.qmv as qmv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    model, tokenizer, *_ = _load_model(args.model)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Explique en détail pourquoi le ciel est bleu et pourquoi il devient rouge au coucher du soleil.'}],
        tokenize=False, add_generation_prompt=True)

    def run(enabled, count=192):
        qmv._USE_SCALED_HADAMARD = enabled
        for layer in model.layers:
            for module in (layer.mlp, layer.router):
                original = module._mlxl3_stateless_original_call
                module._mlxl3_compiled_stateless = mx.compile(lambda x, call=original: call(x))
        for _ in stream_generate(model, tokenizer, prompt, max_tokens=8, sampler=make_sampler(temp=0)):
            pass
        final = None
        tokens, pieces = [], []
        for response in stream_generate(model, tokenizer, prompt, max_tokens=count, sampler=make_sampler(temp=0)):
            final = response
            tokens.append(response.token)
            pieces.append(response.text)
        if final is None:
            raise RuntimeError('generation returned no responses')
        return final, tokens, ''.join(pieces)

    run(False, 8)
    run(True, 8)
    speeds = {False: [], True: []}
    for pair in range(5):
        tokens = {}
        for enabled in ((False, True) if pair % 2 == 0 else (True, False)):
            last, tokens[enabled], text = run(enabled)
            speeds[enabled].append(last.generation_tps)
        print(json.dumps({'pair': pair, 'baseline': speeds[False][-1], 'optimized': speeds[True][-1], 'tokens_equal': tokens[False] == tokens[True], 'tokens': len(tokens[True])}), flush=True)
        assert tokens[False] == tokens[True], 'greedy token regression'
    print(json.dumps({'baseline_median': statistics.median(speeds[False]), 'optimized_median': statistics.median(speeds[True]), 'paired_gain_percent': statistics.median([(b/a-1)*100 for a,b in zip(speeds[False],speeds[True])]), 'text': text, 'peak_gb': mx.get_peak_memory()/1e9}), flush=True)


if __name__ == '__main__':
    main()
