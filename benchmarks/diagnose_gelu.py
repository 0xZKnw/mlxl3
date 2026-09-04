"""Compare each Gemma expert block on the same real reference activations."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlxl3 import moe
from mlxl3.cli import _load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    moe._USE_GELU_DOWN_PREP = False
    model, tokenizer, *_ = _load_model(args.model)
    errors = []
    preparations = []
    original_prepare = moe._fused_glu_down_prepare

    def inspect_prepare(raw, scales, down_scales, selected, activation='silu', logical_hidden=None):
        from mlx import nn

        from mlxl3.kernels.qmv import _scaled_hadamard_input, _scaled_hadamard_output
        actual = original_prepare(raw, scales, down_scales, selected, activation, logical_hidden)
        gu = _scaled_hadamard_output(raw, scales[selected].reshape(raw.shape)).reshape(-1, 2, raw.shape[-1])
        activated = nn.gelu_approx(gu[:, 0])
        hidden = activated * gu[:, 1]
        if logical_hidden is not None:
            hidden = mx.where(mx.arange(raw.shape[-1]) < logical_hidden, hidden, 0)
        expected = _scaled_hadamard_input(hidden, down_scales[selected])
        mx.eval(actual, expected)
        def stats(value):
            finite = bool(mx.all(mx.isfinite(value)).item())
            return {'shape': value.shape, 'dtype': str(value.dtype), 'finite': finite,
                    'max_abs': float(mx.max(mx.abs(value)).item()) if finite else None}
        preparations.append({key: stats(value) for key, value in {
            'raw': raw, 'gate': gu[:, 0], 'up': gu[:, 1], 'activated': activated,
            'hidden': hidden, 'expected': expected, 'actual': actual}.items()})
        return actual

    moe._fused_glu_down_prepare = inspect_prepare
    for name, module in list(model.named_modules()):
        if not isinstance(module, moe.EXL3SwitchGLU):
            continue
        original = module.__call__

        def compare(self, x, indices, scores=None, *, call=original, name=name):
            moe._USE_GELU_DOWN_PREP = False
            expected = call(x, indices, scores=scores)
            try:
                moe._USE_GELU_DOWN_PREP = True
                actual = call(x, indices, scores=scores)
                mx.eval(expected, actual)
            finally:
                moe._USE_GELU_DOWN_PREP = False
            finite = bool(mx.all(mx.isfinite(actual)).item())
            errors.append({'module': name, 'finite': finite,
                           'input_dtype': str(x.dtype),
                           'max_input': float(mx.max(mx.abs(x)).item()),
                           'max_ref': float(mx.max(mx.abs(expected)).item()),
                           'max_error': float(mx.max(mx.abs(expected - actual)).item()) if finite else None})
            return expected

        module.__class__ = type('Diagnosed' + type(module).__name__, (type(module),), {'__call__': compare})
    tokens = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Explique pourquoi le ciel est bleu.'}],
                                           tokenize=True, add_generation_prompt=True)
    logits = model(mx.array(tokens)[None], cache=make_prompt_cache(model))
    mx.eval(logits)
    print(json.dumps({'layers': errors, 'preparations': preparations}, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
