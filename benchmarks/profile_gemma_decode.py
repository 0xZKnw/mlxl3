"""Synchronized stage attribution (diagnostic, not a throughput benchmark)."""
import json
import sys
import time
from collections import defaultdict

import mlx.core as mx
import mlxl3.recurrent as recurrent
import mlxl3.kernels.qmv as qmv
from mlxl3.checkpoint import load_exl3_model
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.utils import load_tokenizer

recurrent._USE_COMPILED_RECURRENT_LAYERS = False
model, _, _ = load_exl3_model(sys.argv[1], lazy=False)
tokenizer = load_tokenizer(sys.argv[1])
prompt = tokenizer.apply_chat_template([{'role':'user','content':'Explique pourquoi le ciel est bleu.'}], tokenize=True, add_generation_prompt=True)
cache = make_prompt_cache(model)
logits = model(mx.array([prompt]), cache=cache)
mx.eval(logits)
token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
for _ in range(4):
    logits = model(token, cache=cache)
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]

stats = defaultdict(lambda: [0, 0.0])

def timed(name, function):
    def call(*args, **kwargs):
        mx.synchronize()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        mx.eval(result)
        stats[name][0] += 1
        stats[name][1] += time.perf_counter()-start
        return result
    return call

for name in ('_scaled_hadamard_input', '_scaled_hadamard_output', '_scaled_hadamard_output_reduce'):
    setattr(qmv, name, timed(name, getattr(qmv, name)))

for name in ('_qmv_tile_kernel', '_qmv_mapped_tile_kernel', '_qmv_grouped_tile_kernel'):
    if not hasattr(qmv, name):
        continue
    original = getattr(qmv, name)
    def factory(*args, original=original, name=name, **kwargs):
        label = name + ':' + str(args[:4])
        return timed(label, original(*args, **kwargs))
    setattr(qmv, name, factory)

start = time.perf_counter()
for _ in range(8):
    logits = model(token, cache=cache)
    mx.eval(logits)
    token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
elapsed = time.perf_counter()-start
print(json.dumps({'synchronized_ms_per_token': elapsed*1000/8,
                  'stages': [{'name': k, 'calls_per_token': v[0]/8, 'ms_per_token': v[1]*1000/8} for k,v in sorted(stats.items(), key=lambda item:-item[1][1])]}, indent=2))
