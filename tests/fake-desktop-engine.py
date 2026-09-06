#!/usr/bin/env python3
"""Isolated engine lifecycle fixture, never imports MLX or user configuration."""
import json
import signal
import sys
import time

model = sys.argv[2]
signal.signal(signal.SIGUSR1, lambda *_: None)
def emit(kind, **values):
    print(json.dumps({'type': kind, **values}), flush=True)
emit('ready', model=model, modules=1, resident_gb=0.01, context_limit=2048, model_context_limit=2048)
for line in sys.stdin:
    request = json.loads(line)
    if request['type'] != 'generate':
        continue
    emit('delta', request_id=request['request_id'], phase='answer', text='hello')
    if model == 'crash':
        sys.exit(1)
    time.sleep(0.2)
    emit('complete', request_id=request['request_id'], assistant_context='done', cache_context='done')
