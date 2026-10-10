#!/usr/bin/env python3
"""Isolated engine lifecycle fixture, never imports MLX or user configuration."""
import json
import os
import signal
import sys
import time

if len(sys.argv) > 1 and sys.argv[1] == 'hub':
    command = sys.argv[2]
    query = sys.argv[3]
    if command == 'search':
        if query == 'error':
            print('fixture catalogue unavailable', file=sys.stderr)
            sys.exit(3)
        if query == 'slow':
            time.sleep(0.7)
        if query == 'empty':
            rows = []
        elif query == 'invalid':
            rows = [{'id': 'fixture/bad-exl3', 'downloads': 0, 'likes': 0, 'gated': []}]
        else:
            rows = [
                {'id': f'fixture/{query}-exl3', 'downloads': 73, 'likes': 2, 'gated': None},
                {'id': 'fixture/manual-exl3', 'downloads': 9, 'likes': 1, 'gated': 'manual'},
                {'id': 'fixture/open-exl3', 'downloads': 0, 'likes': 0, 'gated': False},
            ]
        print(json.dumps(rows), flush=True)
    elif command in ('download', 'resume'):
        total = 100_000_000
        for completed in [40_000_000, 50_000_000, 70_000_000, total]:
            print(json.dumps({'type': 'progress', 'completed': completed, 'total': total}), flush=True)
            if query == 'fixture/fail-exl3':
                print('fixture download failed', file=sys.stderr)
                sys.exit(3)
            time.sleep(0.35)
        print(json.dumps({'type': 'installed', 'model': {
            'name': 'fixture-model', 'path': '/tmp/mlxl3-fixture-model', 'model_type': 'audit',
            'format': 'EXL3', 'bits': 2.49, 'size_bytes': total, 'modules': 1,
            'added_at': '', 'size': '100 MB'}}), flush=True)
    elif command == 'pending':
        print(json.dumps([{'id': 'fixture-download', 'repo': 'fixture/download-exl3',
                           'size_bytes': 100_000_000, 'retained_bytes': 40_000_000}]), flush=True)
    elif command == 'details':
        print(json.dumps({'id': query, 'revision': 'main', 'commit': 'fixture',
                          'branches': ['main'], 'variants': [
                              {'id': '.', 'label': '2.49 bpw', 'size_bytes': 128}],
                          'readme': '# Fixture', 'gated': False, 'downloads': 73, 'likes': 2}), flush=True)
    else:
        sys.exit(2)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == 'list':
    print('[]', flush=True)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == 'mcp-fixture':
    for line in sys.stdin:
        request = json.loads(line)
        if 'id' not in request:
            continue
        method = request['method']
        if method == 'initialize':
            result = {'protocolVersion': '2025-11-25', 'capabilities': {'tools': {}},
                      'serverInfo': {'name': 'mlxl3-local-check', 'version': '1'}}
        elif method == 'tools/list':
            result = {'tools': [{'name': 'echo', 'description': 'Returns the provided message for a local test.',
                                 'inputSchema': {'type': 'object', 'properties': {'message': {'type': 'string'}},
                                                 'required': ['message']}}]}
        elif method == 'tools/call':
            time.sleep(0.2)
            result = {'content': [{'type': 'text', 'text': request['params']['arguments']['message']}], 'isError': False}
        else:
            result = {}
        print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == 'dflash-draft':
    print(json.dumps({'type': 'installed', 'path': '/tmp/mlxl3-fixture-dflash'}), flush=True)
    sys.exit(0)

if len(sys.argv) > 1 and sys.argv[1] == 'mtp-head':
    target = sys.argv[sys.argv.index('--target') + 1] if '--target' in sys.argv else ''
    if '/auto-' in target:
        with open(os.path.join(os.environ['MLXL3_HOME'], 'mtp-operations.jsonl'), 'a') as log:
            log.write(json.dumps({'args': sys.argv[1:], 'pid': os.getpid()}) + '\n')
        if target.endswith('auto-fail'):
            print('fixture head download failed', file=sys.stderr)
            sys.exit(1)
        if target.endswith('auto-slow'):
            print(json.dumps({'type': 'progress', 'completed': 1, 'total': 100}), flush=True)
            time.sleep(1)
    head = '/tmp/mlxl3-fixture-mtp-' + ('dense' if target.endswith(('auto-dense', 'auto-slow')) else 'moe') if '/auto-' in target else '/tmp/mlxl3-fixture-mtp'
    if '--inspect' in sys.argv and sys.argv[sys.argv.index('--inspect') + 1] != head:
        print('fixture head incompatible with target', file=sys.stderr)
        sys.exit(1)
    print(json.dumps({'type': 'installed', 'path': head}), flush=True)
    sys.exit(0)

model = sys.argv[2]
cancelled = False
def cancel(*_):
    global cancelled
    cancelled = True
signal.signal(signal.SIGUSR1, cancel)
def emit(kind, **values):
    print(json.dumps({'type': kind, **values}), flush=True)
mtp_capable = model == 'renamed' or model.startswith(('mtp-', 'auto-'))
tuning_key = 'fixture-runtime:' + model
capabilities = {} if model == 'mtp-old' else {'mtp_max_depth': 3, 'mtp_tune_supported': mtp_capable, 'mtp_tuning_key': tuning_key}
if model.startswith('auto-'):
    capabilities['mtp_configure_supported'] = True
if model == 'mtp-memory-saver':
    capabilities['context_memory'] = {'layers': [{'bytes_per_token': 32768, 'step': 1}], 'fixed_bytes': 4096}
emit('ready', model=model, modules=1, resident_gb=0.01, context_limit=2048, model_context_limit=2048,
     dflash_supported=model == 'renamed', mtp_supported=mtp_capable, mtp_auto_download_supported=mtp_capable,
     **capabilities, memory_saver_supported=model == 'mtp-memory-saver',
     bridge_protocol=1, runtime_commit='fixture', runtime_profile='release', mlx_version='0.32.2')
for line in sys.stdin:
    request = json.loads(line)
    if model == 'mtp-memory-saver':
        with open(os.path.join(os.environ['MLXL3_HOME'], 'memory-saver-requests.jsonl'), 'a') as log:
            log.write(json.dumps(request) + '\n')
    if request['type'] == 'set_mtp':
        with open(os.path.join(os.environ['MLXL3_HOME'], 'mtp-operations.jsonl'), 'a') as log:
            log.write(json.dumps({'model': model, 'request': request, 'pid': os.getpid()}) + '\n')
        if not request['enabled']:
            if model == 'auto-idle-malformed':
                emit('mtp_status', request_id=request['request_id'], mtp_active='invalid boolean')
                continue
            if model in ('auto-idle-error', 'auto-idle-empty-error'):
                emit('error', request_id='' if model.endswith('empty-error') else request['request_id'],
                     message='fixture MTP configuration failed')
                continue
            if model == 'auto-idle-silent':
                continue
        expected = '/tmp/mlxl3-fixture-mtp-' + ('dense' if model in ('auto-dense', 'auto-slow') else 'moe')
        if request['enabled'] and request['mtp_head_path'] != expected:
            emit('error', request_id=request['request_id'], message='foreign head reached bridge')
        else:
            emit('mtp_status', request_id=request['request_id'], mtp_active=request['enabled'])
        continue
    if request['type'] == 'tune_mtp':
        cancelled = False
        for step in range(12):
            emit('mtp_tune_progress', request_id=request['request_id'], phase='warmup' if step < 4 else 'measure',
                 depth=step % 4, completed=step, total=12)
            time.sleep(0.04)
            if cancelled:
                emit('cancelled', request_id=request['request_id'])
                break
        else:
            if model == 'mtp-error':
                emit('error', request_id=request['request_id'], message='fixture tuning failed')
                continue
            if model == 'mtp-malformed':
                emit('error', message='unreadable engine response')
                continue
            rates = [50, 60, 75, 70] if model != 'mtp-baseline' else [50, 49, 51, 50]
            rows = [{"depth": d, "decode_tps": rate, "decode_tokens": 190, "decode_seconds": 190/rate,
                     "accepted_tokens": 0 if d == 0 else 40, "proposed_tokens": 0 if d == 0 else 80,
                     "eligible": True, "reason": None, "token_hashes": ['prompt-a', 'prompt-b']} for d, rate in enumerate(rates)]
            if model == 'mtp-collapse':
                rows[3].update(decode_tps=100, decode_seconds=1.9, accepted_tokens=0, eligible=False, reason='zero_acceptance')
            if model == 'mtp-invalid':
                rows[3]['depth'] = 2
            emit('mtp_tune_complete', request_id=request['request_id'], tuning_key=tuning_key,
                 best_depth=0 if model == 'mtp-baseline' else 2, rows=rows)
        continue
    if request['type'] != 'generate':
        continue
    if model == 'context-count':
        emit('context_usage', request_id=request['request_id'], used_tokens=12, context_limit=2048)
        emit('delta', request_id=request['request_id'], phase='answer', text='hello')
        time.sleep(0.15)
        emit('complete', request_id=request['request_id'], assistant_context='done', cache_context='done',
             stats={'ttft_seconds': 0.1, 'prefill_tps': 100, 'decode_tps': 30,
                    'prompt_tokens': 12, 'generated_tokens': 30, 'context_used': 42, 'context_limit': 2048})
        continue
    if model == 'render-stress':
        emit('delta', request_id=request['request_id'], phase='thinking', text='Internal reasoning')
        emit('delta', request_id=request['request_id'], phase='answer', text='# Result\n\n```html\n')
        for _ in range(40):
            emit('delta', request_id=request['request_id'], phase='answer', text='<div>é 👋</div>\n' * 500)
            time.sleep(0.005)
        emit('delta', request_id=request['request_id'], phase='answer', text='```\n\nFinished: 73.\n')
    else:
        answer = json.dumps(request['messages'], ensure_ascii=False) if model == 'file-import' else (
            json.dumps({'mtp': request.get('mtp'), 'mtp_depth': request.get('mtp_depth')}) if model.startswith('mtp-') else 'hello')
        emit('delta', request_id=request['request_id'], phase='answer', text=answer)
    if model == 'crash':
        sys.exit(1)
    if model == 'auto-idle-silent':
        for _ in range(2000):
            if os.path.exists(os.path.join(os.environ['MLXL3_HOME'], 'release-idle-generation')):
                break
            time.sleep(0.02)
        else:
            emit('error', request_id=request['request_id'], message='fixture generation release timed out')
            continue
    else:
        time.sleep(0.2)
    emit('complete', request_id=request['request_id'], assistant_context='done', cache_context='done')
