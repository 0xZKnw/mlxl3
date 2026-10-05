#!/usr/bin/env python3
"""Isolated engine lifecycle fixture, never imports MLX or user configuration."""
import json
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
    elif command == 'details':
        print(json.dumps({'id': query, 'revision': 'main', 'commit': 'fixture',
                          'branches': ['main'], 'variants': [
                              {'id': '.', 'label': '2.49 bpw', 'size_bytes': 128}],
                          'readme': '# Fixture', 'gated': False, 'downloads': 73, 'likes': 2}), flush=True)
    else:
        sys.exit(2)
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
    print(json.dumps({'type': 'installed', 'path': '/tmp/mlxl3-fixture-mtp'}), flush=True)
    sys.exit(0)

model = sys.argv[2]
signal.signal(signal.SIGUSR1, lambda *_: None)
def emit(kind, **values):
    print(json.dumps({'type': kind, **values}), flush=True)
emit('ready', model=model, modules=1, resident_gb=0.01, context_limit=2048, model_context_limit=2048,
     dflash_supported=model == 'renamed', mtp_supported=model == 'renamed', mtp_auto_download_supported=model == 'renamed',
     bridge_protocol=1, runtime_commit='fixture', runtime_profile='release', mlx_version='0.32.2')
for line in sys.stdin:
    request = json.loads(line)
    if request['type'] != 'generate':
        continue
    if model == 'render-stress':
        emit('delta', request_id=request['request_id'], phase='thinking', text='Internal reasoning')
        emit('delta', request_id=request['request_id'], phase='answer', text='# Result\n\n```html\n')
        for _ in range(40):
            emit('delta', request_id=request['request_id'], phase='answer', text='<div>é 👋</div>\n' * 500)
            time.sleep(0.005)
        emit('delta', request_id=request['request_id'], phase='answer', text='```\n\nFinished: 73.\n')
    else:
        answer = json.dumps(request['messages'], ensure_ascii=False) if model == 'file-import' else 'hello'
        emit('delta', request_id=request['request_id'], phase='answer', text=answer)
    if model == 'crash':
        sys.exit(1)
    time.sleep(0.2)
    emit('complete', request_id=request['request_id'], assistant_context='done', cache_context='done')
