#!/usr/bin/env python3
"""Isolated engine lifecycle fixture, never imports MLX or user configuration."""
import json
import signal
import sys
import time

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
