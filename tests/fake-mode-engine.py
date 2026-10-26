#!/usr/bin/env python3
"""Disposable process-boundary fixture for Desktop's four modes; never loads MLX."""
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

registry = None
if sys.argv[1:2] == ['--registry']:
    registry = sys.argv[2]
    del sys.argv[1:3]


def emit(kind, **fields):
    print(json.dumps({'type': kind, **fields}), flush=True)


command, model = sys.argv[1:3]
if command == 'serve':
    if model == 'serve-fail':
        print('fixture bind failure', file=sys.stderr, flush=True)
        sys.exit(3)
    if model == 'serve-silent':
        time.sleep(60)
        sys.exit(0)
    port = int(sys.argv[sys.argv.index('--port') + 1])
    api_key = sys.argv[sys.argv.index('--api-key') + 1]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get('Authorization') != f'Bearer {api_key}':
                self.send_response(401)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"object":"list","data":[]}')

        def log_message(self, *_):
            pass

    server = HTTPServer(('127.0.0.1', port), Handler)
    emit('ready', port=port, model=model)
    server.serve_forever()
elif command == 'bridge':
    with open(os.path.join(os.environ['MLXL3_HOME'], 'mode-launches.jsonl'), 'a') as log:
        log.write(json.dumps({'registry': registry, 'model': model, 'pid': os.getpid()}) + '\n')
    emit('ready', model=model, modules=1, resident_gb=0.01, context_limit=2048,
         model_context_limit=2048, workspace_features=True, bridge_protocol=1,
         runtime_commit='fixture', runtime_profile='release', mlx_version='0.32.2')
    for line in sys.stdin:
        request = json.loads(line)
        if request['type'] == 'set_mcp':
            emit('mcp_status', mcp_servers=1 if request['enabled'] else 0,
                 mcp_tools=1 if request['enabled'] else 0, mcp_errors={})
        elif request['type'] == 'generate':
            emit('delta', request_id=request['request_id'], phase='answer', text=json.dumps(request))
            emit('complete', request_id=request['request_id'], stats={
                'ttft_seconds': 0.125, 'prefill_tps': 120, 'decode_tps': 30,
                'prompt_tokens': 12, 'generated_tokens': 6, 'context_used': 18,
                'context_limit': 2048})
else:
    sys.exit(2)
