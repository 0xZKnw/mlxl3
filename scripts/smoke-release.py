"""Two real text turns through the same NDJSON protocol as Desktop.

Usage: python scripts/smoke-release.py /path/to/runtime/mlxl3 MODEL
No network, no writes to chat history, no user model download.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time

process = subprocess.Popen([sys.argv[1], 'bridge', sys.argv[2], '--context-length', '4096'],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                           env={**os.environ, 'MLXL3_WARM_MODEL_ON_LOAD': '0'})
events = queue.Queue()
def read():
    for line in process.stdout:
        events.put(json.loads(line))
    events.put({'type': 'error', 'message': 'engine exited'})
threading.Thread(target=read, daemon=True).start()
def until(kind):
    deadline = time.monotonic() + 120
    while True:
        event = events.get(timeout=max(0.1, deadline - time.monotonic()))
        if event['type'] == 'error':
            raise RuntimeError(event)
        if event['type'] == kind:
            return event
        if time.monotonic() > deadline:
            raise TimeoutError('engine did not finish')
def send(value):
    process.stdin.write(json.dumps(value) + '\n')
    process.stdin.flush()
try:
    ready = until('ready')
    assert ready['model'] == sys.argv[2]
    messages = [{'role': 'user', 'content': 'Remember this exact project code: CEDAR-42. Reply briefly.'}]
    for turn in range(2):
        send({'type': 'generate', 'request_id': str(turn), 'conversation_id': 'release-smoke',
              'messages': messages, 'mcp_enabled': False, 'max_tokens': 384,
              'temperature': 0, 'top_k': 0, 'repetition_penalty': 1})
        result = until('complete')
        answer = result.get('assistant_context', '')
        assert answer.strip(), result
        if turn:
            assert 'CEDAR' in answer and '42' in answer, answer
        print(json.dumps({'turn': turn + 1, 'answer': answer, 'stats': result.get('stats')}), flush=True)
        messages += [{'role': 'assistant', 'content': result.get('cache_context') or answer},
                     {'role': 'user', 'content': 'What exact project code did I give you?'}]
    send({'type': 'shutdown'})
    assert process.wait(timeout=10) == 0
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
