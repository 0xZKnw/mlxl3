"""Drive a released engine through the Desktop NDJSON protocol with sampled MTP.

Usage: python mtp_sampling_bridge.py ENGINE MODEL HEAD OUTPUT.json
Bounded waits, fails closed, always reaps the engine.
"""
import json
import queue
import subprocess
import sys
import threading
import time

engine, model, head, output = sys.argv[1:5]
process = subprocess.Popen([engine, 'bridge', model, '--context-length', '4096'],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
events = queue.Queue()


def read():
    for line in process.stdout:
        try:
            events.put(json.loads(line))
        except json.JSONDecodeError:
            events.put({'type': 'error', 'message': 'invalid JSON: ' + line[:200]})
    events.put({'type': 'error', 'message': 'engine exited'})


threading.Thread(target=read, daemon=True).start()


def until(kind, seen=None, limit=300):
    deadline = time.monotonic() + limit
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'no {kind} event')
        event = events.get(timeout=remaining)
        if seen is not None:
            seen.append(event)
        if event['type'] == 'error':
            raise RuntimeError(event)
        if event['type'] == kind:
            return event


def send(value):
    process.stdin.write(json.dumps(value) + '\n')
    process.stdin.flush()


report = {'status': 'failed', 'runs': []}
try:
    ready = until('ready')
    report['ready'] = {key: ready.get(key) for key in (
        'runtime_version', 'runtime_commit', 'mtp_supported', 'mtp_sampling_supported', 'mtp_max_depth')}
    assert ready.get('mtp_sampling_supported') is True, report['ready']
    messages = [{'role': 'user', 'content': 'Write a short poem about autumn rain, then name three rhymes you used.'}]
    cases = [
        ('greedy', {'temperature': 0, 'top_k': 1, 'repetition_penalty': 1}, False),
        ('instruct-preset', {'temperature': 0.7, 'top_k': 20, 'top_p': 0.8, 'presence_penalty': 1.5,
                             'repetition_penalty': 1}, True),
    ]
    for index, (name, sampling, expect_sampling) in enumerate(cases):
        seen = []
        send({'type': 'generate', 'request_id': str(index), 'conversation_id': 'mtp-sampling-' + name,
              'messages': messages, 'mcp_enabled': False, 'max_tokens': 768,
              'mtp': True, 'mtp_head_path': head, 'mtp_depth': 3, **sampling})
        complete = until('complete', seen)
        mode = next(e for e in seen if e['type'] == 'generation_mode')
        stats = complete.get('stats') or {}
        answer = complete.get('assistant_context', '')
        raw = complete.get('cache_context') or ''
        run = {'case': name, 'mtp_active': mode.get('mtp_active'), 'mtp_sampling': mode.get('mtp_sampling'),
               'mtp_depth': mode.get('mtp_depth'), 'generated_tokens': stats.get('generated_tokens'),
               'mtp_proposed_tokens': stats.get('mtp_proposed_tokens'),
               'mtp_accepted_tokens': stats.get('mtp_accepted_tokens'),
               'decode_tokens_per_second': stats.get('decode_tokens_per_second'), 'answer_chars': len(answer), 'raw_chars': len(raw),
               'stop_reason': complete.get('stop_reason'), 'token_hash': complete.get('token_hash')}
        report['runs'].append(run)
        assert mode.get('mtp_active') is True and mode.get('mtp_sampling') is expect_sampling, run
        assert raw.strip() and (stats.get('generated_tokens') or 0) > 0, run
        assert (stats.get('mtp_accepted_tokens') or 0) > 0, run
    send({'type': 'shutdown'})
    assert process.wait(timeout=30) == 0
    report['status'] = 'passed'
except Exception as error:  # noqa: BLE001 - recorded in the report
    report['error'] = repr(error)
finally:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
    with open(output, 'w') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
sys.exit(0 if report['status'] == 'passed' else 1)
