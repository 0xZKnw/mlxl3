"""Exercise the real Swift MCP helper, including transport loss and invalid input."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

helper = str(Path(sys.argv[1]).resolve())
checks = 0


def check(value, message):
    global checks
    assert value, message
    checks += 1


def request(method, params=None, identifier=1):
    return {'jsonrpc': '2.0', 'id': identifier, 'method': method, 'params': params or {}}


with tempfile.TemporaryDirectory(prefix='mlxl3-agent-protocol-') as directory:
    metadata = Path(directory) / "permissions.json"
    root = Path(directory) / "files"
    root.mkdir()

    def exchange(messages, raw=None, permissions=None):
        payload = raw if raw is not None else ''.join(json.dumps(item) + '\n' for item in messages)
        arguments = [helper, '--agent-tools', str(root)]
        if permissions is not None:
            metadata.write_text(json.dumps({'permissions': permissions}))
            arguments.append(str(metadata))
        result = subprocess.run(arguments, input=payload, text=True,
                                capture_output=True, timeout=12, check=False)
        return result, [json.loads(line) for line in result.stdout.splitlines()]

    result, responses = exchange([
        request('initialize', identifier=4),
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        request('tools/list', identifier=5),
        request('tools/call', {'name': 'run', 'arguments': {'command': 'printf protocol-ok > result.txt; cat result.txt'}}, 6),
    ])
    check(result.returncode == 0 and [item['id'] for item in responses] == [4, 5, 6], 'Handshake IDs/notifications')
    check(responses[0]['result']['capabilities'] == {'tools': {}}, 'Tools capability')
    check([tool['name'] for tool in responses[1]['result']['tools']] == ['run', 'list_files', 'read_file', 'write_file'], 'Only built-in sandbox tools exposed')
    check(responses[2]['result']['isError'] is False and 'protocol-ok' in responses[2]['result']['content'][0]['text'], 'Real command through MCP')
    check((root / 'result.txt').read_text() == 'protocol-ok', 'Production tool writes actual file')
    for params in [
        {'name': 'unknown', 'arguments': {'command': 'true'}},
        {'name': 'run', 'arguments': {}},
        {'name': 'run', 'arguments': {'command': ''}},
        {'name': 'run', 'arguments': {'command': 'x' * 16385}},
        {'name': 'run', 'arguments': {'command': 'echo\0bad'}},
    ]:
        result, responses = exchange([request('tools/call', params)])
        check(result.returncode == 0 and responses[0]['result']['isError'], 'Invalid tool request must be an error')
    _, responses = exchange([request('unrecognized')])
    check(responses[0]['error']['code'] == -32601, 'Unknown method')
    for payload in ['{bad}\n', '{}\n', '{"method":', 'x' * 65537]:
        result, _ = exchange([], raw=payload)
        check(result.returncode != 0, 'Malformed/truncated/oversized input must fail closed')

    _, responses = exchange([
        request('tools/call', {'name': 'write_file', 'arguments': {'path': "nested/quoted'file.txt", 'content': "text \u0024(never-run) ' safely"}}, 10),
        request('tools/call', {'name': 'read_file', 'arguments': {'path': "nested/quoted'file.txt"}}, 11),
        request('tools/call', {'name': 'list_files'}, 12),
    ])
    check(not responses[0]['result']['isError'] and (root / "nested/quoted'file.txt").read_text() == "text $(never-run) ' safely", 'Structured write preserves literal content')
    check("text $(never-run) ' safely" in responses[1]['result']['content'][0]['text'], 'Structured read')
    check('nested/' in responses[2]['result']['content'][0]['text'], 'Structured file listing')
    for path in ('../outside.txt', '/tmp/outside.txt', '.git/config', 'a/../outside', 'bad\nname'):
        _, responses = exchange([request('tools/call', {'name': 'write_file', 'arguments': {'path': path, 'content': 'no'}})])
        check(responses[0]['result']['isError'], 'Unsafe structured path refused')
    policy = {'read': True, 'write': False, 'run': True, 'commandSeconds': 2}
    _, responses = exchange([
        request('tools/list'),
        request('tools/call', {'name': 'write_file', 'arguments': {'path': 'refused.txt', 'content': 'BAD'}}, 2),
        request('tools/call', {'name': 'run', 'arguments': {'command': 'printf BAD > refused.txt'}}, 3),
        request('tools/call', {'name': 'run', 'arguments': {'command': "printf BAD > ../permissions.json"}}, 4),
    ], permissions=policy)
    check('write_file' not in [tool['name'] for tool in responses[0]['result']['tools']], 'Disabled write tool not advertised')
    check(all(row['result']['isError'] for row in responses[1:]) and not (root / 'refused.txt').exists(), 'Write permission cannot be bypassed through run')
    check(json.loads(metadata.read_text())['permissions'] == policy, 'Agent cannot modify its permissions')
    policy.update(run=False, read=False)
    _, responses = exchange([request('tools/list'), request('tools/call', {'name': 'run', 'arguments': {'command': 'true'}}, 2)], permissions=policy)
    check(responses[0]['result']['tools'] == [] and responses[1]['result']['isError'], 'Disabled tools rejected by server, not only hidden')
    # The trusted command supervisor must clean up even if its MCP parent is killed.
    transport = subprocess.Popen([helper, '--agent-tools', str(root)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = None
    try:
        transport.stdin.write(json.dumps(request('tools/call', {
            'name': 'run', 'arguments': {'command': 'echo $$ > running.pid; sleep 30'}})) + '\n')
        transport.stdin.flush()
        deadline = time.monotonic() + 5
        while not (root / 'running.pid').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        check((root / 'running.pid').exists(), 'Command started before parent-death test')
        child = int((root / 'running.pid').read_text())
        transport.kill()
        transport.wait(timeout=2)
        deadline = time.monotonic() + 2
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.02)
        check(not alive, 'Sandbox command survived MCP parent death')
    finally:
        if transport.poll() is None:
            transport.kill()
        transport.communicate(timeout=3)
        if child is not None:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass

print(f'Agent protocol checks passed: {checks}, real MCP commands, errors, bounded input and parent-death cleanup')
