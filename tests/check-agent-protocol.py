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

    # The structured-tools checks name their harness; `permissions=False` omits the file.
    def exchange(messages, raw=None, permissions=None):
        payload = raw if raw is not None else ''.join(json.dumps(item) + '\n' for item in messages)
        arguments = [helper, '--agent-tools', str(root)]
        if permissions is None:
            permissions = {'harness': 'tools'}
        if permissions is not False:
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
    policy = {'read': True, 'write': False, 'run': True, 'commandSeconds': 2, 'harness': 'tools'}
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
    # Without saved permissions (and for project.json files older than the harness field) the
    # helper uses the default harness.
    _, responses = exchange([request('tools/list')], permissions=False)
    check([tool['name'] for tool in responses[0]['result']['tools']] == ['javascript'], 'Default harness is JavaScript')
    _, responses = exchange([request('tools/list')], permissions={'read': True, 'write': True, 'run': True, 'commandSeconds': 5})
    check([tool['name'] for tool in responses[0]['result']['tools']] == ['javascript'], 'Legacy permissions use the default harness')

    # Single-tool JavaScript harness through the real MCP helper.
    js_policy = {'read': True, 'write': True, 'run': True, 'commandSeconds': 3, 'harness': 'javascript'}
    _, responses = exchange([request('tools/list')], permissions=js_policy)
    tools = responses[0]['result']['tools']
    check([tool['name'] for tool in tools] == ['javascript'], 'JavaScript harness exposes one tool')
    check(tools[0]['inputSchema']['required'] == ['code'] and 'edit(path' in tools[0]['description'], 'JavaScript tool schema and API')
    code = "write('js/é.txt', 'naïve 👩\u200d💻'); console.log(list().join(',')); return read('js/é.txt') + ' ' + run('cat js/é.txt').status"
    _, responses = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': code}})], permissions=js_policy)
    text = responses[0]['result']['content'][0]['text']
    check(responses[0]['result']['isError'] is False and 'naïve 👩\u200d💻 0' in text and 'js/é.txt' in text, f'JavaScript call through MCP: {text!r}')
    check((root / 'js/é.txt').read_text() == 'naïve 👩\u200d💻', 'JavaScript write reached the workspace')
    # The engine decodes XML parameters that look like JSON: a number is still the model's code.
    _, responses = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': 42}})], permissions=js_policy)
    check(responses[0]['result']['isError'] is False and 'without output' in responses[0]['result']['content'][0]['text'], 'Decoded JSON code runs as text')
    for params in [
        {'name': 'run', 'arguments': {'command': 'printf BAD > crossed.txt'}},
        {'name': 'write_file', 'arguments': {'path': 'crossed.txt', 'content': 'BAD'}},
        {'name': 'javascript', 'arguments': {}},
        {'name': 'javascript', 'arguments': {'code': ''}},
        {'name': 'javascript', 'arguments': {'code': 'x' * 32769}},
        {'name': 'javascript', 'arguments': {'code': None}},
    ]:
        result, responses = exchange([request('tools/call', params)], permissions=js_policy)
        check(result.returncode == 0 and responses[0]['result']['isError'], f'Refused under the JavaScript harness: {params["name"]}')
    check(not (root / 'crossed.txt').exists(), 'Structured tools cannot run under the JavaScript harness')
    tools_policy = dict(js_policy, harness='tools')
    _, responses = exchange([request('tools/list'), request('tools/call', {'name': 'javascript', 'arguments': {'code': "write('crossed.txt', 'BAD')"}}, 2)], permissions=tools_policy)
    check('javascript' not in [tool['name'] for tool in responses[0]['result']['tools']] and responses[1]['result']['isError'] and not (root / 'crossed.txt').exists(), 'JavaScript refused under the structured harness')
    _, responses = exchange([request('tools/list')], permissions=dict(js_policy, read=False, write=False, run=False))
    check(responses[0]['result']['tools'] == [], 'No JavaScript tool without any permission')
    readonly = dict(js_policy, write=False, run=False)
    _, responses = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': "return typeof write + typeof run"}})], permissions=readonly)
    check(responses[0]['result']['content'][0]['text'] == 'Result:\nundefinedundefined', 'Read-only JavaScript API')
    for code in ['é' * 16000, '// x\n' * 30000]:
        result, responses = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': code}}, 7),
                                      request('tools/list', identifier=8)], permissions=js_policy)
        check(result.returncode == 0 and [row['id'] for row in responses] == [7, 8] and responses[0]['result']['isError'],
              f'Oversized code refused without ending the server ({len(code.encode())} bytes)')
    _, responses = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': "return 'r'.repeat(100000)"}})], permissions=js_policy)
    big = responses[0]['result']['content'][0]['text']
    check(not responses[0]['result']['isError'] and len(big.encode()) < 33100 and 'truncated' in big, 'Large result bounded')
    _, responses = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': "write('wide.txt', 'é'.repeat(130000)); return read('wide.txt').length"}})], permissions=js_policy)
    check(responses[0]['result']['content'][0]['text'] == 'Result:\n130000', f"Widest write fits the protocol: {responses[0]['result']['content'][0]['text'][:200]!r}")
    # search(): a grep error is a tool error (it used to be an empty list), CRLF files match.
    (root / 'win.txt').write_bytes(b'alpha one\r\nbeta two\r\ngamma one\r\n')
    _, responses = exchange([
        request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return search("fetchUser(")'}}, 1),
        request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return search("one", "missing-dir")'}}, 2),
        request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return search("one", "win.txt").map((m) => m.line + ":" + m.text).join("|")'}}, 3),
    ], permissions=readonly)
    invalid, missing, crlf = (row['result'] for row in responses)
    check(invalid['isError'] and 'parentheses not balanced' in invalid['content'][0]['text'], f"Invalid regex is an error through MCP: {invalid['content'][0]['text']!r}")
    check(missing['isError'] and 'No such file' in missing['content'][0]['text'], f"Missing search path is an error through MCP: {missing['content'][0]['text']!r}")
    check(not crlf['isError'] and crlf['content'][0]['text'] == 'Result:\n1:alpha one|3:gamma one', f"CRLF matches through MCP: {crlf['content'][0]['text']!r}")

    # The JavaScript supervisor and its command die with the MCP parent.
    metadata.write_text(json.dumps({'permissions': js_policy}))
    transport = subprocess.Popen([helper, '--agent-tools', str(root), str(metadata)], stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    child = None
    try:
        transport.stdin.write(json.dumps(request('tools/call', {
            'name': 'javascript', 'arguments': {'code': "run('echo $$ > js-running.pid; sleep 30')"}})) + '\n')
        transport.stdin.flush()
        deadline = time.monotonic() + 5
        while not (root / 'js-running.pid').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        check((root / 'js-running.pid').exists(), 'JavaScript command started before parent-death test')
        child = int((root / 'js-running.pid').read_text())
        transport.kill()
        transport.wait(timeout=2)
        deadline = time.monotonic() + 3
        alive = True
        while time.monotonic() < deadline:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                alive = False
                break
            time.sleep(0.02)
        check(not alive, 'JavaScript command survived MCP parent death')
        lingering = subprocess.run(['/usr/bin/pgrep', '-f', 'JavaScriptCore.framework.*useJIT=false'], capture_output=True, text=True)
        check(lingering.stdout.strip() == '' or all(int(pid) != transport.pid for pid in lingering.stdout.split()), 'Interpreter reaped')
    finally:
        if transport.poll() is None:
            transport.kill()
        transport.communicate(timeout=3)
        if child is not None:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass

    # The trusted command supervisor must clean up even if its MCP parent is killed.
    metadata.write_text(json.dumps({'permissions': {'harness': 'tools'}}))
    transport = subprocess.Popen([helper, '--agent-tools', str(root), str(metadata)], stdin=subprocess.PIPE,
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

print(f'Agent protocol checks passed: {checks}, real MCP commands, errors, bounded input and parent-death cleanup; JavaScript harness tools, cross-harness refusal and cleanup')
