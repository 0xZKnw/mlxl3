"""Exercise the real Swift MCP helper, including transport loss and invalid input."""
import json
import socket
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
    def exchange(messages, raw=None, permissions=None, long_calls=False):
        payload = raw if raw is not None else ''.join(json.dumps(item) + '\n' for item in messages)
        arguments = [helper, '--agent-tools', str(root)]
        if permissions is None:
            permissions = {'harness': 'tools'}
        if permissions is not False:
            metadata.write_text(json.dumps({'permissions': permissions}))
            arguments.append(str(metadata))
            if long_calls:
                arguments.append('long-calls')
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
    # Explicitly offline: Internet is on by default since the open sandbox (2026-11-23).
    js_policy = {'read': True, 'write': True, 'run': True, 'commandSeconds': 3, 'harness': 'javascript', 'internet': False}
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
    # The result keeps its start and end (it used to keep the start and say "truncated").
    check(not responses[0]['result']['isError'] and len(big.encode()) < 33100 and 'omitted' in big
          and big.startswith('Result:\nrrr') and big.endswith('rrr'), f'Large result bounded, start and end kept: {big[:80]!r}…{big[-80:]!r}')
    _, responses = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': "write('wide.txt', 'é'.repeat(130000)); return read('wide.txt').length"}})], permissions=js_policy)
    check(responses[0]['result']['content'][0]['text'] == 'Result:\n130000', f"Widest write fits the protocol: {responses[0]['result']['content'][0]['text'][:200]!r}")
    # run() lets a talkative command finish: start and end of its output, real status.
    chatty = 'i=0; while [ $i -lt 2000 ]; do echo line-$i-xxxxxxxxxxxxxxxxxxxx; i=$((i+1)); done; echo FINISHED > run-done.txt; echo tests: 42 passed'
    _, responses = exchange([
        request('tools/call', {'name': 'javascript', 'arguments': {'code': f'const r = run({json.dumps(chatty)}); return [r.status, r.stopped, r.omitted > 0, r.output.endsWith("tests: 42 passed\\n")]'}}, 1),
    ], permissions=js_policy)
    text = responses[0]['result']['content'][0]['text']
    check(text == 'Result:\n[\n  0,\n  false,\n  true,\n  true\n]' and (root / 'run-done.txt').read_text() == 'FINISHED\n', f'JavaScript run() finishes a talkative command: {text!r}')
    (root / 'run-done.txt').unlink()
    _, responses = exchange([request('tools/call', {'name': 'run', 'arguments': {'command': chatty}}, 1)], permissions=dict(js_policy, harness='tools'))
    text = responses[0]['result']['content'][0]['text']
    check(not responses[0]['result']['isError'] and text.endswith('tests: 42 passed\n\nExit status: 0') and 'bytes of output omitted' in text
          and text.startswith('line-0-') and (root / 'run-done.txt').exists(), f'Structured run finishes a talkative command: {text[:80]!r}…{text[-80:]!r}')
    # `state` survives between calls and server processes, beside the permissions file.
    state_file = Path(directory) / 'js-state.json'
    _, first = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': "state.k = 'v'; return 1"}}, 1)], permissions=js_policy)
    _, second = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return state.k'}}, 1)], permissions=js_policy)
    check(first[0]['result']['content'][0]['text'] == 'Result:\n1' and second[0]['result']['content'][0]['text'] == 'Result:\nv'
          and json.loads(state_file.read_text()) == {'k': 'v'} and not (root / 'js-state.json').exists(),
          f"state kept between calls, outside the workspace: {second[0]['result']['content'][0]['text']!r}")
    _, listed = exchange([request('tools/list')], permissions=dict(js_policy, commandSeconds=17), long_calls=True)
    _, fresh = exchange([request('tools/list')], permissions=False)
    long_description, fresh_description = listed[0]['result']['tools'][0]['description'], fresh[0]['result']['tools'][0]['description']
    check('Limits per call: 20 s' in long_description and '`state` is an object kept between calls' in long_description
          and 'Each call starts fresh.' in fresh_description and 'Limits per call: 10 s' in fresh_description, 'Call wall and state stated in the description')
    _, unchanged = exchange([request('tools/list')], permissions=dict(js_policy, commandSeconds=17))
    check('Limits per call: 10 s' in unchanged[0]['result']['tools'][0]['description'], 'Without long calls the wall stays 10 s')
    # Internet through the host: offered only with the setting, refused before any request.
    online = dict(js_policy, internet=True, internetHosts=['github.com'])
    _, listed = exchange([request('tools/list')], permissions=online)
    _, offline = exchange([request('tools/list')], permissions=js_policy)
    check('fetch(url: string)' in listed[0]['result']['tools'][0]['description'] and 'fetch(' not in offline[0]['result']['tools'][0]['description'],
          'fetch offered only with Internet')
    _, refused = exchange([
        request('tools/call', {'name': 'javascript', 'arguments': {'code': "return fetch('https://example.org/')"}}, 1),
        request('tools/call', {'name': 'javascript', 'arguments': {'code': "return fetch('http://github.com/')"}}, 2),
        request('tools/call', {'name': 'javascript', 'arguments': {'code': "return typeof fetch"}}, 3),
    ], permissions=online)
    _, hidden = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': "return typeof fetch"}}, 1)], permissions=js_policy)
    check(refused[0]['result']['isError'] and 'not in the allowed hosts' in refused[0]['result']['content'][0]['text']
          and refused[1]['result']['isError'] and 'only https' in refused[1]['result']['content'][0]['text']
          and refused[2]['result']['content'][0]['text'] == 'Result:\nfunction'
          and hidden[0]['result']['content'][0]['text'] == 'Result:\nundefined', 'Internet refusals and visibility through MCP')
    check('fetch(url: string)' in fresh_description and 'allowed hosts (*)' in fresh_description
          and 'with Internet (curl, git clone, pip, npm)' in fresh_description, 'Internet on by default for a new task')
    # Commands get the network with Internet only, through both harnesses' production paths
    # (supervisor `net:` argument, `--agent-command ... tail+net`): a local listener, no Internet.
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(8)
    port = listener.getsockname()[1]
    probe = ("/usr/bin/perl -MIO::Socket::INET -e 'IO::Socket::INET->new(PeerAddr => \"127.0.0.1:%d\", Timeout => 3)"
             " or die \"$!\\n\"; print \"CONNECTED\\n\"' 2>&1" % port)
    try:
        _, js_net = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return run(%s).output' % json.dumps(probe)}}, 1)],
                             permissions=online)
        _, js_off = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return run(%s).output' % json.dumps(probe)}}, 1)],
                             permissions=js_policy)
        _, tools_net = exchange([request('tools/call', {'name': 'run', 'arguments': {'command': probe}}, 1)], permissions={'harness': 'tools'})
        _, tools_off = exchange([request('tools/call', {'name': 'run', 'arguments': {'command': probe}}, 1)],
                                permissions={'harness': 'tools', 'internet': False})
    finally:
        listener.close()
    texts = [r[0]['result']['content'][0]['text'] for r in (js_net, js_off, tools_net, tools_off)]
    check('CONNECTED' in texts[0] and 'CONNECTED' in texts[2]
          and all('Operation not permitted' in t and 'CONNECTED' not in t for t in (texts[1], texts[3])),
          f'Command network follows the Internet setting: {texts!r}')
    if os.environ.get('MLXL3_NETWORK_TESTS') == '1':
        # The production path: MCP server, supervisor arguments, curl, a real GitHub file.
        _, live = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code':
            "const r = fetch('https://raw.githubusercontent.com/octocat/Hello-World/master/README'); return r.status + ' ' + r.text.trim()"}}, 1)],
            permissions=dict(online, internetHosts=['raw.githubusercontent.com']))
        check(live[0]['result']['content'][0]['text'] == 'Result:\n200 Hello World!', f"Live fetch through MCP: {live[0]['result']['content'][0]['text']!r}")
    else:
        print('Live MCP network check not run (set MLXL3_NETWORK_TESTS=1)')
    # Loop note through the real server: a third identical program with the same result.
    _, repeated = exchange([request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return 1 + 1'}}, n) for n in (1, 2, 3)]
                           + [request('tools/call', {'name': 'javascript', 'arguments': {'code': 'return 2 + 2'}}, 4)], permissions=js_policy)
    notes = ['has now run 3 times with the same result' in r['result']['content'][0]['text'] for r in repeated]
    check(notes == [False, False, True, False] and repeated[0]['result']['content'][0]['text'] == 'Result:\n2',
          f"Repeated program noted on the third call only: {[r['result']['content'][0]['text'] for r in repeated]!r}")
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
