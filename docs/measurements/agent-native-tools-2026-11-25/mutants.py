"""Temporary mutants of the native Agent tools (AGENT-2026-11-25-NATIVE-TOOLS).

Each mutant removes or inverts one guard, rebuilds the check binary (or the app for the
timeline check), runs the suite that should notice, and restores the source. A mutant is
detected when its suite fails; a suite that still passes is a surviving mutant. The sources are
compared with their hashes from before the run at the end.

usage: python3 mutants.py <output.json> [mutant-id ...]
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SOURCES = REPO / 'apps/MLXL3Studio/Sources/MLXL3Studio'
JS, TOOLS, SANDBOX, SHOWN = (SOURCES / name for name in (
    'AgentJavaScript.swift', 'AgentTools.swift', 'AgentSandbox.swift', 'ToolCallPresentation.swift'))

# id, file, text to replace (exactly once), replacement, suite, what it removes
MUTANTS = [
    ('M01', TOOLS, 'guard names(permissions).contains(name) else {\n      throw AgentSandbox.Refusal(text: "Error: \\(name) is not enabled in this task.")\n    }',
     '', 'swift', 'argument check accepts a tool the permissions do not offer'),
    ('M02', SANDBOX, 'guard AgentTools.names(permissions).contains(name) else {', 'guard true || AgentTools.names(permissions).contains(name) else {',
     'protocol', 'server no longer refuses names outside the catalogue with the list of tools'),
    ('M03', SANDBOX, 'case "tools/call" where permissions.harness == .native:', 'case "tools/call" where permissions.harness != .tools:',
     'protocol', 'native calls served under the JavaScript harness'),
    ('M04', TOOLS, 'return AgentSandbox.validPath(path) ? path : nil', 'return path', 'swift', 'paths outside the project accepted'),
    ('M05', JS, 'include.utf8.count <= 256, !include.contains("/"), !include.hasPrefix("!"),', 'include.utf8.count <= 256, !include.hasPrefix("!"),',
     'swift', 'include filter accepts a folder'),
    ('M06', JS, 'nativeMode = typeof request.tool === "string";\n        if (nativeMode) {', 'nativeMode = false;\n        if (typeof request.tool === "string") {',
     'swift', 'tool errors keep the JavaScript wording'),
    ('M07', JS, 'if (a.old_string === a.new_string) {', 'if (false) {', 'swift', 'no-op edit accepted'),
    ('M08', JS, 'let maximumText = plain ? maximumText + 2048 : maximumText', 'let maximumText = plain ? maximumText : maximumText',
     'swift', 'no room for the line naming the spilled output'),
    ('M09', JS, 'let result = outcome(finished, warnings: warnings, plain: tool != nil)', 'let result = outcome(finished, warnings: warnings, plain: false)',
     'swift', 'native results labelled like program results'),
    ('M10', JS, 'filters = globs.map { " --include=" + quote($0) }.joined()', 'filters = ""', 'swift', 'include filter ignored'),
    ('M11', SANDBOX, 'case harness = "toolHarness"', 'case harness', 'swift', 'the earlier harness key is read again'),
    ('M12', SANDBOX, 'var harness = AgentHarness.native', 'var harness = AgentHarness.javascript', 'swift', 'JavaScript is the default again'),
    ('M13', TOOLS, '    case let value?:\n      guard JSONSerialization.isValidJSONObject([value]),', '    case let value?:\n      guard false, JSONSerialization.isValidJSONObject([value]),',
     'swift', 'an argument the engine decoded as JSON is refused'),
    ('M14', JS, 'let finishedJobs = jobs.notices(native: tool != nil)', 'let finishedJobs = jobs.notices(native: false)', 'swift', 'finished job announced with wait()'),
    ('M15', SHOWN, '            if let pattern = arguments.object?["pattern"] as? String, !Self.oneLine(pattern).isEmpty {', '            if false, let pattern = arguments.object?["pattern"] as? String {',
     'timeline', 'a search shows its folder instead of its pattern'),
    ('M16', JS, 'plural(a.content === "" ? 0 : splitLines(a.content).length, "line")', 'plural(splitLines(a.content).length, "line")', 'swift', 'an empty file reported as one line'),
    ('M17', JS, '\\(call ? "call has now been made" : "program has now run")', '\\("program has now run")', 'swift', 'repeat reminder speaks of a program'),
    ('M18', JS, '.replace(": view(path, offset) or search(regex, path) ...]", ": read it with offset and limit, or grep it ...]");', ';',
     'swift', 'spilled output named with JavaScript functions'),
    ('M19', TOOLS, '      guard value.utf8.count <= limit else {\n        throw refuse("`\\(key)` is \\(value.utf8.count) bytes; the limit is \\(limit)")\n      }', '',
     'swift', 'no size limit on an argument'),
    ('M20', SHOWN, 'if last.hasSuffix("]"), let mark = last.range(of: "[exit code: ", options: .backwards),', 'if false, let mark = last.range(of: "[exit code: ", options: .backwards),',
     'timeline', 'native exit code not shown in the timeline'),
    ('M21', TOOLS, 'if permissions.read && permissions.write { names.append("edit") }', 'if permissions.write { names.append("edit") }', 'swift', 'edit offered without read'),
    ('M22', SANDBOX, 'let note = repeats.note(code: AgentTools.signature(name, arguments), outcome: outcome, call: true)', 'let note: String? = nil',
     'protocol', 'no repeat reminder for native calls'),
]


def digest(paths):
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run(command, timeout, **options):
    try:
        done = subprocess.run(command, cwd=REPO, capture_output=True, text=True, timeout=timeout, check=False, **options)
        return done.returncode, (done.stdout + done.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return None, 'timeout'


def main():
    output = Path(sys.argv[1])
    wanted = set(sys.argv[2:])
    files = [JS, TOOLS, SANDBOX, SHOWN]
    before = digest(files)
    environment = dict(os.environ, DEVELOPER_DIR='/Applications/Xcode.app/Contents/Developer')
    results = []
    with tempfile.TemporaryDirectory(prefix='mlxl3-mutants-') as directory:
        binary = str(Path(directory) / 'agent-check')
        compile_check = ['swiftc', '-swift-version', '6', '-warnings-as-errors', '-parse-as-library', '-module-cache-path',
                         str(Path(directory) / 'modules'), str(SANDBOX), str(JS), str(SOURCES / 'AgentNetwork.swift'), str(TOOLS),
                         'tests/agent-sandbox-check.swift', '-o', binary]
        for identifier, path, old, new, suite, meaning in MUTANTS:
            if wanted and identifier not in wanted:
                continue
            original = path.read_text(encoding='utf-8')
            row = {'id': identifier, 'file': path.name, 'suite': suite, 'removes': meaning}
            if original.count(old) != 1:
                row.update(status='not applied', detail=f'{original.count(old)} occurrences')
                results.append(row)
                continue
            try:
                path.write_text(original.replace(old, new), encoding='utf-8')
                if suite == 'timeline':
                    code, tail = run(['swift', 'build', '--package-path', 'apps/MLXL3Studio'], 900, env=environment)
                    if code == 0:
                        where, _ = subprocess.Popen(['swift', 'build', '--package-path', 'apps/MLXL3Studio', '--show-bin-path'], cwd=REPO,
                                                    stdout=subprocess.PIPE, text=True, env=environment).communicate(timeout=120)
                        code, tail = run([where.strip() + '/MLXL3Studio', '--check-chat-timeline'], 300)
                        row['built'] = True
                    else:
                        row['built'] = False
                else:
                    code, tail = run(compile_check, 900)
                    row['built'] = code == 0
                    if code == 0:
                        command = [binary] if suite == 'swift' else [sys.executable, 'tests/check-agent-protocol.py', binary]
                        code, tail = run(command, 900)
                # A mutant that does not build proves nothing: it is neither detected nor surviving.
                row['status'] = 'does not build' if not row['built'] else 'detected' if code not in (0,) else 'SURVIVED'
                row['exit'] = code
                row['tail'] = tail[-300:]
            finally:
                path.write_text(original, encoding='utf-8')
            results.append(row)
            print(row['id'], row['status'], row.get('tail', '').strip().splitlines()[-1][:160] if row.get('tail', '').strip() else '', flush=True)
    after = digest(files)
    report = {'sources_restored': before == after, 'sources': after, 'mutants': results,
              'detected': sum(row['status'] == 'detected' for row in results), 'total': len(results)}
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f"{report['detected']}/{report['total']} detected; sources restored: {report['sources_restored']}")
    return 0 if before == after else 2


if __name__ == '__main__':
    raise SystemExit(main())
