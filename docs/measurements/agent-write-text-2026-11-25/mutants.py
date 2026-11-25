"""Temporary mutants of AGENT-2026-11-25-WRITE-TEXT: each must be caught by a check.

Run from the worktree root. Every source is restored from its backup, also on error.
"""
import json
import pathlib
import subprocess
import sys

S = pathlib.Path(sys.argv[1])
A = pathlib.Path("apps/MLXL3Studio/Sources/MLXL3Studio")
JS, SB = A / "AgentJavaScript.swift", A / "AgentSandbox.swift"
BINARY = S / "build" / "mutant-check"
COMPILE = ["swiftc", "-warnings-as-errors", "-swift-version", "6", "-parse-as-library",
           "-module-cache-path", str(S / "build" / "modules"), "-O", str(SB), str(JS),
           str(A / "AgentNetwork.swift"), "tests/agent-sandbox-check.swift", "-o", str(BINARY)]

MUTANTS = [
    ("M1 text leaks into the next call", JS,
     "} else if (textGiven) { delete globalThis.text; textGiven = false; }", "}"),
    ("M2 syntax line off by one", JS,
     'const at = typeof error.line === "number" ? error.line - 2 : 0;',
     'const at = typeof error.line === "number" ? error.line - 1 : 0;'),
    ("M3 remedy shown although text was sent", JS,
     'if (request.api.includes("write") && typeof request.text !== "string" && /',
     'if (request.api.includes("write") && /'),
    ("M4 remedy shown without write permission", JS,
     'if (request.api.includes("write") && typeof request.text !== "string" && /',
     'if (typeof request.text !== "string" && /'),
    ("M5 repeat guard ignores text", JS,
     '+ (text.map { "\\u{0}\\($0.utf8.count):\\($0.hashValue)" } ?? "")', ''),
    ("M6 oversized text not refused by the kernel", JS,
     "if let text, text.utf8.count > maximumFileBytes { return textTooLarge(text.utf8.count) }", ""),
    ("M7 text not writable", JS,
     "{ value: request.text, writable: true, configurable: true, enumerable: true }",
     "{ value: request.text, writable: false, configurable: true, enumerable: true }"),
    ("M8 end-of-code case dropped", JS,
     "} else if (at > lines.length) {", "} else if (false) {"),
    ("M9 server drops text", SB, 'if let text { call["text"] = text }', 'if text != nil { call["text"] = NSNull() }'),
    ("M10 server does not refuse oversized text", SB,
     "if let text, text.utf8.count > AgentJavaScript.maximumFileBytes {", "if let text, text.utf8.count < 0 {"),
    ("M11 server repeat guard ignores text", SB,
     "repeats.note(code: code, text: text, outcome: outcome)", "repeats.note(code: code, outcome: outcome)"),
    ("M12 kernel mode drops text", JS,
     'code: code, text: request["text"] as? String, permissions: permissions,',
     "code: code, text: nil, permissions: permissions,"),
]


def run(command, timeout):
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return done.returncode, (done.stdout + done.stderr)
    except subprocess.TimeoutExpired:
        return None, "timeout"


results = []
MUTANTS = [m for m in MUTANTS if not sys.argv[2:] or m[0].split()[0] in sys.argv[2:]]
for name, path, old, new in MUTANTS:
    original = path.read_text()
    entry = {"mutant": name}
    try:
        if original.count(old) != 1:
            entry["state"] = f"not applied ({original.count(old)} matches)"
            results.append(entry)
            continue
        path.write_text(original.replace(old, new))
        code, output = run(COMPILE, 300)
        if code != 0:
            entry["state"] = "compile failed: " + output[-300:]
            results.append(entry)
            continue
        code, output = run([str(BINARY)], 300)
        first = next((line for line in output.splitlines() if "Fatal error" in line), "")
        entry["sandbox"] = "caught: " + first.split("Fatal error: ")[-1][:110] if code != 0 else "SURVIVED"
        if code == 0:
            code, output = run([sys.executable, "tests/check-agent-protocol.py", str(BINARY)], 400)
            last = output.strip().splitlines()[-1][:150] if output.strip() else ""
            entry["protocol"] = "caught: " + last if code != 0 else "SURVIVED"
        entry["state"] = "caught" if "SURVIVED" not in (entry.get("protocol") or entry["sandbox"]) else "SURVIVED"
    finally:
        path.write_text(original)
        results.append(entry) if entry not in results else None
    print(json.dumps(entry, ensure_ascii=False), flush=True)

(S / ("mutants-" + "-".join(sys.argv[2:]) + ".json" if sys.argv[2:] else "mutants.json")).write_text(json.dumps(results, ensure_ascii=False, indent=1))
print("caught", sum(e.get("state") == "caught" for e in results), "of", len(results))
