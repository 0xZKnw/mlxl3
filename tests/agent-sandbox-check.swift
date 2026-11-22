import Darwin
import Foundation

@main
struct AgentSandboxCheck {
  /// The single-tool harness, through the production supervisor path on the real sandbox.
  static func checkJavaScript(temporary: URL, secret: URL, count: inout Int) throws {
    func check(_ condition: Bool, _ message: String) {
      guard condition else { fatalError(message) }
      count += 1
    }
    let root = temporary.appendingPathComponent("js project")
    try FileManager.default.createDirectory(
      at: root.appendingPathComponent("src"), withIntermediateDirectories: true)
    try Data("def add(a, b):\n    return a - b\n".utf8).write(to: root.appendingPathComponent("src/app.py"))
    try Data("one\r\ntwo\r\n".utf8).write(to: root.appendingPathComponent("crlf.txt"))
    try Data([0x61, 0xff, 0x62]).write(to: root.appendingPathComponent("latin.bin"))
    var all = AgentPermissions()
    all.harness = .javascript
    func js(_ code: String, _ permissions: AgentPermissions? = nil) throws -> AgentJavaScript.Outcome {
      try AgentJavaScript.execute(root: root, code: code, permissions: permissions ?? all)
    }

    // Permissions decide the API; old project.json files still decode (no harness key).
    check(AgentJavaScript.functions(all) == ["list", "read", "search", "write", "edit", "run"], "full API")
    var readOnly = all
    readOnly.write = false
    readOnly.run = false
    check(AgentJavaScript.functions(readOnly) == ["list", "read", "search"], "read-only API")
    var nothing = readOnly
    nothing.read = false
    check(AgentJavaScript.functions(nothing).isEmpty, "no API without permissions")
    let legacy = try JSONDecoder().decode(
      AgentPermissions.self, from: Data(#"{"read":false,"write":true,"run":false,"commandSeconds":7}"#.utf8))
    check(
      !legacy.read && legacy.write && !legacy.run && legacy.commandSeconds == 7
        && legacy.harness == AgentPermissions().harness, "legacy permissions decode with defaults")
    let empty = try JSONDecoder().decode(AgentPermissions.self, from: Data("{}".utf8))
    check(empty == AgentPermissions(), "empty permissions decode to defaults")
    let tools = try JSONDecoder().decode(AgentPermissions.self, from: Data(#"{"harness":"tools"}"#.utf8))
    check(tools.harness == .tools, "explicit harness decodes")
    do {
      _ = try JSONDecoder().decode(AgentPermissions.self, from: Data(#"{"harness":"python"}"#.utf8))
      fatalError("unknown harness accepted")
    } catch { count += 1 }
    var classic = AgentPermissions()
    classic.harness = .tools
    check(
      AgentSandbox.instructions(for: classic)
        == "You are a local coding agent. Use list_files and read_file to explore the project, write_file to edit it, and run to execute commands and verify your changes. Only enabled tools are available. You work on an isolated copy; originals are changed only after user review. Commands use macOS system tools, have no network or personal-file access, and have a configurable deadline (at most 30 seconds) and 32 KiB output limit. Use relative paths. Never claim to have executed a command unless its tool result confirms it. Explain failures and limits honestly.",
      "structured-tools prompt changed")
    check(AgentSandbox.instructions(for: all).contains("`javascript`"), "JavaScript prompt")
    check(
      AgentJavaScript.toolDescription(readOnly).contains("read(path: string)")
        && !AgentJavaScript.toolDescription(readOnly).contains("write(path")
        && !AgentJavaScript.toolDescription(readOnly).contains("run(command"),
      "description lists only enabled functions")
    var slow = all
    slow.commandSeconds = 30
    check(AgentJavaScript.toolDescription(slow).contains("at most 10 s"), "run limit shown as the call limit")

    // One call explores, edits and verifies; values come back as JSON.
    let batch = try js("""
      const files = list();
      const before = read("src/app.py");
      edit("src/app.py", "a - b", "a + b");
      const hits = search("return a [+]");
      const test = run("grep -c 'a + b' src/app.py");
      console.log("files", files.length);
      return { files, changed: before !== read("src/app.py"), hits, test };
      """)
    check(!batch.isError && batch.text.hasPrefix("files 3\nResult:\n"), "batched call: \(batch.text)")
    check(
      batch.text.contains(#""changed": true"#) && batch.text.contains(#""path": "src/app.py""#)
        && batch.text.contains(#""line": 2"#) && batch.text.contains(#""output": "1\n""#)
        && batch.text.contains(#""status": 0"#),
      "batched results: \(batch.text)")
    let edited = try Data(contentsOf: root.appendingPathComponent("src/app.py"))
    check(String(decoding: edited, as: UTF8.self) == "def add(a, b):\n    return a + b\n", "edit wrote the file")
    // Exact text round trip: CRLF, quotes, $(…), Unicode; quoting needs several printf chunks.
    let tricky = "it's $(never) `run` \\ \"q\" é 👩‍💻\r\n" + String(repeating: "'x'", count: 3000)
    let written = try js("write('deep/a b/t.txt', \(jsonString(tricky))); return read('deep/a b/t.txt') === \(jsonString(tricky))")
    check(written.text == "Result:\ntrue", "exact write/read round trip: \(written.text)")
    let onDisk = try Data(contentsOf: root.appendingPathComponent("deep/a b/t.txt"))
    check(onDisk == Data(tricky.utf8), "bytes on disk")
    check(try js("return read('crlf.txt')").text == "Result:\none\r\ntwo\r\n", "CRLF preserved")
    // Raw UTF-8 code and results: jsc's readline is Latin-1, so the protocol must stay ASCII.
    let unicode = try js("console.log('café 👩‍💻'); write('u.txt', 'é'); return ['café 👩‍💻'.length, read('u.txt') === 'é']")
    check(unicode.text == "café 👩‍💻\nResult:\n[\n  10,\n  true\n]", "Unicode code and results: \(unicode.text)")
    let leftovers = try FileManager.default.subpathsOfDirectory(atPath: root.path).filter {
      $0.contains(".mlxl3-write-")
    }
    check(leftovers.isEmpty, "temporary write files left: \(leftovers)")
    let binary = try js("edit('latin.bin', 'a', 'c')")
    let binaryBytes = try Data(contentsOf: root.appendingPathComponent("latin.bin"))
    check(
      binary.isError && binary.text.contains("not UTF-8") && binaryBytes == Data([0x61, 0xff, 0x62]),
      "non-UTF-8 file must not be rewritten: \(binary.text)")

    // Errors the model can act on.
    let syntax = try js("let x = ;")
    check(syntax.isError && syntax.text.hasPrefix("Error: SyntaxError"), "syntax error: \(syntax.text)")
    let reference = try js("let a = 1;\nmissing.call();")
    check(reference.isError && reference.text.contains("(line 2)"), "runtime error line: \(reference.text)")
    let partial = try js("write('done.txt', 'kept'); read('absent.txt')")
    check(
      partial.isError && partial.text.contains(#"read("absent.txt") failed"#)
        && partial.text.contains(#"Completed before the error: write("done.txt")"#)
        && FileManager.default.fileExists(atPath: root.appendingPathComponent("done.txt").path),
      "failed call reports earlier effects: \(partial.text)")
    let ambiguous = try js("write('twice.txt', 'x x'); edit('twice.txt', 'x', 'y')")
    check(ambiguous.isError && ambiguous.text.contains("occurs 2 times"), "ambiguous edit")
    check(try js("return 6 * 7").text == "Result:\n42", "return value")
    check(try js("```javascript\nreturn 'fenced'\n```").text == "Result:\nfenced", "fenced code")
    check(try js(#"const v = 5;\nreturn v + 1"#).text == "Result:\n6", "literal \\n escapes")
    check(try js("1 + 1").text.hasPrefix("Done, without output"), "no output hint")
    check(try js("throw 'plain'").text == "Error: uncaught plain", "non-Error throw")
    let hanging = try js("await new Promise(() => {})")
    check(hanging.isError && hanging.text.contains("ended before finishing"), "never-settling await")

    // Isolation: no shell helper survives, the interpreter cannot read or write files.
    let names = try jscGlobalCandidates()
    check(names.count > 100, "jsc name extraction")
    let probe = try js("""
      const names = \(jsonString(names));
      const api = ["list", "read", "search", "write", "edit", "run", "console", "print"];
      const standard = new Set(\(jsonString(standardGlobals)));
      return names.filter((n) => !api.includes(n) && !standard.has(n)
        && Object.getOwnPropertyDescriptor(globalThis, n) !== undefined).join(",") || "none";
      """)
    check(probe.text == "Result:\nnone", "shell globals survived: \(probe.text)")
    for escape in [
      "return (0, eval)('typeof readFile')",
      "return Function('return typeof this.load')()",
      "return typeof globalThis.$262 + typeof createGlobalObject",
    ] {
      let result = try js(escape)
      check(!result.text.contains("function") && !result.text.contains("object"), "escape \(escape): \(result.text)")
    }
    let outside = try js("return [read(\(jsonString("../" + secret.lastPathComponent)))]")
    check(outside.isError && !outside.text.contains("PRIVATE"), "relative escape read")
    let absolute = try js("return run('cat \(secret.path)')")
    check(absolute.text.contains(#""status": 1"#) && !absolute.text.contains("PRIVATE"), "absolute read through run")
    check(try js("write('.git/config', 'x')").isError, ".git refused")

    // Permissions are enforced by the host, not only by hiding functions.
    let hidden = try js("return typeof write + typeof run + typeof edit", readOnly)
    check(hidden.text == "Result:\nundefinedundefinedundefined", "disabled functions hidden")
    let forged = AgentJavaScript.perform(
      "write", ["forged.txt", "x"], root: root, permissions: readOnly,
      deadline: ProcessInfo.processInfo.systemUptime + 5, parent: getppid())
    check(
      forged["ok"] as? Bool == false && !FileManager.default.fileExists(atPath: root.appendingPathComponent("forged.txt").path),
      "host refuses a disabled function")
    var noWrites = all
    noWrites.write = false
    let blocked = try js("return run('printf x > blocked.txt; echo $?')", noWrites)
    check(
      !FileManager.default.fileExists(atPath: root.appendingPathComponent("blocked.txt").path)
        && !blocked.text.contains(#""output": "0\n""#), "run cannot write without the write permission")

    // Limits: wall time, calls, memory, sizes.
    var started = Date()
    let spin = try js("while (true) {}")
    check(
      spin.isError && spin.text.contains("time limit") && Date().timeIntervalSince(started) < 13,
      "time limit: \(spin.text)")
    let many = try js("for (let i = 0; i < 300; i++) list()")
    check(many.isError && many.text.contains("more than 200"), "call limit: \(many.text)")
    started = Date()
    let memory = try js("const a = []; for (;;) a.push(new Array(1e6).fill(1.5))")
    check(memory.isError && Date().timeIntervalSince(started) < 13, "memory runaway bounded: \(memory.text)")
    let huge = try js("write('huge.txt', 'x'.repeat(300000))")
    check(huge.isError && huge.text.contains("256 KiB"), "write size limit")
    let large = try js("return console.log('y'.repeat(50000))")
    check(large.text.utf8.count < 33_000 && large.text.contains("truncated"), "output limit")
    let emptyCode = try js("")
    let longCode = try js(String(repeating: "x", count: 32_769))
    check(emptyCode.isError && longCode.isError, "code size")
    check(try js("a\0b").isError, "NUL refused")
  }

  static func jsonString(_ value: Any) -> String {
    String(decoding: try! JSONSerialization.data(withJSONObject: value, options: [.fragmentsAllowed]), as: UTF8.self)
  }

  static let standardGlobals = [
    "Object", "Function", "Array", "Number", "parseFloat", "parseInt", "Infinity", "NaN", "undefined",
    "Boolean", "String", "Symbol", "Date", "Promise", "RegExp", "Error", "AggregateError", "EvalError",
    "RangeError", "ReferenceError", "SyntaxError", "TypeError", "URIError", "SuppressedError",
    "globalThis", "JSON", "Math", "Intl", "ArrayBuffer", "SharedArrayBuffer", "Atomics", "Uint8Array",
    "Int8Array", "Uint16Array", "Int16Array", "Uint32Array", "Int32Array", "Float16Array",
    "Float32Array", "Float64Array", "Uint8ClampedArray", "BigUint64Array", "BigInt64Array", "DataView",
    "Map", "BigInt", "Set", "WeakMap", "WeakSet", "Proxy", "Reflect", "FinalizationRegistry", "WeakRef",
    "Iterator", "DisposableStack", "AsyncDisposableStack", "decodeURI", "decodeURIComponent",
    "encodeURI", "encodeURIComponent", "escape", "unescape", "eval", "isFinite", "isNaN",
  ]

  /// Identifier-like strings of this Mac's jsc binary: a new shell global on a future macOS
  /// fails the isolation check instead of surviving silently.
  static func jscGlobalCandidates() throws -> [String] {
    let data = try Data(contentsOf: URL(fileURLWithPath: AgentJavaScript.resolvedEngine))
    var names = Set<String>()
    var current = [UInt8]()
    for byte in data {
      if (byte >= 0x41 && byte <= 0x5a) || (byte >= 0x61 && byte <= 0x7a) || (byte >= 0x30 && byte <= 0x39)
        || byte == 0x5f || byte == 0x24
      {
        current.append(byte)
      } else {
        if (3...40).contains(current.count), let first = current.first, !(0x30...0x39).contains(first) {
          names.insert(String(decoding: current, as: UTF8.self))
        }
        current.removeAll(keepingCapacity: true)
      }
    }
    return names.sorted()
  }

  static func main() throws {
    let args = CommandLine.arguments
    if (args.count == 4 || args.count == 6) && args[1] == "--agent-command" {
      AgentSandbox.commandMode(root: args[2], command: args[3], seconds: args.count == 6 ? Double(args[4]) ?? 0 : 5, writable: args.count == 4 || args[5] == "write")
    }
    if args.count == 7 && args[1] == "--agent-js" { AgentJavaScript.supervisorMode(arguments: args) }
    if (args.count == 3 || args.count == 4) && args[1] == "--agent-tools" { AgentSandbox.serve(root: args[2], permissionsPath: args.count == 4 ? args[3] : nil) }
    let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-agent-\(UUID())"
    ).resolvingSymlinksInPath()
    let root = temporary.appendingPathComponent("workspace \"quoted\"")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }
    let secret = temporary.appendingPathComponent("private.txt")
    try Data("PRIVATE-CONTENT-DO-NOT-LEAK".utf8).write(to: secret)
    func quote(_ value: String) -> String {
      "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
    var count = 0
    func check(_ condition: Bool, _ message: String) {
      guard condition else { fatalError(message) }
      count += 1
    }
    func run(_ command: String, seconds: Double = 5) throws -> AgentSandbox.Result {
      try AgentSandbox.execute(root: root, command: command, seconds: seconds)
    }
    let success = try run("printf hello > greeting.txt; cat greeting.txt")
    check(
      success.status == 0 && !success.interrupted && success.output == "hello",
      "real sandbox must execute and round-trip a nonempty file: \(success)")
    let read = try run("cat \(quote(secret.path))")
    check(
      read.status != 0 && !read.output.contains("PRIVATE-CONTENT"), "absolute read escaped sandbox")
    let write = try run("printf changed > \(quote(secret.path))")
    let hardlink = try run("ln \(quote(secret.path)) hardlink; cat hardlink")
    check(
      hardlink.status != 0 && !hardlink.output.contains("PRIVATE-CONTENT")
        && !FileManager.default.fileExists(atPath: root.appendingPathComponent("hardlink").path),
      "hardlink source escaped sandbox")
    let unchanged = String(decoding: try Data(contentsOf: secret), as: UTF8.self)
    check(
      write.status != 0 && unchanged == "PRIVATE-CONTENT-DO-NOT-LEAK",
      "absolute write escaped sandbox")
    try FileManager.default.createSymbolicLink(
      at: root.appendingPathComponent("escape"), withDestinationURL: secret)
    let symlink = try run("cat escape; printf changed > escape")
    check(
      symlink.status != 0 && !symlink.output.contains("PRIVATE-CONTENT"), "symlink escaped sandbox")
    let network = try run("/usr/bin/curl --max-time 1 --noproxy '*' http://127.0.0.1:9")
    check(
      network.status != 0 && network.output.localizedCaseInsensitiveContains("permitted"),
      "network should be refused by Seatbelt: \(network.output)")
    let inherited = try run("/bin/sh -c \(quote("cat " + quote(secret.path)))")
    check(
      inherited.status != 0 && !inherited.output.contains("PRIVATE-CONTENT"),
      "child process escaped sandbox")
    let environment = try run("/usr/bin/env")
    let canonicalRoot = try AgentSandbox.canonicalRoot(root)
    check(
      !environment.output.contains("MLXL3_EXECUTABLE=") && !environment.output.contains("TOKEN=")
        && environment.output.contains("HOME=\(canonicalRoot.path)"), "host environment leaked")
    let timeout = try run("sleep 30", seconds: 0.2)
    check(timeout.interrupted, "silent process did not time out")
    let large = try run("yes data")
    check(
      large.interrupted && large.output.utf8.count <= AgentSandbox.maximumOutput,
      "output limit failed")
    let descendants = try run("sleep 30 & echo $! > child.pid; wait", seconds: 0.2)
    check(descendants.interrupted, "background child did not time out")
    let pid = Int32(
      String(
        decoding: try Data(contentsOf: root.appendingPathComponent("child.pid")), as: UTF8.self
      ).trimmingCharacters(in: .whitespacesAndNewlines))!
    usleep(100_000)
    check(kill(pid, 0) != 0, "background child survived cleanup")
    let sessions = try run(
      "/usr/bin/perl -MPOSIX -e 'my $a=POSIX::setsid(); my $b=POSIX::setpgid(0,0); print \"$a \", (defined($b) ? $b : -1), \"\\n\"'"
    )
    check(
      sessions.status == 0 && sessions.output == "-1 -1\n",
      "descendants can escape the process group: \(sessions)")
    let spawn = try run("/usr/bin/perl -e 'print syscall(244,0,0,0,0,0,0), \" $!\\n\"'")
    check(
      spawn.output == "-1 Operation not permitted\n",
      "posix_spawn group attributes remain reachable: \(spawn)")
    let limits = try run("ulimit -t 1000; ulimit -f 100000; ulimit -n 1000")
    check(
      limits.status != 0 && limits.output.contains("permitted"),
      "child can raise resource hard limits")
    for command in [
      "", String(repeating: "a", count: AgentSandbox.maximumCommand + 1), "echo\0oops",
    ] {
      do {
        _ = try run(command)
        fatalError("invalid command accepted")
      } catch { count += 1 }
    }
    for seconds in [0.0, -1, 30.1, .infinity, .nan] {
      do {
        _ = try run("true", seconds: seconds)
        fatalError("invalid timeout accepted")
      } catch { count += 1 }
    }
    do {
      _ = try AgentSandbox.profile(root: URL(fileURLWithPath: "/"))
      fatalError("root filesystem accepted")
    } catch { count += 1 }
    let profile = try AgentSandbox.profile(root: root)
    check(
      profile.contains("(deny default)") && !profile.contains("(allow network"),
      "policy is not deny-default")
    try checkJavaScript(temporary: temporary, secret: secret, count: &count)
    print(
      "Agent sandbox checks passed: \(count), real commands, file/network/child isolation, environment, limits, process cleanup and invalid input; JavaScript harness API, errors, isolation, permissions and limits"
    )
  }
}
