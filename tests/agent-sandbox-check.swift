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

    // search(): grep errors are errors (a pipe to head used to hide them as []), CRLF files
    // match (Swift splits "\r\n" as one character), capped results keep a warning.
    let invalid = try js(#"return search("add(")"#)
    check(
      invalid.isError && invalid.text.contains(#"search("add(") failed: parentheses not balanced"#)
        && invalid.text.contains("extended regex"), "invalid regex is an error: \(invalid.text)")
    let brackets = try js(#"return search("[")"#)
    check(brackets.isError && brackets.text.contains("brackets"), "unbalanced brackets: \(brackets.text)")
    let nowhere = try js(#"return search("add", "nowhere")"#)
    check(
      nowhere.isError && nowhere.text.contains("nowhere: No such file or directory")
        && !nowhere.text.contains("extended regex"), "missing path is an error: \(nowhere.text)")
    check(try js(#"return search("zzz-absent")"#).text == "Result:\n[]", "no match is an empty list")
    check(
      try js(#"return search("add\\(", "src").map((m) => m.path + ":" + m.line)"#).text
        == "Result:\n[\n  \"src/app.py:1\"\n]", "escaped parenthesis")
    let crlf = try js(#"return search("o", "crlf.txt").map((m) => [m.line, m.text])"#)
    check(
      crlf.text == "Result:\n[\n  [\n    1,\n    \"one\"\n  ],\n  [\n    2,\n    \"two\"\n  ]\n]",
      "CRLF matches without carriage returns: \(crlf.text)")
    try FileManager.default.createDirectory(
      at: root.appendingPathComponent("many"), withIntermediateDirectories: true)
    for file in 0..<5 {
      let lines = (0..<60).map { "hit \(file) \($0)\n" }.joined()
      try Data(lines.utf8).write(to: root.appendingPathComponent("many/f\(file).txt"))
    }
    let capped = try js(#"const r = search("^hit", "many"); return [r.length, new Set(r.map((m) => m.path)).size]"#)
    check(
      !capped.isError && capped.text.hasSuffix("Result:\n[\n  200,\n  4\n]")
        && capped.text.contains(#"warning: search("^hit"): stopped at 200 matches"#)
        && capped.text.contains("at most 50 matches per file are listed (many/f0.txt, many/f1.txt, many/f2.txt, ...)"),
      "capped search warns: \(capped.text)")
    let perFile = try js(#"return search("^hit 3 ", "many/f3.txt").length"#)
    check(
      perFile.text == "warning: search(\"^hit 3 \"): at most 50 matches per file are listed (many/f3.txt)\nResult:\n50",
      "per-file cap warns: \(perFile.text)")
    try FileManager.default.removeItem(at: root.appendingPathComponent("many"))

    // The reply itself, on outputs the sandbox can produce but a test project cannot force.
    func reply(_ text: String, interrupted: Bool = false, path: String = ".") -> [String: Any] {
      AgentJavaScript.searchReply(Data(text.utf8), interrupted: interrupted, path: path)
    }
    func paths(_ reply: [String: Any]) -> [String] {
      ((reply["value"] as? [[String: Any]]) ?? []).map { "\($0["path"]!):\($0["line"]!):\($0["text"]!)" }
    }
    let done = "\n\u{1E}\(AgentJavaScript.searchStatus)"
    let partly = reply("./a.txt:1:x\ngrep: ./b.txt: Permission denied\(done)2\n")
    check(
      partly["ok"] as? Bool == true && paths(partly) == ["a.txt:1:x"]
        && (partly["note"] as? String) == "some paths could not be searched: ./b.txt: Permission denied",
      "errors beside matches keep the matches: \(partly)")
    let unfinished = reply("./a.txt:1:x\n", interrupted: true)
    check(
      paths(unfinished) == ["a.txt:1:x"] && (unfinished["note"] as? String)?.hasPrefix("time limit reached") == true,
      "interrupted search is marked incomplete: \(unfinished)")
    let overflow = AgentJavaScript.searchReply(
      Data(("./a.txt:1:" + String(repeating: "y", count: AgentJavaScript.searchBytes)).utf8).prefix(AgentJavaScript.searchBytes),
      interrupted: true, path: ".")
    check(
      paths(overflow).count == 1 && (overflow["note"] as? String)?.hasPrefix("output limit reached") == true,
      "output limit is marked incomplete: \(overflow["note"] ?? "none")")
    check(
      (reply("", interrupted: true)["error"] as? String)?.contains("did not finish in time") == true,
      "interrupted search without a match is an error")
    check(
      (reply("./a.txt:1:x\n")["note"] as? String)?.hasPrefix("time limit reached") == true
        && reply("")["ok"] as? Bool == false,
      "a missing status line is never a complete result")
    check(
      paths(reply("grep: x:3:y\(done)0\n", path: "grep: x")) == ["grep: x:3:y"],
      "a searched path named like an error stays a match")
    check(
      paths(reply("./a.txt:1:\u{1E}\(AgentJavaScript.searchStatus)0\n./b.txt:2:z\(done)0\n")) == [
        "a.txt:1:\u{1E}\(AgentJavaScript.searchStatus)0", "b.txt:2:z",
      ], "only the last line is the status")
    check(reply("\(done)1\n")["ok"] as? Bool == true && paths(reply("\(done)1\n")).isEmpty, "status 1 = no match")
    check(
      (reply("grep: parentheses not balanced\(done)2\n")["error"] as? String)?.hasPrefix(
        "parentheses not balanced; the pattern is an extended regex") == true, "status 2 without a match")
    // searchLine against the previous parser (Swift Regex) on deterministic random lines.
    var seed: UInt64 = 0x9E37_79B9_7F4A_7C15
    let alphabet = Array("a::1:23x é/".unicodeScalars)
    var mismatch: String?
    var parsedLines = 0
    for _ in 0..<20_000 where mismatch == nil {
      var line = ""
      seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      for _ in 0..<Int(seed >> 60) {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        line.unicodeScalars.append(alphabet[Int(seed >> 33) % alphabet.count])
      }
      let parsed = AgentJavaScript.searchLine(Data(line.utf8))
      let oracle = line.firstMatch(of: /^(.+?):(\d+):(.*)$/).map {
        (String($0.1), Int($0.2) ?? -1, String($0.3))
      }
      if parsed != nil { parsedLines += 1 }
      if parsed?.path != oracle?.0 || parsed?.line != oracle?.1 || parsed?.text != oracle?.2 {
        mismatch = "searchLine(\(line.debugDescription)) = \(String(describing: parsed)), regex \(String(describing: oracle))"
      }
    }
    // Both outcomes must be exercised, or the comparison proves nothing.
    check(mismatch == nil && parsedLines > 1_000 && parsedLines < 19_000, mismatch ?? "searchLine coverage \(parsedLines)")
    check(
      AgentJavaScript.searchLine(Data("x.txt:12:a\r".utf8)).map { "\($0.path) \($0.line) \($0.text)" } == "x.txt 12 a",
      "carriage return removed")

    // run(): a talkative command finishes; its status and the end of its output reach the model.
    let talkative = try js("""
      return run("i=0; while [ $i -lt 2000 ]; do echo line-$i-xxxxxxxxxxxxxxxxxxxx; i=$((i+1)); done; echo FINISHED > ran.txt; echo tests: 42 passed")
      """)
    check(
      !talkative.isError
        && talkative.text.hasPrefix("Result:\n{\n  \"status\": 0,\n  \"stopped\": false,\n  \"omitted\": ")
        && talkative.text.hasSuffix("tests: 42 passed\\n\"\n}") && talkative.text.utf8.count < 33_000
        && FileManager.default.fileExists(atPath: root.appendingPathComponent("ran.txt").path),
      "talkative run(): \(talkative.text.prefix(120).debugDescription)…\(talkative.text.suffix(80).debugDescription)")
    var quick = all
    quick.commandSeconds = 1
    let stoppedRun = try js("const r = run('sleep 30'); return [r.status, r.stopped, r.omitted]", quick)
    check(stoppedRun.text == "Result:\n[\n  137,\n  true,\n  0\n]", "stopped run() status: \(stoppedRun.text)")
    // Results past 32 KiB keep their start and their end (the end used to be cut).
    let both = try js("console.log('a'.repeat(20000)); return 'b'.repeat(20000) + 'END'")
    check(
      !both.isError && both.text.hasPrefix("aaaa") && both.text.hasSuffix("bbbEND")
        && both.text.contains("bytes of output omitted (32 KiB limit)") && both.text.utf8.count < 33_000,
      "long outcome keeps start and end: \(both.text.utf8.count) bytes")
    let long = try js("return 'q'.repeat(50000) + 'END'")
    check(
      !long.isError && long.text.hasSuffix("qqqEND") && long.text.contains("characters of the result omitted")
        && long.text.utf8.count < 33_000,
      "long return value keeps start and end: \(long.text.suffix(40).debugDescription)")

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
    if [4, 6, 7].contains(args.count) && args[1] == "--agent-command" {
      AgentSandbox.commandMode(
        root: args[2], command: args[3], seconds: args.count >= 6 ? Double(args[4]) ?? 0 : 5,
        writable: args.count == 4 || args[5] == "write", mode: args.count == 7 && args[6] == "tail" ? .headAndTail : .stop)
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
      large.interrupted && large.output.utf8.count <= AgentSandbox.maximumOutput && large.omitted == nil
        && !large.output.contains("omitted"),
      "output limit failed")
    // A stopped command reports the signal that ended it, not the 0 of an uncollected process.
    check(timeout.status == 137 && large.status == 137, "stopped status: \(timeout.status) \(large.status)")

    // `run` commands finish: the output keeps its first 8 KiB and last 24 KiB.
    func command(_ text: String, seconds: Double = 5) throws -> AgentSandbox.Result {
      try AgentSandbox.execute(root: root, command: text, seconds: seconds, mode: .headAndTail)
    }
    let chatty = try command(
      "i=0; while [ $i -lt 2000 ]; do echo line-$i-xxxxxxxxxxxxxxxxxxxx; i=$((i+1)); done; echo FINISHED > done.txt; echo tests: 42 passed")
    // Independent oracle: the same output built here, then cut at the documented sizes.
    let expected = Data(((0..<2000).map { "line-\($0)-xxxxxxxxxxxxxxxxxxxx\n" }.joined() + "tests: 42 passed\n").utf8)
    let omittedBytes = expected.count - 8192 - 24576
    check(
      chatty.status == 0 && !chatty.interrupted && chatty.omitted == omittedBytes
        && chatty.raw == expected.prefix(8192) + Data("\n[... \(omittedBytes) bytes of output omitted ...]\n".utf8)
          + expected.suffix(24576)
        && (try? String(contentsOf: root.appendingPathComponent("done.txt"), encoding: .utf8)) == "FINISHED\n",
      "talkative command finishes with its start and end: status \(chatty.status) omitted \(String(describing: chatty.omitted)) end \(chatty.output.suffix(60).debugDescription)")
    for (size, cut) in [(20_000, false), (32_768, false), (32_769, true)] {
      let exact = try command("/usr/bin/perl -e 'print \"a\" x \(size)'")
      check(
        exact.status == 0 && (exact.omitted != nil) == cut
          && (cut ? exact.omitted == 1 : exact.raw == Data(repeating: 97, count: size)),
        "\(size) bytes: omitted \(String(describing: exact.omitted))")
    }
    // Cuts fall on UTF-8 boundaries: "x" + 20,000 "é" + "y" splits both ends inside an "é".
    let accents = try command("/usr/bin/perl -e 'print \"x\", \"\\xc3\\xa9\" x 20000, \"y\"'")
    check(
      accents.status == 0 && accents.omitted == 40_002 - 8191 - 24575 && !accents.output.contains("\u{FFFD}")
        && accents.output.hasPrefix("xé") && accents.output.hasSuffix("éy"),
      "UTF-8 boundaries: omitted \(String(describing: accents.omitted))")
    // Drained, not throttled: 10 MB well inside the deadline.
    let started10 = Date()
    let bulk = try command("/usr/bin/perl -e 'print \"z\" x 10000000'; echo END")
    check(
      bulk.status == 0 && !bulk.interrupted && bulk.omitted == 10_000_004 - 8192 - 24576
        && bulk.output.hasSuffix("zEND\n") && Date().timeIntervalSince(started10) < 4,
      "10 MB drained: status \(bulk.status) omitted \(String(describing: bulk.omitted)) in \(Date().timeIntervalSince(started10)) s")
    let endless = try command("yes data", seconds: 0.5)
    check(
      endless.interrupted && endless.status == 137 && (endless.omitted ?? 0) > 0
        // The deadline can stop it mid-line: the end is the latest "data" lines, maybe cut.
        && endless.output.suffix(64).contains("data\ndata") && endless.output.utf8.count < AgentSandbox.maximumOutput + 100,
      "endless output stops at the deadline with its end: status \(endless.status)")
    let silent = try command("sleep 30", seconds: 0.2)
    check(silent.interrupted && silent.status == 137 && silent.omitted == nil, "silent command stopped: \(silent)")
    let failing = try command("printf ok; exit 3")
    check(failing.status == 3 && failing.output == "ok" && !failing.interrupted, "exit status kept: \(failing)")
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
