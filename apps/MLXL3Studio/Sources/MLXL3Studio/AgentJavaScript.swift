import Darwin
import Foundation

/// How Agent mode offers the sandbox to the model.
enum AgentHarness: String, Codable, CaseIterable, Sendable {
  /// Four structured tools: `run`, `list_files`, `read_file`, `write_file`.
  case tools
  /// One `javascript` tool: the model writes a short program that calls the same operations
  /// as functions, so one call can explore, edit and test (CodeAct / "code mode").
  case javascript
}

/// The single-tool harness. The model's code runs in macOS's JavaScriptCore shell (`jsc`,
/// interpreter only) inside a compute-only Seatbelt profile: no project files, no network,
/// no processes. Each host function is a line-delimited RPC to this supervisor, which runs
/// the operation through the same sandboxed commands as the structured tools, so every
/// model-controlled executable text still crosses the kernel-enforced boundary.
enum AgentJavaScript {
  static let toolName = "javascript"
  static let maximumCode = 32_768
  /// Wall time of one call, below the engine's 12 s MCP call deadline.
  static let seconds = 10.0
  static let maximumCalls = 200
  static let maximumFileBytes = 256 * 1024
  static let maximumFootprint: UInt64 = 512 * 1024 * 1024
  static let maximumText = 32_768
  static let engine = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"

  struct Outcome: Codable, Equatable {
    let text: String
    let isError: Bool
  }

  /// Host functions offered under `permissions`, in documentation order.
  static func functions(_ permissions: AgentPermissions) -> [String] {
    var names: [String] = []
    if permissions.read { names += ["list", "read", "search"] }
    if permissions.write { names.append("write") }
    if permissions.read && permissions.write { names.append("edit") }
    if permissions.run { names.append("run") }
    return names
  }

  static func toolDescription(_ permissions: AgentPermissions) -> String {
    let signatures: [String: String] = [
      "list": "list(): string[]  // every file path in the project, relative, sorted",
      "read": "read(path: string): string  // whole UTF-8 file, at most 256 KiB",
      "search":
        "search(regex: string, path?: string): {path: string, line: number, text: string}[]  // grep -E, throws on an invalid regex or path; at most 200 matches, 50 per file (a warning says when capped)",
      "write":
        "write(path: string, text: string): void  // create or replace, parent folders created, at most 256 KiB",
      "edit":
        "edit(path: string, find: string, replace: string): void  // `find` must occur exactly once",
      "run":
        "run(command: string): {status: number, stopped: boolean, omitted: number, output: string}  // /bin/sh in the project, at most \(min(permissions.commandSeconds, Int(seconds))) s; past 32 KiB, output keeps its first 8 KiB and last 24 KiB (omitted: bytes cut)",
    ]
    let api = functions(permissions).compactMap { signatures[$0] }.joined(separator: "\n")
    return """
      Run JavaScript in your private macOS sandbox to work on the project. The code is the body of an async function: `return` a value to see it, and use console.log for notes. Batch steps: one call can read, edit and verify. These functions are synchronous (await is optional) and throw an Error on failure:
      \(api)
      No network, files outside the project, require/import or timers. Limits per call: \(Int(seconds)) s, \(maximumCalls) function calls, 32 KiB of output. Each call starts fresh: keep state in files. Writing is \(permissions.write ? "enabled" : "disabled").
      """
  }

  static var inputSchema: [String: Any] {
    [
    "type": "object",
    "properties": [
      "code": [
        "type": "string", "maxLength": maximumCode,
        "description": "JavaScript, the body of an async function",
      ]
    ],
      "required": ["code"], "additionalProperties": false,
    ]
  }

  /// `jsc` lives in the OS cryptex; Seatbelt matches the resolved path.
  static var resolvedEngine: String {
    guard let path = realpath(engine, nil) else { return engine }
    defer { free(path) }
    return String(cString: path)
  }

  /// Compute-only Seatbelt policy for the interpreter: system libraries only.
  static func profile() -> String {
    func literal(_ path: String) -> String {
      "\"" + path.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }
    return """
    (version 1)
    (deny default)
    (allow process-exec (literal "/bin/sh") (literal "/bin/bash") (literal "/bin/zsh")
      (literal "/bin/dash") (literal \(literal(engine))) (literal \(literal(resolvedEngine))))
    (allow syscall-unix)
    (deny syscall-unix (syscall-number SYS_setsid) (syscall-number SYS_setpgid)
      (syscall-number SYS_posix_spawn))
    (allow signal (target same-sandbox))
    (allow sysctl-read (sysctl-name-regex #"^hw[.]") (sysctl-name "kern.clockrate")
      (sysctl-name "kern.ostype") (sysctl-name "kern.osrelease") (sysctl-name "kern.osversion")
      (sysctl-name "kern.argmax") (sysctl-name "kern.ngroups")
      (sysctl-name "kern.maxfilesperproc") (sysctl-name "kern.maxprocperuid"))
    (allow file-read-metadata)
    (allow file-read* (literal "/") (subpath "/System/Library") (subpath "/usr/lib")
      (subpath "/usr/share") (subpath "/bin") (subpath "/usr/bin")
      (subpath "/System/Cryptexes/OS/System/Library")
      (subpath "/System/Volumes/Preboot/Cryptexes/OS/System/Library")
      (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom")
      (literal "/private/var/select/sh"))
    (allow file-write-data (literal "/dev/null"))
    """
  }

  // MARK: - Running one call

  /// Runs `code` with the host functions `permissions` allow. Never throws for model errors:
  /// they are part of the outcome the model reads.
  static func execute(root: URL, code: String, permissions: AgentPermissions) throws -> Outcome {
    guard !code.isEmpty, code.utf8.count <= maximumCode, !code.contains("\0") else {
      return Outcome(text: "Error: code must be 1 to \(maximumCode) bytes of text.", isError: true)
    }
    guard FileManager.default.isExecutableFile(atPath: engine) else {
      return Outcome(
        text: "Error: JavaScriptCore (jsc) is not available on this Mac; use the structured tools.",
        isError: true)
    }
    let root = try AgentSandbox.canonicalRoot(root)
    signal(SIGPIPE, SIG_IGN)
    let started = ProcessInfo.processInfo.systemUptime
    let deadline = started + seconds
    let parent = getppid()

    var input: [Int32] = [0, 0]
    var output: [Int32] = [0, 0]
    var errors: [Int32] = [0, 0]
    guard pipe(&input) == 0, pipe(&output) == 0, pipe(&errors) == 0 else {
      throw POSIXError(.EMFILE)
    }
    var actions: posix_spawn_file_actions_t?
    var attributes: posix_spawnattr_t?
    posix_spawn_file_actions_init(&actions)
    posix_spawnattr_init(&attributes)
    defer {
      posix_spawn_file_actions_destroy(&actions)
      posix_spawnattr_destroy(&attributes)
    }
    posix_spawn_file_actions_adddup2(&actions, input[0], STDIN_FILENO)
    posix_spawn_file_actions_adddup2(&actions, output[1], STDOUT_FILENO)
    posix_spawn_file_actions_adddup2(&actions, errors[1], STDERR_FILENO)
    for descriptor in input + output + errors {
      posix_spawn_file_actions_addclose(&actions, descriptor)
    }
    posix_spawn_file_actions_addchdir(&actions, "/")
    posix_spawnattr_setflags(
      &attributes, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT))
    posix_spawnattr_setpgroup(&attributes, 0)
    let arguments = [
      "/usr/bin/sandbox-exec", "-p", profile(), "/bin/sh", "-c",
      "ulimit -t \(Int(seconds) + 1) && ulimit -f 0 && ulimit -n 32 || exit 125; exec \"$0\" \"$@\"",
      engine, "--useJIT=false", "--useWasm=false", "--gcMaxHeapSize=268435456", "-e", prelude,
    ]
    let environment = ["PATH=/usr/bin:/bin", "LC_ALL=C"]
    let argv = arguments.map { strdup($0) } + [nil]
    let envp = environment.map { strdup($0) } + [nil]
    defer { for value in argv + envp { if let value { free(value) } } }
    var pid: pid_t = 0
    let spawned = posix_spawn(&pid, arguments[0], &actions, &attributes, argv, envp)
    close(input[0])
    close(output[1])
    close(errors[1])
    defer {
      close(input[1])
      close(output[0])
      close(errors[0])
    }
    guard spawned == 0 else { throw POSIXError(POSIXErrorCode(rawValue: spawned) ?? .EIO) }
    var reaped = false
    var status: Int32 = 0
    defer {
      kill(-pid, SIGKILL)
      if !reaped { while waitpid(pid, &status, 0) < 0 && errno == EINTR {} }
    }
    _ = fcntl(output[0], F_SETFL, O_NONBLOCK)
    _ = fcntl(errors[0], F_SETFL, O_NONBLOCK)

    let request: [String: Any] = ["code": code, "api": functions(permissions)]
    guard writeLine(request, to: input[1]) else {
      return Outcome(text: "Error: the JavaScript sandbox did not start.", isError: true)
    }
    var lines = LineBuffer()
    var diagnostics = Data()
    var finished: [String: Any]?
    var failure: String?
    var bytes = [UInt8](repeating: 0, count: 65_536)
    loop: while true {
      let count = read(output[0], &bytes, bytes.count)
      if count > 0 {
        guard lines.append(bytes.prefix(count)) else {
          failure = "the program produced an oversized message"
          break
        }
        while let line = lines.next() {
          guard let message = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else {
            failure = "the program broke the host protocol"
            break loop
          }
          if message["done"] as? Bool == true {
            finished = message
            break loop
          }
          guard let op = message["op"] as? String, let args = message["args"] as? [Any] else {
            failure = "the program broke the host protocol"
            break loop
          }
          let reply = perform(
            op, args, root: root, permissions: permissions, deadline: deadline, parent: parent)
          guard writeLine(reply, to: input[1]) else {
            failure = "the program stopped while a function was answering"
            break loop
          }
        }
      }
      let more = read(errors[0], &bytes, bytes.count)
      if more > 0 && diagnostics.count < 4096 { diagnostics.append(contentsOf: bytes.prefix(more)) }
      if !reaped, waitpid(pid, &status, WNOHANG) == pid { reaped = true }
      if reaped && count <= 0 {
        failure = "the program ended before finishing (an await that never resolves?)"
        break
      }
      if getppid() != parent {
        failure = "the agent stopped"
        break
      }
      if ProcessInfo.processInfo.systemUptime >= deadline {
        failure = "time limit of \(Int(seconds)) s reached"
        break
      }
      if footprint(pid) > maximumFootprint {
        failure = "memory limit of \(maximumFootprint >> 20) MiB reached"
        break
      }
      var descriptors = [
        pollfd(fd: output[0], events: Int16(POLLIN), revents: 0),
        pollfd(fd: errors[0], events: Int16(POLLIN), revents: 0),
      ]
      _ = poll(&descriptors, 2, 20)
    }
    if let finished { return outcome(finished) }
    var text = "Error: \(failure ?? "the program failed")."
    let detail = String(decoding: diagnostics, as: UTF8.self).trimmingCharacters(
      in: .whitespacesAndNewlines)
    if !detail.isEmpty { text += "\n" + String(detail.prefix(600)) }
    return Outcome(text: text, isError: true)
  }

  /// Text the model reads: console output, then the result or the error.
  static func outcome(_ message: [String: Any]) -> Outcome {
    var parts: [String] = []
    let logs = (message["logs"] as? [String]) ?? []
    if !logs.isEmpty { parts.append(logs.joined(separator: "\n")) }
    let error = message["error"] as? String
    if let value = message["value"] as? String { parts.append("Result:\n" + value) }
    if let error {
      parts.append("Error: " + error)
      // The error names the failed call; list the ones that completed (their effects stay).
      let calls = ((message["calls"] as? [String]) ?? []).filter { !$0.hasPrefix("FAILED ") }
      if !calls.isEmpty {
        parts.append("Completed before the error: " + calls.suffix(20).joined(separator: ", "))
      }
    }
    if parts.isEmpty {
      parts.append("Done, without output: `return` a value or console.log to see results.")
    }
    var text = parts.joined(separator: "\n")
    let bytes = Data(text.utf8)
    if bytes.count > maximumText {
      // Start and end: a result, an error or a command summary comes last.
      let head = AgentSandbox.utf8Prefix(bytes.prefix(maximumText / 4))
      let tail = AgentSandbox.utf8Suffix(Data(bytes.suffix(maximumText - maximumText / 4)))
      text = String(decoding: head, as: UTF8.self)
        + "\n[... \(bytes.count - head.count - tail.count) bytes of output omitted (32 KiB limit) ...]\n"
        + String(decoding: tail, as: UTF8.self)
    }
    return Outcome(text: text, isError: error != nil)
  }

  /// JSON with every non-ASCII scalar escaped: jsc's `readline` decodes bytes as Latin-1.
  static func asciiJSON(_ object: Any) -> Data? {
    guard let data = try? JSONSerialization.data(withJSONObject: object) else { return nil }
    var escaped = ""
    escaped.reserveCapacity(data.count)
    for scalar in String(decoding: data, as: UTF8.self).unicodeScalars {
      if scalar.isASCII {
        escaped.unicodeScalars.append(scalar)
      } else {
        for unit in String(scalar).utf16 { escaped += String(format: "\\u%04x", unit) }
      }
    }
    return Data(escaped.utf8)
  }

  private static func writeLine(_ object: Any, to descriptor: Int32) -> Bool {
    guard var data = asciiJSON(object) else { return false }
    data.append(10)
    return data.withUnsafeBytes { buffer in
      var offset = 0
      while offset < buffer.count {
        let written = write(descriptor, buffer.baseAddress! + offset, buffer.count - offset)
        if written < 0 && errno == EINTR { continue }
        guard written > 0 else { return false }
        offset += written
      }
      return true
    }
  }

  private static func footprint(_ pid: pid_t) -> UInt64 {
    var info = rusage_info_v4()
    let result = withUnsafeMutablePointer(to: &info) { pointer in
      pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
        proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
      }
    }
    return result == 0 ? info.ri_phys_footprint : 0
  }

  /// Newline-delimited messages, each at most 2 MiB (a 256 KiB write of non-ASCII text
  /// grows about sixfold once escaped).
  struct LineBuffer {
    private var data = Data()
    static let maximum = 2 << 20

    mutating func append(_ bytes: some Sequence<UInt8>) -> Bool {
      data.append(contentsOf: bytes)
      return data.count <= Self.maximum || data.contains(10)
    }

    mutating func next() -> Data? {
      guard let end = data.firstIndex(of: 10) else { return nil }
      let line = Data(data[data.startIndex..<end])
      data.removeSubrange(data.startIndex...end)
      return line
    }
  }

  // MARK: - Host functions

  /// One host function, answered as `{ok, value}` or `{ok: false, error}`.
  static func perform(
    _ op: String, _ args: [Any], root: URL, permissions: AgentPermissions, deadline: Double,
    parent: pid_t
  ) -> [String: Any] {
    func failure(_ text: String) -> [String: Any] { ["ok": false, "error": text] }
    func text(_ index: Int) -> String? { index < args.count ? args[index] as? String : nil }
    let remaining = deadline - ProcessInfo.processInfo.systemUptime
    guard remaining > 0.3 else { return failure("no time left in this call") }
    guard getppid() == parent else { return failure("the agent stopped") }
    func sandboxed(
      _ command: String, seconds: Double? = nil, writable: Bool = false, limit: Int,
      mode: AgentSandbox.OutputMode = .stop
    ) -> AgentSandbox.Result? {
      let budget = min(seconds ?? 5, remaining - 0.2)
      return try? AgentSandbox.execute(
        root: root, command: command, seconds: max(0.1, budget), writable: writable, limit: limit,
        mode: mode)
    }
    func quote(_ value: String) -> String {
      "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
    // `edit` is read then write, both of which it requires.
    guard functions(permissions).contains(op) else {
      return failure("\(op) is not enabled in this task")
    }
    switch op {
    case "list":
      guard
        let result = sandboxed(
          "find . -type f -not -path './.git/*' -print", limit: maximumFileBytes),
        !result.interrupted, result.status == 0
      else { return failure("the file list is too long or unavailable; use search or run") }
      let paths = result.output.split(separator: "\n").map {
        $0.hasPrefix("./") ? String($0.dropFirst(2)) : String($0)
      }.sorted()
      return ["ok": true, "value": paths]
    case "read":
      guard let path = text(0), AgentSandbox.validPath(path) else {
        return failure("use a relative path inside the project")
      }
      guard
        let result = sandboxed("cat -- \(quote(path))", limit: maximumFileBytes + 1)
      else { return failure("the sandbox is unavailable") }
      if result.interrupted && result.output.utf8.count > maximumFileBytes {
        return failure("\(path) is larger than 256 KiB; use search or run")
      }
      guard result.status == 0, !result.interrupted else {
        return failure(
          result.output.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? "cannot read \(path)" : result.output.trimmingCharacters(in: .whitespacesAndNewlines))
      }
      guard let raw = result.raw, let decoded = String(data: raw, encoding: .utf8),
        !decoded.contains("\0")
      else { return failure("\(path) is not UTF-8 text") }
      return ["ok": true, "value": decoded]
    case "search":
      guard let pattern = text(0), !pattern.isEmpty, pattern.utf8.count <= 1024,
        !pattern.contains("\0")
      else { return failure("the pattern must be 1 to 1024 bytes") }
      let path = text(1) ?? "."
      guard path == "." || AgentSandbox.validPath(path) else {
        return failure("use a relative path inside the project")
      }
      // No pipe: the status line carries grep's own exit status (a pipe reports the last
      // command's), and the output limit stops a large search early.
      guard
        let result = sandboxed(
          "/usr/bin/grep -rnIE --exclude-dir=.git -m \(searchPerFile) -e \(quote(pattern)) -- \(quote(path)); printf '\\n\\036\(searchStatus)%d\\n' \"$?\"",
          limit: searchBytes)
      else { return failure("the sandbox is unavailable") }
      return searchReply(
        result.raw ?? Data(result.output.utf8), interrupted: result.interrupted, path: path)
    case "write":
      guard let path = text(0), AgentSandbox.validPath(path), let content = text(1) else {
        return failure("use a relative path inside the project and text content")
      }
      guard content.utf8.count <= maximumFileBytes, !content.contains("\0") else {
        return failure("content must be at most 256 KiB of text")
      }
      let parent = (path as NSString).deletingLastPathComponent
      let temporary =
        (parent.isEmpty ? "" : parent + "/") + ".mlxl3-write-" + UUID().uuidString.prefix(8)
      // printf arguments stay below the 16 KiB command limit, quotes included.
      var chunks: [String] = []
      var current = ""
      var quoted = 0
      for character in content {
        let cost = character == "'" ? 4 : character.utf8.count
        if quoted + cost > 12_000 {
          chunks.append(current)
          current = ""
          quoted = 0
        }
        current.append(character)
        quoted += cost
      }
      chunks.append(current)
      for (index, chunk) in chunks.enumerated() {
        let prefix = index == 0 && !parent.isEmpty ? "mkdir -p -- \(quote(parent)) && " : ""
        let redirect = index == 0 ? ">" : ">>"
        guard
          let result = sandboxed(
            prefix + "printf '%s' \(quote(chunk)) \(redirect) \(quote(temporary))",
            writable: true, limit: 4096),
          result.status == 0, !result.interrupted
        else {
          _ = sandboxed("rm -f -- \(quote(temporary))", writable: true, limit: 4096)
          return failure("cannot write \(path)")
        }
      }
      guard
        let moved = sandboxed(
          "mv -f -- \(quote(temporary)) \(quote(path))", writable: true, limit: 4096),
        moved.status == 0, !moved.interrupted
      else {
        _ = sandboxed("rm -f -- \(quote(temporary))", writable: true, limit: 4096)
        return failure("cannot write \(path)")
      }
      return ["ok": true, "value": NSNull()]
    case "run":
      guard let command = text(0), !command.isEmpty, command.utf8.count <= AgentSandbox.maximumCommand,
        !command.contains("\0")
      else { return failure("the command must be 1 to 16384 bytes") }
      // The command runs to its end: long output keeps its start and end, never stops it.
      guard
        let result = sandboxed(
          command, seconds: Double(permissions.commandSeconds), writable: permissions.write,
          limit: AgentSandbox.maximumOutput, mode: .headAndTail)
      else { return failure("the sandbox refused the command") }
      return [
        "ok": true,
        "value": [
          "output": result.output, "status": Int(result.status), "stopped": result.interrupted,
          "omitted": result.omitted ?? 0,
        ],
      ]
    default:
      return failure("unknown function \(op)")
    }
  }

  // MARK: - search

  static let searchMatches = 200
  static let searchPerFile = 50
  static let searchBytes = 64 * 1024
  /// Last line of a search that ran to the end, after a record separator (0x1E).
  static let searchStatus = "mlxl3-grep-status="

  /// One `path:line:text` line of `grep -n`; the first `:<digits>:` ends the path, as before.
  /// Bytes, not `Character`s: Swift reads "\r\n" as one character, which merged the lines of
  /// a CRLF file so that none of its matches was returned.
  static func searchLine(_ bytes: Data) -> (path: String, line: Int, text: String)? {
    var line = [UInt8](bytes)
    if line.last == 13 { line.removeLast() }
    var index = 1
    while index < line.count {
      if line[index] == 58 {
        var end = index + 1
        while end < line.count, (48...57).contains(line[end]) { end += 1 }
        if end > index + 1, end < line.count, line[end] == 58,
          let number = Int(String(decoding: line[(index + 1)..<end], as: UTF8.self))
        {
          return (
            String(decoding: line[..<index], as: UTF8.self), number,
            String(decoding: line[(end + 1)...], as: UTF8.self)
          )
        }
      }
      index += 1
    }
    return nil
  }

  /// Host reply for one search. `output` is grep's stdout and stderr (they share the pipe),
  /// then the status line, missing when the sandbox stopped the command (time or size).
  /// An invalid pattern or path is an error, never an empty list; capped or incomplete
  /// results keep their matches and add a `note` the program logs as a warning.
  static func searchReply(_ output: Data, interrupted: Bool, path: String) -> [String: Any] {
    var lines = output.split(separator: 10, omittingEmptySubsequences: true)
    let marker = Data([0x1E] + Array(searchStatus.utf8))
    var status: Int?
    if !interrupted, let last = lines.last, last.starts(with: marker) {
      status = Int(String(decoding: last.dropFirst(marker.count), as: UTF8.self))
      lines.removeLast()
    }
    // Match lines start with the searched path ("./" for the project root).
    let matchPrefix = path == "." ? "./" : path
    var matches: [[String: Any]] = []
    var errors: [String] = []
    var perFile: [String: Int] = [:]
    var found = 0
    for bytes in lines {
      let text = String(decoding: bytes, as: UTF8.self)
      if text.hasPrefix("grep: ") && !text.hasPrefix(matchPrefix) {
        errors.append(String(text.dropFirst(6)).trimmingCharacters(in: .whitespacesAndNewlines))
        continue
      }
      guard var match = searchLine(Data(bytes)) else { continue }
      if match.path.hasPrefix("./") { match.path.removeFirst(2) }
      found += 1
      perFile[match.path, default: 0] += 1
      if matches.count < searchMatches {
        matches.append(["path": match.path, "line": match.line, "text": String(match.text.prefix(300))])
      }
    }
    func failure(_ text: String) -> [String: Any] { ["ok": false, "error": text] }
    let detail = errors.prefix(3).joined(separator: "; ")
    if found == 0 {
      if let status, status > 1 {
        let pathProblem = errors.contains {
          $0.contains("No such file") || $0.contains("Not a directory") || $0.contains("Permission denied")
        }
        let reason = detail.isEmpty ? "grep ended with status \(status)" : detail
        return failure(
          pathProblem
            ? reason
            : reason
              + "; the pattern is an extended regex (grep -E): put \\ before ( ) [ ] { } . * + ? | ^ $ to match them literally"
        )
      }
      if interrupted {
        return failure(
          output.count >= searchBytes
            ? "the search output exceeded 64 KiB before a match line; narrow the path or the pattern"
            : "the search did not finish in time; narrow the path or the pattern")
      }
      if status == nil { return failure("the search did not finish" + (detail.isEmpty ? "" : ": " + detail)) }
    }
    var notes: [String] = []
    if found > searchMatches {
      notes.append("stopped at \(searchMatches) matches; narrow the pattern or the path")
    } else if interrupted || status == nil {
      notes.append(
        output.count >= searchBytes
          ? "output limit reached, results are incomplete; narrow the pattern or the path"
          : "time limit reached, results are incomplete; narrow the pattern or the path")
    }
    let capped = perFile.filter { $0.value >= searchPerFile }.keys.sorted()
    if !capped.isEmpty {
      notes.append(
        "at most \(searchPerFile) matches per file are listed ("
          + capped.prefix(3).joined(separator: ", ") + (capped.count > 3 ? ", ..." : "") + ")")
    }
    if let status, status > 1 { notes.append("some paths could not be searched: " + detail) }
    var reply: [String: Any] = ["ok": true, "value": matches]
    if !notes.isEmpty { reply["note"] = notes.joined(separator: "; ") }
    return reply
  }

  // MARK: - Supervisor process

  /// `--agent-js <root> <read|-> <write|-> <run|-> <seconds>`: reads the code on stdin and
  /// prints the outcome as JSON. A separate process survives loss of the MCP transport and
  /// reaps its interpreter.
  static func supervisorMode(arguments: [String]) -> Never {
    guard arguments.count == 7, let seconds = Int(arguments[6]), (1...30).contains(seconds) else {
      exit(2)
    }
    var permissions = AgentPermissions()
    permissions.read = arguments[3] == "read"
    permissions.write = arguments[4] == "write"
    permissions.run = arguments[5] == "run"
    permissions.commandSeconds = seconds
    permissions.harness = .javascript
    var code = Data()
    while code.count <= maximumCode {
      let chunk = FileHandle.standardInput.availableData
      if chunk.isEmpty { break }
      code.append(chunk)
    }
    do {
      let result =
        code.count > maximumCode
        ? Outcome(text: "Error: code must be 1 to \(maximumCode) bytes of text.", isError: true)
        : try execute(
          root: URL(fileURLWithPath: arguments[2]), code: String(decoding: code, as: UTF8.self),
          permissions: permissions)
      try FileHandle.standardOutput.write(contentsOf: JSONEncoder().encode(result))
      exit(0)
    } catch { exit(1) }
  }

  // MARK: - Prelude

  /// Runs before the model's code: removes the shell's host functions (file, load, realm and
  /// timer helpers), installs `console` and the allowed host functions, then compiles the
  /// code as an async function body and reports one `done` message.
  static let prelude = #"""
    "use strict";
    (function () {
      // The shell's readline decodes Latin-1: both directions carry ASCII-only JSON.
      const shellPrint = print, receive = readline;
      const send = (line) => shellPrint(line.replace(/[\u0080-\uffff]/g,
        (unit) => "\\u" + unit.charCodeAt(0).toString(16).padStart(4, "0")));
      const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
      const standard = new Set(["Object", "Function", "Array", "Number", "parseFloat", "parseInt",
        "Infinity", "NaN", "undefined", "Boolean", "String", "Symbol", "Date", "Promise", "RegExp",
        "Error", "AggregateError", "EvalError", "RangeError", "ReferenceError", "SyntaxError",
        "TypeError", "URIError", "SuppressedError", "globalThis", "JSON", "Math", "Intl",
        "ArrayBuffer", "SharedArrayBuffer", "Atomics", "Uint8Array", "Int8Array", "Uint16Array",
        "Int16Array", "Uint32Array", "Int32Array", "Float16Array", "Float32Array", "Float64Array",
        "Uint8ClampedArray", "BigUint64Array", "BigInt64Array", "DataView", "Map", "BigInt", "Set",
        "WeakMap", "WeakSet", "Proxy", "Reflect", "FinalizationRegistry", "WeakRef", "Iterator",
        "DisposableStack", "AsyncDisposableStack", "decodeURI", "decodeURIComponent", "encodeURI",
        "encodeURIComponent", "escape", "unescape", "eval", "isFinite", "isNaN"]);
      // jsc shell globals are reified lazily: getOwnPropertyNames misses them until used.
      const shell = ["$262", "MemoryFootprint", "OSRExit", "WebAssemblyMemoryMode", "addressOf", "asDoubleNumber",
        "asyncTestPassed", "asyncTestStart", "atob", "btoa", "callerIsBBQOrOMGCompiled", "callerSourceOrigin",
        "checkModuleSyntax", "checkScriptSyntax", "checkSyntax", "console", "createGlobalObject",
        "createHeapBigInt", "createNonRopeNonAtomString", "createNoopNativeFunctionWithCapture",
        "createWebAssemblyMemoryWithMode", "debug", "describe", "describeArray", "disableRichSourceInfo",
        "disassembleBase64", "drainMicrotasks", "dropAllLocks", "dumpBytecodeProfile", "dumpTypesForAllVariables",
        "edenGC", "effectful42", "ensureArrayStorage", "failNextNewCodeBlock", "fiatInt52",
        "finalizationRegistryDeadCount", "finalizationRegistryLiveCount", "flashHeapAccess",
        "fullGC", "gc", "gcHeapSize", "generateHeapSnapshot", "generateHeapSnapshotForGCDebugging",
        "getRandomSeed", "hasCustomProperties", "heapCapacity", "is32BitPlatform", "is8BitString",
        "isBigInt32", "isFinalTier", "isHeapBigInt", "isInt32", "isPureNaN", "isRope", "jscOptions",
        "jscStack", "load", "loadString", "makeMasquerader", "mallocInALoop", "maxArguments",
        "memoryUsageStatistics", "neverInlineFunction", "noDFG", "noFTL", "noInline", "noOSRExitFuzzing",
        "numberOfDFGCompiles", "openFile", "optimizeNextInvocation", "os", "performance", "platformSupportsSamplingProfiler",
        "preciseTime", "predictInt32", "prettyPrint", "print", "printErr", "quit", "read",
        "readFile", "readline", "releaseWeakRefs", "reoptimizationRetryCount", "resetMemoryPeak",
        "resetSuperSamplerState", "run", "runString", "samplingProfilerStackTraces", "setRandomSeed",
        "setTimeout", "setUnhandledRejectionCallback", "sleepSeconds", "startSamplingProfiler",
        "testLoopCount", "totalCompileTime", "transferArrayBuffer", "useBigInt32", "version",
        "waitForReport", "waiterListSize", "wasmTestLoopCount", "write", "writeFile", "$vm",
        "$", "agent", "Worker", "WebAssembly"];
      const present = (name) => Object.getOwnPropertyDescriptor(globalThis, name) !== undefined;
      for (const name of shell.concat(Object.getOwnPropertyNames(globalThis))) {
        if (!standard.has(name)) { try { delete globalThis[name]; } catch (_) {} }
      }
      const logs = [], calls = [];
      let logSize = 0, callCount = 0;
      const finish = (fields) => send(JSON.stringify(Object.assign({ done: true, logs, calls }, fields)));
      // Fail closed: no shell helper (files, scripts, realms, timers) may survive.
      const leftover = shell.concat(Object.getOwnPropertyNames(globalThis))
        .filter((name) => !standard.has(name) && present(name));
      if (leftover.length) { finish({ error: "sandbox setup failed: " + leftover.join(", ") }); return; }

      const show = (value) => {
        if (typeof value === "string") return value;
        if (value === undefined) return "undefined";
        if (typeof value === "function") return "[Function " + (value.name || "anonymous") + "]";
        if (typeof value === "bigint" || typeof value === "symbol") return String(value);
        if (value instanceof Error) return value.name + ": " + value.message;
        if (value instanceof Map) value = Object.fromEntries(value);
        if (value instanceof Set) value = Array.from(value);
        try {
          const text = JSON.stringify(value, (_, item) => typeof item === "bigint" ? String(item) : item, 2);
          return text === undefined ? String(value) : text;
        } catch (_) { return String(value); }
      };
      const logger = (prefix) => (...values) => {
        if (logSize > 30000) return;
        const line = prefix + values.map(show).join(" ");
        logSize += line.length + 1;
        logs.push(logSize > 30000 ? line.slice(0, 2000) + "\n[console output truncated]" : line);
      };
      globalThis.console = Object.freeze({ log: logger(""), info: logger(""), debug: logger(""),
        warn: logger("warning: "), error: logger("error: ") });
      globalThis.print = console.log;

      const warn = logger("warning: ");
      const hostError = (Type, message) => { const error = new Type(message); error.host = true; return error; };
      const label = (name, value) => name + "(" + (value === undefined ? "" :
        JSON.stringify(value.length > 60 ? value.slice(0, 57) + "..." : value)) + ")";
      const host = (op, args, name) => {
        if (++callCount > 200) throw hostError(Error, "more than 200 function calls in one run");
        send(JSON.stringify({ op, args }));
        const line = receive();
        if (typeof line !== "string") throw hostError(Error, "the host stopped answering");
        const reply = JSON.parse(line);
        calls.push((reply.ok ? "" : "FAILED ") + name);
        if (!reply.ok) throw hostError(Error, name + " failed: " + reply.error);
        // Capped or incomplete results stay usable; the warning says so.
        if (typeof reply.note === "string") warn(name + ": " + reply.note);
        return reply.value;
      };
      const text = (value, what, fn) => {
        if (typeof value !== "string") throw hostError(TypeError, fn + ": " + what + " must be a string");
        return value;
      };
      const api = {
        list: () => host("list", [], "list()"),
        read: (path) => host("read", [text(path, "path", "read")], label("read", path)),
        search: (pattern, path = ".") => host("search",
          [text(pattern, "regex", "search"), text(path, "path", "search")], label("search", pattern)),
        write: (path, content) => {
          text(path, "path", "write");
          if (content === undefined || content === null) throw hostError(TypeError, "write: text must be a string");
          const body = typeof content === "string" ? content : show(content);
          host("write", [path, body], label("write", path));
        },
        edit: (path, find, replace) => {
          text(path, "path", "edit"); text(find, "find", "edit"); text(replace, "replace", "edit");
          if (!find) throw hostError(Error, "edit: find must not be empty");
          const source = host("read", [path], label("read", path));
          const count = source.split(find).length - 1;
          if (count !== 1) throw hostError(Error, count === 0
            ? "edit: the text to find is not in " + path
            : "edit: the text to find occurs " + count + " times in " + path + "; include more context");
          host("write", [path, source.replace(find, () => replace)], label("write", path));
        },
        // Fixed key order: the status stays readable even when a long output is cut.
        run: (command) => {
          const r = host("run", [text(command, "command", "run")], label("run", command));
          return { status: r.status, stopped: r.stopped, omitted: r.omitted, output: r.output };
        },
      };

      let request;
      try { request = JSON.parse(receive()); } catch (_) { finish({ error: "invalid request" }); return; }
      for (const name of request.api) {
        Object.defineProperty(globalThis, name, { value: api[name], writable: false, configurable: false });
      }
      const describe = (error) => {
        if (!(error instanceof Error)) return "uncaught " + show(error);
        let line = error.name === "Error" ? error.message : error.name + ": " + error.message;
        if (!error.host && typeof error.line === "number" && error.line > 2 && !error.sourceURL) {
          line += " (line " + (error.line - 2) + ")";
        }
        return line;
      };
      // Small models often fence their code or send it with literal "\n" escapes.
      const code = String(request.code);
      const fenced = /^\s*```[\w+-]*[ \t]*\n([\s\S]*?)\n?```\s*$/.exec(code);
      const candidates = fenced ? [fenced[1], code] : [code];
      for (const source of candidates.slice()) {
        if (!source.includes("\n") && source.includes("\\n")) {
          candidates.push(source.replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\"/g, "\""));
        }
      }
      let program, firstError;
      for (const source of candidates) {
        try { program = new AsyncFunction(source); break; } catch (error) { firstError ??= error; }
      }
      if (!program) { finish({ error: describe(firstError) }); return; }
      program().then(
        (value) => {
          if (value === undefined) { finish({}); return; }
          const text = show(value);
          finish({ value: text.length <= 33000 ? text : text.slice(0, 8000) + "\n[... " + (text.length - 33000)
            + " characters of the result omitted ...]\n" + text.slice(-25000) });
        },
        (error) => finish({ error: describe(error) }));
    })();
    """#
}
