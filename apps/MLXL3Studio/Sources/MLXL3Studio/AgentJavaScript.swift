import Darwin
import Foundation

/// How Agent mode offers the sandbox to the model.
enum AgentHarness: String, Codable, CaseIterable, Sendable {
  /// Four structured tools: `run`, `list_files`, `read_file`, `write_file`.
  case tools
  /// One `javascript` tool: the model writes a short program that calls the same operations
  /// as functions, so one call can explore, edit and test (CodeAct / "code mode").
  case javascript
  /// Native function calls with flat arguments (deepseek-harness' default executor): `read`,
  /// `edit`, `write`, `grep`, `glob`, `bash` and the job tools (`AgentTools`). The model
  /// writes no program; the same host functions do the work.
  case native
}

/// The single-tool harness. The model's code runs in macOS's JavaScriptCore shell (`jsc`,
/// interpreter only) inside a compute-only Seatbelt profile: no project files, no network,
/// no processes. Each host function is a line-delimited RPC to this supervisor, which runs
/// the operation through the same sandboxed commands as the structured tools, so every
/// model-controlled executable text still crosses the kernel-enforced boundary.
enum AgentJavaScript {
  static let toolName = "javascript"
  /// Code of one call and text of one file: 1 MiB, far more than a model writes in one step
  /// (over 250,000 tokens), so a whole program fits in one `write`.
  static let maximumCode = 1 << 20
  /// Default wall time of one call, below the 12 s MCP call deadline of older engines.
  static let seconds = 10.0
  /// Longest wall an engine with per-server MCP deadlines may grant (commands: 30 s + 3).
  static let longestWall = 33.0
  static let maximumCalls = 200
  static let maximumFileBytes = 1 << 20
  /// Paths `list()` returns before it says the list is partial.
  static let maximumListBytes = 256 * 1024
  static let maximumFootprint: UInt64 = 512 * 1024 * 1024
  static let maximumText = 32_768
  /// `JSON.stringify(state)` length kept between calls (characters; the protocol escapes
  /// non-ASCII, so a line stays far below `LineBuffer.maximum`).
  static let maximumState = 256 * 1024
  static let engine = "/System/Library/Frameworks/JavaScriptCore.framework/Versions/A/Helpers/jsc"

  struct Outcome: Codable, Equatable {
    let text: String
    let isError: Bool
  }

  /// What the model reads for code over the limit: the sizes and what to do, nothing run.
  static func tooLarge(_ bytes: Int) -> Outcome {
    Outcome(
      text: "Error: this code is \(bytes) bytes; one call takes at most \(maximumCode) bytes (1 MiB). Nothing was run. Write a long file in several calls (write the first part, then append the rest with write(path, read(path) + part)).",
      isError: true)
  }

  /// The same for the `text` argument, which one `write` must be able to take.
  static func textTooLarge(_ bytes: Int) -> Outcome {
    Outcome(
      text: "Error: `text` is \(bytes) bytes; one call takes at most \(maximumFileBytes) bytes of text (1 MiB). Nothing was run. Write a long file in several calls: write(path, text) for the first part, then write(path, read(path) + text) for each next part.",
      isError: true)
  }

  /// Per-call settings from the MCP server.
  struct Settings: Sendable {
    /// Wall time of the whole call, host functions included.
    var wall = AgentJavaScript.seconds
    /// File keeping `state` between calls (outside the workspace); nil: no persistence.
    var stateFile: URL?
    var commandFolder: URL?
  }

  /// Wall of one call for `commandSeconds`, when the engine accepts a longer MCP deadline.
  static func wall(commandSeconds: Int, longCalls: Bool) -> Double {
    longCalls ? min(longestWall, max(seconds, Double(commandSeconds) + 3)) : seconds
  }

  /// Host functions offered under `permissions`, in documentation order.
  static func functions(_ permissions: AgentPermissions) -> [String] {
    var names: [String] = []
    if permissions.read { names += ["list", "glob", "view", "read", "search"] }
    if permissions.read && permissions.write { names += ["edit", "edits"] }
    if permissions.write { names.append("write") }
    if permissions.run { names += ["run", "spawn", "jobs", "tail", "wait", "kill"] }
    if permissions.internet {
      names.append("fetch")
      if permissions.write { names.append("download") }
    }
    return names
  }

  static func toolDescription(
    _ permissions: AgentPermissions, wall: Double = seconds, persistent: Bool = false
  ) -> String {
    var signatures: [String: String] = [
      "list": "list(): string[]  // every project file path, relative, sorted",
      "glob": "glob(pattern: string): string[]  // e.g. \"src/**/*.ts\"; without / it matches file names anywhere",
      "view":
        "view(path: string, offset?: number, limit?: number): string  // numbered `N| ` lines, 200 by default",
      "read": "read(path: string): string  // raw text, at most 1 MiB",
      "search":
        "search(regex: string, path?: string): {path: string, line: number, text: string}[]  // grep -E; at most 200 matches, 50 per file",
      "edit":
        "edit(path: string, find: string, replace: string, all?: boolean): {line: number, count: number, snippet: string}  // `find` once (all: each); tolerates indentation, `N| ` prefixes, curly quotes",
      "edits": "edits(path: string, changes: [find: string, replace: string][]): {count: number, lines: number[]}  // several edits of one file, all or nothing",
      "write":
        "write(path: string, text: string): void  // creates folders, at most 1 MiB",
      "run":
        "run(command: string): {status: number | null, stopped: boolean, omitted: number, output: string, job?: number}  // /bin/sh in the project, \(permissions.internet ? "with Internet (curl, git clone, pip, npm)" : "no network"); still running after \(min(permissions.commandSeconds, Int(wall))) s: status null, continues as job `job`; output over 32 KiB keeps its first 8 and last 24 KiB, all of it in the file the result names",
      "spawn": "spawn(command: string): number  // background job (server, build), ≤\(Jobs.running) at once, ≤\(Int(AgentSandbox.longestJob) / 60) min; its end is announced in a later result",
      "jobs": "jobs(): {id, command, running, status, seconds}[];  tail(id, lines = 40): string;  wait(id, seconds?): {id, running, status, seconds, output};  kill(id)",
    ]
    let hosts = permissions.internetHosts.prefix(16).joined(separator: ", ")
      + (permissions.internetHosts.count > 16 ? ", ..." : "")
    signatures["fetch"] =
      "fetch(url: string): {status: number, url: string, type: string, text: string}  // HTTPS GET to the allowed hosts (\(hosts)), text only, at most 2 MiB; redirects are checked"
    signatures["download"] =
      "download(url: string, path: string): {status: number, url: string, bytes: number}  // saves into the project, at most 100 MiB; put test material under .mlxl3-scratch/ (never part of the reviewed changes)"
    let api = functions(permissions).compactMap { signatures[$0] }.joined(separator: "\n")
    return """
      Run JavaScript in your private macOS sandbox to work on the project. The code is the body of an async function: `return` a value to see it (a lone expression is returned as is), and use console.log for notes. Batch steps: one call can read, edit and verify. These functions are synchronous (await is optional) and throw an Error on failure:
      \(api)
      The JavaScript itself has no direct network, files outside the project, require/import or timers. Limits per call: \(Int(wall)) s, \(maximumCalls) function calls, 32 KiB of output. \(persistent
        ? "`$` keeps any value between calls (emptied if the kernel restarts; a result says so). `state` is also kept on disk (JSON values, \(maximumState / 1024) Ki characters)."
        : "Each call starts fresh.") Writing is \(permissions.write ? "enabled. A file's content goes in the `text` argument, not in the code: write(path, text) then needs no escaping" : "disabled").
      """
  }

  static var inputSchema: [String: Any] {
    [
    "type": "object",
    "properties": [
      "code": [
        "type": "string", "maxLength": maximumCode,
        "description": "JavaScript, the body of an async function",
      ],
      "text": [
        "type": "string", "maxLength": maximumFileBytes,
        "description": "Optional raw text (a file's content), read in the code as the variable `text`",
      ],
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

  /// Background commands of a kernel (Unharnessed's `bash()` handles, opencode's background
  /// shell): they outlive the call that started them, never the kernel that owns them.
  final class Jobs: @unchecked Sendable {
    struct Job {
      let id: Int
      let command: String
      let started: Double
      let watcher: AgentSandbox.Watcher
      /// Its end was reported (a notice) or its final output was read.
      var noticed = false
    }
    static let running = 4
    static let kept = 16
    private let lock = NSLock()
    private var table: [Int: Job] = [:]
    private var next = 1

    func start(root: URL, command: String, writable: Bool, network: Bool, commandFolder: URL? = nil) throws -> Int {
      lock.lock()
      defer { lock.unlock() }
      guard table.values.filter({ $0.watcher.result == nil }).count < Self.running else {
        throw AgentNetwork.Refusal(
          "\(Self.running) jobs are already running; wait for one or kill one first")
      }
      // Only the newest finished jobs are kept.
      let finished = table.values.filter { $0.watcher.result != nil }.map(\.id).sorted()
      for id in finished.dropLast(Self.kept - 1) { table[id] = nil }
      let id = next
      next += 1
      // P4: a cut output is kept whole in `.mlxl3-scratch/outputs/` (writable copies only).
      let spill =
        writable
        ? AgentSandbox.Spill(root: root, name: "job-\(id)-\(UUID().uuidString.prefix(8).lowercased()).log")
        : nil
      let watcher = AgentSandbox.Watcher(spill: spill)
      table[id] = Job(id: id, command: command, started: ProcessInfo.processInfo.systemUptime, watcher: watcher)
      let recording = AgentCommandReceipt.start(command: command, in: commandFolder)
      Thread {
        let result =
          (try? AgentSandbox.execute(
            root: root, command: command, seconds: AgentSandbox.longestJob, writable: writable,
            limit: AgentSandbox.maximumOutput, mode: .headAndTail, network: network, watcher: watcher,
            spill: spill))
          ?? AgentSandbox.Result(
            output: "the sandbox refused the command", status: 126, interrupted: false, raw: nil, omitted: nil)
        recording?.finish(result)
        watcher.finish(result)
      }.start()
      return id
    }

    /// What the model sees of a job; `output` adds its output so far (or final), `read` marks
    /// a finished job as seen (no notice later).
    func describe(_ id: Int, output: Bool = false, read: Bool = false) -> [String: Any]? {
      lock.lock()
      defer { lock.unlock() }
      guard var job = table[id] else { return nil }
      let result = job.watcher.result
      let end = job.watcher.endedAt ?? ProcessInfo.processInfo.systemUptime
      var value: [String: Any] = [
        "id": id, "command": job.command.count > 300 ? String(job.command.prefix(297)) + "..." : job.command,
        "running": result == nil,
        "status": result.map { Int($0.status) as Any } ?? NSNull(),
        "seconds": Int((end - job.started).rounded()),
      ]
      if output {
        if let result {
          value["output"] = result.output
          value["omitted"] = result.omitted ?? 0
          value["stopped"] = result.interrupted
        } else {
          let current = job.watcher.output
          value["output"] = current.text
          value["omitted"] = current.omitted
        }
      }
      if read && result != nil {
        job.noticed = true
        table[id] = job
      }
      return value
    }

    func list() -> [[String: Any]] {
      lock.lock()
      let ids = table.keys.sorted()
      lock.unlock()
      return ids.compactMap { describe($0) }
    }

    /// Waits until the job ends or `seconds` pass (polled every 20 ms).
    func wait(_ id: Int, seconds: Double) -> [String: Any]? {
      lock.lock()
      let watcher = table[id]?.watcher
      lock.unlock()
      guard let watcher else { return nil }
      let until = ProcessInfo.processInfo.systemUptime + max(0, seconds)
      while watcher.result == nil && ProcessInfo.processInfo.systemUptime < until { usleep(20_000) }
      return describe(id, output: true, read: true)
    }

    func kill(_ id: Int) -> [String: Any]? {
      lock.lock()
      let watcher = table[id]?.watcher
      lock.unlock()
      guard let watcher else { return nil }
      watcher.requestStop()
      return wait(id, seconds: 3)
    }

    /// One line per job that ended since the last call and was not read: the next result
    /// carries it (Unharnessed's `[bash-done …]`).
    func notices(native: Bool = false) -> [String] {
      lock.lock()
      defer { lock.unlock() }
      var lines: [String] = []
      for id in table.keys.sorted() {
        guard var job = table[id], !job.noticed, let result = job.watcher.result else { continue }
        job.noticed = true
        table[id] = job
        let seconds = Int(((job.watcher.endedAt ?? job.started) - job.started).rounded())
        let command = job.command.count > 80 ? String(job.command.prefix(77)) + "..." : job.command
        lines.append(
          "[job \(id) finished: status \(result.status) after \(seconds) s] \(command) "
            + (native ? "(job_output with job_id \(id) shows its output)" : "(wait(\(id)) shows its output)"))
      }
      return lines
    }

    /// Stops every job (end of the kernel) and waits up to 3 s for them.
    func stopAll() {
      lock.lock()
      let watchers = table.values.map(\.watcher)
      lock.unlock()
      watchers.forEach { $0.requestStop() }
      let until = ProcessInfo.processInfo.systemUptime + 3
      while watchers.contains(where: { $0.result == nil }) && ProcessInfo.processInfo.systemUptime < until {
        usleep(20_000)
      }
    }
  }

  /// A JavaScript kernel (Unharnessed's persistent REPL): one sandboxed `jsc` kept between
  /// calls, so `$` and what the model defines there survive; its background jobs; and the
  /// note the next result carries after the kernel had to be restarted.
  final class Kernel {
    let root: URL
    let jobs = Jobs()
    private var process: (pid: pid_t, input: Int32, output: Int32, errors: Int32)?
    private var lines = LineBuffer()
    /// The previous kernel ended between calls (crash, unresolved await): said once.
    private var lost = false
    /// CPU seconds the kernel may use over its whole life (`ulimit -t`): enough for many calls
    /// (host functions wait, they do not count), short for a loop left behind by a dead host.
    static let cpuSeconds = 300

    init(root: URL) throws { self.root = try AgentSandbox.canonicalRoot(root) }

    deinit { shutdown() }

    func shutdown() {
      stopProcess()
      jobs.stopAll()
    }

    private func stopProcess() {
      guard let process else { return }
      self.process = nil
      kill(-process.pid, SIGKILL)
      var status: Int32 = 0
      while waitpid(process.pid, &status, 0) < 0 && errno == EINTR {}
      close(process.input)
      close(process.output)
      close(process.errors)
      lines = LineBuffer()
    }

    private func startProcess() throws {
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
        "ulimit -t \(Self.cpuSeconds) && ulimit -f 0 && ulimit -n 32 || exit 125; exec \"$0\" \"$@\"",
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
      guard spawned == 0 else {
        close(input[1])
        close(output[0])
        close(errors[0])
        throw POSIXError(POSIXErrorCode(rawValue: spawned) ?? .EIO)
      }
      _ = fcntl(output[0], F_SETFL, O_NONBLOCK)
      _ = fcntl(errors[0], F_SETFL, O_NONBLOCK)
      _ = fcntl(input[1], F_SETNOSIGPIPE, 1)
      process = (pid, input[1], output[0], errors[0])
    }

    /// Runs `code` with the host functions `permissions` allow. Never throws for model errors:
    /// they are part of the outcome the model reads.
    /// `text` is the call's `text` argument: the program reads it as the variable `text`.
    /// `tool` is a native call (`AgentTools`): the kernel runs the app's own function of that
    /// name with these arguments as data, and `code` is not read.
    func run(
      code: String, text: String? = nil, tool: (name: String, arguments: [String: Any])? = nil,
      permissions: AgentPermissions, settings: Settings = Settings()
    ) throws -> Outcome {
      if tool == nil {
        guard !code.isEmpty, code.utf8.count <= maximumCode, !code.contains("\0") else {
          return code.utf8.count > maximumCode
            ? tooLarge(code.utf8.count) : Outcome(text: "Error: the code is empty or contains NUL.", isError: true)
        }
      }
      if let text, text.utf8.count > maximumFileBytes { return textTooLarge(text.utf8.count) }
      guard settings.wall.isFinite, (1...longestWall).contains(settings.wall) else {
        throw CocoaError(.fileReadCorruptFile)
      }
      let seconds = settings.wall
      // `state` from the previous call; an unreadable file starts over, and says so.
      var state: Any = [String: Any]()
      var notes: [String] = []
      if let file = settings.stateFile, FileManager.default.fileExists(atPath: file.path) {
        if let data = try? Data(contentsOf: file), data.count <= 6 * maximumState,
          let value = try? JSONSerialization.jsonObject(with: data), value is [String: Any]
        {
          state = value
        } else {
          notes.append("the saved state could not be read; `state` starts empty")
        }
      }
      guard FileManager.default.isExecutableFile(atPath: engine) else {
        return Outcome(
          text: tool == nil
            ? "Error: JavaScriptCore (jsc) is not available on this Mac; use the structured tools."
            : "Error: the tool runtime (JavaScriptCore, jsc) is not available on this Mac.",
          isError: true)
      }
      signal(SIGPIPE, SIG_IGN)
      let deadline = ProcessInfo.processInfo.systemUptime + seconds
      let parent = getppid()
      // A kernel that ended since the last call is replaced, and the model is told.
      if let process {
        var status: Int32 = 0
        if waitpid(process.pid, &status, WNOHANG) == process.pid {
          self.process = nil
          close(process.input)
          close(process.output)
          close(process.errors)
          lines = LineBuffer()
          lost = true
        }
      }
      if process == nil { try startProcess() }
      if lost {
        // A native call keeps nothing in the kernel: nothing to tell the model.
        if tool == nil {
          notes.append("the JavaScript kernel had stopped and was restarted: `$` and what was kept in it are gone; `state` was kept")
        }
        lost = false
      }
      guard let process else { throw POSIXError(.EIO) }
      var request: [String: Any] = [
        "code": code, "api": functions(permissions), "state": state,
        "persist": settings.stateFile != nil,
      ]
      if let text { request["text"] = text }
      if let tool {
        request["tool"] = tool.name
        request["args"] = tool.arguments
      }
      var diagnostics = Data()
      var finished: [String: Any]?
      var failure: String?
      var bytes = [UInt8](repeating: 0, count: 65_536)
      if !writeLine(request, to: process.input) { failure = "the JavaScript kernel did not start" }
      loop: while failure == nil {
        let count = read(process.output, &bytes, bytes.count)
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
              op, args, root: root, permissions: permissions, deadline: deadline, parent: parent,
              jobs: jobs, commandFolder: settings.commandFolder)
            guard writeLine(reply, to: process.input) else {
              failure = "the program stopped while a function was answering"
              break loop
            }
          }
        }
        let more = read(process.errors, &bytes, bytes.count)
        if more > 0 && diagnostics.count < 4096 { diagnostics.append(contentsOf: bytes.prefix(more)) }
        var status: Int32 = 0
        if count <= 0, waitpid(process.pid, &status, WNOHANG) == process.pid {
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
        if footprint(process.pid) > maximumFootprint {
          failure = "memory limit of \(maximumFootprint >> 20) MiB reached"
          break
        }
        var descriptors = [
          pollfd(fd: process.output, events: Int16(POLLIN), revents: 0),
          pollfd(fd: process.errors, events: Int16(POLLIN), revents: 0),
        ]
        _ = poll(&descriptors, 2, 20)
      }
      let finishedJobs = jobs.notices(native: tool != nil)
      if let finished {
        var warnings = notes
        if let file = settings.stateFile {
          if let text = finished["state"] as? String, let data = text.data(using: .utf8),
            (try? JSONSerialization.jsonObject(with: data)) is [String: Any]
          {
            do { try data.write(to: file, options: .atomic) } catch {
              warnings.append("state was not saved: \(error.localizedDescription)")
            }
          } else {
            warnings.append("state was not saved: \((finished["stateError"] as? String) ?? "it is not a JSON object")")
          }
        }
        let result = outcome(finished, warnings: warnings, plain: tool != nil)
        return finishedJobs.isEmpty
          ? result : Outcome(text: result.text + "\n" + finishedJobs.joined(separator: "\n"), isError: result.isError)
      }
      // The kernel is stopped: what it held is gone, the next call starts a new one.
      stopProcess()
      var text = "Error: \(failure ?? "the program failed")."
      let detail = String(decoding: diagnostics, as: UTF8.self).trimmingCharacters(
        in: .whitespacesAndNewlines)
      if !detail.isEmpty { text += "\n" + String(detail.prefix(600)) }
      if tool == nil {
        text += "\nThe JavaScript kernel was restarted: `$` is empty again"
          + (settings.stateFile != nil ? "; `state` keeps its value from before this call." : ".")
      }
      if !notes.isEmpty { text += "\n" + notes.map { "warning: " + $0 }.joined(separator: "\n") }
      if !finishedJobs.isEmpty { text += "\n" + finishedJobs.joined(separator: "\n") }
      return Outcome(text: text, isError: true)
    }
  }

  /// A one-call kernel: what `run` does in a kernel that is then stopped (with its jobs).
  static func execute(
    root: URL, code: String, text: String? = nil, permissions: AgentPermissions, settings: Settings = Settings()
  ) throws -> Outcome {
    let kernel = try Kernel(root: root)
    defer { kernel.shutdown() }
    return try kernel.run(code: code, text: text, permissions: permissions, settings: settings)
  }

  /// Text the model reads: console output, then the result or the error.
  /// Advisory notes on loops (OpenHands' stuck detector, deepseek-harness' repeat reminder):
  /// the same program with the same result a third time, or the same error three calls in a
  /// row. Results that differ (a test rerun after an edit) are never flagged.
  struct RepeatGuard {
    private var history: [(code: String, text: String)] = []
    private var errors: [String] = []

    /// `call`: the key is a native call (tool name and arguments), not a program.
    mutating func note(code: String, text: String? = nil, outcome: Outcome, call: Bool = false) -> String? {
      // The same code with another `text` is another program (the text itself is not kept).
      let key = code.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        + (text.map { "\u{0}\($0.utf8.count):\($0.hashValue)" } ?? "")
      let same = history.filter { $0.code == key && $0.text == outcome.text }.count
      history.append((key, outcome.text))
      if history.count > 32 { history.removeFirst(history.count - 32) }
      // The error itself, without the logs before it or the calls that completed.
      let error =
        outcome.isError
        ? (outcome.text.range(of: "Error: ", options: .backwards).map { String(outcome.text[$0.lowerBound...]) }
          ?? outcome.text).components(separatedBy: "\nCompleted before the error:")[0] : ""
      errors = outcome.isError ? (errors.last == error ? errors : []) + [error] : []
      if same >= 2 {
        return "Note: this exact \(call ? "call has now been made" : "program has now run") \(same + 1) times with the same result. Do not \(call ? "make" : "run") it again: use the result above, or change the approach."
      }
      if errors.count >= 3 {
        return "Note: the same error \(errors.count) times in a row. Change the approach instead of retrying the same fix."
      }
      return nil
    }
  }

  /// `plain` (a native tool call): the tool's own text first, then its warnings; no
  /// "Result:" label and no list of completed functions, which are the tool's inner steps.
  static func outcome(_ message: [String: Any], warnings: [String] = [], plain: Bool = false) -> Outcome {
    var parts: [String] = []
    let logs = ((message["logs"] as? [String]) ?? []) + warnings.map { "warning: " + $0 }
    if !plain, !logs.isEmpty { parts.append(logs.joined(separator: "\n")) }
    let error = message["error"] as? String
    if let value = message["value"] as? String { parts.append(plain ? value : "Result:\n" + value) }
    if let error {
      parts.append("Error: " + error)
      // The error names the failed call; list the ones that completed (their effects stay).
      let calls = ((message["calls"] as? [String]) ?? []).filter { !$0.hasPrefix("FAILED ") }
      if !plain, !calls.isEmpty {
        parts.append("Completed before the error: " + calls.suffix(20).joined(separator: ", "))
      }
    }
    if plain, !logs.isEmpty { parts.append(logs.joined(separator: "\n")) }
    if parts.isEmpty {
      parts.append(plain ? "Done." : "Done, without output: `return` a value or console.log to see results.")
    }
    var text = parts.joined(separator: "\n")
    let bytes = Data(text.utf8)
    // A native `bash` result is a command's output, already cut by the host to 8 + 24 KiB
    // around the line naming the file that holds all of it, then its exit code: room for
    // those lines, or this cut (same sizes, same place) would remove exactly that line.
    let maximumText = plain ? maximumText + 2048 : maximumText
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

  /// Newline-delimited messages, each at most 8 MiB (a 1 MiB write of non-ASCII text
  /// grows about sixfold once escaped).
  struct LineBuffer {
    private var data = Data()
    static let maximum = 8 << 20

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
    parent: pid_t, jobs: Jobs = Jobs(), commandFolder: URL? = nil
  ) -> [String: Any] {
    func failure(_ text: String) -> [String: Any] { ["ok": false, "error": text] }
    func text(_ index: Int) -> String? { index < args.count ? args[index] as? String : nil }
    let remaining = deadline - ProcessInfo.processInfo.systemUptime
    guard remaining > 0.3 else { return failure("no time left in this call") }
    guard getppid() == parent else { return failure("the agent stopped") }
    func sandboxed(
      _ command: String, seconds: Double? = nil, writable: Bool = false, limit: Int,
      mode: AgentSandbox.OutputMode = .stop, network: Bool = false
    ) -> AgentSandbox.Result? {
      let budget = min(seconds ?? 5, remaining - 0.2)
      return try? AgentSandbox.execute(
        root: root, command: command, seconds: max(0.1, budget), writable: writable, limit: limit,
        mode: mode, network: network)
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
      // NUL-separated bytes: any file name survives ("\r\n" is one Swift `Character`).
      guard
        let result = sandboxed(
          "find . -type f -not -path './.git/*' -not -path './\(AgentSandbox.Spill.folder)/*' -print0",
          limit: maximumListBytes)
      else { return failure("the file list is unavailable") }
      let raw = result.raw ?? Data(result.output.utf8)
      let full = result.interrupted && raw.count >= maximumListBytes
      guard full || (!result.interrupted && result.status == 0) else {
        return failure("the file list did not finish in time; use search or run")
      }
      var names = raw.split(separator: 0, omittingEmptySubsequences: true)
      if full, raw.last != 0, !names.isEmpty { names.removeLast() }  // cut inside a name
      let paths = names.map { name -> String in
        let path = String(decoding: name, as: UTF8.self)
        return path.hasPrefix("./") ? String(path.dropFirst(2)) : path
      }.sorted()
      var reply: [String: Any] = ["ok": true, "value": paths]
      if full {
        reply["note"] = "stopped at 256 KiB of paths, \(paths.count) shown; use search(regex, path) or run(\"find ...\") for the rest"
      }
      return reply
    case "read":
      guard let path = text(0), AgentSandbox.validPath(path) else {
        return failure("use a relative path inside the project")
      }
      guard
        let result = sandboxed("cat -- \(quote(path))", limit: maximumFileBytes + 1)
      else { return failure("the sandbox is unavailable") }
      if result.interrupted && result.output.utf8.count > maximumFileBytes {
        return failure("\(path) is larger than 1 MiB; use search or run")
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
      // The native `grep` tool's file-name filter: one `--include` per alternative.
      var filters = ""
      if args.count > 2 {
        guard let include = text(2), let globs = includeGlobs(include) else {
          return failure("include must be a file-name glob such as *.py or *.{js,ts} (no folder: put it in path)")
        }
        filters = globs.map { " --include=" + quote($0) }.joined()
      }
      // No pipe: the status line carries grep's own exit status (a pipe reports the last
      // command's), and the output limit stops a large search early.
      guard
        let result = sandboxed(
          "/usr/bin/grep -rnIE --exclude-dir=.git\(filters) -m \(searchPerFile) -e \(quote(pattern)) -- \(quote(path)); printf '\\n\\036\(searchStatus)%d\\n' \"$?\"",
          limit: searchBytes)
      else { return failure("the sandbox is unavailable") }
      return searchReply(
        result.raw ?? Data(result.output.utf8), interrupted: result.interrupted, path: path)
    case "write":
      guard let path = text(0), AgentSandbox.validPath(path), let content = text(1) else {
        return failure("use a relative path inside the project and text content")
      }
      guard content.utf8.count <= maximumFileBytes, !content.contains("\0") else {
        return failure("content must be at most 1 MiB of text")
      }
      let parent = (path as NSString).deletingLastPathComponent
      let temporary =
        (parent.isEmpty ? "" : parent + "/") + ".mlxl3-write-" + UUID().uuidString.prefix(8)
      // printf arguments stay below the 256 KiB command limit, quotes included.
      var chunks: [String] = []
      var current = ""
      var quoted = 0
      for character in content {
        let cost = character == "'" ? 4 : character.utf8.count
        if quoted + cost > 240_000 {
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
      // Syntax errors of the file just written (opencode's diagnostics after an edit): a
      // warning, never a failure. Only a checker's own syntax report counts.
      var checked: [String: Any] = ["ok": true, "value": NSNull()]
      if let check = syntaxCheck(path), remaining > 1.5,
        let result = sandboxed(check, seconds: min(5, remaining - 1), limit: 4096),
        !result.interrupted, result.status == 3, !result.output.isEmpty
      {
        checked["note"] = "syntax error in \(path): " + String(result.output.trimmingCharacters(in: .whitespacesAndNewlines).prefix(600))
      }
      return checked
    case "run":
      guard let command = text(0), !command.isEmpty, command.utf8.count <= AgentSandbox.maximumCommand,
        !command.contains("\0")
      else { return failure("the command must be 1 to \(AgentSandbox.maximumCommand) bytes") }
      // A background job waited for `commandSeconds`: one still running keeps running (opencode
      // v2's shell), and the model gets its id and output so far instead of a killed command.
      let id: Int
      do {
        id = try jobs.start(root: root, command: command, writable: permissions.write, network: permissions.internet, commandFolder: commandFolder)
      } catch { return failure("\(error)") }
      guard let job = jobs.wait(id, seconds: min(Double(permissions.commandSeconds), remaining - 0.3)) else {
        return failure("the job disappeared")
      }
      if job["running"] as? Bool == true {
        return [
          "ok": true,
          "value": [
            "output": job["output"] ?? "", "status": NSNull(), "stopped": false,
            "omitted": job["omitted"] ?? 0, "job": id,
          ],
        ]
      }
      return [
        "ok": true,
        "value": [
          "output": job["output"] ?? "", "status": job["status"] ?? NSNull(),
          "stopped": job["stopped"] ?? false, "omitted": job["omitted"] ?? 0,
        ],
      ]
    case "spawn":
      guard let command = text(0), !command.isEmpty, command.utf8.count <= AgentSandbox.maximumCommand,
        !command.contains("\0")
      else { return failure("the command must be 1 to \(AgentSandbox.maximumCommand) bytes") }
      do {
        return ["ok": true, "value": try jobs.start(root: root, command: command, writable: permissions.write, network: permissions.internet, commandFolder: commandFolder)]
      } catch { return failure("\(error)") }
    case "jobs":
      return ["ok": true, "value": jobs.list()]
    case "tail", "wait", "kill":
      guard args.count >= 1, let number = args[0] as? NSNumber, number.doubleValue.rounded() == number.doubleValue,
        let id = Int(exactly: number.doubleValue)
      else { return failure("the job id must be a number") }
      let value: [String: Any]?
      switch op {
      case "kill": value = jobs.kill(id)
      case "wait":
        var seconds = min(25, remaining - 0.5)
        if args.count >= 2, let asked = args[1] as? NSNumber {
          guard asked.doubleValue.isFinite, asked.doubleValue >= 0 else { return failure("seconds must be 0 or more") }
          seconds = min(asked.doubleValue, remaining - 0.5)
        }
        value = jobs.wait(id, seconds: seconds)
      default:
        var count = 40
        if args.count >= 2, let asked = args[1] as? NSNumber {
          guard asked.intValue > 0 else { return failure("lines must be a positive number") }
          count = min(asked.intValue, 2000)
        }
        guard let job = jobs.describe(id, output: true, read: true) else { value = nil; break }
        let output = (job["output"] as? String) ?? ""
        let lines = output.split(separator: "\n", omittingEmptySubsequences: false)
        return ["ok": true, "value": lines.suffix(count + (output.hasSuffix("\n") ? 1 : 0)).joined(separator: "\n")]
      }
      guard let value else { return failure("there is no job \(id); jobs() lists them") }
      return ["ok": true, "value": value]
    case "fetch":
      guard let address = text(0) else { return failure("the URL must be a string") }
      let body = FileManager.default.temporaryDirectory.appendingPathComponent(
        "mlxl3-fetch-\(UUID().uuidString)")
      defer { try? FileManager.default.removeItem(at: body) }
      do {
        let fetched = try AgentNetwork.get(
          address, hosts: permissions.internetHosts, deadline: deadline - 0.2,
          limit: AgentNetwork.maximumFetchBytes, output: body)
        let data = (try? Data(contentsOf: body)) ?? Data()
        guard let content = AgentNetwork.text(data, type: fetched.type) else {
          return failure("\(fetched.type.isEmpty ? "binary" : fetched.type) content: use download(url, path)")
        }
        return [
          "ok": true,
          "value": ["status": fetched.status, "url": fetched.url, "type": fetched.type, "text": content],
        ]
      } catch { return failure(String(describing: error)) }
    case "download":
      guard let address = text(0), let path = text(1), AgentSandbox.validPath(path) else {
        return failure("use a URL and a relative path inside the project")
      }
      do {
        let fetched = try AgentNetwork.download(
          address, to: path, root: root, hosts: permissions.internetHosts, deadline: deadline - 0.2)
        return ["ok": true, "value": ["status": fetched.status, "url": fetched.url, "bytes": fetched.bytes]]
      } catch { return failure(String(describing: error)) }
    default:
      return failure("unknown function \(op)")
    }
  }

  // MARK: - search

  /// Checker of `path` by extension, ending with status 3 and a short report on a syntax
  /// error: Python, shell scripts, Swift. Other files are not checked.
  static func syntaxCheck(_ path: String) -> String? {
    func quote(_ value: String) -> String { "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'" }
    let file = quote(path)
    switch (path as NSString).pathExtension.lowercased() {
    case "py":
      return "python3 -c " + quote("import ast, sys\ntry:\n    ast.parse(open(sys.argv[1], encoding='utf-8').read(), sys.argv[1])\nexcept SyntaxError as e:\n    print(f'{e.msg} (line {e.lineno})'); sys.exit(3)") + " " + file + " 2>/dev/null"
    case "sh", "bash", "zsh":
      let shell = (path as NSString).pathExtension.lowercased() == "zsh" ? "/bin/zsh" : "/bin/sh"
      return "out=$(\(shell) -n \(file) 2>&1) || { printf '%s\\n' \"$out\" | head -5; exit 3; }"
    case "swift":
      return "out=$(swiftc -parse \(file) 2>&1) || { printf '%s\\n' \"$out\" | grep ': error:' | head -5 | grep . || exit 0; exit 3; }"
    default: return nil
    }
  }

  /// The file-name globs of a `grep` filter: `*.py` → itself, `*.{js,ts}` → `*.js`, `*.ts`
  /// (one brace group; grep's own `--include` has no alternation). Nil: not a file-name glob.
  static func includeGlobs(_ include: String) -> [String]? {
    guard !include.isEmpty, include.utf8.count <= 256, !include.contains("/"), !include.hasPrefix("!"),
      !include.unicodeScalars.contains(where: { $0.value < 32 || $0.value == 127 })
    else { return nil }
    guard let open = include.firstIndex(of: "{") else { return include.contains("}") ? nil : [include] }
    guard let close = include[open...].firstIndex(of: "}") else { return nil }
    let head = include[..<open], tail = include[include.index(after: close)...]
    let parts = include[include.index(after: open)..<close].split(separator: ",", omittingEmptySubsequences: false)
    guard !tail.contains("{"), !tail.contains("}"), !parts.contains(where: { $0.contains("{") }),
      (1...16).contains(parts.count), !parts.contains(where: \.isEmpty)
    else { return nil }
    return parts.map { String(head) + String($0) + String(tail) }
  }

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
  /// `--agent-js-kernel <root> <state file|->`: the persistent kernel of an MCP server. Each
  /// stdin line is a call (`code`, `text`, the permissions, `wall`), each stdout line its outcome; the
  /// end of stdin (the server stopped) ends the kernel and its jobs.
  static func kernelMode(arguments: [String]) -> Never {
    guard arguments.count == 4, let kernel = try? Kernel(root: URL(fileURLWithPath: arguments[2])) else { exit(2) }
    // Jobs a previous kernel left behind when it was killed outright (orphans, still sandboxed,
    // working in this task's copy or home) are stopped first. In a user's folder (a home was
    // assigned) only the home is swept: a sandboxed orphan working there may be another
    // tool's, which no start time ties to this task.
    if let home = try? AgentSandbox.privateHome(for: kernel.root) {
      AgentSandbox.stop(
        [], since: 0,
        inside: AgentSandbox.assignedHome == nil ? [kernel.root.path, home.path] : [home.path])
    }
    // So are JavaScript kernels whose supervisor was killed (they no longer read requests).
    for pid in AgentSandbox.orphans(of: engine) { kill(pid, SIGKILL) }
    let stateFile = arguments[3] == "-" ? nil : URL(fileURLWithPath: arguments[3])
    signal(SIGPIPE, SIG_IGN)
    var buffer = Data()
    while true {
      let chunk = FileHandle.standardInput.availableData
      if chunk.isEmpty { break }
      buffer.append(chunk)
      if buffer.count > 16 << 20 { break }
      while let end = buffer.firstIndex(of: 10) {
        let line = Data(buffer[..<end])
        buffer.removeSubrange(...end)
        var outcome = Outcome(text: "Error: the JavaScript request was invalid.", isError: true)
        if let request = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
          let code = request["code"] as? String, let seconds = request["commandSeconds"] as? Int,
          (1...30).contains(seconds), let wall = request["wall"] as? Double, wall.isFinite,
          (1...longestWall).contains(wall)
        {
          var permissions = AgentPermissions()
          permissions.harness = .javascript
          permissions.read = request["read"] as? Bool == true
          permissions.write = request["write"] as? Bool == true
          permissions.run = request["run"] as? Bool == true
          permissions.commandSeconds = seconds
          permissions.internet = request["internet"] as? Bool == true
          permissions.internetHosts = AgentNetwork.parseHosts(((request["hosts"] as? [String]) ?? []).joined(separator: ","))
          let tool = (request["tool"] as? String).map { (name: $0, arguments: (request["args"] as? [String: Any]) ?? [:]) }
          outcome =
            (try? kernel.run(
              code: code, text: request["text"] as? String, tool: tool, permissions: permissions,
              settings: Settings(wall: wall, stateFile: stateFile,
                commandFolder: (request["commandFolder"] as? String).map { URL(fileURLWithPath: $0) })))
            ?? Outcome(text: "Error: the JavaScript sandbox refused the call.", isError: true)
        }
        guard let data = try? JSONEncoder().encode(outcome),
          (try? FileHandle.standardOutput.write(contentsOf: data + Data([10]))) != nil
        else {
          kernel.shutdown()
          exit(0)
        }
      }
    }
    kernel.shutdown()
    exit(0)
  }

  /// `--agent-js <root> <read|-> <write|-> <run|-> <commandSeconds> <wall> <state file|->
  /// <-|net:host,host>` (Internet off, or on with its allowed hosts).
  static func supervisorMode(arguments: [String]) -> Never {
    guard arguments.count == 10, let seconds = Int(arguments[6]), (1...30).contains(seconds),
      arguments[9] == "-" || arguments[9].hasPrefix("net:"),
      let wall = Double(arguments[7]), wall.isFinite, (1...longestWall).contains(wall)
    else {
      exit(2)
    }
    var permissions = AgentPermissions()
    permissions.read = arguments[3] == "read"
    permissions.write = arguments[4] == "write"
    permissions.run = arguments[5] == "run"
    permissions.commandSeconds = seconds
    permissions.harness = .javascript
    permissions.internet = arguments[9] != "-"
    permissions.internetHosts = AgentNetwork.parseHosts(String(arguments[9].dropFirst(4)))
    var code = Data()
    while code.count <= maximumCode {
      let chunk = FileHandle.standardInput.availableData
      if chunk.isEmpty { break }
      code.append(chunk)
    }
    do {
      let result =
        code.count > maximumCode
        ? tooLarge(code.count)
        : try execute(
          root: URL(fileURLWithPath: arguments[2]), code: String(decoding: code, as: UTF8.self),
          permissions: permissions,
          settings: Settings(
            wall: wall,
            stateFile: arguments[8] == "-" ? nil : URL(fileURLWithPath: arguments[8])))
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
        "setTimeout", "setUnhandledRejectionCallback", "sleepSeconds", "startSamplingProfiler", "structuredClone",
        "testLoopCount", "totalCompileTime", "transferArrayBuffer", "useBigInt32", "version",
        "waitForReport", "waiterListSize", "wasmTestLoopCount", "write", "writeFile", "$vm",
        "$", "agent", "Worker", "WebAssembly"];
      const present = (name) => Object.getOwnPropertyDescriptor(globalThis, name) !== undefined;
      for (const name of shell.concat(Object.getOwnPropertyNames(globalThis))) {
        if (!standard.has(name)) { try { delete globalThis[name]; } catch (_) {} }
      }
      const logs = [], calls = [];
      let logSize = 0, callCount = 0, persist = false;
      // A native tool call (`request.tool`): messages name tools and arguments, not functions.
      let nativeMode = false;
      // `state` goes back to the host with every finished call (also after an error).
      const finish = (fields) => {
        const message = Object.assign({ done: true, logs, calls }, fields);
        if (persist) {
          try {
            const value = globalThis.state;
            if (value === null || typeof value !== "object" || Array.isArray(value)) {
              message.stateError = "`state` must stay an object";
            } else {
              const text = JSON.stringify(value);
              if (text.length > 262144) message.stateError = "`state` is larger than 256 Ki characters";
              else message.state = text;
            }
          } catch (error) { message.stateError = "`state` is not JSON: " + ((error && error.message) || error); }
        }
        send(JSON.stringify(message));
      };
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
      // Lines of a file: "\r\n" files keep their "\r" in the text, not in what is shown.
      const splitLines = (source) => {
        const lines = source.split("\n");
        if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
        return lines;
      };
      const numbered = (lines, first) =>
        lines.map((line, index) => (first + index) + "| " + line.replace(/\r$/, "")).join("\n");
      const lineOf = (source, offset) => source.slice(0, offset).split("\n").length;
      const around = (source, line, count = 1) => {
        const lines = splitLines(source), first = Math.max(1, line - 3);
        return numbered(lines.slice(first - 1, Math.min(lines.length, line + count + 2)), first);
      };
      // Copied from view(): every non-empty line starts with "N| ".
      const unnumber = (text) => {
        const lines = text.split("\n");
        const prefix = /^\s*\d+\| ?/;
        if (!lines.some((line) => prefix.test(line)) || !lines.every((line) => line === "" || prefix.test(line))) return null;
        return lines.map((line) => line.replace(prefix, "")).join("\n");
      };
      // Codex's apply_patch cascade: exact, then lines compared without trailing spaces, then
      // without spaces at both ends, then with typographic punctuation folded to ASCII.
      const fold = (line) => line.trim()
        .replace(/[‘’‚‛′]/g, "'").replace(/[“”„‟″]/g, '"')
        .replace(/[‐-―−]/g, "-").replace(/…/g, "...")
        .replace(/[  -   　]/g, " ");
      const levels = [(line) => line.replace(/\s+$/, ""), (line) => line.trim(), fold];
      const locate = (source, find) => {
        const exact = [];
        for (let at = source.indexOf(find); at !== -1; at = source.indexOf(find, at + find.length)) {
          exact.push({ start: at, end: at + find.length, exact: true });
        }
        if (exact.length) return exact;
        const wanted = find.replace(/\r?\n$/, "").split("\n");
        if (!wanted.some((line) => line.trim())) return [];
        const lines = source.split("\n"), starts = [];
        for (let index = 0, at = 0; index < lines.length; at += lines[index].length + 1, index++) starts.push(at);
        for (const level of levels) {
          const target = wanted.map(level), found = [];
          for (let index = 0; index + target.length <= lines.length; index++) {
            let same = true;
            for (let k = 0; k < target.length && same; k++) same = level(lines[index + k]) === target[k];
            if (same) {
              const last = index + target.length - 1;
              // The match ends before its last line's "\r", which the replacement keeps.
              found.push({ start: starts[index], end: starts[last] + lines[last].replace(/\r$/, "").length });
              index = last;
            }
          }
          if (found.length) return found;
        }
        return [];
      };
      const bigrams = (text) => {
        const counts = new Map();
        for (let index = 0; index + 1 < text.length; index++) {
          const pair = text.slice(index, index + 2);
          counts.set(pair, (counts.get(pair) || 0) + 1);
        }
        return counts;
      };
      const similarity = (a, b) => {
        if (a === b) return 1;
        if (a.length < 2 || b.length < 2) return 0;
        const left = bigrams(a), right = bigrams(b);
        let shared = 0;
        for (const [pair, count] of left) shared += Math.min(count, right.get(pair) || 0);
        return (2 * shared) / (a.length + b.length - 2);
      };
      // The error says how to fix the call: where the text is, or what is closest to it.
      const missing = (path, source, find, replace) => {
        if (replace && source.includes(replace)) {
          return "the text to find is not in " + path + ", but the replacement already is (line "
            + lineOf(source, source.indexOf(replace)) + "): this edit may already be done";
        }
        const wanted = find.split("\n").map((line) => line.trim()).filter((line) => line);
        const lines = splitLines(source);
        let best = -1, score = 0;
        for (let index = 0; index < lines.length && wanted.length; index++) {
          const value = similarity(fold(lines[index]), fold(wanted[0]));
          if (value > score) { score = value; best = index; }
        }
        if (best < 0 || score < 0.5) {
          return "the text to find is not in " + path + " and nothing is close; "
            + (nativeMode ? "read or grep the current text" : "view() or search() the current text");
        }
        const first = Math.max(1, best - 1);
        return "the text to find is not in " + path + ". Closest lines:\n"
          + numbered(lines.slice(first - 1, Math.min(lines.length, best + wanted.length + 2)), first)
          + "\nCopy the exact current text (without the `N| ` prefixes).";
      };
      // One edit of `source` in memory: the new text, and the first changed line.
      const change = (path, source, find, replace, all, name) => {
        text(find, "find", name); text(replace, "replace", name);
        if (!find) throw hostError(Error, name + ": find must not be empty");
        let matches = locate(source, find);
        const plain = matches.length ? null : unnumber(find);
        if (plain !== null) {
          matches = locate(source, plain);
          if (matches.length) replace = unnumber(replace) ?? replace;
        }
        if (!matches.length) throw hostError(Error, name + ": " + missing(path, source, plain ?? find, replace));
        if (matches.length > 1 && !all) {
          const lines = matches.slice(0, 10).map((match) => lineOf(source, match.start));
          throw hostError(Error, name + ": the text to find occurs " + matches.length + " times in " + path
            + " (lines " + lines.join(", ") + (matches.length > 10 ? ", ..." : "")
            + "); include more surrounding text, or "
            + (nativeMode ? "set replace_all to true" : "pass true") + " to replace every occurrence");
        }
        // A line match replaces whole lines: its final newline stays in the file, and a
        // "\r\n" file keeps its line endings.
        let result = source;
        const crlf = source.includes("\r\n");
        for (const match of matches.slice().reverse()) {
          let value = match.exact ? replace : replace.replace(/\r?\n$/, "");
          if (!match.exact && crlf) value = value.replace(/\r?\n/g, "\r\n");
          result = result.slice(0, match.start) + value + result.slice(match.end);
        }
        return { text: result, line: lineOf(source, matches[0].start), count: matches.length };
      };
      const checkJSON = (path, content, name) => {
        if (!/\.json$/i.test(path)) return;
        try { JSON.parse(content); } catch (error) {
          warn(name + ": " + path + " is no longer valid JSON (" + ((error && error.message) || error) + ")");
        }
      };
      // Terminal noise: colors and cursor codes, and progress lines a "\r" overwrote.
      const plainOutput = (output) => output
        .replace(/\x1b\[[0-?]*[ -\/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-Z\\-_]/g, "")
        .replace(/[^\n]*\r(?!\n)/g, "");
      // A job is named by its number, or by the value run() or jobs() gave for it.
      const jobId = (id) => (id !== null && typeof id === "object" ? (id.job ?? id.id) : id);
      // glob: `**/` any folders, `*` and `?` within a name, `{a,b}` alternatives; a pattern
      // without "/" matches file names anywhere (opencode's glob, over list()).
      const globPattern = (pattern) => {
        const literal = (part) => part.replace(/[.+^$()|[\]\\{}*?]/g, "\\$&");
        let source = "";
        for (let index = 0; index < pattern.length; index++) {
          const character = pattern[index];
          if (pattern.startsWith("**/", index)) { source += "(?:.*/)?"; index += 2; }
          else if (pattern.startsWith("**", index)) { source += ".*"; index += 1; }
          else if (character === "*") source += "[^/]*";
          else if (character === "?") source += "[^/]";
          else if (character === "{" && pattern.indexOf("}", index) > index) {
            const end = pattern.indexOf("}", index);
            source += "(?:" + pattern.slice(index + 1, end).split(",").map(literal).join("|") + ")";
            index = end;
          } else source += literal(character);
        }
        return new RegExp("^" + source + "$");
      };
      const api = {
        list: () => host("list", [], "list()"),
        glob: (pattern) => {
          text(pattern, "pattern", "glob");
          const clean = pattern.replace(/^\.\//, "");
          const matcher = globPattern(clean);
          const names = !clean.includes("/");
          const found = host("list", [], label("glob", pattern))
            .filter((path) => matcher.test(names ? path.slice(path.lastIndexOf("/") + 1) : path));
          if (found.length > 500) {
            warn("glob: " + found.length + " matches; the first 500 are returned, narrow the pattern");
            return found.slice(0, 500);
          }
          return found;
        },
        view: (path, offset = 1, limit = 200) => {
          text(path, "path", "view");
          if (!Number.isInteger(offset) || offset < 1 || !Number.isInteger(limit) || limit < 1) {
            throw hostError(TypeError, "view: offset and limit must be positive integers");
          }
          const source = host("read", [path], label("view", path));
          if (source === "") return "(" + path + " is empty)";
          const lines = splitLines(source), total = lines.length;
          if (offset > total) throw hostError(RangeError, "view: " + path + " has " + total + " lines");
          const end = Math.min(total, offset - 1 + Math.min(limit, 2000));
          let shown = numbered(lines.slice(offset - 1, end), offset);
          if (offset > 1 || end < total) {
            shown += "\n(lines " + offset + "-" + end + " of " + total
              + (end < total ? "; view(" + JSON.stringify(path) + ", " + (end + 1) + ") shows more" : "") + ")";
          }
          return shown;
        },
        read: (path) => host("read", [text(path, "path", "read")], label("read", path)),
        search: (pattern, path = ".") => host("search",
          [text(pattern, "regex", "search"), text(path, "path", "search")], label("search", pattern)),
        write: (path, content) => {
          text(path, "path", "write");
          if (content === undefined || content === null) throw hostError(TypeError, "write: text must be a string");
          const body = typeof content === "string" ? content : show(content);
          host("write", [path, body], label("write", path));
          checkJSON(path, body, "write");
        },
        edit: (path, find, replace, all = false) => {
          text(path, "path", "edit");
          const every = all === true || (all !== null && typeof all === "object" && all.all === true);
          const source = host("read", [path], label("read", path));
          const edited = change(path, source, find, replace, every, "edit");
          host("write", [path, edited.text], label("write", path));
          checkJSON(path, edited.text, "edit");
          return { line: edited.line, count: edited.count, snippet: around(edited.text, edited.line, splitLines(replace).length) };
        },
        edits: (path, changes) => {
          text(path, "path", "edits");
          if (!Array.isArray(changes) || !changes.length) throw hostError(TypeError, "edits: changes must be a non-empty array of [find, replace]");
          let source = host("read", [path], label("read", path));
          const lines = [];
          changes.forEach((item, index) => {
            const pair = Array.isArray(item) ? item : item && typeof item === "object" ? [item.find, item.replace] : [];
            try {
              const edited = change(path, source, pair[0], pair[1], false, "edits");
              source = edited.text;
              lines.push(edited.line);
            } catch (error) {
              throw hostError(Error, "edits: change " + (index + 1) + " of " + changes.length + ": "
                + String(error.message).replace(/^edits: /, "") + "; nothing was written");
            }
          });
          host("write", [path, source], label("write", path));
          checkJSON(path, source, "edits");
          return { count: changes.length, lines };
        },
        fetch: (url) => host("fetch", [text(url, "url", "fetch")], label("fetch", url)),
        download: (url, path) => host("download",
          [text(url, "url", "download"), text(path, "path", "download")], label("download", url)),
        // Fixed key order: the status stays readable even when a long output is cut.
        run: (command) => {
          const r = host("run", [text(command, "command", "run")], label("run", command));
          const value = { status: r.status, stopped: r.stopped, omitted: r.omitted, output: plainOutput(r.output) };
          if (r.job !== undefined) value.job = r.job;
          return value;
        },
        spawn: (command) => host("spawn", [text(command, "command", "spawn")], label("spawn", command)),
        jobs: () => host("jobs", [], "jobs()"),
        tail: (id, lines) => plainOutput(host("tail", lines === undefined ? [jobId(id)] : [jobId(id), lines], "tail(" + jobId(id) + ")")),
        wait: (id, seconds) => {
          const job = host("wait", seconds === undefined ? [jobId(id)] : [jobId(id), seconds], "wait(" + jobId(id) + ")");
          if (typeof job.output === "string") job.output = plainOutput(job.output);
          return job;
        },
        kill: (id) => host("kill", [jobId(id)], "kill(" + jobId(id) + ")"),
      };

      // Native tools (`AgentTools`, deepseek-harness' base set): the app's own functions over
      // the same host operations. Their arguments arrive as data, checked by the server; the
      // model writes no code. Each returns the text the model reads.
      const clip = (value) => value.length <= 35000 ? value : value.slice(0, 8000) + "\n[... "
        + (value.length - 33000) + " characters of the result omitted ...]\n" + value.slice(-25000);
      const plural = (count, word) => count + " " + word + (count === 1 ? "" : "s");
      const jobText = (job) => {
        // The cut marker names the JavaScript functions; a tool call names the tools.
        let shown = plainOutput(String(job.output ?? ""))
          .replace(": view(path, offset) or search(regex, path) ...]", ": read it with offset and limit, or grep it ...]");
        if (shown && !shown.endsWith("\n")) shown += "\n";
        return shown;
      };
      const jobState = (job) => job.running
        ? "[job " + job.id + " still running after " + job.seconds + " s]"
        : "[job " + job.id + (job.stopped ? " stopped" : " finished") + " after " + job.seconds + " s; exit code: " + job.status + "]";
      const tools = {
        read: (a) => {
          const offset = a.offset ?? 1, limit = a.limit ?? 200;
          const source = host("read", [a.file_path], "read");
          if (source === "") return "(" + a.file_path + " is empty)";
          const lines = splitLines(source), total = lines.length;
          if (offset > total) throw hostError(Error, "read: " + a.file_path + " has " + plural(total, "line"));
          const end = Math.min(total, offset - 1 + Math.min(limit, 2000));
          let shown = numbered(lines.slice(offset - 1, end), offset);
          if (offset > 1 || end < total) {
            shown += "\n(lines " + offset + "-" + end + " of " + total
              + (end < total ? "; read again with offset " + (end + 1) + " for more" : "") + ")";
          }
          return shown;
        },
        write: (a) => {
          host("write", [a.file_path, a.content], "write");
          checkJSON(a.file_path, a.content, "write");
          return "Wrote " + a.file_path + " (" + plural(a.content === "" ? 0 : splitLines(a.content).length, "line") + ").";
        },
        edit: (a) => {
          if (a.old_string === a.new_string) {
            throw hostError(Error, "edit: old_string and new_string are the same, nothing would change");
          }
          const source = host("read", [a.file_path], "edit");
          const edited = change(a.file_path, source, a.old_string, a.new_string, a.replace_all === true, "edit");
          host("write", [a.file_path, edited.text], "edit");
          checkJSON(a.file_path, edited.text, "edit");
          return "Edited " + a.file_path + ": " + plural(edited.count, "replacement") + ", first at line " + edited.line
            + ".\n" + around(edited.text, edited.line, splitLines(a.new_string).length);
        },
        grep: (a) => {
          const args = [a.pattern, a.path ?? "."];
          if (a.include !== undefined) args.push(a.include);
          const matches = host("search", args, "grep");
          return matches.length
            ? matches.map((match) => match.path + ":" + match.line + ": " + match.text).join("\n") : "No matches.";
        },
        glob: (a) => {
          const pattern = a.pattern.replace(/^\.\//, "");
          const folder = (a.path ?? "").replace(/^\.\//, "").replace(/\/+$/, "");
          const inside = folder && folder !== "." ? folder + "/" : "";
          const matcher = globPattern(pattern), names = !pattern.includes("/");
          const found = host("list", [], "glob")
            .filter((path) => path.startsWith(inside))
            .filter((path) => {
              const rest = path.slice(inside.length);
              return matcher.test(names ? rest.slice(rest.lastIndexOf("/") + 1) : rest);
            });
          if (!found.length) return "No files match.";
          if (found.length > 500) warn("glob: " + found.length + " matches; the first 500 are listed, narrow the pattern");
          return found.slice(0, 500).join("\n");
        },
        bash: (a) => {
          if (a.run_in_background === true) {
            const id = host("spawn", [a.command], "bash");
            return "Started background job " + id + ": job_output with job_id " + id + " reads its output, job_kill stops it.";
          }
          const result = host("run", [a.command], "bash");
          const shown = jobText(result);
          if (result.job !== undefined) {
            return shown + "[still running: it continues as background job " + result.job
              + "; job_output with job_id " + result.job + " reads it (wait: true blocks until it ends), job_kill stops it]";
          }
          return shown + (result.stopped ? "[stopped by the time or output limit] " : "") + "[exit code: " + result.status + "]";
        },
        job_output: (a) => {
          const job = host("wait", a.wait === true ? [a.job_id] : [a.job_id, 0], "job_output");
          return jobText(job) + jobState(job);
        },
        job_list: () => {
          const jobs = host("jobs", [], "job_list");
          return jobs.length ? jobs.map((job) => jobState(job) + " " + job.command).join("\n") : "No background jobs.";
        },
        job_kill: (a) => {
          const job = host("kill", [a.job_id], "job_kill");
          return jobText(job) + jobState(job);
        },
        fetch: (a) => {
          const page = host("fetch", [a.url], "fetch");
          return "HTTP " + page.status + " " + page.type + " " + page.url + "\n\n" + page.text;
        },
        download: (a) => {
          const saved = host("download", [a.url, a.file_path], "download");
          return "Saved " + plural(saved.bytes, "byte") + " to " + a.file_path + " (HTTP " + saved.status + ", " + saved.url + ").";
        },
      };

      // The kernel serves calls one after the other: `$` and everything kept in it survive
      // until the kernel stops (Unharnessed's persistent REPL); `state` comes from the host.
      globalThis.$ = {};
      let textGiven = false;
      const serveNext = () => {
        const line = receive();
        // The host closed the kernel: jsc's readline then gives undefined or "" (never a request).
        if (typeof line !== "string" || line === "") return;
        let request;
        try { request = JSON.parse(line); } catch (_) { finish({ error: "invalid request" }); Promise.resolve().then(serveNext); return; }
        logs.length = 0; calls.length = 0; logSize = 0; callCount = 0;
        persist = request.persist === true;
        const saved = request.state;
        globalThis.state = saved && typeof saved === "object" && !Array.isArray(saved) ? saved : {};
        // The functions of this call's permissions, and only those.
        for (const name of Object.keys(api)) {
          if (request.api.includes(name)) {
            Object.defineProperty(globalThis, name, { value: api[name], writable: false, configurable: true });
          } else { delete globalThis[name]; }
        }
        // The call's `text` argument: a file's content kept out of the code, so it crosses no
        // JavaScript literal. Writable (a program may assign its own `text`), and removed when
        // the next call gives none.
        if (typeof request.text === "string") {
          Object.defineProperty(globalThis, "text",
            { value: request.text, writable: true, configurable: true, enumerable: true });
          textGiven = true;
        } else if (textGiven) { delete globalThis.text; textGiven = false; }
        // The failing line and its text (smolagents): the model sees what to fix.
        let programLines = [];
        const describe = (error) => {
          if (!(error instanceof Error)) return "uncaught " + show(error);
          let line = error.name === "Error" ? error.message : error.name + ": " + error.message;
          if (!error.host && typeof error.line === "number" && error.line > 2 && !error.sourceURL) {
            const source = (programLines[error.line - 3] || "").trim();
            line += " (line " + (error.line - 2) + (source ? ": " + (source.length > 120 ? source.slice(0, 117) + "..." : source) : "") + ")";
          }
          return line;
        };
        // A native tool call: no code to compile, one finished message.
        nativeMode = typeof request.tool === "string";
        if (nativeMode) {
          let reply;
          try {
            if (!Object.prototype.hasOwnProperty.call(tools, request.tool)) {
              throw hostError(Error, "there is no tool named " + request.tool);
            }
            const given = request.args;
            reply = { value: clip(String(tools[request.tool](given && typeof given === "object" ? given : {}))) };
          } catch (error) { reply = { error: describe(error) }; }
          finish(reply);
          Promise.resolve().then(serveNext);
          return;
        }
        // A program that does not compile. jsc counts its lines after the two-line function
        // header and names the prelude as their source, so `describe` would drop the line. A
        // long literal that ended early (a backtick inside a file's content) is the usual
        // cause when the program writes a file: the remedy is the `text` argument.
        const describeSyntax = (error, source) => {
          if (!(error instanceof SyntaxError)) return describe(error);
          let line = "SyntaxError: " + error.message;
          const lines = source.split("\n");
          const at = typeof error.line === "number" ? error.line - 2 : 0;
          if (at >= 1 && at <= lines.length) {
            const shown = lines[at - 1].trim();
            line += " (line " + at + (shown ? ": " + (shown.length > 120 ? shown.slice(0, 117) + "..." : shown) : "") + ")";
          } else if (at > lines.length) {
            line += " (at the end of the code: a string, a template literal or a bracket is not closed)";
          }
          if (request.api.includes("write") && typeof request.text !== "string" && /\bwrite\s*\(/.test(source)) {
            line += "\nNothing was run. If this code holds a file's content, send that content in the `text` argument"
              + " instead and call write(path, text): there it needs no escaping.";
          }
          return line;
        };
        // Small models often fence their code, send it with literal "\n" escapes or as
        // {"code": ...}, define a function they never call, or write a lone expression: each
        // is run as meant.
        let code = String(request.code);
        if (/^\s*\{\s*"code"\s*:/.test(code)) {
          try { const parsed = JSON.parse(code); if (typeof parsed.code === "string") code = parsed.code; } catch (_) {}
        }
        const fenced = /^\s*```[\w+-]*[ \t]*\n([\s\S]*?)\n?```\s*$/.exec(code);
        // Only the code inside the fences: the fences themselves compile (tagged templates)
        // and would hide a syntax error of that code behind a TypeError.
        const bases = fenced ? [fenced[1]] : [code];
        for (const source of bases.slice()) {
          if (!source.includes("\n") && source.includes("\\n")) {
            bases.push(source.replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\"/g, "\""));
          }
        }
        const called = (source) => {
          const declared = /^\s*(?:async\s+function\s*\*?|function\s*\*?)\s*([A-Za-z_$][\w$]*)\s*\(\s*\)/.exec(source)
            || /^\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?(?:\(\s*\)|function\s*\(\s*\))\s*(?:=>)?/.exec(source);
          if (declared) {
            const uses = source.match(new RegExp("(^|[^\\w$.])" + declared[1].replace(/\$/g, "\\$") + "(?![\\w$])", "g")) || [];
            if (uses.length === 1) return source + "\n;return await " + declared[1] + "();";
          }
          if (/^\s*\(\s*(?:async\s*)?(?:\(\s*\)\s*=>|function\s*\w*\s*\(\s*\))[\s\S]*\)\s*\(\s*\)\s*;?\s*$/.test(source)) {
            return "return await " + source.trim().replace(/;$/, "") + ";";
          }
          return null;
        };
        // A lone expression is its own result (opencode's code mode, a REPL's last value).
        const expression = (source) => "return (" + source.trim().replace(/;\s*$/, "") + "\n);";
        const candidates = [];
        for (const source of bases) {
          const call = called(source);
          if (call !== null) candidates.push(call);
          if (!/^\s*return\b/.test(source)) candidates.push(expression(source));
          candidates.push(source);
        }
        let program, firstError;
        for (const source of candidates) {
          try { program = new AsyncFunction(source); programLines = source.split("\n"); break; } catch (error) {
            // The model's own text explains a syntax error best, not a rewritten one.
            if (!firstError && source === bases[0]) { firstError = error; }
          }
        }
        if (!program) {
          finish({ error: describeSyntax(firstError, bases[0]) });
          Promise.resolve().then(serveNext);
          return;
        }
        program().then(
          (value) => {
            if (value === undefined) finish({});
            else {
              const text = show(value);
              finish({ value: text.length <= 33000 ? text : text.slice(0, 8000) + "\n[... " + (text.length - 33000)
                + " characters of the result omitted ...]\n" + text.slice(-25000) });
            }
            serveNext();
          },
          (error) => { finish({ error: describe(error) }); serveNext(); });
      };
      serveNext();
    })();
    """#
}
