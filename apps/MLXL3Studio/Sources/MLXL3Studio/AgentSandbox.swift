import Darwin
import Foundation

struct AgentPermissions: Codable, Equatable, Sendable {
  var read = true
  var write = true
  var run = true
  var commandSeconds = 5
}

/// All model-controlled executable text crosses this kernel-enforced boundary.
enum AgentSandbox {
  static let maximumOutput = 32_768
  static let maximumCommand = 16_384

  static func canonicalRoot(_ root: URL) throws -> URL {
    guard let path = realpath(root.path, nil) else { throw POSIXError(.ENOENT) }
    defer { free(path) }
    return URL(fileURLWithPath: String(cString: path), isDirectory: true)
  }

  static func profile(root: URL, writable: Bool = true) throws -> String {
    let path = try canonicalRoot(root).path
    guard path != "/", path.utf8.count < 3000,
      !path.unicodeScalars.contains(where: { $0.value < 32 })
    else { throw CocoaError(.fileWriteInvalidFileName) }
    let literal = path.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(
      of: "\"", with: "\\\"")
    var ancestor = URL(fileURLWithPath: path).deletingLastPathComponent()
    var parents: [String] = []
    while ancestor.path != "/" {
      let escaped = ancestor.path.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(
        of: "\"", with: "\\\"")
      parents.append("(literal \"\(escaped)\")")
      ancestor.deleteLastPathComponent()
    }
    return """
      (version 1)
      (deny default)
      (allow process-exec process-fork)
      (allow syscall-unix)
      ;; Keep descendants in the supervisor's process group, including spawn attributes.
      (deny syscall-unix (syscall-number SYS_setsid) (syscall-number SYS_setpgid)
        (syscall-number SYS_posix_spawn))
      (allow signal (target same-sandbox))
      (allow sysctl-read (sysctl-name-regex #"^hw[.]") (sysctl-name "kern.clockrate")
        (sysctl-name "kern.ostype") (sysctl-name "kern.osrelease") (sysctl-name "kern.osversion")
        (sysctl-name "kern.argmax") (sysctl-name "kern.ngroups")
        (sysctl-name "kern.maxfilesperproc") (sysctl-name "kern.maxprocperuid"))
      (allow file-read-metadata)
      (allow file-read* \(parents.joined(separator: " ")))
      (allow file-read* (literal "/") (subpath "/System/Library") (subpath "/usr/lib")
        (subpath "/usr/share") (subpath "/bin") (subpath "/usr/bin")
        (literal "/dev/null") (literal "/dev/random") (literal "/dev/urandom"))
      (allow file-read* \(writable ? "file-write*" : "") (subpath "\(literal)"))
      (allow file-write-data (literal "/dev/null"))
      """
  }

  struct Result: Codable {
    let output: String
    let status: Int32
    let interrupted: Bool
  }

  /// A separate trusted supervisor survives loss of the MCP transport and reaps its job.
  static func execute(root: URL, command: String, seconds: Double = 5, writable: Bool = true) throws -> Result {
    guard !command.isEmpty, command.utf8.count <= maximumCommand, !command.contains("\0"),
      seconds.isFinite, seconds > 0, seconds <= 30
    else { throw CocoaError(.fileReadCorruptFile) }
    let root = try canonicalRoot(root)
    let policy = try profile(root: root, writable: writable)
    let parent = getppid()
    var descriptors: [Int32] = [0, 0]
    guard pipe(&descriptors) == 0 else { throw POSIXError(.EMFILE) }
    defer {
      close(descriptors[0])
      close(descriptors[1])
    }
    var actions: posix_spawn_file_actions_t?
    var attributes: posix_spawnattr_t?
    posix_spawn_file_actions_init(&actions)
    posix_spawnattr_init(&attributes)
    defer {
      posix_spawn_file_actions_destroy(&actions)
      posix_spawnattr_destroy(&attributes)
    }
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDOUT_FILENO)
    posix_spawn_file_actions_adddup2(&actions, descriptors[1], STDERR_FILENO)
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0)
    posix_spawn_file_actions_addclose(&actions, descriptors[0])
    posix_spawn_file_actions_addclose(&actions, descriptors[1])
    posix_spawn_file_actions_addchdir(&actions, root.path)
    posix_spawnattr_setflags(
      &attributes, Int16(POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_CLOEXEC_DEFAULT))
    posix_spawnattr_setpgroup(&attributes, 0)
    var query: [Int32] = [CTL_KERN, KERN_PROC, KERN_PROC_UID, Int32(getuid())]
    var processBytes = 0
    guard sysctl(&query, 4, nil, &processBytes, nil, 0) == 0 else { throw POSIXError(.EIO) }
    let processLimit = processBytes / MemoryLayout<kinfo_proc>.stride + 16
    let arguments = [
      "/usr/bin/sandbox-exec", "-p", policy, "/bin/sh", "-c",
      "ulimit -t \(Int(ceil(seconds))) && ulimit -f 8192 && ulimit -n 64 && ulimit -u \(processLimit) || exit 125; exec /bin/sh -c \"$1\"",
      "mlxl3-agent", command,
    ]
    let environment = [
      "PATH=/usr/bin:/bin", "HOME=\(root.path)", "TMPDIR=\(root.path)", "LC_ALL=C",
    ]
    let argv = arguments.map { strdup($0) } + [nil]
    let envp = environment.map { strdup($0) } + [nil]
    defer { for value in argv + envp { if let value { free(value) } } }
    var pid: pid_t = 0
    let error = posix_spawn(&pid, arguments[0], &actions, &attributes, argv, envp)
    guard error == 0 else { throw POSIXError(POSIXErrorCode(rawValue: error) ?? .EIO) }
    close(descriptors[1])
    descriptors[1] = -1
    _ = fcntl(descriptors[0], F_SETFL, O_NONBLOCK)
    let deadline = ProcessInfo.processInfo.systemUptime + seconds
    var data = Data()
    var status: Int32 = 0
    var interrupted = false
    var reaped = false
    defer {
      kill(-pid, SIGKILL)
      if !reaped { while waitpid(pid, &status, 0) < 0 && errno == EINTR {} }
    }
    var bytes = [UInt8](repeating: 0, count: 4096)
    while true {
      let count = read(descriptors[0], &bytes, bytes.count)
      if count > 0 {
        data.append(contentsOf: bytes.prefix(count))
        if data.count > maximumOutput {
          data = data.prefix(maximumOutput)
          interrupted = true
          break
        }
      }
      if !reaped {
        let result = waitpid(pid, &status, WNOHANG)
        if result == pid { reaped = true }
      }
      if reaped && count <= 0 { break }
      if getppid() != parent || ProcessInfo.processInfo.systemUptime >= deadline {
        interrupted = true
        break
      }
      var descriptor = pollfd(fd: descriptors[0], events: Int16(POLLIN), revents: 0)
      _ = poll(&descriptor, 1, 20)
    }
    let exitCode = status & 0x7f == 0 ? (status >> 8) & 0xff : 128 + (status & 0x7f)
    return Result(
      output: String(decoding: data, as: UTF8.self), status: exitCode, interrupted: interrupted)
  }

  static func commandMode(root: String, command: String, seconds: Double = 5, writable: Bool = true) -> Never {
    do {
      let result = try execute(root: URL(fileURLWithPath: root), command: command, seconds: seconds, writable: writable)
      let data = try JSONEncoder().encode(result)
      try FileHandle.standardOutput.write(contentsOf: data)
      exit(0)
    } catch { exit(1) }
  }

  static func serve(root: String, permissionsPath: String? = nil) -> Never {
    // The native engine already bounds MCP responses; bound requests here as well.
    var buffer = Data()
    while true {
      let chunk = FileHandle.standardInput.availableData
      if chunk.isEmpty { break }
      buffer.append(chunk)
      if buffer.count > 65_536 { exit(1) }
      while let end = buffer.firstIndex(of: 10) {
        let line = Data(buffer[..<end])
        buffer.removeSubrange(...end)
        guard let request = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
          let method = request["method"] as? String
        else { exit(1) }
        guard let id = request["id"] else { continue }
        let permissions: AgentPermissions
        if let permissionsPath {
          struct Envelope: Decodable { let permissions: AgentPermissions }
          guard let data = try? Data(contentsOf: URL(fileURLWithPath: permissionsPath)), data.count <= 2_000_000,
            let envelope = try? JSONDecoder().decode(Envelope.self, from: data),
            (1...30).contains(envelope.permissions.commandSeconds) else { exit(1) }
          permissions = envelope.permissions
        } else { permissions = AgentPermissions() }
        var response: [String: Any] = ["jsonrpc": "2.0", "id": id]
        switch method {
        case "initialize":
          response["result"] = [
            "protocolVersion": "2025-11-25", "capabilities": ["tools": [:]],
            "serverInfo": ["name": "mlxl3-sandbox", "version": "1"],
          ]
        case "tools/list":
          response["result"] = [
            "tools": ([
              [
                "name": "run",
                "description":
                  "Run a shell command in your private macOS sandbox. No network or personal files. Use relative paths. Deadline: \(permissions.commandSeconds) seconds; output: 32 KiB. Writing is \(permissions.write ? "enabled" : "disabled"). System tools in /usr/bin and /bin only.",
                "inputSchema": [
                  "type": "object",
                  "properties": ["command": ["type": "string", "maxLength": maximumCommand]],
                  "required": ["command"], "additionalProperties": false,
                ],
              ],
              ["name": "list_files", "description": "List files in the private workspace.", "inputSchema": ["type": "object", "properties": [:], "additionalProperties": false]],
              ["name": "read_file", "description": "Read a UTF-8 file using a relative workspace path. Output limit: 32 KiB.", "inputSchema": ["type": "object", "properties": ["path": ["type": "string", "maxLength": 1024]], "required": ["path"], "additionalProperties": false]],
              ["name": "write_file", "description": "Create or replace a UTF-8 file in the private workspace. Parent folders are created. Content limit: 8192 bytes.", "inputSchema": ["type": "object", "properties": ["path": ["type": "string", "maxLength": 1024], "content": ["type": "string", "maxLength": 8192]], "required": ["path", "content"], "additionalProperties": false]]
            ] as [[String: Any]]).filter {
              switch $0["name"] as? String {
              case "run": permissions.run
              case "write_file": permissions.write
              default: permissions.read
              }
            }
          ]
        case "tools/call":
          do {
            guard let params = request["params"] as? [String: Any],
              let name = params["name"] as? String,
              params["arguments"] == nil || params["arguments"] is [String: Any]
            else { throw CocoaError(.fileReadCorruptFile) }
            let args = params["arguments"] as? [String: Any] ?? [:]
            func quote(_ value: String) -> String { "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'" }
            func path() throws -> String {
              guard let path = args["path"] as? String, !path.isEmpty, path.utf8.count <= 1024,
                !path.hasPrefix("/"), !path.unicodeScalars.contains(where: { $0.value < 32 || $0.value == 127 }),
                path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && $0 != ".git" }) else { throw CocoaError(.fileReadNoPermission) }
              return path
            }
            let command: String
            switch name {
            case "run":
              guard permissions.run, let text = args["command"] as? String else { throw CocoaError(.fileReadNoPermission) }
              command = text
            case "list_files":
              guard permissions.read else { throw CocoaError(.fileReadNoPermission) }
              command = "find . -type f -not -path './.git/*' -print"
            case "read_file":
              guard permissions.read else { throw CocoaError(.fileReadNoPermission) }
              command = "cat -- " + quote(try path())
            case "write_file":
              guard permissions.write, let content = args["content"] as? String,
                content.utf8.count <= 8192, !content.contains("\0") else { throw CocoaError(.fileWriteNoPermission) }
              let file = try path(), parent = (file as NSString).deletingLastPathComponent
              command = (parent.isEmpty ? "" : "mkdir -p -- " + quote(parent) + " && ") + "printf '%s' " + quote(content) + " > " + quote(file)
            default: throw CocoaError(.fileReadNoPermission)
            }
            guard !command.isEmpty, command.utf8.count <= maximumCommand, !command.contains("\0") else { throw CocoaError(.fileReadCorruptFile) }
            let supervisor = Process()
            supervisor.executableURL = URL(fileURLWithPath: CommandLine.arguments[0])
            supervisor.arguments = ["--agent-command", root, command, String(permissions.commandSeconds), permissions.write ? "write" : "readonly"]
            supervisor.environment = ["PATH": "/usr/bin:/bin"]
            supervisor.standardInput = FileHandle.nullDevice
            supervisor.standardError = FileHandle.nullDevice
            let output = Pipe()
            supervisor.standardOutput = output
            try supervisor.run()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            supervisor.waitUntilExit()
            guard supervisor.terminationStatus == 0 else {
              throw CocoaError(.executableRuntimeMismatch)
            }
            let result = try JSONDecoder().decode(Result.self, from: data)
            response["result"] = [
              "content": [
                [
                  "type": "text",
                  "text": result.output
                    + (result.interrupted
                      ? "\nCommand stopped: time/output limit or parent exited."
                      : "\nExit status: \(result.status)")
                    ,
                ]
              ],
              "isError": result.interrupted || result.status != 0,
            ]
          } catch {
            response["result"] = [
              "content": [["type": "text", "text": "Sandbox command refused or unavailable."]],
              "isError": true,
            ]
          }
        default: response["error"] = ["code": -32601, "message": "Unknown method"]
        }
        guard var data = try? JSONSerialization.data(withJSONObject: response) else { exit(1) }
        data.append(10)
        do { try FileHandle.standardOutput.write(contentsOf: data) } catch { exit(0) }
      }
    }
    exit(buffer.isEmpty ? 0 : 1)
  }
}
