import CryptoKit
import Darwin
import Foundation

struct AgentProjectState: Codable, Sendable {
  var source: String?
  var baseline: String?
  var originalHashes: [String: String] = [:]
  var permissions = AgentPermissions()
  var skippedFiles = 0
}

struct AgentCheckpoint: Identifiable, Sendable, Hashable {
  let id: String
  let date: Date
  let title: String
}

struct AgentProjectStore: Sendable {
  private static let mutation = NSRecursiveLock()
  let workspace: URL
  init(workspace: URL) {
    self.workspace = URL(fileURLWithPath: workspace.standardizedFileURL.path, isDirectory: true)
  }
  var metadata: URL { workspace.deletingLastPathComponent().appendingPathComponent("studio") }
  private var repository: URL { metadata.appendingPathComponent("history.git") }
  static let maximumFiles = 4_000
  static let maximumBytes = 64 * 1024 * 1024
  static let excludedNames: Set<String> = [
    ".git", ".venv", "venv", "node_modules", ".build", "target", "dist", "build", ".DS_Store",
    ".ssh", ".aws",
  ]

  func state() throws -> AgentProjectState {
    let url = metadata.appendingPathComponent("project.json")
    guard FileManager.default.fileExists(atPath: url.path) else { return AgentProjectState() }
    return try JSONDecoder().decode(AgentProjectState.self, from: Data(contentsOf: url))
  }

  func save(_ state: AgentProjectState) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    guard (1...30).contains(state.permissions.commandSeconds) else {
      throw CocoaError(.fileWriteInvalidFileName)
    }
    try FileManager.default.createDirectory(
      at: metadata, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    try JSONEncoder().encode(state).write(
      to: metadata.appendingPathComponent("project.json"), options: .atomic)
  }

  func ensureState() throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    if !FileManager.default.fileExists(atPath: metadata.appendingPathComponent("project.json").path)
    {
      try save(AgentProjectState())
    }
  }

  func setPermissions(_ permissions: AgentPermissions) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    var state = try state()
    state.permissions = permissions
    try save(state)
  }

  static func validPath(_ path: String) -> Bool {
    !path.isEmpty && path.utf8.count <= 1024 && !path.hasPrefix("/")
      && !path.unicodeScalars.contains { $0.value < 32 || $0.value == 127 }
      && path.split(separator: "/", omittingEmptySubsequences: false).allSatisfy {
        !$0.isEmpty && $0 != "." && $0 != ".." && $0 != ".git"
      }
  }

  static func digest(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  func files(in folder: URL? = nil, importing: Bool = false) throws -> (
    paths: [String], skipped: Int
  ) {
    let root = try AgentSandbox.canonicalRoot(folder ?? workspace)
    guard
      let enumerator = FileManager.default.enumerator(
        at: root,
        includingPropertiesForKeys: [
          .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
        ])
    else { throw CocoaError(.fileReadUnknown) }
    var paths: [String] = []
    var skipped = 0
    var bytes = 0
    var visited = 0
    for case let url as URL in enumerator {
      visited += 1
      guard visited <= 20_000 else { throw CocoaError(.fileReadTooLarge) }
      let name = url.lastPathComponent
      if Self.excludedNames.contains(name) || name == ".env"
        || name.hasPrefix(".env.") && name != ".env.example"
      {
        enumerator.skipDescendants()
        skipped += 1
        continue
      }
      let values = try url.resourceValues(forKeys: [
        .isDirectoryKey, .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
      ])
      if values.isSymbolicLink == true {
        enumerator.skipDescendants()
        if importing {
          skipped += 1
          continue
        }
        throw CocoaError(.fileReadNoPermission)
      }
      if values.isDirectory == true { continue }
      let path = String(url.path.dropFirst(root.path.count + 1))
      guard Self.validPath(path), values.isRegularFile == true,
        let size = values.fileSize, size <= HTMLPreview.maximumBytes
      else {
        if importing {
          skipped += 1
          continue
        }
        throw CocoaError(.fileReadTooLarge)
      }
      bytes += size
      guard paths.count < Self.maximumFiles, bytes <= Self.maximumBytes else {
        throw CocoaError(.fileReadTooLarge)
      }
      _ = try HTMLPreview.data(at: url, in: root)
      paths.append(path)
    }
    return (paths.sorted(), skipped)
  }

  func importProject(_ source: URL) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    let root = try AgentSandbox.canonicalRoot(source)
    guard root.path != "/",
      root != FileManager.default.homeDirectoryForCurrentUser.resolvingSymlinksInPath(),
      !workspace.path.hasPrefix(root.path + "/"), !root.path.hasPrefix(workspace.path + "/")
    else { throw CocoaError(.fileReadNoPermission) }
    try FileManager.default.createDirectory(
      at: workspace, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    guard try FileManager.default.contentsOfDirectory(atPath: workspace.path).isEmpty else {
      throw CocoaError(.fileWriteFileExists)
    }
    let inventory = try files(in: root, importing: true)
    guard !inventory.paths.isEmpty else { throw CocoaError(.fileReadUnknown) }
    var state = try state()
    state.source = root.path
    state.skippedFiles = inventory.skipped
    for path in inventory.paths {
      let data = try HTMLPreview.data(at: root.appendingPathComponent(path), in: root)
      let mode =
        (try FileManager.default.attributesOfItem(atPath: root.appendingPathComponent(path).path)[
          .posixPermissions] as? NSNumber)?.uint16Value
      try Self.write(data, path: path, root: workspace, mode: mode)
      state.originalHashes[path] = Self.digest(data)
    }
    try save(state)
    state.baseline = try checkpoint("Project imported")
    try save(state)
  }

  /// Anchor every directory to descriptors; an Agent-created symlink cannot redirect a write.
  static func write(_ data: Data, path: String, root: URL, mode: mode_t? = nil) throws {
    mutation.lock()
    defer { mutation.unlock() }
    guard validPath(path), data.count <= HTMLPreview.maximumBytes else {
      throw CocoaError(.fileWriteInvalidFileName)
    }
    let root = try AgentSandbox.canonicalRoot(root)
    var descriptor = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    guard descriptor >= 0 else { throw CocoaError(.fileWriteNoPermission) }
    defer { close(descriptor) }
    let parts = path.split(separator: "/").map(String.init)
    for part in parts.dropLast() {
      if mkdirat(descriptor, part, 0o755) != 0 && errno != EEXIST { throw POSIXError(.EACCES) }
      let next = openat(descriptor, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
      guard next >= 0 else { throw CocoaError(.fileWriteNoPermission) }
      close(descriptor)
      descriptor = next
    }
    let temporary = ".mlxl3-\(UUID().uuidString).tmp"
    let file = openat(
      descriptor, temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o644)
    guard file >= 0 else { throw CocoaError(.fileWriteNoPermission) }
    defer {
      close(file)
      unlinkat(descriptor, temporary, 0)
    }
    var previous = stat()
    if let mode {
      guard fchmod(file, mode & 0o777) == 0 else { throw POSIXError(.EACCES) }
    } else if fstatat(descriptor, parts.last!, &previous, AT_SYMLINK_NOFOLLOW) == 0,
      previous.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG)
    {
      guard fchmod(file, previous.st_mode & 0o777) == 0 else { throw POSIXError(.EACCES) }
    }
    let handle = FileHandle(fileDescriptor: file, closeOnDealloc: false)
    try handle.write(contentsOf: data)
    try handle.synchronize()
    guard renameat(descriptor, temporary, descriptor, parts.last!) == 0 else {
      throw POSIXError(.EACCES)
    }
  }

  private func git(_ arguments: [String]) throws -> Data {
    let process = Process()
    let output = Pipe()
    let errors = metadata.appendingPathComponent("git-error-\(UUID()).log")
    FileManager.default.createFile(atPath: errors.path, contents: nil)
    let errorHandle = try FileHandle(forWritingTo: errors)
    defer {
      try? errorHandle.close()
      try? FileManager.default.removeItem(at: errors)
    }
    process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
    process.currentDirectoryURL = workspace
    process.arguments =
      [
        "-c", "core.hooksPath=/dev/null", "-c", "core.attributesFile=/dev/null", "-c",
        "core.fsmonitor=false", "-c", "core.autocrlf=false", "-c", "user.name=MLXL3", "-c",
        "user.email=local@mlxl3.invalid",
      ]
      + (arguments.first == "init"
        ? [] : ["--git-dir=\(repository.path)", "--work-tree=\(workspace.path)"]) + arguments
    process.environment = [
      "PATH": "/usr/bin:/bin", "HOME": metadata.path, "GIT_CONFIG_NOSYSTEM": "1",
      "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0",
    ]
    process.standardOutput = output
    process.standardError = errorHandle
    process.standardInput = FileHandle.nullDevice
    try process.run()
    let deadline = DispatchWorkItem { if process.isRunning { process.terminate() } }
    DispatchQueue.global().asyncAfter(deadline: .now() + 20, execute: deadline)
    defer { deadline.cancel() }
    var data = Data()
    while true {
      let bytes = output.fileHandleForReading.availableData
      if bytes.isEmpty { break }
      data.append(bytes)
      if data.count > Self.maximumBytes {
        process.terminate()
        break
      }
    }
    process.waitUntilExit()
    guard process.terminationStatus == 0, data.count <= Self.maximumBytes else {
      throw NSError(
        domain: "MLXL3.Git", code: Int(process.terminationStatus),
        userInfo: [
          NSLocalizedDescriptionKey: String(
            decoding: (try? Data(contentsOf: errors)) ?? Data(), as: UTF8.self)
        ])
    }
    return data
  }

  @discardableResult func checkpoint(_ title: String, force: Bool = true) throws -> String {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    try FileManager.default.createDirectory(
      at: workspace, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    try FileManager.default.createDirectory(
      at: metadata, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    let inventory = try files()
    if !FileManager.default.fileExists(atPath: repository.path) {
      _ = try git(["init", "--bare", repository.path])
    }
    _ = try git(["read-tree", "--empty"])
    // Explicit paths prevent .gitignore or an Agent's exclusions from hiding a changed file.
    for start in stride(from: 0, to: inventory.paths.count, by: 100) {
      _ = try git(
        ["add", "-f", "--"]
          + Array(inventory.paths[start..<min(start + 100, inventory.paths.count)]))
    }
    let indexed = String(decoding: try git(["ls-files", "--stage", "-z"]), as: UTF8.self).split(
      separator: "\0")
    guard indexed.allSatisfy({ $0.hasPrefix("100644 ") || $0.hasPrefix("100755 ") }) else {
      throw CocoaError(.fileReadNoPermission)
    }
    if !force, let previous = try? git(["rev-parse", "HEAD^{tree}"]),
      previous == (try git(["write-tree"]))
    {
      return String(decoding: try git(["rev-parse", "HEAD"]), as: UTF8.self).trimmingCharacters(
        in: .whitespacesAndNewlines)
    }
    _ = try git(["commit", "--allow-empty", "-m", String(title.prefix(100))])
    return String(decoding: try git(["rev-parse", "HEAD"]), as: UTF8.self).trimmingCharacters(
      in: .whitespacesAndNewlines)
  }

  func versions() throws -> [AgentCheckpoint] {
    guard FileManager.default.fileExists(atPath: repository.path) else { return [] }
    return String(decoding: try git(["log", "-100", "--format=%H%x09%ct%x09%s"]), as: UTF8.self)
      .split(separator: "\n").compactMap { line in
        let parts = line.split(separator: "\t", maxSplits: 2).map(String.init)
        guard parts.count == 3, let timestamp = TimeInterval(parts[1]) else { return nil }
        return AgentCheckpoint(
          id: parts[0], date: Date(timeIntervalSince1970: timestamp), title: parts[2])
      }
  }

  private func requireRevision(_ revision: String) throws {
    guard revision.utf8.count == 40,
      revision.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
    else { throw CocoaError(.fileReadCorruptFile) }
  }

  func contents(_ path: String, revision: String) throws -> Data {
    try requireRevision(revision)
    guard Self.validPath(path) else { throw CocoaError(.fileReadNoPermission) }
    let data = try git(["show", "\(revision):\(path)"])
    guard data.count <= HTMLPreview.maximumBytes else { throw CocoaError(.fileReadTooLarge) }
    return data
  }

  func diff(from before: String, to after: String) throws -> String {
    try requireRevision(before)
    try requireRevision(after)
    return String(
      decoding: try git([
        "diff", "--no-ext-diff", "--no-textconv", "--unified=3", before, after, "--",
      ]), as: UTF8.self)
  }

  func apply() throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    var state = try state()
    guard let source = state.source, let baseline = state.baseline,
      let latest = try versions().first?.id
    else { throw CocoaError(.fileWriteNoPermission) }
    let root = URL(fileURLWithPath: source)
    let changes = String(
      decoding: try git(["diff", "--name-only", "-z", baseline, latest, "--"]), as: UTF8.self
    ).split(separator: "\0").map(String.init)
    // Check the whole change set before writing any original file.
    var replacements: [String: Data] = [:]
    var originals: [String: Data] = [:]
    for path in changes {
      guard Self.validPath(path) else { throw CocoaError(.fileWriteInvalidFileName) }
      let url = root.appendingPathComponent(path)
      let old =
        FileManager.default.fileExists(atPath: url.path)
        ? try HTMLPreview.data(at: url, in: root) : nil
      let current = old.map(Self.digest)
      guard current == state.originalHashes[path] else {
        throw NSError(
          domain: "MLXL3.Project", code: 1,
          userInfo: [
            NSLocalizedDescriptionKey: "Conflict: \(path). The original changed after import."
          ])
      }
      let local = workspace.appendingPathComponent(path)
      originals[path] = old
      if FileManager.default.fileExists(atPath: local.path) {
        replacements[path] = try HTMLPreview.data(at: local, in: workspace)
      }
    }
    var applied: [String] = []
    do {
      for path in changes {
        if let data = replacements[path] {
          try Self.write(data, path: path, root: root)
          state.originalHashes[path] = Self.digest(data)
        } else {
          try Self.remove(path: path, root: root)
          state.originalHashes[path] = nil
        }
        applied.append(path)
      }
    } catch {
      for path in applied.reversed() {
        if let original = originals[path] {
          try Self.write(original, path: path, root: root)
        } else {
          try Self.remove(path: path, root: root)
        }
      }
      throw error
    }
    state.baseline = latest
    try save(state)
  }

  private static func remove(path: String, root: URL) throws {
    guard validPath(path) else { throw CocoaError(.fileWriteNoPermission) }
    let root = try AgentSandbox.canonicalRoot(root)
    var descriptor = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    guard descriptor >= 0 else { throw CocoaError(.fileWriteNoPermission) }
    defer { close(descriptor) }
    let parts = path.split(separator: "/").map(String.init)
    for part in parts.dropLast() {
      let next = openat(descriptor, part, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
      guard next >= 0 else { throw CocoaError(.fileWriteNoPermission) }
      close(descriptor)
      descriptor = next
    }
    guard unlinkat(descriptor, parts.last!, 0) == 0 else { throw POSIXError(.EACCES) }
  }

  func restore(_ revision: String) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    try requireRevision(revision)
    _ = try checkpoint("Before restore")
    _ = try git(["restore", "--source", revision, "--worktree", "--", "."])
    _ = try checkpoint("Restored \(revision.prefix(8))")
  }

  func exportZIP(_ destination: URL, revision: String) throws {
    try requireRevision(revision)
    _ = try git(["archive", "--format=zip", "--output", destination.path, revision])
  }

  func materialize(_ revision: String) throws -> URL {
    try requireRevision(revision)
    let destination = metadata.appendingPathComponent("previews/\(revision)")
    let marker = destination.appendingPathComponent(".mlxl3-preview-complete")
    if (try? String(contentsOf: marker, encoding: .utf8)) == revision { return destination }
    let staging = metadata.appendingPathComponent("previews/.\(UUID().uuidString)")
    try FileManager.default.createDirectory(
      at: staging, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    defer { try? FileManager.default.removeItem(at: staging) }
    let paths = String(
      decoding: try git(["ls-tree", "-r", "--name-only", "-z", revision]), as: UTF8.self
    ).split(separator: "\0").map(String.init)
    guard paths.count <= Self.maximumFiles else { throw CocoaError(.fileReadTooLarge) }
    for path in paths {
      try Self.write(contents(path, revision: revision), path: path, root: staging)
    }
    try Data(revision.utf8).write(to: staging.appendingPathComponent(".mlxl3-preview-complete"))
    do { try FileManager.default.moveItem(at: staging, to: destination) } catch {
      guard (try? String(contentsOf: marker, encoding: .utf8)) == revision else { throw error }
    }
    return destination
  }
}
