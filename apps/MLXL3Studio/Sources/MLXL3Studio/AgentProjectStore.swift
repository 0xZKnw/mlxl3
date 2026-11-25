import CryptoKit
import Darwin
import Foundation

struct AgentProjectState: Codable, Sendable {
  var source: String?
  var baseline: String?
  var originalHashes: [String: String] = [:]
  // Optional for pre-checkpoint metadata. Never infer historical state from today's files.
  var messageCheckpoints: [String: String]?
  // Earlier turns must still fork as independent tasks without access to the attached source.
  var messagesBeforeAttachment: Set<String>?
  var activeCommandTurn: UUID?
  var commandReceipts: [String: AgentCommandSummary]?
  var permissions = AgentPermissions()
  var skippedFiles = 0
}

struct AgentCheckpoint: Identifiable, Sendable, Hashable {
  let id: String
  let date: Date
  let title: String
}

struct AgentCommandSummary: Codable, Equatable, Sendable {
  var commands: [String: Int] = [:]
  var tests: [String: Int] = [:]
  var isEmpty: Bool { commands.isEmpty && tests.isEmpty }

  init() {}
  init(folder: URL) throws {
    for receipt in try AgentCommandReceipt.read(from: folder) {
      if receipt.testCommand { tests[receipt.outcome.rawValue, default: 0] += 1 }
      else { commands[receipt.outcome.rawValue, default: 0] += 1 }
    }
  }
}

struct AgentTurnResult: Equatable, Sendable {
  struct File: Equatable, Sendable {
    let path: String
    let added: Bool
  }
  let before: String
  let after: String
  let files: [File]
  let commands: AgentCommandSummary
  let newDocument: String?
}

struct AgentProjectStore: Sendable {
  private static let mutation = NSRecursiveLock()
  let workspace: URL
  /// History, `project.json` and command receipts of the task. Beside the task's own copy,
  /// or the folder given at init when the task works in a user's folder (`direct`).
  let metadata: URL
  /// The files are a user's folder, edited in place: no copy, no review before applying.
  let direct: Bool
  /// `metadata`: the task's folder in the app's data when `workspace` is a user's folder.
  init(workspace: URL, metadata: URL? = nil) {
    let workspace = URL(fileURLWithPath: workspace.standardizedFileURL.path, isDirectory: true)
    self.workspace = workspace
    self.metadata = metadata.map { URL(fileURLWithPath: $0.standardizedFileURL.path, isDirectory: true) }
      ?? workspace.deletingLastPathComponent().appendingPathComponent("studio")
    direct = metadata != nil
  }

  /// A user's folder an Agent task may work in directly, as its canonical path. Refused:
  /// what is not a folder, `/`, the home folder itself, and any folder that holds or sits
  /// inside `reserved` (the app's own data: histories, private homes, other tasks' copies).
  static func workingFolder(_ url: URL, reserved: [URL]) throws -> URL {
    let refusal = { (text: String) in
      NSError(domain: "MLXL3.Project", code: 6, userInfo: [NSLocalizedDescriptionKey: text])
    }
    var isDirectory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue,
      let root = try? AgentSandbox.canonicalRoot(url)
    else { throw refusal(L("Ce dossier n’existe pas ou n’est pas lisible.", "This folder does not exist or cannot be read.")) }
    let home = (try? AgentSandbox.canonicalRoot(FileManager.default.homeDirectoryForCurrentUser))?.path
    guard root.path != "/", root.path != home, AgentSandbox.validRootPath(root.path) else {
      throw refusal(L("Choisissez un dossier de projet, pas le disque ni votre dossier personnel.",
                      "Choose a project folder, not the disk or your home folder."))
    }
    for folder in reserved {
      // A reserved folder may not exist yet: its existing part is resolved like the root.
      let path = AgentSandbox.canonicalLocation(folder).path
      guard root.path != path, !root.path.hasPrefix(path + "/"), !path.hasPrefix(root.path + "/") else {
        throw refusal(L("Ce dossier contient les données de MLXL3 : choisissez-en un autre.",
                        "This folder holds MLXL3’s own data: choose another one."))
      }
    }
    return root
  }
  private var repository: URL { metadata.appendingPathComponent("history.git") }
  static let maximumFiles = 4_000
  static let maximumBytes = 64 * 1024 * 1024
  static let excludedNames: Set<String> = [
    ".git", ".venv", "venv", "node_modules", ".build", "target", "dist", "build", ".DS_Store",
    ".ssh", ".aws",
    // Agent test material (downloaded repositories, archives): never part of a review.
    scratchFolder,
  ]
  static let scratchFolder = ".mlxl3-scratch"

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

  func checkpoint(messageID: UUID, startingTurn: Bool = false) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    let revision = try checkpoint("Conversation boundary", force: false)
    var state = try state()
    var checkpoints = state.messageCheckpoints ?? [:]
    checkpoints[messageID.uuidString] = revision
    state.messageCheckpoints = checkpoints
    let previous = state.activeCommandTurn
    if startingTurn {
      let folder = AgentCommandReceipt.folder(metadata: metadata, turn: messageID)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700])
      state.activeCommandTurn = messageID
    } else if let previous {
      // A missing or unreadable receipt must not lose the response's file checkpoint.
      let folder = AgentCommandReceipt.folder(metadata: metadata, turn: previous)
      if let receipts = try? AgentCommandSummary(folder: folder), !receipts.isEmpty {
        state.commandReceipts = state.commandReceipts ?? [:]
        state.commandReceipts?[messageID.uuidString] = receipts
      }
      state.activeCommandTurn = nil
    }
    try save(state)
    // A job completing after this boundary must not change a historical result.
    if let previous, previous != state.activeCommandTurn {
      try? FileManager.default.removeItem(at: AgentCommandReceipt.folder(metadata: metadata, turn: previous))
    }
  }

  /// Immutable boundaries for exactly one exchange. Older histories can lack either one.
  func turnResult(user: UUID, assistant: UUID) throws -> AgentTurnResult? {
    let state = try state()
    guard let before = state.messageCheckpoints?[user.uuidString],
      let after = state.messageCheckpoints?[assistant.uuidString] else { return nil }
    try requireRevision(before)
    try requireRevision(after)
    var files: [AgentTurnResult.File] = []
    if before != after {
      let raw = try git(["diff", "--name-status", "--no-renames", "-z", before, after, "--"])
      let fields = String(decoding: raw, as: UTF8.self).split(separator: "\0")
      guard fields.count.isMultiple(of: 2), fields.count <= 4 * Self.maximumFiles else {
        throw CocoaError(.fileReadCorruptFile)
      }
      for index in stride(from: 0, to: fields.count, by: 2) {
        let path = String(fields[index + 1])
        guard Self.validPath(path) else { throw CocoaError(.fileReadCorruptFile) }
        files.append(.init(path: path, added: fields[index] == "A"))
      }
    }
    var newDocument: String?
    if files.count == 1, let file = files.first, file.added,
      try Self.editorText(contents(file.path, revision: after)) != nil {
      newDocument = file.path
    }
    return AgentTurnResult(before: before, after: after, files: files,
      commands: state.commandReceipts?[assistant.uuidString] ?? AgentCommandSummary(), newDocument: newDocument)
  }

  /// Each branch retains its own baseline. Applying one branch makes overlapping
  /// changes in other branches conflict until explicitly reviewed/re-imported.
  func fork(to destination: URL, at messageID: UUID, retaining messages: [UUID]) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    var state = try state()
    guard let revision = state.messageCheckpoints?[messageID.uuidString] else {
      throw NSError(domain: "MLXL3.Project", code: 3, userInfo: [NSLocalizedDescriptionKey:
        "No workspace checkpoint exists for this message. Start a new task instead."])
    }
    try requireRevision(revision)
    // Never rewrites a user's folder to an earlier state; the caller keeps the same folder.
    guard !direct else { throw CocoaError(.fileWriteNoPermission) }
    let branch = AgentProjectStore(workspace: destination)
    guard !FileManager.default.fileExists(atPath: destination.deletingLastPathComponent().path) else {
      throw CocoaError(.fileWriteFileExists)
    }
    do {
      try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700])
      try FileManager.default.createDirectory(at: branch.metadata, withIntermediateDirectories: true,
        attributes: [.posixPermissions: 0o700])
      try FileManager.default.copyItem(at: repository, to: branch.repository)
      _ = try branch.git(["reset", "--hard", revision])
      let retained = Set(messages.map(\.uuidString))
      state.messageCheckpoints = state.messageCheckpoints?.filter { retained.contains($0.key) }
      if state.messagesBeforeAttachment?.contains(messageID.uuidString) == true {
        state.source = nil
        state.baseline = nil
        state.originalHashes = [:]
        state.skippedFiles = 0
        state.messagesBeforeAttachment = nil
      } else {
        state.messagesBeforeAttachment = state.messagesBeforeAttachment?.intersection(retained)
      }
      state.commandReceipts = state.commandReceipts?.filter { retained.contains($0.key) }
      state.activeCommandTurn = nil
      try branch.save(state)
    } catch {
      try? FileManager.default.removeItem(at: destination.deletingLastPathComponent())
      throw error
    }
  }

  private struct FileSnapshot {
    let data: Data
    let mode: mode_t
  }

  private func treeFile(_ path: String, revision: String) throws -> FileSnapshot {
    try requireRevision(revision)
    guard Self.validPath(path) else { throw CocoaError(.fileReadNoPermission) }
    let entry = String(decoding: try git(["ls-tree", "-z", revision, "--", ":(literal)" + path]), as: UTF8.self)
    guard entry.hasPrefix("100644 blob ") || entry.hasPrefix("100755 blob ") else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return FileSnapshot(data: try contents(path, revision: revision), mode: entry.hasPrefix("100755") ? 0o755 : 0o644)
  }

  private static func diskFile(_ path: String, root: URL) throws -> FileSnapshot {
    let url = root.appendingPathComponent(path)
    let data = try HTMLPreview.data(at: url, in: root)
    let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
    guard let mode = attrs[.posixPermissions] as? NSNumber else { throw CocoaError(.fileReadCorruptFile) }
    return FileSnapshot(data: data, mode: mode.uint16Value & 0o777)
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
    // A user's folder is taken as it is: what a copy would refuse (a link, a file or a
    // project too large to version) is left out of the history instead of failing the turn.
    let lenient = direct && folder == nil
    let importing = importing || lenient
    for case let url as URL in enumerator {
      visited += 1
      guard visited <= 20_000 else {
        if lenient {
          skipped += 1
          break
        }
        throw CocoaError(.fileReadTooLarge)
      }
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
      guard paths.count < Self.maximumFiles, bytes + size <= Self.maximumBytes else {
        if lenient {
          skipped += 1
          continue
        }
        throw CocoaError(.fileReadTooLarge)
      }
      do { _ = try HTMLPreview.data(at: url, in: root) } catch {
        guard lenient else { throw error }
        skipped += 1
        continue
      }
      bytes += size
      paths.append(path)
    }
    return (paths.sorted(), skipped)
  }

  func importProject(_ source: URL) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    // A task in a user's folder has no copy to import into.
    guard !direct else { throw CocoaError(.fileWriteNoPermission) }
    let root = try AgentSandbox.canonicalRoot(source)
    // realpath requires an existing item; new tasks may have several missing parents.
    // Resolve both sides identically, including a symlinked Application Support directory.
    var ancestor = workspace.deletingLastPathComponent()
    var missing: [String] = []
    while !FileManager.default.fileExists(atPath: ancestor.path) {
      missing.append(ancestor.lastPathComponent)
      ancestor.deleteLastPathComponent()
    }
    let task = try missing.reversed().reduce(AgentSandbox.canonicalRoot(ancestor)) { $0.appendingPathComponent($1) }
    let home = try AgentSandbox.canonicalRoot(FileManager.default.homeDirectoryForCurrentUser)
    guard root.path != "/", root.path != home.path,
      root.path != task.path, !task.path.hasPrefix(root.path + "/"), !root.path.hasPrefix(task.path + "/")
    else { throw CocoaError(.fileReadNoPermission) }
    var state = try state()
    guard state.source == nil else {
      throw NSError(domain: "MLXL3.Project", code: 4, userInfo: [NSLocalizedDescriptionKey:
        L("Un dossier est déjà rattaché à cette tâche.", "A folder is already attached to this task.")])
    }
    let inventory = try files(in: root, importing: true)
    if inventory.paths.isEmpty {
      // The enumerator can produce no entries for a file or an unreadable directory.
      // A newly created, readable project folder is a valid starting point.
      _ = try FileManager.default.contentsOfDirectory(atPath: root.path)
    }
    try FileManager.default.createDirectory(
      at: workspace, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    guard try AgentSandbox.canonicalRoot(workspace).path == task.appendingPathComponent(workspace.lastPathComponent).path
    else { throw CocoaError(.fileWriteNoPermission) }
    let existing = try files().paths
    let previous = try versions().first?.id
    state.messagesBeforeAttachment = Set(state.messageCheckpoints?.keys.map { $0 } ?? [])
    state.source = root.path
    state.skippedFiles = inventory.skipped
    var installed: [String] = []
    var createdDirectories: [String] = []
    var checkpointStarted = false
    do {
      for path in inventory.paths {
        let file = try Self.diskFile(path, root: root)
        // Never overwrite a task file, including one created since the inventory.
        try Self.write(file.data, path: path, root: workspace, mode: file.mode, replacing: false) {
          createdDirectories.append($0)
        }
        installed.append(path)
        state.originalHashes[path] = Self.digest(file.data)
      }
      checkpointStarted = true
      // Only source files belong to the original project's baseline. Existing task files
      // remain visible as additions in the review before anything is applied to the source.
      state.baseline = try checkpoint("Project imported", includedPaths: inventory.paths)
      if !existing.isEmpty { _ = try checkpoint("Attached project and existing task files") }
      try save(state)
    } catch {
      for path in installed.reversed() {
        if let live = try? Self.diskFile(path, root: workspace),
          Self.digest(live.data) == state.originalHashes[path] {
          try Self.remove(path: path, root: workspace)
        }
      }
      for path in createdDirectories.reversed() {
        try Self.remove(path: path, root: workspace, emptyDirectory: true)
      }
      if checkpointStarted && FileManager.default.fileExists(atPath: repository.path) {
        if let previous { _ = try git(["update-ref", "HEAD", previous]) }
        else { try FileManager.default.removeItem(at: repository) }
      }
      throw error
    }
  }

  /// Anchor every directory to descriptors; an Agent-created symlink cannot redirect a write.
  static func write(_ data: Data, path: String, root: URL, mode: mode_t? = nil, replacing: Bool = true,
    onCreateDirectory: ((String) -> Void)? = nil) throws {
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
    for (index, part) in parts.dropLast().enumerated() {
      if mkdirat(descriptor, part, 0o755) == 0 {
        onCreateDirectory?(parts[...index].joined(separator: "/"))
      } else if errno != EEXIST { throw POSIXError(.EACCES) }
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
    if replacing {
      guard renameat(descriptor, temporary, descriptor, parts.last!) == 0 else { throw POSIXError(.EACCES) }
    } else if linkat(descriptor, temporary, descriptor, parts.last!, 0) != 0 {
      if errno == EEXIST {
        throw NSError(domain: "MLXL3.Project", code: 5, userInfo: [NSLocalizedDescriptionKey:
          L("Un fichier de la tâche existe déjà à cet emplacement : ", "A task file already exists at this path: ") + path])
      }
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
    try checkpoint(title, force: force, includedPaths: nil)
  }

  private func checkpoint(_ title: String, force: Bool = true, includedPaths: [String]?) throws -> String {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    try FileManager.default.createDirectory(
      at: workspace, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    try FileManager.default.createDirectory(
      at: metadata, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    var inventory = try files()
    if let includedPaths { inventory.paths = includedPaths }
    if !FileManager.default.fileExists(atPath: repository.path) {
      _ = try git(["init", "--bare", repository.path])
    }
    // Checkpoints and ZIPs preserve bytes; project attributes cannot filter or hide files.
    let info = repository.appendingPathComponent("info")
    try FileManager.default.createDirectory(at: info, withIntermediateDirectories: true)
    try Data(
      "* -text -eol -ident -filter -working-tree-encoding -export-ignore -export-subst !diff\n".utf8
    )
    .write(to: info.appendingPathComponent("attributes"), options: .atomic)
    _ = try git(["read-tree", "--empty"])
    // Explicit paths prevent .gitignore or an Agent's exclusions from hiding a changed file.
    for start in stride(from: 0, to: inventory.paths.count, by: 100) {
      _ = try git(
        ["add", "-f", "--"]
          + inventory.paths[start..<min(start + 100, inventory.paths.count)].map { ":(literal)" + $0 })
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

  static func editorText(_ bytes: Data) -> String? {
    bytes.contains(0) ? nil : String(data: bytes, encoding: .utf8)
  }

  func contents(_ path: String, revision: String) throws -> Data {
    try requireRevision(revision)
    guard Self.validPath(path) else { throw CocoaError(.fileReadNoPermission) }
    let data = try git(["show", "\(revision):\(path)"])
    guard data.count <= HTMLPreview.maximumBytes else { throw CocoaError(.fileReadTooLarge) }
    return data
  }

  func paths(at revision: String) throws -> [String] {
    try requireRevision(revision)
    let paths = String(
      decoding: try git(["ls-tree", "-r", "--name-only", "-z", revision]), as: UTF8.self
    )
    .split(separator: "\0").map(String.init)
    guard paths.count <= Self.maximumFiles, paths.allSatisfy(Self.validPath) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return paths
  }

  func saveFile(_ path: String, text: String, revision: String) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    guard try versions().first?.id == revision,
      try HTMLPreview.data(at: workspace.appendingPathComponent(path), in: workspace)
        == contents(path, revision: revision)
    else { throw Self.changedWorkspace() }
    try Self.write(Data(text.utf8), path: path, root: workspace)
    _ = try checkpoint("Edited \(path)")
  }

  private static func changedWorkspace() -> NSError {
    NSError(
      domain: "MLXL3.Project", code: 2,
      userInfo: [
        NSLocalizedDescriptionKey: L(
          "Les fichiers ont changé. Actualisez et examinez la nouvelle version avant de continuer.",
          "Files changed. Refresh and review the new version before continuing.")
      ])
  }

  func diff(from before: String, to after: String, path: String? = nil) throws -> String {
    try requireRevision(before)
    try requireRevision(after)
    if let path, !Self.validPath(path) { throw CocoaError(.fileReadNoPermission) }
    return String(
      decoding: try git(
        [
          "diff", "--no-ext-diff", "--no-textconv", "--unified=3", before, after, "--",
        ] + (path.map { [":(literal)" + $0] } ?? [])), as: UTF8.self)
  }

  func apply(reviewedRevision: String) throws {
    Self.mutation.lock()
    defer { Self.mutation.unlock() }
    var state = try state()
    guard !direct, let source = state.source, let baseline = state.baseline,
      let latest = try versions().first?.id
    else { throw CocoaError(.fileWriteNoPermission) }
    let root = URL(fileURLWithPath: source)
    guard latest == reviewedRevision, try Set(files().paths) == Set(paths(at: latest)),
      try git(["diff", "--no-ext-diff", "--no-textconv", "--name-only", latest, "--"]).isEmpty
    else { throw Self.changedWorkspace() }
    let reviewedPaths = Set(try paths(at: latest))
    var fields = String(
      decoding: try git(["diff", "--find-renames", "--name-status", "-z", baseline, latest, "--"]), as: UTF8.self
    ).split(separator: "\0").makeIterator()
    var changes: [String] = []
    var renamedFrom: [String: String] = [:]
    while let status = fields.next() {
      guard let path = fields.next() else { throw CocoaError(.fileReadCorruptFile) }
      changes.append(String(path))
      if status.first == "R" {
        guard let destination = fields.next() else { throw CocoaError(.fileReadCorruptFile) }
        changes.append(String(destination))
        renamedFrom[String(destination)] = String(path)
      }
    }
    // Check the whole change set before writing any original file.
    var replacements: [String: FileSnapshot] = [:]
    var originals: [String: FileSnapshot] = [:]
    let baselinePaths = Set(try paths(at: baseline))
    for path in changes {
      guard Self.validPath(path) else { throw CocoaError(.fileWriteInvalidFileName) }
      let url = root.appendingPathComponent(path)
      let old =
        FileManager.default.fileExists(atPath: url.path)
        ? try Self.diskFile(path, root: root) : nil
      let current = old.map { Self.digest($0.data) }
      let baselineFile = try baselinePaths.contains(path) ? treeFile(path, revision: baseline) : nil
      guard current == state.originalHashes[path],
        old.map({ $0.mode & 0o100 }) == baselineFile.map({ $0.mode & 0o100 }) else {
        throw NSError(
          domain: "MLXL3.Project", code: 1,
          userInfo: [
            NSLocalizedDescriptionKey: "Conflict: \(path). The original changed after import."
          ])
      }
      originals[path] = old
    }
    for path in changes where reviewedPaths.contains(path) {
      let reviewed = try treeFile(path, revision: latest)
      let live = try Self.diskFile(path, root: workspace)
      guard reviewed.data == live.data, reviewed.mode & 0o100 == live.mode & 0o100
      else { throw Self.changedWorkspace() }
      var mode = reviewed.mode & live.mode
      if let original = originals[renamedFrom[path] ?? path] {
        // Git tracks only the owner's execute bit. Preserve current source permissions
        // across edits and renames, including restrictions applied after import.
        mode = original.mode
        if reviewed.mode & 0o100 != mode & 0o100 {
          if reviewed.mode & 0o100 == 0 {
            mode &= ~0o111
          } else {
            // Grant execution to group/others only if they can still read the original.
            let executeForCurrentReaders = (mode & 0o444) >> 2
            mode |= live.mode & (0o100 | executeForCurrentReaders)
          }
        }
      }
      replacements[path] = FileSnapshot(data: reviewed.data, mode: mode)
    }
    var applied: [String] = []
    do {
      for path in changes {
        if let file = replacements[path] {
          try Self.write(file.data, path: path, root: root, mode: file.mode)
          state.originalHashes[path] = Self.digest(file.data)
        } else {
          try Self.remove(path: path, root: root)
          state.originalHashes[path] = nil
        }
        applied.append(path)
      }
      state.baseline = latest
      try save(state)
    } catch {
      for path in applied.reversed() {
        if let original = originals[path] {
          try Self.write(original.data, path: path, root: root, mode: original.mode)
        } else {
          try Self.remove(path: path, root: root)
        }
      }
      throw error
    }
  }

  private static func remove(path: String, root: URL, emptyDirectory: Bool = false) throws {
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
    if unlinkat(descriptor, parts.last!, emptyDirectory ? AT_REMOVEDIR : 0) != 0 {
      if emptyDirectory && (errno == ENOTEMPTY || errno == ENOENT) { return }
      throw POSIXError(.EACCES)
    }
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
