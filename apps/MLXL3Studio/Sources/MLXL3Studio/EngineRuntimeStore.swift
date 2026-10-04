import Foundation

/// An engine update lives outside the signed application. Activation changes
/// one atomic pointer only after every file and the actual executable pass.
enum EngineRuntimeStore {
  static let protocolVersion = 1
  static let runtimeFiles = Set(["mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"])
  static let maximumFileBytes = 256 * 1_024 * 1_024

  struct FileRecord: Codable, Sendable {
    let size: Int
    let sha256: String
  }

  struct Manifest: Codable, Sendable {
    let schema: Int
    let version: String
    let bridgeProtocol: Int
    let architecture: String
    let minimumMacOS: String
    let minimumAppVersion: String
    let files: [String: FileRecord]

    enum CodingKeys: String, CodingKey {
      case schema, version, architecture, files
      case bridgeProtocol = "bridge_protocol"
      case minimumMacOS = "minimum_macos"
      case minimumAppVersion = "minimum_app_version"
    }

    func validate(
      appVersion: String, expectedVersion: String? = nil,
      osVersion: String = EngineRuntimeStore.osVersion
    ) throws {
      guard schema == 1, bridgeProtocol == protocolVersion, architecture == "arm64",
        let incoming = SemanticVersion(version), version.split(separator: ".").count == 3,
        version.utf8.allSatisfy({ (48...57).contains($0) || $0 == 46 }),
        let minimumApp = SemanticVersion(minimumAppVersion),
        let app = SemanticVersion(appVersion), app >= minimumApp,
        let minimumOS = SemanticVersion(minimumMacOS),
        let os = SemanticVersion(osVersion), os >= minimumOS,
        expectedVersion == nil || version == expectedVersion,
        Set(files.keys) == runtimeFiles,
        files.values.allSatisfy({
          $0.size > 0 && $0.size <= maximumFileBytes
            && UpdateManager.validDigest("sha256:" + $0.sha256)
        }),
        incoming >= SemanticVersion("1.2.0")!
      else {
        throw EngineError.incompatible
      }
    }
  }

  private struct Pointer: Codable { let version: String }

  static var osVersion: String {
    let os = ProcessInfo.processInfo.operatingSystemVersion
    return "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)"
  }

  static var root: URL {
    FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
      .appendingPathComponent("io.mlxl3.desktop/Engines", isDirectory: true)
  }

  static var bundledVersion: String {
    if let resource = Bundle.main.resourceURL,
      let manifest = try? readManifest(resource.appendingPathComponent("runtime"))
    {
      return manifest.version
    }
    return Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
      ?? "1.2.0"
  }

  static func installedVersion() -> String {
    guard
      let executable = resolve(
        appVersion: Bundle.main.object(
          forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.2.0"),
      let manifest = try? readManifest(executable.deletingLastPathComponent())
    else { return bundledVersion }
    return manifest.version
  }

  static func resolve(
    appVersion: String, root: URL = root,
    bundledVersion: String = bundledVersion
  ) -> URL? {
    do {
      let pointer = try readPointer(root.appendingPathComponent("current.json"))
      guard let incoming = SemanticVersion(pointer.version),
        let bundled = SemanticVersion(bundledVersion), incoming >= bundled
      else { return nil }
      let directory = root.appendingPathComponent("v" + pointer.version, isDirectory: true)
      guard
        !FileManager.default.fileExists(atPath: directory.appendingPathComponent("disabled").path)
      else { return nil }
      try verify(directory, appVersion: appVersion, expectedVersion: pointer.version)
      let executable = directory.appendingPathComponent("mlxl3")
      return FileManager.default.isExecutableFile(atPath: executable.path) ? executable : nil
    } catch { return nil }
  }

  /// Called only when a managed executable fails before its first ready event.
  /// Existing engines remain on disk for diagnostics; subsequent loads fall
  /// back to the bundled runtime without another download.
  @discardableResult
  static func reject(_ executable: URL, root: URL = root) -> Bool {
    guard let active = resolve(appVersion: "999.0.0", root: root, bundledVersion: "1.2.0"),
      active.standardizedFileURL == executable.standardizedFileURL
    else { return false }
    do {
      try Data().write(
        to: active.deletingLastPathComponent().appendingPathComponent("disabled"), options: .atomic)
      return true
    } catch { return false }
  }

  static func useBundled(root: URL = root) throws {
    let pointer = root.appendingPathComponent("current.json")
    if FileManager.default.fileExists(atPath: pointer.path) {
      try FileManager.default.removeItem(at: pointer)
    }
  }

  static func readManifest(_ directory: URL) throws -> Manifest {
    let url = directory.appendingPathComponent("engine.json")
    try requireRegularFile(url, maximum: 64 * 1_024)
    return try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: url))
  }

  private static func readPointer(_ url: URL) throws -> Pointer {
    try requireRegularFile(url, maximum: 1_024)
    let pointer = try JSONDecoder().decode(Pointer.self, from: Data(contentsOf: url))
    guard pointer.version.split(separator: ".").count == 3,
      SemanticVersion(pointer.version) != nil,
      pointer.version.utf8.allSatisfy({ (48...57).contains($0) || $0 == 46 })
    else { throw EngineError.incompatible }
    return pointer
  }

  static func verify(_ directory: URL, appVersion: String, expectedVersion: String) throws {
    let info = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
    guard info.isDirectory == true, info.isSymbolicLink != true else {
      throw EngineError.invalidArchive
    }
    let manifest = try readManifest(directory)
    try manifest.validate(appVersion: appVersion, expectedVersion: expectedVersion)
    for (name, record) in manifest.files {
      let file = directory.appendingPathComponent(name)
      try requireRegularFile(file, maximum: maximumFileBytes)
      guard try file.resourceValues(forKeys: [.fileSizeKey]).fileSize == record.size,
        try UpdateManager.sha256(at: file) == record.sha256.lowercased()
      else { throw EngineError.integrity }
    }
  }

  private static func requireRegularFile(_ url: URL, maximum: Int) throws {
    let values = try url.resourceValues(forKeys: [
      .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
    ])
    guard values.isRegularFile == true, values.isSymbolicLink != true,
      let size = values.fileSize, size >= 0, size <= maximum
    else { throw EngineError.invalidArchive }
  }

  /// Avoid archive extraction entirely: read the five exact members to
  /// stdout, then write them to paths we construct. Links, duplicates, extra
  /// files and traversal names are rejected before any member is materialized.
  static func stage(archive: URL, directory: URL, releaseVersion: String, appVersion: String) throws
  {
    let allowed = runtimeFiles.union(["engine.json"])
    let listing = try UpdateManager.runProcess(
      "/usr/bin/tar", arguments: ["-tzf", archive.path], outputLimit: 64 * 1_024)
    guard let text = String(data: listing, encoding: .utf8) else {
      throw EngineError.invalidArchive
    }
    let members = text.split(separator: "\n").map(String.init)
    guard members.count == allowed.count, Set(members) == allowed else {
      throw EngineError.invalidArchive
    }
    let verbose = try UpdateManager.runProcess(
      "/usr/bin/tar", arguments: ["-tvzf", archive.path], outputLimit: 64 * 1_024)
    guard let types = String(data: verbose, encoding: .utf8),
      types.split(separator: "\n").count == allowed.count,
      types.split(separator: "\n").allSatisfy({ $0.hasPrefix("-") })
    else { throw EngineError.invalidArchive }
    let manifestData = try UpdateManager.runProcess(
      "/usr/bin/tar", arguments: ["-xOzf", archive.path, "engine.json"], outputLimit: 64 * 1_024)
    guard manifestData.count <= 64 * 1_024 else { throw EngineError.invalidArchive }
    let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
    try manifest.validate(appVersion: appVersion, expectedVersion: releaseVersion)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    try manifestData.write(to: directory.appendingPathComponent("engine.json"), options: .atomic)
    for name in runtimeFiles.sorted() {
      let data = try UpdateManager.runProcess(
        "/usr/bin/tar", arguments: ["-xOzf", archive.path, name],
        outputLimit: manifest.files[name]!.size)
      guard data.count == manifest.files[name]?.size else { throw EngineError.integrity }
      let file = directory.appendingPathComponent(name)
      try data.write(to: file, options: .atomic)
      try FileManager.default.setAttributes(
        [.posixPermissions: name == "mlx.metallib" ? 0o600 : 0o700], ofItemAtPath: file.path)
    }
    try verify(directory, appVersion: appVersion, expectedVersion: releaseVersion)
  }

  static func install(archive: URL, release: AppUpdateRelease, appVersion: String, root: URL = root)
    throws
  {
    try requireRegularFile(archive, maximum: 1_500_000_000)
    guard
      release.asset.size
        == Int64(try archive.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? -1),
      let digest = release.asset.digest, UpdateManager.validDigest(digest),
      try UpdateManager.sha256(at: archive) == digest.dropFirst(7).lowercased()
    else { throw EngineError.integrity }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let stageURL = root.appendingPathComponent(".stage-" + UUID().uuidString, isDirectory: true)
    defer { try? FileManager.default.removeItem(at: stageURL) }
    try stage(
      archive: archive, directory: stageURL, releaseVersion: release.version, appVersion: appVersion
    )
    for name in ["mlxl3", "libmlx.dylib", "libjaccl.dylib"] {
      let file = stageURL.appendingPathComponent(name)
      _ = try UpdateManager.runProcess(
        "/usr/bin/lipo", arguments: ["-verify_arch", "arm64", file.path])
      _ = try UpdateManager.runProcess(
        "/usr/bin/codesign", arguments: ["--verify", "--strict", file.path])
    }
    let executable = stageURL.appendingPathComponent("mlxl3")
    let info = try UpdateManager.runProcess(executable.path, arguments: ["runtime-info"])
    let payload = try JSONSerialization.jsonObject(with: info) as? [String: Any]
    guard payload?["bridge_protocol"] as? Int == protocolVersion,
      payload?["version"] as? String == release.version,
      payload?["mlx_enabled"] as? Bool == true
    else { throw EngineError.incompatible }
    _ = try UpdateManager.runProcess(executable.path, arguments: ["list", "--json"])
    let destination = root.appendingPathComponent("v" + release.version, isDirectory: true)
    if FileManager.default.fileExists(atPath: destination.path) {
      try verify(destination, appVersion: appVersion, expectedVersion: release.version)
      guard
        !FileManager.default.fileExists(atPath: destination.appendingPathComponent("disabled").path)
      else {
        throw EngineError.incompatible
      }
    } else {
      try FileManager.default.moveItem(at: stageURL, to: destination)
    }
    try activate(version: release.version, appVersion: appVersion, root: root)
  }

  static func activate(version: String, appVersion: String, root: URL) throws {
    // Validation precedes constructing a path from a version supplied by a caller.
    guard version.split(separator: ".").count == 3,
      SemanticVersion(version) != nil,
      version.utf8.allSatisfy({ (48...57).contains($0) || $0 == 46 })
    else { throw EngineError.incompatible }
    try verify(
      root.appendingPathComponent("v" + version), appVersion: appVersion, expectedVersion: version)
    let pointer = root.appendingPathComponent("current.json")
    if let old = try? Data(contentsOf: pointer) {
      try old.write(to: root.appendingPathComponent("previous.json"), options: .atomic)
    }
    try JSONEncoder().encode(Pointer(version: version)).write(to: pointer, options: .atomic)
  }
}

enum EngineError: LocalizedError {
  case incompatible, invalidArchive, integrity
  var errorDescription: String? {
    switch self {
    case .incompatible:
      L(
        "Ce moteur est incompatible avec cette app ou ce Mac. Le moteur embarqué reste disponible.",
        "This engine is incompatible with this app or Mac. The bundled engine remains available.")
    case .invalidArchive:
      L(
        "L’archive du moteur est invalide. Aucun moteur n’a été activé.",
        "Invalid engine archive. No engine was activated.")
    case .integrity:
      L(
        "Les fichiers du moteur ne correspondent pas aux empreintes attendues.",
        "Engine files do not match their expected checksums.")
    }
  }
}
