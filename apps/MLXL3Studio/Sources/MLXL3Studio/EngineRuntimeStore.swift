import Foundation

/// An engine update lives outside the signed application. Activation changes
/// one atomic pointer only after every file and the actual executable pass.
enum EngineRuntimeStore {
  // Keep native support available in source while v2 ships only MLX.
  static let nativeEngineEnabled = false
  static let protocolVersion = 1
  static let identityChanged = Notification.Name("MLXL3EngineIdentityChanged")
  static let runtimeFiles = Set(["mlxl3", "libmlx.dylib", "libjaccl.dylib", "mlx.metallib"])
  static let maximumFileBytes = 256 * 1_024 * 1_024

  struct FileRecord: Codable, Sendable {
    let size: Int
    let sha256: String
  }

  struct Manifest: Codable, Sendable {
    let schema: Int
    let backend: String?
    let version: String
    let releaseTag: String?
    let bridgeProtocol: Int
    let architecture: String
    let minimumMacOS: String
    let minimumAppVersion: String
    let files: [String: FileRecord]

    enum CodingKeys: String, CodingKey {
      case schema, backend, version, architecture, files
      case releaseTag = "release_tag"
      case bridgeProtocol = "bridge_protocol"
      case minimumMacOS = "minimum_macos"
      case minimumAppVersion = "minimum_app_version"
    }

    var runtimeFiles: Set<String> {
      schema == 2 ? Set(["mlxl3"]) : EngineRuntimeStore.runtimeFiles
    }

    func validateRuntimeInfo(_ payload: [String: Any]?) throws {
      guard (nativeEngineEnabled || payload?["backend"] as? String == "mlx"),
        payload?["bridge_protocol"] as? Int == protocolVersion,
        payload?["version"] as? String == version,
        payload?["mlx_enabled"] as? Bool == (schema == 1),
        schema == 1 || (payload?["backend"] as? String == "metal"
          && payload?["chat_enabled"] as? Bool == true
          && payload?["architecture"] as? String == "aarch64")
      else { throw EngineError.incompatible }
    }

    func validate(
      appVersion: String, expectedVersion: String? = nil,
      osVersion: String = EngineRuntimeStore.osVersion
    ) throws {
      guard (nativeEngineEnabled || schema == 1),
        ((schema == 1 && backend == nil) || (schema == 2 && backend == "metal")), bridgeProtocol == protocolVersion, architecture == "arm64",
        let incoming = SemanticVersion(version), version.split(separator: ".").count == 3,
        version.utf8.allSatisfy({ (48...57).contains($0) || $0 == 46 }),
        let minimumApp = SemanticVersion(minimumAppVersion),
        schema != 2 || minimumApp >= SemanticVersion("2.0.0")!,
        let app = SemanticVersion(appVersion), app >= minimumApp,
        let minimumOS = SemanticVersion(minimumMacOS),
        let os = SemanticVersion(osVersion), os >= minimumOS,
        expectedVersion == nil || version == expectedVersion,
        releaseTag == nil || (EngineReleaseIdentity(releaseTag!)?.version == incoming
          && EngineReleaseIdentity(releaseTag!)?.legacy == false),
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

  private struct Pointer: Codable {
    let version: String
    var releaseTag: String? = nil
    var archiveSHA256: String? = nil
    var manifestSHA256: String? = nil
    var directoryName: String {
      if let releaseTag, let archiveSHA256 { return releaseTag + "-" + archiveSHA256 }
      return "v" + version
    }
    var displayVersion: String {
      releaseTag.flatMap { EngineReleaseIdentity($0)?.rawValue } ?? version + "-legacy"
    }
  }

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
      return manifest.releaseTag.flatMap { EngineReleaseIdentity($0)?.rawValue } ?? manifest.version
    }
    return Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String
      ?? "1.2.0"
  }

  static func installedVersion() -> String {
    let appVersion = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.2.0"
    guard let executable = resolve(appVersion: appVersion),
          let pointer = try? readPointer(root.appendingPathComponent("current.json")),
          executable.deletingLastPathComponent().lastPathComponent == pointer.directoryName
    else { return bundledVersion }
    return pointer.displayVersion
  }

  static func resolve(
    appVersion: String, root: URL = root,
    bundledVersion: String = bundledVersion
  ) -> URL? {
    do {
      let pointer = try readPointer(root.appendingPathComponent("current.json"))
      guard let incoming = EngineReleaseIdentity(pointer.displayVersion),
        let bundled = EngineReleaseIdentity(bundledVersion), incoming > bundled
      else { return nil }
      // Never mask an equally based bundle with unidentifiable historical bytes.
      guard pointer.releaseTag != nil || incoming.version > bundled.version else { return nil }
      let directory = root.appendingPathComponent(pointer.directoryName, isDirectory: true)
      guard
        !FileManager.default.fileExists(atPath: directory.appendingPathComponent("disabled").path)
      else { return nil }
      try verifyPointer(pointer, directory: directory, appVersion: appVersion)
      let executable = directory.appendingPathComponent("mlxl3")
      guard FileManager.default.isExecutableFile(atPath: executable.path) else { return nil }
      // Historical hybrid archives also use schema 1: inspect the real backend.
      let data = try UpdateManager.runProcess(executable.path, arguments: ["runtime-info"], outputLimit: 64 * 1_024, timeout: 3)
      try readManifest(directory).validateRuntimeInfo(JSONSerialization.jsonObject(with: data) as? [String: Any])
      return executable
    } catch { return nil }
  }

  /// Only executable/protocol defects justify persistent quarantine, never model/resource errors.
  /// Existing engines remain on disk for diagnostics; subsequent loads fall
  /// back to the bundled runtime without another download.
  @discardableResult
  static func reject(_ executable: URL, root: URL = root) -> Bool {
    // Ignore bundle preference only when disabling this exact failing managed path.
    guard let active = resolve(appVersion: "999.0.0", root: root, bundledVersion: "0.0.0"),
      active.standardizedFileURL == executable.standardizedFileURL
    else { return false }
    do {
      try Data().write(
        to: active.deletingLastPathComponent().appendingPathComponent("disabled"), options: .atomic)
      NotificationCenter.default.post(name: identityChanged, object: nil)
      return true
    } catch { return false }
  }

  static func isExecutableLaunchFailure(_ failure: Error) -> Bool {
    let error = failure as NSError
    // Foundation can wrap a resource failure in a generic executable-load error.
    // That is not evidence that the installed binary is defective.
    if let underlying = error.userInfo[NSUnderlyingErrorKey] as? NSError,
       underlying.domain == NSPOSIXErrorDomain,
       [POSIXErrorCode.ENOMEM, .EAGAIN, .EMFILE, .ENFILE].contains(where: { Int($0.rawValue) == underlying.code }) {
      return false
    }
    if error.domain == NSPOSIXErrorDomain {
      return [POSIXErrorCode.ENOEXEC, .ENOENT, .EACCES].contains { Int($0.rawValue) == error.code }
    }
    if error.domain == NSCocoaErrorDomain {
      return [CocoaError.Code.executableNotLoadable, .executableArchitectureMismatch,
              .executableRuntimeMismatch, .executableLoad, .fileNoSuchFile,
              .fileReadNoSuchFile, .fileReadNoPermission].contains { $0.rawValue == error.code }
    }
    return false
  }

  static func shouldQuarantine(receivedReady: Bool, protocolMismatch: Bool) -> Bool {
    !receivedReady && protocolMismatch
  }

  static func useBundled(root: URL = root) throws {
    let pointer = root.appendingPathComponent("current.json")
    if FileManager.default.fileExists(atPath: pointer.path) {
      try FileManager.default.removeItem(at: pointer)
      NotificationCenter.default.post(name: identityChanged, object: nil)
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
    let fields = [pointer.releaseTag, pointer.archiveSHA256, pointer.manifestSHA256]
    if fields.contains(where: { $0 != nil }) {
      guard let tag = pointer.releaseTag, let identity = EngineReleaseIdentity(tag), !identity.legacy,
            tag == "engine-v" + identity.rawValue, identity.version == SemanticVersion(pointer.version),
            let archive = pointer.archiveSHA256, let manifest = pointer.manifestSHA256,
            [archive, manifest].allSatisfy({ UpdateManager.validDigest("sha256:" + $0) && $0 == $0.lowercased() })
      else { throw EngineError.incompatible }
    }
    return pointer
  }

  private static func verifyPointer(_ pointer: Pointer, directory: URL, appVersion: String) throws {
    try verify(directory, appVersion: appVersion, expectedVersion: pointer.version)
    if let expected = pointer.manifestSHA256 {
      guard try UpdateManager.sha256(at: directory.appendingPathComponent("engine.json")) == expected
      else { throw EngineError.integrity }
    }
    if let tag = try readManifest(directory).releaseTag, let expected = pointer.releaseTag,
       tag != expected { throw EngineError.integrity }
  }

  static func verify(_ directory: URL, appVersion: String, expectedVersion: String) throws {
    let info = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
    guard info.isDirectory == true, info.isSymbolicLink != true else {
      throw EngineError.invalidArchive
    }
    let manifest = try readManifest(directory)
    try manifest.validate(appVersion: appVersion, expectedVersion: expectedVersion)
    if manifest.schema == 2 {
      let names = Set(try FileManager.default.contentsOfDirectory(atPath: directory.path))
      guard names.subtracting(manifest.runtimeFiles.union(["engine.json", "disabled"])).isEmpty
      else { throw EngineError.invalidArchive }
    }
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

  /// Avoid archive extraction entirely: read the manifest-selected exact members to
  /// stdout, then write them to paths we construct. Links, duplicates, extra
  /// files and traversal names are rejected before any member is materialized.
  static func stage(archive: URL, directory: URL, releaseVersion: String, appVersion: String) throws
  {
    let listing = try UpdateManager.runProcess(
      "/usr/bin/tar", arguments: ["-tzf", archive.path], outputLimit: 64 * 1_024)
    guard let text = String(data: listing, encoding: .utf8) else {
      throw EngineError.invalidArchive
    }
    let members = text.split(separator: "\n").map(String.init)
    guard members.count >= 2, members.count <= 5,
      members.filter({ $0 == "engine.json" }).count == 1,
      Set(members).isSubset(of: runtimeFiles.union(["engine.json"])) else {
      throw EngineError.invalidArchive
    }
    let verbose = try UpdateManager.runProcess(
      "/usr/bin/tar", arguments: ["-tvzf", archive.path], outputLimit: 64 * 1_024)
    guard let types = String(data: verbose, encoding: .utf8),
      types.split(separator: "\n").count == members.count,
      types.split(separator: "\n").allSatisfy({ $0.hasPrefix("-") })
    else { throw EngineError.invalidArchive }
    let manifestData = try UpdateManager.runProcess(
      "/usr/bin/tar", arguments: ["-xOzf", archive.path, "engine.json"], outputLimit: 64 * 1_024)
    guard manifestData.count <= 64 * 1_024 else { throw EngineError.invalidArchive }
    let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
    try manifest.validate(appVersion: appVersion, expectedVersion: releaseVersion)
    let allowed = manifest.runtimeFiles.union(["engine.json"])
    guard members.count == allowed.count, Set(members) == allowed else {
      throw EngineError.invalidArchive
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    try manifestData.write(to: directory.appendingPathComponent("engine.json"), options: .atomic)
    for name in manifest.runtimeFiles.sorted() {
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
    guard let identity = EngineReleaseIdentity(release.tag), !identity.legacy,
          release.tag == "engine-v" + identity.rawValue, identity.version == SemanticVersion(release.version)
    else { throw EngineError.incompatible }
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
    let manifest = try readManifest(stageURL)
    guard manifest.releaseTag == nil || manifest.releaseTag == release.tag else { throw EngineError.incompatible }
    for name in manifest.runtimeFiles.sorted() where name != "mlx.metallib" {
      let file = stageURL.appendingPathComponent(name)
      _ = try UpdateManager.runProcess(
        "/usr/bin/lipo", arguments: [file.path, "-verify_arch", "arm64"])
      _ = try UpdateManager.runProcess(
        "/usr/bin/codesign", arguments: ["--verify", "--strict", file.path])
    }
    let executable = stageURL.appendingPathComponent("mlxl3")
    let info = try UpdateManager.runProcess(executable.path, arguments: ["runtime-info"])
    let payload = try JSONSerialization.jsonObject(with: info) as? [String: Any]
    try manifest.validateRuntimeInfo(payload)
    if manifest.schema == 2 {
      let linked = try UpdateManager.runProcess("/usr/bin/otool", arguments: ["-L", executable.path])
      guard let text = String(data: linked, encoding: .utf8),
        !["libmlx.", "libjaccl.", "mlx.metallib"].contains(where: { text.lowercased().contains($0) })
      else { throw EngineError.incompatible }
    }
    _ = try UpdateManager.runProcess(executable.path, arguments: ["list", "--json"])
    try publishValidatedStage(stageURL, release: release, appVersion: appVersion, root: root)
  }

  /// Commit only after install validates the archive, signatures and runtime-info.
  /// Existing bytes must match the candidate manifest, not merely their own manifest.
  static func publishValidatedStage(_ stageURL: URL, release: AppUpdateRelease,
                                    appVersion: String, root: URL) throws {
    guard let identity = EngineReleaseIdentity(release.tag), !identity.legacy,
          release.tag == "engine-v" + identity.rawValue,
          identity.version == SemanticVersion(release.version),
          let digest = release.asset.digest, UpdateManager.validDigest(digest)
    else { throw EngineError.incompatible }
    let pointer = Pointer(version: release.version, releaseTag: release.tag,
      archiveSHA256: String(digest.dropFirst(7)).lowercased(),
      manifestSHA256: try UpdateManager.sha256(at: stageURL.appendingPathComponent("engine.json")))
    try verifyPointer(pointer, directory: stageURL, appVersion: appVersion)
    if let active = try? readPointer(root.appendingPathComponent("current.json")),
       let previous = EngineReleaseIdentity(active.displayVersion), identity < previous {
      throw EngineError.incompatible
    }
    let destination = root.appendingPathComponent(pointer.directoryName, isDirectory: true)
    if FileManager.default.fileExists(atPath: destination.path) {
      try verifyPointer(pointer, directory: destination, appVersion: appVersion)
      guard !FileManager.default.fileExists(atPath: destination.appendingPathComponent("disabled").path)
      else { throw EngineError.incompatible }
    } else {
      try FileManager.default.moveItem(at: stageURL, to: destination)
    }
    try writePointer(pointer, root: root)
  }

  static func activate(version: String, appVersion: String, root: URL) throws {
    // Validation precedes constructing a path from a version supplied by a caller.
    guard version.split(separator: ".").count == 3,
      SemanticVersion(version) != nil,
      version.utf8.allSatisfy({ (48...57).contains($0) || $0 == 46 })
    else { throw EngineError.incompatible }
    try verify(
      root.appendingPathComponent("v" + version), appVersion: appVersion, expectedVersion: version)
    try writePointer(Pointer(version: version), root: root)
  }

  private static func writePointer(_ value: Pointer, root: URL) throws {
    let pointer = root.appendingPathComponent("current.json")
    if let old = try? Data(contentsOf: pointer) {
      try old.write(to: root.appendingPathComponent("previous.json"), options: .atomic)
    }
    try JSONEncoder().encode(value).write(to: pointer, options: .atomic)
    NotificationCenter.default.post(name: identityChanged, object: nil)
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
