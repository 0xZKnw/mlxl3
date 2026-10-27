import Foundation

extension UpdateManager {
  static func transferForCheck(
    _ release: AppUpdateRelease, cache: URL,
    progress: @escaping @MainActor @Sendable (UpdateDownloadProgress) -> Void
  ) async throws -> URL {
    try await downloadAndVerify(
      release: release, session: .shared, cacheDirectory: cache, progress: progress)
  }
}

@main struct UpdaterProgressCheck {
  @MainActor static func main() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-progress-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let metadata = root.appendingPathComponent("server.json")
    let server = Process()
    server.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
    server.arguments = [CommandLine.arguments[2], metadata.path]
    try server.run()
    defer {
      if server.isRunning { server.terminate() }
      server.waitUntilExit()
    }
    for _ in 0..<1000
    where !FileManager.default.fileExists(atPath: metadata.path) && server.isRunning {
      try await Task.sleep(for: .milliseconds(10))
    }
    let info = try JSONSerialization.jsonObject(with: Data(contentsOf: metadata)) as! [String: Any]
    let base = "http://127.0.0.1:\(info["port"]!)"
    let size = (info["size"] as! NSNumber).int64Value
    var count = 0
    func check(_ value: @autoclosure () -> Bool, _ text: String) {
      precondition(value(), text)
      count += 1
    }
    func release(_ path: String, digest: String? = nil) -> AppUpdateRelease {
      AppUpdateRelease(
        version: "2.0.0", tag: "v2.0.0", title: "fixture", notes: "",
        pageURL: URL(string: base)!,
        asset: AppUpdateAsset(
          name: "asset.dmg",
          downloadURL: URL(string: base + path)!, size: size,
          digest: digest ?? (info["digest"] as! String)))
    }
    for path in ["/asset", "/unknown"] {
      var snapshots: [UpdateDownloadProgress] = []
      let cache = root.appendingPathComponent(UUID().uuidString)
      let file = try await UpdateManager.transferForCheck(release(path), cache: cache) {
        snapshots.append($0)
      }
      try await Task.sleep(for: .milliseconds(20))
      let partial = snapshots.filter {
        !$0.isVerifying && $0.transfer.completed > 0 && $0.transfer.completed < Double(size)
      }
      print(
        "Transfer snapshots \(path): \(snapshots.map { "\($0.transfer.completed):\($0.isVerifying)" })"
      )
      check(partial.count >= 2, "Real URLSession byte callbacks must produce partial progress")
      check(
        partial.contains { ($0.transfer.bytesPerSecond ?? 0) > 0 },
        "Transfer rate uses received bytes")
      check(
        partial.allSatisfy { $0.transfer.fraction! > 0 && $0.transfer.fraction! < 1 },
        "Percentage bounded for missing/known Content-Length")
      check(snapshots.contains { $0.isVerifying }, "SHA verification is distinct from download")
      let content = try Data(contentsOf: file)
      check(content.count == Int(size), "Complete verified payload")
      snapshots = []
      _ = try await UpdateManager.transferForCheck(release(path), cache: cache) {
        snapshots.append($0)
      }
      check(snapshots.allSatisfy(\.isVerifying), "Cache hit verifies without inventing a transfer")
    }
    for path in ["/short", "/oversize", "/error", "/asset"] {
      let cache = root.appendingPathComponent(UUID().uuidString)
      do {
        _ = try await UpdateManager.transferForCheck(
          release(path, digest: "sha256:" + String(repeating: "0", count: 64)), cache: cache
        ) { _ in }
        preconditionFailure("Bad or incomplete download accepted: \(path)")
      } catch { count += 1 }
      check(
        (try? FileManager.default.contentsOfDirectory(atPath: cache.path).filter {
          $0.hasSuffix(".download") || $0 == "asset.dmg"
        }.isEmpty) == true,
        "Failed transfer leaves no staged or installable file")
    }
    let cache = root.appendingPathComponent("cancelled")
    let task = Task {
      try await UpdateManager.transferForCheck(release("/asset"), cache: cache) { _ in }
    }
    try await Task.sleep(for: .milliseconds(150))
    task.cancel()
    do {
      _ = try await task.value
      preconditionFailure("Cancellation ignored")
    } catch { count += 1 }
    check(
      !FileManager.default.fileExists(atPath: cache.appendingPathComponent("asset.dmg").path),
      "Cancelled transfer cannot be installed")
    print("Updater transfer checks passed: \(count)")
  }
}
