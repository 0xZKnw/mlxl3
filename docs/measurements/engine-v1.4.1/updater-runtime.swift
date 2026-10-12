import Foundation

@main struct RuntimeCheck {
  @MainActor static func main() async throws {
    let archive = URL(fileURLWithPath: CommandLine.arguments[1])
    let expectedCommit = CommandLine.arguments[2]
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let release = AppUpdateRelease(
      version: "1.4.1", tag: "engine-v1.4.1", title: "Engine integration", notes: "",
      pageURL: URL(string: "https://github.com/0xZKnw/mlxl3/releases/tag/engine-v1.4.1")!,
      asset: AppUpdateAsset(name: archive.lastPathComponent, downloadURL: archive,
        size: Int64(try Data(contentsOf: archive).count),
        digest: "sha256:" + (try UpdateManager.sha256(at: archive))))
    let store = root.appendingPathComponent("engine")
    try EngineRuntimeStore.install(archive: archive, release: release, appVersion: "1.4.0", root: store)
    let executable = EngineRuntimeStore.resolve(appVersion: "1.4.0", root: store, bundledVersion: "1.4.0")!
    let manifest = try EngineRuntimeStore.readManifest(executable.deletingLastPathComponent())
    precondition(manifest.version == "1.4.1" && manifest.minimumAppVersion == "1.4.0")
    do {
      try manifest.validate(appVersion: "1.3.0")
      preconditionFailure("old Desktop accepted")
    } catch {}
    let movedStore = root.appendingPathComponent("moved-engine")
    try FileManager.default.moveItem(at: store, to: movedStore)
    let moved = EngineRuntimeStore.resolve(appVersion: "1.4.0", root: movedStore, bundledVersion: "1.4.0")!
    let data = try UpdateManager.runProcess(moved.path, arguments: ["runtime-info"])
    let info = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    precondition(info["version"] as? String == "1.4.1" && info["commit"] as? String == expectedCommit)
    precondition(info["profile"] as? String == "release" && info["bridge_protocol"] as? Int == 1)
    precondition(info["mlx_enabled"] as? Bool == true && info["chat_enabled"] as? Bool == true)
    let registry = root.appendingPathComponent("empty-registry.json")
    try Data(#"{"version":1,"models":{}}"#.utf8).write(to: registry)
    let listed = try UpdateManager.runProcess(moved.path, arguments: ["--registry", registry.path, "list", "--json"])
    let listedModels = try JSONSerialization.jsonObject(with: listed) as! [Any]
    precondition(listedModels.isEmpty)
    precondition(EngineRuntimeStore.reject(moved, root: movedStore))
    precondition(EngineRuntimeStore.resolve(appVersion: "1.4.0", root: movedStore, bundledVersion: "1.4.0") == nil)
    print("Desktop 1.4.0: signed engine 1.4.1 installed, moved, executed and rejected; old Desktop rejected; disposable store removed")
    print(String(data: data, encoding: .utf8)!)
  }
}
