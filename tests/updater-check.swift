import Foundation

@main struct UpdaterCheck {
  @MainActor static func main() async throws {
    let fixtures = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-updater-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func check(_ value: Bool, _ message: String = "") { precondition(value, message) }
    func fails(_ name: String, _ body: () throws -> Void) {
      do {
        try body()
        preconditionFailure("Expected rejection: " + name)
      } catch {}
    }
    for invalid in [
      "", "vv1.2.0", "1.2.0.4", "01.2.0", "+1.2.0", "1..2", "1.2.0-rc", "1.2.0/..",
      "999999999999999999999.0.0",
    ] {
      check(SemanticVersion(invalid) == nil, invalid)
    }
    check(SemanticVersion("1.0") == SemanticVersion("1.0.0"))
    check(SemanticVersion("1.2.9")! < SemanticVersion("1.2.10")!)
    let digest = "sha256:" + String(repeating: "a", count: 64)
    func payload(_ tag: String, _ asset: String, draft: Bool = false) -> [String: Any] {
      [
        "tag_name": tag, "name": tag, "body": "test",
        "html_url": "https://github.com/0xZKnw/mlxl3/releases/tag/" + tag,
        "draft": draft, "prerelease": false,
        "assets": [
          [
            "name": asset, "size": 5, "digest": digest,
            "browser_download_url": "https://github.com/0xZKnw/mlxl3/releases/download/" + tag + "/"
              + asset,
          ]
        ],
      ]
    }
    let app = payload("v1.2.0", "MLXL3-Desktop-v1.2.0-b19-Apple-Silicon.dmg")
    let engine = payload("engine-v1.3.0", "MLXL3-Engine-v1.3.0-arm64.tar.gz")
    let list = try JSONSerialization.data(withJSONObject: [
      engine, app,
      payload("v9.9.9", "MLXL3-Desktop-v9.9.9-b1-Apple-Silicon.dmg", draft: true),
    ])
    check(try UpdateManager.selectRelease(list, channel: .app)?.version == "1.2.0")
    check(try UpdateManager.selectRelease(list, channel: .engine)?.version == "1.3.0")
    for (channel, item) in [(UpdateManager.ReleaseChannel.app, app), (.engine, engine)] {
      for failure in ["url", "digest", "zero", "huge", "tag", "page", "prerelease"] {
        var changed = item
        var assets = changed["assets"] as! [[String: Any]]
        switch failure {
        case "url":
          assets[0]["browser_download_url"] =
            "https://github.com.attacker.invalid/0xZKnw/mlxl3/releases/download/test/x"
        case "digest": assets[0].removeValue(forKey: "digest")
        case "zero": assets[0]["size"] = 0
        case "huge": assets[0]["size"] = 1_500_000_001
        case "tag": changed["tag_name"] = "vv1.2.0"
        case "page": changed["html_url"] = "https://github.com/other/repo/releases/tag/v1.2.0"
        default: changed["prerelease"] = true
        }
        changed["assets"] = assets
        let data = try JSONSerialization.data(withJSONObject: [changed])
        check(try UpdateManager.selectRelease(data, channel: channel) == nil, failure)
      }
    }
    for url in [
      "http://github.com/test", "https://user@github.com/test", "https://github.com:443/test",
      "https://github.com/test?x", "https://github.com/test#x",
    ] {
      check(!UpdateManager.trustedURL(URL(string: url)!, path: "/test"), url)
    }
    for bad in [
      "sha512:" + String(repeating: "a", count: 64), "sha256:abc",
      "sha256:" + String(repeating: "z", count: 64),
    ] {
      check(!UpdateManager.validDigest(bad))
    }
    let valid = fixtures.appendingPathComponent("valid.tar.gz")
    let staged = root.appendingPathComponent("v1.2.0")
    try EngineRuntimeStore.stage(
      archive: valid, directory: staged, releaseVersion: "1.2.0", appVersion: "1.2.0")
    try EngineRuntimeStore.verify(staged, appVersion: "1.2.0", expectedVersion: "1.2.0")
    let manifest = try EngineRuntimeStore.readManifest(staged)
    fails("old app") { try manifest.validate(appVersion: "1.1.3") }
    fails("old OS") { try manifest.validate(appVersion: "1.2.0", osVersion: "26.1.9") }
    fails("version mismatch") {
      try manifest.validate(appVersion: "1.2.0", expectedVersion: "1.2.1")
    }
    for invalid in [
      "duplicate", "traversal", "extra", "symlink", "hardlink", "hash", "size", "protocol", "huge",
    ] {
      fails("archive " + invalid) {
        try EngineRuntimeStore.stage(
          archive: fixtures.appendingPathComponent(invalid + ".tar.gz"),
          directory: root.appendingPathComponent(invalid), releaseVersion: "1.2.0",
          appVersion: "1.2.0")
      }
      check(
        !FileManager.default.fileExists(atPath: root.appendingPathComponent("current.json").path))
    }
    try EngineRuntimeStore.activate(version: "1.2.0", appVersion: "1.2.0", root: root)
    let executable = staged.appendingPathComponent("mlxl3")
    check(
      EngineRuntimeStore.resolve(appVersion: "1.2.0", root: root, bundledVersion: "1.2.0")
        == executable)
    check(
      EngineRuntimeStore.resolve(appVersion: "1.2.0", root: root, bundledVersion: "1.2.1") == nil)
    let pointer = try Data(contentsOf: root.appendingPathComponent("current.json"))
    for version in ["../1.2.0", "1.2.1", "v1.2.0", "01.2.0"] {
      fails("activation " + version) {
        try EngineRuntimeStore.activate(version: version, appVersion: "1.2.0", root: root)
      }
      check(try Data(contentsOf: root.appendingPathComponent("current.json")) == pointer)
    }
    let asset = AppUpdateAsset(
      name: "test", downloadURL: URL(string: "https://github.com/test")!,
      size: Int64(try Data(contentsOf: executable).count),
      digest: "sha256:" + (try UpdateManager.sha256(at: executable)))
    check(try await UpdateManager.verifyAsset(at: executable, asset: asset))
    try Data("corrupt".utf8).write(to: executable)
    check(!(try await UpdateManager.verifyAsset(at: executable, asset: asset)))
    check(
      EngineRuntimeStore.resolve(appVersion: "1.2.0", root: root, bundledVersion: "1.2.0") == nil)
    try FileManager.default.removeItem(at: staged)
    try EngineRuntimeStore.stage(
      archive: valid, directory: staged, releaseVersion: "1.2.0", appVersion: "1.2.0")
    check(EngineRuntimeStore.reject(executable, root: root))
    check(
      EngineRuntimeStore.resolve(appVersion: "1.2.0", root: root, bundledVersion: "1.2.0") == nil)
    let script = root.appendingPathComponent("install.zsh")
    try UpdateManager.installerScript.write(to: script, atomically: true, encoding: .utf8)
    _ = try UpdateManager.runProcess("/bin/zsh", arguments: ["-n", script.path])
    let installed = root.appendingPathComponent("Working.app")
    let backup = root.appendingPathComponent("Working.app.mlxl3-backup")
    try FileManager.default.createDirectory(at: installed, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: backup, withIntermediateDirectories: true)
    try Data("working".utf8).write(to: installed.appendingPathComponent("keep"))
    try Data("backup".utf8).write(to: backup.appendingPathComponent("keep"))
    fails("failed app copy") {
      _ = try UpdateManager.runProcess(
        "/bin/zsh",
        arguments: [
          script.path, "99999999", root.appendingPathComponent("missing.app").path,
          installed.path, root.path, root.appendingPathComponent("install.log").path, "1.2.0", "19",
        ])
    }
    check(try Data(contentsOf: installed.appendingPathComponent("keep")) == Data("working".utf8))
    check(try Data(contentsOf: backup.appendingPathComponent("keep")) == Data("backup".utf8))
    _ = try UpdateManager.runProcess("/bin/echo", arguments: ["pipe check"])
    fails("helper output bound") {
      _ = try UpdateManager.runProcess("/usr/bin/yes", arguments: [], outputLimit: 1_024)
    }
    if CommandLine.arguments.count > 2 {
      let archive = URL(fileURLWithPath: CommandLine.arguments[2])
      let version = CommandLine.arguments.count > 3 ? CommandLine.arguments[3] : "1.2.0"
      let release = AppUpdateRelease(
        version: version, tag: "engine-v\(version)", title: "Engine integration", notes: "",
        pageURL: URL(string: "https://github.com/0xZKnw/mlxl3/releases/tag/engine-v\(version)")!,
        asset: AppUpdateAsset(
          name: archive.lastPathComponent, downloadURL: archive,
          size: Int64(try Data(contentsOf: archive).count),
          digest: "sha256:" + (try UpdateManager.sha256(at: archive))))
      let store = root.appendingPathComponent("signed-engine")
      try EngineRuntimeStore.install(
        archive: archive, release: release, appVersion: version, root: store)
      let selected = EngineRuntimeStore.resolve(
        appVersion: version, root: store, bundledVersion: version)!
      _ = try UpdateManager.runProcess(selected.path, arguments: ["runtime-info"])
      let realManifest = try EngineRuntimeStore.readManifest(selected.deletingLastPathComponent())
      check(realManifest.version == version)
      if version == "1.4.0" {
        fails("MTP engine on old Desktop") { try realManifest.validate(appVersion: "1.3.0") }
      }
      check(EngineRuntimeStore.reject(selected, root: store))
      check(
        EngineRuntimeStore.resolve(appVersion: version, root: store, bundledVersion: version) == nil
      )
      print("Signed engine installation, relocation, execution and fallback passed")
    }
    print(
      "Updater checks passed: independent channels, versions/URLs/hashes, hostile archives, activation/fallback, app-copy failure, helper bounds"
    )
  }
}
