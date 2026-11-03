import Foundation
import Synchronization

final class ChannelProtocol: URLProtocol, @unchecked Sendable {
  struct State: Sendable {
    var bodies: [String: Data] = [:]
    var blocked: Set<String> = []
    var started: [String] = []
    var stopped: [String] = []
  }
  static let state = Mutex(State())
  static func configure(_ bodies: [String: Data], blocked: Set<String> = []) {
    state.withLock { $0 = State(bodies: bodies, blocked: blocked) }
  }
  static func started(_ url: String) -> Int { state.withLock { $0.started.filter { $0 == url }.count } }
  static func stopped(_ url: String) -> Int { state.withLock { $0.stopped.filter { $0 == url }.count } }
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    guard let url = request.url else { client?.urlProtocol(self, didFailWithError: URLError(.badURL)); return }
    let (body, blocked) = Self.state.withLock { state -> (Data?, Bool) in
      state.started.append(url.absoluteString)
      return (state.bodies[url.absoluteString], state.blocked.contains(url.absoluteString))
    }
    guard let body else { client?.urlProtocol(self, didFailWithError: URLError(.unsupportedURL)); return }
    let response = HTTPURLResponse(url: url, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: ["Content-Length": String(body.count)])!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    if blocked {
      client?.urlProtocol(self, didLoad: Data(body.prefix(1)))
      return
    }
    client?.urlProtocol(self, didLoad: body)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {
    if let url = request.url { Self.state.withLock { $0.stopped.append(url.absoluteString) } }
  }
}

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
    func payload(_ tag: String, _ asset: String, draft: Bool = false, prerelease: Bool = false) -> [String: Any] {
      [
        "tag_name": tag, "name": tag, "body": "test",
        "html_url": "https://github.com/0xZKnw/mlxl3/releases/tag/" + tag,
        "draft": draft, "prerelease": prerelease,
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
    let beta1 = payload("v2.0.0-beta.1", "MLXL3-Desktop-v2.0.0-beta.1-b31-Apple-Silicon.dmg", prerelease: true)
    let beta2 = payload("v2.0.0-beta.2", "MLXL3-Desktop-v2.0.0-beta.2-b32-Apple-Silicon.dmg", prerelease: true)
    let stable = payload("v1.5.1", "MLXL3-Desktop-v1.5.1-b33-Apple-Silicon.dmg")
    let betaEngine1 = payload("engine-v2.0.0-beta.1", "MLXL3-Engine-v2.0.0-beta.1-arm64.tar.gz", prerelease: true)
    let betaEngine2 = payload("engine-v2.0.0-beta.2", "MLXL3-Engine-v2.0.0-beta.2-arm64.tar.gz", prerelease: true)
    let policyData = try JSONSerialization.data(withJSONObject: [betaEngine1, beta1, stable, betaEngine2, beta2, engine])
    check(try UpdateManager.selectRelease(policyData, channel: .app)?.tag == "v1.5.1")
    check(try UpdateManager.selectRelease(policyData, channel: .engine)?.tag == "engine-v1.3.0")
    let selectedBeta = try UpdateManager.selectRelease(policyData, channel: .app, updateChannel: .beta)!
    check(selectedBeta.version == "2.0.0" && selectedBeta.tag == "v2.0.0-beta.2" && selectedBeta.build == 32)
    check(selectedBeta.asset.name == "MLXL3-Desktop-v2.0.0-beta.2-b32-Apple-Silicon.dmg")
    check(try UpdateManager.selectRelease(policyData, channel: .engine, updateChannel: .beta)?.tag == "engine-v2.0.0-beta.2")
    check(SemanticVersion("2.0.0-beta.2") == nil, "Runtime/manifest versions must remain strict numeric")
    for suffix in ["-beta.0", "-beta.01", "-beta.", "-beta.-1", "-beta.1-extra", "-beta.1-beta.2", "-beta.999999999999999999999", "-rc.1"] {
      let raw = "2.0.0" + suffix
      let item = payload("v" + raw, "MLXL3-Desktop-v" + raw + "-b32-Apple-Silicon.dmg", prerelease: true)
      check(try UpdateManager.selectRelease(JSONSerialization.data(withJSONObject: [item]), channel: .app, updateChannel: .beta) == nil, suffix)
    }
    for failure in ["flag", "draft", "wrong-asset", "url", "page", "hash", "size"] {
      var changed = beta2
      var assets = changed["assets"] as! [[String: Any]]
      switch failure {
      case "flag": changed["prerelease"] = false
      case "draft": changed["draft"] = true
      case "wrong-asset": assets[0]["name"] = "MLXL3-Desktop-v2.0.0-b32-Apple-Silicon.dmg"
      case "url": assets[0]["browser_download_url"] = "https://github.com.attacker.invalid/fake"
      case "page": changed["html_url"] = "https://github.com/other/repo/releases/tag/v2.0.0-beta.2"
      case "hash": assets[0]["digest"] = "sha256:invalid"
      default: assets[0]["size"] = 0
      }
      changed["assets"] = assets
      check(try UpdateManager.selectRelease(JSONSerialization.data(withJSONObject: [changed]), channel: .app, updateChannel: .beta) == nil, failure)
    }
    let finalEngine = payload("engine-v2.0.0", "MLXL3-Engine-v2.0.0-arm64.tar.gz")
    check(try UpdateManager.selectRelease(JSONSerialization.data(withJSONObject: [betaEngine2, finalEngine]), channel: .engine, updateChannel: .beta)?.tag == "engine-v2.0.0")

    let suite = "io.mlxl3.channel-check." + UUID().uuidString
    let preferences = UserDefaults(suiteName: suite)!
    defer { preferences.removePersistentDomain(forName: suite) }
    let preferenceCheck = UpdateManager(preferences: preferences)
    check(preferenceCheck.updateChannel == .stable)
    preferenceCheck.setUpdateChannel(.beta)
    check(preferences.string(forKey: "updates.channel") == "beta")
    check(UpdateManager(preferences: preferences).updateChannel == .beta)
    preferences.set("unknown", forKey: "updates.channel")
    check(UpdateManager(preferences: preferences).updateChannel == .stable)
    preferences.removeObject(forKey: "updates.channel")
    func waitFor(_ message: String, _ body: () -> Bool) async throws {
      for _ in 0..<500 {
        if body() { return }
        try await Task.sleep(for: .milliseconds(10))
      }
      throw NSError(domain: "ChannelCheck", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    func makeSession() -> URLSession {
      let configuration = URLSessionConfiguration.ephemeral
      configuration.protocolClasses = [ChannelProtocol.self]
      return URLSession(configuration: configuration)
    }
    do {
      let lookup = "https://updates.invalid/checking"
      ChannelProtocol.configure([lookup: policyData], blocked: [lookup])
      let session = makeSession(); defer { session.invalidateAndCancel() }
      let manager = UpdateManager(currentVersion: "2.0.0", currentBuild: 31, releasesURL: URL(string: lookup)!, preferences: preferences, session: session, cacheDirectory: root.appendingPathComponent("checking-cache"))
      manager.checkForUpdates()
      try await waitFor("Initial lookup did not start") { ChannelProtocol.started(lookup) == 1 }
      manager.setUpdateChannel(.beta)
      check(manager.state == .idle && manager.engineState == .idle && manager.latestRelease == nil && manager.engineRelease == nil)
      check(ChannelProtocol.started(lookup) == 1, "Changing channel must not make a new request by itself")
      manager.checkForUpdates()
      try await waitFor("Lookup cancellation/new generation missing") { ChannelProtocol.stopped(lookup) >= 1 && ChannelProtocol.started(lookup) == 2 }
      try await Task.sleep(for: .milliseconds(60))
      check(manager.state == .checking, "Old cancelled lookup overwrote the new generation")
      manager.setUpdateChannel(.stable)
    }
    do {
      let lookup = "https://updates.invalid/downloading"
      let stableURL = ((stable["assets"] as! [[String: Any]])[0]["browser_download_url"] as! String)
      let betaURL = ((beta2["assets"] as! [[String: Any]])[0]["browser_download_url"] as! String)
      let data = try JSONSerialization.data(withJSONObject: [stable, beta2])
      ChannelProtocol.configure([lookup: data, stableURL: Data("stable".utf8), betaURL: Data("beta".utf8)], blocked: [stableURL, betaURL])
      let session = makeSession(); defer { session.invalidateAndCancel() }
      preferences.set("stable", forKey: "updates.channel")
      let manager = UpdateManager(currentVersion: "1.5.0", releasesURL: URL(string: lookup)!, preferences: preferences, session: session, cacheDirectory: root.appendingPathComponent("download-cache"))
      manager.checkForUpdates()
      try await waitFor("Stable download did not start") { ChannelProtocol.started(stableURL) == 1 }
      manager.setUpdateChannel(.beta)
      manager.checkForUpdates()
      try await waitFor("New beta download/old cancellation missing") { ChannelProtocol.stopped(stableURL) >= 1 && ChannelProtocol.started(betaURL) == 1 }
      try await Task.sleep(for: .milliseconds(60))
      guard case .downloading(let release) = manager.state else { preconditionFailure("Cancelled old download replaced new state: \(manager.state)") }
      check(release.tag == "v2.0.0-beta.2" && !manager.hasReadyUpdate)
      manager.setUpdateChannel(.stable)
      try await waitFor("Beta download was not cancelled") { ChannelProtocol.stopped(betaURL) >= 1 }
      check(manager.state == .idle && manager.engineState == .idle && !manager.hasReadyUpdate)
      if case .some = await manager.beginInstallation() { preconditionFailure("Channel change installed a cleared update") }
    }
    do {
      let lookup = "https://updates.invalid/cached-beta"
      let cache = root.appendingPathComponent("ready-cache")
      try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
      let bytes = Data("verified update fixture".utf8)
      let file = cache.appendingPathComponent("MLXL3-Desktop-v2.0.0-beta.2-b32-Apple-Silicon.dmg")
      try bytes.write(to: file)
      var cached = beta2; var assets = cached["assets"] as! [[String: Any]]
      assets[0]["size"] = bytes.count; assets[0]["digest"] = "sha256:" + (try UpdateManager.sha256(at: file)); cached["assets"] = assets
      ChannelProtocol.configure([lookup: try JSONSerialization.data(withJSONObject: [cached])])
      let session = makeSession(); defer { session.invalidateAndCancel() }
      preferences.set("beta", forKey: "updates.channel")
      let manager = UpdateManager(currentVersion: "2.0.0", currentBuild: 31, releasesURL: URL(string: lookup)!, preferences: preferences, session: session, cacheDirectory: cache)
      manager.checkForUpdates()
      try await waitFor("Cached beta2 did not finish") { !manager.isBusy }
      check(manager.hasAppUpdate && manager.hasReadyUpdate)
      guard case .ready(let release, let prepared) = manager.state else { preconditionFailure("Beta1→beta2 was not prepared: \(manager.state)") }
      let preparedBytes = try Data(contentsOf: prepared)
      check(release.build == 32 && release.version == "2.0.0" && prepared == file && preparedBytes == bytes)
      manager.setUpdateChannel(.stable)
      check(!manager.hasReadyUpdate && manager.latestRelease == nil && manager.engineRelease == nil)
      if case .some = await manager.beginInstallation() { preconditionFailure("Changing channel automatically installed the beta") }
      manager.checkForUpdates()
      try await waitFor("Stable lookup did not finish") { !manager.isBusy }
      check(!manager.hasAppUpdate && !manager.hasReadyUpdate, "Stable channel offered the cached beta")
    }
    do {
      let lookup = "https://updates.invalid/no-downgrade"
      ChannelProtocol.configure([lookup: try JSONSerialization.data(withJSONObject: [stable, beta1, beta2])])
      let session = makeSession(); defer { session.invalidateAndCancel() }
      preferences.set("beta", forKey: "updates.channel")
      let manager = UpdateManager(currentVersion: "3.0.0", currentBuild: 99, releasesURL: URL(string: lookup)!, preferences: preferences, session: session, cacheDirectory: root.appendingPathComponent("no-downgrade"))
      manager.checkForUpdates()
      try await waitFor("No-downgrade lookup did not finish") { !manager.isBusy }
      check(!manager.hasAppUpdate && !manager.hasReadyUpdate)
    }
    print("Update channel checks passed: stable/beta selection, official suffixes/assets, persistence, no downgrade, beta1→beta2 build, lookup/download/ready cancellation")
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
    check(EngineRuntimeStore.resolve(appVersion: "1.2.0", root: root, bundledVersion: "1.2.0") == nil, "An equal managed engine must not mask the newer bundled beta build")
    check(
      EngineRuntimeStore.resolve(appVersion: "1.2.0", root: root, bundledVersion: "1.1.0")
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
      check(EngineRuntimeStore.resolve(appVersion: version, root: store, bundledVersion: version) == nil)
      let selected = EngineRuntimeStore.resolve(
        appVersion: version, root: store, bundledVersion: "1.1.0")!
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
