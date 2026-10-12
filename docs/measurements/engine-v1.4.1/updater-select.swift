import Foundation

@main struct UpdaterCheck {
  @MainActor static func main() async throws {
      let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2]))
      let app = try UpdateManager.selectRelease(data, channel: .app)!
      let engine = try UpdateManager.selectRelease(data, channel: .engine)!
      precondition(app.version == "1.4.0" && app.build == 23)
      precondition(engine.version == "1.4.1")
      precondition(app.asset.name == "MLXL3-Desktop-v1.4.0-b23-Apple-Silicon.dmg")
      precondition(app.asset.size == 73305316)
      precondition(app.asset.digest == "sha256:40b26ed3ca391ad9c673ae084e28c071094b830b47ced892fd7f73cdfb9a649a")
      precondition(engine.asset.name == "MLXL3-Engine-v1.4.1-arm64.tar.gz")
      precondition(engine.asset.size == 66866870)
      precondition(engine.asset.digest == "sha256:e220ae2e4fccae7515f9601a49ec0007730798cf2bf764177de96a09a3a81d0b")
      let proof: [String: Any] = ["desktop_version": app.version, "desktop_build": app.build,
        "desktop_asset": app.asset.name, "desktop_digest": app.asset.digest ?? "",
        "desktop_size": app.asset.size, "desktop_download_url": app.asset.downloadURL.absoluteString,
        "engine_version": engine.version, "engine_asset": engine.asset.name,
        "engine_digest": engine.asset.digest ?? "", "engine_size": engine.asset.size,
        "engine_download_url": engine.asset.downloadURL.absoluteString]
      print(String(data: try JSONSerialization.data(withJSONObject: proof, options: [.prettyPrinted, .sortedKeys]), encoding: .utf8)!)
  }
}
