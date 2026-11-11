import AppKit
import Foundation
import SwiftUI
import Vision

private final class ReleaseNotesProtocol: URLProtocol, @unchecked Sendable {
  override class func canInit(with request: URLRequest) -> Bool { true }
  override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
  override func startLoading() {
    let body = """
      [{"tag_name":"v9.0.0","name":"Test release","body":"## Improvements\\n\\n- **Faster startup** for models.\\n- [Read documentation](https://example.com/docs)\\n\\n## Fixes\\n\\n- Update notes now render correctly.","html_url":"https://github.com/0xZKnw/mlxl3/releases/tag/v9.0.0","draft":false,"prerelease":false,"assets":[{"name":"MLXL3-Desktop-v9.0.0-b99-Apple-Silicon.dmg","size":5,"digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","browser_download_url":"https://github.com/0xZKnw/mlxl3/releases/download/v9.0.0/MLXL3-Desktop-v9.0.0-b99-Apple-Silicon.dmg"}]}]
      """.data(using: .utf8)!
    let response = HTTPURLResponse(
      url: request.url!, statusCode: 200, httpVersion: "HTTP/1.1", headerFields: nil)!
    client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
    client?.urlProtocol(self, didLoad: body)
    client?.urlProtocolDidFinishLoading(self)
  }
  override func stopLoading() {}
}

@main struct UpdateReleaseNotesCheck {
  @MainActor static func main() async throws {
    _ = NSApplication.shared
    NSApp.setActivationPolicy(.accessory)
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-update-notes-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let suite = "io.mlxl3.update-notes." + UUID().uuidString
    let preferences = UserDefaults(suiteName: suite)!
    defer {
      preferences.removePersistentDomain(forName: suite)
      try? FileManager.default.removeItem(at: root)
    }
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [ReleaseNotesProtocol.self]
    let session = URLSession(configuration: configuration)
    defer { session.invalidateAndCancel() }
    let updater = UpdateManager(
      currentVersion: "1.0.0", currentBuild: 1,
      releasesURL: URL(string: "https://example.invalid/releases")!, preferences: preferences,
      session: session, cacheDirectory: root)
    let studio = StudioModel(
      conversationFileURL: root.appendingPathComponent("history.json"),
      updateManager: updater, isPreview: true, preferences: preferences)
    studio.appSettingsPage = .updates
    updater.checkForUpdates()
    for _ in 0..<200 where updater.latestRelease == nil {
      try await Task.sleep(for: .milliseconds(25))
    }
    precondition(
      updater.hasAppUpdate && updater.latestRelease?.notes.contains("## Improvements") == true,
      "Update fixture was not loaded")
    for (width, theme) in [(CGFloat(1000), AppTheme.dark), (CGFloat(700), AppTheme.light)] {
      let host = NSHostingView(
        rootView: AppSettingsView().environmentObject(studio).studioAppearance(theme))
      let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: width, height: 1100),
        styleMask: [.borderless], backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      window.contentView = host
      host.frame = NSRect(x: 0, y: 0, width: width, height: 1100)
      window.orderFront(nil)
      defer {
        window.orderOut(nil)
        window.close()
      }
      var recognized = ""
      for _ in 0..<40 {
        try await Task.sleep(for: .milliseconds(250))
        host.layoutSubtreeIfNeeded()
        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
          fatalError("No bitmap")
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let request = VNRecognizeTextRequest()
        request.recognitionLevel = .accurate
        request.recognitionLanguages = ["en-US"]
        request.usesLanguageCorrection = false
        let stages = try request.supportedComputeStageDevices
        precondition(!stages.isEmpty, "No OCR compute stages")
        for (stage, devices) in stages {
          guard let cpu = devices.first(where: { if case .cpu = $0 { true } else { false } }) else {
            fatalError("No CPU OCR device")
          }
          request.setComputeDevice(cpu, for: stage)
        }
        try VNImageRequestHandler(cgImage: bitmap.cgImage!).perform([request])
        recognized = (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }
          .joined(separator: "\n")
        if recognized.localizedCaseInsensitiveContains("render correctly") {
          if let folder = ProcessInfo.processInfo.environment["MLXL3_UPDATE_NOTES_SNAPSHOT_DIR"] {
            try bitmap.representation(using: .png, properties: [:])!.write(
              to:
                URL(fileURLWithPath: folder).appendingPathComponent(
                  "update-notes-\(Int(width))-\(theme.rawValue).png"))
          }
          break
        }
      }
      print("Recognized update settings (\(Int(width)), \(theme.rawValue)):\n\(recognized)")
      precondition(
        recognized.localizedCaseInsensitiveContains("Improvements")
          && recognized.localizedCaseInsensitiveContains("Faster startup")
          && recognized.localizedCaseInsensitiveContains("Read documentation")
          && recognized.localizedCaseInsensitiveContains("Fixes")
          && recognized.localizedCaseInsensitiveContains("render correctly"),
        "Release note content is missing from the settings view")
      precondition(
        !recognized.contains("##") && !recognized.contains("**"),
        "Raw Markdown appeared in update notes")
    }
    print("Update release notes checks passed: actual update selection and Markdown display")
  }
}
