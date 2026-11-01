import Darwin
import SwiftUI
import WebKit

@MainActor
enum HTMLPreview {
  nonisolated static let maximumBytes = 2 * 1024 * 1024
  static let pageURL = URL(string: "mlxl3-preview://document/index.html")!
  static let remoteOrigins = [
    "https://cdn.jsdelivr.net", "https://unpkg.com", "https://esm.sh",
    "https://cdnjs.cloudflare.com", "https://cdn.skypack.dev", "https://threejs.org",
  ]
  static var remoteSources: String { remoteOrigins.joined(separator: " ") }
  static let policy =
    "default-src 'none'; script-src 'unsafe-inline' data: blob: mlxl3-preview: \(remoteSources); style-src 'unsafe-inline' mlxl3-preview: \(remoteSources); img-src data: blob: mlxl3-preview: \(remoteSources); font-src data: mlxl3-preview: \(remoteSources); media-src data: blob: mlxl3-preview: \(remoteSources); connect-src mlxl3-preview: \(remoteSources); frame-src 'none'; worker-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'; sandbox allow-scripts"

  static func isDocument(_ source: String) -> Bool {
    let start = source.drop(while: { $0.isWhitespace }).prefix(64).lowercased()
    return start.hasPrefix("<!doctype html") || start.hasPrefix("<html>")
      || start.hasPrefix("<html ") || start.hasPrefix("<html\n")
  }

  static func canRun(language: String?, source: String, completed: Bool) -> Bool {
    guard completed, !source.isEmpty, source.utf8.count <= maximumBytes else { return false }
    let language = language?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() ?? ""
    return ["html", "htm", "text/html"].contains(language)
      || (language.isEmpty && isDocument(source))
  }

  nonisolated static func artifactURLs(in workspace: URL) -> [URL] {
    guard
      let enumerator = FileManager.default.enumerator(
        at: workspace,
        includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey],
        options: [.skipsHiddenFiles, .skipsPackageDescendants])
    else { return [] }
    var files: [URL] = []
    for case (let index, let url as URL) in enumerator.enumerated() {
      if index >= 512 || files.count >= 32 { break }
      guard
        let values = try? url.resourceValues(forKeys: [
          .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
        ]),
        values.isSymbolicLink != true
      else {
        enumerator.skipDescendants()
        continue
      }
      if values.isRegularFile == true, ["html", "htm"].contains(url.pathExtension.lowercased()),
        (values.fileSize ?? maximumBytes + 1) <= maximumBytes
      {
        let attributes = try? FileManager.default.attributesOfItem(atPath: url.path)
        if (attributes?[.referenceCount] as? NSNumber)?.intValue == 1 { files.append(url) }
      }
    }
    return files.sorted { $0.path < $1.path }
  }

  nonisolated static func source(at file: URL, in workspace: URL) throws -> String {
    guard ["html", "htm"].contains(file.pathExtension.lowercased()) else {
      throw CocoaError(.fileReadUnknown)
    }
    let data = try data(at: file, in: workspace)
    guard let source = String(data: data, encoding: .utf8) else {
      throw CocoaError(.fileReadCorruptFile)
    }
    return source
  }

  nonisolated static func data(at file: URL, in workspace: URL) throws -> Data {
    let root = try AgentSandbox.canonicalRoot(workspace)
    let canonical = try AgentSandbox.canonicalRoot(file)
    let values = try file.resourceValues(forKeys: [
      .isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey,
    ])
    guard file.isFileURL, canonical.path.hasPrefix(root.path + "/"),
      values.isRegularFile == true, values.isSymbolicLink != true,
      (values.fileSize ?? maximumBytes + 1) <= maximumBytes
    else { throw CocoaError(.fileReadNoPermission) }
    var directory = open(root.path, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
    guard directory >= 0 else { throw CocoaError(.fileReadNoPermission) }
    defer { close(directory) }
    let components = canonical.path.dropFirst(root.path.count + 1).split(separator: "/")
    for component in components.dropLast() {
      let next = openat(
        directory, String(component), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC)
      guard next >= 0 else { throw CocoaError(.fileReadNoPermission) }
      close(directory)
      directory = next
    }
    guard let name = components.last else { throw CocoaError(.fileReadNoPermission) }
    let descriptor = openat(directory, String(name), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK)
    guard descriptor >= 0 else { throw CocoaError(.fileReadNoPermission) }
    defer { close(descriptor) }
    var info = stat()
    guard fstat(descriptor, &info) == 0, info.st_mode & mode_t(S_IFMT) == mode_t(S_IFREG),
      info.st_nlink == 1, info.st_size >= 0, info.st_size <= off_t(maximumBytes)
    else { throw CocoaError(.fileReadNoPermission) }
    let handle = FileHandle(fileDescriptor: descriptor, closeOnDealloc: false)
    let data = try handle.read(upToCount: maximumBytes + 1) ?? Data()
    guard data.count <= maximumBytes else { throw CocoaError(.fileReadTooLarge) }
    return data
  }

  static func makeWebView(source: String, workspace: URL? = nil, file: URL? = nil) async throws
    -> WKWebView
  {
    guard source.utf8.count <= maximumBytes else { throw CocoaError(.fileReadTooLarge) }
    let configuration = WKWebViewConfiguration()
    configuration.websiteDataStore = .nonPersistent()
    var documentURL = pageURL
    if let workspace, let file {
      _ = try data(at: file, in: workspace)
      let root = try AgentSandbox.canonicalRoot(workspace)
      let canonical = try AgentSandbox.canonicalRoot(file)
      documentURL = pageURL.deletingLastPathComponent().appendingPathComponent(
        String(canonical.path.dropFirst(root.path.count + 1)))
    }
    let page = PreviewPage(source: source, documentURL: documentURL, workspace: workspace)
    configuration.setURLSchemeHandler(page, forURLScheme: pageURL.scheme!)
    // Only dependency CDNs can supply remote resources; host files and LAN stay blocked.
    let patterns =
      ["^mlxl3-preview://document/", "^data:", "^blob:"]
      + remoteOrigins.map { "^" + $0.replacingOccurrences(of: ".", with: "[.]") + "/" }
    let rules: [[String: Any]] =
      [["trigger": ["url-filter": ".*"], "action": ["type": "block"]]]
      + patterns.map {
        ["trigger": ["url-filter": $0], "action": ["type": "ignore-previous-rules"]]
      }
    let encoded = String(decoding: try JSONSerialization.data(withJSONObject: rules), as: UTF8.self)
    let list = try await WKContentRuleListStore.default().compileContentRuleList(
      forIdentifier: "MLXL3HTMLDependencies-v2", encodedContentRuleList: encoded)
    try Task.checkCancellation()
    if let list {
      configuration.userContentController.add(list)
    } else {
      throw CocoaError(.fileReadCorruptFile)
    }
    let view = WKWebView(frame: .zero, configuration: configuration)
    view.navigationDelegate = page
    view.uiDelegate = page
    view.load(URLRequest(url: documentURL))
    return view
  }
}

struct HTMLPreviewDocument: Identifiable {
  let id = UUID()
  let source: String
  let workspace: URL
  let file: URL
}

@MainActor
private final class PreviewPage: NSObject, WKURLSchemeHandler, WKNavigationDelegate, WKUIDelegate {
  let source: String
  let documentURL: URL
  let workspace: URL?
  init(source: String, documentURL: URL, workspace: URL?) {
    self.source = source
    self.documentURL = documentURL
    self.workspace = workspace
  }

  func webView(_ webView: WKWebView, start urlSchemeTask: any WKURLSchemeTask) {
    do {
      guard let url = urlSchemeTask.request.url, url.scheme == documentURL.scheme,
        url.host == documentURL.host, urlSchemeTask.request.httpMethod == "GET"
      else { throw URLError(.unsupportedURL) }
      let content: Data
      let type: String
      if url == documentURL {
        content = Data(source.utf8)
        type = "text/html; charset=utf-8"
      } else {
        let types = [
          "js": "text/javascript", "mjs": "text/javascript", "css": "text/css",
          "json": "application/json", "png": "image/png", "jpg": "image/jpeg",
          "jpeg": "image/jpeg", "webp": "image/webp", "gif": "image/gif", "svg": "image/svg+xml",
          "woff": "font/woff", "woff2": "font/woff2", "gltf": "model/gltf+json",
          "glb": "model/gltf-binary", "bin": "application/octet-stream",
        ]
        guard let workspace, let mime = types[url.pathExtension.lowercased()] else {
          throw URLError(.noPermissionsToReadFile)
        }
        content = try HTMLPreview.data(
          at: workspace.appendingPathComponent(String(url.path.dropFirst())), in: workspace)
        type = mime
      }
      guard
        let response = HTTPURLResponse(
          url: url, statusCode: 200,
          httpVersion: "HTTP/1.1",
          headerFields: [
            "Content-Type": type,
            "Access-Control-Allow-Origin": "*",
            "Content-Security-Policy": HTMLPreview.policy,
            "Cache-Control": "no-store",
          ])
      else { throw URLError(.badServerResponse) }
      urlSchemeTask.didReceive(response)
      urlSchemeTask.didReceive(content)
      urlSchemeTask.didFinish()
    } catch { urlSchemeTask.didFailWithError(error) }
  }

  func webView(_ webView: WKWebView, stop urlSchemeTask: any WKURLSchemeTask) {}

  func webView(
    _ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction,
    decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void
  ) {
    var target = navigationAction.request.url.flatMap {
      URLComponents(url: $0, resolvingAgainstBaseURL: false)
    }
    target?.fragment = nil
    decisionHandler(
      target?.url == documentURL
        && !navigationAction.shouldPerformDownload ? .allow : .cancel)
  }

  func webView(
    _ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
    for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures
  ) -> WKWebView? { nil }

  func webView(
    _ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters,
    initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping @MainActor ([URL]?) -> Void
  ) {
    completionHandler(nil)
  }

  func webView(
    _ webView: WKWebView, requestMediaCapturePermissionFor origin: WKSecurityOrigin,
    initiatedByFrame frame: WKFrameInfo, type: WKMediaCaptureType,
    decisionHandler: @escaping @MainActor (WKPermissionDecision) -> Void
  ) { decisionHandler(.deny) }
}

struct HTMLPreviewSheet: View {
  let source: String
  var workspace: URL? = nil
  var file: URL? = nil
  var body: some View {
    ArtifactStudioView(source: source, workspace: workspace, file: file)
  }
}

struct HTMLWebView: NSViewRepresentable {
  let view: WKWebView
  func makeNSView(context: Context) -> WKWebView { view }
  func updateNSView(_ nsView: WKWebView, context: Context) {}
  static func dismantleNSView(_ nsView: WKWebView, coordinator: ()) {
    nsView.stopLoading()
    nsView.navigationDelegate = nil
    nsView.uiDelegate = nil
    nsView.loadHTMLString("", baseURL: nil)
  }
}
