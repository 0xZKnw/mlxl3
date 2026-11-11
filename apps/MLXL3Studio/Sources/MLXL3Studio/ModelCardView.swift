import AppKit
import JavaScriptCore
import SwiftUI
import WebKit

enum ModelCardRenderer {
    static let maximumBytes = 256_000

    // Hugging Face metadata is not part of the rendered model card.
    static func content(_ source: String) -> String {
        var normalized = source.replacingOccurrences(of: "\r\n", with: "\n")
        if normalized.first == "\u{feff}" { normalized.removeFirst() }
        let lines = normalized.components(separatedBy: "\n")
        guard lines.first?.trimmingCharacters(in: .whitespaces) == "---",
              let end = lines.indices.dropFirst().first(where: { lines[$0].trimmingCharacters(in: .whitespaces) == "---" }),
              end > 1,
              lines[1..<end].contains(where: { line in
                  guard let colon = line.firstIndex(of: ":") else { return false }
                  let key = line[..<colon].trimmingCharacters(in: .whitespaces)
                  return !key.isEmpty && key.unicodeScalars.allSatisfy {
                      CharacterSet.alphanumerics.contains($0) || $0 == "_" || $0 == "-"
                  }
              }) else { return normalized }
        return lines[(end + 1)...].joined(separator: "\n").trimmingCharacters(in: .newlines)
    }

    static func baseURL(repo: String, commit: String) -> URL? {
        let parts = repo.split(separator: "/", omittingEmptySubsequences: false)
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-")
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." && $0.unicodeScalars.allSatisfy(allowed.contains) }),
              commit.utf8.count == 40, commit.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) })
        else { return nil }
        return URL(string: "https://huggingface.co/")!
            .appendingPathComponent(String(parts[0]), isDirectory: true)
            .appendingPathComponent(String(parts[1]), isDirectory: true)
            .appendingPathComponent("resolve", isDirectory: true)
            .appendingPathComponent(commit, isDirectory: true)
    }

    static func externalURL(_ url: URL) -> URL? {
        guard let parts = URLComponents(url: url, resolvingAgainstBaseURL: false),
              ["http", "https"].contains(parts.scheme?.lowercased() ?? ""),
              parts.host != nil, parts.user == nil, parts.password == nil else { return nil }
        return url
    }

    private static func parserSource() throws -> String {
        let local = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appending(path: "Resources/marked.umd.js")
        let candidates = [Bundle.main.url(forResource: "marked.umd", withExtension: "js"), local]
        guard let file = candidates.compactMap({ $0 }).first(where: { FileManager.default.fileExists(atPath: $0.path) }) else {
            throw CocoaError(.fileNoSuchFile)
        }
        return try String(contentsOf: file, encoding: .utf8)
    }

    static func document(source: String, dark: Bool) throws -> String {
        guard source.utf8.count <= maximumBytes else { throw CocoaError(.fileReadTooLarge) }
        guard let context = JSContext() else { throw CocoaError(.fileReadUnknown) }
        context.evaluateScript(try parserSource())
        guard context.exception == nil,
              let parse = context.objectForKeyedSubscript("marked")?.objectForKeyedSubscript("parse"),
              let html = parse.call(withArguments: [content(source)])?.toString(), context.exception == nil else {
            throw CocoaError(.fileReadCorruptFile)
        }
        let foreground = dark ? "#e8e8e8" : "#202020"
        let background = dark ? "#121212" : "#f7f7f7"
        let border = dark ? "#444" : "#ccc"
        let link = dark ? "#8dc8ff" : "#145da0"
        return """
        <!doctype html><html><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; img-src data: https://huggingface.co https://*.huggingface.co https://*.hf.co https://raw.githubusercontent.com https://github.com; font-src data:; connect-src 'none'; frame-src 'none'; object-src 'none'; media-src 'none'; base-uri 'none'; form-action 'none'; sandbox">
        <style>
        :root { color-scheme: \(dark ? "dark" : "light"); }
        body { margin: 0; padding: 8px 4px 24px; color: \(foreground); background: \(background); font: 14px -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.55; overflow-wrap: anywhere; }
        h1,h2,h3,h4 { line-height: 1.25; margin: 1.1em 0 .5em; } h1 { font-size: 1.7em; } h2 { font-size: 1.35em; } h3 { font-size: 1.15em; }
        a { color: \(link); } img { max-width: 100%; height: auto; } pre { overflow: auto; padding: 12px; border: 1px solid \(border); border-radius: 7px; }
        code { font-family: ui-monospace, monospace; } blockquote { margin-left: 0; padding-left: 14px; border-left: 3px solid \(border); }
        table { border-collapse: collapse; max-width: 100%; } th,td { padding: 7px 10px; border: 1px solid \(border); } hr { border: 0; border-top: 1px solid \(border); }
        iframe, object, embed, form { display: none !important; }
        </style></head><body>\(html)</body></html>
        """
    }
}

struct ModelCardView: View {
    let source: String
    let repo: String
    let commit: String
    @Environment(\.colorScheme) private var colorScheme
    @State private var html: String?
    @State private var failed = false

    var body: some View {
        Group {
            if let html, let base = ModelCardRenderer.baseURL(repo: repo, commit: commit) {
                ModelCardWebView(html: html, baseURL: base)
            } else if failed || ModelCardRenderer.baseURL(repo: repo, commit: commit) == nil {
                Text(L("Impossible d’afficher la fiche du modèle.", "Unable to display the model card."))
                    .foregroundStyle(StudioTheme.secondary).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            } else {
                ProgressView().frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .task(id: source + (colorScheme == .dark ? "-dark" : "-light")) {
            html = nil
            failed = false
            let dark = colorScheme == .dark
            let result = await Task.detached(priority: .userInitiated) {
                try? ModelCardRenderer.document(source: source, dark: dark)
            }.value
            guard !Task.isCancelled else { return }
            html = result
            failed = result == nil
        }
    }
}

struct ModelCardWebView: NSViewRepresentable {
    let html: String
    let baseURL: URL

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.websiteDataStore = .nonPersistent()
        configuration.defaultWebpagePreferences.allowsContentJavaScript = false
        configuration.preferences.javaScriptCanOpenWindowsAutomatically = false
        let view = WKWebView(frame: .zero, configuration: configuration)
        view.navigationDelegate = context.coordinator
        view.uiDelegate = context.coordinator
        return view
    }

    func updateNSView(_ view: WKWebView, context: Context) {
        guard context.coordinator.html != html || context.coordinator.baseURL != baseURL else { return }
        context.coordinator.html = html
        context.coordinator.baseURL = baseURL
        context.coordinator.loading = true
        view.loadHTMLString(html, baseURL: baseURL)
    }

    static func dismantleNSView(_ view: WKWebView, coordinator: Coordinator) {
        view.stopLoading()
        view.navigationDelegate = nil
        view.uiDelegate = nil
        view.loadHTMLString("", baseURL: nil)
    }

    @MainActor final class Coordinator: NSObject, WKNavigationDelegate, WKUIDelegate {
        var html = ""
        var baseURL: URL?
        var loading = false

        func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) { loading = false }

        func webView(_ webView: WKWebView, decidePolicyFor action: WKNavigationAction,
                     decisionHandler: @escaping @MainActor (WKNavigationActionPolicy) -> Void) {
            guard let url = action.request.url else { decisionHandler(.cancel); return }
            if loading && (url.scheme == "about" || url == baseURL) {
                decisionHandler(.allow)
            } else if action.navigationType == .linkActivated, let external = ModelCardRenderer.externalURL(url) {
                NSWorkspace.shared.open(external)
                decisionHandler(.cancel)
            } else {
                decisionHandler(.cancel)
            }
        }

        func webView(_ webView: WKWebView, createWebViewWith configuration: WKWebViewConfiguration,
                     for navigationAction: WKNavigationAction, windowFeatures: WKWindowFeatures) -> WKWebView? { nil }
    }
}
