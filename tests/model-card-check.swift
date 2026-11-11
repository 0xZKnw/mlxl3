import AppKit
import Foundation
import SwiftUI
import WebKit

@main struct ModelCardCheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        NSApp.setActivationPolicy(.accessory)
        var checks = 0
        func check(_ condition: @autoclosure () -> Bool, _ message: String) {
            checks += 1
            precondition(condition(), message)
        }
        let fixture = try String(contentsOfFile: "tests/fixtures/model-card.md", encoding: .utf8)
        let id = "author/model"
        let revision = String(repeating: "a", count: 40)
        let base = ModelCardRenderer.baseURL(repo: id, commit: revision)!
        check(base.absoluteString == "https://huggingface.co/author/model/resolve/\(revision)/", "Wrong revision-pinned resource base")
        for repo in ["", "author", "author/../secret", "a/..", "a/b?token=x", "https://localhost/x"] {
            check(ModelCardRenderer.baseURL(repo: repo, commit: revision) == nil, "Unsafe model id accepted")
        }
        for sha in ["main", "", String(repeating: "g", count: 40), String(repeating: "a", count: 39)] {
            check(ModelCardRenderer.baseURL(repo: id, commit: sha) == nil, "Unpinned revision accepted")
        }
        for url in ["file:///etc/passwd", "javascript:alert(1)", "https://name:secret@example.org/"] {
            check(ModelCardRenderer.externalURL(URL(string: url)!) == nil, "Unsafe link accepted")
        }
        check(ModelCardRenderer.externalURL(URL(string: "https://example.org/a")!) != nil, "Ordinary HTTPS link rejected")
        check(!ModelCardRenderer.content(fixture).contains("license: apache-2.0"), "YAML metadata shown as content")
        check(ModelCardRenderer.content("---\r\nlicense: mit\r\n---\r\n# Good").hasPrefix("# Good"), "CRLF frontmatter not stripped")
        check(ModelCardRenderer.content("\u{feff}---\nlicense: mit\n---\n# Good").hasPrefix("# Good"), "BOM frontmatter not stripped")
        check(ModelCardRenderer.content("---\nordinary prose\n---\n# Keep") == "---\nordinary prose\n---\n# Keep", "Non-metadata rules were removed")
        check(ModelCardRenderer.content("---\nlicense: mit\nno closing fence").contains("license: mit"), "Unclosed metadata lost content")
        let html = try ModelCardRenderer.document(source: fixture, dark: true)
        check(html.contains("<h1>Example model</h1>"), "Markdown heading not rendered")
        check(html.contains("<strong>Markdown</strong>") && html.contains("<table>"), "Markdown and HTML table not rendered")
        check(html.contains("class=\"card-panel\"") && html.contains("<style>"), "HTML styles lost")
        check(html.contains("&lt;div&gt;This is source code"), "Fenced code became active HTML")
        check(!html.contains("base_model: example/base"), "YAML frontmatter leaked")
        do {
            _ = try ModelCardRenderer.document(source: String(repeating: "x", count: ModelCardRenderer.maximumBytes + 1), dark: false)
            preconditionFailure("Oversized card accepted")
        } catch { checks += 1 }

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-card-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let metadata = root.appendingPathComponent("server.json")
        let server = Process()
        server.executableURL = URL(fileURLWithPath: CommandLine.arguments[1])
        server.arguments = [CommandLine.arguments[2], metadata.path]
        try server.run()
        defer { if server.isRunning { server.terminate() }; server.waitUntilExit() }
        for _ in 0..<500 where !FileManager.default.fileExists(atPath: metadata.path) && server.isRunning {
            try await Task.sleep(for: .milliseconds(10))
        }
        let info = try JSONSerialization.jsonObject(with: Data(contentsOf: metadata)) as! [String: Any]
        let local = "http://127.0.0.1:\(info["port"]!)"
        let hostile = fixture + "\n<script>window.cardPwned = true; fetch('\(local)/script')</script>\n" +
            "<img src=\"\(local)/image\" onerror=\"window.cardPwned=true\">\n" +
            "<style>@import url('\(local)/style');</style>\n<iframe src=\"\(local)/frame\"></iframe>"
        let host = NSHostingView(rootView: ModelCardView(source: hostile, repo: id, commit: revision)
            .frame(width: 800, height: 600).preferredColorScheme(.dark))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 800, height: 600),
                              styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        host.frame = NSRect(x: 0, y: 0, width: 800, height: 600)
        defer { window.close() }
        func webViews(_ view: NSView) -> [WKWebView] {
            (view as? WKWebView).map { [$0] } ?? view.subviews.flatMap(webViews)
        }
        var card: WKWebView?
        for _ in 0..<500 {
            host.layoutSubtreeIfNeeded()
            if let view = webViews(host).first, !view.isLoading,
               (try? await view.evaluateJavaScript("document.readyState") as? String) == "complete",
               (try? await view.evaluateJavaScript("Boolean(document.getElementById('card-panel'))") as? Bool) == true {
                card = view
                break
            }
            try await Task.sleep(for: .milliseconds(20))
        }
        guard let card else { preconditionFailure("Real model card WebKit view did not load") }
        check(!card.configuration.defaultWebpagePreferences.allowsContentJavaScript, "Card JS enabled")
        let panelText = try await card.evaluateJavaScript("document.getElementById('card-panel').innerText") as? String
        check(panelText?.contains("HTML quantization") == true && panelText?.contains("4.00") == true, "HTML card content missing")
        let color = try await card.evaluateJavaScript("getComputedStyle(document.getElementById('card-panel')).color") as? String
        check(color == "rgb(10, 180, 70)", "Card CSS did not render")
        let link = try await card.evaluateJavaScript("document.querySelector('a').href") as? String
        check(link == base.appendingPathComponent("docs/guide.md").absoluteString, "Relative card link resolved incorrectly")
        let image = try await card.evaluateJavaScript("document.getElementById('local-image').src") as? String
        check(image == base.appendingPathComponent("graph.svg").absoluteString, "Relative card image resolved incorrectly")
        let attack = try await card.evaluateJavaScript("typeof window.cardPwned") as? String
        check(attack == "undefined", "Untrusted card script executed")
        try await Task.sleep(for: .milliseconds(250))
        let (data, _) = try await URLSession.shared.data(from: URL(string: local + "/requests")!)
        let requests = try JSONDecoder().decode([String].self, from: data)
        check(requests.isEmpty, "Untrusted card reached loopback: \(requests)")
        if let output = ProcessInfo.processInfo.environment["MLXL3_MODEL_CARD_SNAPSHOTS"] {
            let image = try await card.takeSnapshot(configuration: nil)
            guard let data = image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:))?.representation(using: .png, properties: [:]) else {
                preconditionFailure("Card bitmap missing")
            }
            try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("model-card-fixture.png"))
        }
        if let sample = ProcessInfo.processInfo.environment["MLXL3_MODEL_CARD_SAMPLE"] {
            let detail = try JSONDecoder().decode(HubDetails.self, from: Data(contentsOf: URL(fileURLWithPath: sample)))
            let actual = try ModelCardRenderer.document(source: detail.readme, dark: true)
            check(actual.contains("dashboard-container") && actual.contains("data-table"), "Real Qwen card structure missing")
            check(!actual.contains("base_model_relation: quantized"), "Real Qwen card metadata leaked")
            host.rootView = ModelCardView(source: detail.readme, repo: detail.id, commit: detail.commit)
                .frame(width: 800, height: 600).preferredColorScheme(.dark)
            var actualCard: WKWebView?
            for _ in 0..<750 {
                host.layoutSubtreeIfNeeded()
                if let candidate = webViews(host).first,
                   (try? await candidate.evaluateJavaScript("Boolean(document.querySelector('.dashboard-header h1'))") as? Bool) == true,
                   (try? await candidate.evaluateJavaScript("document.querySelector('.repo-graph').naturalWidth > 0") as? Bool) == true {
                    actualCard = candidate
                    break
                }
                try await Task.sleep(for: .milliseconds(20))
            }
            guard let actualCard else { preconditionFailure("Real Qwen model card or its graph did not load") }
            let actualText = try await actualCard.evaluateJavaScript("document.body.innerText") as? String
            check(actualText?.contains("Qwen3.6-35B-A3B") == true && actualText?.contains("12.10") == true,
                  "Real Qwen card content missing")
            if let output = ProcessInfo.processInfo.environment["MLXL3_MODEL_CARD_SNAPSHOTS"] {
                let image = try await actualCard.takeSnapshot(configuration: nil)
                let data = image.tiffRepresentation.flatMap(NSBitmapImageRep.init(data:))!.representation(using: .png, properties: [:])!
                try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("model-card-qwen35.png"))
            }
        }
        print("Model card checks passed: \(checks), frontmatter/Markdown/HTML, pinned links/images, safe WebKit and loopback blocking")
    }
}
