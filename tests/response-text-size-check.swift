import AppKit
import SwiftUI

private struct LiveTextSizeFixture: View {
    @ObservedObject var studio: StudioModel
    let source: String
    var body: some View {
        MarkdownResponseView(source, textScale: CGFloat(studio.responseTextSize / ResponseTextSize.defaultValue)).equatable()
            .frame(width: 340, alignment: .leading)
            .padding(16)
            .background(StudioTheme.canvas(for: .midnightBlue))
            .studioAppearance(.midnightBlue)
    }
}

@main struct ResponseTextSizeCheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        var checks = 0
        func check(_ condition: Bool, _ label: String) throws {
            checks += 1
            guard condition else { throw NSError(domain: "ResponseTextSizeCheck", code: 1,
                userInfo: [NSLocalizedDescriptionKey: label]) }
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-text-size-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let suite = "io.mlxl3.text-size." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        func make() -> StudioModel {
            StudioModel(conversationFileURL: root.appendingPathComponent("history.json"),
                updateManager: UpdateManager(releasesURL: root.appendingPathComponent("no-network.json")),
                isPreview: true, preferences: prefs)
        }
        let studio = make()
        try check(studio.responseTextSize == 15, "Default changed")
        studio.conversations = [Conversation(title: "Fixture", messages: [
            ChatMessage(role: .user, content: "Question à conserver"),
            ChatMessage(role: .assistant, content: "Réponse à conserver")])]
        studio.selectedConversationID = studio.conversations[0].id
        studio.selectedModelName = "fixture-model"
        studio.engineState = .generating
        let originalID = studio.selectedConversationID
        studio.draft = "Brouillon e\u{301} 👩‍💻"
        let originalMessages = studio.conversations.map { $0.messages.map(\.content) }
        for point in 12...24 {
            studio.setResponseTextSize(Double(point))
            try check(make().responseTextSize == Double(point), "Size did not persist")
            try check(studio.draft == "Brouillon e\u{301} 👩‍💻" && studio.selectedConversationID == originalID
                      && studio.conversations.map { $0.messages.map(\.content) } == originalMessages
                      && studio.selectedModelName == "fixture-model" && studio.engineState == .generating,
                      "Changing size changed conversation data")
        }
        for (input, expected) in [(-100.0, 12.0), (100.0, 24.0), (16.6, 17.0),
                                  (.nan, 15.0), (.infinity, 15.0), (-.infinity, 15.0)] {
            studio.setResponseTextSize(input)
            try check(studio.responseTextSize == expected && make().responseTextSize == expected,
                      "Invalid or fractional input was not normalized")
        }
        for invalid in ["24", true, [15]] as [Any] {
            prefs.set(invalid, forKey: ResponseTextSize.preferenceKey)
            try check(make().responseTextSize == 15, "Malformed saved value accepted")
        }
        for (saved, expected) in [(-100.0, 12.0), (100.0, 24.0)] {
            prefs.set(saved, forKey: ResponseTextSize.preferenceKey)
            try check(make().responseTextSize == expected, "Saved size escaped bounds")
        }
        let documents = [
            ("prose", String(repeating: "Une réponse lisible avec des accents et du texte. ", count: 8)),
            ("heading", "# Une réponse lisible\n\n## Un sous-titre"),
            ("code", "```swift\nlet nom = \"Bonjour\"\nprint(nom)\nlet x = 42\n```"),
            ("table", "| Nom | Valeur |\n| --- | --- |\n| Bonjour | 42 |\n| Monde | 73 |"),
            ("math", "$$\nx^2 + y^2 = z^2\n$$"),
        ]
        for (name, source) in documents {
            studio.setResponseTextSize(15)
            let host = NSHostingView(rootView: LiveTextSizeFixture(studio: studio, source: source))
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 372, height: 800),
                styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false; window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: 372, height: 800)
            defer { window.close() }
            func height() async throws -> CGFloat {
                try await Task.sleep(for: .milliseconds(200))
                var previous: CGFloat = -1
                var stable = 0
                for _ in 0..<100 {
                    try await Task.sleep(for: .milliseconds(30)); host.layoutSubtreeIfNeeded()
                    let value = host.fittingSize.height
                    if value > 40 && abs(value - previous) < 0.1 { stable += 1 } else { stable = 0 }
                    if stable >= 3 { return value }
                    previous = value
                }
                throw NSError(domain: "ResponseTextSizeCheck", code: 2,
                              userInfo: [NSLocalizedDescriptionKey: "Render deadline: " + name])
            }
            let normal = try await height()
            studio.setResponseTextSize(24)
            let large = try await height()
            studio.setResponseTextSize(12)
            let small = try await height()
            studio.setResponseTextSize(15)
            let restored = try await height()
            print("\(name): small=\(small), normal=\(normal), large=\(large), reset=\(restored)")
            fflush(nil)
            try check(large > normal + 2 && small < normal - 2, name + " did not update live through equatable views")
            try check(abs(restored - normal) < 0.5, name + " did not return to default")
        }
        // Verify the production root injects the setting into existing messages.
        studio.engineState = .idle
        studio.setResponseTextSize(15)
        let liveHost = NSHostingView(rootView: StudioView().environmentObject(studio).studioAppearance(.midnightBlue))
        let liveWindow = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1100, height: 800),
            styleMask: .borderless, backing: .buffered, defer: false)
        liveWindow.isReleasedWhenClosed = false; liveWindow.contentView = liveHost
        liveHost.frame = NSRect(x: 0, y: 0, width: 1100, height: 800)
        func messagePixels() async throws -> Data {
            try await Task.sleep(for: .milliseconds(400)); liveHost.layoutSubtreeIfNeeded()
            // The upper message area excludes the blinking composer caret.
            let rect = NSRect(x: 250, y: liveHost.isFlipped ? 80 : 320, width: 800, height: 400)
            let bitmap = liveHost.bitmapImageRepForCachingDisplay(in: rect)!
            liveHost.cacheDisplay(in: rect, to: bitmap)
            return bitmap.representation(using: .png, properties: [:])!
        }
        let normalPixels = try await messagePixels()
        studio.setResponseTextSize(24)
        let largePixels = try await messagePixels()
        studio.setResponseTextSize(15)
        let resetPixels = try await messagePixels()
        liveWindow.close()
        try check(normalPixels != largePixels, "Production StudioView did not update its messages")
        try check(normalPixels == resetPixels, "Production messages did not return to their original rendering")
        studio.setReadingWidth(.wide)
        try check(studio.responseTextSize == 15, "Changing width changed text size")
        if let output = ProcessInfo.processInfo.environment["MLXL3_TEXT_SIZE_SNAPSHOTS"] {
            for size in [12, 15, 24] {
                studio.setResponseTextSize(Double(size))
                let host = NSHostingView(rootView: ConversationDisplaySettings().environmentObject(studio)
                    .padding(20).frame(width: 300).background(StudioTheme.canvas(for: .midnightBlue))
                    .studioAppearance(.midnightBlue))
                let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 300, height: 340),
                    styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false; window.contentView = host
                host.frame = NSRect(x: 0, y: 0, width: 300, height: 340)
                try await Task.sleep(for: .milliseconds(100)); host.layoutSubtreeIfNeeded()
                let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                host.cacheDisplay(in: host.bounds, to: bitmap)
                try bitmap.representation(using: .png, properties: [:])!.write(
                    to: URL(fileURLWithPath: output).appendingPathComponent("display-\(size).png"))
                window.close()
            }
        }
        print("Response text size checks passed: \(checks) assertions, 13 positions, persistence/errors, live prose/heading/code/table/math")
    }
}
