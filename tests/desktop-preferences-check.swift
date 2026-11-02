import AppKit
import SwiftUI

extension StudioModel {
    fileprivate var fixtureRunning: Bool { bridge.isRunning }
}

@main struct DesktopPreferencesCheck {
    @MainActor static func main() async throws {
        var checks = 0
        func check(_ condition: Bool, _ message: String) {
            checks += 1
            precondition(condition, message)
        }
        func wait(_ label: String, until condition: () -> Bool) async throws {
            for _ in 0..<300 {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw NSError(domain: "DesktopPreferencesCheck", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Deadline: " + label])
        }
        let suite = "io.mlxl3.preferences-check." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(suite)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var instances: [StudioModel] = []
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        defer {
            for studio in instances { studio.ejectModel() }
            preferences.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
            unsetenv("MLXL3_FIXTURE_MODELS"); unsetenv("MLXL3_FIXTURE_LIST_DELAY")
        }
        let file = root.appendingPathComponent("conversations.json")
        func make(preview: Bool = false) -> StudioModel {
            let studio = StudioModel(conversationFileURL: file,
                updateManager: UpdateManager(releasesURL: root.appendingPathComponent("no-network.json")),
                isPreview: preview, preferences: preferences)
            instances.append(studio)
            return studio
        }
        check(AppTheme.load(from: preferences) == .dark, "Keep existing dark default")
        check(StartupModelPreference.load(from: preferences), "Keep existing auto-load default")
        let studio = make(preview: true)
        studio.draft = "Unsent text"; let conversation = studio.selectedConversationID
        for theme in AppTheme.allCases {
            studio.setTheme(theme)
            check(make(preview: true).theme == theme, "Theme not persisted")
            check(studio.draft == "Unsent text" && studio.selectedConversationID == conversation,
                  "Changing appearance changed conversation or draft")
            check(!theme.title.isEmpty, "Missing theme title")
        }
        check(AppTheme.system.colorScheme == nil && AppTheme.light.colorScheme == .light
              && AppTheme.dark.colorScheme == .dark, "Incorrect scheme mapping")
        for invalid in ["", "sepia", true, 4] as [Any] {
            preferences.set(invalid, forKey: AppTheme.preferenceKey)
            check(AppTheme.load(from: preferences) == .dark, "Invalid theme must fall back")
        }
        for enabled in [false, true, false] {
            studio.setLoadModelOnStartup(enabled)
            check(make(preview: true).loadModelOnStartup == enabled, "Startup option not persisted")
        }
        for invalid in ["false", "", 0, 1, 3, 0.5, [false]] as [Any] {
            preferences.set(invalid, forKey: StartupModelPreference.key)
            check(StartupModelPreference.load(from: preferences), "Invalid startup value must use default")
        }
        for dark in [false, true] {
            let appearance = NSAppearance(named: dark ? .darkAqua : .aqua)!
            appearance.performAsCurrentDrawingAppearance {
                func luminance(_ color: Color) -> CGFloat {
                    let c = NSColor(color).usingColorSpace(.sRGB)!
                    return (c.redComponent + c.greenComponent + c.blueComponent) / 3
                }
                check(dark ? luminance(StudioTheme.ink) > luminance(StudioTheme.canvas)
                           : luminance(StudioTheme.ink) < luminance(StudioTheme.canvas), "Text contrast inverted")
                check(dark ? luminance(StudioTheme.onAccent) < luminance(StudioTheme.accent)
                           : luminance(StudioTheme.onAccent) > luminance(StudioTheme.accent), "Button text contrast inverted")
            }
        }
        // Render actual settings and rich text, not a parallel palette implementation.
        _ = NSApplication.shared
        let host = NSHostingView(rootView: AppSettingsView().environmentObject(studio))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 650, height: 560),
                              styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        host.frame = NSRect(x: 0, y: 0, width: 650, height: 560)
        defer { window.close() }
        for theme in [AppTheme.light, .dark, .light] {
            studio.setTheme(theme)
            try await Task.sleep(for: .milliseconds(150))
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                preconditionFailure("Settings did not render")
            }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            check(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0, "Empty settings bitmap")
            let corner = bitmap.colorAt(x: 10, y: 10)!.usingColorSpace(.sRGB)!
            check(theme == .light ? corner.redComponent > 0.7 : corner.redComponent < 0.3,
                  "Rendered settings did not follow selected theme")
            if let output = ProcessInfo.processInfo.environment["MLXL3_PREFERENCE_SNAPSHOTS"],
               let data = bitmap.representation(using: .png, properties: [:]) {
                try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("settings-" + theme.rawValue + ".png"))
            }
        }
        // System follows application appearance without a forced scheme.
        studio.setTheme(.system)
        for dark in [false, true] {
            NSApplication.shared.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
            window.appearance = nil
            try await Task.sleep(for: .milliseconds(150))
            host.layoutSubtreeIfNeeded()
            let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let corner = bitmap.colorAt(x: 10, y: 10)!.usingColorSpace(.sRGB)!
            check(dark ? corner.redComponent < 0.3 : corner.redComponent > 0.7,
                  "System theme did not follow application appearance")
        }
        NSApplication.shared.appearance = nil
        let richText = MarkdownResponseView("# Theme check\n\nA formula: $x^2 + y^2$.\n\n```swift\n// Comment\nlet value = 42\nprint(\"Hello\")\n```\n\n| A | B |\n| --- | --- |\n| 1 | 2 |")
        let richHost = NSHostingView(rootView: ScrollView { richText.padding(24) }
            .frame(width: 700, height: 520).background(StudioTheme.canvas))
        let richWindow = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 700, height: 520),
                                  styleMask: .borderless, backing: .buffered, defer: false)
        richWindow.isReleasedWhenClosed = false
        richWindow.contentView = richHost
        richHost.frame = NSRect(x: 0, y: 0, width: 700, height: 520)
        defer { richWindow.close() }
        for dark in [false, true] {
            richWindow.appearance = NSAppearance(named: dark ? .darkAqua : .aqua)
            try await Task.sleep(for: .milliseconds(250))
            richHost.layoutSubtreeIfNeeded()
            let bitmap = richHost.bitmapImageRepForCachingDisplay(in: richHost.bounds)!
            richHost.cacheDisplay(in: richHost.bounds, to: bitmap)
            check(bitmap.pixelsWide >= 700 && bitmap.pixelsHigh >= 520, "Rich text canvas did not render")
            if let output = ProcessInfo.processInfo.environment["MLXL3_PREFERENCE_SNAPSHOTS"],
               let data = bitmap.representation(using: .png, properties: [:]) {
                try data.write(to: URL(fileURLWithPath: output).appendingPathComponent(dark ? "rich-dark.png" : "rich-light.png"))
            }
        }
        func model(_ name: String) -> LocalModel {
            LocalModel(name: name, path: root.appendingPathComponent(name).path,
                       modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1,
                       modules: 1, addedAt: "", size: "1 B")
        }
        let list = String(decoding: try JSONEncoder().encode([model("first"), model("second")]), as: UTF8.self)
        setenv("MLXL3_FIXTURE_MODELS", list, 1)
        let configured = make()
        configured.setStartupModelPath(model("second").path)
        check(make(preview: true).startupModelPath == model("second").path, "Startup model was not persisted")
        configured.start()
        try await wait("explicit startup model", until: { configured.engineState.isReady })
        check(configured.loadedModel?.name == "second", "Explicit startup model ignored")
        configured.ejectModel()
        preferences.set("/missing/model", forKey: "studio.startupModelPath")
        let missing = make(); missing.start()
        try await wait("missing startup choice", until: { missing.models.count == 2 })
        check(!missing.fixtureRunning && missing.engineState == .idle, "Missing choice loaded a different model")
        missing.selectModel("first")
        try await wait("manual recovery from missing choice", until: { missing.engineState.isReady })
        missing.ejectModel()
        preferences.removeObject(forKey: "studio.startupModelPath")
        preferences.set(false, forKey: StartupModelPreference.key)
        let off = make(); off.start()
        try await wait("off lists models", until: { off.models.count == 2 })
        check(!off.fixtureRunning && off.engineState == .idle, "Disabled startup loaded a model")
        check(off.selectedModelName == "first", "No-model selection fallback lost")
        off.selectModel("second")
        try await wait("manual load when off", until: { off.engineState.isReady })
        check(off.fixtureRunning, "Manual model loading disabled by startup option")
        off.setLoadModelOnStartup(true)
        check(off.engineState.isReady, "Toggling preference must not interrupt loaded model")
        off.persistNow(); off.ejectModel()
        let on = make(); on.start()
        try await wait("last selected model startup", until: { on.engineState.isReady })
        check(on.selectedModelName == "second" && on.loadedModel?.name == "second", "Last selected model not restored")
        on.setLoadModelOnStartup(false)
        check(on.fixtureRunning, "Disabling startup ejected current model")
        on.ejectModel(); on.start()
        try await Task.sleep(for: .milliseconds(300))
        check(!on.fixtureRunning, "Repeated onAppear restarted model")
        preferences.set(true, forKey: StartupModelPreference.key)
        setenv("MLXL3_FIXTURE_LIST_DELAY", "1", 1)
        let race = make(); race.start(); race.setLoadModelOnStartup(false)
        try await wait("pending list with opt-out", until: { race.models.count == 2 })
        check(!race.fixtureRunning, "Late startup callback ignored opt-out")
        let enableLater = make(); enableLater.start(); enableLater.setLoadModelOnStartup(true)
        try await wait("pending list with opt-in", until: { enableLater.models.count == 2 })
        check(!enableLater.fixtureRunning, "Opt-in should apply to next launch")
        unsetenv("MLXL3_FIXTURE_LIST_DELAY")
        let preview = make(preview: true); preview.start()
        check(preview.models.isEmpty && !preview.fixtureRunning, "Preview launched engine")
        setenv("MLXL3_FIXTURE_MODELS", "[]", 1)
        let empty = make(); empty.start()
        try await Task.sleep(for: .milliseconds(350))
        check(empty.models.isEmpty && empty.engineState == .idle && !empty.fixtureRunning, "Empty library started engine")
        setenv("MLXL3_FIXTURE_MODELS", "invalid-json", 1)
        let invalid = make(); invalid.start()
        try await wait("invalid catalogue", until: { if case .failed = invalid.engineState { true } else { false } })
        check(!invalid.fixtureRunning, "Invalid catalogue started engine")
        print("Desktop preference checks passed: \(checks) assertions, isolated persistence, appearance, actual startup/manual loading and errors")
    }
}
