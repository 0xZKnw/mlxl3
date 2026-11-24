import AppKit
import SwiftUI

extension StudioModel {
    fileprivate var fixtureRunning: Bool { bridge.isRunning }
    /// Stands for an engine whose ready event announced `reasoning_effort_supported`.
    fileprivate func fixtureReasoningEngine(_ supported: Bool, levels: Set<ReasoningEffort> = ReasoningEffort.legacyLevels) {
        engineReasoningSupported = supported
        engineReasoningLevels = levels
    }
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
        check(studio.mcpConfigurationURL.deletingLastPathComponent() == file.deletingLastPathComponent(),
              "UI preview MCP configuration must stay in the disposable workspace")
        studio.draft = "Unsent text"; let conversation = studio.selectedConversationID
        check(AppTheme.allCases == [.system, .light, .dark], "Only the three supported appearances are selectable")
        for (names, expected) in [(["paper", "sand", "lavender"], AppTheme.light),
                                  (["midnightBlue", "graphite", "coffee", "forest", "plum", "ocean"], .dark)] {
            for name in names {
                preferences.set(name, forKey: AppTheme.preferenceKey)
                check(AppTheme.load(from: preferences) == expected, "Lost legacy appearance for " + name)
            }
        }
        check(AppAccent.load(from: preferences) == .violet, "Default aurora accent")
        for (name, expected) in [("mint", AppAccent.neutral), ("sage", .neutral), ("coral", .rose),
                                  ("cyan", .violet), ("blue", .violet), ("indigo", .violet)] {
            preferences.set(name, forKey: AppAccent.preferenceKey)
            check(AppAccent.load(from: preferences) == expected, "Unexpected legacy accent " + name)
        }
        // Styles: Graphite by default; each style persists and brings its accent; an accent
        // picked afterwards (or a custom colour) sticks; unknown values fall back.
        check(AppStyle.allCases == [.graphite, .minimal, .auroraASCII, .aurora], "Unexpected style list")
        check(AppStyle.load(from: preferences) == .graphite, "Graphite must be the default style")
        for style in AppStyle.allCases {
            studio.setStyle(style)
            let reopened = make(preview: true)
            check(reopened.style == style && StudioTheme.style == style, "Style not persisted: " + style.rawValue)
            check(reopened.accent == style.accent, "Style did not bring its accent: " + style.rawValue)
            check(studio.draft == "Unsent text" && studio.selectedConversationID == conversation,
                  "Changing style changed conversation or draft")
            check(!style.title.isEmpty && !style.detail.isEmpty, "Missing style text")
        }
        check(AppStyle.graphite.isGraphite && AppStyle.minimal.isGraphite
              && !AppStyle.auroraASCII.isGraphite && !AppStyle.aurora.isGraphite, "Style families mixed up")
        studio.setStyle(.auroraASCII)
        studio.setAccent(.amber)
        check(make(preview: true).accent == .amber && make(preview: true).style == .auroraASCII,
              "An accent chosen after the style must stick")
        preferences.removeObject(forKey: AppAccent.preferenceKey)
        check(make(preview: true).accent == .violet, "An Aurora style without a saved accent must open on Iris")
        preferences.set(AppStyle.graphite.rawValue, forKey: AppStyle.preferenceKey)
        check(make(preview: true).accent == .neutral, "Graphite without a saved accent must open on the neutral accent")
        for invalid in ["neon", "", "Graphite"] {
            preferences.set(invalid, forKey: AppStyle.preferenceKey)
            check(AppStyle.load(from: preferences) == .graphite, "Unknown style accepted: " + invalid)
        }
        preferences.removeObject(forKey: AppStyle.preferenceKey)
        for theme in AppTheme.allCases {
            studio.setTheme(theme)
            check(make(preview: true).theme == theme, "Theme not persisted")
            check(studio.draft == "Unsent text" && studio.selectedConversationID == conversation,
                  "Changing appearance changed conversation or draft")
            check(!theme.title.isEmpty, "Missing theme title")
        }
        for accent in AppAccent.allCases {
            studio.setAccent(accent)
            check(make(preview: true).accent == accent, "Accent not persisted")
        }
        for value in ["", "12345", "#FF0000", "GG0000", "１２３４５６"] {
            check(CustomAccent.normalized(value) == CustomAccent.defaultHex, "Invalid custom color accepted")
        }
        check(CustomAccent.normalized("ab12ef") == "AB12EF", "Custom color normalization failed")
        for value in 0...255 {
            let hex = String(format: "%02X%02X%02X", value, 255 - value, value ^ 0x55)
            check(CustomAccent.hex(CustomAccent.color(hex)) == hex, "Custom color round trip failed")
        }
        studio.setCustomAccent(Color(red: 1, green: 0.5, blue: 0))
        check(studio.accent == .custom && studio.customAccentHex == "FF8000", "Custom color selection failed")
        studio.setStyle(.aurora)
        check(studio.accent == .custom, "Choosing a style replaced a custom colour")
        studio.setStyle(.graphite)
        check(make(preview: true).customAccentHex == "FF8000" && make(preview: true).accent == .custom,
              "Custom color did not persist")
        for density in ConversationDensity.allCases {
            studio.setConversationDensity(density)
            check(make(preview: true).conversationDensity == density, "Density not persisted")
        }
        for width in SidebarWidth.allCases {
            studio.setSidebarWidth(width)
            check(make(preview: true).sidebarWidth == width, "Sidebar width not persisted")
        }
        for font in ConversationFont.allCases {
            studio.setConversationFont(font)
            check(make(preview: true).conversationFont == font, "Conversation font not persisted")
        }
        for style in MessageBubbleStyle.allCases {
            studio.setMessageBubbleStyle(style)
            check(make(preview: true).messageBubbleStyle == style, "Message bubble style not persisted")
        }
        preferences.set("unknown", forKey: SidebarWidth.preferenceKey)
        preferences.set(true, forKey: ConversationFont.preferenceKey)
        preferences.set(17, forKey: MessageBubbleStyle.preferenceKey)
        let invalidAppearance = make(preview: true)
        check(invalidAppearance.sidebarWidth == .standard && invalidAppearance.conversationFont == .standard
              && invalidAppearance.messageBubbleStyle == .subtle, "Invalid appearance values must use defaults")
        studio.openUpdateSettings()
        check(studio.showAppSettings && studio.appSettingsPage == .updates, "Update shortcut did not open updates")
        studio.openAppSettings()
        check(studio.showAppSettings && studio.appSettingsPage == .general, "Ordinary settings must open general")
        studio.appSettingsPage = .appearance
        studio.openUpdateSettings()
        check(studio.appSettingsPage == .updates, "Update shortcut failed while settings were open")
        studio.showAppSettings = false
        let studioViewSource = try String(contentsOfFile: "apps/MLXL3Studio/Sources/MLXL3Studio/StudioView.swift", encoding: .utf8)
        check(studioViewSource.contains("SidebarUpdateBadge(updater: studio.updateManager, action: studio.openUpdateSettings)"),
              "Sidebar update button is not wired to updates")
        preferences.set("unknown", forKey: AppAccent.preferenceKey)
        preferences.set(true, forKey: ConversationDensity.preferenceKey)
        check(make(preview: true).accent == .violet && make(preview: true).conversationDensity == .comfortable,
              "Malformed appearance preference accepted")
        studio.setLoadModelOnStartup(false)
        studio.setStartupModelPath("/fixture/startup")
        studio.resetAppearance()
        let reset = make(preview: true)
        check(reset.theme == .dark && reset.style == .graphite && reset.accent == .neutral && reset.conversationDensity == .comfortable
              && reset.readingWidth == .centered && reset.responseTextSize == 15
              && reset.sidebarWidth == .standard && reset.conversationFont == .standard && reset.messageBubbleStyle == .subtle
              && reset.customAccentHex == CustomAccent.defaultHex,
              "Appearance reset did not persist defaults")
        check(!reset.loadModelOnStartup && reset.startupModelPath == "/fixture/startup"
              && studio.draft == "Unsent text" && studio.selectedConversationID == conversation,
              "Appearance reset changed startup or conversation data")
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
        // WCAG contrast of the resolved tokens on the canvas, in every style and scheme.
        func linear(_ v: CGFloat) -> CGFloat { v <= 0.04045 ? v / 12.92 : pow((v + 0.055) / 1.055, 2.4) }
        func composite(_ color: Color, over base: Color) -> (CGFloat, CGFloat, CGFloat) {
            let c = NSColor(color).usingColorSpace(.sRGB)!, b = NSColor(base).usingColorSpace(.sRGB)!
            let a = c.alphaComponent
            return (c.redComponent * a + b.redComponent * (1 - a), c.greenComponent * a + b.greenComponent * (1 - a),
                    c.blueComponent * a + b.blueComponent * (1 - a))
        }
        func ratio(_ text: Color, on base: Color) -> CGFloat {
            func lum(_ rgb: (CGFloat, CGFloat, CGFloat)) -> CGFloat {
                0.2126 * linear(rgb.0) + 0.7152 * linear(rgb.1) + 0.0722 * linear(rgb.2)
            }
            let t = lum(composite(text, over: base)), b = lum(composite(base, over: base))
            return (max(t, b) + 0.05) / (min(t, b) + 0.05)
        }
        for style in AppStyle.allCases {
            StudioTheme.style = style
            for dark in [false, true] {
                NSAppearance(named: dark ? .darkAqua : .aqua)!.performAsCurrentDrawingAppearance {
                    let ink = ratio(StudioTheme.ink, on: StudioTheme.canvas)
                    check(ink >= 12, "Body text below 12:1 in \(style) dark=\(dark): \(ink)")
                    check(ratio(StudioTheme.secondary, on: StudioTheme.canvas) >= 6, "Secondary text below 6:1 in \(style)")
                    check(ratio(StudioTheme.quiet, on: StudioTheme.canvas) >= 4.5, "Quiet text below 4.5:1 in \(style) dark=\(dark)")
                    check(ratio(StudioTheme.quiet, on: StudioTheme.sidebar) >= 4.5, "Sidebar metadata below 4.5:1 in \(style)")
                    if style.isGraphite {
                        // No pure white text and a sidebar darker than the canvas in dark mode.
                        check(!dark || ink <= 15, "Graphite dark text glares: \(ink)")
                        let canvas = composite(StudioTheme.canvas, over: StudioTheme.canvas)
                        let sidebar = composite(StudioTheme.sidebar, over: StudioTheme.sidebar)
                        check(sidebar.0 + sidebar.1 + sidebar.2 < canvas.0 + canvas.1 + canvas.2, "Graphite sidebar must sit below the canvas")
                    }
                }
            }
        }
        StudioTheme.style = .graphite
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
        studio.appSettingsPage = .appearance
        let host = NSHostingView(rootView: AppSettingsView().environmentObject(studio).environment(\.scenePhase, .inactive))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1000, height: 720),
                              styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        host.frame = NSRect(x: 0, y: 0, width: 1000, height: 720)
        defer { window.close() }
        // Aurora ASCII must carry the aurora across the settings header; Graphite (the default)
        // keeps a plain canvas there.
        for style in [AppStyle.auroraASCII, .graphite] {
          studio.setStyle(style)
          for theme in [AppTheme.light, .dark, .light] {
            studio.setTheme(theme)
            try await Task.sleep(for: .milliseconds(150))
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                preconditionFailure("Settings did not render")
            }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            check(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0, "Empty settings bitmap")
            let corner = bitmap.colorAt(x: 10, y: bitmap.pixelsHigh / 2)!.usingColorSpace(.sRGB)!
            check(theme == .light ? corner.redComponent > 0.7 : corner.redComponent < 0.3,
                  "Rendered settings did not follow selected theme")
            let cool = bitmap.colorAt(x: bitmap.pixelsWide * 77 / 100, y: bitmap.pixelsHigh * 2 / 100)!.usingColorSpace(.sRGB)!
            let warm = bitmap.colorAt(x: bitmap.pixelsWide * 78 / 100, y: bitmap.pixelsHigh * 24 / 100)!.usingColorSpace(.sRGB)!
            if style.isGraphite {
                check(abs(cool.blueComponent - cool.redComponent) < 0.03 && abs(warm.redComponent - warm.blueComponent) < 0.03,
                      "Graphite settings must not show a coloured aurora")
            } else {
                check(cool.blueComponent > cool.redComponent + 0.04 && warm.redComponent > warm.blueComponent + 0.06,
                      "Settings must carry the workspace aurora across its header")
            }
            if let output = ProcessInfo.processInfo.environment["MLXL3_PREFERENCE_SNAPSHOTS"],
               let data = bitmap.representation(using: .png, properties: [:]) {
                try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("settings-\(style.rawValue)-" + theme.rawValue + ".png"))
            }
          }
        }
        let darkHost = NSHostingView(rootView: StudioView().environmentObject(studio)
            .studioAppearance(.dark).frame(width: 1280, height: 800))
        let darkWindow = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 800),
                                  styleMask: .borderless, backing: .buffered, defer: false)
        darkWindow.isReleasedWhenClosed = false
        darkWindow.contentView = darkHost
        darkHost.frame = NSRect(x: 0, y: 0, width: 1280, height: 800)
        defer { darkWindow.close() }
        try await Task.sleep(for: .milliseconds(150))
        darkHost.layoutSubtreeIfNeeded()
        let darkBitmap = darkHost.bitmapImageRepForCachingDisplay(in: darkHost.bounds)!
        darkHost.cacheDisplay(in: darkHost.bounds, to: darkBitmap)
        check(darkBitmap.pixelsWide >= 1280, "Main dark theme window did not render")
        if let output = ProcessInfo.processInfo.environment["MLXL3_PREFERENCE_SNAPSHOTS"],
           let data = darkBitmap.representation(using: .png, properties: [:]) {
            try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("studio-dark.png"))
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
            let corner = bitmap.colorAt(x: 10, y: bitmap.pixelsHigh / 2)!.usingColorSpace(.sRGB)!
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
        for mode in [StudioMode.agent, .chat, .agent, .chat] {
            off.selectMode(mode)
            try await Task.sleep(for: .milliseconds(100))
            check(off.selectedMode == mode && !off.fixtureRunning && off.engineState == .idle,
                  "Switching Chat/Agent must not load an unloaded model")
            if mode == .agent {
                let editor = AgentStudioModel()
                await editor.open(workspace: off.agentWorkspaceURL!, file: nil)
                check(editor.error == nil, "Agent workspace must open before loading the model")
            }
        }
        off.selectModel("second")
        try await wait("manual load when off", until: { off.engineState.isReady })
        check(off.fixtureRunning, "Manual model loading disabled by startup option")
        off.selectMode(.agent)
        try await wait("loaded Chat to Agent", until: { off.engineState.isReady })
        check(off.fixtureRunning && off.loadedModel?.name == "second", "Loaded model lost on Agent navigation: state=\(off.engineState), loaded=\(off.loadedModel?.name ?? "nil"), selected=\(off.selectedModelName ?? "nil"), running=\(off.fixtureRunning)")
        off.selectMode(.chat)
        try await wait("loaded Agent to Chat", until: { off.engineState.isReady })
        off.ejectModel()
        for mode in [StudioMode.agent, .chat] {
            off.selectMode(mode)
            try await Task.sleep(for: .milliseconds(100))
            check(!off.fixtureRunning && off.engineState == .idle, "Navigation reversed a manual unload")
        }
        off.selectModel("second")
        try await wait("explicit reload after navigation", until: { off.engineState.isReady })
        off.setLoadModelOnStartup(true)
        check(off.engineState.isReady, "Toggling preference must not interrupt loaded model")
        off.persistNow(); off.ejectModel()
        for mode in [StudioMode.agent, .chat] {
            off.selectMode(mode)
            try await Task.sleep(for: .milliseconds(100))
            check(!off.fixtureRunning && off.engineState == .idle,
                  "Startup opt-in must not reverse a manual unload during navigation")
        }
        off.engineState = .failed("fixture")
        for mode in [StudioMode.agent, .chat] {
            off.selectMode(mode)
            try await Task.sleep(for: .milliseconds(100))
            check(!off.fixtureRunning && off.engineState == .failed("fixture"),
                  "Navigation must not silently retry a failed model")
        }
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
        let previewConversation = Conversation(title: "Preview", messages: [
            ChatMessage(role: .user, content: "A custom message bubble"),
            ChatMessage(role: .assistant, content: "A readable response with **emphasis**.")
        ])
        studio.conversations = [previewConversation]
        studio.selectedConversationID = previewConversation.id
        studio.setTheme(.dark)
        studio.setCustomAccent(Color(red: 0.18, green: 0.65, blue: 0.80))
        studio.setConversationFont(.serif)
        studio.setMessageBubbleStyle(.tinted)
        studio.setSidebarWidth(.wide)
        studio.setReadingWidth(.narrow)
        try await Task.sleep(for: .milliseconds(500))
        darkHost.layoutSubtreeIfNeeded()
        let customBitmap = darkHost.bitmapImageRepForCachingDisplay(in: darkHost.bounds)!
        darkHost.cacheDisplay(in: darkHost.bounds, to: customBitmap)
        check(customBitmap.pixelsWide == darkBitmap.pixelsWide && customBitmap.pixelsHigh == darkBitmap.pixelsHigh,
              "Custom appearance changed the window dimensions")
        if let output = ProcessInfo.processInfo.environment["MLXL3_PREFERENCE_SNAPSHOTS"],
           let data = customBitmap.representation(using: .png, properties: [:]) {
            try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("studio-custom.png"))
        }
        studio.openUpdateSettings()
        try await Task.sleep(for: .milliseconds(500))
        host.layoutSubtreeIfNeeded()
        let updatesBitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
        host.cacheDisplay(in: host.bounds, to: updatesBitmap)
        check(studio.appSettingsPage == .updates && updatesBitmap.pixelsWide > 0,
              "Updates page failed to render after opening the shortcut")
        if let output = ProcessInfo.processInfo.environment["MLXL3_PREFERENCE_SNAPSHOTS"],
           let data = updatesBitmap.representation(using: .png, properties: [:]) {
            try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("settings-updates.png"))
        }
        // Reasoning selector: levels read from the model's own template, the default never sent,
        // a choice kept per model, nothing shown or sent without engine support.
        check(ReasoningSupport.detect(template: "{%- if enable_thinking is undefined or enable_thinking is true %}{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}")
              == ReasoningSupport(levels: [.none, .low, .medium, .xhigh], defaultLevel: .xhigh), "Qwen3.8 reasoning levels")
        check(ReasoningSupport.detect(template: "{{ enable_thinking }}") == nil, "On/off templates get no reasoning selector")
        let reasoningFolder = root.appendingPathComponent("reasoning-model")
        try FileManager.default.createDirectory(at: reasoningFolder, withIntermediateDirectories: true)
        try #"{%- set effort_map = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7, "high": 0.9, "xhigh": 0.99, "max": 0.99} -%}{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}"#
            .write(to: reasoningFolder.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        let reasoningModel = LocalModel(name: "Inkling 2 Mini", path: reasoningFolder.path, modelType: "inkling2",
            format: "EXL3", bits: 3.33, sizeBytes: 12_000_000_000, modules: 0, addedAt: "", size: "12 GB")
        func reasoningStudio() -> StudioModel {
            let value = make(preview: true)
            value.models = [reasoningModel]
            value.selectModel(reasoningModel.name)
            return value
        }
        let thinker = reasoningStudio()
        check(thinker.reasoningSupport == ReasoningSupport(levels: ReasoningEffort.allCases, defaultLevel: .high, nativeAliases: [.xhigh: .max]),
              "Inkling reasoning levels read from the model folder")
        check(!thinker.reasoningControlAvailable && thinker.requestReasoningEffort == nil,
              "No selector and no field for an engine without reasoning support")
        thinker.fixtureReasoningEngine(true)
        check(thinker.reasoningControlAvailable && thinker.reasoningLevel == .high && thinker.requestReasoningEffort == nil,
              "The template default is shown and never sent")
        thinker.setReasoningLevel(.low)
        check(thinker.reasoningLevel == .low && thinker.requestReasoningEffort == "low", "A chosen level is sent")
        let reopened = reasoningStudio()
        reopened.fixtureReasoningEngine(true)
        check(reopened.reasoningLevel == .low, "The reasoning level is kept per model")
        check(reopened.reasoningLevels == [.none, .low, .medium, .high, .max],
              "A legacy engine offers only the historical, distinct native levels")
        let native: [ReasoningEffort] = [.none, .minimal, .low, .medium, .high, .max]
        reopened.fixtureReasoningEngine(true, levels: Set(ReasoningEffort.allCases))
        check(reopened.reasoningLevels == native, "Inkling supports Minimal and one representative for 0.99")
        for level in native {
            reopened.setReasoningLevel(level)
            check(reopened.requestReasoningEffort == level.rawValue, "The native choice is sent, including the explicit default")
        }
        let key = "studio.reasoning." + reasoningModel.path
        preferences.set("xhigh", forKey: key)
        let aliasStudio = reasoningStudio()
        aliasStudio.fixtureReasoningEngine(true)
        check(aliasStudio.effectiveReasoningChoice == .max && aliasStudio.requestReasoningEffort == "max",
              "Existing Inkling XHigh preferences map to the equivalent native Max")
        reopened.setReasoningLevel(.minimal)
        reopened.fixtureReasoningEngine(true)
        check(reopened.requestReasoningEffort == nil, "Older engines use the model default for unsupported Minimal")
        reopened.fixtureReasoningEngine(true, levels: Set(ReasoningEffort.allCases))
        check(reopened.requestReasoningEffort == "minimal", "A compatible engine restores the saved Minimal choice")
        reopened.responseSchema = "{}"
        check(reopened.requestReasoningEffort == "minimal", "Structured output uses the same native effort")
        reopened.responseSchema = ""
        reopened.resetReasoningLevel()
        check(reopened.requestReasoningEffort == nil && reasoningStudio().reasoningChoice == nil,
              "Model default clears the stored override and omits the field")
        let toggleTemplate = "{% if enable_thinking is defined and enable_thinking is false %}<think></think>{% else %}<think>{% endif %}"
        let toggle = ReasoningSupport.detect(template: toggleTemplate)
        check(toggle?.choices(acceptedByEngine: Set(ReasoningEffort.allCases)) == [.none, .max],
              "A native switch has only Off and On")
        let alias = "if resolved_reasoning_effort == 'high', set to 'xhigh'; "
            + "if resolved_reasoning_effort not in ('xhigh', 'medium', 'low'); enable_thinking"
        check(ReasoningSupport.detect(template: alias)?.levels == [.none, .low, .medium, .xhigh],
              "Qwen3.8's high alias does not create a duplicate native level")
        check(ReasoningSupport.detect(template: "{{ reasoning_effort }} Tool priorities: 'high', 'low'.") == nil,
              "Unrelated strings do not invent reasoning levels")
        try toggleTemplate.write(to: reasoningFolder.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
        for stale in ["low", "medium", "high", "unrecognized"] {
            preferences.set(stale, forKey: key)
            let restored = reasoningStudio()
            restored.fixtureReasoningEngine(true, levels: Set(ReasoningEffort.allCases))
            check(restored.requestReasoningEffort == nil && restored.effectiveReasoningChoice == nil
                    && restored.reasoningLevels == [.none, .max],
                  "Old unsupported levels go back to the model default")
        }
        let composer = NSHostingView(rootView: ComposerView().environmentObject(reopened)
            .frame(width: 720).padding(24).background(StudioTheme.canvas).preferredColorScheme(.dark))
        composer.frame = NSRect(x: 0, y: 0, width: 768, height: 150)
        composer.layoutSubtreeIfNeeded()
        let composerBitmap = composer.bitmapImageRepForCachingDisplay(in: composer.bounds)!
        composer.cacheDisplay(in: composer.bounds, to: composerBitmap)
        check(composerBitmap.pixelsWide > 0, "Composer with the reasoning selector failed to render")
        if let output = ProcessInfo.processInfo.environment["MLXL3_PREFERENCE_SNAPSHOTS"],
           let data = composerBitmap.representation(using: .png, properties: [:]) {
            try data.write(to: URL(fileURLWithPath: output).appendingPathComponent("composer-reasoning.png"))
        }
        print("Desktop preference checks passed: \(checks) assertions, isolated persistence, appearance, actual startup/manual loading and errors")
    }
}
