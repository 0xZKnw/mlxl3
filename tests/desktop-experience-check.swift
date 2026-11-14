import AppKit
import Combine
import Foundation
import SwiftUI

@main struct DesktopExperienceCheck {
    @MainActor static func main() async throws {
        var checks = 0
        func check(_ condition: Bool, _ label: String) throws {
            checks += 1
            if !condition { throw NSError(domain: "DesktopExperienceCheck", code: 1,
                                          userInfo: [NSLocalizedDescriptionKey: label]) }
        }
        func wait(_ label: String, until condition: () -> Bool) async throws {
            for _ in 0..<300 {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw NSError(domain: "DesktopExperienceCheck", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Deadline: " + label])
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-experience-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let suite = "io.mlxl3.experience." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        var instances: [StudioModel] = []
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        defer {
            for studio in instances { studio.ejectModel() }
            prefs.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
            unsetenv("MLXL3_FIXTURE_MODELS"); unsetenv("MLXL3_FIXTURE_FLASH_PATH")
        }
        func make(preview: Bool = false, file: URL? = nil) -> StudioModel {
            let studio = StudioModel(conversationFileURL: file ?? root.appendingPathComponent(UUID().uuidString + ".json"),
                updateManager: UpdateManager(releasesURL: root.appendingPathComponent("no-network.json")),
                isPreview: preview, preferences: prefs)
            instances.append(studio)
            return studio
        }
        let file = root.appendingPathComponent("pins.json")
        let visual = make(preview: true, file: file)
        checkAuroraPlayback()
        try await checkSettingsSurfaces()
        try await checkManagerSurfaces(studio: visual)
        try await checkDisabledControls()
        try await checkWidgetSizing(studio: visual)
        try await checkAurora()
        try await checkWorkspaceControls(studio: visual, preferences: prefs)
        let a = Conversation(title: "Récent", messages: [ChatMessage(role: .assistant, content: "A")], createdAt: Date(timeIntervalSince1970: 300))
        let b = Conversation(title: "Travail", messages: [ChatMessage(role: .assistant, content: "B")], createdAt: Date(timeIntervalSince1970: 200))
        let c = Conversation(title: "Notes", messages: [ChatMessage(role: .assistant, content: "C")], createdAt: Date(timeIntervalSince1970: 100))
        visual.conversations = [a, b, c]
        visual.selectedConversationID = a.id; visual.draft = "Brouillon conservé"
        let orderingFile = root.appendingPathComponent("ordering.json")
        let ordering = make(preview: true, file: orderingFile)
        let old = Conversation(title: "Older", createdAt: Date(timeIntervalSince1970: 100))
        var active = old
        active.updatedAt = Date(timeIntervalSince1970: 200)
        ordering.conversations = [active]
        ordering.newConversation()
        let first = ordering.selectedConversationID!
        try check(ordering.sidebarConversations(matching: "").first?.id == first, "New conversation must precede older activity")
        ordering.newConversation()
        let second = ordering.selectedConversationID!
        try check(ordering.sidebarConversations(matching: "").prefix(2).map(\.id) == [second, first], "Repeated new conversations must stay newest first")
        ordering.setConversationPinned(active.id, true)
        try check(ordering.sidebarConversations(matching: "").map(\.id) == [active.id, second, first], "Pinned conversations must remain above new conversations")
        try check(ordering.persistNow(), "New conversation ordering did not persist")
        let reloadedOrdering = make(preview: true, file: orderingFile)
        try check(reloadedOrdering.sidebarConversations(matching: "").map(\.id) == [active.id, second, first], "New conversation ordering changed after reload")
        try check(ConversationReadingWidth.load(from: prefs) == .centered, "Default width changed")
        for width in ConversationReadingWidth.allCases {
            visual.setReadingWidth(width)
            try check(make(preview: true).readingWidth == width, "Reading width did not persist")
            try check(visual.draft == "Brouillon conservé" && visual.selectedConversationID == a.id, "Width changed draft/selection")
            try check(visual.conversations.map { $0.messages[0].content } == ["A", "B", "C"], "Width changed text")
        }
        for invalid in ["unknown", 42, true] as [Any] {
            prefs.set(invalid, forKey: ConversationReadingWidth.preferenceKey)
            try check(ConversationReadingWidth.load(from: prefs) == .centered, "Invalid width did not fall back")
        }
        visual.setConversationPinned(c.id, true)
        visual.setConversationPinned(b.id, true)
        try check(visual.sidebarConversations(matching: "").map(\.id) == [b.id, c.id, a.id], "Pinned ordering is unstable")
        try check(visual.sidebarConversations(matching: " Notes ").map(\.id) == [c.id], "Search lost pinned conversations")
        try check(visual.sidebarConversations(matching: "absent").isEmpty, "Search returned unrelated pins")
        var sidebarNotifications = 0
        let sidebarObservation = visual.objectWillChange.sink { sidebarNotifications += 1 }
        a.messages[0].append(" absent", phase: nil)
        try check(sidebarNotifications == 1, "Excluded streaming row did not publish sidebar invalidation")
        a.messages[0].append(" again", phase: nil)
        try check(sidebarNotifications == 1, "Pending sidebar invalidation did not coalesce")
        try check(visual.sidebarConversations(matching: "absent").map(\.id) == [a.id],
                  "Streaming mutation did not invalidate the cached search")
        _ = visual.sidebarConversations(matching: "")
        a.messages[0].append(" no search", phase: nil)
        try check(sidebarNotifications == 1, "Empty search caused unnecessary sidebar publication")
        sidebarObservation.cancel()
        visual.conversations[0].messages = [ChatMessage(role: .assistant, content: "A")]
        try check(visual.sidebarConversations(matching: "absent").isEmpty,
                  "Replacing message membership did not invalidate search")
        visual.conversations[0].messages[0].append(" absent", phase: nil)
        try check(visual.sidebarConversations(matching: "absent").map(\.id) == [a.id],
                  "Replacement message was not observed")
        visual.conversations[0].messages = [ChatMessage(role: .assistant, content: "A")]
        try check(visual.draft == "Brouillon conservé" && visual.selectedConversationID == a.id, "Pinning selected another chat")
        visual.setConversationPinned(UUID(), true)
        visual.persistNow()
        let restored = make(preview: true, file: file)
        try check(restored.draft == "Brouillon conservé", "Unsent draft did not survive history reload")
        try check(restored.sidebarConversations(matching: "").map(\.id) == [b.id, c.id, a.id], "Pins did not survive history reload")
        restored.setConversationPinned(b.id, false)
        try check(restored.sidebarConversations(matching: "").map(\.id) == [c.id, a.id, b.id], "Unpin changed ordinary ordering")
        if let recent = restored.conversations.firstIndex(where: { $0.id == b.id }) {
            restored.conversations[recent].updatedAt = Date(timeIntervalSince1970: 1234)
        }
        try check(restored.sidebarConversations(matching: "").map(\.id) == [c.id, b.id, a.id], "Completed activity should reorder ordinary chats but preserve pins")
        restored.deleteConversation(c.id)
        restored.undoDeleteConversation()
        try check(restored.conversations.first(where: { $0.id == c.id })?.isPinned == true, "Undo lost pin state")
        // The real workspace decoder must still read histories without the new field.
        var legacy = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
        var rows = legacy["conversations"] as! [[String: Any]]
        for i in rows.indices { rows[i].removeValue(forKey: "isPinned") }
        legacy["conversations"] = rows
        let legacyFile = root.appendingPathComponent("legacy.json")
        try JSONSerialization.data(withJSONObject: legacy).write(to: legacyFile)
        let legacySnapshot = try ConversationStore.load(from: legacyFile)!
        try check(legacySnapshot.conversations.map(Conversation.init(snapshot:)).allSatisfy { !$0.isPinned }, "Legacy history did not default to unpinned")
        try check(legacySnapshot.conversations.count == 3, "Legacy migration lost chats")
        func model(_ name: String, type: String = "qwen3_5") -> LocalModel {
            LocalModel(name: name, path: root.appendingPathComponent(name).path, modelType: type,
                       format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
        }
        let dense = model("auto-dense"), moe = model("auto-moe"), flash = model("flash-native", type: "qwen4_exp")
        // Malformed values must not turn acceleration on, including numeric 1.
        for invalid in ["true", 1, [true]] as [Any] {
            prefs.set([dense.path: invalid], forKey: ModelMTPPreference.key)
            try check(ModelMTPPreference.load(modelPath: dense.path, preferences: prefs) == nil,
                      "Malformed MTP preference enabled a model")
        }
        prefs.removeObject(forKey: ModelMTPPreference.key)
        // Exercise migration through the production loader, then a fresh app instance.
        prefs.set(true, forKey: "studio.mtpEnabled")
        let migration = make(); migration.models = [dense, moe]
        migration.selectModel(dense.name)
        try await wait("legacy MTP migration", until: { migration.engineState.isReady && !migration.mtpDownloading })
        try check(migration.mtpActive == true, "Legacy ON was not migrated")
        migration.ejectModel()
        let migratedNew = make(); migratedNew.models = [dense, moe]; migratedNew.selectModel(moe.name)
        try await wait("new model after migration", until: { migratedNew.engineState.isReady && !migratedNew.mtpDownloading })
        try check(!migratedNew.mtpRememberedEnabled && !migratedNew.mtpEnabled,
                  "Legacy ON leaked to another model after app restart")
        migratedNew.ejectModel()
        prefs.removeObject(forKey: ModelMTPPreference.key)
        prefs.set(false, forKey: "studio.mtpEnabled")
        let studio = make(); studio.models = [dense, moe, flash, model("auto-fail"), model("plain")]
        func ready(_ label: String) async throws {
            try await wait(label, until: { studio.engineState.isReady && !studio.mtpDownloading })
        }
        studio.selectModel(dense.name); try await ready("initial dense")
        let sentTo = studio.selectedConversationID
        for text in ["First send", "Follow-up"] {
            studio.draft = text
            let old = studio.lastSubmittedMessageID
            studio.send()
            let user = studio.currentConversation?.messages.last(where: { $0.role == .user })
            try check(studio.lastSubmittedMessageID != old && studio.lastSubmittedMessageID == user?.id
                      && user?.content == text && studio.selectedConversationID == sentTo && studio.draft.isEmpty,
                      "Send animation must identify the accepted user message")
            let sent = studio.lastSubmittedMessageID
            studio.send()
            try check(studio.lastSubmittedMessageID == sent, "Rejected send replayed the animation")
            try await ready("send completion")
        }
        try check(!studio.mtpEnabled && !studio.mtpRememberedEnabled, "Fresh model did not default OFF")
        studio.setMTPEnabled(true); try await ready("enable dense")
        try check(studio.mtpEnabled && studio.mtpActive == true && studio.mtpRememberedEnabled, "MTP did not activate")
        studio.ejectModel(); studio.selectModel(dense.name); try await ready("reload dense")
        try check(studio.mtpEnabled && studio.mtpActive == true, "Reload lost MTP ON")
        studio.selectModel(moe.name); try await ready("new moe")
        try check(!studio.mtpEnabled && !studio.mtpRememberedEnabled, "Dense ON leaked to a new model")
        studio.setMTPEnabled(true); try await ready("enable moe")
        studio.selectModel(dense.name); try await ready("return dense")
        studio.setMTPEnabled(false)
        try await wait("dense off ack", until: { studio.mtpActive == false })
        studio.selectModel(moe.name); try await ready("return moe")
        try check(studio.mtpRememberedEnabled && studio.mtpActive == true, "Dense OFF erased MoE ON")
        studio.ejectModel()
        // Actual app startup, not only selectModel: remembered ON must be prepared automatically.
        setenv("MLXL3_FIXTURE_MODELS", String(decoding: try JSONEncoder().encode([dense, moe]), as: UTF8.self), 1)
        let startup = make(); startup.setStartupModelPath(moe.path); startup.start()
        try await wait("startup with MTP", until: { startup.engineState.isReady && !startup.mtpDownloading })
        try check(startup.selectedModelName == moe.name && startup.mtpActive == true, "App startup did not restore MTP")
        startup.ejectModel()
        studio.selectModel("auto-fail"); try await ready("failure target")
        studio.setMTPEnabled(true); try await ready("failed activation")
        try check(studio.mtpError != nil && !studio.mtpEnabled && studio.mtpRememberedEnabled, "Transient failure erased the user's intent")
        studio.setMTPEnabled(false)
        studio.ejectModel(); studio.selectModel("auto-fail"); try await ready("failed target opt out")
        try check(!studio.mtpRememberedEnabled && studio.mtpError == nil, "Explicit opt-out retried failed preparation")
        setenv("MLXL3_FIXTURE_FLASH_PATH", flash.path, 1)
        prefs.set("/foreign/mtp/head", forKey: "studio.mtpHead." + flash.path)
        studio.selectModel(flash.name); try await ready("flash off")
        try check(!studio.mtpRememberedEnabled, "New Flash target did not default OFF")
        studio.setMTPEnabled(true); try await ready("flash integrated MTP")
        try check(studio.mtpUsesModelCheckpoint && studio.mtpHeadPath == flash.path && studio.mtpActive == true,
                  "Flash Next incorrectly used the external head installer")
        studio.ejectModel(); studio.selectModel(flash.name); try await ready("flash reload")
        try check(studio.mtpActive == true && studio.mtpHeadPath == flash.path, "Flash MTP did not survive reload")
        studio.ejectModel()
        // Images are evidence, never an oracle copied from the view implementation.
        if let output = ProcessInfo.processInfo.environment["MLXL3_EXPERIENCE_SNAPSHOTS"] {
            _ = NSApplication.shared
            visual.setTheme(.dark)
            visual.models = [flash]; visual.selectedModelName = flash.name
            for width in ConversationReadingWidth.allCases {
                visual.setReadingWidth(width)
                let answer = ChatMessage(role: .assistant, content: "# Une réponse lisible\n\n" + String(repeating: "Le même texte conserve sa taille ; seule la largeur de lecture change. ", count: 8) + "\n\n```swift\nlet resultat = donnees.filter { $0.score > 10 }.sorted { $0.nom < $1.nom }\n```")
                visual.conversations[0].messages = [answer]
                for size in [820, 1280] {
                    let host = NSHostingView(rootView: StudioView().environmentObject(visual).studioAppearance(.dark)
                        .frame(width: CGFloat(size), height: 800))
                    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: size, height: 800), styleMask: .borderless, backing: .buffered, defer: false)
                    window.isReleasedWhenClosed = false; window.contentView = host
                    host.frame = NSRect(x: 0, y: 0, width: size, height: 800)
                    try await Task.sleep(for: .milliseconds(250)); host.layoutSubtreeIfNeeded()
                    let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                    host.cacheDisplay(in: host.bounds, to: bitmap)
                    try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output).appendingPathComponent("\(width.rawValue)-\(size).png"))
                    window.close()
                }
            }
            studio.selectModel(dense.name); try await ready("MTP settings render")
            studio.setMTPEnabled(true); try await ready("MTP settings enabled")
            let host = NSHostingView(rootView: MTPSettingsContent().environmentObject(studio).padding(20)
                .frame(width: 360, height: 540).background(StudioTheme.canvas).studioAppearance(.dark))
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 360, height: 540), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false; window.contentView = host; host.frame = NSRect(x: 0, y: 0, width: 360, height: 540)
            try await Task.sleep(for: .milliseconds(150)); host.layoutSubtreeIfNeeded()
            let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
            host.cacheDisplay(in: host.bounds, to: bitmap)
            try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output).appendingPathComponent("mtp-panel.png"))
            window.close()
        }
        print("Desktop experience checks passed: \(checks) assertions; pin/history migration, width/draft, per-model MTP OFF/ON/reload/startup/failure/Flash integrated head")
    }
}

extension DesktopExperienceCheck {
    // AppKit bitmap caching omits compositor-backed siblings when native Liquid Glass is present.
    @MainActor static func checkAurora() async throws {
        _ = NSApplication.shared
        for (theme, width, height) in [(AppTheme.light, 820, 600), (.dark, 1280, 820)] {
            let bounds = NSRect(x: 0, y: 0, width: width, height: height)
            let window = NSWindow(contentRect: bounds, styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            defer { window.close() }
            window.appearance = NSAppearance(named: theme == .light ? .aqua : .darkAqua)
            func field(_ busy: Bool) -> some View {
                StudioAurora(busy: busy, fullHeight: true).background(StudioTheme.canvas)
                    .studioAppearance(theme).transaction { $0.disablesAnimations = true }
            }
            let host = NSHostingView(rootView: field(false))
            window.contentView = host
            host.frame = bounds

            func snapshot() async throws -> NSBitmapImageRep {
                try await Task.sleep(for: .milliseconds(150))
                host.layoutSubtreeIfNeeded()
                let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                host.cacheDisplay(in: host.bounds, to: bitmap)
                precondition(bitmap.pixelsWide >= width && bitmap.pixelsHigh >= height)
                return bitmap
            }
            func color(_ image: NSBitmapImageRep, x: Double, y: Double) -> NSColor {
                image.colorAt(x: Int(x * Double(image.pixelsWide)), y: Int(y * Double(image.pixelsHigh)))!
                    .usingColorSpace(.sRGB)!
            }
            let idle = try await snapshot()
            let upper = color(idle, x: 0.342, y: 0.045)
            precondition(max(upper.redComponent, upper.blueComponent) - upper.greenComponent > 0.027, "Missing high ASCII reflection")
            let cool = color(idle, x: 0.77, y: 0.32)
            let warm = color(idle, x: 0.80, y: 0.75)
            let reading = color(idle, x: 0.40, y: 0.24)
            precondition(cool.blueComponent > cool.redComponent + 0.05, "Missing violet column")
            precondition(warm.redComponent > warm.blueComponent + 0.10, "Missing amber foot below the composer")
            precondition(abs(reading.redComponent - reading.blueComponent) < 0.03, "Lost quiet space for the title")
            precondition(theme == .light ? reading.redComponent > 0.85 : reading.redComponent < 0.2, "Lost theme canvas")

            // A smooth glow has no repeated stroke edges. Sample an empty patch below the composer.
            var edges = 0
            for row in 0..<12 {
                let y = 0.755 + Double(row) / Double(height)
                for col in 0..<40 {
                    let x = 0.77 + Double(col) / Double(width)
                    let step = 1 / Double(width)
                    let a = color(idle, x: x - step, y: y).greenComponent
                    let b = color(idle, x: x, y: y).greenComponent
                    let c = color(idle, x: x + step, y: y).greenComponent
                    if abs(b - (a + c) / 2) > 0.010 { edges += 1 }
                }
            }
            precondition(edges > 18, "Missing fine ASCII strokes")

            host.rootView = field(true)
            let busy = try await snapshot()
            let after = color(busy, x: 0.80, y: 0.75)
            precondition(abs(after.greenComponent - warm.greenComponent) < 0.01,
                         "Sending must preserve the still composition and brightness")
        }
        print("Aurora passed 14 background checks, compact light and wide dark, idle and generating")
    }
}

extension DesktopExperienceCheck {
    static func checkAuroraPlayback() {
        var playback = AuroraPlayback(at: 0)
        precondition(playback.phase(at: 4) == 4)
        playback.setGenerating(true, at: 4)
        precondition(playback.phase(at: 4) == 4)
        precondition(abs(playback.phase(at: 8) - playback.phase(at: 7) - 2.0) < 1e-9)
        func speed(_ time: Double) -> Double {
            (playback.phase(at: time + 0.0001) - playback.phase(at: time)) / 0.0001
        }
        precondition(speed(4.5) < 1 && speed(5.5) > 2.0)
        for (time, busy) in [(6.0, false), (6.4, true), (6.6, false)] {
            let position = playback.phase(at: time), rate = speed(time)
            playback.setGenerating(busy, at: time)
            precondition(abs(playback.phase(at: time) - position) < 1e-9 && abs(speed(time) - rate) < 0.002)
            for step in 0..<30 { precondition(speed(time + Double(step) / 100) > 0) }
        }
        precondition(abs(speed(12) - 1) < 1e-7)
        for pulse in 0..<12 {
            let time = 15 + Double(pulse) * 0.49
            playback.setGenerating(true, at: time)
            for step in 0...47 { precondition(speed(time + Double(step) / 100) >= 0.12) }
            playback.setGenerating(false, at: time + 0.48)
        }
        print("Aurora playback passed: undershoot, overshoot, 2x cruise, continuous reversals, return to 1x, twelve rapid restarts")
    }
}


extension DesktopExperienceCheck {
    @MainActor static func checkSettingsSurfaces() async throws {
        func sample(_ backdrop: Color, theme: AppTheme) async throws -> NSColor {
            let card = SettingCard(title: "Fixture", icon: "slider.horizontal.3") {
                Text("Control").frame(height: 60)
            }
            .frame(width: 320, height: 160, alignment: .top)
            .background(backdrop)
            .studioAppearance(theme)
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 320, height: 160),
                                  styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            defer { window.close() }
            window.appearance = NSAppearance(named: theme == .light ? .aqua : .darkAqua)
            let host = NSHostingView(rootView: card)
            window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: 320, height: 160)
            try await Task.sleep(for: .milliseconds(100))
            host.layoutSubtreeIfNeeded()
            let image = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
            host.cacheDisplay(in: host.bounds, to: image)
            return image.colorAt(x: image.pixelsWide * 3 / 4, y: image.pixelsHigh / 2)!
                .usingColorSpace(.sRGB)!
        }
        for theme in [AppTheme.dark, .light] {
            let red = try await sample(.red, theme: theme)
            let blue = try await sample(.blue, theme: theme)
            precondition(red.redComponent > blue.redComponent + 0.3 &&
                         blue.blueComponent > red.blueComponent + 0.3,
                         "Settings sections must preserve the underlying surface, without an opaque card")
        }
        print("Settings surfaces passed: real light/dark rendering preserves the backdrop")
    }
}


extension DesktopExperienceCheck {
    @MainActor static func checkWorkspaceControls(studio: StudioModel, preferences: UserDefaults) async throws {
        var agent = Conversation()
        agent.agent = true
        studio.conversations = [agent]
        studio.selectConversation(agent.id)
        studio.selectMode(.lab)
        studio.newConversation()
        precondition(studio.selectedMode == .chat && studio.conversations.count == 2,
                     "New conversation from Lab must create exactly one Chat when only Agent tasks exist")
        let bounds = NSRect(x: 0, y: 0, width: 1230, height: 780)
        let window = NSWindow(contentRect: bounds, styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        defer { window.close() }
        studio.setLanguage(.fr)
        studio.showInspector = false
        studio.newConversation()
        let host = NSHostingView(rootView: StudioView().environmentObject(studio)
            .defaultAppStorage(preferences).studioAppearance(.dark))
        window.contentView = host
        host.frame = bounds
        window.setFrameOrigin(NSPoint(x: -6000, y: -6000))
        window.orderBack(nil)
        try await Task.sleep(for: .milliseconds(150))
        func click(_ x: CGFloat, _ y: CGFloat) async throws {
            for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
                let event = NSEvent.mouseEvent(with: type, location: NSPoint(x: x, y: bounds.height - y),
                    modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                    windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
                window.sendEvent(event)
            }
            try await Task.sleep(for: .milliseconds(100))
        }
        for x in [CGFloat(30), 200] {
            let count = studio.conversations.count
            try await click(x, 334)
            precondition(studio.conversations.count == count + 1,
                         "New conversation must respond across the full sidebar row")
        }
        studio.draft = "Brouillon conservé"
        try await click(1195, 33)
        precondition(studio.showInspector && studio.draft == "Brouillon conservé")
        try await click(1188, 52)
        precondition(!studio.showInspector && studio.draft == "Brouillon conservé")
        print("Workspace controls passed: full-width new conversation, inspector close and retained draft")
    }
}


extension DesktopExperienceCheck {
    @MainActor static func checkDisabledControls() async throws {
        var values: [CGFloat] = []
        for disabled in [false, true] {
            let bounds = NSRect(x: 0, y: 0, width: 100, height: 60)
            let window = NSWindow(contentRect: bounds, styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            defer { window.close() }
            let host = NSHostingView(rootView: Button {} label: {
                Rectangle().fill(.white).frame(width: 50, height: 20)
            }.buttonStyle(StudioToolbarStyle()).disabled(disabled)
                .padding(20).background(.black).studioAppearance(.dark))
            window.contentView = host
            host.frame = bounds
            try await Task.sleep(for: .milliseconds(150))
            host.layoutSubtreeIfNeeded()
            let bitmap = host.bitmapImageRepForCachingDisplay(in: bounds)!
            host.cacheDisplay(in: bounds, to: bitmap)
            let color = bitmap.colorAt(x: bitmap.pixelsWide / 2, y: bitmap.pixelsHigh / 2)!.usingColorSpace(.sRGB)!
            values.append(color.redComponent)
        }
        precondition(values[0] > 0.9 && values[1] < values[0] - 0.3,
                     "Disabled glass controls must visibly dim")
        print("Disabled controls passed: real enabled/disabled toolbar rendering")
    }

    @MainActor static func checkWidgetSizing(studio: StudioModel) async throws {
        let icon = AppDelegate.menuBarIcon()
        precondition(icon?.isTemplate == true && icon?.accessibilityDescription == "MLXL3 Desktop")
        precondition(icon?.tiffRepresentation != nil && StudioBrand.logo?.size != icon?.size)
        var measured = CGSize.zero
        let bounds = NSRect(x: 0, y: 0, width: 340, height: 600)
        let window = NSWindow(contentRect: bounds, styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        defer { window.close(); studio.engineState = .idle; studio.selectedModelName = nil }
        let host = NSHostingView(rootView: MenuBarPanel(sizeChanged: { measured = $0 })
            .environmentObject(studio).studioAppearance(.dark))
        window.contentView = host
        host.frame = bounds
        studio.engineState = .idle
        try await Task.sleep(for: .milliseconds(150))
        host.layoutSubtreeIfNeeded()
        let unloaded = measured
        precondition(unloaded.width == 340 && unloaded.height > 250 && unloaded.height < 500)
        precondition(abs(unloaded.height - host.fittingSize.height) < 1)
        studio.selectedModelName = String(repeating: "Long model name ", count: 8)
        studio.engineState = .loading(studio.selectedModelName!)
        try await Task.sleep(for: .milliseconds(150))
        host.layoutSubtreeIfNeeded()
        precondition(measured.width == 340 && measured.height > unloaded.height)
        precondition(abs(measured.height - host.fittingSize.height) < 1,
                     "The menu widget must resize its popover as the model changes")
        print("Widget sizing passed: intrinsic height updates for unloaded and long loading model")
    }
}


extension DesktopExperienceCheck {
    @MainActor static func checkManagerSurfaces(studio: StudioModel) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-surfaces-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "io.mlxl3.surfaces." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        studio.showModelManager = true
        studio.showMCPManager = false
        studio.showWorkspace = false
        for theme in [AppTheme.dark, .light] {
            studio.setTheme(theme)
            var agentClosed = false
            for (label, view) in [
                ("models", AnyView(ModelManagerView())),
                ("mcp", AnyView(MCPManagerView())),
                ("projects", AnyView(WorkspaceManagerView())),
                ("agent", AnyView(AgentStudioView(workspace: root.appendingPathComponent("files"), onClose: { agentClosed = true })))
            ] {
                let bounds = NSRect(x: 0, y: 0, width: 820, height: 680)
                let window = NSWindow(contentRect: bounds, styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false
                let host = NSHostingView(rootView: view.environmentObject(studio)
                    .defaultAppStorage(prefs).studioAppearance(theme)
                    .environment(\.scenePhase, .inactive))
                window.contentView = host
                host.frame = bounds
                window.setFrameOrigin(NSPoint(x: -6000, y: -6000))
                window.orderBack(nil)
                studio.showModelManager = label == "models"
                studio.showMCPManager = label == "mcp"
                studio.showWorkspace = label == "projects"
                try await Task.sleep(for: .milliseconds(200))
                host.layoutSubtreeIfNeeded()
                let bitmap = host.bitmapImageRepForCachingDisplay(in: bounds)!
                host.cacheDisplay(in: bounds, to: bitmap)
                var chroma = 0.0
                for y in stride(from: 132, to: 180, by: 3) {
                    let color = bitmap.colorAt(x: 8, y: y)!.usingColorSpace(.sRGB)!
                    chroma += max(color.redComponent, color.blueComponent) - color.greenComponent
                }
                precondition(chroma > 0.07, "\(label) must carry the aurora through its sidebar in \(theme)")
                precondition(host.frame.width == 820, "Management page escaped its compact window")
                for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
                    let event = NSEvent.mouseEvent(with: type, location: NSPoint(x: 58, y: 36),
                        modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                        windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
                    window.sendEvent(event)
                }
                try await Task.sleep(for: .milliseconds(80))
                precondition(label != "models" || !studio.showModelManager, "Models Back failed")
                precondition(label != "mcp" || !studio.showMCPManager, "MCP Back failed")
                precondition(label != "projects" || !studio.showWorkspace, "Projects Back failed")
                precondition(label != "agent" || agentClosed, "Agent Studio Back failed")
                window.close()
            }
        }
        studio.setTheme(.dark)
        studio.showModelManager = false
        print("Management surfaces passed: four production pages, light/dark, compact width, translucent sidebar pixels and actual Back clicks")
    }
}
