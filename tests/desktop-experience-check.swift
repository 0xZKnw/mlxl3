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
            visual.setTheme(.midnightBlue)
            visual.models = [flash]; visual.selectedModelName = flash.name
            for width in ConversationReadingWidth.allCases {
                visual.setReadingWidth(width)
                let answer = ChatMessage(role: .assistant, content: "# Une réponse lisible\n\n" + String(repeating: "Le même texte conserve sa taille ; seule la largeur de lecture change. ", count: 8) + "\n\n```swift\nlet resultat = donnees.filter { $0.score > 10 }.sorted { $0.nom < $1.nom }\n```")
                visual.conversations[0].messages = [answer]
                for size in [820, 1280] {
                    let host = NSHostingView(rootView: StudioView().environmentObject(visual).studioAppearance(.midnightBlue)
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
                .frame(width: 360, height: 540).background(StudioTheme.canvas(for: .midnightBlue)).studioAppearance(.midnightBlue))
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
