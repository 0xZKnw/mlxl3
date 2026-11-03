import AppKit
import Foundation
import SwiftUI

@main struct WorkspaceFeaturesCheck {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-workspace-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        let suite = "mlxl3.workspace.\(UUID())"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        var checks = 0
        func check(_ condition: @autoclosure () throws -> Bool, _ message: String) rethrows { let result = try condition(); precondition(result, message); checks += 1 }
        func wait(_ predicate: @MainActor () -> Bool) async throws {
            for _ in 0..<400 { if predicate() { return }; try await Task.sleep(for: .milliseconds(20)) }
            preconditionFailure("Fixture timed out")
        }
        let history = root.appendingPathComponent("history.json")
        let studio = StudioModel(conversationFileURL: history, isPreview: true, preferences: preferences)
        let recommended = ModelLibrary.recommendedModels
        check(recommended.map(\.id) == ["UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw", "turboderp/Qwen3.8-27B-exl3", "turboderp/Qwen3.8-Flash-Next-exl3", "0xzknw/LFM2.5-24B-A2B-EXL3-4bpw"], "Four requested local model origins")
        check(Set(recommended.map(\.id)).count == 4 && recommended.allSatisfy { $0.revision.count == 40 && $0.revision.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) } }, "Unique recommendations with immutable revisions")
        let catalogue = ModelLibrary()
        for model in recommended {
            catalogue.open(model.id, revision: model.revision)
            try await wait { !catalogue.loadingDetail }
            check(catalogue.detail?.id == model.id && catalogue.error == nil, "Recommendation opens its exact repository")
            check(catalogue.downloading == nil, "Opening a recommendation never starts a download")
            catalogue.back()
        }
        studio.addProject("Astronomie")
        let projectA = studio.selectedProjectID!
        studio.addProject("Cuisine")
        let projectB = studio.selectedProjectID!
        check(projectA != projectB && studio.projects.count == 2, "Distinct projects")
        studio.addProject("  "); studio.addProject(String(repeating: "x", count: 101))
        check(studio.projects.count == 2, "Invalid project names rejected")
        studio.chooseProject(projectA)
        let file = root.appendingPathComponent("planetes.txt")
        try Data("Vénus est à 464 degrés. Mars a deux satellites, Phobos et Deimos.".utf8).write(to: file)
        await studio.importLibraryFiles([file])
        check(studio.documents.count == 1 && studio.documents[0].projectID == projectA, "Project-scoped import")
        await studio.importLibraryFiles([file])
        check(studio.documents.count == 1, "Duplicate import")
        let savedDocuments = studio.documents
        studio.documents = (0..<128).map { LibraryDocument(projectID: projectA, fileName: "\($0).txt", originalPath: "", text: "Fixture") }
        await studio.importLibraryFiles([file])
        check(studio.documents.count == 128 && studio.featureError != nil, "Document count limit")
        studio.documents = (0..<64).map { LibraryDocument(projectID: projectA, fileName: "\($0).txt", originalPath: "", text: String(repeating: "x", count: 256 * 1024)) }
        studio.featureError = nil
        await studio.importLibraryFiles([file])
        check(studio.documents.count == 64 && studio.featureError != nil, "Library byte limit")
        studio.documents = savedDocuments; studio.featureError = nil
        let invalidFile = root.appendingPathComponent("invalid.txt")
        try Data([0xff, 0xfe, 0xff]).write(to: invalidFile)
        await studio.importLibraryFiles([invalidFile])
        check(studio.documents.count == 1 && studio.featureError != nil, "Invalid document reports failure")
        studio.featureError = nil
        studio.chooseProject(projectB)
        check(DocumentSearch.search("satellites", documents: studio.documents.filter { $0.projectID == projectB }).isEmpty, "No cross-project retrieval")
        studio.chooseProject(projectA)
        let pdf = LibraryDocument(projectID: projectA, fileName: "book.pdf", originalPath: "/missing/book.pdf", text: "[Page 2]\nVénus est chaude.\n[Page 7]\nMars possède deux satellites : Phobos et Deimos.")
        let results = DocumentSearch.search("satellites Mars", documents: [pdf])
        check(results.count == 1 && results[0].page == 7, "Independent page oracle")
        check(results[0].text.contains("Phobos"), "Relevant passage")
        check(DocumentSearch.search("", documents: [pdf]).isEmpty, "Empty query")
        check(DocumentSearch.search("Mars", documents: [pdf], limit: 0).isEmpty, "Zero limit")
        check(DocumentSearch.search("absent", documents: [pdf]).isEmpty, "No fabricated result")
        check(DocumentSearch.search("VENUS", documents: [pdf]).first?.page == 2, "Accent-insensitive retrieval")
        for length in [0, 1, 1198, 1199, 1200, 1201, 2400, 10000] {
            let document = LibraryDocument(projectID: projectA, fileName: "long.txt", originalPath: "", text: "Mars " + String(repeating: "a", count: length) + "\nMars\n" + String(repeating: "\n", count: 1400))
            let found = DocumentSearch.search("Mars", documents: [document], limit: 999)
            check(!found.isEmpty && found.count <= 8 && found.allSatisfy { $0.text.count <= 1200 }, "Bounded excerpts \(length)")
            check(DocumentSearch.search("Mars", documents: [document], limit: 999) == found, "Deterministic order")
        }
        // Deterministic generated fixtures exercise Unicode and arbitrary line boundaries.
        var random: UInt64 = 17
        for _ in 0..<128 {
            random = random &* 6364136223846793005 &+ 1
            let text = String(repeating: "é🦀 Mars\n", count: Int(random % 500 + 1))
            let document = LibraryDocument(projectID: projectA, fileName: "unicode.txt", originalPath: "", text: text)
            check(DocumentSearch.search("Mars", documents: [document]).allSatisfy { !$0.text.isEmpty && $0.text.count <= 1200 }, "Generated excerpt boundary")
        }
        studio.draft = "une question"
        let original = studio.currentConversation!.id
        let first = ChatMessage(role: .user, content: "Question originale", attachments: [ChatAttachment(fileName: "x.txt", text: "original", fileBytes: 8)])
        let second = ChatMessage(role: .assistant, content: "Réponse originale")
        second.sources = results
        let index = studio.conversations.firstIndex(where: { $0.id == original })!
        studio.conversations[index].messages = [first, second]
        check(studio.branchConversation(at: second.id), "Branch production entry")
        check(studio.currentConversation?.parentConversationID == original, "Branch lineage")
        check(studio.currentConversation?.projectID == projectA && studio.currentConversation?.messages.count == 2, "Branch scope and length")
        studio.currentConversation?.messages.last?.append(" + modifiée", phase: "answer")
        check(second.content == "Réponse originale", "Branch does not share mutable messages")
        check(studio.currentConversation?.messages.last?.sources == results, "Sources survive branch")
        studio.selectConversation(original)
        studio.editMessage(first.id)
        check(studio.currentConversation?.messages.isEmpty == true && studio.draft == first.content, "Edit creates editable branch")
        check(studio.pendingAttachments.count == 1 && studio.conversations.first(where: { $0.id == original })?.messages.count == 2, "Edit keeps attachments and original")
        check(!studio.branchConversation(at: UUID()), "Unknown message rejected")
        studio.systemPrompt = "Réponds en français."
        studio.responseSchema = "{\"type\":\"object\"}"
        studio.temperature = 0.4; studio.topK = 20; studio.repetitionPenalty = 1.1
        studio.saveProfile("Rédaction")
        let profile = studio.profiles[0]
        studio.systemPrompt = "changed"; studio.responseSchema = ""; studio.temperature = 1
        studio.applyProfile(profile)
        check(studio.systemPrompt == "Réponds en français." && studio.temperature == 0.4 && !studio.responseSchema.isEmpty, "Apply profile")
        var invalidProfile = profile; invalidProfile.temperature = .nan
        studio.applyProfile(invalidProfile)
        check(studio.temperature == 0.4 && studio.featureError != nil, "Invalid profile preserves settings")
        studio.featureError = nil
        let markdown = MarkdownExport.render(studio.conversations.first(where: { $0.id == original })!.snapshot)
        check(markdown.contains("Question originale") && markdown.contains("Réponse originale") && markdown.contains("book.pdf · p. 7") && markdown.contains("Phobos"), "Markdown includes saved sources")
        check(!markdown.contains("Réponds en français"), "No system prompt in export")
        for (weights, memory, expected) in [(Int64(0), UInt64(24), ModelAdvice.Fit.unknown), (10, 100, .comfortable), (55, 100, .tight), (75, 100, .exceeds), (Int64.max, UInt64.max, .comfortable)] {
            check(ModelAdvice(weights: weights, memory: memory).fit == expected, "Memory budget boundaries")
        }
        let configURL = root.appendingPathComponent("isolated-mcp.json")
        try MCPConfiguration.saveServer(name: "local", json: "{\"command\":\"/usr/bin/true\",\"args\":[],\"env\":{\"EXAMPLE\":\"private\"}}", at: configURL)
        try MCPConfiguration.saveServer(name: "remote", json: "{\"url\":\"https://example.com/mcp\",\"enabled\":false}", at: configURL)
        try check(try MCPConfiguration.servers(at: configURL).count == 2, "MCP merge preserves other servers")
        try check((try FileManager.default.attributesOfItem(atPath: configURL.path)[.posixPermissions] as? NSNumber)?.intValue == 0o600, "MCP credentials are owner-only")
        try check(!(try FileManager.default.contentsOfDirectory(atPath: root.path)).contains(where: { $0.hasPrefix(".mcp-") }), "MCP atomic save cleans temporary files")
        let before = try Data(contentsOf: configURL)
        for (name, config) in [("bad/name", "{}"), ("bad", "{\"url\":\"http://example.com\"}"), ("bad", "{\"command\":\"x\",\"args\":[1]}"), ("bad", "{\"url\":\"https://example.com\",\"headers\":{\"X\":\"a\\nb\"}}"), ("bad", "[]"), ("bad", "{\"url\":\"https://example.com\",\"enabled\":1}"), ("bad", "{\"url\":\"https://example.com\",\"command\":4}")] {
            do { try MCPConfiguration.saveServer(name: name, json: config, at: configURL); preconditionFailure("Invalid config accepted: \(name) \(config)") } catch { checks += 1 }
            try check(try Data(contentsOf: configURL) == before, "Invalid MCP leaves configuration untouched")
        }
        try MCPConfiguration.saveServer(name: "remote", json: nil, at: configURL)
        try check(try MCPConfiguration.servers(at: configURL)["local"]?["env"] as? [String: String] == ["EXAMPLE":"private"], "MCP removal preserves secrets")
        studio.settingsDidChange()
        try await wait { (try? ConversationStore.load(from: history))?.profiles?.count == 1 }
        let restored = StudioModel(conversationFileURL: history, isPreview: true, preferences: preferences)
        check(restored.projects.count == 2 && restored.documents.count == 1 && restored.profiles.count == 1, "Workspace persistence")
        var renamed = studio.currentProject!
        renamed.name = String(repeating: "x", count: 101)
        studio.updateProject(renamed)
        check(studio.currentProject!.name.count <= 100, "Rename preserves the persistence name limit")
        let transitions = StudioModel(conversationFileURL: root.appendingPathComponent("transitions.json"), isPreview: true, preferences: preferences)
        transitions.addProject("A")
        let transitionA = transitions.selectedProjectID!
        transitions.addProject("B")
        let transitionB = transitions.selectedProjectID!
        let deletedID = transitions.selectedConversationID!
        transitions.deleteConversation(deletedID)
        check(transitions.selectedProjectID == transitionA && transitions.currentConversation?.projectID == transitionA, "Deletion selects the next conversation's project")
        transitions.undoDeleteConversation()
        check(transitions.selectedProjectID == transitionB && transitions.selectedConversationID == deletedID, "Undo restores the conversation's project permissions")
        transitions.conversations = [transitions.currentConversation!]
        transitions.deleteConversation(deletedID)
        check(transitions.currentConversation?.projectID == transitionB && transitions.currentConversation?.messages.isEmpty == true, "Clearing the last conversation preserves its project")
        let recoveryURL = root.appendingPathComponent("recovery.json")
        try Data(contentsOf: history).write(to: recoveryURL.appendingPathExtension("backup"))
        try Data("broken".utf8).write(to: recoveryURL)
        let recovery = StudioModel(conversationFileURL: recoveryURL, isPreview: true, preferences: preferences)
        recovery.recoverConversations()
        check(recovery.projects.count == 2 && recovery.profiles.count == 1 && recovery.documents.count == 1, "Recovery restores project documents and profiles together")
        let saved = try JSONSerialization.jsonObject(with: Data(contentsOf: history)) as! [String: Any]
        var excessive = saved
        let savedProfile = (saved["profiles"] as! [[String: Any]])[0]
        excessive["profiles"] = (0..<128).map { _ in var profile = savedProfile; profile["id"] = UUID().uuidString; return profile }
        let excessiveURL = root.appendingPathComponent("excessive.json")
        try JSONSerialization.data(withJSONObject: excessive).write(to: excessiveURL)
        let beforeImport = try Data(contentsOf: history)
        let conversationCount = studio.conversations.count
        studio.importWorkspace(from: excessiveURL)
        check(studio.storageError != nil && studio.profiles.count == 1 && studio.conversations.count == conversationCount, "Oversized merge preserves the in-memory workspace")
        try check(try Data(contentsOf: history) == beforeImport, "Rejected import leaves disk history untouched")
        studio.storageError = nil
        var orphan = saved
        orphan["projects"] = []
        orphan["documents"] = []
        let orphanURL = root.appendingPathComponent("orphan.json")
        try JSONSerialization.data(withJSONObject: orphan).write(to: orphanURL)
        do { _ = try ConversationStore.load(from: orphanURL); preconditionFailure("Orphaned project references accepted") } catch { checks += 1 }
        let imported = StudioModel(conversationFileURL: root.appendingPathComponent("imported.json"), isPreview: true, preferences: preferences)
        imported.importWorkspace(from: history)
        check(imported.projects.count == 2 && imported.documents.count == 1 && imported.profiles.count == 1, "File import restores all workspace collections")
        var legacy = saved
        legacy["version"] = 1
        for key in ["projects", "profiles", "documents", "responseSchema"] { legacy.removeValue(forKey: key) }
        legacy["conversations"] = (legacy["conversations"] as! [[String: Any]]).map { row in
            var row = row; row.removeValue(forKey: "projectID"); row.removeValue(forKey: "parentConversationID"); return row
        }
        let legacyURL = root.appendingPathComponent("legacy.json")
        try JSONSerialization.data(withJSONObject: legacy).write(to: legacyURL)
        try check(try ConversationStore.load(from: legacyURL)?.conversations.count == studio.conversations.count, "v1 migration")
        let engine = StudioModel(conversationFileURL: root.appendingPathComponent("engine.json"), preferences: preferences)
        engine.models = [LocalModel(name: "workspace", path: root.path, modelType: "fixture", format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
        engine.selectModel("workspace")
        try await wait { engine.engineState.isReady }
        engine.addProject("Flight")
        var project = engine.currentProject!; project.instructions = "Project instruction"; project.allowedTools = ["local.read"]
        engine.updateProject(project)
        engine.documents = [LibraryDocument(projectID: project.id, fileName: "flight.txt", originalPath: "", text: "The launch vehicle is named Orion.")]
        engine.draft = "What is the launch vehicle?"; engine.send()
        try await wait { !engine.isGenerating }
        let response = engine.currentConversation!.messages.last!
        let payload = try JSONSerialization.jsonObject(with: Data(response.content.utf8)) as! [String: Any]
        check(payload["allowed_tools"] as? [String] == ["local.read"], "Permissions cross real subprocess boundary")
        let messages = payload["messages"] as! [[String: Any]]
        check((messages[0]["content"] as! String).contains("Project instruction"), "Project instruction reaches engine")
        check((messages.last!["content"] as! String).contains("Orion") && response.sources.count == 1, "Retrieval reaches production bridge")
        let count = engine.conversations.count
        engine.regenerate(response.id)
        try await wait { !engine.isGenerating }
        check(engine.conversations.count == count + 1 && engine.currentConversation?.messages.count == 2, "Regeneration end-to-end preserves original")
        engine.responseSchema = "{\"type\":\"object\"}"; engine.draft = "JSON"; engine.send()
        try await wait { !engine.isGenerating }
        let jsonPayload = try JSONSerialization.jsonObject(with: Data(engine.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
        check(jsonPayload["response_schema"] as? String == engine.responseSchema, "Schema reaches bridge")
        engine.responseSchema = "broken"; engine.draft = "invalid"; engine.send()
        check(!engine.isGenerating && engine.featureError != nil && engine.draft == "invalid", "Invalid schema keeps draft")
        engine.ejectModel()
        let oldEngine = StudioModel(conversationFileURL: root.appendingPathComponent("old-engine.json"), preferences: preferences)
        oldEngine.models = [LocalModel(name: "legacy-features", path: root.path, modelType: "fixture", format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
        oldEngine.selectModel("legacy-features")
        try await wait { oldEngine.engineState.isReady }
        oldEngine.responseSchema = "{\"type\":\"object\"}"; oldEngine.draft = "JSON"
        oldEngine.send()
        check(!oldEngine.isGenerating && oldEngine.featureError != nil && oldEngine.currentConversation!.messages.isEmpty, "Older runtimes cannot silently ignore JSON constraints")
        oldEngine.ejectModel()
        studio.deleteProject(projectA)
        check(studio.documents.isEmpty && studio.conversations.contains(where: { $0.id == original && $0.projectID == nil }), "Project removal retains conversations")
        studio.voice.stop(); studio.voice.stop()
        check(!studio.voice.recording && !studio.voice.starting, "Voice stop is idempotent without permissions")
        _ = NSApplication.shared
        NSApp.setActivationPolicy(.prohibited)
        func checkPanel<V: View>(_ view: V, size: NSSize) async throws {
            let host = NSHostingView(rootView: view.environmentObject(studio).preferredColorScheme(.dark))
            let window = NSWindow(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless], backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.appearance = NSAppearance(named: .darkAqua)
            window.contentView = host
            defer { window.close() }
            host.frame = NSRect(origin: .zero, size: size)
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(100))
            let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
            host.cacheDisplay(in: host.bounds, to: bitmap)
            let corner = bitmap.colorAt(x: 2, y: 2)!.usingColorSpace(.deviceRGB)!
            check(corner.alphaComponent > 0.99, "Panel background is opaque")
            check(max(corner.redComponent, corner.greenComponent, corner.blueComponent) < 0.15, "Panel uses the dark app background")
        }
        try await checkPanel(WorkspaceManagerView(), size: NSSize(width: 740, height: 590))
        try await checkPanel(MCPManagerView(), size: NSSize(width: 850, height: 620))
        try await checkPanel(DocumentSourceView(source: results[0]), size: NSSize(width: 720, height: 560))
        print("Workspace checks passed: \(checks), including deterministic retrieval, persistence, branching, profiles, MCP validation and bridge journeys")
    }
}
