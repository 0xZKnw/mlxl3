import AppKit
import Foundation

@main struct HardeningCheck {
    @MainActor static func main() async throws {
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-check-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "io.mlxl3.check." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        func make(_ name: String) async throws -> StudioModel {
            let model = StudioModel(conversationFileURL: root.appendingPathComponent(name + ".json"), preferences: prefs)
            model.models = [LocalModel(name: name, path: root.path, modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
            model.selectModel(name)
            for _ in 0..<200 where !model.engineState.isReady { try await Task.sleep(for: .milliseconds(20)) }
            precondition(model.engineState.isReady, "Fixture not ready")
            return model
        }
        let crash = try await make("crash")
        crash.draft = "test"; crash.send()
        try await Task.sleep(for: .milliseconds(500))
        precondition(!crash.engineState.isReady, "Dead engine still ready")

        let deleted = try await make("delete")
        deleted.draft = "test"; deleted.send()
        deleted.deleteConversation(deleted.selectedConversationID!)
        try await Task.sleep(for: .milliseconds(600))
        precondition(!deleted.isGenerating, "Deleted response wedged generation")
        deleted.ejectModel()
        deleted.newConversation(); deleted.draft = "draft one"
        let first = deleted.selectedConversationID!
        deleted.newConversation(); deleted.draft = "draft two"
        deleted.selectConversation(first)
        precondition(deleted.draft == "draft one")

        let broken = root.appendingPathComponent("broken.json")
        try "{broken".write(to: broken, atomically: true, encoding: .utf8)
        let recovered = StudioModel(conversationFileURL: broken, isPreview: true, preferences: prefs)
        precondition(recovered.storageError != nil && !recovered.persistNow())
        let original = try String(contentsOf: broken, encoding: .utf8)
        precondition(original == "{broken")
        recovered.recoverConversations()
        precondition(recovered.persistNow())
        let store = ConversationStore(fileURL: root.appendingPathComponent("ordered.json"))
        let a = WorkspaceSnapshot(selectedConversationID: nil, selectedModelName: "old", conversations: [], temperature: 0, topK: 0, repetitionPenalty: 1, systemPrompt: "")
        let b = WorkspaceSnapshot(selectedConversationID: nil, selectedModelName: "new", conversations: [], temperature: 0, topK: 0, repetitionPenalty: 1, systemPrompt: "")
        try store.save(b, revision: 2); try store.save(a, revision: 1)
        let latest = try ConversationStore.load(from: store.fileURL)
        precondition(latest?.selectedModelName == "new")
        let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
        message.startTool(id: "tool", serverName: "local", toolName: "test")
        message.fail("interrupted")
        precondition(ChatMessage(snapshot: message.snapshot)!.toolActivities[0].state == .failed)
        precondition(SemanticVersion("1.0") == SemanticVersion("1.0.0"))
        MarkdownRegressionCheck.run()
        print("Desktop hardening checks passed: crash, deletion, drafts, recovery, save ordering, tool state")
    }
}
