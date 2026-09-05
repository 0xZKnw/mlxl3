import Foundation

@MainActor
enum ChatTimelineCheck {
    static func run() throws {
        func check(_ condition: Bool, _ label: String) throws {
            if !condition { throw MCPPreferenceCheck.Failure(message: label) }
        }
        let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
        message.processing("Préparation du contexte")
        message.processing("Lecture · 500 tokens")
        try check(message.parts.count == 1, "Progress updates must not append blocks")
        message.append("Je cherche.", phase: "thinking")
        let firstThinkingID = message.parts.last!.id
        message.startTool(id: "exa-1", serverName: "exa", toolName: "web_search_exa")
        message.finishTool(id: "exa-1", result: "Documentation", isError: false)
        message.processing("Lecture des résultats MCP")
        message.append("Je compare.", phase: "thinking")
        message.append(" Voici.", phase: "thinking")
        message.append("La réponse.", phase: "answer")
        message.finish(stats: nil, fallbackAnswer: nil, cacheContext: "full context")
        try check(message.parts.map(\.kind) == [.processing, .thinking, .tool, .processing, .thinking, .answer],
                  "Tool rounds must remain chronological")
        try check(message.parts[1].id == firstThinkingID && message.parts[1].text == "Je cherche.",
                  "Later reasoning mutated the initial block")
        try check(message.parts[4].text == "Je compare. Voici.", "Streaming text lost")
        try check(message.parts.filter { $0.kind == .processing }.allSatisfy { $0.elapsed != nil },
                  "Processing never completed")
        let encoded = try JSONEncoder().encode(message.snapshot)
        let snapshot = try JSONDecoder().decode(ChatMessageSnapshot.self, from: encoded)
        let restored = ChatMessage(snapshot: snapshot)!
        try check(restored.parts.map(\.id) == message.parts.map(\.id), "Timeline not persisted")
        try check(restored.toolActivities[0].state == .complete, "Tool state lost on relaunch")
        var legacy = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
        legacy.removeValue(forKey: "parts")
        let oldSnapshot = try JSONDecoder().decode(ChatMessageSnapshot.self, from: JSONSerialization.data(withJSONObject: legacy))
        try check(ChatMessage(snapshot: oldSnapshot)!.content == message.content, "Legacy chat migration failed")
        let interrupted = ChatMessage(role: .assistant, content: "", isStreaming: true)
        interrupted.processing("Traitement")
        interrupted.fail("Arrêtée")
        try check(interrupted.parts[0].elapsed != nil && !interrupted.isStreaming, "Cancellation left a live spinner")
    }
}
