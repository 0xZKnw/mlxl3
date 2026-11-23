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
        // The code a tool ran is kept with it; conversations saved before decode without it.
        let agent = ChatMessage(role: .assistant, content: "", isStreaming: true)
        agent.startTool(id: "js-1", serverName: "sandbox", toolName: "javascript", arguments: "const x = 1;\nreturn x")
        agent.finishTool(id: "js-1", result: "Result:\n1", isError: false)
        let agentData = try JSONEncoder().encode(agent.snapshot)
        let agentRestored = ChatMessage(snapshot: try JSONDecoder().decode(ChatMessageSnapshot.self, from: agentData))!
        try check(agentRestored.toolActivities[0].arguments == "const x = 1;\nreturn x", "Tool code lost on relaunch")
        try check(restored.toolActivities[0].arguments == nil, "A tool without arguments gained some")
        let oldActivity = try JSONDecoder().decode(
            ToolActivity.self,
            from: Data(#"{"id":"a","serverName":"exa","toolName":"t","state":"complete","result":"r"}"#.utf8))
        try check(oldActivity.arguments == nil && oldActivity.result == "r", "Saved tool activities must still decode")
        func shown(_ json: String) throws -> String? {
            try JSONDecoder().decode(
                BridgeEvent.self,
                from: Data((#"{"type":"tool_start","request_id":"r","tool_call_id":"c","tool_name":"t""# + json + "}").utf8)
            ).arguments?.text
        }
        try check(try shown(#","arguments":{"code":"const x = 1;\nreturn x"}"#) == "const x = 1;\nreturn x", "JavaScript code shown as written")
        try check(try shown(#","arguments":{"path":"a.txt","limit":2}"#) == #"{"limit":2,"path":"a.txt"}"#, "Other arguments shown as sorted JSON")
        try check(try shown(#","arguments":{"code":"x","path":"a"}"#) == #"{"code":"x","path":"a"}"#, "Code beside other arguments stays JSON")
        try check(try shown(#","arguments":"{\"code\":\"…""#) == #"{"code":"…"#, "Truncated arguments shown as text")
        try check(try shown("") == nil && (try shown(#","arguments":null"#)) == nil, "Older engines send no arguments")
        let interrupted = ChatMessage(role: .assistant, content: "", isStreaming: true)
        interrupted.processing("Traitement")
        interrupted.fail("Arrêtée")
        try check(interrupted.parts[0].elapsed != nil && !interrupted.isStreaming, "Cancellation left a live spinner")
    }
}
