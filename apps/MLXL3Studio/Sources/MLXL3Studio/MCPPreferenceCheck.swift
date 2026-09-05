import Foundation

/// Runs without XCTest/Xcode, a window, a model or the user's real preferences.
/// Invoke MLXL3Studio --check-mcp-preferences on a packaged release as well.
@MainActor
enum MCPPreferenceCheck {
    struct Failure: Error { let message: String }

    static func run() throws {
        let suite = "io.mlxl3.test.\(UUID().uuidString)"
        guard let preferences = UserDefaults(suiteName: suite) else {
            throw Failure(message: "Cannot create isolated preferences")
        }
        defer { preferences.removePersistentDomain(forName: suite) }
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString).appendingPathComponent("chats.json")
        func instance() -> StudioModel {
            StudioModel(conversationFileURL: file, isPreview: true, preferences: preferences)
        }
        func check(_ condition: Bool, _ message: String) throws {
            if !condition { throw Failure(message: message) }
        }
        let first = instance()
        try check(!first.mcpEnabled, "MCP must default off")
        first.setMCPEnabled(true)
        try check(first.mcpEnabled, "Enabling MCP failed")
        let relaunched = instance()
        try check(relaunched.mcpEnabled, "Enabled state was not persisted")
        relaunched.setMCPEnabled(false)
        try check(!instance().mcpEnabled, "Disabled state was not persisted")
        try check(first.mcpServerCount == 0, "Preview connected to a server")

        var request = GenerationRequest(
            requestID: "test", conversationID: "test", messages: [], maxTokens: -1,
            temperature: 0.2, topK: 80, repetitionPenalty: 1.05
        )
        for enabled in [false, true] {
            request.mcpEnabled = enabled
            let data = try JSONEncoder().encode(request)
            let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            try check(object?["mcp_enabled"] as? Bool == enabled, "MCP request flag missing")
        }
    }
}
