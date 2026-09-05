import Foundation
import Metal

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
        // Regression: a GPU-private allocation must be counted, unlike process RSS.
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue(),
              let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder(),
              let before = MLXL3Bridge.memoryFootprintBytes(for: getpid()) else {
            throw Failure(message: "Cannot measure Metal memory footprint")
        }
        let bytes = 64 * 1024 * 1024
        guard let buffer = device.makeBuffer(length: bytes, options: .storageModePrivate) else {
            throw Failure(message: "Cannot allocate Metal test buffer")
        }
        blit.fill(buffer: buffer, range: 0..<bytes, value: 42)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        try check(command.status == .completed, "Metal memory test command failed")
        let after = MLXL3Bridge.memoryFootprintBytes(for: getpid()) ?? 0
        try check(after > before && after - before >= UInt64(bytes * 3 / 4), "Memory counter omitted GPU-private allocation")
        try check(MLXL3Bridge.memoryFootprintBytes(for: -1) == nil, "Failed memory read must not report zero")
        withExtendedLifetime(buffer) { }
        print("Metal footprint check passed: +\(after - before) bytes for a 64 MiB GPU buffer")
        let first = instance()
        try check(first.language == .fr, "Language must default to French")
        first.draft = "Do not translate this"
        first.setLanguage(.en)
        try check(L("Réglages", "Settings") == "Settings", "English translation not applied")
        try check(instance().language == .en, "Language was not persisted")
        try check(first.draft == "Do not translate this", "Language changed user text")
        first.setLanguage(.fr)

        first.models = [LocalModel(name: "test", path: "/nonexistent", modelType: "test", format: "EXL3",
                                  bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
        first.selectModel("test")
        let initialLimit = first.activeContextLimit
        first.contextLengthDraft = 8192
        try check(first.activeContextLimit == initialLimit, "Draft changed applied context before saving")
        try check(first.canSaveContext, "Valid context cannot be saved")
        first.saveContextAndReload()
        try check(first.activeContextLimit == 8192, "Reload did not apply saved context")
        let restored = instance()
        restored.selectedModelName = "test"
        try check(restored.savedContextLength == 8192, "Context limit was not persisted")
        restored.selectedModelName = "other"
        try check(restored.savedContextLength == 0, "Context preference leaked to another model")
        first.contextLengthDraft = -1
        try check(!first.canSaveContext, "Negative context accepted")
        first.contextLengthDraft = 999999999
        try check(!first.canSaveContext, "Beyond-architecture context accepted")
        let event = try JSONDecoder().decode(BridgeEvent.self, from: Data(
            #"{"type":"context_usage","request_id":"r","used_tokens":7000,"context_limit":8192}"#.utf8))
        try check(event.usedTokens == 7000 && event.contextLimit == 8192, "Context event decoding failed")
        var chat = Conversation()
        chat.contextUsage = ContextUsage(used: 7000, limit: 8192, model: "test")
        let decoded = try JSONDecoder().decode(ConversationSnapshot.self, from: JSONEncoder().encode(chat.snapshot))
        try check(decoded.contextUsage?.used == 7000, "Context usage not saved with conversation")
        let profile = try JSONDecoder().decode(ContextMemoryProfile.self, from: Data(
            #"{"layers":[{"bytes_per_token":384,"max_tokens":null,"step":256},{"bytes_per_token":384,"max_tokens":512,"step":256}],"fixed_bytes":4096}"#.utf8))
        try check(profile.bytes(tokens: 100) == 4096 + 384 * 256 * 2, "KV block rounding failed")
        try check(profile.bytes(tokens: 1024) == 4096 + 384 * (1024 + 512), "Sliding-window cap failed")
        first.contextLengthDraft = 4096
        let small = first.draftContextBytes
        first.contextLengthDraft = 8192
        try check((first.draftContextBytes ?? 0) > (small ?? 0), "Memory estimate did not update with draft")
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
