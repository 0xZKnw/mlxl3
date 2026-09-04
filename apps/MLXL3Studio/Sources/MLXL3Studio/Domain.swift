import Combine
import Foundation

struct LocalModel: Codable, Hashable, Identifiable {
    let name: String
    let path: String
    let modelType: String
    let format: String
    let bits: Double?
    let sizeBytes: Int64
    let modules: Int
    let addedAt: String
    let size: String

    var id: String { name }

    enum CodingKeys: String, CodingKey {
        case name, path, format, bits, modules, size
        case modelType = "model_type"
        case sizeBytes = "size_bytes"
        case addedAt = "added_at"
    }

    var architectureLabel: String {
        modelType.replacingOccurrences(of: "_", with: " ").uppercased()
    }
}

struct GenerationStats: Codable, Hashable, Sendable {
    let ttftSeconds: Double
    let prefillTps: Double
    let decodeTps: Double
    let promptTokens: Int
    let generatedTokens: Int
    let peakMemoryGB: Double
    let cachedPromptTokens: Int?
    let evaluatedPromptTokens: Int?

    var cacheHitPercent: Double {
        let cached = cachedPromptTokens ?? 0
        let evaluated = evaluatedPromptTokens ?? promptTokens
        let total = cached + evaluated
        return total > 0 ? 100 * Double(cached) / Double(total) : 0
    }

    enum CodingKeys: String, CodingKey {
        case ttftSeconds = "ttft_seconds"
        case prefillTps = "prefill_tps"
        case decodeTps = "decode_tps"
        case promptTokens = "prompt_tokens"
        case generatedTokens = "generated_tokens"
        case peakMemoryGB = "peak_memory_gb"
        case cachedPromptTokens = "cached_prompt_tokens"
        case evaluatedPromptTokens = "evaluated_prompt_tokens"
    }
}

struct BridgeEvent: Decodable {
    let type: String
    let model: String?
    let modules: Int?
    let loadSeconds: Double?
    let residentGB: Double?
    let requestID: String?
    let phase: String?
    let text: String?
    let assistantContext: String?
    let cacheContext: String?
    let stats: GenerationStats?
    let message: String?
    let mcpServers: Int?
    let mcpTools: Int?
    let mcpErrors: [String: String]?
    let toolCallID: String?
    let toolName: String?
    let serverName: String?
    let isError: Bool?

    enum CodingKeys: String, CodingKey {
        case type, model, modules, phase, text, stats, message
        case loadSeconds = "load_seconds"
        case residentGB = "resident_gb"
        case requestID = "request_id"
        case assistantContext = "assistant_context"
        case cacheContext = "cache_context"
        case mcpServers = "mcp_servers"
        case mcpTools = "mcp_tools"
        case mcpErrors = "mcp_errors"
        case toolCallID = "tool_call_id"
        case toolName = "tool_name"
        case serverName = "server_name"
        case isError = "is_error"
    }
}

struct ToolActivity: Codable, Hashable, Identifiable, Sendable {
    enum State: String, Codable, Sendable {
        case running
        case complete
        case failed
    }

    let id: String
    let serverName: String?
    let toolName: String
    var state: State
    var result: String?
}

struct PromptMessage: Codable, Hashable {
    let role: String
    let content: String
}

struct GenerationRequest: Encodable {
    let type = "generate"
    let requestID: String
    let conversationID: String
    let messages: [PromptMessage]
    let maxTokens: Int
    let temperature: Double
    let topK: Int
    let repetitionPenalty: Double

    enum CodingKeys: String, CodingKey {
        case type, messages, temperature
        case requestID = "request_id"
        case conversationID = "conversation_id"
        case maxTokens = "max_tokens"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
    }
}

final class ChatMessage: ObservableObject, Identifiable {
    enum Role: String {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    private(set) var content: String
    private(set) var thinking: String
    private(set) var isStreaming: Bool
    private(set) var stats: GenerationStats?
    private(set) var error: String?
    private(set) var toolActivities: [ToolActivity]
    private(set) var cacheContext: String?
    private(set) var streamRevision = 0

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        thinking: String = "",
        isStreaming: Bool = false,
        stats: GenerationStats? = nil,
        error: String? = nil,
        toolActivities: [ToolActivity] = [],
        cacheContext: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.thinking = thinking
        self.isStreaming = isStreaming
        self.stats = stats
        self.error = error
        self.toolActivities = toolActivities
        self.cacheContext = cacheContext
    }

    func append(_ text: String, phase: String?) {
        guard !text.isEmpty else { return }
        objectWillChange.send()
        if phase == "thinking" {
            thinking += text
        } else {
            content += text
        }
        streamRevision &+= 1
    }

    func finish(stats: GenerationStats?, fallbackAnswer: String?, cacheContext: String?) {
        objectWillChange.send()
        isStreaming = false
        self.stats = stats
        self.cacheContext = cacheContext
        if content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let fallbackAnswer {
            content = fallbackAnswer
        }
        streamRevision &+= 1
    }

    func fail(_ message: String) {
        objectWillChange.send()
        isStreaming = false
        error = message
        streamRevision &+= 1
    }

    func startTool(id: String, serverName: String?, toolName: String) {
        objectWillChange.send()
        toolActivities.append(
            ToolActivity(
                id: id,
                serverName: serverName,
                toolName: toolName,
                state: .running,
                result: nil
            )
        )
        streamRevision &+= 1
    }

    func finishTool(id: String, result: String?, isError: Bool) {
        guard let index = toolActivities.firstIndex(where: { $0.id == id }) else { return }
        objectWillChange.send()
        toolActivities[index].state = isError ? .failed : .complete
        toolActivities[index].result = result
        streamRevision &+= 1
    }

    var snapshot: ChatMessageSnapshot {
        ChatMessageSnapshot(
            id: id,
            role: role.rawValue,
            content: content,
            thinking: thinking,
            wasStreaming: isStreaming,
            stats: stats,
            error: error,
            toolActivities: toolActivities,
            cacheContext: cacheContext
        )
    }

    convenience init?(snapshot: ChatMessageSnapshot) {
        guard let role = Role(rawValue: snapshot.role) else { return nil }
        self.init(
            id: snapshot.id,
            role: role,
            content: snapshot.content,
            thinking: snapshot.thinking,
            isStreaming: false,
            stats: snapshot.stats,
            error: snapshot.error ?? (snapshot.wasStreaming ? "Génération interrompue" : nil),
            toolActivities: snapshot.toolActivities ?? [],
            cacheContext: snapshot.cacheContext
        )
    }
}

struct Conversation: Identifiable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    let createdAt: Date

    init(
        id: UUID = UUID(),
        title: String = "Nouvelle conversation",
        messages: [ChatMessage] = [],
        createdAt: Date = Date()
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
    }

    var snapshot: ConversationSnapshot {
        ConversationSnapshot(
            id: id,
            title: title,
            messages: messages.map(\.snapshot),
            createdAt: createdAt
        )
    }

    init(snapshot: ConversationSnapshot) {
        self.init(
            id: snapshot.id,
            title: snapshot.title,
            messages: snapshot.messages.compactMap(ChatMessage.init(snapshot:)),
            createdAt: snapshot.createdAt
        )
    }
}

enum EngineState: Equatable {
    case idle
    case loading(String)
    case ready(model: String, modules: Int, residentGB: Double)
    case generating
    case failed(String)

    var label: String {
        switch self {
        case .idle: "Modèle éjecté"
        case let .loading(model): "Chargement de \(model)…"
        case .ready: "Prêt sur Metal"
        case .generating: "Génération…"
        case .failed: "Indisponible"
        }
    }

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}
