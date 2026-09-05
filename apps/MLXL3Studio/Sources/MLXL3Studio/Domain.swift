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
    var contextUsed: Int? = nil
    var contextLimit: Int? = nil

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
        case contextUsed = "context_used"
        case contextLimit = "context_limit"
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
    var usedTokens: Int? = nil
    var contextLimit: Int? = nil
    var modelContextLimit: Int? = nil
    var contextFull: Bool? = nil
    var contextMemory: ContextMemoryProfile? = nil

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
        case usedTokens = "used_tokens"
        case contextLimit = "context_limit"
        case modelContextLimit = "model_context_limit"
        case contextFull = "context_full"
        case contextMemory = "context_memory"
    }
}

struct ContextMemoryProfile: Decodable {
    struct Layer: Decodable {
        let bytesPerToken: Int
        let maxTokens: Int?
        let step: Int
        enum CodingKeys: String, CodingKey {
            case bytesPerToken = "bytes_per_token"
            case maxTokens = "max_tokens"
            case step
        }
    }
    let layers: [Layer]
    let fixedBytes: Int
    enum CodingKeys: String, CodingKey {
        case layers
        case fixedBytes = "fixed_bytes"
    }
    func bytes(tokens: Int) -> Double {
        guard tokens > 0 else { return 0 }
        return layers.reduce(Double(fixedBytes)) { total, layer in
            let step = max(layer.step, 1)
            let allocated = ((tokens + step - 1) / step) * step
            return total + Double(min(allocated, layer.maxTokens ?? allocated)) * Double(layer.bytesPerToken)
        }
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

/// Stable, chronological blocks. Never move later reasoning above a tool call.
struct AssistantPart: Codable, Identifiable, Sendable {
    enum Kind: String, Codable, Sendable { case thinking, answer, tool, processing }
    let id: UUID
    let kind: Kind
    var text: String
    var toolID: String?
    var startedAt: Date?
    var elapsed: Double?
    var interrupted: Bool?

    init(kind: Kind, text: String = "", toolID: String? = nil) {
        id = UUID()
        self.kind = kind
        self.text = text
        self.toolID = toolID
        startedAt = kind == .processing ? Date() : nil
    }
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
    var mcpEnabled: Bool = false

    enum CodingKeys: String, CodingKey {
        case type, messages, temperature
        case requestID = "request_id"
        case conversationID = "conversation_id"
        case maxTokens = "max_tokens"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case mcpEnabled = "mcp_enabled"
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
    private(set) var parts: [AssistantPart]
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
        cacheContext: String? = nil,
        parts: [AssistantPart]? = nil
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
        // Old snapshots remain readable; their original ordering was not saved.
        self.parts = parts ?? (
            (thinking.isEmpty ? [] : [AssistantPart(kind: .thinking, text: thinking)])
            + toolActivities.map { AssistantPart(kind: .tool, toolID: $0.id) }
            + (content.isEmpty ? [] : [AssistantPart(kind: .answer, text: content)])
        )
    }

    func append(_ text: String, phase: String?) {
        guard !text.isEmpty else { return }
        objectWillChange.send()
        endProcessing()
        let kind: AssistantPart.Kind = phase == "thinking" ? .thinking : .answer
        if parts.last?.kind != kind { parts.append(AssistantPart(kind: kind)) }
        parts[parts.count - 1].text += text
        if phase == "thinking" {
            thinking += text
        } else {
            content += text
        }
        streamRevision &+= 1
    }

    func finish(stats: GenerationStats?, fallbackAnswer: String?, cacheContext: String?) {
        objectWillChange.send()
        endProcessing()
        isStreaming = false
        self.stats = stats
        self.cacheContext = cacheContext
        if content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
           let fallbackAnswer {
            content = fallbackAnswer
            if !fallbackAnswer.isEmpty { parts.append(AssistantPart(kind: .answer, text: fallbackAnswer)) }
        }
        streamRevision &+= 1
    }

    func fail(_ message: String) {
        objectWillChange.send()
        endProcessing(interrupted: true)
        isStreaming = false
        error = message
        streamRevision &+= 1
    }

    func startTool(id: String, serverName: String?, toolName: String) {
        objectWillChange.send()
        endProcessing()
        parts.append(AssistantPart(kind: .tool, toolID: id))
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

    func processing(_ label: String) {
        objectWillChange.send()
        if parts.last?.kind != .processing || parts.last?.elapsed != nil {
            parts.append(AssistantPart(kind: .processing))
        }
        parts[parts.count - 1].text = label
        streamRevision &+= 1
    }

    private func endProcessing(interrupted: Bool = false) {
        guard let last = parts.last, last.kind == .processing,
              last.elapsed == nil, let start = last.startedAt else { return }
        parts[parts.count - 1].elapsed = Date().timeIntervalSince(start)
        parts[parts.count - 1].interrupted = interrupted
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
            cacheContext: cacheContext,
            parts: parts.map { part in
                var saved = part
                if saved.kind == .processing, saved.elapsed == nil, let start = saved.startedAt {
                    saved.elapsed = Date().timeIntervalSince(start)
                    saved.interrupted = true
                }
                return saved
            }
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
            error: snapshot.error ?? (snapshot.wasStreaming ? L("Génération interrompue", "Generation interrupted") : nil),
            toolActivities: snapshot.toolActivities ?? [],
            cacheContext: snapshot.cacheContext,
            parts: snapshot.parts
        )
        endProcessing(interrupted: snapshot.wasStreaming)
    }
}

struct ContextUsage: Codable, Sendable {
    let used: Int
    let limit: Int
    let model: String
}

struct Conversation: Identifiable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    let createdAt: Date
    var contextUsage: ContextUsage?

    init(
        id: UUID = UUID(),
        title: String = L("Nouvelle conversation", "New conversation"),
        messages: [ChatMessage] = [],
        createdAt: Date = Date(),
        contextUsage: ContextUsage? = nil
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.createdAt = createdAt
        self.contextUsage = contextUsage
    }

    var snapshot: ConversationSnapshot {
        ConversationSnapshot(
            id: id,
            title: title,
            messages: messages.map(\.snapshot),
            createdAt: createdAt,
            contextUsage: contextUsage
        )
    }

    init(snapshot: ConversationSnapshot) {
        self.init(
            id: snapshot.id,
            title: snapshot.title,
            messages: snapshot.messages.compactMap(ChatMessage.init(snapshot:)),
            createdAt: snapshot.createdAt,
            contextUsage: snapshot.contextUsage
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
        case .idle: L("Modèle éjecté", "Model unloaded")
        case let .loading(model): L("Chargement de \(model)…", "Loading \(model)…")
        case .ready: L("Prêt sur Metal", "Ready on Metal")
        case .generating: L("Génération…", "Generating…")
        case .failed: L("Indisponible", "Unavailable")
        }
    }

    var isReady: Bool {
        if case .ready = self { return true }
        return false
    }
}

enum ModelInstallState: Equatable {
    case idle
    case working(String)
    case succeeded(String)
    case failed(String)

    var isWorking: Bool {
        if case .working = self { return true }
        return false
    }
}
