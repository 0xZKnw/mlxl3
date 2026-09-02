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

struct GenerationStats: Codable, Hashable {
    let ttftSeconds: Double
    let prefillTps: Double
    let decodeTps: Double
    let promptTokens: Int
    let generatedTokens: Int
    let peakMemoryGB: Double

    enum CodingKeys: String, CodingKey {
        case ttftSeconds = "ttft_seconds"
        case prefillTps = "prefill_tps"
        case decodeTps = "decode_tps"
        case promptTokens = "prompt_tokens"
        case generatedTokens = "generated_tokens"
        case peakMemoryGB = "peak_memory_gb"
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
    let stats: GenerationStats?
    let message: String?

    enum CodingKeys: String, CodingKey {
        case type, model, modules, phase, text, stats, message
        case loadSeconds = "load_seconds"
        case residentGB = "resident_gb"
        case requestID = "request_id"
        case assistantContext = "assistant_context"
    }
}

struct PromptMessage: Codable, Hashable {
    let role: String
    let content: String
}

struct GenerationRequest: Encodable {
    let type = "generate"
    let requestID: String
    let messages: [PromptMessage]
    let maxTokens: Int
    let temperature: Double
    let topK: Int
    let repetitionPenalty: Double

    enum CodingKeys: String, CodingKey {
        case type, messages, temperature
        case requestID = "request_id"
        case maxTokens = "max_tokens"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
    }
}

struct ChatMessage: Identifiable, Hashable {
    enum Role: String, Hashable {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    var content: String
    var thinking: String
    var isStreaming: Bool
    var stats: GenerationStats?
    var error: String?

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        thinking: String = "",
        isStreaming: Bool = false,
        stats: GenerationStats? = nil,
        error: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.thinking = thinking
        self.isStreaming = isStreaming
        self.stats = stats
        self.error = error
    }
}

struct Conversation: Identifiable, Hashable {
    let id: UUID
    var title: String
    var messages: [ChatMessage]
    let createdAt: Date

    init(id: UUID = UUID(), title: String = "Nouvelle conversation") {
        self.id = id
        self.title = title
        self.messages = []
        self.createdAt = Date()
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
