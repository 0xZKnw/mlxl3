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

struct EngineMemoryStats: Codable, Hashable, Sendable {
    let mlxActiveBytes: UInt64?
    let mlxCacheBytes: UInt64?
    let mlxPeakBytes: UInt64?
    var nativeMetalAllocatedBytes: UInt64? = nil
    let processFootprintBytes: UInt64?
    let processLifetimePeakBytes: UInt64?

    enum CodingKeys: String, CodingKey {
        case nativeMetalAllocatedBytes = "native_metal_allocated_bytes"
        case mlxActiveBytes = "mlx_active_bytes"
        case mlxCacheBytes = "mlx_cache_bytes"
        case mlxPeakBytes = "mlx_peak_bytes"
        case processFootprintBytes = "process_footprint_bytes"
        case processLifetimePeakBytes = "process_lifetime_peak_bytes"
    }
}

struct GenerationStats: Codable, Hashable, Sendable {
    let ttftSeconds: Double
    let prefillTps: Double
    let decodeTps: Double
    let promptTokens: Int
    let generatedTokens: Int
    let peakMemoryGB: Double? // Legacy history used this key for checkpoint size, not process peak.
    let modelSizeGB: Double?
    let cachedPromptTokens: Int?
    let evaluatedPromptTokens: Int?
    var contextUsed: Int? = nil
    var contextLimit: Int? = nil
    var elapsedSeconds: Double? = nil
    var endToEndTTFTSeconds: Double? = nil
    var toolRounds: Int? = nil
    var dflashProposedTokens: Int? = nil
    var dflashAcceptedTokens: Int? = nil
    var dflashBlocks: Int? = nil
    var mtpProposedTokens: Int? = nil
    var mtpAcceptedTokens: Int? = nil
    var mtpBlocks: Int? = nil
    var dflashBlockSeconds: Double? = nil
    var memory: EngineMemoryStats? = nil

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
        case modelSizeGB = "model_size_gb"
        case cachedPromptTokens = "cached_prompt_tokens"
        case evaluatedPromptTokens = "evaluated_prompt_tokens"
        case contextUsed = "context_used"
        case contextLimit = "context_limit"
        case elapsedSeconds = "elapsed_seconds"
        case endToEndTTFTSeconds = "end_to_end_ttft_seconds"
        case toolRounds = "tool_rounds"
        case dflashProposedTokens = "dflash_proposed_tokens"
        case dflashAcceptedTokens = "dflash_accepted_tokens"
        case dflashBlocks = "dflash_blocks"
        case mtpProposedTokens = "mtp_proposed_tokens"
        case mtpAcceptedTokens = "mtp_accepted_tokens"
        case mtpBlocks = "mtp_blocks"
        case dflashBlockSeconds = "dflash_block_seconds"
        case memory
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
    var stopReason: String? = nil
    var contextMemory: ContextMemoryProfile? = nil
    var nativeContextMemory: ContextMemoryProfile? = nil
    var backend: String? = nil
    var version: String? = nil
    var qwenProfile: String? = nil
    var qwenEffectiveFlags: [String: Bool]? = nil
    var qwenDeviceKV: Bool? = nil
    var qwenPrefillLastHead: Bool? = nil
    var qwenGroupExpertQMV: Bool? = nil
    var qwenGroupExpertStages: Bool? = nil
    var qwenDeviceGreedy: Bool? = nil
    var effectiveOptions: String {
        if let flags = qwenEffectiveFlags {
            return flags.keys.sorted().map { "qwen_" + $0 + "=" + String(flags[$0]!) }.joined(separator: ",")
        }
        return [("qwen_device_kv", qwenDeviceKV), ("qwen_prefill_last_head", qwenPrefillLastHead),
         ("qwen_group_expert_qmv", qwenGroupExpertQMV), ("qwen_group_expert_stages", qwenGroupExpertStages),
         ("qwen_device_greedy", qwenDeviceGreedy)].compactMap { key, value in
            value.map { key + "=" + String($0) }
        }.joined(separator: ",")
    }
    var turnContext: String? = nil
    var runtimeCommit: String? = nil
    var runtimeProfile: String? = nil
    var mlxVersion: String? = nil
    var runtimeExecutable: String? = nil
    var dflashRequested: Bool? = nil
    var dflashActive: Bool? = nil
    var dflashProposals: Int? = nil
    var dflashMode: Int? = nil
    var dflashDraftPath: String? = nil
    var dflashSupported: Bool? = nil
    var dflashTuneSupported: Bool? = nil
    var dflashConfigureSupported: Bool? = nil
    var dflashTuningKey: String? = nil
    var dflashFamily: String? = nil
    var mtpSupported: Bool? = nil
    var mtpAutoDownloadSupported: Bool? = nil
    var mtpConfigureSupported: Bool? = nil
    var mtpActive: Bool? = nil
    var mtpReason: String? = nil
    var mtpMaxDepth: Int? = nil
    var mtpTuneSupported: Bool? = nil
    var mtpTuningKey: String? = nil
    var depth: Int? = nil
    var completed: Int? = nil
    var total: Int? = nil
    var bestDepth: Int? = nil
    var tuningKey: String? = nil
    var rows: [MTPTuningRow]? = nil
    var workspaceFeatures: Bool? = nil
    var bridgeProtocol: Int? = nil
    var memorySaverSupported: Bool? = nil
    var mcpToolCatalog: [MCPToolDescriptor]? = nil

    enum CodingKeys: String, CodingKey {
        case type, model, modules, phase, text, stats, message
        case memorySaverSupported = "memory_saver_supported"
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
        case stopReason = "stop_reason"
        case contextMemory = "context_memory"
        case nativeContextMemory = "context_memory_profile"
        case backend
        case version = "runtime_version"
        case qwenProfile = "qwen_profile"
        case qwenEffectiveFlags = "qwen_effective_flags"
        case qwenDeviceKV = "qwen_device_kv"
        case qwenPrefillLastHead = "qwen_prefill_last_head"
        case qwenGroupExpertQMV = "qwen_group_expert_qmv"
        case qwenGroupExpertStages = "qwen_group_expert_stages"
        case qwenDeviceGreedy = "qwen_device_greedy"
        case turnContext = "turn_context"
        case runtimeCommit = "runtime_commit"
        case runtimeProfile = "runtime_profile"
        case mlxVersion = "mlx_version"
        case runtimeExecutable = "runtime_executable"
        case dflashRequested = "dflash_requested"
        case dflashActive = "dflash_active"
        case dflashProposals = "dflash_proposals"
        case dflashDraftPath = "dflash_draft_path"
        case dflashMode = "dflash_mode"
        case dflashSupported = "dflash_supported"
        case dflashTuneSupported = "dflash_tune_supported", dflashConfigureSupported = "dflash_configure_supported"
        case dflashTuningKey = "dflash_tuning_key", dflashFamily = "dflash_family"
        case mtpSupported = "mtp_supported"
        case mtpAutoDownloadSupported = "mtp_auto_download_supported"
        case mtpConfigureSupported = "mtp_configure_supported"
        case mtpActive = "mtp_active"
        case mtpReason = "mtp_reason"
        case mtpMaxDepth = "mtp_max_depth", mtpTuneSupported = "mtp_tune_supported", mtpTuningKey = "mtp_tuning_key"
        case depth, completed, total, rows
        case bestDepth = "best_depth", tuningKey = "tuning_key"
        case workspaceFeatures = "workspace_features"
        case bridgeProtocol = "bridge_protocol"
        case mcpToolCatalog = "mcp_tool_catalog"
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
    var turnContext: String? = nil
    enum CodingKeys: String, CodingKey {
        case role, content
        case turnContext = "turn_context"
    }
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
    var dflash2: Bool = false
    var dflashDraftPath: String = ""
    var dflashMode: Int = 1
    var mtp: Bool = false
    var mtpHeadPath: String = ""
    var mtpDepth: Int = 1
    var memorySaver: Bool = false
    var allowedTools: [String]? = nil
    var responseSchema: String? = nil

    enum CodingKeys: String, CodingKey {
        case type, messages, temperature
        case requestID = "request_id"
        case conversationID = "conversation_id"
        case maxTokens = "max_tokens"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case mcpEnabled = "mcp_enabled"
        case dflash2
        case dflashDraftPath = "dflash_draft_path"
        case dflashMode = "dflash_mode"
        case mtp
        case mtpHeadPath = "mtp_head_path"
        case mtpDepth = "mtp_depth"
        case memorySaver = "memory_saver"
        case allowedTools = "allowed_tools"
        case responseSchema = "response_schema"
    }
}

final class ChatMessage: ObservableObject, Identifiable {
    enum Role: String {
        case user
        case assistant
    }

    let id: UUID
    let role: Role
    let attachments: [ChatAttachment]
    var sources: [DocumentExcerpt] = [] { didSet { streamRevision &+= 1 } }
    private(set) var content: String
    private(set) var thinking: String
    private(set) var isStreaming: Bool
    private(set) var stats: GenerationStats?
    private(set) var error: String?
    private(set) var toolActivities: [ToolActivity]
    private(set) var cacheContext: String?
    var turnContext: String? { didSet { streamRevision &+= 1 } }
    private(set) var parts: [AssistantPart]
    let revisionChanged = PassthroughSubject<Void, Never>()
    private(set) var streamRevision = 0 { didSet { revisionChanged.send() } }

    init(
        id: UUID = UUID(),
        role: Role,
        content: String,
        attachments: [ChatAttachment] = [],
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
        self.attachments = attachments
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
        for index in toolActivities.indices where toolActivities[index].state == .running {
            toolActivities[index].state = .failed
        }
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

    var promptContent: String {
        role == .user ? ChatAttachment.promptContent(content, attachments: attachments) + DocumentSearch.context(sources) : content
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
            },
            turnContext: turnContext,
            attachments: attachments.isEmpty ? nil : attachments,
            sources: sources.isEmpty ? nil : sources
        )
    }

    convenience init?(snapshot: ChatMessageSnapshot) {
        guard let role = Role(rawValue: snapshot.role) else { return nil }
        self.init(
            id: snapshot.id,
            role: role,
            content: snapshot.content,
            attachments: snapshot.attachments ?? [],
            thinking: snapshot.thinking,
            isStreaming: false,
            stats: snapshot.stats,
            error: snapshot.error ?? (snapshot.wasStreaming ? L("Génération interrompue", "Generation interrupted") : nil),
            toolActivities: snapshot.toolActivities ?? [],
            cacheContext: snapshot.cacheContext,
            parts: snapshot.parts
        )
        endProcessing(interrupted: snapshot.wasStreaming)
        turnContext = snapshot.turnContext
        sources = snapshot.sources ?? []
        for index in toolActivities.indices where toolActivities[index].state == .running {
            toolActivities[index].state = .failed
        }
    }
}

struct ContextUsage: Codable, Sendable {
    let used: Int
    let limit: Int
    let model: String

    func finalized(with stats: GenerationStats) -> ContextUsage? {
        let finalLimit = stats.contextLimit ?? limit
        guard finalLimit > 0 else { return nil }
        let total: Int
        if let exact = stats.contextUsed {
            total = exact
        } else {
            // Older single-round engines reported separate input/output counts.
            guard (stats.toolRounds ?? 0) == 0, stats.promptTokens >= 0, stats.generatedTokens >= 0 else { return nil }
            let sum = stats.promptTokens.addingReportingOverflow(stats.generatedTokens)
            guard !sum.overflow else { return nil }
            total = sum.partialValue
        }
        guard total >= 0 else { return nil }
        return ContextUsage(used: min(total, finalLimit), limit: finalLimit, model: model)
    }
}

struct Conversation: Identifiable {
    let id: UUID
    var title: String
    var messages: [ChatMessage] {
        didSet { messageSnapshots = ConversationMessageSnapshots(messages) }
    }
    let createdAt: Date
    var isPinned: Bool
    var contextUsage: ContextUsage?
    var projectID: UUID?
    var parentConversationID: UUID?
    var agent = false
    var updatedAt: Date?
    private var messageSnapshots: ConversationMessageSnapshots
    var messageChanges: AnyPublisher<Void, Never> { messageSnapshots.changed.eraseToAnyPublisher() }

    init(
        id: UUID = UUID(),
        title: String = L("Nouvelle conversation", "New conversation"),
        messages: [ChatMessage] = [],
        createdAt: Date = Date(),
        contextUsage: ContextUsage? = nil,
        isPinned: Bool = false
    ) {
        self.id = id
        self.title = title
        self.messages = messages
        self.messageSnapshots = ConversationMessageSnapshots(messages)
        self.createdAt = createdAt
        self.contextUsage = contextUsage
        self.isPinned = isPinned
    }

    var snapshot: ConversationSnapshot {
        ConversationSnapshot(
            id: id,
            title: title,
            messages: messageSnapshots.snapshot(),
            createdAt: createdAt,
            contextUsage: contextUsage,
            isPinned: isPinned,
            projectID: projectID,
            parentConversationID: parentConversationID,
            agent: agent ? true : nil,
            updatedAt: updatedAt
        )
    }

    init(snapshot: ConversationSnapshot) {
        self.init(
            id: snapshot.id,
            title: snapshot.title,
            messages: snapshot.messages.compactMap(ChatMessage.init(snapshot:)),
            createdAt: snapshot.createdAt,
            contextUsage: snapshot.contextUsage,
            isPinned: snapshot.isPinned ?? false
        )
        projectID = snapshot.projectID
        parentConversationID = snapshot.parentConversationID
        agent = snapshot.agent ?? false
        updatedAt = snapshot.updatedAt
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

/// Value snapshots are shared until message identity/revision changes. Conversation
/// metadata is still rebuilt, so title/project/pinning edits cannot be stale.
private final class ConversationMessageSnapshots {
    let changed = PassthroughSubject<Void, Never>()
    private let messages: [ChatMessage]
    private var observations: [AnyCancellable] = []
    private var dirty: Set<Int>
    private var streaming: Set<Int>
    private var saved: [ChatMessageSnapshot] = []

    init(_ messages: [ChatMessage]) {
        self.messages = messages
        dirty = Set(messages.indices)
        streaming = Set(messages.indices.filter { messages[$0].isStreaming })
        observations = messages.enumerated().map { index, message in
            message.revisionChanged.sink { [weak self, weak message] in
                guard let self, let message else { return }
                self.dirty.insert(index)
                if message.isStreaming { self.streaming.insert(index) }
                else { self.streaming.remove(index) }
                self.changed.send()
            }
        }
    }

    func snapshot() -> [ChatMessageSnapshot] {
        if saved.count != messages.count {
            saved = messages.map(\.snapshot)
        } else {
            // Only live processing parts need their elapsed clock refreshed.
            for index in dirty.union(streaming) { saved[index] = messages[index].snapshot }
        }
        dirty.removeAll(keepingCapacity: true)
        return saved
    }
}
