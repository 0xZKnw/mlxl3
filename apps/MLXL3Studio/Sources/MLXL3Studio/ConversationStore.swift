import Foundation

struct WorkspaceSnapshot: Codable, Sendable {
    static let currentVersion = 1

    let version: Int
    let selectedConversationID: UUID?
    let selectedModelName: String?
    let conversations: [ConversationSnapshot]
    let temperature: Double
    let topK: Int
    let repetitionPenalty: Double
    let systemPrompt: String

    init(
        selectedConversationID: UUID?,
        selectedModelName: String?,
        conversations: [ConversationSnapshot],
        temperature: Double,
        topK: Int,
        repetitionPenalty: Double,
        systemPrompt: String
    ) {
        version = Self.currentVersion
        self.selectedConversationID = selectedConversationID
        self.selectedModelName = selectedModelName
        self.conversations = conversations
        self.temperature = temperature
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.systemPrompt = systemPrompt
    }
}

struct ConversationSnapshot: Codable, Sendable {
    let id: UUID
    let title: String
    let messages: [ChatMessageSnapshot]
    let createdAt: Date
    var contextUsage: ContextUsage? = nil
}

struct ChatMessageSnapshot: Codable, Sendable {
    let id: UUID
    let role: String
    let content: String
    let thinking: String
    let wasStreaming: Bool
    let stats: GenerationStats?
    let error: String?
    let toolActivities: [ToolActivity]?
    let cacheContext: String?
    var parts: [AssistantPart]? = nil
    var turnContext: String? = nil
}

final class ConversationStore: @unchecked Sendable {
    let fileURL: URL
    private let queue = DispatchQueue(label: "io.mlxl3.conversation-writer")
    private var latestRevision = -1

    init(fileURL: URL = ConversationStore.defaultFileURL()) {
        self.fileURL = fileURL
    }

    static func defaultFileURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["MLXL3_CONVERSATIONS_PATH"],
           !override.isEmpty {
            return URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
        }
        let root = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
        return root
            .appending(path: "io.mlxl3.desktop", directoryHint: .isDirectory)
            .appending(path: "conversations.json")
    }

    static func load(from fileURL: URL = defaultFileURL()) throws -> WorkspaceSnapshot? {
        guard FileManager.default.fileExists(atPath: fileURL.path) else { return nil }
        let data = try Data(contentsOf: fileURL)
        let snapshot = try JSONDecoder.mlxl3.decode(WorkspaceSnapshot.self, from: data)
        guard snapshot.version == WorkspaceSnapshot.currentVersion else {
            throw CocoaError(.fileReadCorruptFile)
        }
        return snapshot
    }

    func save(_ snapshot: WorkspaceSnapshot, revision: Int) throws {
        try queue.sync {
            guard revision > latestRevision else { return }
            try Self.write(snapshot, to: fileURL)
            latestRevision = revision
        }
    }

    static func write(_ snapshot: WorkspaceSnapshot, to fileURL: URL) throws {
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder.mlxl3.encode(snapshot)
        if FileManager.default.fileExists(atPath: fileURL.path) {
            // Never overwrite unreadable history, including a future schema.
            _ = try load(from: fileURL)
            try Data(contentsOf: fileURL).write(to: fileURL.appendingPathExtension("backup"), options: .atomic)
        }
        try data.write(to: fileURL, options: [.atomic])
    }
}

private extension JSONEncoder {
    static var mlxl3: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        return encoder
    }
}

private extension JSONDecoder {
    static var mlxl3: JSONDecoder {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }
}
