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
}

actor ConversationStore {
    let fileURL: URL

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

    static func load(from fileURL: URL = defaultFileURL()) -> WorkspaceSnapshot? {
        guard let data = try? Data(contentsOf: fileURL),
              let snapshot = try? JSONDecoder.mlxl3.decode(WorkspaceSnapshot.self, from: data),
              snapshot.version == WorkspaceSnapshot.currentVersion
        else { return nil }
        return snapshot
    }

    func save(_ snapshot: WorkspaceSnapshot) throws {
        try Self.write(snapshot, to: fileURL)
    }

    static func write(_ snapshot: WorkspaceSnapshot, to fileURL: URL) throws {
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder.mlxl3.encode(snapshot)
        try data.write(to: fileURL, options: [.atomic])
    }
}

private extension JSONEncoder {
    static var mlxl3: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
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
