import Foundation

struct WorkspaceSnapshot: Codable, Sendable {
    static let currentVersion = 2

    let version: Int
    let selectedConversationID: UUID?
    let selectedModelName: String?
    let conversations: [ConversationSnapshot]
    let temperature: Double
    let topK: Int
    let repetitionPenalty: Double
    let systemPrompt: String
    var projects: [WorkspaceProject]? = nil
    var profiles: [GenerationProfile]? = nil
    var documents: [LibraryDocument]? = nil
    var responseSchema: String? = nil

    init(
        selectedConversationID: UUID?,
        selectedModelName: String?,
        conversations: [ConversationSnapshot],
        temperature: Double,
        topK: Int,
        repetitionPenalty: Double,
        systemPrompt: String,
        projects: [WorkspaceProject] = [],
        profiles: [GenerationProfile] = [],
        documents: [LibraryDocument] = [],
        responseSchema: String = ""
    ) {
        version = Self.currentVersion
        self.selectedConversationID = selectedConversationID
        self.selectedModelName = selectedModelName
        self.conversations = conversations
        self.temperature = temperature
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.systemPrompt = systemPrompt
        self.projects = projects
        self.profiles = profiles
        self.documents = documents
        self.responseSchema = responseSchema
    }
}

struct ConversationSnapshot: Codable, Sendable {
    let id: UUID
    let title: String
    let messages: [ChatMessageSnapshot]
    let createdAt: Date
    var contextUsage: ContextUsage? = nil
    var projectID: UUID? = nil
    var parentConversationID: UUID? = nil
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
    var attachments: [ChatAttachment]? = nil
    var sources: [DocumentExcerpt]? = nil
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
        guard (1...WorkspaceSnapshot.currentVersion).contains(snapshot.version) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        try validate(snapshot)
        return snapshot
    }

    static func validate(_ snapshot: WorkspaceSnapshot) throws {
        let projects = snapshot.projects ?? []
        let profiles = snapshot.profiles ?? []
        let documents = snapshot.documents ?? []
        let projectIDs = Set(projects.map(\.id))
        guard projects.count <= 256, projectIDs.count == projects.count,
            projects.allSatisfy({ !$0.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && $0.name.count <= 100 }),
            snapshot.conversations.allSatisfy({ $0.projectID == nil || projectIDs.contains($0.projectID!) }),
            profiles.count <= 128, Set(profiles.map(\.id)).count == profiles.count,
            documents.count <= DocumentSearch.maximumDocuments, Set(documents.map(\.id)).count == documents.count,
            documents.allSatisfy({ projectIDs.contains($0.projectID) && $0.text.utf8.count <= ChatAttachmentImporter.maxTextBytes }),
            documents.reduce(0, { $0 + $1.text.utf8.count }) <= DocumentSearch.maximumLibraryBytes else {
            throw CocoaError(.fileReadCorruptFile)
        }
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
        try validate(snapshot)
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
