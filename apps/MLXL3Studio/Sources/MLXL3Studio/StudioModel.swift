import AppKit
import Darwin
import SwiftUI

@MainActor
final class StudioModel: ObservableObject {
    @Published var models: [LocalModel] = []
    @Published var selectedModelName: String?
    @Published var conversations: [Conversation] = [Conversation()]
    @Published var selectedConversationID: UUID?
    @Published var draft = ""
    @Published var engineState: EngineState = .idle
    @Published var showInspector = false
    @Published var showModelManager = false
    @Published var showAppSettings = false
    @Published private(set) var modelInstallState: ModelInstallState = .idle
    @Published var temperature = 0.2
    @Published var topK = 80
    @Published var repetitionPenalty = 1.05
    @Published var systemPrompt = ""
    @Published private(set) var mcpServerCount = 0
    @Published private(set) var mcpToolCount = 0
    @Published private(set) var mcpErrors: [String: String] = [:]
    @Published private(set) var mcpEnabled = false
    @Published private(set) var mcpUpdating = false

    let updateManager: UpdateManager
    let isPreview: Bool
    private let bridge = MLXL3Bridge()
    private let conversationStore: ConversationStore
    private let conversationFileURL: URL
    private var persistenceTask: Task<Void, Never>?
    private var didStart = false
    private var activeRequestID: String?
    private var activeResponseID: UUID?
    private var readyInfo: (model: String, modules: Int, residentGB: Double)?
    private let preferences: UserDefaults

    init(
        conversationFileURL: URL = ConversationStore.defaultFileURL(),
        updateManager: UpdateManager = UpdateManager(),
        isPreview: Bool = false,
        preferences: UserDefaults = .standard
    ) {
        self.conversationFileURL = conversationFileURL
        self.updateManager = updateManager
        self.isPreview = isPreview
        self.preferences = preferences
        self.mcpEnabled = preferences.bool(forKey: "studio.mcpEnabled")
        conversationStore = ConversationStore(fileURL: conversationFileURL)
        if let snapshot = ConversationStore.load(from: conversationFileURL),
           !snapshot.conversations.isEmpty {
            conversations = snapshot.conversations.map(Conversation.init(snapshot:))
            selectedConversationID = conversations.contains {
                $0.id == snapshot.selectedConversationID
            } ? snapshot.selectedConversationID : conversations.first?.id
            selectedModelName = snapshot.selectedModelName
            temperature = snapshot.temperature
            topK = snapshot.topK
            repetitionPenalty = snapshot.repetitionPenalty
            systemPrompt = snapshot.systemPrompt
        } else {
            selectedConversationID = conversations.first?.id
        }
        bridge.onEvent = { [weak self] event in self?.handle(event) }
        bridge.onExit = { [weak self] message in
            guard let self, let message, !message.isEmpty else { return }
            if self.activeRequestID != nil {
                self.failActiveTurn(message)
            } else {
                self.engineState = .failed(message)
            }
        }
    }

    var selectedModel: LocalModel? {
        models.first { $0.name == selectedModelName }
    }

    var currentConversation: Conversation? {
        guard let selectedConversationID else { return nil }
        return conversations.first { $0.id == selectedConversationID }
    }

    var canSend: Bool {
        engineState.isReady && !mcpUpdating && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var isGenerating: Bool {
        if case .generating = engineState { return true }
        return false
    }

    var canEject: Bool {
        switch engineState {
        case .loading, .ready, .generating: true
        case .idle, .failed: false
        }
    }

    var loadedModel: LocalModel? {
        canEject ? selectedModel : nil
    }

    var latestGenerationStats: GenerationStats? {
        currentConversation?.messages.reversed().compactMap(\.stats).first
    }

    var residentMemoryBytes: UInt64 {
        bridge.residentMemoryBytes()
    }

    var engineResidentMemoryBytes: UInt64 {
        bridge.engineResidentMemoryBytes()
    }

    var interfaceResidentMemoryBytes: UInt64 {
        bridge.interfaceResidentMemoryBytes()
    }

    var mcpConfigurationURL: URL {
        let environment = ProcessInfo.processInfo.environment
        let root: URL
        if let override = environment["MLXL3_HOME"], !override.isEmpty {
            root = URL(fileURLWithPath: (override as NSString).expandingTildeInPath)
        } else {
            root = FileManager.default.homeDirectoryForCurrentUser
                .appending(path: ".config/mlxl3", directoryHint: .isDirectory)
        }
        return root.appending(path: "mcp.json")
    }

    func start() {
        guard !isPreview else { return }
        guard !didStart else { return }
        didStart = true
        updateManager.startAutomaticCheck()
        refreshModels()
    }

    func refreshModels() {
        guard !isPreview else { return }
        MLXL3Bridge.listModels { [weak self] result in
            guard let self else { return }
            switch result {
            case let .success(models):
                self.models = models
                guard !models.isEmpty else {
                    self.selectedModelName = nil
                    self.engineState = .failed("Aucun modèle enregistré dans MLXL3.")
                    return
                }
                if !models.contains(where: { $0.name == self.selectedModelName }) {
                    self.selectedModelName = models[0].name
                }
                self.loadSelectedModel()
            case let .failure(error):
                self.engineState = .failed(error.localizedDescription)
            }
        }
    }

    func openModelManager() {
        modelInstallState = .idle
        showModelManager = true
    }

    func openAppSettings() {
        showAppSettings = true
    }

    func installUpdateAndRestart() {
        guard !isPreview else { return }
        guard !isGenerating else { return }
        persistNow()
        guard updateManager.beginInstallation() else { return }
        bridge.stop()
        persistNow()
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            Darwin.exit(EXIT_SUCCESS)
        }
        DispatchQueue.main.async {
            NSApplication.shared.terminate(nil)
        }
    }

    func importModelFolder() {
        guard !isPreview else { return }
        guard !modelInstallState.isWorking else { return }
        let panel = NSOpenPanel()
        panel.title = "Importer un modèle EXL3"
        panel.message = "Choisis le dossier contenant config.json et les poids EXL3."
        panel.prompt = "Importer"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.begin { [weak self] response in
            guard response == .OK, let url = panel.url else { return }
            Task { @MainActor [weak self] in
                self?.registerModelFolder(url)
            }
        }
    }

    func downloadModel(repo: String, revision: String, name: String) {
        guard !isPreview else { return }
        let repository = repo.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !repository.isEmpty, !modelInstallState.isWorking else { return }
        let cleanRevision = revision.trimmingCharacters(in: .whitespacesAndNewlines)
        let cleanName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        modelInstallState = .working("Téléchargement de \(repository)…")
        MLXL3Bridge.downloadModel(
            repo: repository,
            revision: cleanRevision.isEmpty ? nil : cleanRevision,
            name: cleanName.isEmpty ? nil : cleanName
        ) { [weak self] result in
            guard let self else { return }
            switch result {
            case let .success(model):
                self.selectedModelName = model.name
                self.modelInstallState = .succeeded("\(model.name) est prêt.")
                self.refreshModels()
            case let .failure(error):
                self.modelInstallState = .failed(error.localizedDescription)
            }
        }
    }

    func selectModel(_ name: String) {
        if name == selectedModelName, bridge.isRunning { return }
        selectedModelName = name
        schedulePersistence()
        loadSelectedModel()
    }

    func ejectModel() {
        guard canEject else { return }
        activeMessage()?.fail("Modèle éjecté")
        activeRequestID = nil
        activeResponseID = nil
        readyInfo = nil
        bridge.stop()
        engineState = .idle
        mcpUpdating = false
        mcpServerCount = 0
        mcpToolCount = 0
        mcpErrors = [:]
        schedulePersistence()
    }

    func newConversation() {
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        selectedConversationID = conversation.id
        draft = ""
        schedulePersistence()
    }

    func deleteConversation(_ id: UUID) {
        guard conversations.count > 1 else {
            if let index = conversations.firstIndex(where: { $0.id == id }) {
                conversations[index] = Conversation(id: id)
            }
            schedulePersistence()
            return
        }
        conversations.removeAll { $0.id == id }
        if selectedConversationID == id {
            selectedConversationID = conversations.first?.id
        }
        schedulePersistence()
    }

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, engineState.isReady, !mcpUpdating, let conversationIndex else { return }

        draft = ""
        if conversations[conversationIndex].messages.isEmpty {
            conversations[conversationIndex].title = title(for: text)
        }
        conversations[conversationIndex].messages.append(
            ChatMessage(role: .user, content: text)
        )
        let assistant = ChatMessage(role: .assistant, content: "", isStreaming: true)
        conversations[conversationIndex].messages.append(assistant)
        schedulePersistence()

        let requestID = UUID().uuidString
        activeRequestID = requestID
        activeResponseID = assistant.id
        engineState = .generating

        var messages: [PromptMessage] = []
        let system = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !system.isEmpty {
            messages.append(PromptMessage(role: "system", content: system))
        }
        messages += conversations[conversationIndex].messages.compactMap { message in
            guard !message.isStreaming else { return nil }
            let context = message.role == .assistant
                ? (message.cacheContext ?? message.content)
                : message.content
            return PromptMessage(role: message.role.rawValue, content: context)
        }

        do {
            try bridge.generate(
                GenerationRequest(
                    requestID: requestID,
                    conversationID: conversations[conversationIndex].id.uuidString,
                    messages: messages,
                    maxTokens: -1,
                    temperature: temperature,
                    topK: topK,
                    repetitionPenalty: repetitionPenalty,
                    mcpEnabled: mcpEnabled
                )
            )
        } catch {
            failActiveTurn(error.localizedDescription)
        }
    }

    func stopGeneration() {
        guard isGenerating else { return }
        if bridge.cancelGeneration() {
            return
        }
        activeMessage()?.fail("Génération arrêtée")
        activeRequestID = nil
        activeResponseID = nil
        bridge.stop()
        schedulePersistence()
        loadSelectedModel()
    }

    func reloadMCPServers() {
        guard !isGenerating, !mcpUpdating else { return }
        updateMCPConnection()
    }

    func setMCPEnabled(_ enabled: Bool) {
        guard !isGenerating, !mcpUpdating, enabled != mcpEnabled else { return }
        mcpEnabled = enabled
        preferences.set(enabled, forKey: "studio.mcpEnabled")
        mcpErrors = [:]
        if !enabled { mcpServerCount = 0; mcpToolCount = 0 }
        if engineState.isReady { updateMCPConnection() }
    }

    private func updateMCPConnection() {
        guard !isPreview, bridge.isRunning else { return }
        mcpUpdating = true
        do {
            try bridge.setMCPEnabled(mcpEnabled)
        } catch {
            mcpUpdating = false
            mcpErrors = ["connexion": error.localizedDescription]
        }
    }

    func openMCPConfiguration() {
        guard !isPreview else { return }
        let url = mcpConfigurationURL
        if !FileManager.default.fileExists(atPath: url.path) {
            try? FileManager.default.createDirectory(
                at: url.deletingLastPathComponent(),
                withIntermediateDirectories: true
            )
            try? "{\n  \"version\": 1,\n  \"mcpServers\": {}\n}\n".write(
                to: url,
                atomically: true,
                encoding: .utf8
            )
        }
        NSWorkspace.shared.open(url)
    }

    func settingsDidChange() {
        schedulePersistence()
    }

    func selectConversation(_ id: UUID) {
        guard conversations.contains(where: { $0.id == id }) else { return }
        selectedConversationID = id
        schedulePersistence()
    }

    func persistNow() {
        persistenceTask?.cancel()
        persistenceTask = nil
        try? ConversationStore.write(workspaceSnapshot, to: conversationFileURL)
    }

    private var conversationIndex: Int? {
        guard let selectedConversationID else { return nil }
        return conversations.firstIndex { $0.id == selectedConversationID }
    }

    private func loadSelectedModel() {
        guard !isPreview else { return }
        guard let selectedModelName, selectedModel != nil else { return }
        engineState = .loading(selectedModelName)
        mcpUpdating = false
        readyInfo = nil
        mcpServerCount = 0
        mcpToolCount = 0
        mcpErrors = [:]
        activeRequestID = nil
        activeResponseID = nil
        do {
            try bridge.start(model: selectedModelName)
        } catch {
            engineState = .failed(error.localizedDescription)
        }
    }

    private func registerModelFolder(_ url: URL) {
        let rawName = url.lastPathComponent
        let name = rawName.replacingOccurrences(
            of: "[^A-Za-z0-9._-]+",
            with: "-",
            options: .regularExpression
        ).trimmingCharacters(in: CharacterSet(charactersIn: ".-"))
        guard !name.isEmpty else {
            modelInstallState = .failed("Le dossier n’a pas de nom utilisable.")
            return
        }
        modelInstallState = .working("Validation de \(rawName)…")
        MLXL3Bridge.registerModel(name: name, path: url) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.selectedModelName = name
                self.modelInstallState = .succeeded("\(name) est prêt.")
                self.refreshModels()
            case let .failure(error):
                self.modelInstallState = .failed(error.localizedDescription)
            }
        }
    }

    private func handle(_ event: BridgeEvent) {
        switch event.type {
        case "loading":
            engineState = .loading(event.model ?? selectedModelName ?? "modèle")
        case "ready":
            let info = (
                model: event.model ?? selectedModelName ?? "modèle",
                modules: event.modules ?? 0,
                residentGB: event.residentGB ?? 0
            )
            readyInfo = info
            mcpServerCount = event.mcpServers ?? 0
            mcpToolCount = event.mcpTools ?? 0
            mcpErrors = event.mcpErrors ?? [:]
            engineState = .ready(
                model: info.model,
                modules: info.modules,
                residentGB: info.residentGB
            )
            if mcpEnabled { updateMCPConnection() }
        case "mcp_status":
            mcpUpdating = false
            mcpServerCount = event.mcpServers ?? 0
            mcpToolCount = event.mcpTools ?? 0
            mcpErrors = event.mcpErrors ?? [:]
        case "delta":
            guard event.requestID == activeRequestID,
                  let text = event.text,
                  let message = activeMessage()
            else { return }
            message.append(text, phase: event.phase)
            schedulePersistence()
        case "tool_start":
            guard event.requestID == activeRequestID,
                  let callID = event.toolCallID,
                  let toolName = event.toolName,
                  let message = activeMessage()
            else { return }
            message.startTool(
                id: callID,
                serverName: event.serverName,
                toolName: toolName
            )
            schedulePersistence()
        case "tool_result":
            guard event.requestID == activeRequestID,
                  let callID = event.toolCallID,
                  let message = activeMessage()
            else { return }
            message.finishTool(
                id: callID,
                result: event.text,
                isError: event.isError ?? false
            )
            schedulePersistence()
        case "complete":
            guard event.requestID == activeRequestID,
                  let message = activeMessage()
            else { return }
            message.finish(
                stats: event.stats,
                fallbackAnswer: event.assistantContext,
                cacheContext: event.cacheContext
            )
            activeRequestID = nil
            activeResponseID = nil
            schedulePersistence()
            restoreReadyState()
        case "cancelled":
            guard event.requestID == activeRequestID else { return }
            activeMessage()?.fail("Génération arrêtée")
            activeRequestID = nil
            activeResponseID = nil
            schedulePersistence()
            restoreReadyState()
        case "error":
            guard event.requestID == nil || event.requestID == activeRequestID else { return }
            failActiveTurn(event.message ?? "Erreur inconnue du moteur")
        default:
            break
        }
    }

    private func restoreReadyState() {
        if let readyInfo {
            engineState = .ready(
                model: readyInfo.model,
                modules: readyInfo.modules,
                residentGB: readyInfo.residentGB
            )
        } else {
            engineState = .idle
        }
    }

    private func failActiveTurn(_ message: String) {
        mcpUpdating = false
        activeMessage()?.fail(message)
        activeRequestID = nil
        activeResponseID = nil
        if readyInfo == nil {
            engineState = .failed(message)
        } else {
            restoreReadyState()
        }
        schedulePersistence()
    }

    private func activeMessage() -> ChatMessage? {
        guard let activeResponseID else { return nil }
        for conversation in conversations.indices {
            if let message = conversations[conversation].messages.first(
                where: { $0.id == activeResponseID }
            ) {
                return message
            }
        }
        return nil
    }

    private func title(for prompt: String) -> String {
        let words = prompt.split(whereSeparator: \.isWhitespace)
        let title = words.prefix(7).joined(separator: " ")
        return title.count > 46 ? String(title.prefix(46)) + "…" : title
    }

    private var workspaceSnapshot: WorkspaceSnapshot {
        WorkspaceSnapshot(
            selectedConversationID: selectedConversationID,
            selectedModelName: selectedModelName,
            conversations: conversations.map(\.snapshot),
            temperature: temperature,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            systemPrompt: systemPrompt
        )
    }

    private func schedulePersistence() {
        guard persistenceTask == nil else { return }
        persistenceTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(1))
            guard !Task.isCancelled, let self else { return }
            persistenceTask = nil
            let snapshot = workspaceSnapshot
            try? await conversationStore.save(snapshot)
        }
    }
}
