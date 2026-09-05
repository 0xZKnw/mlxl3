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
    @Published var contextLengthDraft = 0
    @Published private(set) var activeContextLimit: Int?
    @Published private(set) var modelContextLimit: Int?
    @Published private(set) var contextMemory: ContextMemoryProfile?
    @Published private(set) var modelResidentBytes: Double?
    @Published private(set) var language: AppLanguage = .fr

    let updateManager: UpdateManager
    let modelLibrary = ModelLibrary()
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
        self.language = AppLanguage(rawValue: preferences.string(forKey: "studio.language") ?? "fr") ?? .fr
        self.mcpEnabled = preferences.bool(forKey: "studio.mcpEnabled")
        conversationStore = ConversationStore(fileURL: conversationFileURL)
        AppLocalization.set(language)
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

    var savedContextLength: Int {
        guard let name = selectedModelName else { return 0 }
        return (preferences.dictionary(forKey: "studio.contextLengths")?[name] as? Int) ?? 0
    }

    func setLanguage(_ value: AppLanguage) {
        AppLocalization.set(value)
        language = value
        preferences.set(value.rawValue, forKey: "studio.language")
    }

    var contextLabel: String {
        let used = contextUsed.map { $0.formatted() } ?? "—"
        let usage = currentConversation?.contextUsage
        let limit = activeContextLimit ?? (usage?.model == selectedModelName ? usage?.limit : nil)
        return "\(used) / \(limit.map { $0.formatted() } ?? "—")"
    }

    var contextUsed: Int? {
        guard let conversation = currentConversation else { return nil }
        if conversation.messages.isEmpty { return 0 }
        guard let usage = conversation.contextUsage, usage.model == selectedModelName else { return nil }
        return usage.used
    }

    var canSaveContext: Bool {
        !isGenerating && selectedModel != nil && modelContextLimit != nil
            && contextLengthDraft >= 0 && contextLengthDraft <= (modelContextLimit ?? 0)
            && contextLengthDraft != savedContextLength
    }

    var draftContextBytes: Double? {
        guard contextLengthDraft >= 0, let maximum = modelContextLimit,
              contextLengthDraft <= maximum else { return nil }
        return contextMemory?.bytes(tokens: contextLengthDraft == 0 ? maximum : contextLengthDraft)
    }

    func saveContextAndReload() {
        guard canSaveContext, let name = selectedModelName else { return }
        var lengths = preferences.dictionary(forKey: "studio.contextLengths") ?? [:]
        lengths[name] = contextLengthDraft
        preferences.set(lengths, forKey: "studio.contextLengths")
        loadSelectedModel()
    }

    var currentConversation: Conversation? {
        guard let selectedConversationID else { return nil }
        return conversations.first { $0.id == selectedConversationID }
    }

    var canSend: Bool {
        engineState.isReady && !mcpUpdating && !modelInstallState.isWorking && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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

    var engineMemoryFootprintBytes: UInt64? {
        bridge.engineMemoryFootprintBytes()
    }

    var interfaceMemoryFootprintBytes: UInt64? {
        bridge.interfaceMemoryFootprintBytes()
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

    func refreshModels(autoLoad: Bool = true) {
        guard !isPreview else { return }
        MLXL3Bridge.listModels { [weak self] result in
            guard let self else { return }
            switch result {
            case let .success(models):
                self.models = models
                guard !models.isEmpty else {
                    self.selectedModelName = nil
                    self.engineState = .idle
                    return
                }
                if !models.contains(where: { $0.name == self.selectedModelName }) {
                    self.selectedModelName = models[0].name
                }
                if autoLoad { self.loadSelectedModel() }
            case let .failure(error):
                self.engineState = .failed(error.localizedDescription)
            }
        }
    }

    func openModelManager() {
        if !modelInstallState.isWorking { modelInstallState = .idle }
        showModelManager = true
        refreshModels(autoLoad: false)
    }

    func removeLocalModel(_ model: LocalModel, trashFiles: Bool) {
        guard !isGenerating, !modelInstallState.isWorking, modelLibrary.downloading == nil else { return }
        if isPreview {
            models.removeAll { $0.id == model.id }
            return
        }
        modelInstallState = .working(L("Suppression…", "Removing…"))
        Task {
            let source = URL(fileURLWithPath: model.path).standardizedFileURL.resolvingSymlinksInPath()
            var trashed: NSURL?
            do {
                let fresh = try JSONDecoder().decode([LocalModel].self, from: await CLICommand().output(["list", "--json"]))
                guard fresh.contains(where: { $0.name == model.name && $0.path == model.path }) else {
                    throw MLXL3BridgeError.commandFailed(L("Le modèle a changé. Actualise la bibliothèque.", "The model changed. Refresh the library."))
                }
                if trashFiles {
                    let home = FileManager.default.homeDirectoryForCurrentUser
                    let protected = [home.path, "/", "/Applications", "/Users", FileManager.default.currentDirectoryPath]
                        + ["Documents", "Desktop", "Downloads", "Library", "Applications"].map { home.appendingPathComponent($0).path }
                    let shared = fresh.contains { $0.name != model.name && ($0.path == source.path || $0.path.hasPrefix(source.path + "/")) }
                    guard !protected.contains(source.path), !home.path.hasPrefix(source.path + "/"), !shared,
                          !FileManager.default.fileExists(atPath: source.appendingPathComponent(".git").path),
                          FileManager.default.fileExists(atPath: source.appendingPathComponent("config.json").path),
                          FileManager.default.fileExists(atPath: source.appendingPathComponent("quantization_config.json").path) else {
                        throw MLXL3BridgeError.commandFailed(L("Ce dossier ne peut pas être supprimé depuis l’app. Retire seulement l’entrée de la bibliothèque.", "This folder cannot be deleted from the app. Remove only its library entry."))
                    }
                }
                if selectedModelName == model.name { ejectModel() }
                if trashFiles { try FileManager.default.trashItem(at: source, resultingItemURL: &trashed) }
                _ = try await CLICommand().output(["remove", model.name, "--expected-path", model.path])
                modelInstallState = .succeeded(trashFiles
                    ? L("Dossier placé dans la Corbeille. Récupérable depuis Finder.", "Folder moved to Trash. Recoverable in Finder.")
                    : L("Entrée retirée ; fichiers conservés.", "Entry removed; files kept."))
                refreshModels(autoLoad: false)
            } catch {
                if let trashed, !FileManager.default.fileExists(atPath: source.path) {
                    do { try FileManager.default.moveItem(at: trashed as URL, to: source) }
                    catch {
                        modelInstallState = .failed(L("Échec du retrait. Le dossier reste récupérable dans la Corbeille : ", "Removal failed. The folder can still be recovered from Trash: ") + trashed.path!)
                        return
                    }
                }
                modelInstallState = .failed(error.localizedDescription)
            }
        }
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
        panel.title = L("Importer un modèle EXL3", "Import an EXL3 model")
        panel.message = L("Choisis le dossier contenant config.json et les poids EXL3.", "Choose the folder containing config.json and EXL3 weights.")
        panel.prompt = L("Importer", "Import")
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
        modelInstallState = .working(L("Téléchargement de \(repository)…", "Downloading \(repository)…"))
        MLXL3Bridge.downloadModel(
            repo: repository,
            revision: cleanRevision.isEmpty ? nil : cleanRevision,
            name: cleanName.isEmpty ? nil : cleanName
        ) { [weak self] result in
            guard let self else { return }
            switch result {
            case let .success(model):
                self.selectedModelName = model.name
                self.modelInstallState = .succeeded(L("\(model.name) est prêt.", "\(model.name) is ready."))
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
        activeMessage()?.fail(L("Modèle éjecté", "Model unloaded"))
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
        activeMessage()?.fail(L("Génération arrêtée", "Generation stopped"))
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
        contextLengthDraft = savedContextLength
        if isPreview {
            modelContextLimit = 262144
            activeContextLimit = savedContextLength == 0 ? modelContextLimit : savedContextLength
            modelResidentBytes = 12_000_000_000
            contextMemory = ContextMemoryProfile(layers: [.init(bytesPerToken: 32768, maxTokens: nil, step: 256)], fixedBytes: 4_000_000)
            return
        }
        activeContextLimit = nil
        modelContextLimit = nil
        contextMemory = nil
        modelResidentBytes = nil
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
            try bridge.start(model: selectedModelName, contextLength: savedContextLength)
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
            modelInstallState = .failed(L("Le dossier n’a pas de nom utilisable.", "The folder has no usable name."))
            return
        }
        modelInstallState = .working(L("Validation de \(rawName)…", "Validating \(rawName)…"))
        MLXL3Bridge.registerModel(name: name, path: url) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success:
                self.selectedModelName = name
                self.modelInstallState = .succeeded(L("\(name) est prêt.", "\(name) is ready."))
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
            activeContextLimit = event.contextLimit
            modelContextLimit = event.modelContextLimit
            contextMemory = event.contextMemory
            modelResidentBytes = event.residentGB.map { $0 * 1e9 }
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
        case "context_usage":
            guard event.requestID == activeRequestID,
                  let used = event.usedTokens, let limit = event.contextLimit,
                  let name = selectedModelName,
                  let index = conversations.firstIndex(where: { conversation in
                      conversation.messages.contains { $0.id == activeResponseID }
                  }) else { return }
            conversations[index].contextUsage = ContextUsage(used: max(0, used), limit: limit, model: name)
        case "generation_status":
            guard event.requestID == activeRequestID,
                  let message = activeMessage() else { return }
            message.processing(event.text ?? L("Préparation de la réponse", "Preparing response"))
            schedulePersistence()
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
            if event.contextFull == true {
                message.fail(L("Limite de contexte atteinte. Augmente la limite dans les paramètres du modèle ou ouvre une nouvelle conversation.", "Context limit reached. Increase the limit in model settings or start a new conversation."))
            }
            activeRequestID = nil
            activeResponseID = nil
            schedulePersistence()
            restoreReadyState()
        case "cancelled":
            guard event.requestID == activeRequestID else { return }
            activeMessage()?.fail(L("Génération arrêtée", "Generation stopped"))
            activeRequestID = nil
            activeResponseID = nil
            schedulePersistence()
            restoreReadyState()
        case "error":
            guard event.requestID == nil || event.requestID == activeRequestID else { return }
            failActiveTurn(event.message ?? L("Erreur inconnue du moteur", "Unknown engine error"))
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
        let detail = message.hasPrefix("Context full (")
            ? L("Contexte plein. Augmente la limite dans les paramètres du modèle ou ouvre une nouvelle conversation.", message)
            : message
        activeMessage()?.fail(detail)
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
