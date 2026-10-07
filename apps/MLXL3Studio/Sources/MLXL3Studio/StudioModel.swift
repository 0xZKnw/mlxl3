import AppKit
import Combine
import Darwin
import SwiftUI

private struct DFlashInstallEvent: Decodable {
    let type: String
    let completed: Double?
    let total: Double?
    let path: String?
}

@MainActor
final class StudioModel: ObservableObject {
    @Published var models: [LocalModel] = []
    @Published var selectedModelName: String?
    @Published var conversations: [Conversation] = [Conversation()]
    @Published var selectedConversationID: UUID?
    @Published var draft = ""
    @Published private(set) var attachmentDrafts: [UUID: [ChatAttachment]] = [:]
    @Published private(set) var attachmentImportErrors: [UUID: String] = [:]
    @Published private(set) var isImportingFiles = false
    private var attachmentImportID: UUID?
    private var attachmentImportConversationID: UUID?
    @Published var storageError: String?
    private var persistenceBlocked = false
    private var persistenceRevision = 0
    private var drafts: [UUID: String] = [:]
    private var deletedConversation: Conversation?
    @Published var engineState: EngineState = .idle {
        didSet { updateModelIdleUnload() }
    }
    @Published var showInspector = false
    @Published var showModelManager = false
    @Published var showAppSettings = false
    @Published private(set) var modelInstallState: ModelInstallState = .idle
    @Published var temperature = 0.2
    @Published var topK = 80
    @Published var repetitionPenalty = 1.05
    @Published private(set) var dflash2Enabled = false
    @Published private(set) var dflashDraftPath = ""
    @Published private(set) var dflashDownloading = false {
        didSet { updateModelIdleUnload() }
    }
    @Published private(set) var dflashDownloadCompleted = 0.0
    @Published private(set) var dflashDownloadTotal = 0.0
    @Published private(set) var dflashDownloadError: String?
    @Published private(set) var dflashActive: Bool?
    @Published private(set) var dflashSupported: Bool?
    @Published private(set) var mtpEnabled = false
    @Published private(set) var mtpHeadPath = ""
    @Published private(set) var mtpDownloading = false {
        didSet { updateModelIdleUnload() }
    }
    @Published private(set) var mtpDownloadCompleted = 0.0
    @Published private(set) var mtpDownloadTotal = 0.0
    @Published private(set) var mtpError: String?
    @Published private(set) var mtpSupported: Bool?
    @Published private(set) var mtpAutoDownloadSupported: Bool?
    @Published private(set) var mtpActive: Bool?
    @Published private(set) var mtpDepth = 1
    @Published private(set) var mtpMaxDepth = 1
    @Published private(set) var mtpTuneSupported = false
    @Published private(set) var isTuningMTP = false {
        didSet { updateModelIdleUnload() }
    }
    @Published private(set) var mtpTuneProgress = 0.0
    @Published private(set) var mtpTuneStatus = ""
    @Published private(set) var mtpTuneRows: [MTPTuningRow] = []
    @Published private(set) var mtpTunedAt: Date?
    private var mtpTuningKey: String?
    private var tuneRequestID: String?
    private var tuneConfiguration: MTPConfigurationKey?
    private var tuneCancellationRequested = false
    @Published private(set) var runtimeIdentity: String?
    @Published var systemPrompt = ""
    @Published private(set) var mcpServerCount = 0
    @Published private(set) var mcpToolCount = 0
    @Published private(set) var mcpErrors: [String: String] = [:]
    @Published private(set) var mcpEnabled = false
    @Published private(set) var mcpUpdating = false {
        didSet { updateModelIdleUnload() }
    }
    @Published var contextLengthDraft = 0
    @Published private(set) var activeContextLimit: Int?
    @Published private(set) var modelContextLimit: Int?
    @Published private(set) var contextMemory: ContextMemoryProfile?
    @Published private(set) var modelResidentBytes: Double?
    @Published private(set) var language: AppLanguage = .fr
    @Published private(set) var modelIdleUnloadDelay = ModelIdleUnloadDelay.defaultValue

    let updateManager: UpdateManager
    let modelLibrary = ModelLibrary()
    let isPreview: Bool
    private let bridge = MLXL3Bridge()
    private let conversationStore: ConversationStore
    private let conversationFileURL: URL
    private var persistenceTask: Task<Void, Never>?
    private var dflashDownloadTask: Task<Void, Never>?
    private var mtpDownloadTask: Task<Void, Never>?
    private var mtpOperationID = UUID()
    private var mtpConfigureSupported = false
    private var mtpLegacyModelName: String?
    private var didStart = false
    private var activeRequestID: String?
    private var activeResponseID: UUID?
    private var readyInfo: (model: String, modules: Int, residentGB: Double)?
    private let preferences: UserDefaults
    private var updateObservation: AnyCancellable?
    private let idleUnloadNow: () -> ContinuousClock.Instant
    private var mtpConfigurationPending = false {
        didSet { updateModelIdleUnload() }
    }
    private lazy var modelIdleUnloader = ModelIdleUnloader(now: idleUnloadNow) { [weak self] in
        guard let self, self.canAutomaticallyUnloadModel else { return }
        self.ejectModel()
    }

    init(
        conversationFileURL: URL = ConversationStore.defaultFileURL(),
        updateManager: UpdateManager = UpdateManager(),
        isPreview: Bool = false,
        preferences: UserDefaults = .standard,
        idleUnloadNow: @escaping () -> ContinuousClock.Instant = { .now }
    ) {
        self.conversationFileURL = conversationFileURL
        self.updateManager = updateManager
        self.isPreview = isPreview
        self.preferences = preferences
        self.idleUnloadNow = idleUnloadNow
        self.modelIdleUnloadDelay = ModelIdleUnloadDelay.load(from: preferences)
        self.language = AppLanguage(rawValue: preferences.string(forKey: "studio.language") ?? "fr") ?? .fr
        self.mcpEnabled = preferences.bool(forKey: "studio.mcpEnabled")
        // DFlash remains available to the CLI; Desktop v1.2 moves to native MTP.
        self.dflash2Enabled = false
        preferences.set(false, forKey: "studio.dflash2Enabled")
        self.dflashDraftPath = preferences.string(forKey: "studio.dflashDraftPath") ?? ""
        self.mtpHeadPath = preferences.string(forKey: "studio.mtpHeadPath") ?? ""
        self.mtpEnabled = preferences.bool(forKey: "studio.mtpEnabled")
        conversationStore = ConversationStore(fileURL: conversationFileURL)
        AppLocalization.set(language)
        do {
        if let snapshot = try ConversationStore.load(from: conversationFileURL),
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
        } catch {
            persistenceBlocked = true
            storageError = L("Historique illisible : aucune donnée ne sera écrasée. ", "History could not be read: no data will be overwritten. ") + error.localizedDescription
            selectedConversationID = conversations.first?.id
        }
        if mtpEnabled {
            temperature = 0; topK = 1; repetitionPenalty = 1
        }
        if dflash2Enabled {
            if dflashDraftPath.isEmpty {
                dflash2Enabled = false
                preferences.set(false, forKey: "studio.dflash2Enabled")
            } else {
                temperature = 0
                topK = 1
                repetitionPenalty = 1
            }
        }
        mtpLegacyModelName = selectedModelName
        bridge.onEvent = { [weak self] event in self?.handle(event) }
        bridge.onRuntimeFallback = { [weak self] in
            guard let self else { return }
            self.readyInfo = nil
            self.cancelMTPPreparation()
            self.finishMTPTuning(error: nil)
            self.loadSelectedModel()
        }
        bridge.onExit = { [weak self] message in
            guard let self, let message, !message.isEmpty else { return }
            self.readyInfo = nil
            self.cancelMTPPreparation()
            self.finishMTPTuning(error: message)
            self.mtpTuneSupported = false; self.mtpTuningKey = nil
            self.dflashActive = nil
            self.dflashSupported = nil
            self.mtpActive = nil; self.mtpSupported = nil; self.mtpAutoDownloadSupported = nil
            self.runtimeIdentity = nil
            if self.activeRequestID != nil {
                self.failActiveTurn(message)
            } else {
                self.engineState = .failed(message)
            }
        }
        updateObservation = updateManager.objectWillChange.sink { [weak self] in
            self?.objectWillChange.send()
        }
        if !isPreview { prepareMCPConfiguration() }
    }

    var selectedModel: LocalModel? {
        models.first { $0.name == selectedModelName }
    }

    var dflash2Available: Bool {
        if let dflashSupported { return dflashSupported }
        guard let model = selectedModel, model.modelType == "qwen3_5_moe" else { return false }
        return (model.name + model.path).lowercased().contains("qwen3.6-35b-a3b")
    }

    var mtpAvailable: Bool {
        mtpSupported ?? ["qwen3_5", "qwen3_5_moe"].contains(selectedModel?.modelType ?? "")
    }
    var mtpAutomaticDownloadAvailable: Bool {
        mtpAutoDownloadSupported ?? false
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

    func setModelIdleUnloadDelay(_ delay: ModelIdleUnloadDelay) {
        modelIdleUnloadDelay = delay
        preferences.set(delay.rawValue, forKey: ModelIdleUnloadDelay.preferenceKey)
        updateModelIdleUnload()
    }

    private var canAutomaticallyUnloadModel: Bool {
        !isPreview && engineState.isReady && readyInfo != nil && bridge.isRunning
            && !isGenerating && !mcpUpdating && !mtpDownloading
            && !dflashDownloading && !mtpConfigurationPending
    }

    private func updateModelIdleUnload() {
        modelIdleUnloader.update(isIdle: canAutomaticallyUnloadModel, after: modelIdleUnloadDelay.duration)
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
        if let stats = conversation.messages.last?.stats,
           let finalized = usage.finalized(with: stats) { return finalized.used }
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
        engineState.isReady && !isTuningMTP && !mcpUpdating && !dflashDownloading && !mtpDownloading && !modelInstallState.isWorking && !updateManager.isInstalling
            && !isImportingFiles && (!draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !pendingAttachments.isEmpty)
    }

    var pendingAttachments: [ChatAttachment] {
        selectedConversationID.flatMap { attachmentDrafts[$0] } ?? []
    }

    var attachmentImportError: String? {
        selectedConversationID.flatMap { attachmentImportErrors[$0] }
    }

    var canImportFiles: Bool {
        !isGenerating && !isImportingFiles && currentConversation != nil
    }

    func chooseChatFiles() {
        guard canImportFiles else { return }
        let panel = NSOpenPanel()
        panel.title = L("Joindre des fichiers", "Attach files")
        panel.message = L("PDF et texte · 8 fichiers maximum · 20 Mio par fichier", "PDF and text · up to 8 files · 20 MiB per file")
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = true
        panel.allowedContentTypes = ChatAttachmentImporter.contentTypes
        // Extensionless text files are valid too; unsupported formats get an explicit error.
        panel.allowsOtherFileTypes = true
        guard panel.runModal() == .OK else { return }
        let urls = panel.urls
        Task { await importChatFiles(urls) }
    }

    func importChatFiles(_ urls: [URL]) async {
        guard canImportFiles, !urls.isEmpty, let id = selectedConversationID else { return }
        let token = UUID()
        attachmentImportID = token
        attachmentImportConversationID = id
        isImportingFiles = true
        attachmentImportErrors[id] = nil
        let existing = pendingAttachments
        let task = Task.detached(priority: .userInitiated) {
            ChatAttachmentImporter.load(urls, existing: existing)
        }
        let result = await withTaskCancellationHandler {
            await task.value
        } onCancel: {
            task.cancel()
        }
        guard attachmentImportID == token else { return }
        attachmentImportID = nil
        attachmentImportConversationID = nil
        isImportingFiles = false
        guard !Task.isCancelled, conversations.contains(where: { $0.id == id }) else { return }
        attachmentDrafts[id] = result.attachments
        attachmentImportErrors[id] = result.errors.isEmpty ? nil : result.errors.joined(separator: "\n")
    }

    func removeChatAttachment(_ attachmentID: UUID) {
        guard !isGenerating, !isImportingFiles, let id = selectedConversationID else { return }
        attachmentDrafts[id]?.removeAll { $0.id == attachmentID }
    }

    func dismissAttachmentImportError() {
        guard let id = selectedConversationID else { return }
        attachmentImportErrors[id] = nil
    }

    private func clearAttachmentDraft(_ id: UUID) {
        attachmentDrafts[id] = nil
        attachmentImportErrors[id] = nil
        if attachmentImportConversationID == id {
            attachmentImportID = nil
            attachmentImportConversationID = nil
            isImportingFiles = false
        }
    }

    var isGenerating: Bool {
        if isTuningMTP { return true }
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
        guard canEject, let name = readyInfo?.model else { return nil }
        return models.first { $0.name == name }
    }

    var latestGenerationStats: GenerationStats? {
        guard currentConversation?.contextUsage?.model == readyInfo?.model else { return nil }
        return currentConversation?.messages.reversed().compactMap(\.stats).first
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
                    self.ejectModel()
                    self.selectedModelName = nil
                    self.engineState = .idle
                    return
                }
                if !models.contains(where: { $0.name == self.selectedModelName }) {
                    self.ejectModel()
                    self.selectedModelName = models[0].name
                }
                if autoLoad && !self.isGenerating { self.loadSelectedModel() }
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
        guard persistNow() else { return }
        Task {
        guard let installation = await updateManager.beginInstallation() else { return }
        if installation == .reloadEngine {
            loadSelectedModel()
            return
        }
        bridge.stop()
        persistNow()
        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
            Darwin.exit(EXIT_SUCCESS)
        }
        DispatchQueue.main.async {
            NSApplication.shared.terminate(nil)
        }
        }
    }

    func importModelFolder() {
        chooseModelFolder(replacing: nil)
    }

    func relocateModel(_ model: LocalModel) {
        chooseModelFolder(replacing: model.name)
    }

    private func chooseModelFolder(replacing name: String?) {
        guard !isPreview else { return }
        guard !modelInstallState.isWorking, !isGenerating else { return }
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
                self?.registerModelFolder(url, replacing: name)
            }
        }
    }

    func selectModel(_ name: String) {
        guard !isGenerating else { return }
        if name == selectedModelName, bridge.isRunning { return }
        selectedModelName = name
        schedulePersistence()
        loadSelectedModel()
    }

    func ejectModel() {
        guard canEject else { return }
        modelIdleUnloader.cancel()
        activeMessage()?.fail(L("Modèle éjecté", "Model unloaded"))
        activeRequestID = nil
        activeResponseID = nil
        readyInfo = nil
        dflashActive = nil
        runtimeIdentity = nil
        dflashSupported = nil
        mtpActive = nil
        mtpSupported = nil
        mtpAutoDownloadSupported = nil
        mtpConfigureSupported = false
        finishMTPTuning(error: nil)
        mtpMaxDepth = 1; mtpTuneSupported = false; mtpTuningKey = nil
        mtpTuneRows = []; mtpTunedAt = nil; mtpDepth = 1
        cancelMTPPreparation()
        activeContextLimit = nil
        modelContextLimit = nil
        contextMemory = nil
        modelResidentBytes = nil
        bridge.stop()
        engineState = .idle
        mcpUpdating = false
        mcpServerCount = 0
        mcpToolCount = 0
        mcpErrors = [:]
        schedulePersistence()
    }

    func newConversation() {
        if let id = selectedConversationID { drafts[id] = draft }
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        selectedConversationID = conversation.id
        draft = ""
        schedulePersistence()
    }

    func deleteConversation(_ id: UUID) {
        guard let removed = conversations.first(where: { $0.id == id }) else { return }
        clearAttachmentDraft(id)
        drafts[id] = nil
        if removed.messages.contains(where: { $0.id == activeResponseID }) {
            activeMessage()?.fail(L("Conversation supprimée", "Conversation deleted"))
            _ = bridge.cancelGeneration()
        }
        deletedConversation = removed
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
            draft = selectedConversationID.flatMap { drafts[$0] } ?? ""
        }
        schedulePersistence()
    }

    func send() {
        guard canSend, let conversationIndex else { return }
        let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = trimmed.isEmpty ? L("Analyse les fichiers joints.", "Analyze the attached files.") : trimmed
        let attachments = pendingAttachments

        draft = ""
        drafts[conversations[conversationIndex].id] = nil
        clearAttachmentDraft(conversations[conversationIndex].id)
        if conversations[conversationIndex].messages.isEmpty {
            conversations[conversationIndex].title = title(for: trimmed.isEmpty ? attachments.first?.fileName ?? text : text)
        }
        conversations[conversationIndex].messages.append(
            ChatMessage(role: .user, content: text, attachments: attachments)
        )
        let assistant = ChatMessage(role: .assistant, content: "", isStreaming: true)
        conversations[conversationIndex].messages.append(assistant)
        schedulePersistence()

        let requestID = UUID().uuidString
        activeRequestID = requestID
        activeResponseID = assistant.id
        dflashActive = nil
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
                : message.promptContent
            return PromptMessage(role: message.role.rawValue, content: context, turnContext: message.turnContext)
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
                    mcpEnabled: mcpEnabled,
                    dflash2: dflash2Enabled && dflash2Available,
                    dflashDraftPath: dflashDraftPath,
                    mtp: mtpEnabled && mtpAvailable,
                    mtpHeadPath: mtpHeadPath,
                    mtpDepth: mtpDepth
                )
            )
        } catch {
            failActiveTurn(error.localizedDescription)
        }
    }

    func stopGeneration() {
        if isTuningMTP { cancelMTPTuning(); return }
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
        prepareMCPConfiguration()
        NSWorkspace.shared.open(mcpConfigurationURL)
    }

    private func prepareMCPConfiguration() {
        do { try MCPConfiguration.ensureExa(at: mcpConfigurationURL) }
        catch { mcpErrors["configuration"] = error.localizedDescription }
    }

    func settingsDidChange() {
        schedulePersistence()
    }

    func setDFlash2Enabled(_ enabled: Bool) {
        if !enabled {
            dflashDownloadTask?.cancel()
            dflash2Enabled = false
            preferences.set(false, forKey: "studio.dflash2Enabled")
            return
        }
        guard dflash2Available, !isGenerating, !dflashDownloading else { return }
        dflashDownloadError = nil
        dflashDownloadCompleted = 0
        dflashDownloadTotal = 0
        dflashDownloading = true
        dflashDownloadTask = Task { [self] in
            do {
                var data: Data?
                if !dflashDraftPath.isEmpty {
                    data = try? await CLICommand().output(["dflash-draft", "--inspect", dflashDraftPath])
                }
                try Task.checkCancellation()
                if data == nil {
                    data = try await CLICommand().output(["dflash-draft"]) { [weak self] line in
                        guard let self,
                              let event = try? JSONDecoder().decode(DFlashInstallEvent.self, from: line),
                              event.type == "progress" else { return }
                        self.dflashDownloadCompleted = event.completed ?? self.dflashDownloadCompleted
                        self.dflashDownloadTotal = event.total ?? self.dflashDownloadTotal
                    }
                }
                try Task.checkCancellation()
                guard let data,
                      let event = try? JSONDecoder().decode(DFlashInstallEvent.self, from: data),
                      event.type == "installed", let path = event.path else {
                    throw MLXL3BridgeError.invalidResponse
                }
                dflashDraftPath = path
                preferences.set(path, forKey: "studio.dflashDraftPath")
                enableDFlashGreedy()
            } catch is CancellationError { }
            catch { dflashDownloadError = error.localizedDescription }
            dflashDownloading = false
            dflashDownloadTask = nil
        }
    }

    private func cancelMTPPreparation() {
        mtpOperationID = UUID()
        mtpDownloadTask?.cancel()
        mtpDownloadTask = nil
        mtpDownloading = false
        mtpConfigurationPending = false
    }

    private func configureMTP(requestID: String, enabled: Bool, headPath: String) throws {
        mtpConfigurationPending = true
        do { try bridge.setMTP(requestID: requestID, enabled: enabled, headPath: headPath) }
        catch { mtpConfigurationPending = false; throw error }
        // ON already has the preparation task's timeout; OFF must also recover
        // if a live engine never acknowledges the request.
        if !enabled {
            let operation = mtpOperationID
            mtpDownloadTask = Task { [weak self] in
                do { try await Task.sleep(for: .seconds(30)) }
                catch { return }
                guard let self, operation == self.mtpOperationID, self.mtpConfigurationPending else { return }
                self.mtpError = L("La configuration MTP ne répond pas.", "MTP configuration did not respond.")
                self.mtpActive = nil
                self.cancelMTPPreparation()
            }
        }
    }

    func setMTPEnabled(_ enabled: Bool) {
        guard !isGenerating else { return }
        if !enabled {
            cancelMTPPreparation()
            mtpEnabled = false
            preferences.set(false, forKey: "studio.mtpEnabled")
            saveMTPSelection()
            if mtpConfigureSupported && bridge.isRunning {
                do { try configureMTP(requestID: mtpOperationID.uuidString, enabled: false, headPath: "") }
                catch { mtpError = error.localizedDescription }
            }
            return
        }
        guard mtpAvailable, engineState.isReady, !mtpDownloading, let model = selectedModel else { return }
        cancelMTPPreparation()
        let operation = mtpOperationID
        preferences.set(true, forKey: "studio.mtpEnabled")
        mtpError = nil; mtpDownloadCompleted = 0; mtpDownloadTotal = 0
        mtpDownloading = true
        mtpDownloadTask = Task { [weak self] in
            guard let self else { return }
            do {
                var data: Data?
                let target = mtpConfigureSupported ? ["--target", model.path] : []
                if !mtpHeadPath.isEmpty {
                    do { data = try await CLICommand().output(["mtp-head", "--inspect", mtpHeadPath] + target) }
                    catch is CancellationError { throw CancellationError() }
                    catch { if !mtpAutomaticDownloadAvailable { throw error } }
                }
                if data == nil {
                    guard mtpAutomaticDownloadAvailable else {
                        throw MLXL3BridgeError.commandFailed(L("Choisis la tête MTP de ce modèle.", "Choose this model’s MTP head."))
                    }
                    data = try await CLICommand().output(["mtp-head"] + target) { [weak self] line in
                        guard let self, operation == self.mtpOperationID,
                              let event = try? JSONDecoder().decode(DFlashInstallEvent.self, from: line), event.type == "progress" else { return }
                        self.mtpDownloadCompleted = event.completed ?? self.mtpDownloadCompleted
                        self.mtpDownloadTotal = event.total ?? self.mtpDownloadTotal
                    }
                }
                try Task.checkCancellation()
                guard operation == mtpOperationID, model == selectedModel, mtpAvailable else { throw CancellationError() }
                guard let data, let event = try? JSONDecoder().decode(DFlashInstallEvent.self, from: data),
                      event.type == "installed", let path = event.path, !path.isEmpty else { throw MLXL3BridgeError.invalidResponse }
                mtpHeadPath = path
                preferences.set(path, forKey: "studio.mtpHeadPath")
                preferences.set(path, forKey: "studio.mtpHead.\(model.path)")
                dflash2Enabled = false
                mtpEnabled = true
                temperature = 0; topK = 1; repetitionPenalty = 1
                preferences.set(true, forKey: "studio.mtpEnabled")
                saveMTPSelection()
                schedulePersistence()
                if mtpConfigureSupported {
                    try configureMTP(requestID: operation.uuidString, enabled: true, headPath: path)
                    try await Task.sleep(for: .seconds(30))
                    throw MLXL3BridgeError.commandFailed(L("Le chargement MTP ne répond pas.", "MTP loading did not respond."))
                }
            } catch is CancellationError { }
            catch {
                guard operation == mtpOperationID else { return }
                mtpEnabled = false; mtpActive = false
                mtpError = error.localizedDescription
                cancelMTPPreparation()
                if mtpConfigureSupported && bridge.isRunning {
                    try? configureMTP(requestID: mtpOperationID.uuidString, enabled: false, headPath: "")
                }
                return
            }
            guard operation == mtpOperationID else { return }
            mtpDownloading = false
            mtpDownloadTask = nil
        }
    }

    func chooseMTPHead() {
        guard !isGenerating, !mtpDownloading else { return }
        let picker = NSOpenPanel()
        picker.canChooseDirectories = true; picker.canChooseFiles = false; picker.allowsMultipleSelection = false
        picker.prompt = L("Utiliser cette tête MTP", "Use this MTP head")
        guard picker.runModal() == .OK, let path = picker.url?.path else { return }
        mtpEnabled = false; preferences.set(false, forKey: "studio.mtpEnabled")
        mtpHeadPath = path; preferences.set(path, forKey: "studio.mtpHeadPath")
        mtpDepth = 1; mtpTuneRows = []; mtpTunedAt = nil
        setMTPEnabled(true)
    }

    private var currentMTPConfiguration: MTPConfigurationKey? {
        guard let model = selectedModel, let runtime = mtpTuningKey, !mtpHeadPath.isEmpty else { return nil }
        return MTPConfigurationKey(modelPath: model.path, headPath: mtpHeadPath, runtime: runtime)
    }

    var canTuneMTP: Bool {
        mtpAvailable && mtpTuneSupported && engineState.isReady && !isGenerating
            && !mtpDownloading && !mcpUpdating && !updateManager.isInstalling && !modelInstallState.isWorking
            && currentMTPConfiguration != nil
    }

    func setMTPDepth(_ depth: Int) {
        guard !isGenerating, (1...mtpMaxDepth).contains(depth) else { return }
        mtpDepth = depth
        saveMTPSelection()
    }

    private func saveMTPSelection() {
        guard let key = currentMTPConfiguration else { return }
        MTPTuning.save(MTPSelection(key: key, depth: mtpEnabled ? mtpDepth : 0,
            rows: mtpTuneRows, tunedAt: mtpTunedAt), preferences: preferences)
    }

    private func restoreMTPSelection() {
        mtpDepth = 1; mtpTuneRows = []; mtpTunedAt = nil
        guard let key = currentMTPConfiguration,
              let saved = MTPTuning.load(key: key, preferences: preferences), saved.depth <= mtpMaxDepth else { return }
        mtpDepth = max(1, saved.depth)
        // The global switch owns ON/OFF; saved tuning only restores the depth.
        mtpTuneRows = saved.rows; mtpTunedAt = saved.tunedAt
        if mtpEnabled { temperature = 0; topK = 1; repetitionPenalty = 1 }
    }

    func tuneMTP() {
        guard canTuneMTP, let key = currentMTPConfiguration else { return }
        let request = UUID().uuidString
        tuneRequestID = request; tuneConfiguration = key; tuneCancellationRequested = false
        isTuningMTP = true; mtpTuneProgress = 0; mtpError = nil
        mtpTuneStatus = L("Préparation du test…", "Preparing test…")
        do { try bridge.tuneMTP(requestID: request, headPath: mtpHeadPath) }
        catch { finishMTPTuning(error: error.localizedDescription) }
    }

    func cancelMTPTuning() {
        guard isTuningMTP else { return }
        tuneCancellationRequested = true
        mtpTuneStatus = L("Arrêt du test…", "Stopping test…")
        if !bridge.cancelGeneration() {
            finishMTPTuning(error: L("Test arrêté. Réglage conservé.", "Test stopped. Setting preserved."))
        }
    }

    private func finishMTPTuning(error: String?) {
        isTuningMTP = false; tuneRequestID = nil; tuneConfiguration = nil
        if let error { mtpError = error }
    }

    private func handleMTPTuning(_ event: BridgeEvent) {
        switch event.type {
        case "mtp_tune_progress":
            guard !tuneCancellationRequested else { return }
            if let completed = event.completed, let total = event.total, total > 0 {
                mtpTuneProgress = min(1, max(0, Double(completed) / Double(total)))
            }
            let mode = event.depth.map { $0 == 0 ? "Baseline" : "MTP\($0)" } ?? "MTP"
            mtpTuneStatus = event.phase == "warmup"
                ? L("Échauffement · ", "Warmup · ") + mode
                : event.phase == "loading_head" ? L("Chargement de la tête MTP…", "Loading MTP head…")
                : L("Mesure · ", "Measuring · ") + mode
        case "mtp_tune_complete":
            guard !tuneCancellationRequested, let key = tuneConfiguration,
                  key == currentMTPConfiguration, event.tuningKey == key.runtime,
                  let rows = event.rows, let best = MTPTuning.winner(rows), best == event.bestDepth else {
                finishMTPTuning(error: L("Test incomplet ou annulé. Réglage conservé.", "Incomplete or cancelled test. Setting preserved.")); return
            }
            mtpDepth = max(1, best); mtpEnabled = best > 0; mtpTuneRows = rows.sorted { $0.depth < $1.depth }
            mtpTunedAt = Date(); mtpTuneProgress = 1
            // Baseline is also measured in greedy mode, keeping the comparison valid.
            temperature = 0; topK = 1; repetitionPenalty = 1
            preferences.set(mtpEnabled, forKey: "studio.mtpEnabled")
            saveMTPSelection(); schedulePersistence(); finishMTPTuning(error: nil)
        case "cancelled":
            finishMTPTuning(error: L("Test arrêté. Réglage conservé.", "Test stopped. Setting preserved."))
        case "error":
            finishMTPTuning(error: event.message ?? L("Le test MTP a échoué. Réglage conservé.", "MTP test failed. Setting preserved."))
        default: break
        }
    }

    func chooseDFlashDraft() {
        let picker = NSOpenPanel()
        picker.canChooseDirectories = true
        picker.canChooseFiles = false
        picker.allowsMultipleSelection = false
        picker.prompt = L("Utiliser ce draft", "Use this draft")
        guard picker.runModal() == .OK, let path = picker.url?.path else { return }
        dflashDraftPath = path
        preferences.set(path, forKey: "studio.dflashDraftPath")
    }

    func enableDFlashGreedy() {
        temperature = 0
        topK = 1
        repetitionPenalty = 1
        dflash2Enabled = true
        preferences.set(true, forKey: "studio.dflash2Enabled")
        settingsDidChange()
    }

    func selectConversation(_ id: UUID) {
        guard conversations.contains(where: { $0.id == id }) else { return }
        if let previous = selectedConversationID { drafts[previous] = draft }
        selectedConversationID = id
        draft = drafts[id] ?? ""
        schedulePersistence()
    }

    @discardableResult func persistNow() -> Bool {
        persistenceTask?.cancel()
        persistenceTask = nil
        guard !persistenceBlocked else { return false }
        persistenceRevision += 1
        do {
            try conversationStore.save(workspaceSnapshot, revision: persistenceRevision)
            return true
        } catch {
            storageError = L("Échec de sauvegarde : ", "Save failed: ") + error.localizedDescription
            return false
        }
    }

    func revealConversations() {
        NSWorkspace.shared.activateFileViewerSelecting([conversationFileURL])
    }

    func recoverConversations() {
        do {
            if FileManager.default.fileExists(atPath: conversationFileURL.path) {
                try FileManager.default.moveItem(at: conversationFileURL, to: conversationFileURL.appendingPathExtension("unreadable-" + UUID().uuidString))
            }
            let backupURL = conversationFileURL.appendingPathExtension("backup")
            if let backup = try? ConversationStore.load(from: backupURL) {
                conversations = backup.conversations.map(Conversation.init(snapshot:))
                selectedConversationID = conversations.first?.id
            } else if FileManager.default.fileExists(atPath: backupURL.path) {
                try FileManager.default.moveItem(at: backupURL, to: backupURL.appendingPathExtension("unreadable-" + UUID().uuidString))
            }
            persistenceBlocked = false
            storageError = nil
            persistNow()
        } catch { storageError = error.localizedDescription }
    }

    func undoDeleteConversation() {
        guard let restored = deletedConversation else { return }
        conversations.removeAll { $0.id == restored.id }
        conversations.insert(restored, at: 0)
        selectedConversationID = restored.id
        deletedConversation = nil
        schedulePersistence()
    }

    func exportConversations() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "MLXL3-conversations.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(workspaceSnapshot).write(to: url, options: .atomic)
        } catch { storageError = error.localizedDescription }
    }

    func importConversations() {
        guard !isGenerating else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            guard let snapshot = try ConversationStore.load(from: url) else { return }
            let known = Set(conversations.map(\.id))
            conversations.insert(contentsOf: snapshot.conversations.filter { !known.contains($0.id) }.map(Conversation.init(snapshot:)), at: 0)
            persistNow()
        } catch { storageError = error.localizedDescription }
    }

    private var conversationIndex: Int? {
        guard let selectedConversationID else { return nil }
        return conversations.firstIndex { $0.id == selectedConversationID }
    }

    private func loadSelectedModel() {
        guard !isGenerating else { return }
        modelIdleUnloader.cancel()
        dflashActive = nil
        dflashSupported = nil
        mtpActive = nil
        mtpSupported = nil
        mtpAutoDownloadSupported = nil
        cancelMTPPreparation()
        mtpConfigureSupported = false
        mtpEnabled = preferences.bool(forKey: "studio.mtpEnabled")
        mtpHeadPath = selectedModel.flatMap { preferences.string(forKey: "studio.mtpHead.\($0.path)") }
            ?? (selectedModelName == mtpLegacyModelName ? preferences.string(forKey: "studio.mtpHeadPath") ?? "" : "")
        mtpError = nil
        runtimeIdentity = nil
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
        Task {
            do {
                try await bridge.start(model: selectedModelName, contextLength: savedContextLength)
            } catch is CancellationError { }
            catch {
                engineState = .failed(error.localizedDescription)
            }
        }
    }

    private func registerModelFolder(_ url: URL, replacing existingName: String? = nil) {
        guard !isGenerating else { return }
        let rawName = url.lastPathComponent
        let name = existingName ?? rawName.replacingOccurrences(
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
        if mtpConfigurationPending && event.type == "error" && (event.requestID == nil || event.requestID == "") {
            mtpError = event.message ?? L("La configuration MTP a échoué.", "MTP configuration failed.")
            mtpEnabled = false; mtpActive = nil
            cancelMTPPreparation()
            // Keep dispatching: an uncorrelated error may also fail an active
            // generation or tuning request, whose existing cleanup must run.
        }
        if event.requestID == mtpOperationID.uuidString && ["mtp_status", "error"].contains(event.type) {
            mtpConfigurationPending = false
            mtpDownloadTask?.cancel(); mtpDownloadTask = nil; mtpDownloading = false
            if event.type == "mtp_status" { mtpActive = event.mtpActive }
            else { mtpEnabled = false; mtpActive = false; mtpError = event.message }
            return
        }
        if isTuningMTP && event.type == "error" && (event.requestID == nil || event.requestID == "") {
            _ = bridge.cancelGeneration()
            finishMTPTuning(error: event.message ?? L("Réponse du moteur illisible. Réglage conservé.", "Unreadable engine response. Setting preserved."))
            return
        }
        if event.requestID == tuneRequestID, tuneRequestID != nil {
            handleMTPTuning(event)
            return
        }
        switch event.type {
        case "loading":
            engineState = .loading(event.model ?? selectedModelName ?? "modèle")
        case "ready":
            dflashSupported = event.dflashSupported
            mtpSupported = event.mtpSupported
            mtpAutoDownloadSupported = event.mtpAutoDownloadSupported
            mtpConfigureSupported = event.mtpConfigureSupported == true
            mtpMaxDepth = min(3, max(1, event.mtpMaxDepth ?? 1))
            mtpTuneSupported = event.mtpTuneSupported == true && mtpMaxDepth == 3
            mtpTuningKey = event.mtpTuningKey
            runtimeIdentity = [event.runtimeCommit, event.runtimeProfile, event.mlxVersion.map { "MLX \($0)" }]
                .compactMap { $0 }.joined(separator: " · ")
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
            restoreMTPSelection()
            mcpServerCount = event.mcpServers ?? 0
            mcpToolCount = event.mcpTools ?? 0
            mcpErrors = event.mcpErrors ?? [:]
            engineState = .ready(
                model: info.model,
                modules: info.modules,
                residentGB: info.residentGB
            )
            if mcpEnabled { updateMCPConnection() }
            if mtpEnabled && mtpAvailable { setMTPEnabled(true) }
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
        case "generation_mode":
            guard event.requestID == activeRequestID else { return }
            dflashActive = event.dflashActive
            mtpActive = event.mtpActive
            if let reason = event.mtpReason { mtpError = reason }
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
            guard event.requestID == activeRequestID else { return }
            if let stats = event.stats, let name = selectedModelName,
               let index = conversations.firstIndex(where: { $0.messages.contains { $0.id == activeResponseID } }),
               let limit = stats.contextLimit ?? conversations[index].contextUsage?.limit ?? activeContextLimit,
               let finalUsage = ContextUsage(used: 0, limit: limit, model: name).finalized(with: stats) {
                conversations[index].contextUsage = finalUsage
            }
            let message = activeMessage()
            message?.turnContext = event.turnContext
            message?.finish(
                stats: event.stats,
                fallbackAnswer: event.assistantContext,
                cacheContext: event.cacheContext
            )
            if event.contextFull == true {
                message?.fail(L("Limite de contexte atteinte. Augmente la limite dans les paramètres du modèle ou ouvre une nouvelle conversation.", "Context limit reached. Increase the limit in model settings or start a new conversation."))
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
        if let readyInfo, bridge.isRunning {
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
        if readyInfo == nil || !bridge.isRunning {
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
        guard persistenceTask == nil, !persistenceBlocked else { return }
        persistenceTask = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .seconds(self?.isGenerating == true ? 10 : 1))
            guard !Task.isCancelled, let self else { return }
            persistenceTask = nil
            let snapshot = workspaceSnapshot
            persistenceRevision += 1
            let revision = persistenceRevision
            let store = conversationStore
            do {
                try await Task.detached(priority: .utility) { try store.save(snapshot, revision: revision) }.value
            } catch { storageError = L("Échec de sauvegarde : ", "Save failed: ") + error.localizedDescription }
        }
    }
}
