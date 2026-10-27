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
    @Published private(set) var selectedMode: StudioMode = .chat
    @Published private(set) var labSamples: [LabSample] = []
    let localServer = LocalServer()
    private var labConfiguration: ((GenerationStats) -> LabSample)?
    private var modeTransition: Task<Void, Never>?
    private var serverObservation: AnyCancellable?
    private var bridgeUsesAgentRegistry = false
    @Published var models: [LocalModel] = []
    @Published var selectedModelName: String?
    @Published var conversations: [Conversation] = [Conversation()]
    @Published var selectedConversationID: UUID?
    @Published var projects: [WorkspaceProject] = []
    @Published var profiles: [GenerationProfile] = []
    @Published var documents: [LibraryDocument] = []
    @Published var selectedProjectID: UUID?
    @Published var showWorkspace = false
    @Published var showMCPManager = false
    @Published var responseSchema = ""
    @Published var featureError: String?
    @Published var libraryImporting = false
    @Published var mcpTools: [MCPToolDescriptor] = []
    @Published private(set) var workspaceFeaturesAvailable = false
    let voice = VoiceController()
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
    @Published private(set) var dflashMode = 1
    @Published private(set) var dflashTuneRows: [MTPTuningRow] = []
    @Published private(set) var dflashTuneSupported = false
    private var dflashTunedAt: Date?
    private var dflashTuningKey: String?
    private var dflashConfigureSupported = false
    private var dflashOperationID = UUID()
    private var dflashConfigurationPending = false {
        didSet { updateModelIdleUnload() }
    }
    private var dflashLegacyModelName: String?
    private var tuneIsDFlash = false
    var isTuningDFlash: Bool { isTuningMTP && tuneIsDFlash }
    var dflashTuneProgress: Double { mtpTuneProgress }
    var dflashTuneStatus: String { mtpTuneStatus }
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
    @Published private(set) var memorySaverEnabled = false
    @Published private(set) var memorySaverSupported = false

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
    private var activeRequestID: String? {
        didSet {
            if activeRequestID == nil {
                generationWatchdog?.cancel(); generationWatchdog = nil
                cancellationFallback?.cancel(); cancellationFallback = nil
                cancellationReason = nil
                generationStartedAt = nil; generationLastActivity = nil
            }
        }
    }
    private var generationWatchdog: Task<Void, Never>?
    private var cancellationFallback: Task<Void, Never>?
    private var cancellationReason: String?
    private var generationStartedAt: ContinuousClock.Instant?
    private var generationLastActivity: ContinuousClock.Instant?
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
        self.memorySaverEnabled = preferences.bool(forKey: "studio.memorySaverEnabled")
        self.language = AppLanguage(rawValue: preferences.string(forKey: "studio.language") ?? "fr") ?? .fr
        self.mcpEnabled = preferences.bool(forKey: "studio.mcpEnabled")
        self.dflash2Enabled = preferences.bool(forKey: "studio.dflash2Enabled")
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
            labSamples = Array((snapshot.labSamples ?? []).filter(\.valid).suffix(1000))
            projects = snapshot.projects ?? []
            profiles = snapshot.profiles ?? []
            documents = snapshot.documents ?? []
            responseSchema = snapshot.responseSchema ?? ""
            selectedProjectID = conversations.first(where: { $0.id == selectedConversationID })?.projectID
        } else {
            selectedConversationID = conversations.first?.id
        }
        } catch {
            persistenceBlocked = true
            storageError = L("Historique illisible : aucune donnée ne sera écrasée. ", "History could not be read: no data will be overwritten. ") + error.localizedDescription
            selectedConversationID = conversations.first?.id
        }
        if dflash2Enabled && mtpEnabled {
            mtpEnabled = false; preferences.set(false, forKey: "studio.mtpEnabled")
        }
        if dflash2Enabled {
            if dflashDraftPath.isEmpty {
                dflash2Enabled = false
                preferences.set(false, forKey: "studio.dflash2Enabled")
            }
        }
        mtpLegacyModelName = selectedModelName
        dflashLegacyModelName = selectedModelName
        bridge.onEvent = { [weak self] event in self?.handle(event) }
        serverObservation = localServer.objectWillChange.sink { [weak self] in self?.objectWillChange.send() }
        selectedMode = currentConversation?.agent == true ? .agent : .chat
        bridge.onRuntimeFallback = { [weak self] in
            guard let self else { return }
            self.readyInfo = nil
            self.cancelMTPPreparation()
            self.cancelDFlashPreparation()
            self.finishMTPTuning(error: nil)
            self.loadSelectedModel()
        }
        bridge.onExit = { [weak self] message in
            guard let self, let message, !message.isEmpty else { return }
            self.readyInfo = nil
            self.cancelMTPPreparation()
            self.cancelDFlashPreparation()
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
        mtpSupported ?? ["qwen3_5", "qwen3_5_moe", "qwen4_exp"].contains(selectedModel?.modelType ?? "")
    }
    var mtpBuiltIn: Bool { selectedModel?.modelType == "qwen4_exp" }
    var mtpEffectiveHeadPath: String { mtpBuiltIn ? selectedModel?.path ?? "" : mtpHeadPath }
    // Saved intent can survive an incompatible model; applicability is not engine activity.
    var mtpApplicable: Bool { mtpEnabled && mtpAvailable }
    var dflash2Applicable: Bool { dflash2Enabled && dflash2Available }
    var samplingLocked: Bool { mtpApplicable || dflash2Applicable }
    var samplingTemperature: Double { samplingLocked ? 0 : temperature }
    var samplingTopK: Int { samplingLocked ? 1 : topK }
    var samplingRepetitionPenalty: Double { samplingLocked ? 1 : repetitionPenalty }

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

    func setMemorySaverEnabled(_ enabled: Bool) {
        guard !isGenerating && !isTuningMTP else { return }
        memorySaverEnabled = enabled
        preferences.set(enabled, forKey: "studio.memorySaverEnabled")
    }

    private var canAutomaticallyUnloadModel: Bool {
        !isPreview && engineState.isReady && readyInfo != nil && bridge.isRunning
            && !isGenerating && !mcpUpdating && !mtpDownloading
            && !dflashDownloading && !dflashConfigurationPending && !mtpConfigurationPending
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

    var recommendedContextLength: Int? {
        guard let profile = contextMemory, let maximum = modelContextLimit,
              let resident = modelResidentBytes else { return nil }
        let observed = bridge.engineMemoryFootprintBytes().map(Double.init) ?? 0
        return profile.recommendedTokens(maximum: maximum,
            physicalBytes: Double(ProcessInfo.processInfo.physicalMemory),
            residentBytes: max(resident, observed))
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
        !localServer.active && modeTransition == nil && (selectedMode == .chat || selectedMode == .agent)
            && (selectedMode != .agent || (bridgeUsesAgentRegistry && mcpToolCount > 0 && mcpErrors.isEmpty))
            && engineState.isReady && !isTuningMTP && !mcpUpdating && !dflashDownloading && !dflashConfigurationPending && !mtpDownloading && !modelInstallState.isWorking && !updateManager.isInstalling
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
        guard !isGenerating, !localServer.active, modeTransition == nil else { return }
        if name == selectedModelName, bridge.isRunning { return }
        selectedModelName = name
        schedulePersistence()
        loadSelectedModel()
    }

    func ejectModel() {
        guard canEject else { return }
        modelIdleUnloader.cancel()
        cancelDFlashPreparation()
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
        memorySaverSupported = false
        finishMTPTuning(error: nil)
        mtpMaxDepth = 1; mtpTuneSupported = false; mtpTuningKey = nil
        mtpTuneRows = []; mtpTunedAt = nil; mtpDepth = 1
        cancelMTPPreparation()
        activeContextLimit = nil
        modelContextLimit = nil
        contextMemory = nil
        modelResidentBytes = nil
        bridge.stop()
        bridgeUsesAgentRegistry = false
        engineState = .idle
        mcpUpdating = false
        mcpServerCount = 0
        mcpToolCount = 0
        mcpErrors = [:]
        schedulePersistence()
    }

    func newConversation() {
        guard !isGenerating || currentConversation?.agent != true else { return }
        voice.stop()
        if let id = selectedConversationID { drafts[id] = draft }
        var conversation = Conversation()
        conversation.projectID = selectedProjectID
        conversation.agent = selectedMode == .agent
        if selectedMode == .lab || selectedMode == .server { selectMode(.chat) }
        conversations.insert(conversation, at: 0)
        selectedConversationID = conversation.id
        draft = ""
        schedulePersistence()
        if conversation.agent { refreshAgentWorkspace() }
    }

    func deleteConversation(_ id: UUID) {
        guard let removed = conversations.first(where: { $0.id == id }) else { return }
        clearAttachmentDraft(id)
        drafts[id] = nil
        if removed.messages.contains(where: { $0.id == activeResponseID }) {
            forceStopGeneration(L("Conversation supprimée", "Conversation deleted"))
        }
        deletedConversation = removed
        guard conversations.count > 1 else {
            if let index = conversations.firstIndex(where: { $0.id == id }) {
                conversations[index] = Conversation(id: id)
                conversations[index].projectID = removed.projectID
                conversations[index].agent = removed.agent
            }
            voice.stop(); draft = ""
            schedulePersistence()
            return
        }
        conversations.removeAll { $0.id == id }
        if selectedConversationID == id, let next = conversations.first {
            selectConversation(next.id)
            drafts[id] = nil
        }
        schedulePersistence()
    }

    func send() {
        guard canSend, let conversationIndex else { return }
        if selectedMode != .agent && !responseSchema.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard workspaceFeaturesAvailable else { featureError = L("Ce moteur ne prend pas en charge le JSON validé. Mettez le moteur à jour.", "This engine does not support validated JSON. Update the engine."); return }
            do {
                guard try JSONSerialization.jsonObject(with: Data(responseSchema.utf8)) is [String: Any] else { throw MCPSettingsError.invalid }
            } catch { featureError = L("Le schéma doit être un objet JSON valide.", "The schema must be a valid JSON object."); return }
        }
        if mcpEnabled && currentProject != nil && !workspaceFeaturesAvailable {
            featureError = L("Ce moteur ne prend pas en charge les permissions par projet. Mettez le moteur à jour.", "This engine does not support per-project permissions. Update the engine."); return
        }
        voice.stop()
        let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        let text = trimmed.isEmpty ? L("Analyse les fichiers joints.", "Analyze the attached files.") : trimmed
        let attachments = pendingAttachments

        draft = ""
        drafts[conversations[conversationIndex].id] = nil
        clearAttachmentDraft(conversations[conversationIndex].id)
        if conversations[conversationIndex].messages.isEmpty {
            conversations[conversationIndex].title = title(for: trimmed.isEmpty ? attachments.first?.fileName ?? text : text)
        }
        let projectDocuments = documents.filter { $0.projectID == conversations[conversationIndex].projectID }
        let sources = DocumentSearch.search(text, documents: projectDocuments)
        let user = ChatMessage(role: .user, content: text, attachments: attachments)
        user.sources = sources
        conversations[conversationIndex].messages.append(user)
        let assistant = ChatMessage(role: .assistant, content: "", isStreaming: true)
        assistant.sources = sources
        conversations[conversationIndex].messages.append(assistant)
        schedulePersistence()

        let requestID = UUID().uuidString
        activeRequestID = requestID
        activeResponseID = assistant.id
        if selectedMode == .agent {
            generationStartedAt = .now
            generationLastActivity = .now
            generationWatchdog = Task { [weak self] in
                while !Task.isCancelled {
                    try? await Task.sleep(for: .seconds(1))
                    guard !Task.isCancelled, let self else { return }
                    self.checkAgentGenerationDeadline(at: .now)
                }
            }
        }
        labConfiguration = nil
        if let model = selectedModel {
            let mode = selectedMode.rawValue, runtime = runtimeIdentity ?? "unknown"
            let system = ProcessInfo.processInfo.operatingSystemVersionString, hardware = LabHardware.identity
            let context = activeContextLimit ?? 0, temperature = samplingTemperature, topK = samplingTopK
            let mtp = mtpApplicable, dflash = dflash2Applicable
            labConfiguration = { stats in LabSample(model: model.name, modelPath: model.path, bits: model.bits, mode: mode,
                runtime: runtime, system: system, hardware: hardware, context: context, temperature: temperature,
                topK: topK, mtp: mtp, dflash: dflash, stats: stats) }
        }
        dflashActive = nil
        engineState = .generating

        var messages: [PromptMessage] = []
        let system = [systemPrompt, currentProject?.instructions ?? "", sources.isEmpty ? "" : DocumentSearch.instruction,
            selectedMode == .agent ? "You are a local coding agent. Use the run tool to inspect, create and edit files in your private workspace, then verify your work. Commands use macOS system tools, have no network or access to personal files, and are limited to 5 seconds and 32 KiB of output. Use relative paths. Never claim to have executed a command unless its tool result confirms it. Explain failures and limits honestly." : ""]
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.joined(separator: "\n\n")
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
                    maxTokens: selectedMode == .agent ? 4096 : -1,
                    temperature: samplingTemperature,
                    topK: samplingTopK,
                    repetitionPenalty: samplingRepetitionPenalty,
                    mcpEnabled: selectedMode == .agent || (mcpEnabled && responseSchema.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty),
                    dflash2: dflash2Applicable,
                    dflashDraftPath: dflashDraftPath,
                    dflashMode: dflashMode,
                    mtp: mtpApplicable,
                    mtpHeadPath: mtpEffectiveHeadPath,
                    mtpDepth: mtpDepth,
                    memorySaver: memorySaverEnabled && memorySaverSupported,
                    allowedTools: selectedMode == .agent ? nil : currentProject?.allowedTools,
                    responseSchema: selectedMode == .agent || responseSchema.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : responseSchema
                )
            )
        } catch {
            failActiveTurn(error.localizedDescription)
        }
    }

    func stopGeneration() {
        if isTuningMTP { cancelMTPTuning(); return }
        guard isGenerating else { return }
        requestGenerationStop(L("Génération arrêtée", "Generation stopped"))
    }

    private func requestGenerationStop(_ reason: String) {
        guard let request = activeRequestID else { return }
        cancellationReason = reason
        guard cancellationFallback == nil, bridge.cancelGeneration() else {
            forceStopGeneration(reason); return
        }
        activeMessage()?.processing(L("Arrêt de la génération…", "Stopping generation…"))
        cancellationFallback = Task { [weak self] in
            try? await Task.sleep(for: .seconds(3))
            guard !Task.isCancelled, self?.activeRequestID == request else { return }
            self?.forceStopGeneration(reason)
        }
    }

    private func forceStopGeneration(_ reason: String) {
        activeMessage()?.fail(reason)
        labConfiguration = nil
        activeRequestID = nil; activeResponseID = nil
        ejectModel()
        schedulePersistence()
    }

    private func checkAgentGenerationDeadline(at now: ContinuousClock.Instant) {
        guard activeRequestID != nil, cancellationFallback == nil,
              let start = generationStartedAt, let last = generationLastActivity else { return }
        if now - start >= .seconds(600) {
            requestGenerationStop(L("Limite de 10 minutes atteinte. La réponse partielle est conservée.", "10-minute Agent limit reached. Partial output was kept."))
        } else if now - last >= .seconds(120) {
            requestGenerationStop(L("L’agent ne répond plus depuis 2 minutes. La réponse partielle est conservée.", "Agent has not responded for 2 minutes. Partial output was kept."))
        }
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
            try bridge.setMCPEnabled(bridgeUsesAgentRegistry || mcpEnabled)
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

    private func cancelDFlashPreparation() {
        dflashOperationID = UUID()
        dflashDownloadTask?.cancel(); dflashDownloadTask = nil
        dflashDownloading = false; dflashConfigurationPending = false
    }

    func setDFlash2Enabled(_ enabled: Bool, downloadCompatible: Bool = false) {
        guard !isGenerating else { return }
        if !enabled {
            cancelDFlashPreparation()
            dflash2Enabled = false
            preferences.set(false, forKey: "studio.dflash2Enabled")
            saveDFlashSelection()
            if dflashConfigureSupported && bridge.isRunning {
                do {
                    dflashConfigurationPending = true
                    try bridge.setDFlash(requestID: dflashOperationID.uuidString, enabled: false, draftPath: "")
                    let operation = dflashOperationID
                    dflashDownloadTask = Task { [weak self] in
                        do { try await Task.sleep(for: .seconds(30)) }
                        catch { return }
                        guard let self, operation == self.dflashOperationID, self.dflashConfigurationPending else { return }
                        self.dflashDownloadError = L("La configuration DFlash ne répond pas.", "DFlash configuration did not respond.")
                        self.dflashActive = nil
                        self.cancelDFlashPreparation()
                    }
                }
                catch { dflashDownloadError = error.localizedDescription; dflashActive = nil; cancelDFlashPreparation() }
            }
            return
        }
        guard dflash2Available, engineState.isReady, !dflashDownloading, let model = selectedModel else { return }
        cancelDFlashPreparation()
        let operation = dflashOperationID
        dflashDownloadError = nil; dflashDownloadCompleted = 0; dflashDownloadTotal = 0
        let flashDraft = model.modelType == "qwen4_exp"
        if flashDraft && (!dflashConfigureSupported || (!downloadCompatible && dflashDraftPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)) {
            dflash2Enabled = false
            preferences.set(false, forKey: "studio.dflash2Enabled")
            dflashDownloadError = !dflashConfigureSupported
                ? L("Mets le moteur à jour pour charger un draft Flash Next.", "Update the engine to load a Flash Next draft.")
                : L("Télécharge le draft compatible avec le bouton ci-dessous, ou choisis son dossier local.", "Download the compatible draft with the button below, or choose its local folder.")
            return
        }
        dflashDownloading = true
        dflashDownloadTask = Task { [self] in
            do {
                var data: Data?
                if !flashDraft || downloadCompatible {
                    if !downloadCompatible && !dflashDraftPath.isEmpty {
                        data = try? await CLICommand().output(["dflash-draft", "--target", model.path, "--inspect", dflashDraftPath])
                    }
                    try Task.checkCancellation()
                    if data == nil {
                        data = try await CLICommand().output(["dflash-draft", "--target", model.path]) { [weak self] line in
                            guard let self, operation == self.dflashOperationID,
                                  let event = try? JSONDecoder().decode(DFlashInstallEvent.self, from: line), event.type == "progress" else { return }
                            self.dflashDownloadCompleted = event.completed ?? self.dflashDownloadCompleted
                            self.dflashDownloadTotal = event.total ?? self.dflashDownloadTotal
                        }
                    }
                }
                try Task.checkCancellation()
                guard operation == dflashOperationID, selectedModel?.path == model.path else { return }
                let path: String
                if flashDraft && !downloadCompatible {
                    // Raw Flash drafts are validated by the resident loader and its acknowledgement.
                    path = dflashDraftPath
                } else {
                    guard let data, let event = try? JSONDecoder().decode(DFlashInstallEvent.self, from: data),
                          event.type == "installed", let installedPath = event.path, !installedPath.isEmpty else { throw MLXL3BridgeError.invalidResponse }
                    path = installedPath
                }
                if !downloadCompatible { commitDFlashDraft(path, modelPath: model.path) }
                cancelMTPPreparation(); mtpEnabled = false
                preferences.set(false, forKey: "studio.mtpEnabled")
                if dflashConfigureSupported {
                    dflashConfigurationPending = true
                    try bridge.setDFlash(requestID: operation.uuidString, enabled: true, draftPath: path)
                    // Bound an unresponsive resident engine; stale tasks never enable a new model.
                    for _ in 0..<300 where dflashConfigurationPending {
                        try await Task.sleep(for: .milliseconds(100))
                        try Task.checkCancellation()
                    }
                    guard !dflashConfigurationPending else { throw MLXL3BridgeError.commandFailed(L("Le draft ne répond pas.", "Draft loading did not respond.")) }
                    guard dflashActive == true else { throw MLXL3BridgeError.commandFailed(dflashDownloadError ?? "DFlash loading failed") }
                }
                if downloadCompatible { commitDFlashDraft(path, modelPath: model.path) }
                enableDFlashGreedy()
                saveDFlashSelection()
            } catch is CancellationError { }
            catch {
                if operation == dflashOperationID {
                    dflashDownloadError = (downloadCompatible
                        ? L("Impossible de préparer le draft Flash Next : ", "Unable to prepare the Flash Next draft: ") : "") + error.localizedDescription
                    dflash2Enabled = false
                }
            }
            if operation == dflashOperationID {
                dflashDownloading = false; dflashConfigurationPending = false; dflashDownloadTask = nil
            }
        }
    }

    private var currentDFlashConfiguration: MTPConfigurationKey? {
        guard let model = selectedModel, let runtime = dflashTuningKey, !dflashDraftPath.isEmpty else { return nil }
        return MTPConfigurationKey(modelPath: model.path, headPath: dflashDraftPath, runtime: runtime, packedDraft: true)
    }
    var canTuneDFlash: Bool {
        dflash2Available && dflashTuneSupported && engineState.isReady && !isGenerating
            && !dflashDownloading && !dflashConfigurationPending && !mtpDownloading && !mtpConfigurationPending
            && !mcpUpdating && !updateManager.isInstalling && !modelInstallState.isWorking && currentDFlashConfiguration != nil
    }
    func setDFlashMode(_ mode: Int) {
        guard !isGenerating, (1...3).contains(mode) else { return }
        dflashMode = mode; saveDFlashSelection()
    }
    private func saveDFlashSelection() {
        guard let key = currentDFlashConfiguration else { return }
        MTPTuning.save(MTPSelection(key: key, depth: dflashMode,
            rows: dflashTuneRows, tunedAt: dflashTunedAt), preferences: preferences, storageKey: "studio.dflashSelections")
    }
    private func restoreDFlashSelection() {
        dflashMode = 1; dflashTuneRows = []; dflashTunedAt = nil
        guard let key = currentDFlashConfiguration,
              let saved = MTPTuning.load(key: key, preferences: preferences, storageKey: "studio.dflashSelections") else { return }
        dflashMode = max(1, saved.depth); dflashTuneRows = saved.rows; dflashTunedAt = saved.tunedAt
    }
    func tuneDFlash() {
        guard canTuneDFlash, let key = currentDFlashConfiguration else { return }
        let request = UUID().uuidString
        tuneIsDFlash = true; tuneRequestID = request; tuneConfiguration = key; tuneCancellationRequested = false
        isTuningMTP = true; mtpTuneProgress = 0; dflashDownloadError = nil
        mtpTuneStatus = L("Préparation du test…", "Preparing test…")
        do { try bridge.tuneDFlash(requestID: request, draftPath: dflashDraftPath,
                                  memorySaver: memorySaverEnabled && memorySaverSupported) }
        catch { finishMTPTuning(error: error.localizedDescription) }
    }
    func cancelDFlashTuning() { cancelMTPTuning() }

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
        cancelDFlashPreparation()
        dflash2Enabled = false; preferences.set(false, forKey: "studio.dflash2Enabled")
        cancelMTPPreparation()
        let operation = mtpOperationID
        preferences.set(true, forKey: "studio.mtpEnabled")
        if mtpBuiltIn {
            mtpError = nil
            mtpDepth = 1
            activateMTP()
            if !mtpConfigureSupported { return }
        }
        mtpError = nil; mtpDownloadCompleted = 0; mtpDownloadTotal = 0
        mtpDownloading = true
        mtpDownloadTask = Task { [weak self] in
            guard let self else { return }
            do {
                try Task.checkCancellation()
                guard operation == mtpOperationID, model == selectedModel, mtpAvailable else { throw CancellationError() }
                let path: String
                if mtpBuiltIn {
                    path = model.path
                } else {
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
                          event.type == "installed", let installedPath = event.path, !installedPath.isEmpty else { throw MLXL3BridgeError.invalidResponse }
                    path = installedPath
                    mtpHeadPath = path
                    preferences.set(path, forKey: "studio.mtpHeadPath")
                    preferences.set(path, forKey: "studio.mtpHead.\(model.path)")
                }
                activateMTP()
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

    private func activateMTP() {
        dflash2Enabled = false
        mtpEnabled = true
        // Greedy request values are derived without changing saved ordinary sampling.
        preferences.set(true, forKey: "studio.mtpEnabled")
        saveMTPSelection()
        schedulePersistence()
    }

    func chooseMTPHead() {
        guard !isGenerating, !mtpDownloading, !mtpBuiltIn else { return }
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
        guard let model = selectedModel, let runtime = mtpTuningKey, !mtpEffectiveHeadPath.isEmpty else { return nil }
        return MTPConfigurationKey(modelPath: model.path, headPath: mtpEffectiveHeadPath, runtime: runtime)
    }

    var canTuneMTP: Bool {
        mtpAvailable && mtpTuneSupported && engineState.isReady && !isGenerating
            && !mtpDownloading && !dflashDownloading && !dflashConfigurationPending && !mcpUpdating && !updateManager.isInstalling && !modelInstallState.isWorking
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
        // Integrated heads retain their per-model choice; external heads use the global switch.
        if mtpBuiltIn { mtpEnabled = saved.depth > 0 }
        mtpTuneRows = saved.rows; mtpTunedAt = saved.tunedAt
    }

    func tuneMTP() {
        guard canTuneMTP, let key = currentMTPConfiguration else { return }
        let request = UUID().uuidString
        tuneIsDFlash = false; tuneRequestID = request; tuneConfiguration = key; tuneCancellationRequested = false
        isTuningMTP = true; mtpTuneProgress = 0; mtpError = nil
        mtpTuneStatus = L("Préparation du test…", "Preparing test…")
        do { try bridge.tuneMTP(requestID: request, headPath: mtpHeadPath, memorySaver: memorySaverEnabled && memorySaverSupported) }
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
        if let error { if tuneIsDFlash { dflashDownloadError = error } else { mtpError = error } }
        tuneIsDFlash = false
    }

    private func handleMTPTuning(_ event: BridgeEvent) {
        switch event.type {
        case "mtp_tune_progress", "dflash_tune_progress":
            guard !tuneCancellationRequested else { return }
            if let completed = event.completed, let total = event.total, total > 0 {
                mtpTuneProgress = min(1, max(0, Double(completed) / Double(total)))
            }
            let mode = event.depth.map { tuneIsDFlash ? ["Baseline", "Auto", "2 tokens", "7 tokens"][min(3, max(0, $0))] : $0 == 0 ? "Baseline" : "MTP\($0)" } ?? "Tune"
            mtpTuneStatus = event.phase == "warmup"
                ? L("Échauffement · ", "Warmup · ") + mode
                : event.phase == "loading_head" ? L("Chargement de la tête MTP…", "Loading MTP head…")
                : L("Mesure · ", "Measuring · ") + mode
        case "mtp_tune_complete", "dflash_tune_complete":
            guard !tuneCancellationRequested, let key = tuneConfiguration,
                  key == (tuneIsDFlash ? currentDFlashConfiguration : currentMTPConfiguration), event.tuningKey == key.runtime,
                  let rows = event.rows, let best = MTPTuning.winner(rows), best == event.bestDepth else {
                finishMTPTuning(error: L("Test incomplet ou annulé. Réglage conservé.", "Incomplete or cancelled test. Setting preserved.")); return
            }
            if tuneIsDFlash {
                dflashMode = max(1, best); dflash2Enabled = best > 0
                dflashTuneRows = rows.sorted { $0.depth < $1.depth }; dflashTunedAt = Date()
                mtpEnabled = false; preferences.set(false, forKey: "studio.mtpEnabled")
                preferences.set(dflash2Enabled, forKey: "studio.dflash2Enabled")
                saveDFlashSelection()
                if best == 0 { try? bridge.setDFlash(requestID: UUID().uuidString, enabled: false, draftPath: "") }
            } else {
                mtpDepth = max(1, best); mtpEnabled = best > 0; mtpTuneRows = rows.sorted { $0.depth < $1.depth }
                mtpTunedAt = Date()
                dflash2Enabled = false; preferences.set(false, forKey: "studio.dflash2Enabled")
                preferences.set(mtpEnabled, forKey: "studio.mtpEnabled"); saveMTPSelection()
            }
            mtpTuneProgress = 1
            // Tuning uses greedy internally; retain the user's ordinary sampling settings.
            schedulePersistence(); finishMTPTuning(error: nil)
        case "cancelled":
            finishMTPTuning(error: L("Test arrêté. Réglage conservé.", "Test stopped. Setting preserved."))
        case "error":
            finishMTPTuning(error: event.message ?? L("Le test MTP a échoué. Réglage conservé.", "MTP test failed. Setting preserved."))
        default: break
        }
    }

    private func commitDFlashDraft(_ path: String, modelPath: String) {
        dflashDraftPath = path
        preferences.set(path, forKey: "studio.dflashDraftPath")
        preferences.set(path, forKey: "studio.dflashDraft.\(modelPath)")
        restoreDFlashSelection()
    }

    func downloadCompatibleDFlashDraft() {
        guard selectedModel?.modelType == "qwen4_exp" else { return }
        setDFlash2Enabled(true, downloadCompatible: true)
    }

    func chooseDFlashDraft() {
        let picker = NSOpenPanel()
        picker.canChooseDirectories = true
        picker.canChooseFiles = false
        picker.allowsMultipleSelection = false
        picker.prompt = L("Utiliser ce draft", "Use this draft")
        guard picker.runModal() == .OK, let path = picker.url?.path else { return }
        dflashDraftPath = path
        if let model = selectedModel { preferences.set(path, forKey: "studio.dflashDraft.\(model.path)") }
        preferences.set(path, forKey: "studio.dflashDraftPath")
        setDFlash2Enabled(true)
    }

    func enableDFlashGreedy() {
        mtpEnabled = false; preferences.set(false, forKey: "studio.mtpEnabled")
        dflash2Enabled = true
        preferences.set(true, forKey: "studio.dflash2Enabled")
        settingsDidChange()
    }

    func selectConversation(_ id: UUID) {
        guard let conversation = conversations.first(where: { $0.id == id }) else { return }
        guard !localServer.active, modeTransition == nil else { return }
        if isGenerating && (currentConversation?.agent == true || conversation.agent) && selectedConversationID != id { return }
        let previousID = selectedConversationID
        let previousAgent = bridgeUsesAgentRegistry
        if let previous = selectedConversationID { drafts[previous] = draft }
        voice.stop()
        selectedConversationID = id
        selectedProjectID = currentConversation?.projectID
        draft = drafts[id] ?? ""
        selectedMode = conversation.agent ? .agent : .chat
        if previousAgent != conversation.agent { loadSelectedModel() }
        else if conversation.agent && previousID != id { refreshAgentWorkspace() }
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
                projects = backup.projects ?? []
                profiles = backup.profiles ?? []
                documents = backup.documents ?? []
                responseSchema = backup.responseSchema ?? ""
                selectedProjectID = currentConversation?.projectID
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
        selectConversation(restored.id)
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
        importWorkspace(from: url)
    }

    func importWorkspace(from url: URL) {
        guard !isGenerating else { return }
        do {
            guard let snapshot = try ConversationStore.load(from: url) else { return }
            let known = Set(conversations.map(\.id))
            var merged = workspaceSnapshot
            let knownProjects = Set(projects.map(\.id))
            merged.projects = projects + (snapshot.projects ?? []).filter { !knownProjects.contains($0.id) }
            let knownProfiles = Set(profiles.map(\.id))
            merged.profiles = profiles + (snapshot.profiles ?? []).filter { !knownProfiles.contains($0.id) }
            let knownDocuments = Set(documents.map(\.id))
            merged.documents = documents + (snapshot.documents ?? []).filter { !knownDocuments.contains($0.id) }
            try ConversationStore.validate(merged)
            projects = merged.projects ?? []
            profiles = merged.profiles ?? []
            documents = merged.documents ?? []
            conversations.insert(contentsOf: snapshot.conversations.filter { !known.contains($0.id) }.map(Conversation.init(snapshot:)), at: 0)
            persistNow()
        } catch { storageError = error.localizedDescription }
    }

    private var conversationIndex: Int? {
        guard let selectedConversationID else { return nil }
        return conversations.firstIndex { $0.id == selectedConversationID }
    }

    private func loadSelectedModel() {
        guard !isGenerating, !localServer.active, modeTransition == nil else { return }
        modelIdleUnloader.cancel()
        cancelDFlashPreparation()
        dflashTuneSupported = false; dflashTuningKey = nil; dflashConfigureSupported = false
        dflash2Enabled = preferences.bool(forKey: "studio.dflash2Enabled")
        dflashDraftPath = selectedModel.flatMap { preferences.string(forKey: "studio.dflashDraft.\($0.path)") }
            ?? (selectedModelName == dflashLegacyModelName ? preferences.string(forKey: "studio.dflashDraftPath") ?? "" : "")
        dflashTuneRows = []; dflashMode = 1
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
        let agentMode = selectedMode == .agent
        let contextLength = savedContextLength
        Task {
            do {
                let registry = try agentMode ? prepareAgentRegistry() : nil
                bridgeUsesAgentRegistry = agentMode
                try await bridge.start(model: selectedModelName, contextLength: contextLength, registry: registry)
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
        if let activeRequestID, event.requestID == activeRequestID { generationLastActivity = .now }
        if dflashConfigurationPending && event.type == "error" && (event.requestID == nil || event.requestID == "") {
            dflashDownloadError = event.message ?? L("La configuration DFlash a échoué.", "DFlash configuration failed.")
            dflash2Enabled = false; dflashActive = nil
            cancelDFlashPreparation()
        }
        if event.requestID == dflashOperationID.uuidString && ["dflash_status", "error"].contains(event.type) {
            dflashConfigurationPending = false
            dflashActive = event.type == "dflash_status" ? event.dflashActive : false
            if event.type == "error" { dflashDownloadError = event.message }
            else if event.dflashActive != dflashDownloading {
                dflashDownloadError = L("Réponse de configuration DFlash invalide.", "Invalid DFlash configuration response.")
            }
            if !dflashDownloading { dflashDownloadTask?.cancel(); dflashDownloadTask = nil }
            return
        }
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
            memorySaverSupported = event.memorySaverSupported == true
            workspaceFeaturesAvailable = event.workspaceFeatures == true
            if let tools = event.mcpToolCatalog { mcpTools = tools }
            dflashSupported = event.dflashSupported
            dflashTuneSupported = event.dflashTuneSupported == true
            dflashConfigureSupported = event.dflashConfigureSupported == true
            dflashTuningKey = event.dflashTuningKey
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
            if mtpBuiltIn { mtpEnabled = false }
            restoreMTPSelection()
            restoreDFlashSelection()
            mcpServerCount = event.mcpServers ?? 0
            mcpToolCount = event.mcpTools ?? 0
            mcpErrors = event.mcpErrors ?? [:]
            engineState = .ready(
                model: info.model,
                modules: info.modules,
                residentGB: info.residentGB
            )
            if mcpEnabled || bridgeUsesAgentRegistry { updateMCPConnection() }
            if dflash2Enabled && dflash2Available { setDFlash2Enabled(true) }
            else if mtpEnabled && mtpAvailable { setMTPEnabled(true) }
        case "mcp_status":
            if let tools = event.mcpToolCatalog { mcpTools = tools }
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
            if event.contextFull != true, let stats = event.stats, let sample = labConfiguration?(stats), sample.valid {
                labSamples.append(sample)
                labSamples = Array(labSamples.suffix(1000))
            }
            labConfiguration = nil
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
            } else if generationStartedAt != nil && event.stopReason == "max_tokens" {
                message?.fail(L("Limite de 4 096 tokens atteinte. La réponse partielle est conservée ; continue dans un nouveau tour.", "4,096-token Agent limit reached. Partial output was kept; continue in another turn."))
            }
            activeRequestID = nil
            activeResponseID = nil
            schedulePersistence()
            restoreReadyState()
        case "cancelled":
            guard event.requestID == activeRequestID else { return }
            labConfiguration = nil
            activeMessage()?.fail(cancellationReason ?? L("Génération arrêtée", "Generation stopped"))
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
        labConfiguration = nil
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
        var snapshot = WorkspaceSnapshot(
            selectedConversationID: selectedConversationID,
            selectedModelName: selectedModelName,
            conversations: conversations.map(\.snapshot),
            temperature: temperature,
            topK: topK,
            repetitionPenalty: repetitionPenalty,
            systemPrompt: systemPrompt,
            projects: projects,
            profiles: profiles,
            documents: documents,
            responseSchema: responseSchema
        )
        snapshot.labSamples = labSamples
        return snapshot
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

extension StudioModel {
    func selectMode(_ mode: StudioMode) {
        guard !isGenerating, modeTransition == nil, mode != selectedMode else { return }
        voice.stop()
        if mode == .server || mode == .lab { selectedMode = mode; return }
        if localServer.active {
            localServer.stop()
            modeTransition = Task {
                while localServer.active && !Task.isCancelled { try? await Task.sleep(for: .milliseconds(50)) }
                modeTransition = nil
                guard !Task.isCancelled else { return }
                selectMode(mode)
                loadSelectedModel()
            }
            return
        }
        selectedMode = mode
        if let conversation = conversations.first(where: { $0.agent == (mode == .agent) && $0.projectID == selectedProjectID }) {
            selectConversation(conversation.id)
        } else {
            newConversation()
            if mode != .agent { loadSelectedModel() }
        }
    }

    var agentWorkspaceURL: URL? {
        guard let id = selectedConversationID else { return nil }
        return conversationFileURL.deletingLastPathComponent().resolvingSymlinksInPath()
            .appending(path: "agent-workspaces/\(id.uuidString)/files", directoryHint: .isDirectory)
    }

    func openAgentWorkspace() {
        guard let url = agentWorkspaceURL else { return }
        do {
            try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
            NSWorkspace.shared.open(url)
        } catch { featureError = error.localizedDescription }
    }

    private func prepareAgentRegistry() throws -> URL {
        guard let root = agentWorkspaceURL, let model = selectedModel,
              FileManager.default.isExecutableFile(atPath: "/usr/bin/sandbox-exec") else { throw MLXL3BridgeError.invalidResponse }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        guard root.standardizedFileURL.path == root.resolvingSymlinksInPath().path else { throw CocoaError(.fileReadNoPermission) }
        _ = try AgentSandbox.profile(root: root)
        let metadata = conversationFileURL.deletingLastPathComponent().appending(path: "agent-runtime", directoryHint: .isDirectory)
        try FileManager.default.createDirectory(at: metadata, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let entry = try JSONSerialization.jsonObject(with: JSONEncoder().encode(model))
        let registry = metadata.appending(path: "models.json")
        try JSONSerialization.data(withJSONObject: ["version": 1, "models": [model.name: entry]])
            .write(to: registry, options: .atomic)
        let config: [String: Any] = ["version": 1, "mcpServers": [
            "exa": ["enabled": false, "url": "https://mcp.exa.ai/mcp"],
            "sandbox": ["enabled": true, "command": Bundle.main.executableURL?.path ?? CommandLine.arguments[0],
                "args": ["--agent-tools", root.path]]]]
        try JSONSerialization.data(withJSONObject: config).write(to: metadata.appending(path: "mcp.json"), options: .atomic)
        return registry
    }

    private func refreshAgentWorkspace() {
        if engineState.isReady && bridge.isRunning && bridgeUsesAgentRegistry {
            do { _ = try prepareAgentRegistry(); updateMCPConnection() }
            catch { mcpToolCount = 0; mcpErrors = ["sandbox": error.localizedDescription] }
        } else { loadSelectedModel() }
    }

    func startServer(port: Int) {
        guard !isGenerating, modeTransition == nil, !localServer.active, let name = selectedModelName,
              (1024...65535).contains(port) else { featureError = L("Choisissez un modèle et un port entre 1024 et 65535.", "Choose a model and a port between 1024 and 65535."); return }
        selectedMode = .server
        modeTransition = Task {
            await bridge.stopAndWait()
            ejectModel()
            modeTransition = nil
            guard !Task.isCancelled else { return }
            localServer.start(model: name, port: port, context: savedContextLength)
        }
    }

    func stopServices() {
        generationWatchdog?.cancel(); generationWatchdog = nil
        cancellationFallback?.cancel(); cancellationFallback = nil
        modeTransition?.cancel(); modeTransition = nil
        localServer.stop()
        bridge.stop()
    }

    func exportLab() {
        let panel = NSSavePanel(); panel.nameFieldStringValue = "MLXL3-Lab.json"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]; encoder.dateEncodingStrategy = .iso8601
            try encoder.encode(labSamples).write(to: url, options: .atomic)
        } catch { featureError = error.localizedDescription }
    }

    var currentProject: WorkspaceProject? { projects.first { $0.id == selectedProjectID } }
    var voiceLocale: String { language == .fr ? "fr-FR" : "en-US" }

    func chooseProject(_ id: UUID?) {
        guard !isGenerating else { return }
        voice.stop()
        selectedProjectID = id
        if let conversation = conversations.first(where: { $0.projectID == id && $0.agent == (selectedMode == .agent) }) { selectConversation(conversation.id) }
        else { newConversation() }
        if let profileID = currentProject?.profileID, let profile = profiles.first(where: { $0.id == profileID }) { applyProfile(profile) }
    }

    func addProject(_ name: String) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty, name.count <= 100, projects.count < 256, !isGenerating else { return }
        let project = WorkspaceProject(name: name)
        projects.append(project)
        chooseProject(project.id)
        schedulePersistence()
    }

    func deleteProject(_ id: UUID) {
        guard !isGenerating else { return }
        projects.removeAll { $0.id == id }
        documents.removeAll { $0.projectID == id }
        for index in conversations.indices where conversations[index].projectID == id { conversations[index].projectID = nil }
        if selectedProjectID == id { chooseProject(nil) }
        schedulePersistence()
    }

    func updateProject(_ project: WorkspaceProject) {
        guard !isGenerating, let index = projects.firstIndex(where: { $0.id == project.id }),
            !project.name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, project.name.count <= 100 else { return }
        projects[index] = project
        schedulePersistence()
    }

    func saveProfile(_ name: String) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isGenerating, !name.isEmpty, name.count <= 100, profiles.count < 128 else { return }
        profiles.append(GenerationProfile(name: name, model: selectedModelName, systemPrompt: systemPrompt,
            temperature: temperature, topK: topK, repetitionPenalty: repetitionPenalty,
            contextLength: savedContextLength, responseSchema: responseSchema))
        schedulePersistence()
    }

    func applyProfile(_ profile: GenerationProfile) {
        guard !isGenerating, profile.temperature.isFinite, (0...2).contains(profile.temperature),
            (0...10000).contains(profile.topK), profile.repetitionPenalty.isFinite, (0.1...4).contains(profile.repetitionPenalty), profile.contextLength >= 0 else {
            featureError = L("Profil invalide ou génération en cours.", "Invalid profile or generation in progress."); return
        }
        systemPrompt = profile.systemPrompt
        responseSchema = profile.responseSchema
        temperature = profile.temperature; topK = profile.topK; repetitionPenalty = profile.repetitionPenalty
        if let name = profile.model, models.contains(where: { $0.name == name }) {
            let previousContext = savedContextLength
            var lengths = preferences.dictionary(forKey: "studio.contextLengths") ?? [:]
            lengths[name] = profile.contextLength
            preferences.set(lengths, forKey: "studio.contextLengths")
            if name != selectedModelName { selectModel(name) }
            else if previousContext != profile.contextLength { contextLengthDraft = profile.contextLength; loadSelectedModel() }
        }
        if mtpEnabled { setMTPEnabled(false) }
        schedulePersistence()
    }

    func deleteProfile(_ id: UUID) {
        guard !isGenerating else { return }
        profiles.removeAll { $0.id == id }
        for index in projects.indices where projects[index].profileID == id { projects[index].profileID = nil }
        schedulePersistence()
    }

    @discardableResult
    func branchConversation(at messageID: UUID, includeMessage: Bool = true) -> Bool {
        guard !isGenerating, let original = currentConversation,
            let index = original.messages.firstIndex(where: { $0.id == messageID }) else { return false }
        var branch = Conversation(title: original.title + L(" · branche", " · branch"))
        branch.agent = original.agent
        branch.projectID = original.projectID
        branch.parentConversationID = original.id
        branch.messages = original.messages.prefix(index + (includeMessage ? 1 : 0)).compactMap { ChatMessage(snapshot: $0.snapshot) }
        conversations.insert(branch, at: 0)
        selectConversation(branch.id)
        schedulePersistence()
        return true
    }

    func editMessage(_ id: UUID) {
        guard let message = currentConversation?.messages.first(where: { $0.id == id }), message.role == .user,
            branchConversation(at: id, includeMessage: false), let conversation = selectedConversationID else { return }
        draft = message.content
        attachmentDrafts[conversation] = message.attachments
        drafts[conversation] = draft
    }

    func regenerate(_ id: UUID) {
        guard !isGenerating, engineState.isReady, let messages = currentConversation?.messages,
            let index = messages.firstIndex(where: { $0.id == id }), messages[index].role == .assistant,
            let user = messages[..<index].last(where: { $0.role == .user }) else { return }
        editMessage(user.id)
        send()
    }

    func exportMarkdown() {
        guard let conversation = currentConversation else { return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "MLXL3-conversation.md"
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do { try Data(MarkdownExport.render(conversation.snapshot).utf8).write(to: url, options: .atomic) }
        catch { featureError = error.localizedDescription }
    }

    func chooseLibraryFiles(folder: Bool) {
        guard currentProject != nil, !libraryImporting else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = folder; panel.canChooseFiles = !folder
        panel.allowsMultipleSelection = !folder
        if !folder { panel.allowedContentTypes = ChatAttachmentImporter.contentTypes }
        guard panel.runModal() == .OK else { return }
        Task { await importLibraryFiles(panel.urls) }
    }

    func importLibraryFiles(_ urls: [URL]) async {
        guard let projectID = selectedProjectID, !libraryImporting else { return }
        libraryImporting = true
        defer { libraryImporting = false }
        let existing = documents
        let result = await Task.detached(priority: .userInitiated) {
            var imported: [LibraryDocument] = []
            var failures: [String] = []
            var paths: [URL] = []
            for url in urls {
                let accessed = url.startAccessingSecurityScopedResource()
                defer { if accessed { url.stopAccessingSecurityScopedResource() } }
                if (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
                    let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey], options: [.skipsHiddenFiles, .skipsPackageDescendants])
                    guard enumerator != nil else { failures.append(url.lastPathComponent + L(" : dossier illisible.", ": folder could not be read.")); continue }
                    var visited = 0
                    while let file = enumerator?.nextObject() as? URL {
                        visited += 1
                        if visited > 4000 { failures.append(L("Dossier limité à 4000 entrées.", "Folder limited to 4000 entries.")); break }
                        let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                        if values?.isSymbolicLink == true { enumerator?.skipDescendants(); continue }
                        if values?.isRegularFile == true { paths.append(file) }
                    }
                } else { paths.append(url) }
                var total = existing.reduce(0) { $0 + $1.text.utf8.count } + imported.reduce(0) { $0 + $1.text.utf8.count }
                for file in paths.sorted(by: { $0.path < $1.path }) {
                    guard existing.count + imported.count < DocumentSearch.maximumDocuments else { failures.append(L("Limite de 128 documents atteinte.", "128 document limit reached.")); break }
                    let result = ChatAttachmentImporter.load([file], existing: [])
                    if let attachment = result.attachments.first {
                        if existing.contains(where: { $0.projectID == projectID && $0.originalPath == file.path && $0.text == attachment.text }) || imported.contains(where: { $0.originalPath == file.path }) { continue }
                        guard attachment.text.utf8.count <= DocumentSearch.maximumLibraryBytes - total else { failures.append(L("Bibliothèque limitée à 16 Mio de texte.", "Library limited to 16 MiB of text.")); break }
                        imported.append(LibraryDocument(projectID: projectID, fileName: attachment.fileName, originalPath: file.path, text: attachment.text))
                        total += attachment.text.utf8.count
                    } else { failures += result.errors }
                }
                paths.removeAll()
            }
            return (imported, Array(failures.prefix(10)))
        }.value
        guard projects.contains(where: { $0.id == projectID }) else { return }
        documents += result.0
        if !result.1.isEmpty { featureError = result.1.joined(separator: "\n") }
        persistNow()
    }

    func removeLibraryDocument(_ id: UUID) {
        documents.removeAll { $0.id == id }
        schedulePersistence()
    }

    func startDictation() {
        guard !isGenerating else { return }
        let conversation = selectedConversationID
        let prefix = draft
        Task { [weak self] in
            guard let self else { return }
            await voice.start(locale: voiceLocale) { [weak self] text in
                guard let self, self.selectedConversationID == conversation, !self.isGenerating else { return }
                self.draft = prefix + (prefix.isEmpty ? "" : " ") + text
            }
        }
    }
}
