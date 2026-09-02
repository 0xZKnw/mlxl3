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
    @Published var temperature = 0.2
    @Published var topK = 80
    @Published var repetitionPenalty = 1.05
    @Published var systemPrompt = ""

    private let bridge = MLXL3Bridge()
    private var didStart = false
    private var activeRequestID: String?
    private var activeResponseID: UUID?
    private var readyInfo: (model: String, modules: Int, residentGB: Double)?

    init() {
        selectedConversationID = conversations.first?.id
        bridge.onEvent = { [weak self] event in self?.handle(event) }
        bridge.onExit = { [weak self] message in
            guard let self, let message, !message.isEmpty else { return }
            self.engineState = .failed(message)
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
        engineState.isReady && !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
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

    func start() {
        guard !didStart else { return }
        didStart = true
        refreshModels()
    }

    func refreshModels() {
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

    func selectModel(_ name: String) {
        if name == selectedModelName, bridge.isRunning { return }
        selectedModelName = name
        loadSelectedModel()
    }

    func ejectModel() {
        guard canEject else { return }
        if let location = activeMessageLocation() {
            conversations[location.conversation].messages[location.message].isStreaming = false
            conversations[location.conversation].messages[location.message].error = "Modèle éjecté"
        }
        activeRequestID = nil
        activeResponseID = nil
        readyInfo = nil
        bridge.stop()
        engineState = .idle
    }

    func newConversation() {
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        selectedConversationID = conversation.id
        draft = ""
    }

    func deleteConversation(_ id: UUID) {
        guard conversations.count > 1 else {
            if let index = conversations.firstIndex(where: { $0.id == id }) {
                conversations[index] = Conversation(id: id)
            }
            return
        }
        conversations.removeAll { $0.id == id }
        if selectedConversationID == id {
            selectedConversationID = conversations.first?.id
        }
    }

    func send() {
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, engineState.isReady, let conversationIndex else { return }

        draft = ""
        if conversations[conversationIndex].messages.isEmpty {
            conversations[conversationIndex].title = title(for: text)
        }
        conversations[conversationIndex].messages.append(
            ChatMessage(role: .user, content: text)
        )
        let assistant = ChatMessage(role: .assistant, content: "", isStreaming: true)
        conversations[conversationIndex].messages.append(assistant)

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
            return PromptMessage(role: message.role.rawValue, content: message.content)
        }

        do {
            try bridge.generate(
                GenerationRequest(
                    requestID: requestID,
                    messages: messages,
                    maxTokens: -1,
                    temperature: temperature,
                    topK: topK,
                    repetitionPenalty: repetitionPenalty
                )
            )
        } catch {
            failActiveTurn(error.localizedDescription)
        }
    }

    func stopGeneration() {
        guard isGenerating else { return }
        if let location = activeMessageLocation() {
            conversations[location.conversation].messages[location.message].isStreaming = false
            conversations[location.conversation].messages[location.message].error = "Génération arrêtée"
        }
        activeRequestID = nil
        activeResponseID = nil
        bridge.stop()
        loadSelectedModel()
    }

    private var conversationIndex: Int? {
        guard let selectedConversationID else { return nil }
        return conversations.firstIndex { $0.id == selectedConversationID }
    }

    private func loadSelectedModel() {
        guard let selectedModelName else { return }
        engineState = .loading(selectedModelName)
        readyInfo = nil
        activeRequestID = nil
        activeResponseID = nil
        do {
            try bridge.start(model: selectedModelName)
        } catch {
            engineState = .failed(error.localizedDescription)
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
            engineState = .ready(
                model: info.model,
                modules: info.modules,
                residentGB: info.residentGB
            )
        case "delta":
            guard event.requestID == activeRequestID,
                  let text = event.text,
                  let location = activeMessageLocation()
            else { return }
            if event.phase == "thinking" {
                conversations[location.conversation].messages[location.message].thinking += text
            } else {
                conversations[location.conversation].messages[location.message].content += text
            }
        case "complete":
            guard event.requestID == activeRequestID,
                  let location = activeMessageLocation()
            else { return }
            var message = conversations[location.conversation].messages[location.message]
            message.isStreaming = false
            message.stats = event.stats
            if message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
               let answer = event.assistantContext {
                message.content = answer
            }
            conversations[location.conversation].messages[location.message] = message
            activeRequestID = nil
            activeResponseID = nil
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
        if let location = activeMessageLocation() {
            conversations[location.conversation].messages[location.message].isStreaming = false
            conversations[location.conversation].messages[location.message].error = message
        }
        activeRequestID = nil
        activeResponseID = nil
        if readyInfo == nil {
            engineState = .failed(message)
        } else {
            restoreReadyState()
        }
    }

    private func activeMessageLocation() -> (conversation: Int, message: Int)? {
        guard let activeResponseID else { return nil }
        for conversation in conversations.indices {
            if let message = conversations[conversation].messages.firstIndex(
                where: { $0.id == activeResponseID }
            ) {
                return (conversation, message)
            }
        }
        return nil
    }

    private func title(for prompt: String) -> String {
        let words = prompt.split(whereSeparator: \.isWhitespace)
        let title = words.prefix(7).joined(separator: " ")
        return title.count > 46 ? String(title.prefix(46)) + "…" : title
    }
}
