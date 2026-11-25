import Combine
import Foundation

/// Two fresh bridge sessions. Only the active model is resident, and neither request
/// participates in a Studio conversation, its tools, or the usage ledger.
@MainActor
final class LabComparison: ObservableObject {
    enum Phase { case waiting, loading, generating, complete, failed, cancelled }

    struct Run: Identifiable {
        let model: LocalModel
        var id: String { model.id }
        var phase: Phase = .waiting
        var answer = ""
        var firstTextSeconds: Double?
        var decodeTps: Double?
        var elapsedSeconds: Double?
        var generatedTokens: Int?
        var error: String?
        var hitTokenLimit = false
        var hitContextLimit = false
        var isThinking = false
    }

    @Published private(set) var runs: [Run] = []
    @Published private(set) var isRunning = false
    private(set) var prompt = ""
    private(set) var budget = 0

    private let bridge = MLXL3Bridge()
    private let idleTimeout: Duration
    private var task: Task<Void, Never>?

    init(idleTimeout: Duration = .seconds(300)) {
        self.idleTimeout = idleTimeout
    }

    static func validPrompt(_ text: String) -> Bool {
        !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && text.utf8.count <= 65_536
    }

    func start(_ first: LocalModel, _ second: LocalModel, prompt: String, budget: Int,
               prepare: @escaping @MainActor () async -> Void) {
        guard !isRunning, first.id != second.id, Self.validPrompt(prompt), (1...4096).contains(budget) else { return }
        self.prompt = prompt
        self.budget = budget
        runs = [Run(model: first), Run(model: second)]
        isRunning = true
        task = Task {
            await prepare()
            for index in runs.indices {
                guard !Task.isCancelled else { break }
                do { try await run(index) }
                catch is CancellationError { runs[index].phase = .cancelled }
                catch {
                    runs[index].phase = .failed
                    runs[index].error = error.localizedDescription
                }
                runs[index].isThinking = false
                // A pending SIGTERM is insufficient, particularly for large checkpoints.
                await bridge.stopAndWait()
            }
            if Task.isCancelled {
                for index in runs.indices where runs[index].phase == .waiting { runs[index].phase = .cancelled }
            }
            task = nil
            isRunning = false
        }
    }

    func cancel() {
        task?.cancel()
        bridge.stop()
    }

    func waitUntilStopped() async { await task?.value }

    private enum Signal {
        case event(BridgeEvent)
        case failure(String)
    }

    private func run(_ index: Int) async throws {
        runs[index].phase = .loading
        let (stream, channel) = AsyncStream<Signal>.makeStream()
        bridge.onEvent = { channel.yield(.event($0)) }
        bridge.onExit = { channel.yield(.failure($0 ?? L("Le moteur s’est arrêté.", "The engine stopped."))) }
        let deadline = ContinuousClock.now.advanced(by: .seconds(3600))
        var activity = ContinuousClock.now
        let watchdog = Task {
            while !Task.isCancelled {
                let limit = min(deadline, activity.advanced(by: idleTimeout))
                do { try await Task.sleep(until: limit, clock: .continuous) } catch { return }
                let now = ContinuousClock.now
                if now >= deadline || activity.duration(to: now) >= idleTimeout {
                    channel.yield(.failure(L("Le moteur a dépassé le délai de réponse.", "The engine timed out.")))
                    return
                }
            }
        }
        defer {
            watchdog.cancel()
            bridge.onEvent = nil
            bridge.onExit = nil
            channel.finish()
        }
        try await bridge.start(model: runs[index].model.name)
        let requestID = UUID().uuidString
        var started: ContinuousClock.Instant?
        for await signal in stream {
            try Task.checkCancellation()
            switch signal {
            case .failure(let reason): throw MLXL3BridgeError.commandFailed(reason)
            case .event(let event):
                if let id = event.requestID, id != requestID { continue }
                activity = .now
                if event.type == "error" {
                    throw MLXL3BridgeError.commandFailed(event.message ?? L("Erreur du moteur.", "Engine error."))
                }
                if event.type == "ready", started == nil {
                    let request = GenerationRequest(requestID: requestID, conversationID: UUID().uuidString,
                        messages: [PromptMessage(role: "user", content: prompt)], maxTokens: budget,
                        temperature: 0, topK: 0, repetitionPenalty: 1, dspark: false)
                    started = .now
                    try bridge.generate(request) { channel.yield(.failure($0)) }
                    runs[index].phase = .generating
                    continue
                }
                guard event.requestID == requestID, let started else { continue }
                let elapsed = started.duration(to: .now).components
                let seconds = Double(elapsed.seconds) + Double(elapsed.attoseconds) / 1e18
                switch event.type {
                case "delta":
                    guard let text = event.text, !text.isEmpty else { break }
                    if event.phase == "thinking" {
                        if runs[index].firstTextSeconds == nil, !runs[index].isThinking { runs[index].isThinking = true }
                    } else {
                        if runs[index].firstTextSeconds == nil, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            runs[index].firstTextSeconds = seconds
                            runs[index].isThinking = false
                        }
                        runs[index].answer += text
                    }
                case "complete":
                    runs[index].elapsedSeconds = seconds
                    runs[index].hitTokenLimit = event.stopReason == "max_tokens"
                    runs[index].hitContextLimit = event.stopReason == "context_full" ||
                        (event.stopReason == nil && event.contextFull == true)
                    if let stats = event.stats {
                        runs[index].decodeTps = LabDecodeUsage.measurement(stats)?.average
                        runs[index].generatedTokens = stats.generatedTokens >= 0 ? stats.generatedTokens : nil
                    }
                    if runs[index].firstTextSeconds == nil, let answer = event.assistantContext,
                       !answer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        runs[index].answer = answer
                        runs[index].firstTextSeconds = seconds
                    }
                    if runs[index].firstTextSeconds == nil {
                        runs[index].answer = ""
                        if event.contextFull == true || runs[index].hitContextLimit {
                            throw MLXL3BridgeError.commandFailed(
                                L("Le contexte du modèle est plein. Essaie un prompt plus court.",
                                  "The model’s context is full. Try a shorter prompt."))
                        }
                        guard runs[index].hitTokenLimit else {
                            throw MLXL3BridgeError.commandFailed(
                                L("Le modèle n’a produit aucune réponse visible.", "The model returned no visible answer."))
                        }
                    }
                    runs[index].phase = .complete
                    return
                case "cancelled":
                    if Task.isCancelled { throw CancellationError() }
                    throw MLXL3BridgeError.commandFailed(L("Le moteur a interrompu la réponse.", "The engine interrupted the response."))
                default: break
                }
            }
        }
        throw CancellationError()
    }
}
