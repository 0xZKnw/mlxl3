import Darwin
@preconcurrency import Foundation

enum MLXL3BridgeError: LocalizedError {
    case executableNotFound
    case commandFailed(String)
    case invalidResponse

    var errorDescription: String? {
        switch self {
        case .executableNotFound:
            "Runtime MLXL3 introuvable. Réinstalle MLXL3 Desktop depuis le DMG."
        case let .commandFailed(message):
            message
        case .invalidResponse:
            "La commande mlxl3 a renvoyé une réponse illisible."
        }
    }
}

enum CLIResolver {
    static func executable() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        var candidates: [String] = []
        if let override = environment["MLXL3_EXECUTABLE"], !override.isEmpty {
            candidates.append((override as NSString).expandingTildeInPath)
        }

        if let resources = Bundle.main.resourceURL {
            candidates.append(resources.appending(path: "runtime/mlxl3").path)
        }

        var sourceProbe = URL(fileURLWithPath: #filePath)
        for _ in 0..<8 {
            sourceProbe.deleteLastPathComponent()
            candidates.append(sourceProbe.appending(path: ".venv/bin/mlxl3").path)
        }

        let home = FileManager.default.homeDirectoryForCurrentUser.path
        candidates += [
            FileManager.default.currentDirectoryPath + "/.venv/bin/mlxl3",
            home + "/.local/bin/mlxl3",
            "/opt/homebrew/bin/mlxl3",
            "/usr/local/bin/mlxl3",
        ]

        if let path = environment["PATH"] {
            candidates += path.split(separator: ":").map { String($0) + "/mlxl3" }
        }
        return candidates.first { FileManager.default.isExecutableFile(atPath: $0) }
            .map { URL(fileURLWithPath: $0) }
    }

    static func workingDirectory(for executable: URL) -> URL {
        let bin = executable.deletingLastPathComponent()
        let environment = bin.deletingLastPathComponent()
        if environment.lastPathComponent == ".venv" {
            return environment.deletingLastPathComponent()
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }
}

private struct PendingDelta {
    let requestID: String?
    let phase: String?
    var text: String
}

final class MLXL3Bridge: @unchecked Sendable {
    var onEvent: (@MainActor (BridgeEvent) -> Void)?
    var onExit: (@MainActor (String?) -> Void)?

    private let ioQueue = DispatchQueue(label: "com.mlxl3.studio.bridge.io")
    private var process: Process?
    private var inputPipe: Pipe?
    private var outputBuffer = Data()
    private var stderrBuffer = Data()
    private var pendingDelta: PendingDelta?
    private var deltaFlushWorkItem: DispatchWorkItem?
    private let displayFlushInterval = 0.05

    var isRunning: Bool { process?.isRunning == true }

    func residentMemoryBytes() -> UInt64 {
        interfaceResidentMemoryBytes() + engineResidentMemoryBytes()
    }

    func interfaceResidentMemoryBytes() -> UInt64 {
        residentMemoryBytes(for: getpid())
    }

    func engineResidentMemoryBytes() -> UInt64 {
        guard let process, process.isRunning else { return 0 }
        return residentMemoryBytes(for: process.processIdentifier)
    }

    static func listModels(
        completion: @escaping @MainActor @Sendable (Result<[LocalModel], Error>) -> Void
    ) {
        execute(["list", "--json"]) { result in
            completion(
                result.flatMap { data in
                    do {
                        return .success(try JSONDecoder().decode([LocalModel].self, from: data))
                    } catch {
                        return .failure(error)
                    }
                }
            )
        }
    }

    static func registerModel(
        name: String,
        path: URL,
        completion: @escaping @MainActor @Sendable (Result<Void, Error>) -> Void
    ) {
        execute(["register", name, path.path, "--force"]) { result in
            completion(result.map { _ in () })
        }
    }

    static func downloadModel(
        repo: String,
        revision: String?,
        name: String?,
        completion: @escaping @MainActor @Sendable (Result<LocalModel, Error>) -> Void
    ) {
        var arguments = ["download", repo, "--json"]
        if let revision, !revision.isEmpty {
            arguments += ["--revision", revision]
        }
        if let name, !name.isEmpty {
            arguments += ["--name", name]
        }
        execute(arguments) { result in
            completion(
                result.flatMap { data in
                    do {
                        return .success(try JSONDecoder().decode(LocalModel.self, from: data))
                    } catch {
                        return .failure(error)
                    }
                }
            )
        }
    }

    private static func execute(
        _ arguments: [String],
        completion: @escaping @MainActor @Sendable (Result<Data, Error>) -> Void
    ) {
        guard let executable = CLIResolver.executable() else {
            DispatchQueue.main.async {
                completion(.failure(MLXL3BridgeError.executableNotFound))
            }
            return
        }
        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            let output = Pipe()
            let errorURL = FileManager.default.temporaryDirectory
                .appending(path: "mlxl3-command-\(UUID().uuidString).log")
            FileManager.default.createFile(atPath: errorURL.path, contents: nil)
            process.executableURL = executable
            process.arguments = arguments
            process.currentDirectoryURL = CLIResolver.workingDirectory(for: executable)
            process.standardOutput = output
            do {
                let errorHandle = try FileHandle(forWritingTo: errorURL)
                process.standardError = errorHandle
                try process.run()
                process.waitUntilExit()
                try? errorHandle.close()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                if process.terminationStatus != 0 {
                    let details = String(
                        data: (try? Data(contentsOf: errorURL)) ?? Data(),
                        encoding: .utf8
                    ) ?? "Échec de la commande MLXL3"
                    throw MLXL3BridgeError.commandFailed(
                        details.trimmingCharacters(in: .whitespacesAndNewlines)
                    )
                }
                DispatchQueue.main.async { completion(.success(data)) }
            } catch {
                DispatchQueue.main.async { completion(.failure(error)) }
            }
            try? FileManager.default.removeItem(at: errorURL)
        }
    }

    func start(model: String) throws {
        stop()
        guard let executable = CLIResolver.executable() else {
            throw MLXL3BridgeError.executableNotFound
        }

        let process = Process()
        let input = Pipe()
        let output = Pipe()
        let error = Pipe()
        ioQueue.sync {
            outputBuffer.removeAll(keepingCapacity: true)
            stderrBuffer.removeAll(keepingCapacity: true)
            pendingDelta = nil
            deltaFlushWorkItem?.cancel()
            deltaFlushWorkItem = nil
        }
        process.executableURL = executable
        process.arguments = ["bridge", model]
        // Group shared KV heads for Gemma d=512 decode; other cases retain
        // the reference path. Preserve explicit overrides for comparisons.
        var environment = ProcessInfo.processInfo.environment
        if environment["MLXL3_GEMMA_SDPA512"] == nil {
            environment["MLXL3_GEMMA_SDPA512"] = "grouped"
        }
        process.environment = environment
        process.currentDirectoryURL = CLIResolver.workingDirectory(for: executable)
        process.standardInput = input
        process.standardOutput = output
        process.standardError = error

        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let bridge = self else { return }
            bridge.ioQueue.async { bridge.consumeOutput(data) }
        }
        error.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let bridge = self else { return }
            bridge.ioQueue.async { bridge.stderrBuffer.append(data) }
        }
        process.terminationHandler = { [weak self] terminatedProcess in
            guard let self else { return }
            self.ioQueue.async {
                guard let currentProcess = self.process,
                      currentProcess === terminatedProcess
                else { return }
                output.fileHandleForReading.readabilityHandler = nil
                error.fileHandleForReading.readabilityHandler = nil
                self.flushPendingDelta()
                let details = String(data: self.stderrBuffer, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                let message = details?.isEmpty == false
                    ? details
                    : "Le moteur MLXL3 s’est arrêté de façon inattendue."
                self.process = nil
                self.inputPipe = nil
                DispatchQueue.main.async { self.onExit?(message) }
            }
        }

        try process.run()
        self.process = process
        inputPipe = input
    }

    func generate(_ request: GenerationRequest) throws {
        guard let process, process.isRunning, let inputPipe else {
            throw MLXL3BridgeError.commandFailed("Le moteur MLXL3 n’est pas prêt.")
        }
        var data = try JSONEncoder().encode(request)
        data.append(0x0A)
        try inputPipe.fileHandleForWriting.write(contentsOf: data)
    }

    func cancelGeneration() -> Bool {
        guard let process, process.isRunning else { return false }
        return Darwin.kill(process.processIdentifier, SIGUSR1) == 0
    }

    func stop() {
        inputPipe?.fileHandleForWriting.closeFile()
        if process?.isRunning == true {
            process?.terminate()
        }
        inputPipe = nil
        process = nil
    }

    private func consumeOutput(_ data: Data) {
        outputBuffer.append(data)
        let newline = Data([0x0A])
        while let range = outputBuffer.range(of: newline) {
            let line = outputBuffer.subdata(in: outputBuffer.startIndex..<range.lowerBound)
            outputBuffer.removeSubrange(outputBuffer.startIndex...range.lowerBound)
            guard !line.isEmpty else { continue }
            do {
                let event = try JSONDecoder().decode(BridgeEvent.self, from: line)
                queueForDisplay(event)
            } catch {
                let raw = String(data: line, encoding: .utf8) ?? ""
                let event = BridgeEvent(
                    type: "error",
                    model: nil,
                    modules: nil,
                    loadSeconds: nil,
                    residentGB: nil,
                    requestID: nil,
                    phase: nil,
                    text: nil,
                    assistantContext: nil,
                    cacheContext: nil,
                    stats: nil,
                    message: "Réponse moteur invalide: \(raw)",
                    mcpServers: nil,
                    mcpTools: nil,
                    mcpErrors: nil,
                    toolCallID: nil,
                    toolName: nil,
                    serverName: nil,
                    isError: nil
                )
                queueForDisplay(event)
            }
        }
    }

    private func queueForDisplay(_ event: BridgeEvent) {
        guard event.type == "delta", let text = event.text else {
            flushPendingDelta()
            dispatchToMain(event)
            return
        }

        if pendingDelta?.requestID == event.requestID, pendingDelta?.phase == event.phase {
            pendingDelta?.text += text
        } else {
            flushPendingDelta()
            pendingDelta = PendingDelta(
                requestID: event.requestID,
                phase: event.phase,
                text: text
            )
        }
        scheduleDeltaFlush()
    }

    private func scheduleDeltaFlush() {
        guard deltaFlushWorkItem == nil else { return }
        let workItem = DispatchWorkItem { [weak self] in
            self?.flushPendingDelta()
        }
        deltaFlushWorkItem = workItem
        ioQueue.asyncAfter(
            deadline: .now() + displayFlushInterval,
            execute: workItem
        )
    }

    private func flushPendingDelta() {
        deltaFlushWorkItem?.cancel()
        deltaFlushWorkItem = nil
        guard let pendingDelta else { return }
        self.pendingDelta = nil
        dispatchToMain(
            BridgeEvent(
                type: "delta",
                model: nil,
                modules: nil,
                loadSeconds: nil,
                residentGB: nil,
                requestID: pendingDelta.requestID,
                phase: pendingDelta.phase,
                text: pendingDelta.text,
                assistantContext: nil,
                cacheContext: nil,
                stats: nil,
                message: nil,
                mcpServers: nil,
                mcpTools: nil,
                mcpErrors: nil,
                toolCallID: nil,
                toolName: nil,
                serverName: nil,
                isError: nil
            )
        )
    }

    private func dispatchToMain(_ event: BridgeEvent) {
        DispatchQueue.main.async { [weak self] in self?.onEvent?(event) }
    }

    private func residentMemoryBytes(for pid: pid_t) -> UInt64 {
        var info = proc_taskinfo()
        let size = MemoryLayout<proc_taskinfo>.stride
        let read = withUnsafeMutablePointer(to: &info) { pointer in
            proc_pidinfo(pid, PROC_PIDTASKINFO, 0, pointer, Int32(size))
        }
        return read == size ? info.pti_resident_size : 0
    }
}
