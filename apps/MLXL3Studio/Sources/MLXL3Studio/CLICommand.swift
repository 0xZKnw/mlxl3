@preconcurrency import Foundation

/// Cancellable short-lived catalogue/download commands; never touches the model bridge.
final class CLICommand: @unchecked Sendable {
    private let process = Process()
    private let lock = NSLock()
    private var cancelled = false

    func cancel() {
        lock.lock()
        defer { lock.unlock() }
        cancelled = true
        if process.isRunning { process.terminate() }
    }

    private func launch() throws {
        lock.lock()
        defer { lock.unlock() }
        if cancelled { throw CancellationError() }
        try process.run()
    }

    func output(_ arguments: [String], onLine: (@MainActor @Sendable (Data) -> Void)? = nil) async throws -> Data {
        try await withTaskCancellationHandler {
            try Task.checkCancellation()
            return try await withCheckedThrowingContinuation { continuation in
                DispatchQueue.global(qos: .userInitiated).async { [self] in
                    let errors = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-command-\(UUID()).log")
                    FileManager.default.createFile(atPath: errors.path, contents: nil)
                    defer { try? FileManager.default.removeItem(at: errors) }
                    do {
                        guard let executable = CLIResolver.executable() else { throw MLXL3BridgeError.executableNotFound }
                        let errorHandle = try FileHandle(forWritingTo: errors)
                        defer { try? errorHandle.close() }
                        let pipe = Pipe()
                        process.executableURL = executable
                        process.arguments = arguments
                        process.currentDirectoryURL = CLIResolver.workingDirectory(for: executable)
                        process.standardOutput = pipe
                        process.standardError = errorHandle
                        try launch()
                        var result = Data()
                        var pending = Data()
                        // Drain before waiting: README responses can exceed a pipe's capacity.
                        while let chunk = try pipe.fileHandleForReading.read(upToCount: 16384), !chunk.isEmpty {
                            if let onLine {
                                pending.append(chunk)
                                while let newline = pending.firstIndex(of: 10) {
                                    let line = Data(pending[..<newline])
                                    pending.removeSubrange(...newline)
                                    result = line
                                    Task { @MainActor in onLine(line) }
                                }
                            } else {
                                result.append(chunk)
                            }
                            if result.count + pending.count > 8_000_000 {
                                cancel()
                                throw MLXL3BridgeError.invalidResponse
                            }
                        }
                        process.waitUntilExit()
                        lock.lock()
                        let stopped = cancelled
                        lock.unlock()
                        if stopped { throw CancellationError() }
                        if process.terminationStatus != 0 {
                            let data = (try? Data(contentsOf: errors)) ?? Data()
                            throw MLXL3BridgeError.commandFailed(String(decoding: data.suffix(8000), as: UTF8.self))
                        }
                        continuation.resume(returning: result)
                    } catch {
                        cancel()
                        continuation.resume(throwing: error)
                    }
                }
            }
        } onCancel: { self.cancel() }
    }
}
