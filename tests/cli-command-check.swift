// Run: swiftc -parse-as-library apps/MLXL3Studio/Sources/MLXL3Studio/CLICommand.swift tests/cli-command-check.swift -o /tmp/mlxl3-cli-check && /tmp/mlxl3-cli-check "$PWD/.venv/bin/python"
import Foundation

enum CLIResolver {
    static func executable() -> URL? { URL(fileURLWithPath: CommandLine.arguments[1]) }
    static func workingDirectory(for executable: URL) -> URL { FileManager.default.temporaryDirectory }
}
enum MLXL3BridgeError: Error { case executableNotFound, invalidResponse, commandFailed(String) }

@main struct Check {
    @MainActor static var receivedProgress = false
    static func main() async throws {
        let large = try await CLICommand().output(["-c", "print('x' * 200000)"])
        precondition(large.count == 200001, "Output larger than a pipe must not deadlock or truncate")
        let last = try await CLICommand().output(["-c", "print('progress'); print('installed')"], onLine: { _ in })
        precondition(String(decoding: last, as: UTF8.self) == "installed")
        let streaming = Task {
            try await CLICommand().output(["-c", "import time; print('progress', flush=True); time.sleep(3); print('installed')"], onLine: { line in
                if String(decoding: line, as: UTF8.self) == "progress" { receivedProgress = true }
            })
        }
        try await Task.sleep(for: .seconds(1))
        let arrivedBeforeExit = receivedProgress
        _ = try await streaming.value
        guard arrivedBeforeExit else { throw MLXL3BridgeError.commandFailed("Progress was buffered until process exit") }
        let job = Task { try await CLICommand().output(["-c", "import time; time.sleep(30)"]) }
        try await Task.sleep(for: .milliseconds(200))
        job.cancel()
        do { _ = try await job.value; preconditionFailure("Cancellation was ignored") }
        catch is CancellationError { }
        do { _ = try await CLICommand().output(["-c", "raise RuntimeError('fixture error')"]); preconditionFailure("Failure was ignored") }
        catch MLXL3BridgeError.commandFailed(let error) { precondition(error.contains("fixture error")) }
        for (text, expected) in [("final", "final"), ("progress\nfinal", "final"), ("progress\nfinal\n", "final")] {
            var lines: [String] = []
            let result = try await CLICommand().output(["-c", "import sys; sys.stdout.write(sys.argv[1])", text]) {
                lines.append(String(decoding: $0, as: UTF8.self))
            }
            precondition(String(decoding: result, as: UTF8.self) == expected,
                         "Final event must survive EOF without a newline")
            // CLI completion must follow the last progress callback.
            precondition(lines == text.split(separator: "\n").map(String.init), "Lost or reordered a progress event")
        }
        do {
            _ = try await CLICommand().output(["-c", "import sys; sys.stderr.write('X'*10_000_000+'end'); sys.exit(1)"])
            preconditionFailure("Failure with a large diagnostic was ignored")
        } catch MLXL3BridgeError.commandFailed(let message) {
            precondition(message.utf8.count == 8000 && message.hasSuffix("end"))
        }
        print("CLI transport passed: large output, NDJSON, cancellation, error reporting")
    }
}
