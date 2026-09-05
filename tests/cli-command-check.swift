// Run: swiftc -parse-as-library apps/MLXL3Studio/Sources/MLXL3Studio/CLICommand.swift tests/cli-command-check.swift -o /tmp/mlxl3-cli-check && /tmp/mlxl3-cli-check "$PWD/.venv/bin/python"
import Foundation

enum CLIResolver {
    static func executable() -> URL? { URL(fileURLWithPath: CommandLine.arguments[1]) }
    static func workingDirectory(for executable: URL) -> URL { FileManager.default.temporaryDirectory }
}
enum MLXL3BridgeError: Error { case executableNotFound, invalidResponse, commandFailed(String) }

@main struct Check {
    static func main() async throws {
        let large = try await CLICommand().output(["-c", "print('x' * 200000)"])
        precondition(large.count == 200001, "Output larger than a pipe must not deadlock or truncate")
        let last = try await CLICommand().output(["-c", "print('progress'); print('installed')"], onLine: { _ in })
        precondition(String(decoding: last, as: UTF8.self) == "installed")
        let job = Task { try await CLICommand().output(["-c", "import time; time.sleep(30)"]) }
        try await Task.sleep(for: .milliseconds(200))
        job.cancel()
        do { _ = try await job.value; preconditionFailure("Cancellation was ignored") }
        catch is CancellationError { }
        do { _ = try await CLICommand().output(["-c", "raise RuntimeError('fixture error')"]); preconditionFailure("Failure was ignored") }
        catch MLXL3BridgeError.commandFailed(let error) { precondition(error.contains("fixture error")) }
        print("CLI transport passed: large output, NDJSON, cancellation, error reporting")
    }
}
