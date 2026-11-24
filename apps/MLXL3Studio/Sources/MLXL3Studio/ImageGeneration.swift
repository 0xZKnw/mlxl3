import AppKit
import Metal
import SwiftUI
import UniformTypeIdentifiers

enum Image21Hardware {
    static func supports(gpu: String?, os: OperatingSystemVersion, disabled: Bool) -> Bool {
        guard !disabled, os.majorVersion > 26 || (os.majorVersion == 26 && os.minorVersion >= 3),
              let gpu else { return false }
        return gpu == "Apple M5" || gpu.hasPrefix("Apple M5 ")
    }

    static let available = supports(gpu: MTLCreateSystemDefaultDevice()?.name,
                                    os: ProcessInfo.processInfo.operatingSystemVersion,
                                    disabled: ProcessInfo.processInfo.environment["MLXL3_DISABLE_TENSOR_OPS"] != nil)
    static var requirement: String {
        L("Qwen Image nécessite un GPU Apple M5 et macOS 26.3 ou ultérieur avec TensorOps activé.",
          "Qwen Image requires an Apple M5 GPU and macOS 26.3 or later with TensorOps enabled.")
    }
}

@MainActor
final class ImageGeneration: ObservableObject {
    @Published var prompt = ""
    @Published var size = 512
    @Published var steps = 40
    @Published var seed = UInt32.random(in: 0...UInt32.max)
    @Published private(set) var running = false
    @Published private(set) var status = ""
    @Published private(set) var result: URL?
    @Published private(set) var error: String?
    private var task: Task<Void, Never>?
    private var process: Process?

    private let supported: Bool

    init(supported: Bool = Image21Hardware.available) {
        self.supported = supported
    }

    func start(model: URL, prompt: String, size: Int, steps: Int, seed: UInt32) {
        guard !running else { return }
        running = true
        result = nil
        error = nil
        status = L("Préparation du moteur…", "Preparing the engine…")
        task = Task {
            defer { running = false; task = nil; process = nil }
            do {
                guard supported else { throw failure(Image21Hardware.requirement) }
                guard (2...50).contains(steps) else {
                    throw failure(L("Choisis entre 2 et 50 étapes.", "Choose between 2 and 50 steps."))
                }
                let fm = FileManager.default
                guard fm.fileExists(atPath: model.appendingPathComponent("manifest.json").path) else {
                    throw failure(L("Choisis le dossier EXL3 contenant manifest.json.", "Choose the EXL3 folder containing manifest.json."))
                }
                let home = fm.homeDirectoryForCurrentUser
                let paths = [home.appendingPathComponent(".local/bin/uv").path, "/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
                guard let uv = paths.first(where: fm.isExecutableFile(atPath:)) else {
                    throw failure(L("Installe uv pour préparer le moteur image (docs.astral.sh/uv).", "Install uv to set up the image engine (docs.astral.sh/uv)."))
                }
                guard let script = Bundle.main.url(forResource: "render", withExtension: "py", subdirectory: "Image21") else {
                    throw failure(L("Le composant Image21 manque dans l’application.", "The Image21 component is missing from the app."))
                }
                let work = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
                try fm.createDirectory(at: work, withIntermediateDirectories: true)
                defer { try? fm.removeItem(at: work) }
                let log = work.appendingPathComponent("worker.log")
                let interpreter = work.appendingPathComponent("interpreter.txt")
                try await run(URL(fileURLWithPath: uv),
                              ["run", "--no-project", "--python", "3.12", "--script", script.path, "--interpreter"],
                              log: log, stdout: interpreter, timeout: 1800)
                let python = try String(contentsOf: interpreter, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
                guard python.hasPrefix("/"), fm.isExecutableFile(atPath: python) else {
                    throw failure(L("Interpréteur Python introuvable.", "Python interpreter could not be found."))
                }
                status = L("Chargement du modèle…", "Loading the model…")
                let png = work.appendingPathComponent("image.png")
                try await run(URL(fileURLWithPath: python),
                              ["-I", "-u", script.path, "--model", model.path, "--prompt", prompt,
                               "--size", String(size), "--steps", String(steps), "--seed", String(seed), "--output", png.path],
                              log: log, stdout: log, timeout: 3600, steps: steps)
                try Task.checkCancellation()
                guard let image = NSBitmapImageRep(data: try Data(contentsOf: png)), image.pixelsWide == size, image.pixelsHigh == size else {
                    throw failure(L("Le moteur a produit une image illisible.", "The engine returned an unreadable image."))
                }
                let directory = try fm.url(for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
                    .appendingPathComponent("io.mlxl3.desktop/Generated Images", isDirectory: true)
                try fm.createDirectory(at: directory, withIntermediateDirectories: true)
                let destination = directory.appendingPathComponent("Image-\(seed)-\(UUID().uuidString).png")
                try fm.moveItem(at: png, to: destination)
                result = destination
                status = ""
            } catch is CancellationError {
                status = L("Génération annulée.", "Generation cancelled.")
            } catch {
                self.error = error.localizedDescription
                status = ""
            }
        }
    }

    func cancel() {
        task?.cancel()
        if let process, process.isRunning { process.terminate() }
    }

    private func failure(_ message: String) -> NSError {
        NSError(domain: "MLXL3.ImageGeneration", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func tail(_ url: URL) -> String {
        guard let file = try? FileHandle(forReadingFrom: url) else { return "" }
        defer { try? file.close() }
        let end = (try? file.seekToEnd()) ?? 0
        try? file.seek(toOffset: end > 4096 ? end - 4096 : 0)
        let data = (try? file.readToEnd()) ?? Data()
        return String(decoding: data, as: UTF8.self)
    }

    static func progressMessage(in log: String, steps: Int) -> String? {
        for line in log.split(whereSeparator: \.isNewline).reversed() {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let bar = trimmed.prefixMatch(of: /^\d{1,3}%\|[^|]*\|\s*(\d+)\/(\d+)\s/),
               Int(bar.2) == steps, let done = Int(bar.1), (0...steps).contains(done) {
                return done == steps
                    ? L("Finalisation de l’image…", "Finishing the image…")
                    : L("Étape \(done) sur \(steps)", "Step \(done) of \(steps)")
            }
            if trimmed.hasPrefix("Loading tokenizer") {
                return L("Chargement de l’encodeur de texte et du VAE…", "Loading the text encoder and VAE…")
            }
        }
        return nil
    }

    private func run(_ executable: URL, _ arguments: [String], log: URL, stdout: URL, timeout: Int, steps: Int? = nil) async throws {
        try Task.checkCancellation()
        let fm = FileManager.default
        guard fm.createFile(atPath: log.path, contents: nil) else { throw failure("Cannot create worker log") }
        let errors = try FileHandle(forWritingTo: log)
        defer { try? errors.close() }
        if stdout != log { fm.createFile(atPath: stdout.path, contents: nil) }
        let output = stdout == log ? errors : try FileHandle(forWritingTo: stdout)
        defer { if stdout != log { try? output.close() } }
        let child = Process()
        child.executableURL = executable
        child.arguments = arguments
        child.standardError = errors
        child.standardOutput = output
        child.standardInput = FileHandle.nullDevice
        process = child
        try child.run()
        let deadline = ContinuousClock.now.advanced(by: .seconds(timeout))
        do {
            while child.isRunning {
                try await Task.sleep(for: .milliseconds(400))
                guard ContinuousClock.now < deadline else {
                    throw failure(L("Le moteur image a dépassé le délai.", "The image engine timed out."))
                }
                if let steps, let message = Self.progressMessage(in: tail(log), steps: steps) {
                    status = message
                }
            }
            try Task.checkCancellation()
        } catch {
            if child.isRunning { child.terminate() }
            // Reap the worker before another generation can start.
            for _ in 0..<20 where child.isRunning {
                await withCheckedContinuation { continuation in
                    DispatchQueue.global().asyncAfter(deadline: .now() + 0.1) { continuation.resume() }
                }
            }
            if child.isRunning { kill(child.processIdentifier, SIGKILL) }
            child.waitUntilExit()
            throw error
        }
        guard child.terminationStatus == 0 else {
            let detail = tail(log).split(whereSeparator: \.isNewline).suffix(3).joined(separator: "\n")
            throw failure(detail.isEmpty ? L("Le moteur image s’est arrêté.", "The image engine stopped.") : detail)
        }
    }
}
