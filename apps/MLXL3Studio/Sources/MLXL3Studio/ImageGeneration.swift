import AppKit
import ImageIO
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

enum ImageModelAvailability: Equatable {
    case checking
    case missing(Int64)
    case installed(URL)
    case downloading(completed: Int64, total: Int64)
    case paused(Int64)
    case failed(String, Int64)
}

struct ImageEditReference: Sendable {
    static let size = 512
    static let maximumInputBytes = 32 * 1024 * 1024

    let name: String
    let png: Data

    static func load(_ url: URL) throws -> Self {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        func unsupported() -> NSError {
            NSError(domain: "MLXL3.ImageGeneration", code: 2, userInfo: [
                NSLocalizedDescriptionKey: L("Choisis une image lisible de 32 Mio maximum.",
                                              "Choose a readable image up to 32 MiB.")
            ])
        }
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard url.isFileURL,
              let info = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              info.isRegularFile == true, let count = info.fileSize,
              count > 0, count <= maximumInputBytes else { throw unsupported() }
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let bytes = try handle.read(upToCount: maximumInputBytes + 1),
              bytes.count <= maximumInputBytes,
              let source = CGImageSourceCreateWithData(bytes as CFData, options),
              let type = CGImageSourceGetType(source) as String?,
              UTType(type)?.conforms(to: .image) == true,
              CGImageSourceGetStatusAtIndex(source, 0) == .statusComplete else { throw unsupported() }
        let thumbnail = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                         kCGImageSourceCreateThumbnailWithTransform: true,
                         kCGImageSourceThumbnailMaxPixelSize: size] as CFDictionary
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, thumbnail) else { throw unsupported() }
        let png = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(png, UTType.png.identifier as CFString, 1, nil) else {
            throw unsupported()
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw unsupported() }
        return Self(name: url.lastPathComponent, png: png as Data)
    }
}

struct GeneratedImageOutput: Sendable {
    let id: UUID
    let png: Data
    let savedURL: URL?
    let seed: UInt32
    let isEdit: Bool

    func export(to destination: URL) async throws {
        try await Task.detached(priority: .userInitiated) {
            try png.write(to: destination, options: .atomic)
        }.value
    }
}

@MainActor
final class ImageGeneration: ObservableObject {
    static let autoSavePreferenceKey = "image21AutoSaveGeneratedImages"
    typealias Transfer = @Sendable ([ImageModelDownloadFile], URL, @escaping @Sendable (Int64, Int64) async -> Void) async throws -> Void

    @Published var prompt = ""
    @Published var size = 512
    @Published var baseSteps = 40
    var steps: Int { variant == .turbo ? 8 : baseSteps }
    @Published private(set) var variant: Image21Variant
    @Published private(set) var modelAvailability: ImageModelAvailability = .checking
    @Published private(set) var folderError: String?
    var modelDirectory: URL? {
        if case let .installed(url) = modelAvailability { return url }
        return nil
    }
    var isDownloading: Bool {
        if case .downloading = modelAvailability { return true }
        return false
    }
    @Published private(set) var referenceImage: ImageEditReference?
    @Published private(set) var referenceImporting = false
    @Published private(set) var referenceError: String?
    @Published private var waitingForReference = false
    private var referenceRequest = UUID()
    var editing: Bool { waitingForReference || referenceImage != nil }
    @Published var seed = UInt32.random(in: 0...UInt32.max)
    @Published private(set) var running = false
    @Published private(set) var finalizing = false
    @Published private(set) var status = ""
    @Published private(set) var result: GeneratedImageOutput?
    @Published private(set) var error: String?
    @Published private(set) var history: [GeneratedImageRecord] = []
    @Published private(set) var isLoadingHistory = false
    @Published private(set) var historyError: String?
    private let historyStore: GeneratedImageHistory
    let usageStore: ImageGenerationUsage
    @Published private(set) var usageRevision = 0
    private var historyRevision = 0
    private var historyReadID: UUID?
    private var task: Task<Void, Never>?
    private var process: Process?

    private let supported: Bool
    private let worker: (launcher: URL, script: URL)?
    private let preferences: UserDefaults
    private let catalog: ImageModelCatalog
    private let download: Transfer
    private var downloadTask: Task<Void, Never>?
    private var modelLookup = UUID()

    init(supported: Bool = Image21Hardware.available, historyDirectory: URL = GeneratedImageHistory.defaultDirectory,
         worker: (launcher: URL, script: URL)? = nil, preferences: UserDefaults = .standard,
         catalog: ImageModelCatalog = ImageModelCatalog(),
         download: @escaping Transfer = { files, destination, progress in
             try await ImageModelDownload().install(files: files, to: destination, progress: progress)
         }) {
        self.supported = supported
        self.historyStore = GeneratedImageHistory(directory: historyDirectory)
        self.usageStore = ImageGenerationUsage(historyDirectory: historyDirectory)
        self.worker = worker
        self.preferences = preferences
        self.catalog = catalog
        self.download = download
        variant = preferences.string(forKey: "image21Variant").flatMap(Image21Variant.init(rawValue:)) ?? .turbo
    }

    private var folderKey: String { variant == .base ? "image21ModelDirectory" : "image21TurboModelDirectory" }

    private var preferredFolder: URL? {
        guard let path = preferences.string(forKey: folderKey), !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path, isDirectory: true)
    }

    func refreshModelDirectory() async {
        guard !isDownloading else { return }
        let request = UUID()
        modelLookup = request
        let previous = modelAvailability
        let preferred = preferredFolder
        let selection = variant
        do {
            let (directory, bytes, partial) = try await Task.detached(priority: .userInitiated) { [catalog] in
                let package = try catalog.package(for: selection)
                let directory = catalog.installedDirectory(for: selection, preferred: preferred)
                return (directory, package.totalBytes, directory == nil && catalog.hasPartialDownload(for: selection))
            }.value
            guard request == modelLookup else { return }
            if let directory { modelAvailability = .installed(directory) }
            else if case .paused = previous { modelAvailability = .paused(bytes) }
            else if case let .failed(message, _) = previous { modelAvailability = .failed(message, bytes) }
            else { modelAvailability = partial ? .paused(bytes) : .missing(bytes) }
        } catch {
            if request == modelLookup { modelAvailability = .failed(error.localizedDescription, 0) }
        }
    }

    func selectVariant(_ selection: Image21Variant) {
        guard !running, !isDownloading, selection != variant else { return }
        variant = selection
        preferences.set(selection.rawValue, forKey: "image21Variant")
        modelLookup = UUID()
        modelAvailability = .checking
        folderError = nil
        error = nil
        status = ""
    }

    func selectModelDirectory(_ directory: URL) async {
        guard !running, !isDownloading, modelAvailability != .checking else { return }
        let request = UUID()
        modelLookup = request
        let selection = variant
        do {
            let installed = try await Task.detached(priority: .userInitiated) { [catalog] in
                try catalog.package(for: selection).installedFolder(in: directory)
            }.value
            guard request == modelLookup else { return }
            guard let installed else {
                folderError = L("Choisis le dossier complet de la version sélectionnée.",
                                "Choose the complete folder for the selected variant.")
                return
            }
            preferences.set(installed.path, forKey: folderKey)
            modelAvailability = .installed(installed)
            folderError = nil
            error = nil
        } catch {
            if request == modelLookup { folderError = error.localizedDescription }
        }
    }

    func downloadModel() {
        guard !running, !isDownloading, modelDirectory == nil, modelAvailability != .checking else { return }
        guard supported else {
            modelAvailability = .failed(Image21Hardware.requirement, 0)
            return
        }
        let selection = variant
        let preferred = preferredFolder
        modelLookup = UUID()
        folderError = nil
        modelAvailability = .downloading(completed: 0, total: 0)
        downloadTask = Task {
            var total: Int64 = 0
            defer { downloadTask = nil }
            do {
                let (package, installed) = try await Task.detached(priority: .utility) { [catalog] in
                    let package = try catalog.package(for: selection)
                    return (package, catalog.installedDirectory(for: selection, preferred: preferred))
                }.value
                total = package.totalBytes
                if let installed {
                    modelAvailability = .installed(installed)
                    return
                }
                try Task.checkCancellation()
                let destination = catalog.destination(for: selection)
                modelAvailability = .downloading(completed: 0, total: total)
                try await download(package.files, destination) { completed, expected in
                    await self.updateDownload(completed, expected)
                }
                let complete = await Task.detached(priority: .utility) { package.isInstalled(at: destination) }.value
                guard complete else { throw failure(L("Le modèle téléchargé est incomplet.", "The downloaded model is incomplete.")) }
                modelAvailability = .installed(destination)
            } catch {
                modelAvailability = Task.isCancelled ? .paused(total) : .failed(error.localizedDescription, total)
            }
        }
    }

    private func updateDownload(_ completed: Int64, _ total: Int64) {
        if isDownloading { modelAvailability = .downloading(completed: completed, total: total) }
    }

    func cancelModelDownload() { downloadTask?.cancel() }

    func refreshHistory() async {
        let request = UUID()
        historyReadID = request
        isLoadingHistory = true
        historyError = nil
        defer { if historyReadID == request { isLoadingHistory = false } }
        do {
            while !Task.isCancelled && historyReadID == request {
                let revision = historyRevision
                let scan = Task.detached(priority: .utility) { [historyStore] in try historyStore.records() }
                let records = try await withTaskCancellationHandler {
                    try await scan.value
                } onCancel: { scan.cancel() }
                guard historyReadID == request, !Task.isCancelled else { return }
                // An in-flight scan can predate an image that just finished.
                if revision == historyRevision { history = records; return }
            }
        } catch {
            if historyReadID == request && !Task.isCancelled { historyError = error.localizedDescription }
        }
    }

    func importReference(_ url: URL) async {
        guard !running, !isDownloading, variant == .turbo else { return }
        let request = UUID()
        referenceRequest = request
        referenceImporting = true
        referenceError = nil
        defer { if referenceRequest == request { referenceImporting = false } }
        do {
            let image = try await Task.detached(priority: .userInitiated) { try ImageEditReference.load(url) }.value
            guard referenceRequest == request, !Task.isCancelled, variant == .turbo else { return }
            referenceImage = image
            waitingForReference = false
        } catch {
            if referenceRequest == request && !Task.isCancelled { referenceError = error.localizedDescription }
        }
    }

    func removeReference() {
        guard !running else { return }
        referenceRequest = UUID()
        referenceImporting = false
        referenceImage = nil
        referenceError = nil
        waitingForReference = false
    }

    func requestEditReference() {
        guard !running else { return }
        removeReference()
        waitingForReference = true
    }

    func start(model: URL, prompt: String, size: Int, steps: Int, seed: UInt32) {
        guard !running, !isDownloading, !referenceImporting else { return }
        let variant = self.variant
        let reference = referenceImage
        let isEdit = editing
        let size = isEdit ? ImageEditReference.size : size
        let autoSave = preferences.bool(forKey: Self.autoSavePreferenceKey)
        running = true
        // An unsuccessful retry should not discard the only copy of an unsaved image.
        if result?.savedURL != nil { result = nil }
        error = nil
        status = L("Préparation du moteur…", "Preparing the engine…")
        task = Task {
            defer { finalizing = false; running = false; task = nil; process = nil }
            do {
                guard supported else { throw failure(Image21Hardware.requirement) }
                if isEdit {
                    guard variant == .turbo else { throw failure(L("La retouche utilise Turbo.", "Editing uses Turbo.")) }
                    guard reference != nil else { throw failure(L("Ajoute l’image à retoucher.", "Add the image to edit.")) }
                }
                guard variant.stepRange.contains(steps) else {
                    if variant == .turbo { throw failure(L("Turbo utilise 8 étapes.", "Turbo uses 8 steps.")) }
                    throw failure(L("Choisis entre 2 et 50 étapes.", "Choose between 2 and 50 steps."))
                }
                let fm = FileManager.default
                let manifest = variant == .turbo ? "quantization.json" : "manifest.json"
                guard fm.fileExists(atPath: model.appendingPathComponent(manifest).path) else {
                    if variant == .turbo {
                        throw failure(L("Choisis le dossier EXL3 Turbo 4 bits contenant quantization.json.",
                                        "Choose the Turbo 4-bit EXL3 folder containing quantization.json."))
                    }
                    throw failure(L("Choisis le dossier EXL3 contenant manifest.json.", "Choose the EXL3 folder containing manifest.json."))
                }
                let home = fm.homeDirectoryForCurrentUser
                let paths = [home.appendingPathComponent(".local/bin/uv").path, "/opt/homebrew/bin/uv", "/usr/local/bin/uv"]
                guard let uv = worker?.launcher ?? paths.first(where: fm.isExecutableFile(atPath:)).map({ URL(fileURLWithPath: $0) }) else {
                    throw failure(L("Installe uv pour préparer le moteur image (docs.astral.sh/uv).", "Install uv to set up the image engine (docs.astral.sh/uv)."))
                }
                guard let script = worker?.script ?? Bundle.main.url(forResource: "render", withExtension: "py", subdirectory: "Image21") else {
                    throw failure(L("Le composant Image21 manque dans l’application.", "The Image21 component is missing from the app."))
                }
                let work = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
                try fm.createDirectory(at: work, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
                defer { try? fm.removeItem(at: work) }
                let log = work.appendingPathComponent("worker.log")
                let interpreter = work.appendingPathComponent("interpreter.txt")
                try await run(uv,
                              ["run", "--no-project", "--python", "3.12", "--script", script.path, "--interpreter"],
                              log: log, stdout: interpreter, timeout: 1800)
                let python = try String(contentsOf: interpreter, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
                guard python.hasPrefix("/"), fm.isExecutableFile(atPath: python) else {
                    throw failure(L("Interpréteur Python introuvable.", "Python interpreter could not be found."))
                }
                status = L("Chargement du modèle…", "Loading the model…")
                let png = work.appendingPathComponent("image.png")
                var arguments = ["-I", "-u", script.path, "--variant", variant.rawValue, "--model", model.path, "--prompt=\(prompt)",
                                 "--size", String(size), "--steps", String(steps), "--seed", String(seed), "--output", png.path]
                if let reference {
                    let source = work.appendingPathComponent("reference.png")
                    try reference.png.write(to: source, options: .atomic)
                    arguments += ["--reference-image", source.path]
                }
                try await run(URL(fileURLWithPath: python), arguments, log: log, stdout: log, timeout: 3600, steps: steps)
                try Task.checkCancellation()
                let verified = try await Task.detached(priority: .userInitiated) { [historyStore] in
                    try historyStore.validatedPNG(at: png, size: size)
                }.value
                try Task.checkCancellation()
                // Once accepted, count the completed render even if publishing its PNG fails.
                finalizing = true
                status = L("Finalisation de l’image…", "Finishing the image…")
                let id = UUID()
                let createdAt = Date.now
                let (record, counted, warnings) = await Task.detached(priority: .userInitiated) { [historyStore, usageStore] in
                    var record: GeneratedImageRecord?
                    var warnings: [String] = []
                    if autoSave {
                        do {
                            record = try historyStore.save(verifiedPNG: verified, prompt: prompt, modelName: model.lastPathComponent,
                                                           variant: variant, steps: steps, seed: seed, id: id, createdAt: createdAt,
                                                           isEdit: isEdit)
                        } catch {
                            warnings.append(L("L’enregistrement automatique a échoué.", "Automatic saving failed.") + " " + error.localizedDescription)
                        }
                    }
                    do {
                        try usageStore.record(id: id, variant: variant, at: createdAt)
                        return (record, true, warnings)
                    } catch {
                        warnings.append(L("Le compteur d’images n’a pas été mis à jour.", "The image count could not be updated.") + " " + error.localizedDescription)
                        return (record, false, warnings)
                    }
                }.value
                if let record {
                    historyRevision += 1
                    let position = history.firstIndex { GeneratedImageRecord.newestFirst(record, $0) } ?? history.endIndex
                    history.insert(record, at: position)
                }
                if counted || record != nil { usageRevision += 1 }
                result = GeneratedImageOutput(id: id, png: verified.data, savedURL: record?.fileURL, seed: seed, isEdit: isEdit)
                if !warnings.isEmpty { self.error = warnings.joined(separator: "\n") }
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
        guard !finalizing else { return }
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
            if trimmed.hasPrefix("Loading edit encoder") {
                return L("Préparation de l’encodeur d’images…", "Preparing the image encoder…")
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
