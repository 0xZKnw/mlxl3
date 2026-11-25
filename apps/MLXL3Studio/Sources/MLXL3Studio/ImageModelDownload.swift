import CryptoKit
import Darwin
import Foundation
import Synchronization

struct ImageModelDownloadFile: Sendable {
    let filename: String
    let url: URL
    let size: Int64
    let sha256: String
}

private final class ImageFileProgress: NSObject, URLSessionDownloadDelegate {
    let expected: Int64
    let samples: AsyncStream<Int64>.Continuation

    let target: URL
    private let completion = Mutex<CheckedContinuation<Void, any Error>?>(nil)

    init(expected: Int64, samples: AsyncStream<Int64>.Continuation, target: URL) {
        self.expected = expected
        self.samples = samples
        self.target = target
    }

    func transfer(_ request: URLRequest, session: URLSession) async throws {
        let task = session.downloadTask(with: request)
        task.delegate = self
        let started = Mutex(false)
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                completion.withLock { $0 = continuation }
                let resumed = started.withLock { started in
                    guard !Task.isCancelled else { return false }
                    started = true
                    task.resume()
                    return true
                }
                if !resumed {
                    finish(.failure(CancellationError()))
                    task.cancel()
                }
            }
        } onCancel: {
            started.withLock { if $0 { task.cancel() } }
        }
    }

    private func finish(_ result: Result<Void, any Error>) {
        let pending = completion.withLock { value in
            let pending = value
            value = nil
            return pending
        }
        pending?.resume(with: result)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesWritten <= expected else { downloadTask.cancel(); return }
        samples.yield(totalBytesWritten)
    }

    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        do {
            guard let response = downloadTask.response as? HTTPURLResponse, response.statusCode == 200,
                  response.url?.scheme == "https" else { throw URLError(.badServerResponse) }
            // Foundation unlinks the download as its delegate returns.
            try FileManager.default.moveItem(at: location, to: target)
            finish(.success(()))
        } catch { finish(.failure(error)) }
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        finish(.failure(error ?? URLError(.badServerResponse)))
    }

    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping @Sendable (URLRequest?) -> Void) {
        completionHandler(request.url?.scheme == "https" ? request : nil)
    }
}

struct ImageModelDownload: Sendable {
    private let session: URLSession

    private static let defaultSession: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.httpShouldSetCookies = false
        configuration.timeoutIntervalForRequest = 60
        configuration.timeoutIntervalForResource = 24 * 3600
        return URLSession(configuration: configuration)
    }()

    init(session: URLSession? = nil) {
        self.session = session ?? Self.defaultSession
    }

    private func failure(_ message: String) -> NSError {
        NSError(domain: "MLXL3.ImageModelDownload", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func totalSize(of files: [ImageModelDownloadFile]) throws -> Int64 {
        var names = Set<String>()
        var total: Int64 = 0
        for file in files {
            let (sum, overflow) = total.addingReportingOverflow(file.size)
            guard !file.filename.isEmpty, !file.filename.hasPrefix("."), !file.filename.contains("/"),
                  !file.filename.contains("\\"), !file.filename.utf8.contains(0),
                  names.insert(file.filename.lowercased()).inserted, file.url.scheme == "https",
                  file.url.host != nil, file.size > 0, !overflow,
                  file.sha256.utf8.count == 64, file.sha256.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else {
                throw failure(L("Catalogue de téléchargement invalide.", "Invalid download catalog."))
            }
            total = sum
        }
        guard total > 0 else { throw failure(L("Catalogue vide.", "Empty download catalog.")) }
        return total
    }

    static func stagingDirectory(for destination: URL) -> URL {
        destination.deletingLastPathComponent().appendingPathComponent("." + destination.lastPathComponent + ".partial", isDirectory: true)
    }

    static func matches(_ url: URL, _ file: ImageModelDownloadFile) throws -> Bool {
        // Follow links in user installations (including Hugging Face snapshots) only for reading.
        let resolved = url.resolvingSymlinksInPath()
        guard FileManager.default.fileExists(atPath: resolved.path) else { return false }
        let info = try resolved.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard info.isRegularFile == true, let size = info.fileSize, Int64(size) == file.size else { return false }
        let handle = try FileHandle(forReadingFrom: resolved)
        defer { try? handle.close() }
        var hash = SHA256()
        while let block = try handle.read(upToCount: 1024 * 1024), !block.isEmpty {
            try Task.checkCancellation()
            hash.update(data: block)
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined() == file.sha256
    }

    private func fetch(_ file: ImageModelDownloadFile, into stage: URL,
                       report: @escaping @Sendable (Int64) async -> Void) async throws {
        var request = URLRequest(url: file.url, cachePolicy: .reloadIgnoringLocalCacheData)
        request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
        let temporary = stage.deletingLastPathComponent().appendingPathComponent(".incoming")
        defer { _ = unlink(temporary.path) }
        let (stream, samples) = AsyncStream<Int64>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let delegate = ImageFileProgress(expected: file.size, samples: samples, target: temporary)
        let observer = Task {
            for await count in stream { await report(count) }
        }
        let result: Result<Void, any Error>
        do {
            try await delegate.transfer(request, session: session)
            result = .success(())
        } catch { result = .failure(error) }
        samples.finish()
        await observer.value
        try Task.checkCancellation()
        try result.get()
        guard try Self.matches(temporary, file) else {
            throw failure(L("Le fichier \(file.filename) est incomplet ou endommagé.",
                            "The file \(file.filename) is incomplete or damaged."))
        }
        try FileManager.default.moveItem(at: temporary, to: stage)
    }

    func install(files: [ImageModelDownloadFile], to destination: URL,
                 progress: @escaping @Sendable (Int64, Int64) async -> Void = { _, _ in }) async throws {
        let total = try totalSize(of: files)
        let fm = FileManager.default
        let parent = destination.deletingLastPathComponent()
        let stage = Self.stagingDirectory(for: destination)
        try Task.checkCancellation()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)
        // The managed image model root is shared by Studio windows and processes.
        // Lock the directory itself to avoid a persistent lock file.
        let lock = open(parent.path, O_RDONLY | O_DIRECTORY | O_CLOEXEC)
        guard lock >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO) }
        defer { close(lock) }
        guard flock(lock, LOCK_EX | LOCK_NB) == 0 else {
            throw failure(L("Un autre téléchargement de modèle image est en cours.",
                            "Another image model download is already running."))
        }
        for url in [stage, destination] {
            if let type = try? fm.attributesOfItem(atPath: url.path)[.type], type as? FileAttributeType != .typeDirectory {
                throw failure(L("Dossier de modèle invalide.", "Invalid model directory."))
            }
        }

        // A crash can leave the last unverified HTTP response in the private stage.
        let incoming = stage.appendingPathComponent(".incoming")
        if unlink(incoming.path) != 0 && errno != ENOENT {
            throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
        }
        var completed: Int64 = 0
        var pending: [ImageModelDownloadFile] = []
        await progress(0, total)
        defer {
            // Verified files survive interruption. Remove an empty staging directory.
            _ = rmdir(stage.path)
        }
        for file in files {
            try Task.checkCancellation()
            let installed = destination.appendingPathComponent(file.filename)
            let cached = stage.appendingPathComponent(file.filename)
            let type = try? fm.attributesOfItem(atPath: cached.path)[.type] as? FileAttributeType
            if try Self.matches(installed, file) {
                // An external installation may have completed while this download was paused.
                if (type == .typeRegular || type == .typeSymbolicLink),
                   installed.resolvingSymlinksInPath() != cached.resolvingSymlinksInPath() {
                    try fm.removeItem(at: cached)
                }
            } else {
                try fm.createDirectory(at: stage, withIntermediateDirectories: true)
                guard type == nil || type == .typeRegular || type == .typeSymbolicLink else {
                    throw failure(L("Fichier de reprise invalide.", "Invalid staged file."))
                }
                if try type != .typeRegular || !Self.matches(cached, file) {
                    if type == .typeRegular || type == .typeSymbolicLink { try fm.removeItem(at: cached) }
                    let offset = completed
                    try await fetch(file, into: cached) { await progress(offset + $0, total) }
                }
                pending.append(file)
            }
            completed += file.size
            await progress(completed, total)
        }
        try Task.checkCancellation()
        guard !pending.isEmpty else { return }
        if !fm.fileExists(atPath: destination.path) {
            let stagedNames = try fm.contentsOfDirectory(atPath: stage.path)
            guard Set(stagedNames) == Set(files.map(\.filename)) else {
                throw failure(L("Le dossier de reprise contient des fichiers inattendus.",
                                "The staged download contains unexpected files."))
            }
            try fm.moveItem(at: stage, to: destination)
        } else {
            // Repair an existing directory without disturbing unrelated files. Commit manifests
            // after the verified payload, so a newly completed model can be discovered safely.
            let manifests = ["quantization.json", "manifest.json"]
            let ordered = pending.filter { !manifests.contains($0.filename) } + pending.filter { manifests.contains($0.filename) }
            for file in ordered {
                let source = stage.appendingPathComponent(file.filename)
                let target = destination.appendingPathComponent(file.filename)
                guard rename(source.path, target.path) == 0 else {
                    throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
                }
            }
        }
    }
}
