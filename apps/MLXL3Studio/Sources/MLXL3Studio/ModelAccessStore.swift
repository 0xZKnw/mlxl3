import Foundation

/// Persists the folders the user explicitly selected in an open panel.
///
/// Models downloaded by MLXL3 live in Application Support and never need a
/// Files & Folders prompt. Models registered from Documents, Downloads, an
/// external disk, etc. receive a security-scoped bookmark so the app can
/// restore access before the engine starts on later launches.
final class ModelAccessStore {
    enum AccessError: LocalizedError {
        case wrongFolder(expected: String)

        var errorDescription: String? {
            switch self {
            case let .wrongFolder(expected):
                "Choisis le dossier exact du modèle : \(expected)"
            }
        }
    }

    private struct Archive: Codable {
        var version = 1
        var bookmarks: [String: Data]
    }

    private let fileURL: URL
    private let managedModelsURL: URL
    private var bookmarks: [String: Data] = [:]
    private var activeURLs: [String: URL] = [:]

    init(
        fileURL: URL = ModelAccessStore.defaultFileURL(),
        managedModelsURL: URL = ModelAccessStore.defaultManagedModelsURL()
    ) {
        self.fileURL = fileURL
        self.managedModelsURL = managedModelsURL.standardizedFileURL
    }

    static func defaultFileURL() -> URL {
        let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/Application Support", directoryHint: .isDirectory)
        return support
            .appending(path: "io.mlxl3.desktop", directoryHint: .isDirectory)
            .appending(path: "model-access.json")
    }

    static func defaultManagedModelsURL() -> URL {
        if let override = ProcessInfo.processInfo.environment["MLXL3_MODELS_DIR"],
           !override.isEmpty {
            return URL(
                fileURLWithPath: (override as NSString).expandingTildeInPath,
                isDirectory: true
            )
        }
        let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask
        ).first ?? FileManager.default.homeDirectoryForCurrentUser
            .appending(path: "Library/Application Support", directoryHint: .isDirectory)
        return support
            .appending(path: "io.mlxl3.desktop", directoryHint: .isDirectory)
            .appending(path: "Models", directoryHint: .isDirectory)
    }

    func restoreAccess() {
        guard let data = try? Data(contentsOf: fileURL),
              let archive = try? JSONDecoder().decode(Archive.self, from: data),
              archive.version == 1
        else { return }

        bookmarks = archive.bookmarks
        var changed = false
        for (storedPath, data) in Array(bookmarks) {
            var stale = false
            guard let url = try? URL(
                resolvingBookmarkData: data,
                options: [.withSecurityScope, .withoutUI],
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            ) else {
                bookmarks.removeValue(forKey: storedPath)
                changed = true
                continue
            }
            let path = normalizedPath(url)
            _ = url.startAccessingSecurityScopedResource()
            activeURLs[path] = url
            if path != storedPath {
                bookmarks.removeValue(forKey: storedPath)
                bookmarks[path] = data
                changed = true
            }
            if stale {
                do {
                    bookmarks[path] = try makeBookmark(for: url)
                    changed = true
                } catch {
                    // The existing bookmark is still usable for this session.
                }
            }
        }
        if changed { try? persist() }
    }

    func hasAccess(to url: URL) -> Bool {
        let path = normalizedPath(url)
        if contains(path, in: normalizedPath(managedModelsURL)) { return true }
        return activeURLs.keys.contains { contains(path, in: $0) }
    }

    func authorize(modelURL: URL, selectedURL: URL) throws {
        let modelPath = normalizedPath(modelURL)
        let selectedPath = normalizedPath(selectedURL)
        guard modelPath == selectedPath else {
            throw AccessError.wrongFolder(expected: modelURL.lastPathComponent)
        }

        let data = try makeBookmark(for: selectedURL)
        if let previous = activeURLs.removeValue(forKey: selectedPath) {
            previous.stopAccessingSecurityScopedResource()
        }
        _ = selectedURL.startAccessingSecurityScopedResource()
        activeURLs[selectedPath] = selectedURL
        bookmarks[selectedPath] = data
        try persist()
    }

    private func makeBookmark(for url: URL) throws -> Data {
        try url.bookmarkData(
            options: [.withSecurityScope, .securityScopeAllowOnlyReadAccess],
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        )
    }

    private func persist() throws {
        try FileManager.default.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        let data = try JSONEncoder().encode(Archive(bookmarks: bookmarks))
        try data.write(to: fileURL, options: .atomic)
    }

    private func normalizedPath(_ url: URL) -> String {
        url.standardizedFileURL.path(percentEncoded: false)
    }

    private func contains(_ candidate: String, in directory: String) -> Bool {
        candidate == directory || candidate.hasPrefix(directory + "/")
    }
}
