import CryptoKit
import Foundation

struct ImageModelPackage: Sendable {
    let variant: Image21Variant
    let repository: String
    let revision: String
    let files: [ImageModelDownloadFile]

    var totalBytes: Int64 { files.reduce(0) { $0 + $1.size } }

    /// HF snapshots can be selected at their root or at the Turbo 4bit subfolder.
    func installedFolder(in directory: URL) -> URL? {
        if isInstalled(at: directory) { return directory }
        guard variant == .turbo else { return nil }
        let child = directory.appendingPathComponent("4bit", isDirectory: true)
        return isInstalled(at: child) ? child : nil
    }

    /// Discovery reads the small manifest and stats the shards. The generation
    /// worker separately verifies every shard's SHA-256 before loading weights.
    func isInstalled(at directory: URL) -> Bool {
        let manifestName = variant == .base ? "manifest.json" : "quantization.json"
        guard let manifest = files.first(where: { $0.filename == manifestName }) else { return false }
        let manifestURL = directory.appendingPathComponent(manifestName).resolvingSymlinksInPath()
        guard let size = Self.regularSize(manifestURL), (1...2_000_000).contains(size),
              let data = try? Data(contentsOf: manifestURL), data.count == size else { return false }
        let isPublished = size == manifest.size && SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() == manifest.sha256
        let shards = files.filter { $0.filename.hasSuffix(".safetensors") }
        guard !shards.isEmpty else { return false }
        var total: Int64 = 0
        for file in shards {
            let url = directory.appendingPathComponent(file.filename).resolvingSymlinksInPath()
            guard let length = Self.regularSize(url), length > 0, !isPublished || length == file.size else { return false }
            let sum = total.addingReportingOverflow(length)
            guard !sum.overflow else { return false }
            total = sum.partialValue
        }
        if isPublished { return true }
        // Studio also accepts pre-publication Base conversions. Their safetensors
        // headers can differ even though their tensor contents are identical.
        guard variant == .base, let legacy = try? JSONDecoder().decode(BaseManifest.self, from: data) else { return false }
        return legacy.covers(shards, total: total)
    }

    private static func regularSize(_ url: URL) -> Int64? {
        guard let info = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]),
              info.isRegularFile == true, let size = info.fileSize else { return nil }
        return Int64(size)
    }

    private struct BaseManifest: Decodable {
        struct Part: Decodable { let file: String; let sha256: String }
        struct Projection: Decodable { let bits: Int }
        let source: String
        let source_revision: String
        let bytes: Int64
        let parts: [Part]
        let recipe: [String: Projection]

        func covers(_ files: [ImageModelDownloadFile], total: Int64) -> Bool {
            let projections = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                               "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"]
            let names = (0..<32).flatMap { block in projections.map { "transformer_blocks.\(block).\($0)" } }
            return source == "Qwen/Qwen-Image-2.1" && source_revision == "d26bb61231c349cf6b7896fa83353113880e1ba3"
                && bytes == total && parts.count == files.count && Set(parts.map(\.file)) == Set(files.map(\.filename))
                && parts.allSatisfy { isLowerHex($0.sha256, count: 64) }
                && Set(recipe.keys) == Set(names) && recipe.values.allSatisfy { (3...5).contains($0.bits) }
        }
    }
}

private func isLowerHex(_ string: String, count: Int) -> Bool {
    string.utf8.count == count && string.utf8.allSatisfy { (48...57).contains($0) || (97...102).contains($0) }
}

struct ImageModelCatalog: Sendable {
    static let defaultModelsRoot = URL.applicationSupportDirectory
        .appendingPathComponent("io.mlxl3.desktop/ImageModels", isDirectory: true)

    static var defaultHubCache: URL? {
        let env = ProcessInfo.processInfo.environment
        func location(_ key: String) -> URL? {
            guard let value = env[key], !value.isEmpty else { return nil }
            return URL(fileURLWithPath: (value as NSString).expandingTildeInPath, isDirectory: true)
        }
        if let url = location("HF_HUB_CACHE") ?? location("HUGGINGFACE_HUB_CACHE") { return url }
        if let home = location("HF_HOME") { return home.appendingPathComponent("hub", isDirectory: true) }
        let cache = location("XDG_CACHE_HOME") ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache")
        return cache.appendingPathComponent("huggingface/hub", isDirectory: true)
    }

    private let modelsRoot: URL
    private let hubCache: URL?
    private let packages: [Image21Variant: ImageModelPackage]

    init(bundle: Bundle = .main, modelsRoot: URL = Self.defaultModelsRoot, hubCache: URL? = Self.defaultHubCache) {
        let url = bundle.url(forResource: "image-models", withExtension: "json", subdirectory: "Image21")
        self.init(packages: url.flatMap { try? Self.readPackages(at: $0) } ?? [:], modelsRoot: modelsRoot, hubCache: hubCache)
    }

    init(packages: [Image21Variant: ImageModelPackage], modelsRoot: URL, hubCache: URL? = nil) {
        self.packages = packages
        self.modelsRoot = modelsRoot
        self.hubCache = hubCache
    }

    func package(for variant: Image21Variant) throws -> ImageModelPackage {
        guard let package = packages[variant] else { throw CatalogError.invalid }
        return package
    }

    func destination(for variant: Image21Variant) -> URL {
        let name = variant == .turbo ? "Qwen-Image-2.1-Turbo-EXL3-4bit" : "Qwen-Image-2.1-EXL3-4bpw"
        return modelsRoot.appendingPathComponent(name, isDirectory: true)
    }

    func hasPartialDownload(for variant: Image21Variant) -> Bool {
        let stage = ImageModelDownload.stagingDirectory(for: destination(for: variant))
        let fm = FileManager.default
        guard let package = packages[variant],
              let type = try? fm.attributesOfItem(atPath: stage.path)[.type] as? FileAttributeType, type == .typeDirectory,
              let entries = try? fm.contentsOfDirectory(atPath: stage.path) else { return false }
        let names = Set(entries)
        // The incoming response has no catalog name until the installer verifies
        // it. Look for the smallest reusable file first and stop at the first match.
        let candidates = package.files.filter { names.contains($0.filename) }.sorted { $0.size < $1.size }
        for file in candidates {
            if Task.isCancelled { return false }
            let url = stage.appendingPathComponent(file.filename)
            guard let type = try? fm.attributesOfItem(atPath: url.path)[.type] as? FileAttributeType,
                  type == .typeRegular else { continue }
            if (try? ImageModelDownload.matches(url, file)) == true { return true }
        }
        return false
    }

    func installedDirectory(for variant: Image21Variant, preferred: URL? = nil) -> URL? {
        guard let package = packages[variant] else { return nil }
        if let preferred, let local = package.installedFolder(in: preferred) { return local }
        let managed = destination(for: variant)
        if package.isInstalled(at: managed) { return managed }
        guard let hubCache else { return nil }
        let repo = hubCache.appendingPathComponent("models--" + package.repository.replacingOccurrences(of: "/", with: "--"))
        let snapshots = repo.appendingPathComponent("snapshots")
        func directory(at revision: String) -> URL {
            let snapshot = snapshots.appendingPathComponent(revision, isDirectory: true)
            return variant == .turbo ? snapshot.appendingPathComponent("4bit", isDirectory: true) : snapshot
        }
        let pinned = directory(at: package.revision)
        if package.isInstalled(at: pinned) { return pinned }
        // A download at another immutable revision may have identical weights
        // and no refs/main. Inspect this repository's snapshot directories only.
        guard let revisions = try? FileManager.default.contentsOfDirectory(atPath: snapshots.path) else { return nil }
        for revision in revisions.sorted() where revision != package.revision && isLowerHex(revision, count: 40) {
            let candidate = directory(at: revision)
            if package.isInstalled(at: candidate) { return candidate }
        }
        return nil
    }

    private enum CatalogError: LocalizedError {
        case invalid
        var errorDescription: String? { L("Le catalogue des modèles Image est indisponible.", "The image model catalog is unavailable.") }
    }

    static func readPackages(at url: URL) throws -> [Image21Variant: ImageModelPackage] {
        struct File: Decodable {
            let name: String
            let size: Int64
            let sha256: String
        }
        struct Entry: Decodable {
            let repository: String
            let revision: String
            let files: [File]
        }
        let entries = try JSONDecoder().decode([String: Entry].self, from: Data(contentsOf: url))
        var result: [Image21Variant: ImageModelPackage] = [:]
        for variant in Image21Variant.allCases {
            let repository = variant == .turbo ? "jyohukuchan/Qwen-Image-2.1-Turbo-EXL3" : "Henk0o/mlxl3-image21-exl3-4bpw"
            guard let entry = entries[variant.rawValue], entry.repository == repository, isLowerHex(entry.revision, count: 40),
                  !entry.files.isEmpty, Set(entry.files.map(\.name)).count == entry.files.count else { throw CatalogError.invalid }
            var remote = URL(string: "https://huggingface.co")!.appendingPathComponent(repository)
                .appendingPathComponent("resolve").appendingPathComponent(entry.revision)
            if variant == .turbo { remote.appendPathComponent("4bit") }
            var total: Int64 = 0
            let files = try entry.files.map { item in
                let sum = total.addingReportingOverflow(item.size)
                guard !item.name.isEmpty, item.name.first != ".", item.name.utf8.allSatisfy({
                    (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0) || [45, 46, 95].contains($0)
                }), item.size > 0, !sum.overflow, isLowerHex(item.sha256, count: 64) else { throw CatalogError.invalid }
                total = sum.partialValue
                return ImageModelDownloadFile(filename: item.name, url: remote.appendingPathComponent(item.name),
                                              size: item.size, sha256: item.sha256)
            }
            result[variant] = ImageModelPackage(variant: variant, repository: repository, revision: entry.revision, files: files)
        }
        return result
    }
}
