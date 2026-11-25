import Foundation
import ImageIO
import UniformTypeIdentifiers

struct Image21Size: Hashable, Sendable {
    let width: Int
    let height: Int

    static func square(_ side: Int) -> Self { Self(width: side, height: side) }

    // Published text-to-image sizes of the Qwen Image 2.1 Base and Turbo checkpoints.
    static let nativePresets: [Self] = [
        .square(2048),
        Self(width: 2400, height: 1792), Self(width: 1792, height: 2400),
        Self(width: 2528, height: 1696), Self(width: 1696, height: 2528),
        Self(width: 2752, height: 1536), Self(width: 1536, height: 2752)
    ]
    static let presets: [Self] = [.square(512), .square(768), .square(1024)] + nativePresets

    var label: String { "\(width) × \(height)" }
    var workerArgument: String { width == height ? String(width) : "\(width)x\(height)" }
}

enum Image21Variant: String, Codable, CaseIterable, Sendable {
    case base, turbo

    var title: String { self == .turbo ? "Turbo" : "Base" }
    var stepRange: ClosedRange<Int> { self == .turbo ? 8...8 : 2...50 }
}

struct GeneratedImageRecord: Identifiable, Equatable, Sendable {
    var id: URL { fileURL }
    let fileURL: URL
    let createdAt: Date
    let prompt: String?
    let modelName: String?
    let variant: Image21Variant?
    let width: Int
    let height: Int
    let steps: Int?
    let seed: UInt32?

    func export(to destination: URL) async throws {
        try await Task.detached(priority: .userInitiated) {
            let source = try fileURL.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileResourceIdentifierKey])
            guard source.isRegularFile == true, source.isSymbolicLink != true else {
                throw CocoaError(.fileReadUnknown)
            }
            let target = destination.resolvingSymlinksInPath()
            if target == fileURL.resolvingSymlinksInPath() { return }
            if let sourceID = source.fileResourceIdentifier,
               let targetID = try? target.resourceValues(forKeys: [.fileResourceIdentifierKey]).fileResourceIdentifier,
               sourceID.isEqual(targetID) { return }
            let png = try Data(contentsOf: fileURL, options: .mappedIfSafe)
            try png.write(to: destination, options: .atomic)
        }.value
    }

    static func newestFirst(_ lhs: Self, _ rhs: Self) -> Bool {
        if lhs.createdAt != rhs.createdAt { return lhs.createdAt > rhs.createdAt }
        return lhs.fileURL.lastPathComponent < rhs.fileURL.lastPathComponent
    }
}

struct GeneratedImageHistory: Sendable {
    static let defaultDirectory = URL.applicationSupportDirectory
        .appendingPathComponent("io.mlxl3.desktop/Generated Images", isDirectory: true)
    let directory: URL

    private struct Metadata: Codable {
        let version: Int
        let filename: String
        let createdAt: Date
        let prompt: String
        let modelName: String
        let variant: Image21Variant?
        let size: Int?
        let width: Int?
        let height: Int?
        let steps: Int
        let seed: UInt32

        func matches(_ pixels: Image21Size) -> Bool {
            switch version {
            case 1, 2: width == nil && height == nil && size == pixels.width && size == pixels.height
            case 3: size == nil && width == pixels.width && height == pixels.height
            default: false
            }
        }

        func record(at url: URL, pixels: Image21Size) -> GeneratedImageRecord {
            GeneratedImageRecord(fileURL: url, createdAt: createdAt, prompt: prompt, modelName: modelName,
                                 variant: variant ?? .base, width: pixels.width, height: pixels.height, steps: steps, seed: seed)
        }
    }

    private func sidecar(for image: URL) -> URL {
        directory.appendingPathComponent("." + image.lastPathComponent + ".json")
    }

    private func dimensions(of file: URL) -> Image21Size? {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(file as CFURL, options) else { return nil }
        return dimensions(of: source, verifyingSize: nil)
    }

    private func dimensions(of source: CGImageSource, verifyingSize size: Image21Size?) -> Image21Size? {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard CGImageSourceGetType(source) == UTType.png.identifier as CFString,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int, width > 0,
              let height = properties[kCGImagePropertyPixelHeight] as? Int, height > 0 else { return nil }
        if let size {
            let decode = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
            guard width == size.width, height == size.height,
                  CGImageSourceCreateImageAtIndex(source, 0, decode) != nil,
                  CGImageSourceGetStatusAtIndex(source, 0) == .statusComplete else { return nil }
        }
        return Image21Size(width: width, height: height)
    }

    struct VerifiedPNG: Sendable {
        let data: Data
        fileprivate let file: URL
        fileprivate let size: Image21Size
    }

    func validatedPNG(at url: URL, size: Image21Size) throws -> VerifiedPNG {
        guard let data = try? Data(contentsOf: url),
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              dimensions(of: source, verifyingSize: size) != nil else {
            throw NSError(domain: "MLXL3.ImageGeneration", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: L("Le moteur a produit une image illisible.", "The engine returned an unreadable image.")])
        }
        return VerifiedPNG(data: data, file: url, size: size)
    }

    func save(png: URL, prompt: String, modelName: String, variant: Image21Variant = .base,
              size: Image21Size, steps: Int, seed: UInt32, id: UUID = UUID(), createdAt: Date = .now) throws -> GeneratedImageRecord {
        try save(verifiedPNG: validatedPNG(at: png, size: size), prompt: prompt, modelName: modelName,
                 variant: variant, steps: steps, seed: seed, id: id, createdAt: createdAt)
    }

    func save(verifiedPNG: VerifiedPNG, prompt: String, modelName: String, variant: Image21Variant,
              steps: Int, seed: UInt32, id: UUID, createdAt: Date) throws -> GeneratedImageRecord {
        guard variant.stepRange.contains(steps) else {
            throw NSError(domain: "MLXL3.ImageGeneration", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: L("Nombre d’étapes incompatible avec le modèle.",
                                                                  "Step count is incompatible with the model.")])
        }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let filename = "Image-\(seed)-\(id.uuidString).png"
        let image = directory.appendingPathComponent(filename)
        // Keep square sidecars readable in Studio versions that predate rectangular output.
        let pixels = verifiedPNG.size
        let square = pixels.width == pixels.height
        let metadata = Metadata(version: square ? 2 : 3, filename: filename, createdAt: createdAt, prompt: prompt,
                                modelName: modelName, variant: variant, size: square ? pixels.width : nil,
                                width: square ? nil : pixels.width, height: square ? nil : pixels.height,
                                steps: steps, seed: seed)
        let settings = sidecar(for: image)
        // Stage on the destination volume. Readers see the complete PNG only after
        // its settings have been written, even when the worker's temporary volume differs.
        let staged = directory.appendingPathComponent("." + filename + ".partial")
        try fm.moveItem(at: verifiedPNG.file, to: staged)
        guard !fm.fileExists(atPath: image.path), !fm.fileExists(atPath: settings.path) else {
            try? fm.removeItem(at: staged)
            throw CocoaError(.fileWriteFileExists)
        }
        do {
            try JSONEncoder().encode(metadata).write(to: settings, options: .atomic)
            try fm.moveItem(at: staged, to: image)
        } catch {
            try? fm.removeItem(at: staged)
            try? fm.removeItem(at: settings)
            throw error
        }
        return metadata.record(at: image, pixels: pixels)
    }

    private func metadata(for image: URL) -> Metadata? {
        let file = sidecar(for: image)
        guard let info = try? file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
              info.isRegularFile == true, info.isSymbolicLink != true,
              let size = info.fileSize, size <= 2 * 1024 * 1024,
              let data = try? Data(contentsOf: file),
              let metadata = try? JSONDecoder().decode(Metadata.self, from: data),
              metadata.filename == image.lastPathComponent else { return nil }
        // Version 1 was written when Base was the only supported checkpoint.
        guard (metadata.version == 1 && metadata.variant == nil) || ([2, 3].contains(metadata.version) && metadata.variant != nil),
              (metadata.variant ?? .base).stepRange.contains(metadata.steps) else { return nil }
        return metadata
    }

    static func filenameIdentity(_ url: URL) -> (seed: UInt32, id: UUID)? {
        // Studio's original filenames were Image-<UInt32>-<UUID>.png.
        let stem = url.deletingPathExtension().lastPathComponent
        let parts = stem.split(separator: "-", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "Image",
              let seed = UInt32(parts[1]), String(seed) == parts[1],
              let id = UUID(uuidString: String(parts[2])) else { return nil }
        return (seed, id)
    }

    func records(excluding knownIDs: Set<UUID> = []) throws -> [GeneratedImageRecord] {
        let keys: Set<URLResourceKey> = [.isRegularFileKey, .isSymbolicLinkKey, .creationDateKey, .contentModificationDateKey]
        let files: [URL]
        do {
            files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: Array(keys),
                                                                options: [.skipsHiddenFiles])
        } catch let e as CocoaError where e.code == .fileReadNoSuchFile || e.code == .fileNoSuchFile {
            return []
        }
        var records: [GeneratedImageRecord] = []
        for file in files {
            try Task.checkCancellation()
            let identity = Self.filenameIdentity(file)
            if let identity, knownIDs.contains(identity.id) { continue }
            guard file.pathExtension.lowercased() == "png",
                  let info = try? file.resourceValues(forKeys: keys), info.isRegularFile == true, info.isSymbolicLink != true,
                  let pixels = dimensions(of: file) else { continue }
            let seed = identity?.seed
            // Directory enumeration can expand /var to /private/var. Keep the URL
            // rooted in the library so selection survives saving, refreshing and relaunching.
            let image = directory.appendingPathComponent(file.lastPathComponent)
            if let saved = metadata(for: file), saved.matches(pixels),
               saved.seed == seed {
                records.append(saved.record(at: image, pixels: pixels))
            } else if let date = info.creationDate ?? info.contentModificationDate {
                records.append(GeneratedImageRecord(fileURL: image, createdAt: date, prompt: nil, modelName: nil, variant: nil,
                                                    width: pixels.width, height: pixels.height, steps: nil, seed: seed))
            }
        }
        return records.sorted(by: GeneratedImageRecord.newestFirst)
    }
}
