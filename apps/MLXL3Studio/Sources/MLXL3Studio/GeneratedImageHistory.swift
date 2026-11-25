import Foundation
import ImageIO
import UniformTypeIdentifiers

struct GeneratedImageRecord: Identifiable, Equatable, Sendable {
    var id: URL { fileURL }
    let fileURL: URL
    let createdAt: Date
    let prompt: String?
    let modelName: String?
    let width: Int
    let height: Int
    let steps: Int?
    let seed: UInt32?

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
        let size: Int
        let steps: Int
        let seed: UInt32

        func record(at url: URL) -> GeneratedImageRecord {
            GeneratedImageRecord(fileURL: url, createdAt: createdAt, prompt: prompt, modelName: modelName,
                                 width: size, height: size, steps: steps, seed: seed)
        }
    }

    private func sidecar(for image: URL) -> URL {
        directory.appendingPathComponent("." + image.lastPathComponent + ".json")
    }

    private func dimensions(of file: URL, verifyingSize size: Int? = nil) -> (width: Int, height: Int)? {
        let options = [kCGImageSourceShouldCache: false] as CFDictionary
        guard let source = CGImageSourceCreateWithURL(file as CFURL, options),
              CGImageSourceGetType(source) == UTType.png.identifier as CFString,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, options) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int, width > 0,
              let height = properties[kCGImagePropertyPixelHeight] as? Int, height > 0 else { return nil }
        if let size {
            let decode = [kCGImageSourceShouldCacheImmediately: true] as CFDictionary
            guard width == size, height == size,
                  CGImageSourceCreateImageAtIndex(source, 0, decode) != nil,
                  CGImageSourceGetStatusAtIndex(source, 0) == .statusComplete else { return nil }
        }
        return (width, height)
    }

    func save(png: URL, prompt: String, modelName: String, size: Int, steps: Int, seed: UInt32) throws -> GeneratedImageRecord {
        guard dimensions(of: png, verifyingSize: size) != nil else {
            throw NSError(domain: "MLXL3.ImageGeneration", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: L("Le moteur a produit une image illisible.", "The engine returned an unreadable image.")])
        }
        let fm = FileManager.default
        try fm.createDirectory(at: directory, withIntermediateDirectories: true)
        let filename = "Image-\(seed)-\(UUID().uuidString).png"
        let image = directory.appendingPathComponent(filename)
        let metadata = Metadata(version: 1, filename: filename, createdAt: Date(), prompt: prompt,
                                modelName: modelName, size: size, steps: steps, seed: seed)
        let settings = sidecar(for: image)
        // Stage on the destination volume. Readers see the complete PNG only after
        // its settings have been written, even when the worker's temporary volume differs.
        let staged = directory.appendingPathComponent("." + filename + ".partial")
        try fm.moveItem(at: png, to: staged)
        do {
            try JSONEncoder().encode(metadata).write(to: settings, options: .atomic)
            try fm.moveItem(at: staged, to: image)
        } catch {
            try? fm.removeItem(at: staged)
            try? fm.removeItem(at: settings)
            throw error
        }
        return metadata.record(at: image)
    }

    private func metadata(for image: URL) -> Metadata? {
        let file = sidecar(for: image)
        guard let info = try? file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]),
              info.isRegularFile == true, info.isSymbolicLink != true,
              let size = info.fileSize, size <= 2 * 1024 * 1024,
              let data = try? Data(contentsOf: file),
              let metadata = try? JSONDecoder().decode(Metadata.self, from: data),
              metadata.version == 1, metadata.filename == image.lastPathComponent else { return nil }
        return metadata
    }

    private func filenameSeed(_ url: URL) -> UInt32? {
        // Studio's original filenames were Image-<UInt32>-<UUID>.png.
        let stem = url.deletingPathExtension().lastPathComponent
        let parts = stem.split(separator: "-", maxSplits: 2, omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "Image",
              let seed = UInt32(parts[1]), String(seed) == parts[1],
              UUID(uuidString: String(parts[2])) != nil else { return nil }
        return seed
    }

    func records() throws -> [GeneratedImageRecord] {
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
            guard file.pathExtension.lowercased() == "png",
                  let info = try? file.resourceValues(forKeys: keys), info.isRegularFile == true, info.isSymbolicLink != true,
                  let pixels = dimensions(of: file) else { continue }
            let seed = filenameSeed(file)
            // Directory enumeration can expand /var to /private/var. Keep the URL
            // rooted in the library so selection survives saving, refreshing and relaunching.
            let image = directory.appendingPathComponent(file.lastPathComponent)
            if let saved = metadata(for: file), saved.size == pixels.width, saved.size == pixels.height,
               saved.seed == seed, (2...50).contains(saved.steps) {
                records.append(saved.record(at: image))
            } else if let date = info.creationDate ?? info.contentModificationDate {
                records.append(GeneratedImageRecord(fileURL: image, createdAt: date, prompt: nil, modelName: nil,
                                                    width: pixels.width, height: pixels.height, steps: nil, seed: seed))
            }
        }
        return records.sorted(by: GeneratedImageRecord.newestFirst)
    }
}
