import CryptoKit
import Foundation

/// Tiny local catalogs exercise the same path as a downloaded checkpoint.
struct StudioImageFixtures: Sendable {
    let catalog: ImageModelCatalog
    let remote: URL

    init(root: URL) throws {
        remote = root.appendingPathComponent("fixture-remote", isDirectory: true)
        var packages: [Image21Variant: ImageModelPackage] = [:]
        for variant in Image21Variant.allCases {
            let folder = remote.appendingPathComponent(variant.rawValue)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let manifest = variant == .base ? "manifest.json" : "quantization.json"
            let files = try [manifest, "weights.safetensors"].map { name in
                let data = Data((variant.rawValue + "-" + name).utf8)
                let url = folder.appendingPathComponent(name)
                try data.write(to: url)
                let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                return ImageModelDownloadFile(filename: name, url: url, size: Int64(data.count), sha256: hash)
            }
            packages[variant] = ImageModelPackage(variant: variant, repository: "fixture/" + variant.rawValue,
                                                  revision: String(repeating: "a", count: 40), files: files)
        }
        catalog = ImageModelCatalog(packages: packages, modelsRoot: root.appendingPathComponent("ImageModels"))
    }

    static func publish(_ files: [ImageModelDownloadFile], at destination: URL) throws {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for file in files { try Data(contentsOf: file.url).write(to: destination.appendingPathComponent(file.filename)) }
    }
}
