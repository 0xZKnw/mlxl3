import CryptoKit
import Foundation

@main
struct ImageModelCatalogCheck {
    static func main() throws {
        let fm = FileManager.default
        let temp = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fm.createDirectory(at: temp, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: temp) }
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        let packages = try ImageModelCatalog.readPackages(at: url)
        let base = packages[.base]!, turbo = packages[.turbo]!
        precondition(base.revision == "28b15b0087dbcb83341004c577283f21be135fbe" && base.totalBytes == 3_556_970_982)
        precondition(turbo.revision == "f362145b8e8ebc88d6639c445eb9f31beab7c0b4" && turbo.totalBytes == 3_766_859_807)
        let blocks = (0..<32).map { String(format: "block-%02d.safetensors", $0) }
        precondition(Set(base.files.map(\.filename)) == Set(blocks + ["protected.safetensors", "manifest.json", "config.json", "LICENSE", "Notice"]))
        let projections = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.proj", "img_mlp.gate_layer", "img_mlp.out"]
        let names = (0..<32).flatMap { i in projections.map { "transformer_blocks.\(i).\($0)" } }
        func digest(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
        let weights = names.map { digest(Data($0.utf8)) + ".safetensors" }
        precondition(Set(turbo.files.map(\.filename)) == Set(weights + ["unquantized.safetensors", "quantization.json", "transformer_config.json", "LICENSE", "Notice"]))
        for package in [base, turbo] {
            for file in package.files {
                let prefix = "https://huggingface.co/\(package.repository)/resolve/\(package.revision)/" + (package.variant == .turbo ? "4bit/" : "")
                precondition(file.url.absoluteString == prefix + file.filename)
            }
        }
        let markerData = Data("fixture".utf8), shardData = Data([5, 4, 3])
        func fixtureFile(_ name: String, _ data: Data) -> ImageModelDownloadFile {
            .init(filename: name, url: URL(string: "https://huggingface.co/fixture")!,
                  size: Int64(data.count), sha256: digest(data))
        }
        let marker = fixtureFile("quantization.json", markerData), shard = fixtureFile("weight.safetensors", shardData)
        let package = ImageModelPackage(variant: .turbo, repository: turbo.repository, revision: turbo.revision, files: [marker, shard])
        let hub = temp.appendingPathComponent("hub")
        let catalog = ImageModelCatalog(packages: [.turbo: package], modelsRoot: temp.appendingPathComponent("models"), hubCache: hub)
        let preferred = temp.appendingPathComponent("chosen", isDirectory: true)
        let canonical = catalog.destination(for: .turbo)
        let stage = canonical.deletingLastPathComponent().appendingPathComponent("." + canonical.lastPathComponent + ".partial")
        precondition(!catalog.hasPartialDownload(for: .turbo))
        try fm.createDirectory(at: stage, withIntermediateDirectories: true)
        let stagedShard = stage.appendingPathComponent(shard.filename)
        let incoming = stage.appendingPathComponent(".incoming")
        try shardData.write(to: incoming)
        precondition(!catalog.hasPartialDownload(for: .turbo))
        try fm.createSymbolicLink(at: stagedShard, withDestinationURL: incoming)
        precondition(!catalog.hasPartialDownload(for: .turbo))
        try fm.removeItem(at: stagedShard)
        try Data([42]).write(to: stagedShard)
        precondition(!catalog.hasPartialDownload(for: .turbo))
        try Data([42, 42, 42]).write(to: stagedShard)
        precondition(!catalog.hasPartialDownload(for: .turbo))
        try markerData.write(to: stage.appendingPathComponent(marker.filename))
        precondition(catalog.hasPartialDownload(for: .turbo))
        precondition(!catalog.hasPartialDownload(for: .base))
        let movedStage = temp.appendingPathComponent("foreign-partial")
        try fm.moveItem(at: stage, to: movedStage)
        try fm.createSymbolicLink(at: stage, withDestinationURL: movedStage)
        precondition(!catalog.hasPartialDownload(for: .turbo))
        try fm.removeItem(at: stage)
        let repo = hub.appendingPathComponent("models--jyohukuchan--Qwen-Image-2.1-Turbo-EXL3")
        let snapshot = repo.appendingPathComponent("snapshots").appendingPathComponent(turbo.revision).appendingPathComponent("4bit", isDirectory: true)
        let blobs = repo.appendingPathComponent("blobs")
        try fm.createDirectory(at: blobs, withIntermediateDirectories: true)
        for (file, bytes) in [(marker, markerData), (shard, shardData)] {
            try bytes.write(to: blobs.appendingPathComponent(file.sha256))
        }
        // Exercise all eight combinations of complete / missing preferred,
        // managed and symlink-backed HF snapshots; no process or network involved.
        for mask in 0..<8 {
            let urls = [preferred, canonical, snapshot]
            for (i, directory) in urls.enumerated() {
                if fm.fileExists(atPath: directory.path) { try fm.removeItem(at: directory) }
                try fm.createDirectory(at: directory, withIntermediateDirectories: true)
                let complete = mask & (1 << i) != 0
                for file in (complete ? [marker, shard] : [marker]) {
                    try fm.createSymbolicLink(at: directory.appendingPathComponent(file.filename),
                                              withDestinationURL: blobs.appendingPathComponent(file.sha256))
                }
            }
            let expected = urls.enumerated().first { mask & (1 << $0.offset) != 0 }?.element
            precondition(catalog.installedDirectory(for: .turbo, preferred: preferred)?.path == expected?.path)
        }
        let preferredMarker = preferred.appendingPathComponent(marker.filename)
        try fm.removeItem(at: preferredMarker)
        try Data("invalid".utf8).write(to: preferredMarker)
        precondition(catalog.installedDirectory(for: .turbo, preferred: preferred)?.path == canonical.path)
        let selectedShard = canonical.appendingPathComponent(shard.filename)
        try fm.removeItem(at: selectedShard)
        try Data([42]).write(to: selectedShard)
        precondition(catalog.installedDirectory(for: .turbo, preferred: preferred)?.path == snapshot.path)
        try fm.removeItem(at: blobs.appendingPathComponent(shard.sha256))
        precondition(catalog.installedDirectory(for: .turbo, preferred: preferred) == nil)
        // An unrelated pinned revision of the same package may already be
        // installed by HF without a refs/main pointer.
        let newRevision = String(repeating: "a", count: 40)
        let other = repo.appendingPathComponent("snapshots").appendingPathComponent(newRevision).appendingPathComponent("4bit", isDirectory: true)
        try fm.createDirectory(at: other, withIntermediateDirectories: true)
        try markerData.write(to: other.appendingPathComponent(marker.filename))
        try shardData.write(to: other.appendingPathComponent(shard.filename))
        precondition(catalog.installedDirectory(for: .turbo)?.path == other.path)
        precondition(catalog.installedDirectory(for: .turbo, preferred: other.deletingLastPathComponent())?.path == other.path)

        // Pre-publication Base weights have different safetensors headers. A full
        // historical set is usable, a truncated shard or wrong upstream is not.
        let localBase = temp.appendingPathComponent("base")
        try fm.createDirectory(at: localBase, withIntermediateDirectories: true)
        let shards = base.files.filter { $0.filename.hasSuffix(".safetensors") }
        for file in shards { try shardData.write(to: localBase.appendingPathComponent(file.filename)) }
        var legacy: [String: Any] = [
            "source": "Qwen/Qwen-Image-2.1", "source_revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
            "bytes": shards.count * shardData.count, "parts": shards.map { ["file": $0.filename, "sha256": digest(shardData)] },
            "recipe": Dictionary(uniqueKeysWithValues: names.map { ($0, ["bits": 4]) })
        ]
        let manifest = localBase.appendingPathComponent("manifest.json")
        try JSONSerialization.data(withJSONObject: legacy).write(to: manifest)
        precondition(base.isInstalled(at: localBase))
        try Data([1]).write(to: localBase.appendingPathComponent(shards[0].filename))
        precondition(!base.isInstalled(at: localBase))
        try shardData.write(to: localBase.appendingPathComponent(shards[0].filename))
        legacy["source"] = "wrong/source"
        try JSONSerialization.data(withJSONObject: legacy).write(to: manifest)
        precondition(!base.isInstalled(at: localBase))
        print("Image model catalog passed: 266 pinned files, eight discovery combinations, old Base, partial and corrupt metadata, cached symlinks")
    }
}
