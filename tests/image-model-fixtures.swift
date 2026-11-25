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

    static func worker(in root: URL) throws -> (launcher: URL, script: URL) {
        // Exercise the production process launch, acknowledgement, validation and save path
        // without requiring a physical M5, a model download, or the user's real library.
        let uv = root.appendingPathComponent("fixture-uv")
        let renderer = root.appendingPathComponent("fixture-renderer")
        try Data("""
            #!/bin/sh
            printf '%s\\n' "${MLXL3_TEST_PYTHON:-/usr/bin/python3}"
            """.utf8).write(to: uv)
        try Data("""
            import argparse
            from pathlib import Path
            import shutil
            import time
            parser = argparse.ArgumentParser()
            for key in ("model", "output"):
                parser.add_argument("--" + key, required=True, type=Path)
            for key in ("variant", "prompt", "size", "steps", "seed"):
                parser.add_argument("--" + key, required=True)
            args = parser.parse_args()
            (args.model / "last-output").write_text(str(args.output))
            (args.model / "last-prompt").write_text(args.prompt, encoding="utf-8")
            (args.model / "last-variant").write_text(args.variant + ":" + args.steps)
            if args.prompt == "corrupt":
                args.output.write_bytes(b"bad PNG")
            elif args.prompt == "pause":
                (args.model / "started").touch()
                time.sleep(20)
            else:
                shutil.copyfile(args.model / "sample.png", args.output)
            """.utf8).write(to: renderer)
        for path in [uv, renderer] {
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: path.path)
        }
        return (uv, renderer)
    }

    static func publish(_ files: [ImageModelDownloadFile], at destination: URL) throws {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for file in files { try Data(contentsOf: file.url).write(to: destination.appendingPathComponent(file.filename)) }
    }
}
