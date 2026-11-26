import CryptoKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

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
            parser.add_argument("--reference-image", type=Path)
            parser.add_argument("--low-memory", action="count", default=0)
            args = parser.parse_args()
            (args.model / "last-reference").write_text(str(args.reference_image or ""))
            if args.reference_image:
                (args.model / "last-reference-bytes").write_bytes(args.reference_image.read_bytes())

            (args.model / "last-output").write_text(str(args.output))
            (args.model / "last-prompt").write_text(args.prompt, encoding="utf-8")
            (args.model / "last-variant").write_text(args.variant + ":" + args.steps)
            (args.model / "last-size").write_text(args.size)
            (args.model / "last-low-memory").write_text(str(args.low_memory))
            if args.prompt == "--low-memory":
                raise SystemExit(f"low-memory={args.low_memory}")
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

    static func writePNG(to destination: URL, width: Int, height: Int) throws {
        guard let pixels = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                                     space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let output = CGImageDestinationCreateWithURL(destination as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        pixels.setFillColor(gray: 1, alpha: 1)
        pixels.fill(CGRect(x: 0, y: 0, width: width / 2, height: height))
        guard let image = pixels.makeImage() else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(output, image, nil)
        guard CGImageDestinationFinalize(output) else { throw CocoaError(.fileWriteUnknown) }
    }

    static func publish(_ files: [ImageModelDownloadFile], at destination: URL) throws {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        for file in files { try Data(contentsOf: file.url).write(to: destination.appendingPathComponent(file.filename)) }
    }
}
