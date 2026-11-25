import Foundation

@main
struct ImageGenerationCheck {
    @MainActor static func checkHistory(root: URL) async throws {
        var assertions = 0
        func check(_ value: @autoclosure () -> Bool, _ message: String = "Image history invariant failed") {
            precondition(value(), message)
            assertions += 1
        }
        let fm = FileManager.default
        let images = root.appendingPathComponent("Generated Images")
        let model = root.appendingPathComponent("Qwen-Image-fixture")
        try fm.createDirectory(at: model, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: model.appendingPathComponent("manifest.json"))
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let sample = try Data(contentsOf: repo.appendingPathComponent("docs/images/studio-image21-m5.png"))
        try sample.write(to: model.appendingPathComponent("sample.png"))

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
            for key in ("prompt", "size", "steps", "seed"):
                parser.add_argument("--" + key, required=True)
            parser.add_argument("--low-memory", action="count", default=0)
            args = parser.parse_args()
            (args.model / "last-prompt").write_text(args.prompt, encoding="utf-8")
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
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: path.path)
        }
        let engine = ImageGeneration(supported: true, lowMemoryAvailable: true,
                                     historyDirectory: images, worker: (uv, renderer))
        func wait(_ predicate: @MainActor () -> Bool) async throws {
            for _ in 0..<500 {
                if predicate() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            preconditionFailure("Timed out waiting for the isolated image worker")
        }
        // Read the argument count from the actual child after the interpreter handoff.
        for (size, expected) in [(512, 0), (768, 0), (1024, 1)] {
            engine.size = size
            check(engine.offersLowMemory(at: engine.size) == (expected == 1))
            var draftLowMemory = true
            engine.start(model: model, prompt: "--low-memory", size: engine.size, steps: 2, seed: 123, lowMemory: draftLowMemory)
            draftLowMemory = false
            engine.size = 512
            try await wait { !engine.running }
            check(!draftLowMemory && engine.error == "low-memory=\(expected)")
        }
        engine.start(model: model, prompt: "--low-memory", size: 1024, steps: 2, seed: 123)
        try await wait { !engine.running }
        check(engine.error == "low-memory=0", "The worker must keep its default without an explicit choice")
        let otherHardware = ImageGeneration(supported: true, lowMemoryAvailable: false,
                                            historyDirectory: images, worker: (uv, renderer))
        check(!otherHardware.offersLowMemory(at: 1024))
        otherHardware.start(model: model, prompt: "--low-memory", size: 1024, steps: 2, seed: 123, lowMemory: true)
        try await wait { !otherHardware.running }
        check(otherHardware.error == "low-memory=0")

        await engine.refreshHistory()
        check(engine.history.isEmpty && !engine.isLoadingHistory && engine.historyError == nil)
        check(!fm.fileExists(atPath: images.path), "Opening an empty gallery must not create files")
        let prompt = "Lune 🌙, lignes ‘bleues’.\nDeuxième ligne."
        engine.start(model: model, prompt: prompt, size: 512, steps: 2, seed: 0)
        engine.prompt = "a changed draft"
        engine.seed = 10
        try await wait { !engine.running }
        guard let first = engine.history.first else { preconditionFailure(engine.error ?? "Missing history") }
        check(engine.result == first.fileURL && first.prompt == prompt && first.steps == 2 && first.seed == 0)
        check(first.modelName == "Qwen-Image-fixture" && first.width == 512 && first.height == 512)
        let published = try Data(contentsOf: first.fileURL)
        check(published == sample, "The worker's PNG must survive publication byte-for-byte")
        engine.start(model: model, prompt: "--seed", size: 512, steps: 50, seed: 0)
        try await wait { !engine.running }
        let received = try String(contentsOf: model.appendingPathComponent("last-prompt"), encoding: .utf8)
        check(engine.error == nil && received == "--seed", "Image prompts that look like options reach the worker verbatim")
        guard let second = engine.history.first else { preconditionFailure() }
        check(engine.history == [second, first] && second.id != first.id)
        check(second.prompt == "--seed" && second.steps == 50 && second.createdAt >= first.createdAt)

        let reopened = ImageGeneration(supported: false, historyDirectory: images)
        await reopened.refreshHistory()
        check(reopened.history == [second, first] && reopened.historyError == nil, "Restore after relaunch")
        engine.start(model: model, prompt: "corrupt", size: 512, steps: 2, seed: 5)
        try await wait { !engine.running }
        check(engine.error == "The engine returned an unreadable image." && engine.result == nil)
        engine.start(model: model, prompt: "wrong dimensions", size: 768, steps: 2, seed: 6)
        try await wait { !engine.running }
        check(engine.error == "The engine returned an unreadable image." && engine.history == [second, first])
        engine.start(model: model, prompt: "pause", size: 512, steps: 2, seed: 7)
        try await wait { fm.fileExists(atPath: model.appendingPathComponent("started").path) }
        engine.cancel()
        try await wait { !engine.running }
        await engine.refreshHistory()
        check(engine.status == "Generation cancelled." && engine.error == nil && engine.history == [second, first])

        let legacy = images.appendingPathComponent("Image-4294967295-22C1B2AC-D794-445B-BF0A-10AE664773D0.png")
        let unknown = images.appendingPathComponent("renamed.png")
        try sample.write(to: legacy)
        try sample.write(to: unknown)
        try fm.createSymbolicLink(at: images.appendingPathComponent("shortcut.png"), withDestinationURL: first.fileURL)
        try Data("bad png".utf8).write(to: images.appendingPathComponent("damaged.png"))
        try sample.write(to: images.appendingPathComponent(".incomplete.png"))
        try Data("{}".utf8).write(to: images.appendingPathComponent(".no-such-image.png.json"))
        // A truncated sidecar or one belonging to a different image must never invent a prompt.
        let firstInfo = images.appendingPathComponent("." + first.fileURL.lastPathComponent + ".json")
        let secondInfo = images.appendingPathComponent("." + second.fileURL.lastPathComponent + ".json")
        try fm.copyItem(at: firstInfo, to: images.appendingPathComponent("." + legacy.lastPathComponent + ".json"))
        try Data("{".utf8).write(to: secondInfo, options: .atomic)
        await reopened.refreshHistory()
        check(reopened.history.count == 4 && reopened.historyError == nil)
        guard let old = reopened.history.first(where: { $0.fileURL == legacy }),
              let renamed = reopened.history.first(where: { $0.fileURL == unknown }),
              let incomplete = reopened.history.first(where: { $0.id == second.id }) else { preconditionFailure() }
        check(old.prompt == nil && old.steps == nil && old.seed == .max && old.width == 512 && old.height == 512)
        check(renamed.prompt == nil && renamed.seed == nil && renamed.modelName == nil)
        check(incomplete.prompt == nil && incomplete.seed == 0 && incomplete.steps == nil)
        check(reopened.history.contains(first), "One corrupt entry must not hide the valid entry")
        check(zip(reopened.history, reopened.history.dropFirst()).allSatisfy { $0.createdAt >= $1.createdAt })
        try fm.removeItem(at: first.fileURL)
        await reopened.refreshHistory()
        check(reopened.history.count == 3 && !reopened.history.contains(first), "Missing PNGs are not gallery entries")
        let blocked = ImageGeneration(historyDirectory: unknown)
        await blocked.refreshHistory()
        check(!blocked.isLoadingHistory && blocked.history.isEmpty && blocked.historyError != nil)
        print("Image history: \(assertions) assertions, real subprocess save/cancel/failure, exact PNG, reopen, legacy names, sorting, missing/corrupt metadata and isolated files checked")
    }

    @MainActor static func main() async throws {
        AppLocalization.set(.en)
        let fetching = "Fetching 9 files: 100%|########| 9/9 [00:01<00:00]"
        precondition(ImageGeneration.progressMessage(in: fetching, steps: 9) == nil)
        precondition(ImageGeneration.progressMessage(in: "Loading tokenizer, text encoder and VAE\r" + fetching, steps: 9) == "Loading the text encoder and VAE…")
        let log = fetching + "\r  0%|      | 0/9 [00:00<?, ?it/s]\r 44%|####  | 4/9 [00:06<00:09]"
        precondition(ImageGeneration.progressMessage(in: log, steps: 9) == "Step 4 of 9")
        precondition(ImageGeneration.progressMessage(in: log + "\r100%|######| 9/9 [00:16<00:00]", steps: 9) == "Finishing the image…")
        precondition(ImageGeneration.progressMessage(in: "100%|######| 10/9 [00:01<00:00]", steps: 9) == nil)
        let minimum = OperatingSystemVersion(majorVersion: 26, minorVersion: 3, patchVersion: 0)
        let old = OperatingSystemVersion(majorVersion: 26, minorVersion: 2, patchVersion: 9)
        for gpu in ["Apple M5", "Apple M5 Pro", "Apple M5 Max"] {
            precondition(Image21Hardware.supports(gpu: gpu, os: minimum, disabled: false))
            precondition(!Image21Hardware.supports(gpu: gpu, os: old, disabled: false))
            precondition(!Image21Hardware.supports(gpu: gpu, os: minimum, disabled: true))
        }
        for gpu in [nil, "Apple M1", "Apple M2 Pro", "Apple M3 Ultra", "Apple M4 Max", "Apple M50"] as [String?] {
            precondition(!Image21Hardware.supports(gpu: gpu, os: minimum, disabled: false))
        }
        let gib: UInt64 = 1_024 * 1_024 * 1_024
        precondition(Image21Hardware.supportsLowMemory(gpu: "Apple M5", memory: 24 * gib))
        for gpu in [nil, "Apple M5 Pro", "Apple M5 Max", "Apple M4"] as [String?] {
            precondition(!Image21Hardware.supportsLowMemory(gpu: gpu, memory: 24 * gib))
        }
        for memory in [16 * gib, 32 * gib] {
            precondition(!Image21Hardware.supportsLowMemory(gpu: "Apple M5", memory: memory))
        }
        let engine = ImageGeneration(supported: true)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for (supported, steps, message) in [(false, 40, Image21Hardware.requirement),
                                             (true, 1, "Choose between 2 and 50 steps.")] {
            let rejected = ImageGeneration(supported: supported)
            rejected.start(model: root, prompt: "oak", size: 512, steps: steps, seed: 371)
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while rejected.running && ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            precondition(!rejected.running && rejected.error == message && rejected.result == nil)
        }

        for _ in 0..<2 {
            precondition(!engine.running)
            engine.start(model: root, prompt: "teapot", size: 512, steps: 2, seed: 371)
            precondition(engine.running && engine.error == nil && engine.result == nil)
            // A second click must not replace the in-flight request.
            engine.start(model: root, prompt: "duplicate", size: 512, steps: 2, seed: 372)
            let deadline = ContinuousClock.now.advanced(by: .seconds(5))
            while engine.running && ContinuousClock.now < deadline {
                try await Task.sleep(for: .milliseconds(10))
            }
            precondition(!engine.running && engine.result == nil && engine.status.isEmpty)
            precondition(engine.error == "Choose the EXL3 folder containing manifest.json.")
        }
        try await checkHistory(root: root)
        print("Image generation: overlapping request, missing model and retry checked")
    }
}
