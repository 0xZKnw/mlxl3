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
            for key in ("variant", "prompt", "size", "steps", "seed"):
                parser.add_argument("--" + key, required=True)
            args = parser.parse_args()
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
            try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: path.path)
        }
        let suite = "mlxl3.turbo-test.\(UUID().uuidString)"
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        prefs.set(model.path, forKey: "image21ModelDirectory")
        let turbo = root.appendingPathComponent("Installed Turbo", isDirectory: true)
        let engine = ImageGeneration(supported: true, historyDirectory: images, worker: (uv, renderer),
                                     preferences: prefs, installedTurboDirectory: turbo)
        func wait(_ predicate: @MainActor () -> Bool) async throws {
            for _ in 0..<500 {
                if predicate() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            preconditionFailure("Timed out waiting for the isolated image worker")
        }
        await engine.refreshHistory()
        check(engine.variant == .base && engine.modelDirectory?.path == model.path && engine.steps == 40)
        check(engine.history.isEmpty && !engine.isLoadingHistory && engine.historyError == nil)
        check(!fm.fileExists(atPath: images.path), "Opening an empty gallery must not create files")
        let prompt = "Lune 🌙, lignes ‘bleues’.\nDeuxième ligne."
        engine.start(model: model, prompt: prompt, size: 512, steps: 2, seed: 0)
        engine.prompt = "a changed draft"
        engine.seed = 10
        try await wait { !engine.running }
        guard let first = engine.history.first else { preconditionFailure(engine.error ?? "Missing history") }
        check(engine.result == first.fileURL && first.prompt == prompt && first.steps == 2 && first.seed == 0)
        check(first.modelName == "Qwen-Image-fixture" && first.variant == .base && first.width == 512 && first.height == 512)
        let baseArgs = try String(contentsOf: model.appendingPathComponent("last-variant"), encoding: .utf8)
        check(baseArgs == "base:2", "Base reaches the worker with the selected steps")
        let published = try Data(contentsOf: first.fileURL)
        check(published == sample, "The worker's PNG must survive publication byte-for-byte")
        engine.start(model: model, prompt: "--seed", size: 512, steps: 50, seed: 0)
        try await wait { !engine.running }
        let received = try String(contentsOf: model.appendingPathComponent("last-prompt"), encoding: .utf8)
        check(engine.error == nil && received == "--seed", "Image prompts that look like options reach the worker verbatim")
        guard let second = engine.history.first else { preconditionFailure() }
        check(engine.history == [second, first] && second.id != first.id)
        check(second.prompt == "--seed" && second.steps == 50 && second.createdAt >= first.createdAt)

        let reopened = ImageGeneration(supported: false, historyDirectory: images, preferences: prefs)
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

        engine.baseSteps = 31
        engine.selectVariant(.turbo)
        check(engine.modelDirectory == nil && engine.steps == 8, "Turbo must not silently use the Base folder")
        try fm.createDirectory(at: turbo, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: turbo.appendingPathComponent("quantization.json"))
        try sample.write(to: turbo.appendingPathComponent("sample.png"))
        engine.refreshModelDirectory()
        check(engine.modelDirectory == turbo, "An installed Turbo folder is discovered without a picker")
        for wrongSteps in [1, 7, 9, 40, 51] {
            engine.start(model: turbo, prompt: "turbo", size: 512, steps: wrongSteps, seed: 123)
            try await wait { !engine.running }
            check(engine.error == "Turbo uses 8 steps." && !fm.fileExists(atPath: turbo.appendingPathComponent("last-prompt").path))
        }
        engine.start(model: model, prompt: "turbo", size: 512, steps: 8, seed: 123)
        try await wait { !engine.running }
        check(engine.error?.contains("quantization.json") == true && engine.history.count == 2)
        engine.start(model: turbo, prompt: "turbo", size: 512, steps: engine.steps, seed: 123)
        engine.selectVariant(.base)
        try await wait { !engine.running }
        let turboArgs = try String(contentsOf: turbo.appendingPathComponent("last-variant"), encoding: .utf8)
        guard let fast = engine.history.first else { preconditionFailure() }
        check(turboArgs == "turbo:8" && fast.variant == .turbo && fast.steps == 8 && engine.variant == .turbo)
        let custom = root.appendingPathComponent("My Turbo", isDirectory: true)
        engine.selectModelDirectory(custom)
        let restored = ImageGeneration(supported: false, historyDirectory: images, preferences: prefs, installedTurboDirectory: turbo)
        check(restored.variant == .turbo && restored.modelDirectory == custom && restored.steps == 8)
        await restored.refreshHistory()
        check(restored.history.first == fast && restored.history.count == 3)
        engine.selectVariant(.base)
        check(engine.steps == 31 && engine.modelDirectory?.path == model.path, "Switching back must restore Base settings")
        let store = GeneratedImageHistory(directory: images)
        do {
            _ = try store.save(png: turbo.appendingPathComponent("sample.png"), prompt: "invalid", modelName: "Turbo",
                               variant: .turbo, size: 512, steps: 9, seed: 0)
            preconditionFailure("Saved a Turbo render with a different sampling schedule")
        } catch {
            check(fm.fileExists(atPath: turbo.appendingPathComponent("sample.png").path))
        }
        let turboInfo = images.appendingPathComponent("." + fast.fileURL.lastPathComponent + ".json")
        let originalTurboInfo = try Data(contentsOf: turboInfo)
        for invalid in ["unknown variant", "missing variant", "wrong schedule"] {
            var metadata = try JSONSerialization.jsonObject(with: originalTurboInfo) as! [String: Any]
            if invalid == "unknown variant" { metadata["variant"] = "future-model" }
            if invalid == "missing variant" { metadata.removeValue(forKey: "variant") }
            if invalid == "wrong schedule" { metadata["steps"] = 40 }
            try JSONSerialization.data(withJSONObject: metadata).write(to: turboInfo, options: .atomic)
            await restored.refreshHistory()
            check(restored.history.first?.id == fast.id && restored.history.first?.variant == nil
                  && restored.history.first?.prompt == nil, "Invalid metadata must not be reused as Base")
        }
        try originalTurboInfo.write(to: turboInfo, options: .atomic)

        let legacy = images.appendingPathComponent("Image-4294967295-22C1B2AC-D794-445B-BF0A-10AE664773D0.png")
        let unknown = images.appendingPathComponent("renamed.png")
        try sample.write(to: legacy)
        try sample.write(to: unknown)
        try fm.createSymbolicLink(at: images.appendingPathComponent("shortcut.png"), withDestinationURL: first.fileURL)
        try Data("bad png".utf8).write(to: images.appendingPathComponent("damaged.png"))
        try sample.write(to: images.appendingPathComponent(".incomplete.png"))
        try Data("{}".utf8).write(to: images.appendingPathComponent(".no-such-image.png.json"))
        // Version 1 predates Turbo. A missing or malformed sidecar cannot establish which model rendered an image.
        let firstInfo = images.appendingPathComponent("." + first.fileURL.lastPathComponent + ".json")
        let secondInfo = images.appendingPathComponent("." + second.fileURL.lastPathComponent + ".json")
        var legacyMetadata = try JSONSerialization.jsonObject(with: Data(contentsOf: firstInfo)) as! [String: Any]
        legacyMetadata["version"] = 1
        legacyMetadata.removeValue(forKey: "variant")
        try JSONSerialization.data(withJSONObject: legacyMetadata).write(to: firstInfo, options: .atomic)
        try fm.copyItem(at: firstInfo, to: images.appendingPathComponent("." + legacy.lastPathComponent + ".json"))
        try Data("{".utf8).write(to: secondInfo, options: .atomic)
        await reopened.refreshHistory()
        check(reopened.history.count == 5 && reopened.historyError == nil)
        guard let old = reopened.history.first(where: { $0.fileURL == legacy }),
              let renamed = reopened.history.first(where: { $0.fileURL == unknown }),
              let incomplete = reopened.history.first(where: { $0.id == second.id }) else { preconditionFailure() }
        check(old.prompt == nil && old.steps == nil && old.seed == .max && old.width == 512 && old.height == 512)
        check(renamed.prompt == nil && renamed.seed == nil && renamed.modelName == nil && renamed.variant == nil)
        check(incomplete.prompt == nil && incomplete.seed == 0 && incomplete.steps == nil)
        check(reopened.history.contains(first), "One corrupt entry must not hide the valid entry")
        check(zip(reopened.history, reopened.history.dropFirst()).allSatisfy { $0.createdAt >= $1.createdAt })
        try fm.removeItem(at: first.fileURL)
        await reopened.refreshHistory()
        check(reopened.history.count == 4 && !reopened.history.contains(first), "Missing PNGs are not gallery entries")
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
