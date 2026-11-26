import Foundation

@main
struct ImageGenerationCheck {
    actor TestTransfer {
        private(set) var attempts = 0

        func install(files: [ImageModelDownloadFile], at destination: URL,
                     progress: @escaping @Sendable (Int64, Int64) async -> Void) async throws {
            attempts += 1
            let total = files.reduce(Int64(0)) { $0 + $1.size }
            await progress(1, total)
            let stage = destination.deletingLastPathComponent().appendingPathComponent("." + destination.lastPathComponent + ".partial")
            switch attempts {
            case 1:
                try StudioImageFixtures.publish([files[0]], at: stage)
                try await Task.sleep(for: .seconds(10))
            case 2: throw URLError(.cannotConnectToHost)
            default:
                try StudioImageFixtures.publish(files, at: destination)
                if FileManager.default.fileExists(atPath: stage.path) { try FileManager.default.removeItem(at: stage) }
            }
        }
    }

    @MainActor static func checkDownloads(root: URL) async throws {
        let fixtures = try StudioImageFixtures(root: root.appendingPathComponent("downloads"))
        let suite = "mlxl3.image-downloads." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        let transport = TestTransfer()
        let engine = ImageGeneration(supported: true, preferences: prefs, catalog: fixtures.catalog,
                                     download: { try await transport.install(files: $0, at: $1, progress: $2) })
        let turbo = try fixtures.catalog.package(for: .turbo)
        let base = try fixtures.catalog.package(for: .base)
        let customBase = root.appendingPathComponent("legacy base", isDirectory: true)
        try StudioImageFixtures.publish(base.files, at: customBase)
        prefs.set(customBase.path, forKey: "image21ModelDirectory")
        func wait(for predicate: () -> Bool) async throws {
            let deadline = ContinuousClock.now.advanced(by: .seconds(4))
            while !predicate() && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(10)) }
            precondition(predicate(), "Download state timed out")
        }
        await engine.refreshModelDirectory()
        precondition(engine.variant == .turbo && engine.steps == 8 && engine.modelAvailability == .missing(turbo.totalBytes))
        engine.downloadModel()
        engine.downloadModel()
        try await wait { engine.modelAvailability == .downloading(completed: 1, total: turbo.totalBytes) }
        engine.selectVariant(.base)
        precondition(engine.variant == .turbo && engine.modelDirectory == nil)
        engine.cancelModelDownload()
        try await wait { engine.modelAvailability == .paused(turbo.totalBytes) }
        let reopened = ImageGeneration(supported: true, preferences: prefs, catalog: fixtures.catalog)
        await reopened.refreshModelDirectory()
        precondition(reopened.modelAvailability == .paused(turbo.totalBytes), "The next launch must offer to resume verified files")
        engine.selectVariant(.base)
        await engine.refreshModelDirectory()
        engine.selectVariant(.turbo)
        await engine.refreshModelDirectory()
        precondition(engine.modelAvailability == .paused(turbo.totalBytes))
        let attemptsAfterDiscovery = await transport.attempts
        precondition(attemptsAfterDiscovery == 1, "Reopening or switching a paused download must not start a transfer")
        engine.downloadModel()
        try await wait { if case .failed = engine.modelAvailability { true } else { false } }
        engine.downloadModel()
        let managed = fixtures.catalog.destination(for: .turbo)
        try await wait { engine.modelDirectory == managed }
        engine.downloadModel()
        engine.selectVariant(.base)
        await engine.refreshModelDirectory()
        precondition(engine.modelDirectory?.path == customBase.path)
        engine.downloadModel()
        let calls = await transport.attempts
        precondition(calls == 3, "Only cancellation, failure, and the successful retry reach the transport")
        precondition(!FileManager.default.fileExists(atPath: fixtures.catalog.destination(for: .base).path))
        // Another process can finish installing between the discovery and the click.
        engine.selectVariant(.turbo)
        try FileManager.default.removeItem(at: managed)
        await engine.refreshModelDirectory()
        precondition(engine.modelDirectory == nil)
        try StudioImageFixtures.publish(turbo.files, at: managed)
        engine.downloadModel()
        try await wait { engine.modelDirectory == managed }
        let subsequentCalls = await transport.attempts
        precondition(subsequentCalls == 3, "A model installed before the click must be adopted without any transfer")
        print("Image downloads: Turbo default, legacy Base reuse, progress, duplicate click, cancel, retry and external completion checked")
    }

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
        let fixtures = try StudioImageFixtures(root: root)
        try StudioImageFixtures.publish(fixtures.catalog.package(for: .base).files, at: model)
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        let sample = try Data(contentsOf: repo.appendingPathComponent("docs/images/studio-image21-m5.png"))
        try sample.write(to: model.appendingPathComponent("sample.png"))

        let worker = try StudioImageFixtures.worker(in: root)
        let suite = "mlxl3.turbo-test.\(UUID().uuidString)"
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        prefs.set(model.path, forKey: "image21ModelDirectory")
        prefs.set("base", forKey: "image21Variant")
        let turbo = fixtures.catalog.destination(for: .turbo)
        let engine = ImageGeneration(supported: true, lowMemoryAvailable: true, historyDirectory: images, worker: worker,
                                     preferences: prefs, catalog: fixtures.catalog)
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
                                            historyDirectory: images, worker: worker, preferences: prefs, catalog: fixtures.catalog)
        check(!otherHardware.offersLowMemory(at: 1024))
        otherHardware.start(model: model, prompt: "--low-memory", size: 1024, steps: 2, seed: 123, lowMemory: true)
        try await wait { !otherHardware.running }
        check(otherHardware.error == "low-memory=0")

        await engine.refreshModelDirectory()
        await engine.refreshHistory()
        check(engine.variant == .base && engine.modelDirectory?.path == model.path && engine.steps == 40)
        check(engine.history.isEmpty && !engine.isLoadingHistory && engine.historyError == nil)
        check(!fm.fileExists(atPath: images.path), "Opening an empty gallery must not create files")
        engine.start(model: model, prompt: "preview", size: 512, steps: 2, seed: 12)
        // A changed preference applies to the next render.
        prefs.set(true, forKey: ImageGeneration.autoSavePreferenceKey)
        try await wait { !engine.running }
        guard let preview = engine.result else { preconditionFailure(engine.error ?? "Missing preview") }
        check(preview.png == sample && preview.seed == 12 && preview.savedURL == nil && engine.error == nil,
              "A render with default settings must return an exact in-memory preview")
        let scratch = try String(contentsOf: model.appendingPathComponent("last-output"), encoding: .utf8)
        check(!fm.fileExists(atPath: URL(fileURLWithPath: scratch).deletingLastPathComponent().path))
        check(engine.history.isEmpty && !fm.fileExists(atPath: images.path))
        let export = root.appendingPathComponent("exported.png")
        for _ in 0..<2 { try await preview.export(to: export) }
        let exported = try Data(contentsOf: export)
        check(exported == sample && !fm.fileExists(atPath: images.path))
        do {
            try await preview.export(to: root.appendingPathComponent("absent/fails.png"))
            preconditionFailure("A failed export must be reported")
        } catch { check(engine.result?.id == preview.id) }
        engine.start(model: model, prompt: "corrupt", size: 512, steps: 2, seed: 0)
        try await wait { !engine.running }
        check(engine.error != nil && engine.result?.id == preview.id, "A failed retry must preserve the unsaved preview")
        let previewCounts = try engine.usageStore.summary()
        check(previewCounts.base.allTime == 1 && previewCounts.turbo.allTime == 0 && engine.usageRevision == 1)
        let prompt = "Lune 🌙, lignes ‘bleues’.\nDeuxième ligne."
        engine.start(model: model, prompt: prompt, size: 512, steps: 2, seed: 0)
        engine.prompt = "a changed draft"
        engine.seed = 10
        try await wait { !engine.running }
        guard let first = engine.history.first else { preconditionFailure(engine.error ?? "Missing history") }
        check(engine.result?.savedURL == first.fileURL && first.prompt == prompt && first.steps == 2 && first.seed == 0)
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
        await engine.refreshModelDirectory()
        check(engine.modelDirectory == nil && engine.steps == 8, "Turbo must not silently use the Base folder")
        try fm.createDirectory(at: turbo, withIntermediateDirectories: true)
        try StudioImageFixtures.publish(fixtures.catalog.package(for: .turbo).files, at: turbo)
        try sample.write(to: turbo.appendingPathComponent("sample.png"))
        await engine.refreshModelDirectory()
        check(engine.modelDirectory == turbo, "An installed Turbo folder is discovered without a picker")
        for wrongSteps in [1, 7, 9, 40, 51] {
            engine.start(model: turbo, prompt: "turbo", size: 512, steps: wrongSteps, seed: 123)
            try await wait { !engine.running }
            check(engine.error == "Turbo uses 8 steps." && !fm.fileExists(atPath: turbo.appendingPathComponent("last-prompt").path))
        }
        for size in [512, 768, 1024] {
            check(!engine.offersLowMemory(at: size), "Only Base offers the measured memory policy")
            engine.start(model: turbo, prompt: "--low-memory", size: size, steps: 8, seed: 123, lowMemory: true)
            try await wait { !engine.running }
            check(engine.error == "low-memory=0", "Turbo must keep its native memory policy at every size")
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
        await engine.selectModelDirectory(custom)
        check(engine.modelDirectory == turbo && engine.folderError != nil, "An empty custom folder must not replace an installed model")
        let flat = custom.appendingPathComponent("4bit", isDirectory: true)
        try StudioImageFixtures.publish(fixtures.catalog.package(for: .turbo).files, at: flat)
        await engine.selectModelDirectory(custom)
        check(engine.modelDirectory == flat && prefs.string(forKey: "image21TurboModelDirectory") == flat.path,
              "Selecting a snapshot parent must store its actual 4bit folder")
        let restored = ImageGeneration(supported: false, historyDirectory: images, preferences: prefs, catalog: fixtures.catalog)
        await restored.refreshModelDirectory()
        check(restored.variant == .turbo && restored.modelDirectory == flat && restored.steps == 8)
        await restored.refreshHistory()
        check(restored.history.first == fast && restored.history.count == 3)
        let restoredCounts = try restored.usageStore.summary()
        check(restoredCounts == ImageUsageSummary(base: .init(last30Days: 3, allTime: 3),
                                                  turbo: .init(last30Days: 1, allTime: 1)) && engine.usageRevision == 4,
              "Relaunch counts validated renders once across failed, cancelled and repeated operations")
        engine.selectVariant(.base)
        await engine.refreshModelDirectory()
        check(engine.steps == 31 && engine.modelDirectory?.path == model.path, "Switching back must restore Base settings")
        let store = GeneratedImageHistory(directory: images)
        do {
            _ = try store.validatedPNG(at: root.appendingPathComponent("missing.png"), size: 512)
            preconditionFailure("A worker with no output must report an unreadable image")
        } catch { check(error.localizedDescription == "The engine returned an unreadable image.") }
        let duplicate = root.appendingPathComponent("duplicate.png")
        try sample.write(to: duplicate)
        do {
            _ = try store.save(png: duplicate, prompt: "changed", modelName: "Other",
                               size: 512, steps: 2, seed: first.seed!, id: GeneratedImageHistory.filenameIdentity(first.fileURL)!.id)
            preconditionFailure("Duplicate publication overwrote a previous render")
        } catch {
            let records = try store.records()
            check(records.contains(first), "A repeated publication must preserve the original settings and image")
        }
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
        let renamedPNG = try Data(contentsOf: repo.appendingPathComponent("tests/fixtures/image-input/red.png"))
        try sample.write(to: legacy)
        try renamedPNG.write(to: unknown)
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
        let download = root.appendingPathComponent("download.png")
        let originalID = try fm.attributesOfItem(atPath: first.fileURL.path)[.systemFileNumber] as? NSNumber
        let originalSettings = try Data(contentsOf: firstInfo)
        let countsBeforeExport = try reopened.usageStore.summary()
        for (record, expected) in [(first, sample), (old, sample), (renamed, renamedPNG)] {
            try Data("previous destination".utf8).write(to: download)
            try await record.export(to: download)
            let copy = try Data(contentsOf: download)
            check(copy == expected, "Download must preserve the original PNG bytes, including legacy images")
        }
        let sameFile = root.appendingPathComponent("linked.png")
        try fm.linkItem(at: first.fileURL, to: sameFile)
        for destination in [first.fileURL, sameFile] {
            try await first.export(to: destination)
            let fileNumber = try fm.attributesOfItem(atPath: destination.path)[.systemFileNumber] as? NSNumber
            check(originalID != nil && originalID == fileNumber, "An export onto its source must preserve the file identity")
        }
        let settingsAfterExport = try Data(contentsOf: firstInfo)
        let countsAfterExport = try reopened.usageStore.summary()
        check(settingsAfterExport == originalSettings && countsAfterExport == countsBeforeExport)
        try fm.removeItem(at: first.fileURL)
        do {
            try await first.export(to: download)
            preconditionFailure("A missing source must report an export error")
        } catch {
            let surviving = try Data(contentsOf: download)
            check(surviving == renamedPNG, "A failed download must preserve an existing destination")
        }
        await reopened.refreshHistory()
        check(reopened.history.count == 4 && !reopened.history.contains(first), "Missing PNGs are not gallery entries")
        let blocked = ImageGeneration(historyDirectory: unknown)
        await blocked.refreshHistory()
        check(!blocked.isLoadingHistory && blocked.history.isEmpty && blocked.historyError != nil)
        let inaccessibleGallery = root.appendingPathComponent("unavailable-gallery")
        try Data().write(to: inaccessibleGallery)
        let recovery = ImageGeneration(supported: true, historyDirectory: inaccessibleGallery, worker: worker,
                                       preferences: prefs, catalog: fixtures.catalog)
        recovery.start(model: model, prompt: "save failure", size: 512, steps: 2, seed: 23)
        try await wait { !recovery.running }
        check(recovery.error?.hasPrefix("Automatic saving failed.") == true && recovery.history.isEmpty)
        check(recovery.result?.savedURL == nil && recovery.result?.png == sample)
        try fm.removeItem(at: inaccessibleGallery)
        let recoveryCounts = try recovery.usageStore.summary()
        check(recoveryCounts.base.allTime == 1 && recovery.usageRevision == 1)
        print("Image history and usage: \(assertions) assertions, worker cleanup, unsaved preview, explicit export, automatic save, exact PNG, cancellation, failure, relaunch and legacy records checked")
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
        let suite = "mlxl3.image-errors." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        preferences.set("base", forKey: "image21Variant")
        let engine = ImageGeneration(supported: true, preferences: preferences)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for (supported, steps, message) in [(false, 40, Image21Hardware.requirement),
                                             (true, 1, "Choose between 2 and 50 steps.")] {
            let rejected = ImageGeneration(supported: supported, preferences: preferences)
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
        try await checkDownloads(root: root)
        try await checkHistory(root: root)
        print("Image generation: overlapping request, missing model and retry checked")
    }
}
