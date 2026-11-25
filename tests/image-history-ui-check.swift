import AppKit
import SwiftUI
import Vision

private struct HistoryUIError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

@main
struct ImageHistoryInterfaceCheck {
    @MainActor static func main() {
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        Task { @MainActor in
            do { try await checkWorkspace(); exit(EXIT_SUCCESS) }
            catch {
                FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
                exit(EXIT_FAILURE)
            }
        }
        app.run()
        exit(EXIT_FAILURE)
    }

    @MainActor static func checkWorkspace() async throws {
        guard CommandLine.arguments.count == 2 else { throw HistoryUIError(message: "Expected a 512 px sample PNG") }
        let example = URL(fileURLWithPath: CommandLine.arguments[1])
        let fileManager = FileManager.default
        let temporary = fileManager.temporaryDirectory.appendingPathComponent("mlxl3-history-ui-\(UUID())")
        let images = temporary.appendingPathComponent("Images")
        try fileManager.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: temporary) }
        let suite = "io.mlxl3.history-ui.\(UUID())"
        guard let preferences = UserDefaults(suiteName: suite) else { throw HistoryUIError(message: "No test preferences") }
        defer { preferences.removePersistentDomain(forName: suite) }
        let studio = StudioModel(conversationFileURL: temporary.appendingPathComponent("chat"), isPreview: true, preferences: preferences)
        studio.setLanguage(.en)
        studio.setStyle(.graphite)
        let fixtures = try StudioImageFixtures(root: temporary)
        let engine = ImageGeneration(supported: true, historyDirectory: images,
                                     worker: try StudioImageFixtures.worker(in: temporary), preferences: preferences,
                                     catalog: fixtures.catalog, download: { files, destination, progress in
            await progress(1, files.reduce(0) { $0 + $1.size })
            try StudioImageFixtures.publish(files, at: destination)
        })
        let host = NSHostingView(rootView: AnyView(ImageGenerationView(engine: engine, close: {}).environmentObject(studio)
            .defaultAppStorage(preferences).preferredColorScheme(.dark)))
        let window = NSWindow(contentRect: .init(x: 0, y: 0, width: 1120, height: 760),
                              styleMask: [.titled, .resizable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.setFrameOrigin(.init(x: -6000, y: -6000))
        window.orderBack(nil)
        defer { window.close() }

        typealias Word = (text: String, box: CGRect)
        let ocr = VNRecognizeTextRequest()
        ocr.recognitionLevel = .accurate
        ocr.recognitionLanguages = ["en-US"]
        ocr.usesLanguageCorrection = false
        for (stage, devices) in try ocr.supportedComputeStageDevices {
            guard let cpu = devices.first(where: { if case .cpu = $0 { true } else { false } }) else {
                throw HistoryUIError(message: "CPU OCR unavailable")
            }
            ocr.setComputeDevice(cpu, for: stage)
        }
        var frame = 0
        func words(snapshot: String? = nil) async throws -> [Word] {
            try await Task.sleep(for: .milliseconds(180))
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                throw HistoryUIError(message: "Image workspace did not render")
            }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            guard let image = bitmap.cgImage else { throw HistoryUIError(message: "Empty workspace bitmap") }
            // MLXL3_IMAGE_HISTORY_SNAPSHOT_DIR keeps every frame the check read, for a visual review.
            if let folder = ProcessInfo.processInfo.environment["MLXL3_IMAGE_HISTORY_SNAPSHOT_DIR"] {
                frame += 1
                try fileManager.createDirectory(atPath: folder, withIntermediateDirectories: true)
                try bitmap.representation(using: .png, properties: [:])?.write(
                    to: URL(fileURLWithPath: folder).appendingPathComponent(snapshot ?? "image-history-\(frame).png"))
            }
            try VNImageRequestHandler(cgImage: image).perform([ocr])
            return (ocr.results ?? []).compactMap { item in
                item.topCandidates(1).first.map { ($0.string.lowercased(), item.boundingBox) }
            }
        }
        func click(_ label: String, in cells: [Word]) throws {
            guard let cell = cells.first(where: { $0.text.contains(label) }) else {
                throw HistoryUIError(message: "Missing control '\(label)' in \(cells.map(\.text))")
            }
            let location = CGPoint(x: cell.box.midX * host.bounds.width, y: cell.box.midY * host.bounds.height)
            func event(_ type: NSEvent.EventType) -> NSEvent {
                NSEvent.mouseEvent(with: type, location: location, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                                   windowNumber: window.windowNumber, context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
            }
            NSApp.postEvent(event(.leftMouseUp), atStart: false)
            window.sendEvent(event(.leftMouseDown))
            if let up = NSApp.nextEvent(matching: .leftMouseUp, until: .distantPast, inMode: .default, dequeue: true) {
                window.sendEvent(up)
            }
        }
        func expect(_ snippets: [String], in cells: [Word]) throws {
            let text = cells.map(\.text).joined(separator: " ")
            for snippet in snippets where !text.contains(snippet) {
                throw HistoryUIError(message: "Missing '\(snippet)' in '\(text)'")
            }
        }

        func variantPicker(in view: NSView) -> NSSegmentedControl? {
            if let picker = view as? NSSegmentedControl, picker.segmentCount == 2 { return picker }
            for child in view.subviews {
                if let picker = variantPicker(in: child) { return picker }
            }
            return nil
        }
        let missing = try await words()
        try expect(["base", "turbo", "recommended", "8 steps", "download"], in: missing)
        guard engine.variant == .turbo && engine.modelDirectory == nil else { throw HistoryUIError(message: "First launch must offer Turbo") }
        try click("download", in: missing.filter { $0.box.midX < 0.25 })
        let deadline = ContinuousClock.now.advanced(by: .seconds(4))
        while engine.modelDirectory == nil && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(20)) }
        try expect(["installed on this mac"], in: await words())
        guard engine.modelDirectory == fixtures.catalog.destination(for: .turbo) else { throw HistoryUIError(message: "Download button did not install Turbo") }
        studio.setLanguage(.fr)
        ocr.recognitionLanguages = ["fr-FR"]
        try expect(["installé sur ce mac"], in: await words(snapshot: "studio-image-turbo-installed-fr.png"))
        studio.setLanguage(.en)
        ocr.recognitionLanguages = ["en-US"]
        guard let picker = variantPicker(in: host) else { throw HistoryUIError(message: "Missing image model picker") }
        picker.selectedSegment = 1
        guard picker.sendAction(picker.action, to: picker.target) else { throw HistoryUIError(message: "Image picker has no action") }
        try expect(["2 to 50", "recreate a base", "download"], in: await words())
        guard engine.variant == .base && engine.steps == 40 else { throw HistoryUIError(message: "Base picker failed") }
        try click("history", in: await words())
        try expect(["image history", "no images"], in: await words())
        let sample = temporary.appendingPathComponent("worker.png")
        try fileManager.copyItem(at: example, to: sample)
        let store = GeneratedImageHistory(directory: images)
        let saved = try store.save(png: sample, prompt: "A cobalt blue teapot", modelName: "Fixture model",
                                   size: 512, steps: 17, seed: 371)
        try fileManager.copyItem(at: example, to: sample)
        let fast = try store.save(png: sample, prompt: "A mint bicycle", modelName: "Fixture model", variant: .turbo,
                                  size: 512, steps: 8, seed: 492)
        let legacy = images.appendingPathComponent("Image-29-\(UUID().uuidString).png")
        try fileManager.copyItem(at: example, to: legacy)
        await engine.refreshHistory()
        let grid = try await words()
        try expect(["a cobalt blue teapot"], in: grid)
        try click("generated image", in: grid)
        let older = try await words()
        try expect(["prompt unavailable", "512", "29"], in: older)
        guard !older.contains(where: { $0.text.contains("reuse settings") }) else {
            throw HistoryUIError(message: "A legacy PNG offered to reuse unknown settings")
        }
        try click("all images", in: older)
        try click("a cobalt blue teapot", in: await words())
        let details = try await words()
        try expect(["a cobalt blue teapot", "fixture model", "base", "512", "17", "371", "show png"], in: details)

        window.setContentSize(.init(width: 820, height: 700))
        try click("all images", in: await words())
        let narrow = try await words()
        try expect(["a cobalt blue teapot"], in: narrow)
        try click("a cobalt blue teapot", in: narrow)
        try click("reuse settings", in: await words())
        try await Task.sleep(for: .milliseconds(120))
        guard engine.prompt == "A cobalt blue teapot", engine.variant == .base, engine.size == 512, engine.steps == 17, engine.seed == 371,
              !engine.running, fileManager.fileExists(atPath: saved.fileURL.path),
              fileManager.fileExists(atPath: legacy.path) else {
            throw HistoryUIError(message: "Reusing a saved entry must restore the draft and preserve both images")
        }
        try click("history", in: await words())
        try click("a mint bicycle", in: await words())
        try expect(["variant", "turbo", "492"], in: await words())
        try click("reuse settings", in: await words())
        try expect(["8 steps", "installed on this mac"], in: await words())
        guard engine.variant == .turbo && engine.prompt == "A mint bicycle" && engine.steps == 8 && engine.seed == 492
              && !engine.running && fileManager.fileExists(atPath: fast.fileURL.path) else {
            throw HistoryUIError(message: "Reusing Turbo must select its eight-step generator")
        }
        // One recent image of each variant already exists in the fixture's gallery.
        let now = Date()
        let old = Calendar.current.date(byAdding: .day, value: -40, to: now)!
        for (variant, recent) in [(Image21Variant.base, 11), (.turbo, 35)] {
            for _ in 0..<recent { try engine.usageStore.record(id: UUID(), variant: variant, at: now) }
            for _ in 0..<12 { try engine.usageStore.record(id: UUID(), variant: variant, at: old) }
        }
        func expectImageCounts(_ counts: [Int], for variant: Image21Variant, in cells: [Word]) throws {
            guard let label = cells.first(where: { $0.text == variant.rawValue }) else {
                throw HistoryUIError(message: "Missing \(variant.title) in the Lab")
            }
            let row = cells.filter { $0.box.minX > label.box.midX && abs($0.box.midY - label.box.midY) < 0.016 }
                .sorted { $0.box.minX < $1.box.minX }
            let numbers = row.flatMap { $0.text.split(whereSeparator: { !$0.isNumber }).compactMap { Int($0) } }
            guard numbers == counts else {
                throw HistoryUIError(message: "Expected \(variant.title) \(counts), found \(numbers) in \(cells.map(\.text))")
            }
        }
        func showLab() {
            host.rootView = AnyView(LabWorkspaceView(imageGeneration: engine, onBack: {})
                .background { StudioAurora(busy: false, animated: false, fullHeight: true).background(StudioTheme.canvas) }
                .id(studio.language).environmentObject(studio).environment(\.locale, Locale(identifier: studio.language.rawValue))
                .defaultAppStorage(preferences).preferredColorScheme(.dark))
        }
        showLab()
        try click("images", in: await words())
        let lab = try await words()
        try expect(["generated images", "last 30 days", "all time", "one image in your history is unknown"], in: lab)
        try expectImageCounts([12, 24], for: .base, in: lab)
        try expectImageCounts([36, 48], for: .turbo, in: lab)

        // Keep Lab open while the production controller completes another image.
        let turboFolder = fixtures.catalog.destination(for: .turbo)
        try fileManager.copyItem(at: example, to: turboFolder.appendingPathComponent("sample.png"))
        engine.start(model: turboFolder, prompt: "Usage check", size: 512, steps: 8, seed: 19)
        let finished = ContinuousClock.now.advanced(by: .seconds(12))
        while engine.running && ContinuousClock.now < finished { try await Task.sleep(for: .milliseconds(40)) }
        guard !engine.running && engine.error == nil else {
            engine.cancel()
            throw HistoryUIError(message: "Lab test generation failed: \(engine.error ?? "timeout")")
        }
        try expectImageCounts([37, 49], for: .turbo, in: await words())
        studio.setLanguage(.fr)
        ocr.recognitionLanguages = ["fr-FR"]
        showLab()
        try click("images", in: await words())
        let french = try await words(snapshot: "studio-lab-images-fr.png")
        try expect(["images générées", "30 derniers jours", "depuis le début", "inconnu pour une image"], in: french)
        try expectImageCounts([12, 24], for: .base, in: french)
        try expectImageCounts([37, 49], for: .turbo, in: french)
        let unavailable = temporary.appendingPathComponent("unavailable")
        try Data().write(to: unavailable)
        let fresh = ImageGeneration(supported: false, historyDirectory: unavailable.appendingPathComponent("Images"),
                                    preferences: preferences, catalog: fixtures.catalog)
        func showFreshLab(_ id: String) {
            host.rootView = AnyView(LabImageUsageView(engine: fresh).padding(24)
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .background(StudioTheme.canvas).id(id).preferredColorScheme(.dark))
            window.setContentSize(.init(width: 820, height: 700))
        }
        showFreshLab("unavailable")
        let errorPage = try await words()
        try expect(["indisponibles", "réessayer"], in: errorPage)
        try fileManager.removeItem(at: unavailable)
        try click("réessayer", in: errorPage)
        let empty = try await words()
        // Vision may omit an isolated zero. Check the empty table still exposes both
        // modes and periods; the four distinct non-zero cells were checked above.
        try expect(["base", "turbo", "30 derniers jours", "depuis le début"], in: empty)
        guard try fresh.usageStore.summary() == ImageUsageSummary() else {
            throw HistoryUIError(message: "Retry did not open an empty image ledger")
        }
        for _ in 0..<12 { try fresh.usageStore.record(id: UUID(), variant: .turbo) }
        let missingHistory = unavailable.appendingPathComponent("Images")
        try Data().write(to: missingHistory)
        showFreshLab("partial")
        let partial = try await words()
        try expect(["inaccessible", "réessayer"], in: partial)
        try expectImageCounts([12, 12], for: .turbo, in: partial)
        try fileManager.removeItem(at: missingHistory)
        try click("réessayer", in: partial)
        let recovered = try await words()
        guard !recovered.contains(where: { $0.text.contains("inaccessible") }) else {
            throw HistoryUIError(message: "Retry did not refresh the recovered gallery")
        }
        try expectImageCounts([12, 12], for: .turbo, in: recovered)
        try await checkImageSaveSettings(studio: studio, window: window, preferences: preferences)
        print("Image history interface passed: image library, reuse, Lab counts in both periods and live refresh, saving preference")
    }
}
