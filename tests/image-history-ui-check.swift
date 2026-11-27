import AppKit
import SwiftUI
import Vision

private struct HistoryUIError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// The fixture drives the same configured panel as Studio without relying on a window-server
/// dialog on a hosted runner. A physical Mac can also exercise the app-modal interaction.
@MainActor
private final class SavePanelDriver {
    private(set) var requests = 0
    private(set) var filename: String?
    private(set) var directory: URL?
    private(set) var isPNG = false
    var destination: URL?

    func select(_ panel: NSSavePanel) -> URL? {
        requests += 1
        filename = panel.nameFieldStringValue
        directory = panel.directoryURL
        isPNG = panel.allowedContentTypes == [.png] && panel.canCreateDirectories
        return destination
    }
}

@main
struct ImageHistoryInterfaceCheck {
    @MainActor static func main() {
        // Foundation resolves Downloads relative to this disposable home before AppKit starts.
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-history-ui-\(UUID())")
        setenv("CFFIXED_USER_HOME", temporary.path, 1)
        let app = NSApplication.shared
        app.setActivationPolicy(.accessory)
        Task { @MainActor in
            do { try await checkWorkspace(temporary: temporary); exit(EXIT_SUCCESS) }
            catch {
                FileHandle.standardError.write(Data((error.localizedDescription + "\n").utf8))
                exit(EXIT_FAILURE)
            }
        }
        app.run()
        exit(EXIT_FAILURE)
    }

    @MainActor static func checkWorkspace(temporary: URL) async throws {
        guard CommandLine.arguments.count == 2 else { throw HistoryUIError(message: "Expected a 512 px sample PNG") }
        let example = URL(fileURLWithPath: CommandLine.arguments[1])
        let fileManager = FileManager.default
        let images = temporary.appendingPathComponent("Images")
        let downloads = temporary.appendingPathComponent("Downloads", isDirectory: true)
        try fileManager.createDirectory(at: downloads, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: temporary) }
        guard URL.downloadsDirectory.resolvingSymlinksInPath().path == downloads.resolvingSymlinksInPath().path else {
            throw HistoryUIError(message: "The export check needs an isolated Downloads folder")
        }
        let suite = "io.mlxl3.history-ui.\(UUID())"
        guard let preferences = UserDefaults(suiteName: suite) else { throw HistoryUIError(message: "No test preferences") }
        defer { preferences.removePersistentDomain(forName: suite) }
        let studio = StudioModel(conversationFileURL: temporary.appendingPathComponent("chat"), isPreview: true, preferences: preferences)
        studio.setLanguage(.en)
        studio.setStyle(.graphite)
        let fixtures = try StudioImageFixtures(root: temporary)
        let engine = ImageGeneration(supported: true, lowMemoryAvailable: true, historyDirectory: images,
                                     worker: try StudioImageFixtures.worker(in: temporary), preferences: preferences,
                                     catalog: fixtures.catalog, download: { files, destination, progress in
            await progress(1, files.reduce(0) { $0 + $1.size })
            try StudioImageFixtures.publish(files, at: destination)
        })
        let dialog = SavePanelDriver()
        let useSystemPanel = ProcessInfo.processInfo.environment["MLXL3_TEST_NATIVE_IMAGE_PANEL"] == "1"
        let select: @MainActor (NSSavePanel) -> URL? = useSystemPanel ? selectImageSaveDestination : dialog.select
        func workspace() -> AnyView {
            AnyView(ImageGenerationView(engine: engine, close: {}, selectDestination: select).environmentObject(studio)
                .environment(\.locale, Locale(identifier: studio.language.rawValue))
                .defaultAppStorage(preferences).preferredColorScheme(.dark))
        }
        let host = NSHostingView(rootView: workspace())
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

        func download(_ cells: [Word], filename: String, accept: Bool, source: URL? = nil) async throws {
            let suggested = downloads.appendingPathComponent(filename)
            let destination = accept && !useSystemPanel ? temporary.appendingPathComponent("Export " + filename) : suggested
            let before = try engine.usageStore.summary()
            let gallery = engine.history
            if useSystemPanel {
                var matched = false
                let reply = Timer(timeInterval: 0.1, repeats: true) { _ in
                    MainActor.assumeIsolated {
                        guard !matched, let panel = NSApp.modalWindow as? NSSavePanel, panel.isVisible,
                              panel.allowedContentTypes == [.png], panel.nameFieldStringValue == filename,
                              panel.url?.resolvingSymlinksInPath().path == destination.resolvingSymlinksInPath().path else { return }
                        matched = true
                        // AppKit hosts the system Save button in a separate process.
                        if accept { NSApp.stopModal(withCode: .OK) } else { panel.cancel(nil) }
                    }
                }
                let timeout = Timer(fire: .now.addingTimeInterval(5), interval: 0, repeats: false) { _ in
                    MainActor.assumeIsolated { (NSApp.modalWindow as? NSSavePanel)?.cancel(nil) }
                }
                RunLoop.main.add(reply, forMode: .modalPanel)
                RunLoop.main.add(timeout, forMode: .modalPanel)
                defer { reply.invalidate(); timeout.invalidate() }
                try click("download png", in: cells)
                let shown = ContinuousClock.now.advanced(by: .seconds(6))
                while !matched && ContinuousClock.now < shown { try await Task.sleep(for: .milliseconds(25)) }
                guard matched else { throw HistoryUIError(message: "The native PNG save panel did not show the selected filename and destination") }
            } else {
                let requests = dialog.requests
                dialog.destination = accept ? destination : nil
                try click("download png", in: cells)
                let deadline = ContinuousClock.now.advanced(by: .seconds(4))
                while dialog.requests == requests && ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(25)) }
                guard dialog.requests == requests + 1 else { throw HistoryUIError(message: "Download PNG did not request one save dialog") }
                guard dialog.isPNG, dialog.filename == filename else {
                    throw HistoryUIError(message: "Download PNG configured an incorrect format or filename: \(dialog.filename ?? "nil")")
                }
                guard dialog.directory?.resolvingSymlinksInPath().path == downloads.resolvingSymlinksInPath().path else {
                    throw HistoryUIError(message: "Download PNG did not request the Downloads directory: \(dialog.directory?.path ?? "nil")")
                }
            }
            if accept {
                let deadline = ContinuousClock.now.advanced(by: .seconds(4))
                while !fileManager.fileExists(atPath: destination.path) && ContinuousClock.now < deadline {
                    try await Task.sleep(for: .milliseconds(25))
                }
                guard try Data(contentsOf: destination) == Data(contentsOf: source ?? example) else {
                    throw HistoryUIError(message: "The downloaded PNG differed from the selected image")
                }
            } else if fileManager.fileExists(atPath: destination.path) {
                throw HistoryUIError(message: "Cancelling downloaded a PNG")
            }
            guard try before == engine.usageStore.summary(), engine.history == gallery else {
                throw HistoryUIError(message: "Export changed the gallery or the Lab counters")
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
        engine.size = .square(2048)
        try expect(["2048", "réduire la mémoire"],
                   in: await words(snapshot: "studio-image-turbo-native-size-fr.png"))
        studio.setLanguage(.en)
        ocr.recognitionLanguages = ["en-US"]
        engine.size = .square(1024)
        let turbo1024 = try await words(snapshot: "studio-image-turbo-memory-1024.png")
        try expect(["reduce memory use"], in: turbo1024)
        engine.prompt = "--low-memory"
        func checkMemorySubmission(_ expected: Int, size: String) async throws {
            let controls = try await words()
            try click("generate", in: controls.filter { $0.box.midX > 0.5 && $0.box.midY < 0.3 })
            let submitted = ContinuousClock.now.advanced(by: .seconds(12))
            while (engine.running || engine.error == nil) && ContinuousClock.now < submitted {
                try await Task.sleep(for: .milliseconds(25))
            }
            guard !engine.running, engine.error?.contains("low-memory=\(expected)") == true else {
                engine.cancel()
                throw HistoryUIError(message: "Wrong memory mode for \(engine.variant): \(engine.error ?? "no response")")
            }
            guard let model = engine.modelDirectory,
                  try String(contentsOf: model.appendingPathComponent("last-size"), encoding: .utf8) == size else {
                throw HistoryUIError(message: "The \(engine.variant) worker did not receive size \(size)")
            }
        }
        try await checkMemorySubmission(0, size: "1024")
        try click("reduce memory use", in: await words())
        studio.setLanguage(.fr)
        ocr.recognitionLanguages = ["fr-FR"]
        try expect(["réduire la mémoire"], in: await words(snapshot: "studio-image-turbo-memory-1024-enabled-fr.png"))
        studio.setLanguage(.en)
        ocr.recognitionLanguages = ["en-US"]
        try await checkMemorySubmission(1, size: "1024")
        let twoK = Image21Size.square(2048)
        engine.size = twoK
        let native = try await words()
        try expect(["2048", "reduce memory use"], in: native)
        try await checkMemorySubmission(1, size: "2048")
        engine.prompt = ""
        let reference = temporary.appendingPathComponent("Retouch reference.png")
        try fileManager.copyItem(at: example, to: reference)
        let imageView = ImageGenerationView(engine: engine, close: {})
        try expect(["drop an image"], in: await words())
        guard !imageView.acceptReferenceDrop([reference, reference]),
              !imageView.acceptReferenceDrop([URL(string: "https://example.com/reference.png")!]),
              imageView.acceptReferenceDrop([reference]) else {
            throw HistoryUIError(message: "Reference drop must accept exactly one local file")
        }
        let importDeadline = ContinuousClock.now.advanced(by: .seconds(4))
        while engine.referenceImage == nil && engine.referenceError == nil && ContinuousClock.now < importDeadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        let editControls = try await words(snapshot: "studio-image-retouch-en.png")
        try expect(["edit an image", "retouch reference.png", "describe what to change", "512"], in: editControls)
        guard !editControls.contains(where: { $0.text.contains("reduce memory") }) else {
            throw HistoryUIError(message: "Retouching must hide the 2K memory switch")
        }
        guard !engine.referenceImporting, engine.referenceImage?.name == "Retouch reference.png", engine.size == twoK else {
            throw HistoryUIError(message: "Import must enter edit mode without changing the text-to-image size")
        }
        engine.removeReference()
        try expect(["generate an image", "drop an image", "reduce memory use"], in: await words())
        engine.size = .square(1024)
        engine.prompt = ""
        guard let picker = variantPicker(in: host) else { throw HistoryUIError(message: "Missing image model picker") }
        picker.selectedSegment = 1
        guard picker.sendAction(picker.action, to: picker.target) else { throw HistoryUIError(message: "Image picker has no action") }
        let base1024 = try await words(snapshot: "studio-image-base-memory-1024.png")
        try expect(["2 to 50", "recreate a base", "download", "reduce memory use"], in: base1024)
        try click("download", in: base1024.filter { $0.box.midX < 0.25 })
        let baseReady = ContinuousClock.now.advanced(by: .seconds(4))
        while engine.modelDirectory == nil && ContinuousClock.now < baseReady { try await Task.sleep(for: .milliseconds(20)) }
        guard engine.modelDirectory == fixtures.catalog.destination(for: .base) else { throw HistoryUIError(message: "Base download failed") }

        engine.prompt = "--low-memory"  // The fixture worker reports how many independent flags it received.
        try await checkMemorySubmission(0, size: "1024")  // Selecting Turbo's option does not opt Base in.
        try click("reduce memory use", in: await words())
        try await checkMemorySubmission(1, size: "1024")
        engine.prompt = ""
        engine.size = .square(768)
        let base768 = try await words()
        guard !base768.contains(where: { $0.text.contains("reduce memory") }) else {
            throw HistoryUIError(message: "768 px uses the automatic Base policy and must not offer a switch")
        }
        engine.size = .square(512)
        guard engine.variant == .base && engine.steps == 40 else { throw HistoryUIError(message: "Base picker failed") }
        try expect(["use turbo"], in: await words())
        guard !imageView.acceptReferenceDrop([reference]) else {
            throw HistoryUIError(message: "Base accepted a Turbo-only image drop")
        }
        try click("history", in: await words())
        try expect(["image history", "no images"], in: await words())
        let sample = temporary.appendingPathComponent("worker.png")
        try fileManager.copyItem(at: example, to: sample)
        let store = GeneratedImageHistory(directory: images)
        let saved = try store.save(png: sample, prompt: "A cobalt blue teapot", modelName: "Fixture model",
                                   size: .square(512), steps: 17, seed: 371)
        try StudioImageFixtures.writePNG(to: sample, width: twoK.width, height: twoK.height)
        let fast = try store.save(png: sample, prompt: "A mint bicycle", modelName: "Fixture model", variant: .turbo,
                                  size: twoK, steps: 8, seed: 492)
        let legacy = images.appendingPathComponent("Image-29-\(UUID().uuidString).png")
        try fileManager.copyItem(at: example, to: legacy)
        await engine.refreshHistory()
        let grid = try await words()
        try expect(["a cobalt blue teapot"], in: grid)
        try click("generated image", in: grid)
        let older = try await words()
        try expect(["prompt unavailable", "512", "29"], in: older)
        try await download(older, filename: "Image-29.png", accept: false)
        guard !older.contains(where: { $0.text.contains("reuse settings") }) else {
            throw HistoryUIError(message: "A legacy PNG offered to reuse unknown settings")
        }
        try click("all images", in: older)
        try click("a cobalt blue teapot", in: await words())
        let details = try await words()
        try expect(["a cobalt blue teapot", "fixture model", "base", "512", "17", "371", "show png"], in: details)
        try await download(details, filename: "Image-371.png", accept: true)
        guard !fileManager.fileExists(atPath: downloads.appendingPathComponent("Image-29.png").path) else {
            throw HistoryUIError(message: "The cancelled export created a file")
        }
        studio.setLanguage(.fr)
        host.rootView = workspace()
        ocr.recognitionLanguages = ["fr-FR"]
        try expect(["télécharger le png", "réutiliser les réglages"],
                   in: await words(snapshot: "studio-image-history-download-fr.png"))
        studio.setLanguage(.en)
        host.rootView = workspace()
        ocr.recognitionLanguages = ["en-US"]

        window.setContentSize(.init(width: 820, height: 700))
        try click("all images", in: await words())
        let narrow = try await words()
        try expect(["a cobalt blue teapot"], in: narrow)
        try click("a cobalt blue teapot", in: narrow)
        try click("reuse settings", in: await words())
        try await Task.sleep(for: .milliseconds(120))
        guard engine.prompt == "A cobalt blue teapot", engine.variant == .base, engine.size == .square(512), engine.steps == 17, engine.seed == 371,
              !engine.running, fileManager.fileExists(atPath: saved.fileURL.path),
              fileManager.fileExists(atPath: legacy.path) else {
            throw HistoryUIError(message: "Reusing a saved entry must restore the draft and preserve both images")
        }
        try click("history", in: await words())
        try click("a mint bicycle", in: await words())
        let twoKDetails = try await words()
        try expect(["variant", "turbo", "2048", "492"], in: twoKDetails)
        try await download(twoKDetails, filename: "Image-492.png", accept: true, source: fast.fileURL)
        try click("reuse settings", in: twoKDetails)
        try expect(["8 steps", "installed on this", "2048"], in: await words())
        guard engine.variant == .turbo && engine.prompt == "A mint bicycle" && engine.steps == 8 && engine.seed == 492
              && engine.modelDirectory == fixtures.catalog.destination(for: .turbo)
              && engine.size == twoK && !engine.running && fileManager.fileExists(atPath: fast.fileURL.path) else {
            throw HistoryUIError(message: "Reusing Turbo must restore its 2048 px setting and eight steps")
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
        engine.start(model: turboFolder, prompt: "Usage check", size: .square(512), steps: 8, seed: 19)
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
        print("Image history interface passed: reference drop and Turbo settings, PNG download and cancellation, image library, reuse, Lab counts in both periods and live refresh, saving preference")
    }
}
