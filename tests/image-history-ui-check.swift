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
        let engine = ImageGeneration(supported: true, historyDirectory: images)
        let host = NSHostingView(rootView: ImageGenerationView(engine: engine, close: {}).environmentObject(studio)
            .defaultAppStorage(preferences).preferredColorScheme(.dark))
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
        func words() async throws -> [Word] {
            try await Task.sleep(for: .milliseconds(180))
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                throw HistoryUIError(message: "Image workspace did not render")
            }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            guard let image = bitmap.cgImage else { throw HistoryUIError(message: "Empty workspace bitmap") }
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

        try click("history", in: await words())
        try expect(["image history", "no images"], in: await words())
        let sample = temporary.appendingPathComponent("worker.png")
        try fileManager.copyItem(at: example, to: sample)
        let store = GeneratedImageHistory(directory: images)
        let saved = try store.save(png: sample, prompt: "A cobalt blue teapot", modelName: "Fixture model",
                                   size: 512, steps: 17, seed: 371)
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
        try expect(["a cobalt blue teapot", "fixture model", "512", "17", "371", "show png"], in: details)

        window.setContentSize(.init(width: 820, height: 700))
        try click("all images", in: await words())
        let narrow = try await words()
        try expect(["a cobalt blue teapot"], in: narrow)
        try click("a cobalt blue teapot", in: narrow)
        try click("reuse settings", in: await words())
        try await Task.sleep(for: .milliseconds(120))
        guard engine.prompt == "A cobalt blue teapot", engine.size == 512, engine.steps == 17, engine.seed == 371,
              !engine.running, fileManager.fileExists(atPath: saved.fileURL.path),
              fileManager.fileExists(atPath: legacy.path) else {
            throw HistoryUIError(message: "Reusing a saved entry must restore the draft and preserve both images")
        }
        print("Image history interface passed: empty library, image and prompt, saved settings, compact navigation, reuse without generation")
    }
}
