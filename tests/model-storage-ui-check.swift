import AppKit
import SwiftUI
import Vision

@MainActor enum ModelStorageInterfaceCheck {
    static func run() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mlxl3-storage-ui-\(UUID())").resolvingSymlinksInPath()
        let managed = root.appendingPathComponent("Models")
        let owned = managed.appendingPathComponent("Qwen-4bpw")
        let external = root.appendingPathComponent("External").appendingPathComponent("Birch")
        let hub = root.appendingPathComponent("hub")
        for folder in [owned, external, hub] { try fm.createDirectory(at: folder, withIntermediateDirectories: true) }
        defer { try? fm.removeItem(at: root) }
        let variables = ["MLXL3_MODELS_DIR": managed.path, "HF_HUB_CACHE": hub.path]
        let original = ProcessInfo.processInfo.environment
        for (key, value) in variables { setenv(key, value, 1) }
        defer {
            for key in variables.keys {
                if let previous = original[key] { setenv(key, previous, 1) } else { unsetenv(key) }
            }
        }
        let ownedName = "Qwen Image"
        let record: [String: Any] = ["attempt": "fixture-id", "request": ["name": ownedName, "destination": owned.path]]
        try JSONSerialization.data(withJSONObject: record).write(to: owned.appendingPathComponent(".mlxl3-install.json"))
        for name in ["config.json", "quantization_config.json"] {
            try Data("{}".utf8).write(to: owned.appendingPathComponent(name))
        }
        let shared = owned.appendingPathComponent("encoder.safetensors")
        try Data(repeating: 0x5a, count: 900_000).write(to: shared)
        try fm.linkItem(at: shared, to: external.appendingPathComponent("encoder.safetensors"))
        try Data(repeating: 0x91, count: 1_100_000).write(to: owned.appendingPathComponent("transformer.safetensors"))
        let suite = "mlxl3.storage-ui.\(UUID())"
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("history.json"), isPreview: true, preferences: prefs)
        let priorLanguage = studio.language, priorStyle = StudioTheme.style
        defer { studio.setLanguage(priorLanguage); StudioTheme.style = priorStyle }
        studio.setLanguage(.en)
        studio.setStyle(.graphite)
        studio.models = [
            LocalModel(name: ownedName, path: owned.path, modelType: "qwen", format: "EXL3", bits: 4, sizeBytes: 2_000_000,
                       modules: 1, addedAt: "", size: "2 MB", installationID: "fixture-id"),
            LocalModel(name: "Birch", path: external.path, modelType: "qwen", format: "EXL3", bits: 4, sizeBytes: 900_000,
                       modules: 1, addedAt: "", size: "900 KB")
        ]
        let inventory = try ModelStorageInventory.scan(models: studio.models, pending: [])
        let entry = inventory.items.first { $0.registryName == ownedName }!
        precondition(entry.reclaimableBytes >= 1_100_000 && entry.reclaimableBytes < 1_500_000)
        let expectedEstimate = ByteCountFormatter.string(fromByteCount: entry.reclaimableBytes, countStyle: .file).lowercased()
        let bounds = NSRect(x: 0, y: 0, width: 900, height: 650)
        let window = NSWindow(contentRect: bounds, styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        let host = NSHostingView(rootView: ModelManagerView().environmentObject(studio)
            .defaultAppStorage(prefs).studioAppearance(.dark))
        window.contentView = host
        window.setFrameOrigin(NSPoint(x: -6000, y: -6000))
        window.orderBack(nil)
        defer { window.close() }

        func snapshot() async throws -> NSBitmapImageRep {
            try await Task.sleep(for: .milliseconds(220))
            host.layoutSubtreeIfNeeded()
            let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
            host.cacheDisplay(in: host.bounds, to: bitmap)
            return bitmap
        }
        typealias Cell = (text: String, box: CGRect)
        func read(_ bitmap: NSBitmapImageRep) throws -> [Cell] {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.recognitionLanguages = ["en-US"]
            request.usesLanguageCorrection = false
            let stages = try request.supportedComputeStageDevices
            precondition(!stages.isEmpty)
            for (stage, choices) in stages {
                guard let cpu = choices.first(where: { if case .cpu = $0 { true } else { false } }) else {
                    preconditionFailure("Missing CPU OCR device")
                }
                request.setComputeDevice(cpu, for: stage)
            }
            try VNImageRequestHandler(cgImage: bitmap.cgImage!).perform([request])
            return (request.results ?? []).compactMap { word in
                word.topCandidates(1).first.map { ($0.string.lowercased(), word.boundingBox) }
            }
        }
        func click(_ cell: Cell) {
            for type in [NSEvent.EventType.leftMouseDown, .leftMouseUp] {
                let position = CGPoint(x: cell.box.midX * bounds.width, y: cell.box.midY * bounds.height)
                let event = NSEvent.mouseEvent(with: type, location: position, modifierFlags: [],
                    timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
                    context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
                window.sendEvent(event)
            }
        }
        let library = try read(await snapshot())
        guard let link = library.first(where: { $0.text.contains("manage storage") }) else {
            preconditionFailure("Library did not offer storage management: \(library.map(\.text))")
        }
        click(link)
        let listing = try read(await snapshot())
        precondition(listing.contains { $0.text.contains("installed models") }
            && listing.contains { $0.text.contains("external folders") }
            && listing.contains { $0.text.contains("recoverable") }
            && listing.contains { $0.text.contains(expectedEstimate) },
            "Storage must show installed/external items and the estimate: \(listing.map(\.text))")
        guard let row = listing.first(where: { $0.text.contains("qwen image") }) else {
            preconditionFailure("Installed model missing from storage")
        }
        click(row)
        let opened = try await snapshot()
        let details = try read(opened)
        let labels = details.map(\.text).joined(separator: " ")
        precondition(labels.contains("shared with birch") && labels.contains("move to trash") && labels.contains("emptying trash"),
                     "Missing ownership or Trash estimate before removal: \(labels)")
        precondition(fm.fileExists(atPath: shared.path) && studio.models.count == 2, "Review removed a model")
        if let destination = ProcessInfo.processInfo.environment["MLXL3_STORAGE_SNAPSHOT_DIR"] {
            try fm.createDirectory(atPath: destination, withIntermediateDirectories: true)
            try opened.representation(using: .png, properties: [:])?.write(to: URL(fileURLWithPath: destination)
                .appendingPathComponent("studio-storage-review.png"))
        }
        guard let externalRow = details.first(where: { $0.text == "birch" }) else {
            preconditionFailure("The external model is missing: \(details.map(\.text))")
        }
        click(externalRow)
        let externalDetails = try read(await snapshot()).map(\.text).joined(separator: " ")
        precondition(externalDetails.contains("managed outside studio") && externalDetails.contains("remove from library")
                     && !externalDetails.contains("move to trash"), "External folder protection missing: \(externalDetails)")
        precondition(fm.fileExists(atPath: shared.path) && studio.models.count == 2)
        print("Storage UI passed: library navigation, ownership, recovery estimate, external folders and non-destructive review")
    }
}
