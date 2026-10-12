import AppKit
import SwiftUI

@main struct HeaderCheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-pr30-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        setenv("MLXL3_HOME", root.path, 1)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "io.mlxl3.pr30." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        prefs.set(true, forKey: "studio.sidebarVisible")
        defer { prefs.removePersistentDomain(forName: suite) }
        let out = URL(fileURLWithPath: CommandLine.arguments[2], isDirectory: true)
        func wait(_ message: String, _ done: () -> Bool) async throws {
            for _ in 0..<400 {
                if done() { return }
                try await Task.sleep(for: .milliseconds(10))
            }
            throw NSError(domain: "PR30", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
        }
        func model(_ name: String) async throws -> StudioModel {
            let model = StudioModel(conversationFileURL: root.appendingPathComponent(name + ".json"), preferences: prefs)
            model.models = [LocalModel(name: name, path: root.path, modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
            model.selectModel(name)
            try await wait("fixture ready", { model.engineState.isReady && !model.mtpDownloading })
            model.setMTPEnabled(true)
            try await wait("head configured", { !model.mtpDownloading && model.canTuneMTP })
            return model
        }
        func capture(_ model: StudioModel, _ name: String, width: CGFloat) throws {
            let host = NSHostingView(rootView: StudioView().environmentObject(model).defaultAppStorage(prefs).preferredColorScheme(.dark))
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: 700), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: width, height: 700)
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { throw NSError(domain: "PR30", code: 2) }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            guard let data = bitmap.representation(using: .png, properties: [:]) else { throw NSError(domain: "PR30", code: 3) }
            try data.write(to: out.appendingPathComponent(name + ".png"))
            window.close()
        }
        var records: [[String: Any]] = []
        for name in ["mtp-error", "mtp-malformed", "mtp-good"] {
            let studio = try await model(name)
            defer { studio.ejectModel() }
            precondition(!studio.showInspector)
            if name == "mtp-error" { try capture(studio, "ready-820", width: 820) }
            // WorkspaceHeader's idle button calls exactly this production entry point.
            studio.tuneMTP()
            precondition(studio.isTuningMTP)
            try await wait("tune completed", { !studio.isTuningMTP })
            precondition(studio.engineState.isReady && studio.canTuneMTP)
            if name != "mtp-good" { precondition(studio.mtpError != nil && studio.mtpTuneRows.isEmpty) }
            else { precondition(studio.mtpTuneRows.count == 4 && studio.mtpDepth == 2) }
            try capture(studio, name + "-closed", width: 1280)
            let record: [String: Any] = ["fixture": name, "inspector_visible": studio.showInspector, "error": studio.mtpError ?? "", "rows": studio.mtpTuneRows.count, "depth": studio.mtpDepth, "can_tune_again": studio.canTuneMTP]
            records.append(record)
            print(record)
        }
        let oldEntry = try await model("mtp-error")
        defer { oldEntry.ejectModel() }
        oldEntry.showInspector = true // The previous button was inside this inspector.
        oldEntry.tuneMTP()
        try await wait("old entry completed", { !oldEntry.isTuningMTP })
        precondition(oldEntry.showInspector && oldEntry.mtpError != nil)
        records.append(["fixture": "previous-entry-mtp-error", "inspector_visible": oldEntry.showInspector, "error": oldEntry.mtpError ?? ""])
        try JSONSerialization.data(withJSONObject: records, options: [.prettyPrinted, .sortedKeys]).write(to: out.appendingPathComponent("header-counterexamples.json"))
        print("Counterexamples reproduced: error, malformed reply and successful results remain in the closed inspector; previous entry keeps it visible. Finite fixture checks, no GPU.")
    }
}
