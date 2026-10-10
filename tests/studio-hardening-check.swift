import AppKit
import Combine
import Foundation
import SwiftUI

// Combined with the production source by check-desktop.sh; no test setter ships.
extension UpdateManager {
    func setStatesForCheck(app: AppUpdateState, engine: AppUpdateState) {
        state = app
        engineState = engine
    }
}

@main struct HardeningCheck {
    @MainActor static func main() async throws {
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-check-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        setenv("MLXL3_HOME", root.path, 1)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "io.mlxl3.check." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        func make(_ name: String) async throws -> StudioModel {
            let model = StudioModel(conversationFileURL: root.appendingPathComponent(name + ".json"), preferences: prefs)
            model.models = [LocalModel(name: name, path: root.path, modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
            model.selectModel(name)
            for _ in 0..<200 where !model.engineState.isReady || model.mtpDownloading { try await Task.sleep(for: .milliseconds(20)) }
            precondition(model.engineState.isReady, "Fixture not ready")
            return model
        }
        // shortcut: the fixture error is the only red content; use accessibility text if another red control is added.
        func tuningErrorIsVisible(_ model: StudioModel) async throws -> Bool {
            _ = NSApplication.shared
            let host = NSHostingView(rootView: GenerationInspector().environmentObject(model).preferredColorScheme(.dark))
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 286, height: 600),
                styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: 286, height: 600)
            defer { window.close() }
            for _ in 0..<100 {
                try await Task.sleep(for: .milliseconds(20))
                host.layoutSubtreeIfNeeded()
                guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                    preconditionFailure("Inspector bitmap unavailable")
                }
                host.cacheDisplay(in: host.bounds, to: bitmap)
                var redPixels = 0
                for y in stride(from: 0, to: bitmap.pixelsHigh, by: 4) {
                    for x in stride(from: 0, to: bitmap.pixelsWide, by: 4) {
                        if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
                           color.redComponent > 0.6 && color.greenComponent < 0.45 && color.blueComponent < 0.45 {
                            redPixels += 1
                            if redPixels >= 5 { return true }
                        }
                    }
                }
            }
            return false
        }
        let updates = try await make("updates")
        defer { updates.ejectModel() }
        updates.draft = "send during background update"
        updates.updateManager.setStatesForCheck(app: .checking, engine: .checking)
        if !updates.canSend {
            FileHandle.standardError.write(Data("Composer regression: ready=\(updates.engineState.isReady), updateBusy=\(updates.updateManager.isBusy), canSend=false\n".utf8))
        }
        precondition(updates.canSend, "A background update check must not disable Send")
        let release = AppUpdateRelease(version: "9.0.0", tag: "v9.0.0", title: "fixture", notes: "",
            pageURL: root, asset: AppUpdateAsset(name: "fixture", downloadURL: root, size: 1, digest: nil))
        let states: [AppUpdateState] = [.idle, .checking, .upToDate(checkedAt: Date()),
            .downloading(release), .ready(release: release, diskImage: root), .installing(release), .failed("fixture")]
        var notifications = 0
        let observation = updates.objectWillChange.sink { notifications += 1 }
        for (appIndex, app) in states.enumerated() {
            for (engineIndex, engine) in states.enumerated() {
                let before = notifications
                updates.updateManager.setStatesForCheck(app: app, engine: engine)
                precondition(updates.canSend == (appIndex != 5 && engineIndex != 5), "Only installation blocks ready inference")
                precondition(notifications > before, "Update transitions must refresh the composer")
            }
        }
        observation.cancel()
        updates.updateManager.setStatesForCheck(app: .downloading(release), engine: .downloading(release))
        updates.draft = " \n "; precondition(!updates.canSend)
        updates.draft = "send during background update"; updates.send()
        precondition(updates.isGenerating && !updates.canSend && updates.draft.isEmpty)
        for _ in 0..<200 where updates.isGenerating { try await Task.sleep(for: .milliseconds(20)) }
        precondition(updates.engineState.isReady && updates.conversations[0].messages.last?.role == .assistant
            && updates.conversations[0].messages.last?.content.isEmpty == false, "Send must reach the bridge and complete")
        updates.draft = "next message"; precondition(updates.canSend)
        updates.ejectModel(); precondition(!updates.canSend)
        precondition(!updates.memorySaverEnabled, "Memory saving must be opt-in")
        updates.setMemorySaverEnabled(true)
        precondition(prefs.bool(forKey: "studio.memorySaverEnabled"), "Memory preference not persisted")
        let saver = try await make("mtp-memory-saver")
        defer { saver.ejectModel() }
        precondition(saver.memorySaverEnabled && saver.memorySaverSupported, "Memory preference/capability not restored")
        let advised = saver.recommendedContextLength
        precondition(advised == 2048, "Ready profile must reach the production context adviser")
        precondition(saver.contextLengthDraft == 0, "Recommendation must preserve the user's setting")
        saver.contextLengthDraft = advised!
        precondition(saver.canSaveContext)
        saver.saveContextAndReload()
        for _ in 0..<200 where !saver.engineState.isReady { try await Task.sleep(for: .milliseconds(20)) }
        precondition(saver.engineState.isReady && !saver.canSaveContext)
        precondition((prefs.dictionary(forKey: "studio.contextLengths")?["mtp-memory-saver"] as? Int) == advised)
        saver.setMTPEnabled(true)
        for _ in 0..<200 where saver.mtpDownloading { try await Task.sleep(for: .milliseconds(20)) }
        precondition(saver.canTuneMTP)
        saver.draft = "memory fixture"; saver.send()
        saver.setMemorySaverEnabled(false)
        precondition(saver.memorySaverEnabled, "Memory policy changed during generation")
        for _ in 0..<200 where saver.isGenerating { try await Task.sleep(for: .milliseconds(20)) }
        precondition(!saver.isGenerating && saver.conversations[0].messages.count == 2)
        saver.tuneMTP()
        saver.setMemorySaverEnabled(false)
        precondition(saver.memorySaverEnabled, "Memory policy changed during tuning")
        for _ in 0..<200 where saver.isTuningMTP { try await Task.sleep(for: .milliseconds(20)) }
        precondition(!saver.isTuningMTP && saver.mtpTuneRows.count == 4)
        saver.setMemorySaverEnabled(false)
        saver.draft = "normal fixture"; saver.send()
        for _ in 0..<200 where saver.isGenerating { try await Task.sleep(for: .milliseconds(20)) }
        precondition(!saver.isGenerating)
        let wire = try String(contentsOf: root.appendingPathComponent("memory-saver-requests.jsonl"), encoding: .utf8)
        let requests = try wire.split(separator: "\n").map {
            try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any]
        }
        let generations = requests.filter { $0["type"] as? String == "generate" }
        let tunes = requests.filter { $0["type"] as? String == "tune_mtp" }
        precondition(generations.count == 2 && tunes.count == 1, "Missing production requests")
        precondition(generations[0]["memory_saver"] as? Bool == true && generations[1]["memory_saver"] as? Bool == false)
        precondition(tunes[0]["memory_saver"] as? Bool == true, "Tune did not receive memory policy")
        precondition(generations[0]["mtp"] as? Bool == true && saver.mtpEnabled, "Memory saving disabled MTP")
        precondition(saver.conversations[0].messages.count == 4, "Memory saving lost history")
        saver.ejectModel()
        precondition(!saver.memorySaverSupported, "Old engine capability survived unload")
        saver.setMemorySaverEnabled(true)
        let legacySaver = try await make("mtp-old")
        precondition(legacySaver.memorySaverEnabled && !legacySaver.memorySaverSupported, "Legacy capability assumed")
        precondition(legacySaver.recommendedContextLength == nil, "Legacy engine must not fabricate a recommendation")
        legacySaver.ejectModel(); legacySaver.setMemorySaverEnabled(false)
        print("Memory saving checks passed: opt-in/persistence, generation/Tune wire, busy guards, MTP/history, legacy capability")
        print("Composer update checks passed: 49 app/engine states, UI notifications, empty draft, send/completion, ejection")
        let tuned = try await make("mtp-good")
        tuned.setMTPEnabled(true)
        for _ in 0..<200 where tuned.mtpDownloading { try await Task.sleep(for: .milliseconds(20)) }
        precondition(tuned.canTuneMTP && tuned.mtpMaxDepth == 3)
        tuned.updateManager.setStatesForCheck(app: .checking, engine: .downloading(release))
        precondition(tuned.canTuneMTP, "Background updates must not disable MTP tuning")
        tuned.updateManager.setStatesForCheck(app: .idle, engine: .installing(release))
        precondition(!tuned.canTuneMTP)
        tuned.tuneMTP()
        precondition(!tuned.showInspector && !tuned.isTuningMTP, "An ineligible tune must leave the inspector closed")
        tuned.updateManager.setStatesForCheck(app: .idle, engine: .idle)
        let before = tuned.conversations[0].messages.count
        precondition(!tuned.showInspector)
        tuned.tuneMTP()
        precondition(tuned.showInspector, "Header tuning must reveal progress, results and errors")
        precondition(tuned.isTuningMTP && tuned.isGenerating)
        tuned.draft = "should wait"; tuned.send()
        precondition(tuned.conversations[0].messages.count == before && !tuned.canSend)
        for _ in 0..<200 where tuned.isTuningMTP { try await Task.sleep(for: .milliseconds(20)) }
        precondition(tuned.mtpEnabled && tuned.mtpDepth == 2 && tuned.mtpTuneRows.count == 4 && tuned.mtpTuneProgress == 1)
        precondition(tuned.showInspector, "Completed tuning results must stay visible")
        precondition(tuned.conversations[0].messages.count == before, "Tuning must not enter chat history")
        tuned.send()
        for _ in 0..<200 where tuned.isGenerating { try await Task.sleep(for: .milliseconds(20)) }
        precondition(tuned.conversations[0].messages.last!.content.contains("\"mtp_depth\": 2"), "Selected depth must reach the bridge")
        tuned.ejectModel()
        let restoredTune = try await make("mtp-good")
        precondition(restoredTune.mtpEnabled && restoredTune.mtpDepth == 2 && restoredTune.mtpTuneRows.count == 4)
        restoredTune.tuneMTP()
        for _ in 0..<200 where restoredTune.mtpTuneProgress == 0 { try await Task.sleep(for: .milliseconds(10)) }
        restoredTune.cancelMTPTuning()
        for _ in 0..<200 where restoredTune.isTuningMTP { try await Task.sleep(for: .milliseconds(20)) }
        precondition(restoredTune.mtpDepth == 2 && restoredTune.mtpTuneRows.count == 4 && restoredTune.mtpError != nil)
        precondition(restoredTune.showInspector, "Tuning cancellation must stay visible")
        restoredTune.setMTPDepth(3); restoredTune.ejectModel()
        let manual = try await make("mtp-good")
        precondition(manual.mtpDepth == 3, "Manual depth selection must persist")
        manual.ejectModel()
        for name in ["mtp-baseline", "mtp-collapse", "mtp-invalid", "mtp-error", "mtp-malformed"] {
            let candidate = try await make(name)
            candidate.setMTPEnabled(true)
            for _ in 0..<200 where candidate.mtpDownloading { try await Task.sleep(for: .milliseconds(20)) }
            precondition(!candidate.showInspector)
            candidate.tuneMTP()
            for _ in 0..<200 where candidate.isTuningMTP { try await Task.sleep(for: .milliseconds(20)) }
            if name == "mtp-baseline" {
                precondition(!candidate.mtpEnabled && candidate.mtpTuneRows.count == 4)
            } else if name == "mtp-collapse" {
                precondition(candidate.mtpDepth == 2 && candidate.mtpEnabled, "Zero acceptance must not win")
            } else {
                precondition(candidate.mtpDepth == 1 && candidate.mtpEnabled && candidate.mtpError != nil)
            }
            precondition(candidate.showInspector, "Tuning feedback must stay visible, including malformed replies and errors")
            if name == "mtp-error" || name == "mtp-malformed" {
                let visible = try await tuningErrorIsVisible(candidate)
                precondition(visible, "Tuning error must be rendered inside the 600-point inspector viewport")
            }
            candidate.ejectModel()
        }
        print("MTP inspector checks passed: eligible start, success, cancellation, errors, malformed replies and ineligible start")
        let old = try await make("mtp-old")
        precondition(old.mtpAvailable && old.mtpMaxDepth == 1 && !old.canTuneMTP)
        old.tuneMTP()
        precondition(!old.showInspector && !old.isTuningMTP, "An unsupported engine must leave the inspector closed")
        old.setMTPDepth(3); precondition(old.mtpDepth == 1)
        old.ejectModel()
        prefs.set(true, forKey: "studio.mtpEnabled")
        let baselineAgain = try await make("mtp-baseline")
        precondition(baselineAgain.mtpEnabled && baselineAgain.mtpTuneRows.count == 4
            && MTPTuning.winner(baselineAgain.mtpTuneRows) == 0,
            "Saved baseline results must persist without overriding an explicit global ON")
        let measuredBaselineRows = baselineAgain.mtpTuneRows
        baselineAgain.ejectModel()
        // The real StudioModel/CLI pipe/bridge must select per-target heads,
        // preserve the user's ON preference, and suppress cancelled callbacks.
        let automatic = StudioModel(conversationFileURL: root.appendingPathComponent("automatic.json"), preferences: prefs)
        automatic.models = ["auto-dense", "auto-moe", "auto-slow", "auto-fail", "plain"].map {
            LocalModel(name: $0, path: root.appendingPathComponent($0).path, modelType: "qwen3_5",
                format: "EXL3", bits: 2, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
        }
        func waitAutomatic() async throws {
            for _ in 0..<300 where !automatic.engineState.isReady || automatic.mtpDownloading {
                try await Task.sleep(for: .milliseconds(20))
            }
            precondition(automatic.engineState.isReady && !automatic.mtpDownloading, "Automatic head load timed out")
        }
        prefs.set(true, forKey: "studio.mtpEnabled")
        // Migrate an old, foreign head: inspection must reject it and install
        // the managed head selected from the new target, never forward it.
        prefs.set("/tmp/mlxl3-fixture-mtp-moe", forKey: "studio.mtpHead.\(root.appendingPathComponent("auto-dense").path)")
        automatic.selectModel("auto-dense"); try await waitAutomatic()
        precondition(automatic.mtpHeadPath == "/tmp/mlxl3-fixture-mtp-dense" && automatic.mtpEnabled && automatic.mtpActive == true,
            "Dense head: \(automatic.mtpHeadPath), enabled=\(automatic.mtpEnabled), active=\(String(describing: automatic.mtpActive)), error=\(String(describing: automatic.mtpError))")
        automatic.selectModel("auto-moe"); try await waitAutomatic()
        precondition(automatic.mtpHeadPath == "/tmp/mlxl3-fixture-mtp-moe" && automatic.mtpEnabled && automatic.mtpActive == true)
        MTPTuning.save(MTPSelection(key: MTPConfigurationKey(
            modelPath: root.appendingPathComponent("auto-dense").path,
            headPath: "/tmp/mlxl3-fixture-mtp-dense", runtime: "fixture-runtime:auto-dense"),
            depth: 0, rows: measuredBaselineRows, tunedAt: Date()), preferences: prefs)
        automatic.selectModel("auto-dense"); try await waitAutomatic()
        precondition(automatic.mtpHeadPath == "/tmp/mlxl3-fixture-mtp-dense" && automatic.mtpActive == true,
            "An explicit global ON must override an old measured baseline")
        automatic.setMTPEnabled(false)
        for _ in 0..<200 where automatic.mtpActive != false { try await Task.sleep(for: .milliseconds(10)) }
        precondition(automatic.mtpActive == false && !prefs.bool(forKey: "studio.mtpEnabled"))
        automatic.selectModel("auto-moe"); try await waitAutomatic()
        precondition(!automatic.mtpEnabled && automatic.mtpActive == nil, "MTP OFF must stay off on model switch")
        automatic.setMTPEnabled(true); try await waitAutomatic()
        automatic.selectModel("auto-slow")
        for _ in 0..<200 where automatic.mtpDownloadCompleted == 0 { try await Task.sleep(for: .milliseconds(10)) }
        precondition(automatic.mtpDownloading)
        automatic.selectModel("auto-moe"); try await waitAutomatic()
        try await Task.sleep(for: .milliseconds(1100))
        precondition(automatic.selectedModelName == "auto-moe" && automatic.mtpHeadPath.hasSuffix("-moe")
            && automatic.mtpActive == true && !automatic.mtpDownloading && automatic.mtpError == nil)
        automatic.selectModel("auto-fail"); try await waitAutomatic()
        precondition(automatic.mtpError?.contains("fixture head download failed") == true && !automatic.mtpEnabled)
        for _ in 0..<100 {
            let log = try String(contentsOf: root.appendingPathComponent("mtp-operations.jsonl"), encoding: .utf8)
            if log.contains("\"model\": \"auto-fail\"") { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        automatic.selectModel("plain"); try await waitAutomatic()
        precondition(!automatic.mtpAvailable && automatic.mtpHeadPath.isEmpty && !automatic.mtpDownloading)
        automatic.selectModel("auto-dense"); try await waitAutomatic()
        precondition(automatic.mtpActive == true && automatic.mtpError == nil, "Failure must not disable the saved ON preference")
        automatic.ejectModel()
        let operations = try String(contentsOf: root.appendingPathComponent("mtp-operations.jsonl"), encoding: .utf8)
            .split(separator: "\n").map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
        let configured = operations.compactMap { $0["request"] as? [String: Any] }
        precondition(configured.contains { $0["enabled"] as? Bool == false }, "MTP OFF did not reach the engine")
        precondition(operations.contains {
            $0["model"] as? String == "auto-fail"
                && ($0["request"] as? [String: Any])?["enabled"] as? Bool == false
        }, "Failed preparation did not unload the engine head")
        for _ in 0..<200 where operations.contains(where: {
            ($0["pid"] as? Int32).map { kill($0, 0) == 0 } ?? false
        }) { try await Task.sleep(for: .milliseconds(20)) }
        for operation in operations {
            if let pid = operation["pid"] as? Int32 {
                precondition(kill(pid, 0) == -1 && errno == ESRCH, "Previous engine/download process still lives")
            }
        }
        print("Automatic MTP checks passed: dense→MoE→dense, foreign head, OFF, cancellation, download failure, unsupported target, process cleanup")
        let key = MTPConfigurationKey(modelPath: root.path, headPath: "/tmp/mlxl3-fixture-mtp", runtime: "fixture-runtime:mtp-good")
        precondition(MTPTuning.load(key: key, preferences: prefs)?.depth == 3)
        precondition(MTPTuning.load(key: MTPConfigurationKey(modelPath: root.path, headPath: key.headPath, runtime: "new-engine"), preferences: prefs) == nil)
        let tuningHead = root.appendingPathComponent("head")
        try FileManager.default.createDirectory(at: tuningHead, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: tuningHead.appendingPathComponent("config.json"))
        let keyBefore = MTPConfigurationKey(modelPath: root.path, headPath: tuningHead.path, runtime: "engine-context-4096")
        MTPTuning.save(MTPSelection(key: keyBefore, depth: 2, rows: [], tunedAt: nil), preferences: prefs)
        try Data("{\"changed\":true}".utf8).write(to: tuningHead.appendingPathComponent("config.json"))
        let keyAfter = MTPConfigurationKey(modelPath: root.path, headPath: tuningHead.path, runtime: "engine-context-4096")
        precondition(keyAfter != keyBefore && MTPTuning.load(key: keyAfter, preferences: prefs) == nil)
        let interrupted = try await make("mtp-good")
        interrupted.tuneMTP(); interrupted.ejectModel()
        precondition(!interrupted.isTuningMTP && interrupted.mtpTuneRows.isEmpty)
        let counted = try await make("context-count")
        counted.draft = "salut"; counted.send()
        for _ in 0..<200 where counted.isGenerating { try await Task.sleep(for: .milliseconds(20)) }
        precondition(counted.contextUsed == 42, "Context only shows input tokens after completion")
        precondition(counted.currentConversation?.contextUsage?.used == 42, "Final context was not saved to the conversation")
        let mcp = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("mcp.json"))) as! [String: Any]
        precondition((mcp["mcpServers"] as! [String: Any])["exa"] != nil, "Startup did not register Exa")
        counted.conversations[0].contextUsage = ContextUsage(used: 12, limit: 2048, model: "context-count")
        precondition(counted.contextUsed == 42, "Old history still displays only input tokens")
        counted.settingsDidChange()
        for _ in 0..<600 where !FileManager.default.fileExists(atPath: root.appendingPathComponent("context-count.json").path) {
            try await Task.sleep(for: .milliseconds(20))
        }
        let reopened = StudioModel(conversationFileURL: root.appendingPathComponent("context-count.json"), isPreview: true, preferences: prefs)
        precondition(reopened.contextUsed == 42, "Context count did not survive relaunch")
        reopened.selectedModelName = "other"
        precondition(reopened.contextUsed == nil, "Token count leaked to another model")
        var stats = counted.currentConversation!.messages.last!.stats!
        let baseline = ContextUsage(used: 12, limit: 2048, model: "test")
        stats.contextUsed = 77; stats.toolRounds = 3
        precondition(baseline.finalized(with: stats)?.used == 77, "MCP aggregated counts replaced final context")
        stats.contextUsed = nil
        precondition(baseline.finalized(with: stats) == nil, "Unknown multi-round context was guessed")
        stats.toolRounds = 0
        precondition(baseline.finalized(with: stats)?.used == 42)
        for value in [-1, 0, 1, 12, 42, 2048, Int.max] {
            stats.contextUsed = value
            let actual = baseline.finalized(with: stats)
            precondition(value < 0 ? actual == nil : actual?.used == min(value, 2048))
        }
        stats.contextUsed = 42; stats.contextLimit = 0
        precondition(baseline.finalized(with: stats) == nil)
        for (input, output) in [(Int.max, 1), (-1, 30), (12, -1)] {
            let payload = "{\"ttft_seconds\":0,\"prefill_tps\":0,\"decode_tps\":0,\"prompt_tokens\":\(input),\"generated_tokens\":\(output)}"
            let invalid = try JSONDecoder().decode(GenerationStats.self, from: Data(payload.utf8))
            precondition(baseline.finalized(with: invalid) == nil, "Invalid legacy token counts were accepted")
        }
        // Hub metadata crosses the real CLI pipe before ModelLibrary decodes it.
        // The previous Bool-only decoder rejected null/auto/manual.
        for (fragment, expected) in [
            ("", false), (",\"gated\":null", false),
            (",\"gated\":false", false), (",\"gated\":true", true),
            (",\"gated\":\"auto\"", true), (",\"gated\":\"manual\"", true),
            (",\"gated\":\"false\"", false), (",\"gated\":\"true\"", true),
        ] {
            let data = Data(("{\"id\":\"fixture/model-exl3\",\"downloads\":73,\"likes\":0" + fragment + "}").utf8)
            let value = try JSONDecoder().decode(HubModel.self, from: data)
            precondition(value.id == "fixture/model-exl3" && value.downloads == 73 && value.likes == 0)
            precondition(value.gated == expected, fragment)
        }
        for invalid in ["0", "1", "[]", "{}", "\"unknown\""] {
            do {
                _ = try JSONDecoder().decode(HubModel.self, from: Data(
                    ("{\"id\":\"fixture/model\",\"downloads\":1,\"likes\":0,\"gated\":" + invalid + "}").utf8))
                preconditionFailure("Invalid access metadata accepted: " + invalid)
            } catch is DecodingError {}
        }
        let library = ModelLibrary()
        func search(_ query: String) async throws {
            library.query = query
            library.search()
            for _ in 0..<200 where library.searching { try await Task.sleep(for: .milliseconds(20)) }
            precondition(!library.searching, "Catalogue search timed out")
        }
        try await search("qwen3.6")
        precondition(library.error == nil && library.results.count == 3)
        precondition(library.results[0].id == "fixture/qwen3.6-exl3" && !library.results[0].gated)
        precondition(library.results[1].gated && !library.results[2].gated)
        library.open(library.results[0].id)
        for _ in 0..<200 where library.loadingDetail { try await Task.sleep(for: .milliseconds(20)) }
        precondition(library.detail?.id == "fixture/qwen3.6-exl3")
        precondition(library.selectedVariant?.sizeBytes == 128)
        library.back()
        try await search("error")
        precondition(library.error?.contains("fixture catalogue unavailable") == true)
        try await search("invalid")
        precondition(library.error != nil)
        try await search("empty")
        precondition(library.error == nil && library.results.isEmpty)
        library.query = "slow"; library.search()
        try await Task.sleep(for: .milliseconds(400))
        try await search("latest")
        try await Task.sleep(for: .milliseconds(750))
        precondition(library.results[0].id == "fixture/latest-exl3" && library.error == nil,
                     "Cancelled search replaced the current results")
        // Previously cancelled/failed commands must not poison cached results.
        try await search("qwen3.6")
        precondition(library.results[0].id == "fixture/qwen3.6-exl3" && library.error == nil)
        library.moreResults()
        for _ in 0..<200 where library.searching { try await Task.sleep(for: .milliseconds(20)) }
        precondition(library.error == nil && library.results.count == 3)
        library.refresh()
        for _ in 0..<200 where library.searching { try await Task.sleep(for: .milliseconds(20)) }
        precondition(library.error == nil && library.results.count == 3)
        let downloadStudio = StudioModel(conversationFileURL: root.appendingPathComponent("downloads.json"),
                                         isPreview: true, preferences: prefs)
        library.open("fixture/download-exl3")
        for _ in 0..<200 where library.loadingDetail { try await Task.sleep(for: .milliseconds(20)) }
        library.download(studio: downloadStudio)
        for _ in 0..<200 where library.downloadProgress.bytesPerSecond == nil {
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(library.downloadStatus == .transferring && library.downloading != nil)
        precondition((library.downloadProgress.bytesPerSecond ?? 0) > 0)
        precondition((library.downloadProgress.fraction ?? 0) > 0)
        library.cancelDownload()
        for _ in 0..<200 where library.downloading != nil || library.pending.isEmpty {
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(library.downloadStatus == .paused && library.downloadProgress.completed > 0)
        precondition(library.lastDownloadRepo == "fixture/download-exl3")
        let paused = library.pending.first!
        library.resume(paused, studio: downloadStudio)
        precondition(library.downloadProgress.bytesPerSecond == nil && library.downloadStatus == .transferring,
                     "Resume retained the previous transfer's speed")
        for _ in 0..<200 where library.downloading != nil { try await Task.sleep(for: .milliseconds(20)) }
        precondition(library.downloadStatus == .complete && library.downloadProgress.fraction == 1)
        precondition(library.downloadMessage?.contains("fixture-model") == true)
        library.open("fixture/fail-exl3")
        for _ in 0..<200 where library.loadingDetail { try await Task.sleep(for: .milliseconds(20)) }
        library.download(studio: downloadStudio)
        for _ in 0..<200 where library.downloading != nil { try await Task.sleep(for: .milliseconds(20)) }
        precondition(library.downloadStatus == .failed && library.downloadMessage?.contains("fixture download failed") == true)
        let crash = try await make("crash")
        crash.draft = "test"; crash.send()
        try await Task.sleep(for: .milliseconds(500))
        precondition(!crash.engineState.isReady, "Dead engine still ready")

        let deleted = try await make("delete")
        deleted.draft = "test"; deleted.send()
        deleted.deleteConversation(deleted.selectedConversationID!)
        try await Task.sleep(for: .milliseconds(600))
        precondition(!deleted.isGenerating, "Deleted response wedged generation")
        deleted.ejectModel()
        deleted.newConversation(); deleted.draft = "draft one"
        let first = deleted.selectedConversationID!
        deleted.newConversation(); deleted.draft = "draft two"
        deleted.selectConversation(first)
        precondition(deleted.draft == "draft one")

        let broken = root.appendingPathComponent("broken.json")
        try "{broken".write(to: broken, atomically: true, encoding: .utf8)
        let recovered = StudioModel(conversationFileURL: broken, isPreview: true, preferences: prefs)
        precondition(recovered.storageError != nil && !recovered.persistNow())
        let original = try String(contentsOf: broken, encoding: .utf8)
        precondition(original == "{broken")
        recovered.recoverConversations()
        precondition(recovered.persistNow())
        let store = ConversationStore(fileURL: root.appendingPathComponent("ordered.json"))
        let a = WorkspaceSnapshot(selectedConversationID: nil, selectedModelName: "old", conversations: [], temperature: 0, topK: 0, repetitionPenalty: 1, systemPrompt: "")
        let b = WorkspaceSnapshot(selectedConversationID: nil, selectedModelName: "new", conversations: [], temperature: 0, topK: 0, repetitionPenalty: 1, systemPrompt: "")
        try store.save(b, revision: 2); try store.save(a, revision: 1)
        let latest = try ConversationStore.load(from: store.fileURL)
        precondition(latest?.selectedModelName == "new")
        let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
        message.startTool(id: "tool", serverName: "local", toolName: "test")
        message.fail("interrupted")
        precondition(ChatMessage(snapshot: message.snapshot)!.toolActivities[0].state == .failed)
        precondition(SemanticVersion("1.0") == SemanticVersion("1.0.0"))
        let dflash = StudioModel(conversationFileURL: root.appendingPathComponent("dflash.json"), preferences: prefs)
        dflash.models = [LocalModel(name: "qwen3.6-35b-a3b", path: root.path + "/Qwen3.6-35B-A3B",
                                    modelType: "qwen3_5_moe", format: "EXL3", bits: 2.49,
                                    sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
        dflash.selectedModelName = "qwen3.6-35b-a3b"
        dflash.setDFlash2Enabled(true)
        for _ in 0..<200 where dflash.dflashDownloading { try await Task.sleep(for: .milliseconds(20)) }
        precondition(dflash.dflash2Enabled && dflash.dflashDraftPath == "/tmp/mlxl3-fixture-dflash")
        precondition(dflash.temperature == 0 && dflash.topK == 1 && dflash.repetitionPenalty == 1)
        dflash.setDFlash2Enabled(false)
        precondition(!dflash.dflash2Enabled)
        let renamed = try await make("renamed")
        precondition(renamed.mtpAvailable, "MTP capability must come from the engine")
        renamed.setMTPEnabled(true)
        for _ in 0..<200 where renamed.mtpDownloading { try await Task.sleep(for: .milliseconds(20)) }
        precondition(renamed.mtpEnabled && renamed.mtpHeadPath == "/tmp/mlxl3-fixture-mtp")
        precondition(renamed.temperature == 0 && renamed.topK == 1 && renamed.repetitionPenalty == 1)
        precondition(renamed.dflash2Available, "Engine capability must override the directory name")
        precondition(renamed.runtimeIdentity == "fixture · release · MLX 0.32.2")
        renamed.ejectModel()
        precondition(!renamed.mtpAvailable, "Ejection must discard the MTP capability")
        precondition(!renamed.dflash2Available && renamed.runtimeIdentity == nil)
        let complete = try JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"complete","stats":{"ttft_seconds":0.4,"prefill_tps":300,"decode_tps":16,"prompt_tokens":300,"generated_tokens":42,"model_size_gb":12.5,"elapsed_seconds":8,"end_to_end_ttft_seconds":0.5,"tool_rounds":1,"dflash_proposed_tokens":50,"dflash_accepted_tokens":28,"memory":{"mlx_active_bytes":12000000000,"mlx_cache_bytes":500000000,"mlx_peak_bytes":13000000000,"process_footprint_bytes":14000000000,"process_lifetime_peak_bytes":15000000000}}}"#.utf8))
        precondition(complete.stats?.peakMemoryGB == nil && complete.stats?.modelSizeGB == 12.5)
        precondition(complete.stats?.memory?.processFootprintBytes == 14_000_000_000)
        precondition(complete.stats?.endToEndTTFTSeconds == 0.5 && complete.stats?.toolRounds == 1)
        precondition(complete.stats?.dflashAcceptedTokens == 28)
        let legacy = try JSONDecoder().decode(GenerationStats.self, from: Data(#"{"ttft_seconds":1,"prefill_tps":1,"decode_tps":1,"prompt_tokens":1,"generated_tokens":1,"peak_memory_gb":12}"#.utf8))
        precondition(legacy.peakMemoryGB == 12 && legacy.memory == nil)
        MarkdownRegressionCheck.run()
        print("Desktop hardening checks passed: Hub metadata/search/detail/cancellation/errors, transfer rate/pause/resume/success/failure, crash, deletion, drafts, recovery, save ordering, tool state")
    }
}
