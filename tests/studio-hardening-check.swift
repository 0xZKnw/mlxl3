import AppKit
import Foundation

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
            for _ in 0..<200 where !model.engineState.isReady { try await Task.sleep(for: .milliseconds(20)) }
            precondition(model.engineState.isReady, "Fixture not ready")
            return model
        }
        let tuned = try await make("mtp-good")
        tuned.setMTPEnabled(true)
        for _ in 0..<200 where tuned.mtpDownloading { try await Task.sleep(for: .milliseconds(20)) }
        precondition(tuned.canTuneMTP && tuned.mtpMaxDepth == 3)
        let before = tuned.conversations[0].messages.count
        tuned.tuneMTP()
        precondition(tuned.isTuningMTP && tuned.isGenerating)
        tuned.draft = "should wait"; tuned.send()
        precondition(tuned.conversations[0].messages.count == before && !tuned.canSend)
        for _ in 0..<200 where tuned.isTuningMTP { try await Task.sleep(for: .milliseconds(20)) }
        precondition(tuned.mtpEnabled && tuned.mtpDepth == 2 && tuned.mtpTuneRows.count == 4 && tuned.mtpTuneProgress == 1)
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
        restoredTune.setMTPDepth(3); restoredTune.ejectModel()
        let manual = try await make("mtp-good")
        precondition(manual.mtpDepth == 3, "Manual depth selection must persist")
        manual.ejectModel()
        for name in ["mtp-baseline", "mtp-collapse", "mtp-invalid", "mtp-error", "mtp-malformed"] {
            let candidate = try await make(name)
            candidate.setMTPEnabled(true)
            for _ in 0..<200 where candidate.mtpDownloading { try await Task.sleep(for: .milliseconds(20)) }
            candidate.tuneMTP()
            for _ in 0..<200 where candidate.isTuningMTP { try await Task.sleep(for: .milliseconds(20)) }
            if name == "mtp-baseline" {
                precondition(!candidate.mtpEnabled && candidate.mtpTuneRows.count == 4)
            } else if name == "mtp-collapse" {
                precondition(candidate.mtpDepth == 2 && candidate.mtpEnabled, "Zero acceptance must not win")
            } else {
                precondition(candidate.mtpDepth == 1 && candidate.mtpEnabled && candidate.mtpError != nil)
            }
            candidate.ejectModel()
        }
        let old = try await make("mtp-old")
        precondition(old.mtpAvailable && old.mtpMaxDepth == 1 && !old.canTuneMTP)
        old.setMTPDepth(3); precondition(old.mtpDepth == 1)
        old.ejectModel()
        let baselineAgain = try await make("mtp-baseline")
        precondition(!baselineAgain.mtpEnabled && baselineAgain.mtpTuneRows.count == 4, "Baseline winner must also persist")
        baselineAgain.ejectModel()
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
