import AppKit
import Foundation

// check-desktop.sh concatenates this file with StudioModel.swift: the scenarios
// drive the production automatic Tune (MTP-AUTOTUNE-20261115) through the real
// bridge, CLI head installer and tests/fake-desktop-engine.py, without a test
// entry point in the shipped application. Only the two debounce delays are shortened.
extension StudioModel {
    fileprivate func fixtureAutoTuneDelays(start: Duration, idle: Duration) {
        mtpAutoTuneDelay = start; mtpAutoTuneIdleDelay = idle
    }
    fileprivate var fixtureLedger: MTPAutoTuneLedger { mtpAutoTuneLedger }
    fileprivate var fixtureTuneKey: MTPConfigurationKey? { tuneConfiguration }
    fileprivate var fixtureCurrentKey: MTPConfigurationKey? { currentMTPConfiguration }
    fileprivate var fixtureTuneCancelRequested: Bool { tuneCancellationRequested }
}

@main struct MTPAutoTuneCheck {
    @MainActor static func main() async throws {
        var checks = 0
        func check(_ condition: Bool, _ message: String) {
            checks += 1
            precondition(condition, message)
        }
        func wait(_ description: String, attempts: Int = 500, until condition: () -> Bool) async throws {
            for _ in 0..<attempts {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw NSError(domain: "MTPAutoTuneCheck", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Deadline waiting for " + description])
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-autotune-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let requestLog = root.appendingPathComponent("requests.jsonl")
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        setenv("MLXL3_SAMPLING_CHECK_REQUESTS", requestLog.path, 1)
        unsetenv("MLXL3_FIXTURE_TUNING_SUFFIX")
        let suite = "io.mlxl3.autotune-check." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite) }
        var instances: [StudioModel] = []
        defer { for studio in instances { studio.ejectModel() } }

        func requests(_ type: String, model: String) -> [[String: Any]] {
            guard let text = try? String(contentsOf: requestLog, encoding: .utf8) else { return [] }
            return text.split(separator: "\n").compactMap {
                try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
            }.filter { $0["model"] as? String == model }
             .compactMap { $0["request"] as? [String: Any] }
             .filter { $0["type"] as? String == type }
        }
        func tunes(_ model: String) -> Int { requests("tune_mtp", model: model).count }
        func local(_ name: String) -> LocalModel {
            LocalModel(name: name, path: root.appendingPathComponent(name).path, modelType: "audit",
                       format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
        }
        func make(_ name: String, others: [String] = [], mtp: Bool = true) async throws -> StudioModel {
            if mtp { ModelMTPPreference.save(true, modelPath: root.appendingPathComponent(name).path, preferences: prefs) }
            let studio = StudioModel(conversationFileURL: root.appendingPathComponent(UUID().uuidString + ".json"),
                                     preferences: prefs)
            instances.append(studio)
            studio.fixtureAutoTuneDelays(start: .milliseconds(40), idle: .milliseconds(400))
            studio.models = ([name] + others).map(local)
            studio.selectModel(name)
            try await wait("\(name) ready", until: { studio.engineState.isReady && !studio.mtpDownloading })
            return studio
        }
        func settle(_ studio: StudioModel) async throws {
            try await wait("tune finished", until: { !studio.isTuningMTP })
        }
        // Longer than both debounce delays and a full fixture Tune (~0.5 s).
        func quiet() async throws { try await Task.sleep(for: .milliseconds(1200)) }

        // 1. First load without a stored result tunes once, in the background, and applies the winner.
        let first = try await make("mtp-good")
        try await wait("automatic tune start", until: { first.isTuningMTP })
        check(first.mtpAutoTuneActive && first.isGenerating && !first.composerBusy,
              "An automatic Tune must keep the composer usable")
        first.draft = "typing during tune"
        check(first.canSend && first.sendBlockReason == nil, "Chat must not be blocked by an automatic Tune")
        first.draft = ""
        try await settle(first)
        check(tunes("mtp-good") == 1, "First load must tune exactly once")
        check(first.mtpEnabled && first.mtpDepth == 2 && first.mtpTuneRows.count == 4 && first.mtpTunedAt != nil
              && first.mtpError == nil && !first.mtpAutoTuneActive, "Automatic result must be applied")
        let goodKey = try require(first.fixtureCurrentKey, "missing configuration key")
        check(MTPAutoTuneLedger.isTuned(MTPTuning.load(key: goodKey, preferences: prefs), for: goodKey),
              "Automatic result must persist under the runtime key")
        first.draft = "after tune"; first.send()
        try await wait("generation", until: { !first.isGenerating && first.conversations[0].messages.count == 2 })
        check(first.conversations[0].messages.last!.content.contains("\"mtp_depth\": 2"), "Tuned depth must reach the bridge")
        try await quiet()
        check(tunes("mtp-good") == 1, "A tuned key must not retune after chat")
        first.ejectModel()

        // 2. Same key on a later load: reuse without a new Tune, including a later manual choice.
        let reused = try await make("mtp-good")
        try await quiet()
        check(tunes("mtp-good") == 1 && !reused.isTuningMTP, "Same key must reuse the stored result")
        check(reused.mtpDepth == 2 && reused.mtpTuneRows.count == 4 && reused.mtpAutoTuneDecision == .reuse,
              "Stored depth must be restored")
        reused.setMTPDepth(3); reused.ejectModel()
        let manual = try await make("mtp-good")
        try await quiet()
        check(tunes("mtp-good") == 1 && manual.mtpDepth == 3, "A manual depth after Tune must be respected")
        manual.ejectModel()

        // 3. A runtime option change (new engine key) retunes; restoring the option reuses the old record.
        setenv("MLXL3_FIXTURE_TUNING_SUFFIX", ":mtpevict-v1=0", 1)
        let changed = try await make("mtp-good")
        try await wait("retune after key change", until: { tunes("mtp-good") == 2 })
        try await settle(changed)
        let changedKey = try require(changed.fixtureCurrentKey, "missing changed key")
        check(changedKey != goodKey && changedKey.runtime.hasSuffix(":mtpevict-v1=0"), "Key must follow the engine")
        check(changed.mtpDepth == 2 && changed.mtpTunedAt != nil, "Retune must apply its own winner")
        changed.ejectModel()
        unsetenv("MLXL3_FIXTURE_TUNING_SUFFIX")
        let back = try await make("mtp-good")
        try await quiet()
        check(tunes("mtp-good") == 2 && back.mtpDepth == 3, "Original key must still reuse its record")
        back.ejectModel()

        // 4. Failure keeps the current default depth and is not retried in this app run.
        let failing = try await make("mtp-error")
        try await wait("failing tune", until: { tunes("mtp-error") == 1 })
        try await settle(failing)
        let failingKey = try require(failing.fixtureCurrentKey, "missing failing key")
        check(failing.mtpEnabled && failing.mtpDepth == 1 && failing.mtpTuneRows.isEmpty && failing.mtpTunedAt == nil,
              "Failure must fail closed on the default depth")
        check(failing.mtpError != nil && failing.fixtureLedger.failed.contains(failingKey), "Failure must be recorded")
        check(MTPTuning.load(key: failingKey, preferences: prefs)?.tunedAt == nil, "Failure must not persist a Tune")
        failing.draft = "after failure"; failing.send()
        try await wait("generation after failure", until: { !failing.isGenerating && failing.conversations[0].messages.count == 2 })
        check(failing.conversations[0].messages.last!.content.contains("\"mtp_depth\": 1"), "Fallback depth must reach the bridge")
        try await quiet()
        check(tunes("mtp-error") == 1 && failing.mtpAutoTuneDecision == .skip, "A failed key must not retry this run")
        failing.ejectModel()

        // 5. Sending a message preempts the Tune, the message is answered, then an idle retry completes it.
        let preempted = try await make("mtp-preempt")
        try await wait("preemptible tune", until: { preempted.isTuningMTP && preempted.mtpTuneProgress > 0 })
        let preemptKey = try require(preempted.fixtureTuneKey, "missing preempt key")
        preempted.draft = "hello during tune"
        check(preempted.canSend, "Send must be accepted during an automatic Tune")
        preempted.send()
        try await wait("preempted generation", until: {
            !preempted.isGenerating && preempted.conversations[0].messages.count == 2
                && preempted.conversations[0].messages.last?.content.isEmpty == false
        })
        check(preempted.conversations[0].messages.first?.content == "hello during tune", "Preempting send lost its text")
        check(preempted.conversations[0].messages.last!.content.contains("\"mtp_depth\": 1"),
              "The interrupted Tune must leave the default depth")
        check(preempted.mtpError == nil && preempted.fixtureLedger.interruptions[preemptKey] == 1,
              "Preemption is an interruption, not an error")
        let order = (try String(contentsOf: requestLog, encoding: .utf8)).split(separator: "\n").compactMap {
            try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any]
        }.filter { $0["model"] as? String == "mtp-preempt" }.compactMap { ($0["request"] as? [String: Any])?["type"] as? String }
        check(Array(order.prefix(2)) == ["tune_mtp", "generate"], "Send must follow the cancelled Tune: \(order)")
        try await wait("idle retry", until: { tunes("mtp-preempt") == 2 })
        try await settle(preempted)
        check(preempted.mtpDepth == 2 && preempted.mtpTunedAt != nil && preempted.fixtureLedger.interruptions[preemptKey] == nil,
              "Idle retry must complete and clear the interruption count")
        preempted.ejectModel()

        // 6. Unload interrupts without an error; the next load tries again.
        let unloaded = try await make("mtp-unload")
        // Progress proves the engine received the request (and logged it) before it is killed.
        try await wait("tune before unload", until: { unloaded.isTuningMTP && unloaded.mtpTuneProgress > 0 })
        let unloadKey = try require(unloaded.fixtureTuneKey, "missing unload key")
        // The toolbar unload stays available during an automatic Tune and takes the same path.
        check(unloaded.mtpAutoTuneActive && unloaded.modelUnloadControl == .enabled,
              "Toolbar unload must stay enabled during an automatic Tune")
        unloaded.unloadModelFromToolbar()
        check(!unloaded.isTuningMTP && !unloaded.mtpAutoTuneActive && unloaded.mtpError == nil
              && unloaded.fixtureLedger.interruptions[unloadKey] == 1, "Unload must interrupt cleanly")
        unloaded.selectModel("mtp-unload")
        try await wait("reload", until: { unloaded.engineState.isReady && !unloaded.mtpDownloading })
        try await wait("tune after reload", until: { tunes("mtp-unload") == 2 })
        try await settle(unloaded)
        check(unloaded.mtpTunedAt != nil && unloaded.mtpDepth == 2, "Reload after unload must finish tuning")
        unloaded.ejectModel()

        // 7. Switching to another model interrupts the Tune and loads the new model.
        let switching = try await make("mtp-switch", others: ["plain"])
        try await wait("tune before switch", until: { switching.isTuningMTP && switching.mtpTuneProgress > 0 })
        let switchKey = try require(switching.fixtureTuneKey, "missing switch key")
        switching.selectModel("plain")
        check(!switching.isTuningMTP && switching.fixtureLedger.interruptions[switchKey] == 1,
              "Model switch must interrupt the automatic Tune")
        try await wait("switched model", until: { switching.engineState.isReady && switching.selectedModelName == "plain" })
        check(switching.mtpError == nil, "Model switch must not report a Tune error")
        switching.ejectModel()

        // 8. An explicit stop from the Tune control is honoured and not retried this run.
        let stopped = try await make("mtp-stop")
        try await wait("tune before stop", until: { stopped.isTuningMTP && stopped.mtpTuneProgress > 0 })
        let stopKey = try require(stopped.fixtureTuneKey, "missing stop key")
        stopped.stopGeneration()
        // Cancellation is asynchronous (SIGUSR1), so check the request flag, then give a
        // wrongly sent signal time to land before checking the Tune is still running.
        check(stopped.isTuningMTP && !stopped.fixtureTuneCancelRequested,
              "The composer stop must not cancel a background Tune")
        try await Task.sleep(for: .milliseconds(150))
        check(stopped.isTuningMTP && stopped.mtpError == nil && !stopped.fixtureLedger.failed.contains(stopKey),
              "The composer stop must leave the background Tune running")
        stopped.cancelMTPTuning()
        try await settle(stopped)
        check(stopped.mtpDepth == 1 && stopped.mtpError != nil && stopped.fixtureLedger.failed.contains(stopKey),
              "Explicit stop keeps the setting and is remembered")
        stopped.draft = "after stop"; stopped.send()
        try await wait("generation after stop", until: { !stopped.isGenerating && stopped.conversations[0].messages.count == 2 })
        try await quiet()
        check(tunes("mtp-stop") == 1, "An explicitly stopped key must not retry this run")
        stopped.ejectModel()

        // 9. Without an MTP head enabled, or with the preference off, nothing is tuned.
        let plain = try await make("mtp-plain", mtp: false)
        try await quiet()
        check(!plain.mtpEnabled && tunes("mtp-plain") == 0, "MTP off must not tune")
        plain.ejectModel()
        prefs.set(false, forKey: StudioModel.mtpAutoTunePreferenceKey)
        let disabled = try await make("mtp-disabled")
        try await quiet()
        check(disabled.mtpEnabled && tunes("mtp-disabled") == 0 && disabled.canTuneMTP, "Opt-out must keep manual Tune only")
        disabled.ejectModel()
        prefs.removeObject(forKey: StudioModel.mtpAutoTunePreferenceKey)

        print("MTP auto-tune checks passed: \(checks) checks over first tune, reuse, manual override, key change, failure, chat preemption, unload, model switch, explicit stop and opt-out")
    }

    static func require<T>(_ value: T?, _ message: String) throws -> T {
        guard let value else {
            throw NSError(domain: "MTPAutoTuneCheck", code: 2, userInfo: [NSLocalizedDescriptionKey: message])
        }
        return value
    }
}
