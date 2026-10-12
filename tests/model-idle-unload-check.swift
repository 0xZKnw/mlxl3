import AppKit
import Foundation

// check-desktop.sh concatenates this file with StudioModel.swift, as the bridge
// checks do. This exercises the real private deadline/event path without adding
// a testing command or synthetic event entry point to the shipped application.
extension StudioModel {
    fileprivate func checkIdleDeadline() { modelIdleUnloader.checkForExpiry() }
    fileprivate var fixtureEngineRunning: Bool { bridge.isRunning }
    fileprivate var fixtureMTPPending: Bool { mtpConfigurationPending }
    fileprivate var fixtureMTPOperation: String { mtpOperationID.uuidString }
    fileprivate var fixtureMTPTaskExists: Bool { mtpDownloadTask != nil }
    fileprivate func observeFixtureEvents(_ observe: @escaping @MainActor (BridgeEvent) -> Void) {
        bridge.onEvent = { [weak self] event in
            self?.handle(event)
            observe(event)
        }
    }
    fileprivate func deliverFixtureEvent(_ json: String) throws {
        handle(try JSONDecoder().decode(BridgeEvent.self, from: Data(json.utf8)))
    }
}

@MainActor private final class TestIdleClock {
    var instant = ContinuousClock.now
    func advance(_ duration: Duration) { instant = instant.advanced(by: duration) }
}

@main struct ModelIdleUnloadCheck {
    @MainActor static func main() async throws {
        var checks = 0
        func check(_ condition: Bool, _ message: String) {
            checks += 1
            precondition(condition, message)
        }
        func require(_ condition: Bool, _ message: String) throws {
            checks += 1
            if !condition { throw NSError(domain: "ModelIdleUnloadCheck", code: 1,
                                          userInfo: [NSLocalizedDescriptionKey: message]) }
        }
        func wait(_ description: String, attempts: Int = 250, until condition: () -> Bool) async throws {
            for _ in 0..<attempts {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw NSError(domain: "ModelIdleUnloadCheck", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Deadline waiting for " + description])
        }
        let suite = "io.mlxl3.idle-check." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }

        check(ModelIdleUnloadDelay.load(from: preferences) == .fifteenMinutes, "Default must be 15 minutes")
        let durations: [Duration?] = [nil, .seconds(60), .seconds(300), .seconds(600),
                                     .seconds(900), .seconds(1800), .seconds(3600), .seconds(7200)]
        for (delay, expected) in zip(ModelIdleUnloadDelay.allCases, durations) {
            preferences.set(delay.rawValue, forKey: ModelIdleUnloadDelay.preferenceKey)
            check(ModelIdleUnloadDelay.load(from: preferences) == delay, "Preference round-trip failed")
            check(delay.duration == expected, "Incorrect delay duration")
            check(!delay.title.isEmpty, "Empty picker title")
        }
        for invalid in ["unknown", "", "-1", true, 15, 15.5] as [Any] {
            preferences.set(invalid, forKey: ModelIdleUnloadDelay.preferenceKey)
            check(ModelIdleUnloadDelay.load(from: preferences) == .fifteenMinutes, "Invalid preference accepted")
        }
        preferences.removeObject(forKey: ModelIdleUnloadDelay.preferenceKey)

        let clock = TestIdleClock()
        var expired = 0
        let unloader = ModelIdleUnloader(now: { clock.instant }) { expired += 1 }
        unloader.update(isIdle: true, after: .seconds(60))
        clock.advance(.seconds(59)); unloader.checkForExpiry()
        check(expired == 0, "Unloaded before deadline")
        unloader.update(isIdle: true, after: .seconds(60))
        clock.advance(.seconds(1)); unloader.checkForExpiry()
        check(expired == 1, "Duplicate idle status reset the deadline")
        unloader.checkForExpiry()
        check(expired == 1, "Deadline callback fired more than once")

        unloader.update(isIdle: true, after: .seconds(60))
        clock.advance(.seconds(59)); unloader.update(isIdle: false, after: .seconds(60))
        clock.advance(.seconds(3600)); unloader.checkForExpiry()
        check(expired == 1, "Busy model unloaded")
        unloader.update(isIdle: true, after: .seconds(60))
        clock.advance(.seconds(59)); unloader.checkForExpiry()
        check(expired == 1, "Busy time was counted as inactivity")
        clock.advance(.seconds(1)); unloader.checkForExpiry()
        check(expired == 2, "Full idle interval did not resume")

        unloader.update(isIdle: true, after: .seconds(60))
        clock.advance(.seconds(30)); unloader.update(isIdle: true, after: .seconds(300))
        clock.advance(.seconds(269)); unloader.checkForExpiry()
        check(expired == 2, "Changed delay used the old deadline")
        clock.advance(.seconds(31)); unloader.checkForExpiry()
        check(expired == 3, "Changed delay did not take effect")
        for disabled in [nil, .zero, .seconds(-1)] as [Duration?] {
            unloader.update(isIdle: true, after: .seconds(60))
            unloader.update(isIdle: true, after: disabled)
            clock.advance(.seconds(7200)); unloader.checkForExpiry()
            check(expired == 3, "Disabled deadline fired")
        }

        // The actual sleeping Task must also deliver and cancel; manual clock
        // checks above alone would not catch a disconnected scheduling callback.
        var timerCalls = 0
        let realTimer = ModelIdleUnloader { timerCalls += 1 }
        realTimer.update(isIdle: true, after: .milliseconds(40))
        try await wait("real deadline", until: { timerCalls == 1 })
        realTimer.update(isIdle: true, after: .milliseconds(40))
        realTimer.cancel()
        try await Task.sleep(for: .milliseconds(80))
        check(timerCalls == 1, "Cancelled sleeping task fired")
        var disposable: ModelIdleUnloader? = ModelIdleUnloader { timerCalls += 1 }
        weak var weakTimer = disposable
        disposable?.update(isIdle: true, after: .milliseconds(40))
        disposable = nil
        check(weakTimer == nil, "Sleeping task retained its controller")
        weakTimer = nil

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-idle-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        defer { try? FileManager.default.removeItem(at: root) }
        var instances: [StudioModel] = []
        defer { for studio in instances { studio.ejectModel() } }
        func make(_ name: String) async throws -> (StudioModel, TestIdleClock) {
            let clock = TestIdleClock()
            let studio = StudioModel(conversationFileURL: root.appendingPathComponent(UUID().uuidString + ".json"),
                                     preferences: preferences, idleUnloadNow: { clock.instant })
            instances.append(studio)
            studio.models = [LocalModel(name: name, path: root.appendingPathComponent(name).path,
                                       modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1,
                                       modules: 1, addedAt: "", size: "1 B")]
            studio.setModelIdleUnloadDelay(.oneMinute)
            studio.selectModel(name)
            try await wait("model ready", until: { studio.engineState.isReady && !studio.mtpDownloading })
            return (studio, clock)
        }
        func tick(_ studio: StudioModel, _ clock: TestIdleClock, _ seconds: Int) {
            clock.advance(.seconds(seconds))
            studio.checkIdleDeadline()
        }

        let (idle, idleClock) = try await make("idle")
        tick(idle, idleClock, 59)
        check(idle.fixtureEngineRunning && idle.engineState.isReady, "Idle model unloaded too early")
        tick(idle, idleClock, 1)
        check(idle.engineState == .idle && !idle.fixtureEngineRunning, "Deadline did not stop the real bridge")
        check(idle.selectedModelName == "idle" && idle.engineMemoryFootprintBytes == 0, "Selection or engine release lost")
        idle.selectModel("idle")
        try await wait("same-model reload", until: { idle.engineState.isReady })
        tick(idle, idleClock, 59)
        check(idle.engineState.isReady, "Old deadline survived reload")
        tick(idle, idleClock, 1)
        check(idle.engineState == .idle, "Reload did not create a fresh interval")

        let (generation, generationClock) = try await make("context-count")
        tick(generation, generationClock, 59)
        generation.draft = "Keep this question"; generation.send()
        tick(generation, generationClock, 3600)
        check(generation.isGenerating && generation.fixtureEngineRunning, "Generation was interrupted")
        try await wait("generation complete", until: { !generation.isGenerating })
        let history = generation.conversations.first!.messages.map { $0.content }
        tick(generation, generationClock, 59)
        check(generation.engineState.isReady, "Countdown did not restart after generation")
        tick(generation, generationClock, 1)
        check(generation.engineState == .idle, "Completed generation never unloaded")
        check(generation.conversations.first!.messages.map { $0.content } == history, "Auto-unload changed history")
        check(generation.persistNow(), "History could not be saved after unload")
        let restored = StudioModel(conversationFileURL: root.appendingPathComponent("preferences.json"),
                                   isPreview: true, preferences: preferences)
        check(restored.modelIdleUnloadDelay == .oneMinute, "Studio preference did not persist")

        let (stopping, stopClock) = try await make("stop-idle")
        stopping.draft = "Finish stopping first"; stopping.send(); stopping.stopGeneration()
        tick(stopping, stopClock, 3600)
        check(stopping.isGenerating && stopping.fixtureEngineRunning, "Pending cancellation was treated as idle")
        try await wait("generation stopped", until: { !stopping.isGenerating })
        tick(stopping, stopClock, 59)
        check(stopping.engineState.isReady, "Cancellation did not restart a full interval")
        tick(stopping, stopClock, 1)
        check(stopping.engineState == .idle, "Stopped generation never unloaded")

        let (never, neverClock) = try await make("never")
        never.setModelIdleUnloadDelay(.never)
        tick(never, neverClock, 10800)
        check(never.engineState.isReady && never.fixtureEngineRunning, "Never setting unloaded model")
        never.setModelIdleUnloadDelay(.fiveMinutes)
        tick(never, neverClock, 299)
        check(never.engineState.isReady, "New setting unloaded early")
        tick(never, neverClock, 1)
        check(never.engineState == .idle, "New setting did not unload")

        let (mcp, mcpClock) = try await make("mcp-idle")
        tick(mcp, mcpClock, 59); mcp.setMCPEnabled(true)
        tick(mcp, mcpClock, 3600)
        check(mcp.mcpUpdating && mcp.fixtureEngineRunning, "MCP configuration was interrupted")
        // The generic lifecycle fixture ignores set_mcp. Deliver its completion
        // through the real decoder/handler so the busy boundary stays explicit.
        try mcp.deliverFixtureEvent(#"{"type":"mcp_status","mcp_servers":1,"mcp_tools":1}"#)
        tick(mcp, mcpClock, 59)
        check(mcp.engineState.isReady, "MCP completion did not restart interval")
        tick(mcp, mcpClock, 1)
        check(mcp.engineState == .idle, "MCP completion never unloaded")
        preferences.set(false, forKey: "studio.mcpEnabled")

        let (mtp, mtpClock) = try await make("auto-dense")
        mtp.setMTPEnabled(true)
        tick(mtp, mtpClock, 3600)
        check(mtp.mtpDownloading && mtp.fixtureEngineRunning, "MTP preparation was interrupted")
        try await wait("MTP configured", until: { !mtp.mtpDownloading && !mtp.fixtureMTPPending })
        check(mtp.canTuneMTP, "Fixture tuning not available")
        tick(mtp, mtpClock, 59); mtp.tuneMTP()
        tick(mtp, mtpClock, 3600)
        check(mtp.isTuningMTP && mtp.fixtureEngineRunning, "MTP tuning was interrupted")
        try await wait("tuning finished", until: { !mtp.isTuningMTP })
        tick(mtp, mtpClock, 59)
        check(mtp.engineState.isReady, "Tuning completion did not restart interval")
        mtp.tuneMTP(); mtp.cancelMTPTuning()
        tick(mtp, mtpClock, 3600)
        check(mtp.isTuningMTP && mtp.fixtureEngineRunning, "Pending tuning cancellation was treated as idle")
        try await wait("tuning cancelled", until: { !mtp.isTuningMTP })
        tick(mtp, mtpClock, 59)
        check(mtp.engineState.isReady, "Tuning cancellation did not restart interval")
        mtp.setMTPEnabled(false)
        tick(mtp, mtpClock, 3600)
        check(mtp.fixtureMTPPending && mtp.fixtureEngineRunning, "MTP OFF configuration was interrupted")
        try await wait("MTP OFF acknowledged", until: { !mtp.fixtureMTPPending })
        tick(mtp, mtpClock, 59)
        check(mtp.engineState.isReady, "MTP OFF completion did not restart interval")
        tick(mtp, mtpClock, 1)
        check(mtp.engineState == .idle, "MTP OFF completion never unloaded")

        let (switching, switchClock) = try await make("first-idle")
        tick(switching, switchClock, 59)
        switching.models.append(LocalModel(name: "second-idle", path: root.path, modelType: "audit",
                                          format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B"))
        switching.selectModel("second-idle")
        tick(switching, switchClock, 3600)
        check(switching.engineState == .loading("second-idle"), "Load was interrupted by old deadline")
        try await wait("switched model", until: { switching.engineState.isReady })
        tick(switching, switchClock, 59)
        check(switching.loadedModel?.name == "second-idle", "Switched model inherited old deadline")
        switching.ejectModel()
        tick(switching, switchClock, 10800)
        check(switching.engineState == .idle, "Manual unload left a live deadline")

        let (crash, crashClock) = try await make("crash")
        crash.draft = "crash fixture"; crash.send()
        try await wait("crash handled", until: { !crash.isGenerating })
        let crashedState = crash.engineState
        tick(crash, crashClock, 10800)
        check(!crash.fixtureEngineRunning && crash.engineState == crashedState, "Old deadline overwrote crash state")

        for name in ["auto-idle-malformed", "auto-idle-empty-error", "auto-idle-error"] {
            let (recovery, recoveryClock) = try await make(name)
            var responseHandled = false
            recovery.observeFixtureEvents { event in
                if event.type == "error" { responseHandled = true }
            }
            tick(recovery, recoveryClock, 59)
            recovery.setMTPEnabled(false)
            let oldOperation = recovery.fixtureMTPOperation
            try await wait("MTP failure handled", until: { responseHandled })
            try require(!recovery.fixtureMTPPending && !recovery.fixtureMTPTaskExists,
                        "MTP failure left the idle countdown blocked: " + name)
            try require(recovery.mtpError != nil && recovery.engineState.isReady,
                        "MTP error was hidden or stopped a responsive engine")
            tick(recovery, recoveryClock, 59)
            try require(recovery.engineState.isReady, "Error recovery reused the old idle deadline")
            recovery.draft = "Continue after the MTP error"; recovery.send()
            try await wait("recovery generation", until: { !recovery.isGenerating })
            try require(recovery.conversations.first?.messages.last?.content == "hello",
                        "Responsive engine could not generate after MTP error")
            tick(recovery, recoveryClock, 59)
            try require(recovery.engineState.isReady, "Recovery generation reused an old deadline")
            tick(recovery, recoveryClock, 1)
            try require(recovery.engineState == .idle && !recovery.fixtureEngineRunning,
                        "Recovered MTP error still prevented automatic unload")
            recovery.selectModel(name)
            try await wait("recovery reload", until: { recovery.engineState.isReady })
            let errorBeforeStaleReply = recovery.mtpError
            try recovery.deliverFixtureEvent("{\"type\":\"error\",\"request_id\":\"\(oldOperation)\",\"message\":\"obsolete\"}")
            try require(recovery.mtpError == errorBeforeStaleReply && !recovery.fixtureMTPPending,
                        "A stale configuration reply changed the reloaded model")
            recovery.ejectModel()
        }

        let (activeRecovery, _) = try await make("auto-idle-silent")
        activeRecovery.setMTPEnabled(false)
        activeRecovery.draft = "Fail this turn explicitly"; activeRecovery.send()
        try activeRecovery.deliverFixtureEvent(#"{"type":"error","message":"uncorrelated failure"}"#)
        try require(!activeRecovery.fixtureMTPPending && !activeRecovery.isGenerating,
                    "MTP recovery swallowed generation error cleanup")
        try require(activeRecovery.conversations.first?.messages.last?.isStreaming == false,
                    "Generation remained streaming after an uncorrelated error")
        activeRecovery.ejectModel()

        let (tuningRecovery, _) = try await make("auto-dense")
        tuningRecovery.setMTPEnabled(true)
        try await wait("recovery tuning head", until: { !tuningRecovery.mtpDownloading })
        tuningRecovery.setMTPEnabled(false)
        tuningRecovery.tuneMTP()
        try require(tuningRecovery.fixtureMTPPending && tuningRecovery.isTuningMTP,
                    "Fixture did not exercise simultaneous pending MTP and tuning")
        try tuningRecovery.deliverFixtureEvent(#"{"type":"error","request_id":""}"#)
        try require(!tuningRecovery.fixtureMTPPending && !tuningRecovery.isTuningMTP && tuningRecovery.mtpError != nil,
                    "MTP recovery swallowed tuning cleanup or hid a missing-message error")
        tuningRecovery.ejectModel()

        // Start a second OFF operation, then unload/reload. Its cancelled timeout
        // must not affect the reloaded session while the next case waits 30s.
        let (cancelledOFF, cancelledClock) = try await make("auto-idle-silent")
        cancelledOFF.setMTPEnabled(false)
        let cancelledOperation = cancelledOFF.fixtureMTPOperation
        try require(cancelledOFF.fixtureMTPPending && cancelledOFF.fixtureMTPTaskExists,
                    "MTP OFF did not establish a bounded pending operation")
        cancelledOFF.ejectModel()
        cancelledOFF.selectModel("auto-idle-silent")
        try await wait("cancelled OFF reload", until: { cancelledOFF.engineState.isReady })
        try require(!cancelledOFF.fixtureMTPPending && !cancelledOFF.fixtureMTPTaskExists,
                    "Unload failed to cancel the old OFF timeout")

        let (silentOFF, silentClock) = try await make("auto-idle-silent")
        silentOFF.setMTPEnabled(false)
        try silentOFF.deliverFixtureEvent(#"{"type":"mtp_status","request_id":"obsolete","mtp_active":false}"#)
        try require(silentOFF.fixtureMTPPending, "A stale status acknowledged the current OFF operation")
        silentOFF.draft = "Do not interrupt this generation"; silentOFF.send()
        tick(silentOFF, silentClock, 3600)
        try require(silentOFF.isGenerating && silentOFF.fixtureEngineRunning,
                    "A pending OFF request interrupted generation")
        try await wait("silent OFF timeout", attempts: 1800, until: { silentOFF.mtpError != nil })
        try require(!silentOFF.fixtureMTPPending && !silentOFF.fixtureMTPTaskExists,
                    "MTP OFF timeout did not release the pending guard")
        try require(silentOFF.isGenerating && silentOFF.fixtureEngineRunning,
                    "MTP OFF timeout interrupted an active generation")
        tick(silentOFF, silentClock, 3600)
        try require(silentOFF.isGenerating, "Idle expiry interrupted generation after timeout")
        try Data().write(to: root.appendingPathComponent("release-idle-generation"))
        try await wait("generation after OFF timeout", until: { !silentOFF.isGenerating })
        try require(silentOFF.conversations.first?.messages.last?.content == "hello",
                    "Generation was lost after OFF timeout")
        tick(silentOFF, silentClock, 59)
        try require(silentOFF.engineState.isReady, "Timeout used the busy interval as idle time")
        tick(silentOFF, silentClock, 1)
        try require(silentOFF.engineState == .idle && !silentOFF.fixtureEngineRunning,
                    "Silent OFF timeout still prevented unload after generation")
        try cancelledOFF.deliverFixtureEvent("{\"type\":\"error\",\"request_id\":\"\(cancelledOperation)\",\"message\":\"obsolete\"}")
        tick(cancelledOFF, cancelledClock, 59)
        try require(cancelledOFF.mtpError == nil && cancelledOFF.engineState.isReady,
                    "Cancelled OFF timeout or late error affected the reloaded model")
        tick(cancelledOFF, cancelledClock, 1)
        try require(cancelledOFF.engineState == .idle, "Reloaded model did not receive a fresh idle deadline")
        print("Model idle unload checks passed: \(checks) assertions, real deadline, preferences and Studio/bridge lifecycle")
    }
}
