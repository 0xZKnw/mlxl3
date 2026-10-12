import Foundation

extension StudioModel {
    fileprivate func reviewTick() { modelIdleUnloader.checkForExpiry() }
    fileprivate var reviewPending: Bool { mtpConfigurationPending }
    fileprivate var reviewRunning: Bool { bridge.isRunning }
    fileprivate func reviewDuplicateReady() { restoreReadyState() }
    fileprivate func reviewObserve(_ observe: @escaping @MainActor (BridgeEvent) -> Void) {
        bridge.onEvent = { [weak self] event in
            self?.handle(event)
            observe(event)
        }
    }
}

@MainActor private final class ReviewClock {
    var instant = ContinuousClock.now
    func advance(_ seconds: Int) { instant = instant.advanced(by: .seconds(seconds)) }
}

private enum ReviewError: Error { case deadline(String) }

@main struct ReviewIdleCheck {
    @MainActor static func main() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-pr28-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "io.mlxl3.pr28-review." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        let modes = CommandLine.arguments.dropFirst(3)
        var results: [[String: Any]] = []
        var studios: [StudioModel] = []
        defer { for studio in studios { studio.ejectModel() } }
        func wait(_ description: String, until condition: () -> Bool) async throws {
            for _ in 0..<250 {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw ReviewError.deadline(description)
        }
        for mode in modes {
            let clock = ReviewClock()
            preferences.set(false, forKey: "studio.mtpEnabled")
            let studio = StudioModel(conversationFileURL: root.appendingPathComponent(mode + ".json"),
                                     preferences: preferences, idleUnloadNow: { clock.instant })
            studios.append(studio)
            let name = "auto-" + mode
            studio.models = [LocalModel(name: name, path: root.appendingPathComponent(name).path,
                                       modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1,
                                       modules: 1, addedAt: "", size: "1 B")]
            studio.setModelIdleUnloadDelay(.oneMinute)
            studio.selectModel(name)
            try await wait("ready", until: { studio.engineState.isReady })
            clock.advance(59)
            studio.reviewDuplicateReady()
            var configurationResponse = ""
            studio.reviewObserve { event in
                if ["mtp_status", "error"].contains(event.type) { configurationResponse = event.type }
            }
            studio.setMTPEnabled(false)
            try await wait("configuration response handled", until: { !configurationResponse.isEmpty })
            studio.draft = "Prove this engine still responds after configuration"
            studio.send()
            try await wait("generation completed", until: { !studio.isGenerating })
            let completed = studio.conversations[0].messages.last?.content == "hello"
            let pendingAfterCompletion = studio.reviewPending
            clock.advance(60)
            studio.reviewTick()
            let unloaded = studio.engineState == .idle && !studio.reviewRunning
            results.append(["mode": mode, "generation_completed": completed,
                            "configuration_response": configurationResponse,
                            "mtp_pending_after_generation": pendingAfterCompletion,
                            "unloaded_after_full_interval": unloaded,
                            "pass": completed && unloaded])
            print("\(mode): completed=\(completed), pending=\(pendingAfterCompletion), unloaded=\(unloaded)")
            studio.ejectModel()
        }
        let data = try JSONSerialization.data(withJSONObject: results, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
        // Exit after explicit cleanup so an expected failing review check cannot
        // leave a fixture process or a preference domain behind.
        for studio in studios { studio.ejectModel() }
        preferences.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: root)
        if results.contains(where: { $0["pass"] as? Bool != true }) { exit(1) }
    }
}
