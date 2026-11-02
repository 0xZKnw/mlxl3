import AppKit
import Foundation

@MainActor private final class NotificationSpy {
    var allowed = true
    var requests = 0
    var ids: [String] = []
    var delay: Duration = .zero
    var failDelivery = false
    var client: CompletionNotificationClient {
        CompletionNotificationClient(requestAuthorization: { [self] in
            requests += 1
            try await Task.sleep(for: delay)
            return allowed
        }, deliver: { [self] id in
            if failDelivery { throw NSError(domain: "fixture-delivery", code: 1) }
            ids.append(id)
        })
    }
}

extension StudioModel {
    fileprivate func fixtureEvent(_ json: String) throws {
        handle(try JSONDecoder().decode(BridgeEvent.self, from: Data(json.utf8)))
    }
    fileprivate var fixtureRequestID: String? { activeRequestID }
}

@main struct CompletionNotificationsCheck {
    @MainActor static func main() async throws {
        var checks = 0
        func check(_ condition: Bool, _ message: String) {
            checks += 1; precondition(condition, message)
        }
        func wait(_ label: String, until condition: () -> Bool) async throws {
            for _ in 0..<300 {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw NSError(domain: "CompletionNotificationsCheck", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: label])
        }
        let suite = "io.mlxl3.notifications." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(suite)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        let spy = NotificationSpy()
        var active = false
        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("history.json"),
            updateManager: UpdateManager(releasesURL: root.appendingPathComponent("no-network.json")),
            preferences: prefs, notificationClient: spy.client, applicationIsActive: { active })
        defer {
            studio.ejectModel()
            prefs.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }
        check(!studio.completionNotificationsEnabled && spy.requests == 0, "Must default off without an OS prompt")
        spy.allowed = false
        studio.setCompletionNotificationsEnabled(true)
        try await wait("permission rejection", until: { studio.notificationStatus != nil })
        check(!studio.completionNotificationsEnabled, "Denied permission enabled notifications")
        spy.allowed = true
        spy.delay = .milliseconds(150)
        studio.setCompletionNotificationsEnabled(true)
        studio.setCompletionNotificationsEnabled(false)
        try await Task.sleep(for: .milliseconds(250))
        check(!studio.completionNotificationsEnabled, "Late permission callback reversed opt-out")
        spy.delay = .zero
        studio.setCompletionNotificationsEnabled(true)
        try await wait("permission grant", until: { studio.completionNotificationsEnabled })
        check(prefs.bool(forKey: "studio.completionNotificationsEnabled"), "Notification preference was not persisted")
        studio.models = [LocalModel(name: "notification-fixture", path: root.path,
            modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")]
        studio.selectModel("notification-fixture")
        try await wait("model ready", until: { studio.engineState.isReady })
        func generate() async throws {
            studio.draft = "A private prompt that must never appear in notifications"
            studio.send()
            try await wait("generation finished", until: { !studio.isGenerating })
            try await Task.sleep(for: .milliseconds(30))
        }
        try await generate()
        check(spy.ids.count == 1, "Successful background generation did not notify")
        check(!spy.ids[0].contains("private"), "Notification identifier leaked prompt")
        try studio.fixtureEvent("{\"type\":\"complete\"}")
        try studio.fixtureEvent("{\"type\":\"complete\",\"request_id\":\"" + spy.ids[0] + "\"}")
        try await Task.sleep(for: .milliseconds(30))
        check(spy.ids.count == 1, "Unsolicited or duplicate completion notified")
        active = true
        try await generate()
        check(spy.ids.count == 1, "Foreground generation notified")
        active = false
        studio.draft = "Cancel this"; studio.send()
        studio.stopGeneration()
        try await wait("cancel race completion", until: { !studio.isGenerating })
        try await Task.sleep(for: .milliseconds(30))
        check(spy.ids.count == 1, "Completion after stop notified")
        studio.draft = "Error"; studio.send()
        let requestID = studio.fixtureRequestID!
        try studio.fixtureEvent("{\"type\":\"error\",\"request_id\":\"" + requestID + "\",\"message\":\"fixture failure\"}")
        try await Task.sleep(for: .milliseconds(300))
        check(spy.ids.count == 1, "Failed generation notified")
        studio.setCompletionNotificationsEnabled(false)
        try await generate()
        check(spy.ids.count == 1, "Opt-out ignored")
        studio.setCompletionNotificationsEnabled(true)
        try await wait("permission regrant", until: { studio.completionNotificationsEnabled })
        spy.failDelivery = true
        try await generate()
        check(studio.notificationStatus != nil && studio.engineState.isReady,
              "Notification delivery error broke model or vanished")
        // A failed first load can leave an engine alive. Selecting the same
        // model must retry instead of requiring an unrelated model switch.
        studio.ejectModel()
        studio.models.append(LocalModel(name: "first-load-fails", path: root.path + "/first-load-fails",
            modelType: "audit", format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B"))
        studio.selectModel("first-load-fails")
        try await wait("first-load error", until: { if case .failed = studio.engineState { true } else { false } })
        studio.retrySelectedModel()
        try await wait("same-model retry", until: { studio.engineState.isReady })
        check(studio.loadedModel?.name == "first-load-fails", "Failed model required an unrelated model switch")
        // Enqueued loads and cancellation must preserve the latest intent.
        studio.selectModel("notification-fixture")
        studio.selectModel("first-load-fails")
        studio.ejectModel()
        try await Task.sleep(for: .milliseconds(250))
        check(studio.engineState == .idle && studio.fixtureRequestID == nil, "A superseded load survived ejection")
        print("Completion notification checks passed: \(checks) assertions; permission, persistence, background/foreground, cancellation, errors, duplicates")
    }
}
