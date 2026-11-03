// Physical local-model check, selected explicitly outside hosted CI.
import AppKit
import Foundation

extension MLXL3Bridge {
    fileprivate var fixtureChild: Process? { process }
}
extension StudioModel {
    fileprivate func stopAndReapFixture() async {
        let child = bridge.fixtureChild
        ejectModel()
        if let child { await Task.detached { child.waitUntilExit() }.value }
    }
}

@main struct LocalModelStartupCheck {
    @MainActor static func main() async throws {
        guard CommandLine.arguments.count == 4 else {
            throw NSError(domain: "LocalModelStartupCheck", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "engine model registry required"])
        }
        let engine = CommandLine.arguments[1], name = CommandLine.arguments[2]
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-startup-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let registry = root.appendingPathComponent("models.json")
        try FileManager.default.copyItem(atPath: CommandLine.arguments[3], toPath: registry.path)
        guard let data = try JSONSerialization.jsonObject(with: Data(contentsOf: registry)) as? [String: Any],
              let models = data["models"] as? [String: [String: Any]],
              let path = models[name]?["path"] as? String, !path.isEmpty else {
            throw NSError(domain: "LocalModelStartupCheck", code: 4,
                          userInfo: [NSLocalizedDescriptionKey: "Missing model metadata"])
        }
        func quote(_ value: String) -> String { "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'" }
        let wrapper = root.appendingPathComponent("engine")
        try ("#!/bin/sh\nexec " + quote(engine) + " --registry " + quote(registry.path) + " \"$@\"\n")
            .write(to: wrapper, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: wrapper.path)
        setenv("MLXL3_EXECUTABLE", wrapper.path, 1)
        setenv("MLXL3_HOME", root.path, 1)
        let suite = "io.mlxl3.local-startup." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer {
            prefs.removePersistentDomain(forName: suite)
        }
        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("history.json"),
            updateManager: UpdateManager(releasesURL: root.appendingPathComponent("no-network.json")), preferences: prefs)
        studio.setStartupModelPath(path)
        studio.start()
        do {
            let deadline = ContinuousClock.now.advanced(by: .seconds(180))
            while !studio.engineState.isReady {
                if case let .failed(message) = studio.engineState {
                    throw NSError(domain: "LocalModelStartupCheck", code: 2,
                                  userInfo: [NSLocalizedDescriptionKey: message])
                }
                guard ContinuousClock.now < deadline else {
                    throw NSError(domain: "LocalModelStartupCheck", code: 3,
                                  userInfo: [NSLocalizedDescriptionKey: "Startup deadline"])
                }
                try await Task.sleep(for: .milliseconds(100))
            }
            guard studio.loadedModel?.name == name else {
                throw NSError(domain: "LocalModelStartupCheck", code: 5,
                              userInfo: [NSLocalizedDescriptionKey: "Wrong model loaded"])
            }
            print("Real Desktop startup passed: " + name + "; " + (studio.runtimeIdentity ?? "unknown runtime"))
            await studio.stopAndReapFixture()
            print("Local model engine stopped and reaped")
        } catch {
            await studio.stopAndReapFixture()
            throw error
        }
    }
}
