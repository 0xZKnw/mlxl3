import AppKit
import Foundation
import SwiftUI

// check-desktop.sh concatenates this file with StudioModel.swift so the fixture
// can observe the real bridge process without a shipped testing entry point.
extension StudioModel {
    fileprivate var fixtureEngineRunning: Bool { bridge.isRunning }
}

@main struct ModelUnloadControlCheck {
    @MainActor static func main() async throws {
        var checks = 0
        func require(_ condition: Bool, _ message: String) throws {
            checks += 1
            if !condition {
                throw NSError(domain: "ModelUnloadControlCheck", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: message])
            }
        }
        func wait(_ description: String, attempts: Int = 250, until condition: () -> Bool) async throws {
            for _ in 0..<attempts {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            throw NSError(domain: "ModelUnloadControlCheck", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "Deadline waiting for " + description])
        }

        // Finite exhaustive check of the pure state: nothing to release hides the
        // button whatever the activity; busy work only ever disables it.
        let table: [(Bool, Bool, ModelUnloadControl)] = [
            (false, false, .hidden), (false, true, .hidden),
            (true, false, .enabled), (true, true, .disabled),
        ]
        for (canEject, busy, expected) in table {
            try require(ModelUnloadControl(canEject: canEject, busy: busy) == expected,
                        "State table mismatch for canEject=\(canEject) busy=\(busy)")
        }

        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-unload-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        setenv("MLXL3_HOME", root.path, 1)
        let suite = "io.mlxl3.unload-check." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        // Manual Tune scenarios; the automatic Tune and the toolbar are covered by tests/mtp-autotune-check.swift.
        preferences.set(false, forKey: StudioModel.mtpAutoTunePreferenceKey)
        var instances: [StudioModel] = []
        defer {
            for studio in instances { studio.ejectModel() }
            preferences.removePersistentDomain(forName: suite)
            try? FileManager.default.removeItem(at: root)
        }
        func make(_ names: [String]) -> StudioModel {
            let studio = StudioModel(conversationFileURL: root.appendingPathComponent(UUID().uuidString + ".json"),
                                     preferences: preferences)
            instances.append(studio)
            studio.models = names.map {
                LocalModel(name: $0, path: root.appendingPathComponent($0).path, modelType: "audit",
                           format: "EXL3", bits: 3, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
            }
            return studio
        }

        // Hosted toolbar capsule: the button exists only when something can be
        // unloaded, so the shared capsule grows by exactly one 30 pt control.
        var snapshots = 0
        func capsuleWidth(_ studio: StudioModel, _ label: String) async throws -> CGFloat {
            let host = NSHostingView(rootView: ModelSelectionMenu().environmentObject(studio)
                .studioAppearance(.dark).padding(10).background(StudioTheme.canvas))
            let size = host.fittingSize
            try require(size.width.isFinite && size.width > 0 && size.height > 0 && size.height <= 60,
                        label + ": model capsule has invalid bounds \(size)")
            if let output = ProcessInfo.processInfo.environment["MLXL3_UNLOAD_SNAPSHOTS"] {
                let frame = NSRect(x: 0, y: 0, width: max(size.width, 120), height: size.height)
                let window = NSWindow(contentRect: frame, styleMask: .borderless, backing: .buffered, defer: false)
                window.isReleasedWhenClosed = false; window.contentView = host
                host.frame = frame
                try await Task.sleep(for: .milliseconds(120)); host.layoutSubtreeIfNeeded()
                let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                host.cacheDisplay(in: host.bounds, to: bitmap)
                try bitmap.representation(using: .png, properties: [:])!.write(
                    to: URL(fileURLWithPath: output).appendingPathComponent("unload-" + label + ".png"))
                window.close()
                snapshots += 1
            }
            return size.width
        }

        let empty = make(["unload-ready"])
        try require(empty.modelUnloadControl == .hidden, "Unload offered with no engine")
        empty.unloadModelFromToolbar()
        try require(empty.engineState == .idle && !empty.fixtureEngineRunning, "Hidden unload changed the engine")
        _ = try await capsuleWidth(empty, "no-model")

        // Loading: the button cancels the load and a late ready event is ignored.
        let loading = make(["unload-loading"])
        loading.selectModel("unload-loading")
        try require(loading.engineState == .loading("unload-loading"), "Fixture did not observe the loading state")
        try require(loading.modelUnloadControl == .enabled, "Loading model cannot be cancelled")
        loading.unloadModelFromToolbar()
        try require(loading.engineState == .idle && !loading.fixtureEngineRunning, "Load cancellation kept the engine")
        try await Task.sleep(for: .milliseconds(400))
        try require(loading.engineState == .idle && loading.loadedModel == nil, "Late ready event revived a cancelled load")

        // Ready → generating (disabled, no-op) → finished → unloaded → reload.
        let studio = make(["auto-idle-silent"])
        studio.selectModel("auto-idle-silent")
        try await wait("model ready", until: { studio.engineState.isReady && !studio.mtpDownloading })
        try require(studio.modelUnloadControl == .enabled, "Ready model cannot be unloaded")
        try require(studio.loadedModel?.name == "auto-idle-silent" && studio.fixtureEngineRunning, "Model not loaded")
        let enabledWidth = try await capsuleWidth(studio, "enabled")

        studio.draft = "Keep this reply"; studio.send()
        try require(studio.isGenerating, "Fixture did not start the generation under test")
        try require(studio.modelUnloadControl == .disabled, "Unload enabled during a generation")
        let disabledWidth = try await capsuleWidth(studio, "generating")
        try require(abs(disabledWidth - enabledWidth) < 0.5, "Disabled button changed the capsule layout")
        studio.unloadModelFromToolbar()
        try require(studio.isGenerating && studio.fixtureEngineRunning && studio.engineState != .idle,
                    "Disabled unload interrupted the generation")
        try require(studio.conversations.first?.messages.last?.isStreaming == true, "Reply stopped streaming")
        try Data().write(to: root.appendingPathComponent("release-idle-generation"))
        try await wait("generation complete", until: { !studio.isGenerating })
        try FileManager.default.removeItem(at: root.appendingPathComponent("release-idle-generation"))
        let history = studio.conversations.first!.messages.map(\.content)
        try require(history.last == "hello", "Generation was lost")
        try require(studio.modelUnloadControl == .enabled, "Unload not re-enabled after the generation")

        studio.unloadModelFromToolbar()
        try require(studio.engineState == .idle && !studio.fixtureEngineRunning, "Unload kept the engine process")
        try require(studio.loadedModel == nil && studio.modelContextLimit == nil && (studio.engineMemoryFootprintBytes ?? 0) == 0,
                    "Unload left loaded-model state or engine memory")
        try require(studio.modelUnloadControl == .hidden && !studio.canSend, "UI still offers a loaded model")
        try require(studio.selectedModelName == "auto-idle-silent", "Unload lost the model selection")
        // Same model name, so the only width change is the attached 30 pt button.
        let hiddenWidth = try await capsuleWidth(studio, "unloaded")
        try require(enabledWidth - hiddenWidth >= 30 && enabledWidth - hiddenWidth <= 40,
                    "Unload button is not attached to the model capsule (\(hiddenWidth) → \(enabledWidth))")
        try require(studio.conversations.first!.messages.map(\.content) == history, "Unload changed history")
        try await Task.sleep(for: .milliseconds(200))
        try require(studio.engineState == .idle, "Engine reloaded by itself after unload")
        studio.selectModel("auto-idle-silent")
        try await wait("reload", until: { studio.engineState.isReady })
        try require(studio.modelUnloadControl == .enabled, "Reloaded model cannot be unloaded")

        studio.unloadModelFromToolbar()
        try require(studio.engineState == .idle && !studio.fixtureEngineRunning, "Second unload failed")

        // Stopping a reply first is the supported path to unload during work.
        // This fixture model honours cancellation (auto-idle-silent does not, and
        // would exercise the force-stop fallback instead).
        let stopping = make(["stop-unload"])
        stopping.selectModel("stop-unload")
        try await wait("stop model ready", until: { stopping.engineState.isReady && !stopping.mtpDownloading })
        stopping.draft = "Stop me"; stopping.send()
        try require(stopping.modelUnloadControl == .disabled, "Unload enabled during the stopped generation")
        stopping.stopGeneration()
        try await wait("generation stopped", attempts: 400, until: { !stopping.isGenerating })
        try require(stopping.engineState.isReady && stopping.modelUnloadControl == .enabled,
                    "Stopped reply did not re-enable unload")
        stopping.unloadModelFromToolbar()
        try require(stopping.engineState == .idle && !stopping.fixtureEngineRunning, "Unload after stop failed")

        // MTP tuning also owns the engine.
        let tuning = make(["auto-dense"])
        tuning.selectModel("auto-dense")
        try await wait("tuning model ready", until: { tuning.engineState.isReady && !tuning.mtpDownloading })
        tuning.setMTPEnabled(true)
        try await wait("MTP head", until: { !tuning.mtpDownloading })
        try require(tuning.canTuneMTP, "Fixture tuning not available")
        tuning.tuneMTP()
        try require(tuning.isTuningMTP && tuning.modelUnloadControl == .disabled, "Unload enabled during MTP tuning")
        tuning.unloadModelFromToolbar()
        try require(tuning.isTuningMTP && tuning.fixtureEngineRunning, "Disabled unload interrupted tuning")
        try await wait("tuning finished", until: { !tuning.isTuningMTP })
        try require(tuning.modelUnloadControl == .enabled, "Unload not re-enabled after tuning")
        tuning.unloadModelFromToolbar()
        try require(tuning.engineState == .idle && !tuning.fixtureEngineRunning, "Unload after tuning failed")

        print("Model unload control checks passed: \(checks) assertions, \(snapshots) snapshots")
    }
}
