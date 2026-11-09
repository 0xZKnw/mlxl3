import AppKit
import Darwin
import Foundation
import SwiftUI

// Concatenated with StudioModel.swift; exercises the production ready/cancel paths.
extension StudioModel {
    func checkRuntimeReadyAndCancellation(executable: URL) async throws {
        try await bridge.start(model: "ignore")
        for _ in 0..<200 {
            if engineState.isReady { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(engineState.isReady)
        precondition(contextMemory?.bytes(tokens: 2048) == 125829220)
        precondition(contextMemory?.bytes(tokens: 8192) == 503316580)
        precondition(runtimeIdentity?.contains("direct-metal") == true)
        precondition(runtimeIdentity?.contains("qwen_device_kv=true") == true)
        precondition(runtimeIdentity?.contains("qwen_moe_join=false") == true)
        precondition(runtimeQwenProfile != nil && runtimeIdentity?.contains("Qwen profile") == true)
        // The configured selection is intentionally independent of the running Tune.
        let key = MTPConfigurationKey(modelPath: executable.path, headPath: executable.path, runtime: "fixture")
        MTPTuning.save(MTPSelection(key: key, depth: 2, rows: [], tunedAt: nil), preferences: preferences)
        let savedSelection = preferences.data(forKey: "studio.mtpSelections")
        mtpDepth = 2
        isTuningMTP = true; tuneRequestID = "old"; tuneCancellationRequested = false
        cancelMTPTuning()
        precondition(tuneCancellationFallback != nil)
        finishMTPTuning(error: nil)
        isTuningMTP = true; tuneRequestID = "new"; tuneCancellationRequested = false
        let stale = try JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"mtp_tune_cancelled","request_id":"old"}"#.utf8))
        handle(stale)
        try await Task.sleep(for: .milliseconds(3200))
        precondition(isTuningMTP && tuneRequestID == "new" && bridge.isRunning,
                     "Old cancellation/ack stopped the new Tune")
        let started = ContinuousClock.now
        cancelMTPTuning()
        for _ in 0..<400 {
            if !isTuningMTP && !bridge.isRunning { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(!isTuningMTP && !bridge.isRunning)
        precondition(preferences.data(forKey: "studio.mtpSelections") == savedSelection, "Cancellation overwrote saved tuning")
        precondition(ContinuousClock.now - started < .seconds(8))
        await bridge.stopAndWait()
        // A new load and Tune ticket remain possible after forced recovery.
        try await bridge.start(model: "good")
        for _ in 0..<200 {
            if engineState.isReady { break }
            try await Task.sleep(for: .milliseconds(20))
        }
        precondition(engineState.isReady)
        isTuningMTP = true; tuneRequestID = "after-recovery"; tuneCancellationRequested = false
        precondition(isTuningMTP && bridge.isRunning)
        finishMTPTuning(error: nil)
        await bridge.stopAndWait()
    }
}

@main struct Issue163RuntimeContractCheck {
    @MainActor static func main() async throws {
        let files = FileManager.default
        let root = files.temporaryDirectory.appendingPathComponent("issue163-runtime-" + UUID().uuidString)
        try files.createDirectory(at: root, withIntermediateDirectories: true)
        let suite = "io.mlxl3.runtime-check." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite); try? files.removeItem(at: root) }
        let overrides = ["MLXL3_NATIVE_QWEN_DEVICE_KV": "0", "MLXL3_NATIVE_QWEN_PROFILE": "optimized", "OTHER": "keep"]
        precondition(QwenRuntimeProfile.environment(overrides, requested: "reference") == overrides)
        precondition(QwenRuntimeProfile.environment([:], requested: nil)[QwenRuntimeProfile.environmentKey] == "reference")
        precondition(QwenRuntimeProfile.environment([:], requested: "invalid")[QwenRuntimeProfile.environmentKey] == "reference")
        let requested = QwenRuntimeProfile.environment(["MLXL3_NATIVE_QWEN_DEVICE_KV": "0"], requested: "optimized")
        precondition(requested[QwenRuntimeProfile.environmentKey] == "optimized" && requested["MLXL3_NATIVE_QWEN_DEVICE_KV"] == "0")
        let script = #"""
        #!/bin/sh
        case "$2" in
          bad-model|bad-context|low-memory) echo 'model setup failed' >&2; exit 1;;
          ignore) trap '' USR1 TERM;;
        esac
        protocol=1
        if [ "$2" = bad-protocol ]; then protocol=99; fi
        printf '{"type":"ready","model":"fixture","bridge_protocol":%s,"backend":"direct-metal","runtime_version":"2.0.2","qwen_profile":"%s","qwen_effective_flags":{"device_kv":true,"moe_join":false,"device_greedy":false},"qwen_device_kv":true,"context_memory_profile":{"fixed_bytes":100,"layers":[{"bytes_per_token":61440,"step":64}]}}\n' "$protocol" "${MLXL3_NATIVE_QWEN_PROFILE:-reference}"
        while IFS= read -r line; do :; done
        """
        func stage(_ suffix: String, alternate: Bool = false) throws -> URL {
            let directory = root.appendingPathComponent("stage-" + UUID().uuidString)
            try files.createDirectory(at: directory, withIntermediateDirectories: true)
            let executable = directory.appendingPathComponent("mlxl3")
            try Data((script + (alternate ? "\n# alternate" : "")).utf8).write(to: executable)
            try files.setAttributes([.posixPermissions: 0o700], ofItemAtPath: executable.path)
            let metadata: [String: Any] = ["schema": 2, "backend": "metal", "version": "2.0.2",
                "release_tag": "engine-v2.0.2" + suffix, "bridge_protocol": 1, "architecture": "arm64",
                "minimum_macos": "1.0.0", "minimum_app_version": "2.0.0",
                "files": ["mlxl3": ["size": try Data(contentsOf: executable).count,
                                    "sha256": try UpdateManager.sha256(at: executable)]]]
            try JSONSerialization.data(withJSONObject: metadata, options: [.sortedKeys]).write(to: directory.appendingPathComponent("engine.json"))
            return directory
        }
        func release(_ suffix: String, _ digest: String) -> AppUpdateRelease {
            AppUpdateRelease(version: "2.0.2", tag: "engine-v2.0.2" + suffix, title: "fixture", notes: "",
                pageURL: root, asset: AppUpdateAsset(name: "fixture", downloadURL: root, size: 1,
                    digest: "sha256:" + String(repeating: digest, count: 64)))
        }
        func resolve(_ bundled: String = "2.0.1") -> URL? {
            EngineRuntimeStore.resolve(appVersion: "2.0.0", root: root, bundledVersion: bundled)
        }
        func fails(_ reason: String, _ body: () throws -> Void) {
            do { try body(); preconditionFailure(reason) } catch {}
        }
        let identities = ["2.0.2-beta.1", "2.0.2-beta.2", "2.0.2-legacy", "2.0.2", "2.0.3-beta.1"]
        for i in identities.indices {
            for j in identities.indices {
                precondition((EngineReleaseIdentity(identities[i])! < EngineReleaseIdentity(identities[j])!) == (i < j))
            }
        }
        for invalid in ["../2.0.2", "2.0.2-beta.01", "2.0.2-beta.0", "2.0.2-beta.1/..", "2.0.2-rc.1"] {
            precondition(EngineReleaseIdentity(invalid) == nil)
        }
        // Legacy pointers migrate by activation, preserving the original directory.
        let legacyRoot = root.appendingPathComponent("legacy")
        try files.createDirectory(at: legacyRoot, withIntermediateDirectories: true)
        let legacyStage = try stage("")
        let legacy = legacyRoot.appendingPathComponent("v2.0.2")
        try files.moveItem(at: legacyStage, to: legacy)
        try EngineRuntimeStore.activate(version: "2.0.2", appVersion: "2.0.0", root: legacyRoot)
        precondition(EngineRuntimeStore.resolve(appVersion: "2.0.0", root: legacyRoot, bundledVersion: "2.0.1") == legacy.appendingPathComponent("mlxl3"))
        precondition(EngineRuntimeStore.resolve(appVersion: "2.0.0", root: legacyRoot, bundledVersion: "2.0.2-beta.2") == nil)
        fails("Unknown historical stable must not downgrade to beta") {
            try EngineRuntimeStore.publishValidatedStage(stage("-beta.2"), release: release("-beta.2", "b"), appVersion: "2.0.0", root: legacyRoot)
        }
        try EngineRuntimeStore.publishValidatedStage(stage(""), release: release("", "c"), appVersion: "2.0.0", root: legacyRoot)
        precondition(files.fileExists(atPath: legacy.path))
        precondition(EngineRuntimeStore.resolve(appVersion: "2.0.0", root: legacyRoot, bundledVersion: "2.0.2-beta.2") != nil)
        var selected: [URL] = []
        for (suffix, digest) in [("-beta.1", "a"), ("-beta.2", "b"), ("", "c")] {
            try EngineRuntimeStore.publishValidatedStage(stage(suffix), release: release(suffix, digest), appVersion: "2.0.0", root: root)
            selected.append(resolve()!)
        }
        precondition(Set(selected).count == 3)
        precondition(resolve("2.0.2-beta.2") == selected.last && resolve("2.0.2") == nil)
        let pointer = try Data(contentsOf: root.appendingPathComponent("current.json"))
        fails("Stable downgraded to beta") {
            try EngineRuntimeStore.publishValidatedStage(stage("-beta.1"), release: release("-beta.1", "a"), appVersion: "2.0.0", root: root)
        }
        fails("Candidate content substituted under same digest") {
            try EngineRuntimeStore.publishValidatedStage(stage("", alternate: true), release: release("", "c"), appVersion: "2.0.0", root: root)
        }
        let unchangedPointer = try Data(contentsOf: root.appendingPathComponent("current.json"))
        precondition(unchangedPointer == pointer)
        try EngineRuntimeStore.publishValidatedStage(stage(""), release: release("", "c"), appVersion: "2.0.0", root: root)
        precondition(resolve() == selected.last, "Identical archive not idempotent")
        try EngineRuntimeStore.publishValidatedStage(stage("", alternate: true), release: release("", "d"), appVersion: "2.0.0", root: root)
        let active = resolve()!
        precondition(active != selected.last)
        let originalManifest = try Data(contentsOf: active.deletingLastPathComponent().appendingPathComponent("engine.json"))
        let originalExecutable = try Data(contentsOf: active)
        let substitute = try stage("")
        try Data(contentsOf: substitute.appendingPathComponent("engine.json")).write(to: active.deletingLastPathComponent().appendingPathComponent("engine.json"))
        try Data(contentsOf: substitute.appendingPathComponent("mlxl3")).write(to: active)
        precondition(resolve() == nil, "Self-consistent manifest replacement bypassed pointer binding")
        try originalManifest.write(to: active.deletingLastPathComponent().appendingPathComponent("engine.json"))
        try originalExecutable.write(to: active)
        precondition(resolve() == active)
        // Run the actual resolver and bridge, with an isolated managed store.
        let oldOverride = ProcessInfo.processInfo.environment["MLXL3_EXECUTABLE"]
        unsetenv("MLXL3_EXECUTABLE")
        defer {
            if let oldOverride { setenv("MLXL3_EXECUTABLE", oldOverride, 1) }
            else { unsetenv("MLXL3_EXECUTABLE") }
        }
        let bridge = MLXL3Bridge(runtimeRoot: root, appVersion: "2.0.0")
        var exits = 0, ready = 0, fallbacks = 0
        bridge.onExit = { _ in exits += 1 }
        bridge.onEvent = { if $0.type == "ready" { ready += 1 } }
        bridge.onRuntimeFallback = { fallbacks += 1 }
        func wait(_ condition: () -> Bool) async throws {
            for _ in 0..<250 {
                if condition() { return }
                try await Task.sleep(for: .milliseconds(20))
            }
            preconditionFailure("Runtime fixture timed out")
        }
        for model in ["bad-model", "bad-context", "low-memory"] {
            let previous = exits
            try await bridge.start(model: model)
            try await wait { exits > previous }
            precondition(resolve() == active && fallbacks == 0)
            precondition(!files.fileExists(atPath: active.deletingLastPathComponent().appendingPathComponent("disabled").path))
        }
        try await bridge.start(model: "good")
        try await wait { ready == 1 }
        precondition(bridge.isRunning && resolve() == active)
        await bridge.stopAndWait()
        try await bridge.start(model: "bad-protocol")
        try await wait { fallbacks == 1 }
        precondition(resolve() == nil)
        await bridge.stopAndWait()
        for code in [POSIXErrorCode.ENOMEM, .EAGAIN, .EMFILE] {
            precondition(!EngineRuntimeStore.isExecutableLaunchFailure(NSError(domain: NSPOSIXErrorDomain, code: Int(code.rawValue))))
        }
        precondition(EngineRuntimeStore.isExecutableLaunchFailure(POSIXError(.ENOEXEC)))
        let wrappedMemoryFailure = NSError(domain: NSCocoaErrorDomain, code: CocoaError.Code.executableLoad.rawValue,
            userInfo: [NSUnderlyingErrorKey: POSIXError(.ENOMEM)])
        precondition(!EngineRuntimeStore.isExecutableLaunchFailure(wrappedMemoryFailure))
        // Telemetry round-trips retain native allocation without inventing MLX zeros.
        for raw in [#"{"native_metal_allocated_bytes":123,"mlx_peak_bytes":0}"#, #"{"mlx_peak_bytes":456}"#, "{}"] {
            let memory = try JSONDecoder().decode(EngineMemoryStats.self, from: Data(raw.utf8))
            let roundtrip = try JSONDecoder().decode(EngineMemoryStats.self, from: JSONEncoder().encode(memory))
            precondition(roundtrip == memory)
            if raw == "{}" { precondition(memory.mlxPeakBytes == nil && memory.nativeMetalAllocatedBytes == nil) }
        }
        let legacyReady = try JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"ready","context_memory":{"fixed_bytes":100,"layers":[{"bytes_per_token":61440,"step":64}]}}"#.utf8))
        precondition(legacyReady.contextMemory?.bytes(tokens: 2048) == 125829220 && legacyReady.nativeContextMemory == nil)
        let unavailable = try JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"ready","context_memory_profile":null}"#.utf8))
        precondition(unavailable.nativeContextMemory == nil && unavailable.contextMemory == nil)
        let standalone = root.appendingPathComponent("engine-fixture")
        try Data(script.utf8).write(to: standalone)
        try files.setAttributes([.posixPermissions: 0o700], ofItemAtPath: standalone.path)
        setenv("MLXL3_EXECUTABLE", standalone.path, 1)
        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("conversation.json"), preferences: prefs)
        try await studio.checkRuntimeReadyAndCancellation(executable: standalone)
        print("Runtime contracts passed: release transitions, model-local failure, protocol fallback, native telemetry, bounded tuner recovery")
    }
}
