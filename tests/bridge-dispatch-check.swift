// Run from the repository root in zsh; the fixture never loads MLX:
// test_dir="$(mktemp -d -t mlxl3-bridge-check)"
// CLANG_MODULE_CACHE_PATH="$test_dir/modules" swift build --disable-sandbox --package-path apps/MLXL3Studio
// sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
// sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
// sources=("${(@)sources:#*/MLXL3Bridge.swift}")
// awk '{ print }' apps/MLXL3Studio/Sources/MLXL3Studio/MLXL3Bridge.swift tests/bridge-dispatch-check.swift > "$test_dir/BridgeDispatchCombined.swift"
// swiftc -O -swift-version 6 -parse-as-library -module-cache-path "$test_dir/modules" -I apps/MLXL3Studio/.build/arm64-apple-macosx/debug/Modules "${sources[@]}" apps/MLXL3Studio/.build/arm64-apple-macosx/debug/SwiftMath.build/*.swift.o "$test_dir/BridgeDispatchCombined.swift" -o "$test_dir/check"
// "$test_dir/check" "$PWD/tests/fake-desktop-engine.py"
// Concatenation grants this test extension access to the actual private dispatcher, without production test hooks.
import Foundation

// This file-local decoder only instruments the concatenated bridge translation unit.
// Actual JSON decoding still uses Foundation; the one-shot gate makes an in-flight stop deterministic.
private enum DecodeGate {
    static let next = Mutex<(@Sendable () -> Void)?>(nil)
}

private final class JSONDecoder {
    func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        let gate = DecodeGate.next.withLock { next in
            defer { next = nil }
            return next
        }
        gate?()
        return try Foundation.JSONDecoder().decode(type, from: data)
    }
}

@MainActor private final class DispatchSamples {
    var eventDelay: Double?
    var readyModels: [String] = []
    var exitCount = 0
    var deliveries = 0
}

extension MLXL3Bridge {
    @MainActor fileprivate func checkDispatchLatency(legacy: Bool) async -> Double {
        let event = try! JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"ready"}"#.utf8))
        let samples = DispatchSamples()
        let started = ProcessInfo.processInfo.systemUptime
        onEvent = { _ in samples.eventDelay = ProcessInfo.processInfo.systemUptime - started }
        let enteredIO = DispatchSemaphore(value: 0)
        let releaseIO = DispatchSemaphore(value: 0)
        let ticket = generationID.withLock { $0 }
        let uiDelay: Double = await withCheckedContinuation { continuation in
            ioQueue.async {
                if legacy {
                    DispatchQueue.main.async {
                        guard self.ioQueue.sync(execute: { self.generationID.withLock { $0 == ticket } }) else { return }
                        self.onEvent?(event)
                    }
                } else {
                    self.dispatchToMain(event, ticket: ticket)
                }
                enteredIO.signal()
                precondition(releaseIO.wait(timeout: .now() + 2) == .success)
            }
            precondition(enteredIO.wait(timeout: .now() + 2) == .success)
            DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(50)) { releaseIO.signal() }
            DispatchQueue.main.async {
                continuation.resume(returning: ProcessInfo.processInfo.systemUptime - started)
            }
        }
        await withCheckedContinuation { continuation in ioQueue.async { continuation.resume() } }
        precondition(samples.eventDelay != nil && samples.eventDelay! <= uiDelay)
        return uiDelay
    }

    @MainActor fileprivate func checkStaleDispatch() async {
        let event = try! JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"ready"}"#.utf8))
        let samples = DispatchSamples()
        onEvent = { _ in samples.deliveries += 1 }
        let oldTicket = generationID.withLock { $0 }
        ioQueue.sync { dispatchToMain(event, ticket: oldTicket) }
        stop() // Invalidate after scheduling the callback, before MainActor can deliver it.
        await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
        precondition(samples.deliveries == 0, "Stale callback survived stop")
        let newTicket = generationID.withLock { $0 }
        ioQueue.sync { dispatchToMain(event, ticket: newTicket) }
        await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
        precondition(samples.deliveries == 1, "Current generation callback was discarded")
    }

    @MainActor fileprivate func checkStopDuringDecode() async {
        let event = try! JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"ready"}"#.utf8))
        let samples = DispatchSamples()
        onEvent = { _ in samples.deliveries += 1 }
        let ticket = generationID.withLock { $0 }
        let enteredDecode = DispatchSemaphore(value: 0)
        let releaseDecode = DispatchSemaphore(value: 0)
        let invalidatedBeforeRelease = Mutex(false)
        DecodeGate.next.withLock { next in
            next = {
                enteredDecode.signal()
                precondition(releaseDecode.wait(timeout: .now() + 2) == .success)
            }
        }
        ioQueue.async {
            self.dispatchToMain(event, ticket: ticket)
            // Cover pending-delta flushing, direct events, and the decode-error path after invalidation.
            let lines = [#"{"type":"delta","request_id":"old","text":"obsolete"}"#,
                         #"{"type":"complete","request_id":"old"}"#, "invalid JSON"]
            self.consumeOutput(Data((lines.joined(separator: "\n") + "\n").utf8), ticket: ticket)
        }
        precondition(enteredDecode.wait(timeout: .now() + 2) == .success)
        DispatchQueue.global().asyncAfter(deadline: .now() + .milliseconds(50)) {
            let invalidated = self.generationID.withLock { $0 != ticket }
            invalidatedBeforeRelease.withLock { $0 = invalidated }
            releaseDecode.signal()
        }
        stop() // The queued MainActor callback is still waiting while IO is inside consumeOutput.
        await withCheckedContinuation { continuation in DispatchQueue.main.async { continuation.resume() } }
        precondition(invalidatedBeforeRelease.withLock { $0 }, "Stop invalidated only after waiting for IO")
        precondition(samples.deliveries == 0, "In-flight decoding relabeled old output as the new generation")
    }
}

@main struct BridgeDispatchCheck {
    @MainActor static func main() async throws {
        let bridge = MLXL3Bridge()
        let legacy = await bridge.checkDispatchLatency(legacy: true)
        let independent = await bridge.checkDispatchLatency(legacy: false)
        precondition(legacy >= 0.04, "Fixture did not block the legacy queue check")
        precondition(independent < legacy / 2, "UI callback still waited for the occupied IO queue")
        await bridge.checkStaleDispatch()
        await bridge.checkStopDuringDecode()

        precondition(CommandLine.arguments.count == 2)
        setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
        let samples = DispatchSamples()
        bridge.onEvent = { event in
            if event.type == "ready", let model = event.model { samples.readyModels.append(model) }
        }
        bridge.onExit = { _ in samples.exitCount += 1 }
        for model in ["first", "second", "crash"] {
            try await bridge.start(model: model)
            for _ in 0..<100 where !samples.readyModels.contains(model) { try await Task.sleep(for: .milliseconds(10)) }
            precondition(samples.readyModels.contains(model) && bridge.isRunning)
            precondition(samples.exitCount == 0, "Previous process exit leaked into a new generation")
            precondition(bridge.cancelGeneration(), "Cooperative cancellation signal failed")
        }
        try bridge.generate(GenerationRequest(requestID: "check", conversationID: "check",
                                             messages: [PromptMessage(role: "user", content: "hello")],
                                             maxTokens: 1, temperature: 0, topK: 0, repetitionPenalty: 1))
        for _ in 0..<100 where samples.exitCount == 0 { try await Task.sleep(for: .milliseconds(10)) }
        precondition(samples.exitCount == 1 && !bridge.isRunning, "Current process exit was discarded")
        bridge.stop()
        precondition(!bridge.cancelGeneration())
        print(String(format: "Bridge callback with 50 ms occupied IO: legacy %.3f ms; independent UUID %.3f ms", legacy * 1_000, independent * 1_000))
        print("Bridge checks passed: stale/current callbacks, stop during decoding, reload, cancellation, stale/current process exit")
    }
}
