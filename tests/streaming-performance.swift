// Compile with the production sources; concatenate after MLXL3Bridge.swift
// so the benchmark can terminate and join its own disposable engine process.
import AppKit
import CryptoKit
import Darwin
import SwiftUI

extension MLXL3Bridge {
  fileprivate func stopAndJoin() async {
    let child = process
    stop()
    if let child { await Task.detached { child.waitUntilExit() }.value }
  }
}

private enum PerformanceError: Error {
  case invalid(String)
}

@main struct StreamingPerformance {
  @MainActor private static func window(_ message: ChatMessage, studio: StudioModel) -> NSWindow {
    let host = NSHostingView(rootView: MessagesView(messages: [message]).environmentObject(studio))
    let window = NSWindow(
      contentRect: NSRect(x: 80, y: 80, width: 900, height: 700),
      styleMask: [.titled], backing: .buffered, defer: false)
    window.title = "MLXL3 streaming performance fixture"
    window.level = .floating
    window.contentView = host
    window.orderFrontRegardless()
    return window
  }

  private static func cpu() -> Double {
    var usage = rusage()
    getrusage(RUSAGE_SELF, &usage)
    return Double(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec)
      + Double(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1_000_000
  }

  private static func mainCPU() -> Double {
    var time = timespec()
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &time)
    return Double(time.tv_sec) + Double(time.tv_nsec) / 1_000_000_000
  }

  private static func hash(_ source: String) -> String {
    SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined()
  }

  @MainActor private static func wait(
    _ label: String, seconds: Double = 120,
    until condition: () -> Bool
  ) async throws {
    let end = ProcessInfo.processInfo.systemUptime + seconds
    while !condition() {
      guard ProcessInfo.processInfo.systemUptime < end else {
        throw PerformanceError.invalid("Timeout: " + label)
      }
      try await Task.sleep(for: .milliseconds(10))
    }
  }

  @MainActor static func main() {
    _ = NSApplication.shared
    NSApp.setActivationPolicy(.regular)
    Task { @MainActor in
      NSApp.activate()
      await run()
      NSApp.terminate(nil)
    }
    NSApp.run()
  }

  @MainActor private static func run() async {
    guard CommandLine.arguments.count >= 3 else { exit(2) }
    let output = URL(fileURLWithPath: CommandLine.arguments[2])
    var samples: [[String: Any]] = []
    func save(_ status: String, error: String? = nil) {
      var report: [String: Any] = [
        "status": status, "complete": status == "complete", "samples": samples,
      ]
      if let error { report["error"] = error }
      do {
        try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
          .write(to: output, options: .atomic)
      } catch {
        FileHandle.standardError.write(Data("Cannot save performance report: \(error)\n".utf8))
        exit(3)
      }
    }
    save("running")
    do {
      let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-streaming-" + UUID().uuidString)
      let suite = "io.mlxl3.streaming-check." + UUID().uuidString
      let prefs = UserDefaults(suiteName: suite)!
      defer {
        prefs.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: root)
      }
      let studio = StudioModel(conversationFileURL: root.appendingPathComponent("history.json"), isPreview: true, preferences: prefs)
      if CommandLine.arguments[1] == "--ui" {
        for size in [65_536, 262_144, 1_048_576] {
          for kind in ["prose", "code", "thinking"] {
            for cadence in size == 65_536 && kind == "prose" ? ["50ms", "slow", "burst"] : ["50ms"]
            {
              let line =
                kind == "code" ? "let item = \"é 👩‍💻\"\n" : "Une phrase avec **du texte** et é 👩‍💻.\n\n"
              let source =
                (kind == "code" ? "```swift\n" : "")
                + String(repeating: line, count: size / line.utf8.count)
              let message = ChatMessage(
                role: .assistant, content: kind == "thinking" ? "" : source,
                thinking: kind == "thinking" ? source : "", isStreaming: true)
              let host = window(message, studio: studio)
              defer { host.orderOut(nil) }
              try await Task.sleep(for: .milliseconds(400))
              for _ in 0..<6 {
                message.append(line, phase: kind == "thinking" ? "thinking" : "answer")
                try await Task.sleep(for: .milliseconds(50))
              }
              let start = ProcessInfo.processInfo.systemUptime
              guard host.occlusionState.contains(.visible) else {
                throw PerformanceError.invalid("Replay window is not visible")
              }
              let startCPU = cpu()
              let startMain = mainCPU()
              var delays: [Double] = []
              var peak = MLXL3Bridge().interfaceMemoryFootprintBytes() ?? 0
              for index in 1...24 {
                let offset =
                  cadence == "slow"
                  ? Double(index) / 15
                  : cadence == "burst"
                    ? Double((index - 1) / 3) * 0.15 + Double((index - 1) % 3 + 1) * 0.003
                    : Double(index) * 0.05
                let deadline = start + offset
                let remaining = deadline - ProcessInfo.processInfo.systemUptime
                if remaining > 0 { try await Task.sleep(for: .seconds(remaining)) }
                delays.append(max(0, ProcessInfo.processInfo.systemUptime - deadline))
                message.append(line, phase: kind == "thinking" ? "thinking" : "answer")
                peak = max(peak, MLXL3Bridge().interfaceMemoryFootprintBytes() ?? 0)
              }
              // Include the final asynchronous preparation and the entire last fade.
              try await Task.sleep(for: .milliseconds(180))
              let cpuSeconds = cpu() - startCPU
              let mainSeconds = mainCPU() - startMain
              let wall = ProcessInfo.processInfo.systemUptime - start
              guard host.occlusionState.contains(.visible) else {
                throw PerformanceError.invalid("Replay window became occluded")
              }
              message.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
              let actual = kind == "thinking" ? message.thinking : message.content
              let expected = source + String(repeating: line, count: 30)
              guard actual.utf8.elementsEqual(expected.utf8) else {
                throw PerformanceError.invalid("UI replay lost text")
              }
              samples.append([
                "kind": kind, "bytes": size, "cadence": cadence, "cpu_seconds": cpuSeconds,
                "main_cpu_seconds": mainSeconds, "wall_seconds": wall,
                "arrival_p95_seconds": delays.sorted()[22], "arrival_max_seconds": delays.max()!,
                "footprint_bytes": peak, "text_sha256": hash(actual), "appends": 24,
                "window_visible": true,
              ])
              save("running")
            }
          }
        }
      } else if CommandLine.arguments[1] == "--engine", CommandLine.arguments.count == 6 {
        let engine = CommandLine.arguments[3]
        let model = CommandLine.arguments[4]
        let head = CommandLine.arguments[5]
        setenv("MLXL3_EXECUTABLE", engine, 1)
        let bridge = MLXL3Bridge()
        let message = ChatMessage(
          role: .assistant,
          content: String(repeating: "A previous paragraph with **Markdown**.\n\n", count: 1_700),
          isStreaming: true)
        let host = window(message, studio: studio)
        defer { host.orderOut(nil) }
        var ready = false
        var completed = false
        var failure: String?
        var stats: GenerationStats?
        var text = ""
        bridge.onEvent = { event in
          switch event.type {
          case "ready": ready = true
          case "delta":
            text += event.text ?? ""
            message.append(event.text ?? "", phase: event.phase)
          case "complete":
            stats = event.stats
            completed = true
          case "error":
            failure = event.message ?? "Engine error"
            completed = true
          default: break
          }
        }
        bridge.onExit = { error in
          failure = error ?? "Unexpected engine exit"
          completed = true
          ready = true
        }
        do {
          try await bridge.start(model: model, contextLength: 4096)
          try await wait("ready") { ready }
          let prompts = [
            "Explique en français le fonctionnement d'un cache LRU, avec un exemple détaillé et plusieurs paragraphes.",
            "Écris une fonction Python merge_intervals, puis des tests, et explique sa complexité en détail.",
          ]
          for index in 0..<3 {
            if let failure { throw PerformanceError.invalid(failure) }
            completed = false
            stats = nil
            text = ""
            let start = ProcessInfo.processInfo.systemUptime
            guard host.occlusionState.contains(.visible) else {
              throw PerformanceError.invalid("Engine replay window is not visible")
            }
            let startCPU = cpu()
            let startMain = mainCPU()
            let request = GenerationRequest(
              requestID: UUID().uuidString, conversationID: UUID().uuidString,
              messages: [PromptMessage(role: "user", content: prompts[max(0, index - 1)])],
              maxTokens: index == 0 ? 48 : 128, temperature: 0, topK: 1, repetitionPenalty: 1,
              mtp: !head.isEmpty, mtpHeadPath: head)
            try bridge.generate(request)
            try await wait("generation") { completed }
            if let failure { throw PerformanceError.invalid(failure) }
            guard let stats, !text.isEmpty, stats.generatedTokens >= (index == 0 ? 16 : 64),
              stats.generatedTokens <= request.maxTokens, stats.decodeTps.isFinite,
              stats.decodeTps > 0,
              stats.ttftSeconds.isFinite, stats.ttftSeconds >= 0
            else { throw PerformanceError.invalid("Incomplete generation statistics") }
            try await Task.sleep(for: .milliseconds(180))
            guard host.occlusionState.contains(.visible) else {
              throw PerformanceError.invalid("Engine replay window became occluded")
            }
            if index > 0 {
              samples.append([
                "kind": "engine", "prompt": index - 1, "text_sha256": hash(text),
                "cpu_seconds": cpu() - startCPU, "main_cpu_seconds": mainCPU() - startMain,
                "wall_seconds": ProcessInfo.processInfo.systemUptime - start,
                "stats": try JSONSerialization.jsonObject(with: JSONEncoder().encode(stats)),
                "footprint_bytes": bridge.interfaceMemoryFootprintBytes() ?? 0,
                "window_visible": true,
              ])
              save("running")
            }
          }
          await bridge.stopAndJoin()
        } catch {
          await bridge.stopAndJoin()
          throw error
        }
      } else {
        throw PerformanceError.invalid("Unknown benchmark mode")
      }
      save("complete")
    } catch {
      save("failed", error: String(describing: error))
      FileHandle.standardError.write(Data("Performance fixture failed: \(error)\n".utf8))
      exit(1)
    }
  }
}
