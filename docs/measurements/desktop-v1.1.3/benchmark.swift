import AppKit
import SwiftUI

@main struct ResponseUIBenchmark {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        for bytes in [65_536, 262_144, 1_048_576] {
            for code in [false, true] {
                let line = code ? "<div class=\"item\">Hello 👋</div>\n" : "A sentence with **words** and punctuation.\n\n"
                let initial = (code ? "```html\n" : "") + String(repeating: line, count: bytes / line.utf8.count)
                let message = ChatMessage(role: .assistant, content: initial, isStreaming: true)
                let host = NSHostingView(rootView: MessagesView(messages: [message]))
                host.frame = NSRect(x: 0, y: 0, width: 900, height: 700)
                let window = NSWindow(contentRect: NSRect(x: -20_000, y: 0, width: 900, height: 700), styleMask: [], backing: .buffered, defer: false)
                window.contentView = host
                window.orderFront(nil)
                host.layoutSubtreeIfNeeded()
                try await Task.sleep(for: .milliseconds(200))
                var samples: [Double] = []
                var delays: [Double] = []
                for _ in 0..<22 {
                    let start = Date()
                    message.append(line, phase: "answer")
                    host.layoutSubtreeIfNeeded()
                    samples.append(Date().timeIntervalSince(start) * 1_000)
                    let sleepStart = Date()
                    try await Task.sleep(for: .milliseconds(50))
                    delays.append(max(0, Date().timeIntervalSince(sleepStart) * 1_000 - 50))
                }
                let measured = Array(samples.dropFirst(2)).sorted()
                let drift = Array(delays.dropFirst(2)).sorted()
                print(String(format: "AppKit %d bytes code=%d: median %.3f ms/update, p95 %.3f, max %.3f; timer drift p95 %.3f ms", bytes, code ? 1 : 0, measured[10], measured[18], measured[19], drift[18]))
                message.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
                host.layoutSubtreeIfNeeded()
                window.orderOut(nil)
            }
        }
    }
}
