// Compiled in the same file as MarkdownResponseView.swift to exercise its
// production parser/worker without exporting implementation details.
import AppKit
import SwiftUI

@main struct ResponseUICheck {
  private static func description(_ blocks: [MarkdownBlock]) -> [String] {
    blocks.map { block in
      let value: String
      switch block.kind {
      case .paragraph(let text): value = "paragraph:\(text)"
      case .heading(let level, let text): value = "heading:\(level):\(text)"
      case .unordered(let indent, let text): value = "unordered:\(indent):\(text)"
      case .ordered(let indent, let number, let text): value = "ordered:\(indent):\(number):\(text)"
      case .quote(let text): value = "quote:\(text)"
      case .code(let language, let text): value = "code:\(language ?? ""):\(text)"
      case .math(let text): value = "math:\(text)"
      case .table(let header, let alignments, let rows):
        value = "table:\(header):\(alignments):\(rows)"
      case .rule: value = "rule"
      }
      return "\(block.id):\(value)"
    }
  }

  @MainActor static func main() async throws {
    _ = NSApplication.shared
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    let answer = "# Été 👩‍💻\n\nFirst **paragraph**.\n\n```swift\nlet x = 73\n```\n\n$x^2$\n"
    let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
    pasteboard.setString("unchanged", forType: .string)
    precondition(!MessageClipboard.copy(message, to: pasteboard))
    precondition(pasteboard.string(forType: .string) == "unchanged", "Empty copy cleared clipboard")
    message.append("Private reasoning", phase: "thinking")
    message.append(answer, phase: "answer")
    message.startTool(id: "tool", serverName: "test", toolName: "lookup")
    message.finishTool(id: "tool", result: "Internal tool output", isError: false)
    message.append("\nFinal paragraph e\u{301}.\n", phase: "answer")
    let entire = answer + "\nFinal paragraph e\u{301}.\n"
    precondition(MessageClipboard.copy(message, to: pasteboard))
    precondition(
      pasteboard.string(forType: .string)?.utf8.elementsEqual(entire.utf8) == true,
      "Copy lost paragraphs/code/Unicode or included internal reasoning/tools")
    message.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
    precondition(MessageClipboard.copy(message, to: pasteboard))
    precondition(pasteboard.string(forType: .string)?.utf8.elementsEqual(entire.utf8) == true)
    let fallback = ChatMessage(role: .assistant, content: "", isStreaming: true)
    fallback.finish(stats: nil, fallbackAnswer: "Fallback", cacheContext: nil)
    precondition(MessageClipboard.copy(fallback, to: pasteboard))
    precondition(pasteboard.string(forType: .string) == "Fallback")
    let interrupted = ChatMessage(role: .assistant, content: "Partial", isStreaming: true)
    interrupted.fail("Cancelled")
    precondition(MessageClipboard.copy(interrupted, to: pasteboard))
    precondition(pasteboard.string(forType: .string) == "Partial")
    let restored = ChatMessage(snapshot: message.snapshot)!
    precondition(MessageClipboard.copy(restored, to: pasteboard))
    precondition(pasteboard.string(forType: .string)?.utf8.elementsEqual(entire.utf8) == true)
    let user = ChatMessage(role: .user, content: "User question 👋")
    precondition(MessageClipboard.copy(user, to: pasteboard))
    precondition(pasteboard.string(forType: .string) == "User question 👋")

    // Native control rendering and its production clipboard action use only a
    // private pasteboard. Headless NSHostingView has no native AX descendants.
    let button = NSHostingView(
      rootView: MessageCopyButton(message: message, pasteboard: pasteboard))
    button.frame = NSRect(x: 0, y: 0, width: 140, height: 40)
    button.layoutSubtreeIfNeeded()
    precondition(button.fittingSize.height > 0, "Copy button failed to render")
    precondition(MessageClipboard.copy(button.rootView.message, to: button.rootView.pasteboard))
    precondition(pasteboard.string(forType: .string)?.utf8.elementsEqual(entire.utf8) == true)

    let worker = MarkdownPreparation()
    var cases = 0
    let prose = String(repeating: "é **bold** 👩‍💻.\n\n", count: 1_200)
    let table = "| A | B |\n| --- | ---: |\n" + String(repeating: "| é | `x\\|y` |\n", count: 2_000)
    let documents = [
      "", answer, prose, table, prose + table,
      "````html\n" + String(repeating: "<div>é 👋</div>\n", count: 20_000),
      "~~~swift\nlet x = 73\n~~~\n\n## End\n",
      "\n$$\na^2+b^2\n$$\n\n> Quote\n\n- One\n1. Two\n---\n",
    ]
    for document in documents {
      for candidate in [
        document, document + "e", document + "e\u{301}", document + "\n```\nEnd.",
        String(document.prefix(7)), "Replacement", "",
      ] {
        let prepared = await worker.prepare(candidate)!
        let expectedChunks = StreamingTextChunker.chunks(candidate)
        precondition(
          prepared.map(\.chunk) == expectedChunks, "Worker changed chunk IDs/source/table state")
        for (actual, expected) in zip(prepared, expectedChunks) {
          precondition(
            description(actual.blocks) == description(MarkdownParser.parse(expected.source)),
            "Background preparation changed Markdown/fence/math/table rendering")
        }
        cases += 1
      }
    }
    let renderer = MarkdownRenderModel()
    func settle(_ expected: String) async throws {
      for _ in 0..<500 {
        if renderer.chunks.map(\.chunk.source).joined() == expected { return }
        try await Task.sleep(for: .milliseconds(10))
      }
      preconditionFailure("Preparation did not reach newest source")
    }
    renderer.request(documents[5])
    await Task.yield()
    renderer.request("Latest answer **73**")
    try await settle("Latest answer **73**")
    try await Task.sleep(for: .milliseconds(50))
    precondition(
      renderer.chunks.map(\.chunk.source).joined() == "Latest answer **73**",
      "Stale preparation replaced the latest answer")
    renderer.request("Cancelled replacement")
    renderer.cancel()
    try await Task.sleep(for: .milliseconds(50))
    precondition(
      renderer.chunks.map(\.chunk.source).joined() == "Latest answer **73**",
      "Cancelled/disappeared view published stale content")
    renderer.request("")
    try await settle("")
    var streamed = "```html\n"
    for _ in 0..<100 {
      streamed += "<div>é 👋</div>\n"
      renderer.request(streamed)
    }
    try await settle(streamed)
    renderer.cancel()
    renderer.request("Reappeared")
    try await settle("Reappeared")

    // Real bridge -> StudioModel -> full rendered reply -> copy -> history.
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-response-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let suite = "io.mlxl3.response-check." + UUID().uuidString
    let prefs = UserDefaults(suiteName: suite)!
    defer { prefs.removePersistentDomain(forName: suite) }
    setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
    let history = root.appendingPathComponent("conversations.json")
    let studio = StudioModel(conversationFileURL: history, preferences: prefs)
    defer { studio.ejectModel() }
    studio.models = [
      LocalModel(
        name: "render-stress", path: root.path, modelType: "fixture", format: "EXL3", bits: 3,
        sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
    ]
    studio.selectModel("render-stress")
    for _ in 0..<250 where !studio.engineState.isReady {
      try await Task.sleep(for: .milliseconds(20))
    }
    precondition(studio.engineState.isReady)
    studio.draft = "Generate a long file"
    studio.send()
    let host = NSHostingView(rootView: MessagesView(messages: studio.currentConversation!.messages))
    host.frame = NSRect(x: 0, y: 0, width: 900, height: 700)
    for _ in 0..<500 where studio.isGenerating {
      host.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(10))
    }
    precondition(!studio.isGenerating, "Response journey did not complete")
    let reply = studio.currentConversation!.messages.last!
    let expected =
      "# Result\n\n```html\n" + String(repeating: "<div>é 👋</div>\n", count: 20_000)
      + "```\n\nFinished: 73.\n"
    precondition(
      reply.content.utf8.elementsEqual(expected.utf8), "Bridge dropped/reordered fragments")
    precondition(MessageClipboard.copy(reply, to: pasteboard))
    precondition(pasteboard.string(forType: .string)?.utf8.elementsEqual(expected.utf8) == true)
    precondition(studio.persistNow())
    let reopened = StudioModel(conversationFileURL: history, isPreview: true, preferences: prefs)
    precondition(
      reopened.currentConversation?.messages.last?.content.utf8.elementsEqual(expected.utf8) == true
    )
    host.layoutSubtreeIfNeeded()
    print(
      "Response UI checks passed: complete copy, native control/clipboard action, \(cases) parser/chunk parity cases, stale/cancelled updates, long streaming bridge/render/history"
    )
  }
}
