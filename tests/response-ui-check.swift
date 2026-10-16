// Compiled in the same file as MarkdownResponseView.swift to exercise its
// production parser/worker without exporting implementation details.
import AppKit
import SwiftUI

private struct ReferenceFadeRenderer: TextRenderer {
  let tails: [(Int, Double)]

  func draw(layout: Text.Layout, in context: inout GraphicsContext) {
    let runs = layout.flatMap { $0 }
    var remaining = runs.reduce(0) { $0 + $1.count }
    for run in runs {
      if remaining - run.count >= tails.reduce(0, { $0 + $1.0 }) || tails.allSatisfy({ $0.1 == 1 })
      {
        context.draw(run)
        remaining -= run.count
        continue
      }
      for glyph in run {
        var opacity = 1.0
        var boundary = 0
        for (count, value) in tails.reversed() {
          boundary += count
          if remaining <= boundary {
            opacity = value
            break
          }
        }
        var copy = context
        copy.opacity *= opacity
        copy.draw(glyph)
        remaining -= 1
      }
    }
  }
}

@main struct ResponseUICheck {
  @MainActor private static func checkScroll() async throws {
    func require(_ condition: Bool, _ message: String) {
      if !condition {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        preconditionFailure(message)
      }
    }
    let message = ChatMessage(
      role: .assistant,
      content: String(
        repeating: "A paragraph with enough words to wrap onto another line.\n\n", count: 40),
      isStreaming: true)
    let host = NSHostingView(rootView: MessagesView(messages: [message]))
    let window = NSWindow(
      contentRect: NSRect(x: 80, y: 80, width: 700, height: 400),
      styleMask: [.titled], backing: .buffered, defer: false)
    window.contentView = host
    window.makeKeyAndOrderFront(nil)
    defer { window.orderOut(nil) }
    func scrollView(_ view: NSView) -> NSScrollView? {
      if let scroll = view as? NSScrollView { return scroll }
      return view.subviews.lazy.compactMap { scrollView($0) }.first
    }
    try await Task.sleep(for: .milliseconds(500))
    guard let scroll = scrollView(host), let document = scroll.documentView else {
      FileHandle.standardError.write(Data("Transcript did not create a native scroll view\n".utf8))
      preconditionFailure("Transcript did not create a native scroll view")
    }
    func offset() -> CGFloat { scroll.contentView.bounds.minY }
    func distance() -> CGFloat {
      document.isFlipped
        ? document.bounds.maxY - scroll.contentView.bounds.maxY : scroll.contentView.bounds.minY
    }
    for _ in 0..<200 {
      host.layoutSubtreeIfNeeded()
      if document.bounds.height > scroll.contentView.bounds.height + 100 && abs(distance()) < 30 {
        break
      }
      try await Task.sleep(for: .milliseconds(10))
    }
    // The bottom marker precedes the transcript's 26 pt bottom padding.
    require(abs(distance()) < 30, "Initial transcript is not at the bottom: \(distance())")
    let before = offset()
    message.append("\n\nA new paragraph.\n\nAnother new paragraph.\n\n", phase: "answer")
    var positions: [CGFloat] = []
    for _ in 0..<35 {
      try await Task.sleep(for: .milliseconds(10))
      host.layoutSubtreeIfNeeded()
      positions.append(offset())
    }
    let moving = Set(positions.map { Int(($0 * 10).rounded()) })
    require(
      abs(offset() - before) > 10
        && (NSWorkspace.shared.accessibilityDisplayShouldReduceMotion || moving.count > 2),
      "Scroll jumped instead of interpolating: delta=\(offset() - before) positions=\(positions)")
    require(abs(distance()) < 30, "Scroll missed the final prepared height: \(distance())")

    var follow = StreamingScrollState()
    require(follow.shouldFollow, "Initial follow state")
    follow.phaseChanged(.tracking)
    require(!follow.shouldFollow, "Tracking must suspend follow before a drag")
    follow.phaseChanged(.interacting)
    follow.proximityChanged(false)
    follow.phaseChanged(.decelerating)
    require(!follow.shouldFollow, "Deceleration must suspend follow")
    follow.phaseChanged(.idle)
    follow.proximityChanged(false)
    require(!follow.shouldFollow, "Reading older messages must persist after the gesture")
    follow.proximityChanged(true)
    follow.phaseChanged(.animating)
    require(follow.shouldFollow, "Returning to bottom must resume follow")
    print(
      "Scroll checks passed: native initial/final anchor and interpolation; production state transitions for tracking/drag/deceleration/reading/resume"
    )
  }

  @MainActor private static func checkFade() {
    var state = TokenFadeState()
    state.update("abc", at: 0, animated: true)
    state.update("abcdef", at: 0.05, animated: true)
    precondition(state.batches.map(\.count) == [3, 3])
    precondition(
      state.batches.map(\.started) == [0, 0.05], "New fragment restarted the previous fade")
    precondition(abs(state.batches[0].opacity(at: 0.09) - 0.7875) < 1e-12)
    precondition(state.batches[0].opacity(at: 0.12) == 1)
    precondition(state.batches[1].opacity(at: 0.12) < 1)
    precondition(state.batches[1].opacity(at: 0.18) == 1)
    let late = TokenFadeState.Batch(count: 1, started: 12_345_678)
    precondition(
      late.opacity(at: late.started + TokenFadeState.duration) == 1,
      "Clock rounding left the finished renderer on its slow path")
    state.update("abcdef!", at: 0.2, animated: true)
    precondition(state.batches.map(\.count) == [1], "Expired fragments were kept")
    state.update("replacement", at: 0.21, animated: true)
    precondition(state.batches.isEmpty, "Replacement animated unrelated old text")
    state.update("", at: 0.22, animated: true)
    precondition(state.batches.isEmpty)
    state.update("no motion", at: 0.23, animated: false)
    precondition(state.batches.isEmpty)
    state.update("no motion!", at: 0.24, animated: true)
    precondition(state.batches.map(\.count) == [1])

    var cases = 0
    for first in [0, 1, 63, 64, 65, 128] {
      for second in 0...130 {
        var bounded = TokenFadeState()
        bounded.update(String(repeating: "a", count: first), at: 0, animated: true)
        bounded.update(String(repeating: "a", count: first + second), at: 0.01, animated: true)
        precondition(bounded.batches.reduce(0) { $0 + $1.count } == min(64, first + second))
        precondition(bounded.batches.allSatisfy { $0.count > 0 })
        for batch in bounded.batches {
          let samples = (-1...20).map { batch.opacity(at: Double($0) / 100) }
          precondition(samples.allSatisfy { $0 >= 0.15 && $0 <= 1 && $0.isFinite })
          precondition(zip(samples, samples.dropFirst()).allSatisfy { $0 <= $1 })
        }
        cases += 1
      }
    }
    var unicode = TokenFadeState()
    var source = ""
    for (index, fragment) in ["e", "\u{301}", "👩", "‍", "💻", "🇫", "🇷", "\r", "\n"].enumerated() {
      source += fragment
      unicode.update(source, at: Double(index) / 1_000, animated: true)
      precondition(unicode.batches.reduce(0) { $0 + $1.count } <= 64)
    }

    func pixels<V: View>(_ view: V) -> (Int, Int, [UInt8]) {
      let renderer = ImageRenderer(
        content: view.font(.system(size: 18)).foregroundStyle(.white)
          .frame(width: 320, alignment: .leading).frame(minHeight: 1)
          .fixedSize(horizontal: false, vertical: true)
          .background(.black))
      renderer.scale = 1
      guard let image = renderer.cgImage else {
        FileHandle.standardError.write(Data("ImageRenderer returned no image\n".utf8))
        preconditionFailure("Missing native drawing")
      }
      var bytes = [UInt8](repeating: 0, count: image.width * image.height * 4)
      let context = CGContext(
        data: &bytes, width: image.width, height: image.height,
        bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
      context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
      return (image.width, image.height, bytes)
    }
    let texts: [Text] = [
      Text(
        "A long paragraph with words and a newly arriving suffix. "
          + String(repeating: "abc fi 👩‍💻 e\u{301} ", count: 12)),
      Text("Italic ffi AV e\u{301} 👩‍💻 العربية עברית 中文").italic(),
      Text(
        "\(Text("let value = ").foregroundColor(.orange))\(Text("\"hello 👋\"\n// suffix").foregroundColor(.green))"
      ),
      Text(
        try! AttributedString(
          markdown: "**Bold** and *italic* and `code` with [link](https://example.com).")),
      Text(""), Text("x"), Text("👩‍💻"),
    ]
    var images = 0
    func compare(_ actual: (Int, Int, [UInt8]), _ expected: (Int, Int, [UInt8]), label: String) {
      precondition(actual.0 == expected.0 && actual.1 == expected.1)
      let differences = zip(actual.2, expected.2).map { abs(Int($0) - Int($1)) }
      let changed = differences.filter { $0 > 2 }.count
      if (differences.max() ?? 0) > 32 || Double(changed) > Double(differences.count) * 0.005 {
        FileHandle.standardError.write(
          Data(
            "Drawing mismatch \(label) max=\(differences.max() ?? 0) changed=\(changed)/\(differences.count)\n"
              .utf8))
        preconditionFailure("Grouped drawing changed opacity or glyph placement")
      }
    }
    for text in texts {
      for opacity in [0.15, 0.5, 1.0] {
        let time = (opacity - 0.15) / 0.85 * TokenFadeState.duration
        let actual = pixels(
          text.textRenderer(
            TokenFadeRenderer(
              batches: [.init(count: 10_000, started: 0)], time: time)))
        compare(actual, pixels(text.opacity(opacity)), label: "native opacity=\(opacity)")
        images += 1
      }
    }
    for text in [
      Text(String(repeating: "ABCDE 12345\n", count: 12)).monospaced(), Text(""), Text("x"),
    ] {
      for count in [1, 3, 63, 64] {
        for opacity in [0.15, 0.5, 1.0] {
          let batch = TokenFadeState.Batch(count: count, started: 0)
          let time = (opacity - 0.15) / 0.85 * TokenFadeState.duration
          let actual = pixels(text.textRenderer(TokenFadeRenderer(batches: [batch], time: time)))
          let expected = pixels(text.textRenderer(ReferenceFadeRenderer(tails: [(count, opacity)])))
          compare(actual, expected, label: "suffix count=\(count) opacity=\(opacity)")
          images += 1
        }
      }
    }
    for length in [0, 1, 2, 5, 8, 80] {
      let text = Text(String(repeating: "X", count: length)).monospaced()
      let actual = pixels(
        text.textRenderer(
          TokenFadeRenderer(
            batches: [.init(count: 3, started: 0), .init(count: 2, started: 0.08)], time: 0.09)))
      let expected = pixels(
        text.textRenderer(
          ReferenceFadeRenderer(
            tails: [(3, 0.7875), (2, 0.22083333333333333)])))
      compare(actual, expected, label: "overlapping fragment ages length=\(length)")
      images += 1
    }
    print(
      "Token fade checks passed: continuous ages, replacement/motion, \(cases) bounded append cases, Unicode and \(images) native drawing comparisons"
    )
  }

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
    NSApp.setActivationPolicy(.regular)
    NSApp.activate()
    checkFade()
    try await checkScroll()
    // These exact reply formulas used to fall back to raw $$ text.
    let formulas = [
      #"\mathbb{E}[T_n] = \sum_{i=0}^{n-1}\mathbb{E}[T_{i+1}-T_i] = e\,n\sum_{i=0}^{n-1}\frac{1}{n-i} = e\,n\left(1+\frac12+\frac13+\dots+\frac1n\right)."#,
      #"\boxed{\mathbb{E}[T_n] \approx e\,n\,\ln n}."#,
      #"\boxed{}"#, #"\boxed{x}"#, #"\boxed{\frac{1}{n}}"#,
      #"\boxed{\boxed{x+1}}"#, #"\boxed{x}_n^2"#,
      #"\frac{\boxed{x}}{1+\dots+n}"#, #"\left(\boxed{x}\right)"#,
    ]
    for formula in formulas {
      var parseError: NSError?
      let list = MTMathListBuilder.build(fromString: formula, error: &parseError)
      precondition(list != nil && parseError == nil, "Reply LaTeX rejected: \(formula)")
      let roundTrip = MTMathListBuilder.mathListToString(list)
      precondition(MTMathListBuilder.build(fromString: roundTrip) != nil)
      let copied = MTMathList(atoms: list!.atoms.map { $0.copy() })
      precondition(MTMathListBuilder.mathListToString(copied) == roundTrip,
                   "Copy dropped the box or introduced invisible delimiters")
      let source = "Before.\n\n$$\n" + formula + "\n$$\n\nAfter."
      precondition(description(MarkdownParser.parse(source)) ==
        ["0:paragraph:Before.", "1:math:" + formula, "2:paragraph:After."])
      for mode in [MTMathUILabelMode.text, .display] {
        let image = LatexImageCache.image(latex: formula, fontSize: 18, mode: mode)
        precondition(image != nil, "Production math image fell back to text")
        precondition(image!.size.width.isFinite && image!.size.height.isFinite &&
                     image!.size.width > 0 && image!.size.height > 0)
        let bitmap = NSBitmapImageRep(data: image!.tiffRepresentation!)!
        precondition(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0)
        precondition((0..<bitmap.pixelsHigh).contains { y in
          (0..<bitmap.pixelsWide).contains { x in
            bitmap.colorAt(x: x, y: y)!.alphaComponent > 0
          }
        }, "Math image was blank")
      }
    }
    for formula in formulas.prefix(2) {
      for (open, close) in [("$", "$"), (#"\("#, #"\)"#), ("$$", "$$"), (#"\["#, #"\]"#)] {
        let pieces = InlineMathParser.parse("Before " + open + formula + close + " after")
        precondition(pieces.count == 3)
        guard case .math(let latex, let raw, _) = pieces[1] else {
          preconditionFailure("Inline formula lost its math renderer")
        }
        precondition(latex == formula && raw == open + formula + close)
      }
    }
    let plain = LatexImageCache.image(latex: "x", fontSize: 18, mode: .display)!
    let boxed = LatexImageCache.image(latex: #"\boxed{x}"#, fontSize: 18, mode: .display)!
    precondition(boxed.size.width > plain.size.width && boxed.size.height > plain.size.height,
                 "Box command was stripped instead of framed")
    let emptyBox = LatexImageCache.image(latex: #"\boxed{}"#, fontSize: 18, mode: .display)!
    let frame = NSBitmapImageRep(data: emptyBox.tiffRepresentation!)!
    let w = frame.pixelsWide, h = frame.pixelsHigh
    precondition(frame.colorAt(x: w / 2, y: h / 2)!.alphaComponent == 0,
                 "Frame was filled over its content")
    precondition((0..<min(2, h)).contains { y in
      frame.colorAt(x: w / 2, y: y)!.alphaComponent > 0
    } && (max(0, h - 2)..<h).contains { y in
      frame.colorAt(x: w / 2, y: y)!.alphaComponent > 0
    }, "Top or bottom frame stroke missing")
    precondition((0..<min(2, w)).contains { x in
      frame.colorAt(x: x, y: h / 2)!.alphaComponent > 0
    } && (max(0, w - 2)..<w).contains { x in
      frame.colorAt(x: x, y: h / 2)!.alphaComponent > 0
    }, "Left or right frame stroke missing")
    for invalid in [#"\boxed{x"#, #"\notASupportedCommand{x}"#,
                    String(repeating: "x", count: 4097)] {
      precondition(LatexImageCache.image(latex: invalid, fontSize: 18, mode: .display) == nil)
    }
    let codeMath = "```latex\n" + formulas[1] + "\n```"
    precondition(description(MarkdownParser.parse(codeMath)) == ["0:code:latex:" + formulas[1]])
    print("LaTeX rendering checks passed: 9 formulas, 2 styles, parse/copy/roundtrip, actual pixels/frame, invalid input")
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    let answer = "# Été 👩‍💻\n\nFirst **paragraph**.\n\n```swift\nlet x = 73\n```\n\n$x^2$\n"
      + "\n$$\n" + formulas[0] + "\n$$\n\n$$\n" + formulas[1] + "\n$$\n"
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
      "Before.\n\n$$\n" + formulas[0] + "\n$$\n\n$$\n" + formulas[1] + "\n$$",
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
