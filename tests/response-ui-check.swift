// Compiled in the same file as MarkdownResponseView.swift to exercise its
// production parser/worker without exporting implementation details.
import AppKit
import CoreML
import SwiftUI
import Vision
import WebKit

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

extension MarkdownResponseView {
  @MainActor static func verifyHTMLPreviewLifetime() {
    let prefix =
      "```html\n<button>Example</button>\n```\n\n" + String(repeating: "x\n\n", count: 6000)
    precondition(!MarkdownResponseView(prefix, streaming: true).htmlPreviewEnabled)
    precondition(!MarkdownResponseView(prefix, htmlPreviewEnabled: false).htmlPreviewEnabled)
    precondition(MarkdownResponseView(prefix).htmlPreviewEnabled)
  }
}

extension CodeBlockView {
  @MainActor fileprivate static func previewForLifetimeCheck(_ source: String) -> Self {
    var view = Self(language: "html", source: source, streaming: false, htmlPreviewEnabled: true)
    view._showingPreview = State(initialValue: true)
    return view
  }
}

@main struct ResponseUICheck {
  @MainActor private static func checkInlinePreviewLifetime() async throws {
    func report(_ message: String) {
      FileHandle.standardError.write(Data(("Inline lifetime: " + message + "\n").utf8))
    }
    func require(_ condition: Bool, _ message: String) {
      report((condition ? "PASS " : "FAIL ") + message)
      precondition(condition, message)
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-inline-lifetime-\(UUID())")
    let suite = "mlxl3.inline-lifetime.\(UUID())"
    let prefs = UserDefaults(suiteName: suite)!
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer {
      prefs.removePersistentDomain(forName: suite)
      try? FileManager.default.removeItem(at: root)
    }
    let studio = StudioModel(
      conversationFileURL: root.appendingPathComponent("history.json"), isPreview: true,
      preferences: prefs)
    let source = """
      <!doctype html><html><body><p>Disposable lifetime check</p>
      <script>window.mlxl3Ticks = 0; setInterval(() => { window.mlxl3Ticks += 1; }, 25);</script>
      </body></html>
      """
    let host = NSHostingView(
      rootView: CodeBlockView.previewForLifetimeCheck(source).environmentObject(studio)
        .padding(24).background(StudioTheme.canvas).preferredColorScheme(.dark))
    let window = NSWindow(
      contentRect: NSRect(x: 40, y: 40, width: 800, height: 560), styleMask: [.titled],
      backing: .buffered, defer: false)
    window.contentView = host
    window.orderFront(nil)
    defer {
      window.contentView = nil
      window.orderOut(nil)
      studio.studioDocument = nil
    }
    func webViews(_ view: NSView) -> [WKWebView] {
      (view as? WKWebView).map { [$0] } ?? view.subviews.flatMap(webViews)
    }
    var loaded: WKWebView?
    var loadedTicks = 0
    var lastJavaScriptError: String?
    for _ in 0..<80 {
      host.layoutSubtreeIfNeeded()
      if let candidate = webViews(host).first, !candidate.isLoading {
        do {
          if let ticks = try await candidate.evaluateJavaScript("window.mlxl3Ticks") as? NSNumber,
            ticks.intValue > 0
          {
            loaded = candidate
            loadedTicks = ticks.intValue
            break
          }
        } catch { lastJavaScriptError = error.localizedDescription }
      }
      try await Task.sleep(for: .milliseconds(100))
    }
    if loaded == nil {
      let actual = webViews(host)
      report(
        "load timeout: count=\(actual.count), views=\(actual.map { "url=\($0.url?.absoluteString ?? "nil") loading=\($0.isLoading)" }), lastJavaScriptError=\(lastJavaScriptError ?? "none")"
      )
    } else {
      report(
        "actual timer loaded: ticks=\(loadedTicks), url=\(loaded!.url?.absoluteString ?? "nil")")
    }
    require(loaded != nil, "The real inline preview did not run the disposable CPU timer")
    let oldView = loaded!
    studio.studioDocument = StudioDocument(workspace: root)
    for _ in 0..<40 {
      host.layoutSubtreeIfNeeded()
      if webViews(host).isEmpty { break }
      try await Task.sleep(for: .milliseconds(100))
    }
    report(
      "after Agent Studio: native WebKit count=\(webViews(host).count), oldURL=\(oldView.url?.absoluteString ?? "nil")"
    )
    require(webViews(host).isEmpty, "Inline preview kept running behind Agent Studio")
    var cleared = false
    for _ in 0..<40 {
      if let kind = (try? await oldView.evaluateJavaScript("typeof window.mlxl3Ticks")) as? String,
        kind == "undefined"
      {
        cleared = true
        break
      }
      try await Task.sleep(for: .milliseconds(100))
    }
    report(
      "after dismantle: oldURL=\(oldView.url?.absoluteString ?? "nil"), loading=\(oldView.isLoading)"
    )
    require(cleared, "Removed inline preview retained its executable page")
    try await Task.sleep(for: .milliseconds(150))
    let kind = try await oldView.evaluateJavaScript("typeof window.mlxl3Ticks") as? String
    report("final timer type=\(kind ?? "nil")")
    require(kind == "undefined", "Removed preview timer restarted")
    print(
      "Inline preview lifetime passed: actual CPU timer starts, Agent Studio removes WebKit and clears its executable page"
    )
  }
  @MainActor private static func checkHTMLControls() async throws {
    MarkdownResponseView.verifyHTMLPreviewLifetime()
    func require(_ condition: Bool, _ message: String) {
      if !condition {
        FileHandle.standardError.write(Data((message + "\n").utf8))
        preconditionFailure(message)
      }
    }
    func changedHeaderPixels(_ a: NSBitmapImageRep, _ b: NSBitmapImageRep) -> Int {
      require(
        a.pixelsWide == b.pixelsWide && a.pixelsHigh == b.pixelsHigh,
        "HTML header comparison changed scale")
      var changed = 0
      // Only the mode-switch area; exclude Writing/Copy and the native code body.
      for y in (75 * a.pixelsHigh / 600)..<(108 * a.pixelsHigh / 600) {
        for x in (535 * a.pixelsWide / 800)..<(670 * a.pixelsWide / 800) {
          guard let first = a.colorAt(x: x, y: y)?.usingColorSpace(.sRGB),
            let second = b.colorAt(x: x, y: y)?.usingColorSpace(.sRGB)
          else { preconditionFailure("Missing header pixels") }
          if abs(first.redComponent - second.redComponent)
            + abs(first.greenComponent - second.greenComponent)
            + abs(first.blueComponent - second.blueComponent) > 0.08
          {
            changed += 1
          }
        }
      }
      return changed
    }
    func nativeViews(_ view: NSView) -> [NSView] {
      [view] + view.subviews.flatMap(nativeViews)
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-html-ui-\(UUID())")
    let suite = "mlxl3.html-ui.\(UUID())"
    let prefs = UserDefaults(suiteName: suite)!
    defer {
      prefs.removePersistentDomain(forName: suite)
      try? FileManager.default.removeItem(at: root)
    }
    let studio = StudioModel(
      conversationFileURL: root.appendingPathComponent("history.json"), isPreview: true,
      preferences: prefs)
    for mode in [StudioMode.chat, .agent] {
      studio.selectMode(mode)
      let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
      message.append(
        "```html\n<button>Example</button>\n```\n\n"
          + String(repeating: "Later content.\n\n", count: 2), phase: "answer")
      let host = NSHostingView(
        rootView: MessagesView(messages: [message]).environmentObject(studio).background(
          StudioTheme.canvas
        ).preferredColorScheme(.dark))
      let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 800, height: 600), styleMask: [.titled],
        backing: .buffered, defer: false)
      window.contentView = host
      window.orderFront(nil)
      defer {
        window.contentView = nil
        window.orderOut(nil)
      }
      func scrollView(_ view: NSView) -> NSScrollView? {
        if let scroll = view as? NSScrollView { return scroll }
        return view.subviews.lazy.compactMap { scrollView($0) }.first
      }
      func snapshot(_ phase: String) async throws -> NSBitmapImageRep {
        for _ in 0..<3 {
          if let scroll = scrollView(host), let document = scroll.documentView {
            scroll.contentView.scroll(
              to: NSPoint(
                x: 0,
                y: document.isFlipped
                  ? 0 : max(0, document.bounds.height - scroll.contentView.bounds.height)))
            scroll.reflectScrolledClipView(scroll.contentView)
          }
          host.layoutSubtreeIfNeeded()
          try await Task.sleep(for: .milliseconds(50))
        }
        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
          preconditionFailure("HTML transcript did not render")
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        require(
          !nativeViews(host).contains { $0 is WKWebView },
          "HTML preview ran without user action in \(mode)/\(phase)")
        if let path = ProcessInfo.processInfo.environment["MLXL3_HTML_UI_SNAPSHOTS"] {
          let folder = URL(fileURLWithPath: path)
          try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
          try bitmap.representation(using: .png, properties: [:])!.write(
            to: folder.appendingPathComponent("html-\(mode.rawValue)-\(phase).png"))
        }
        return bitmap
      }
      try await Task.sleep(for: .milliseconds(400))
      let streaming = try await snapshot("streaming")
      message.processing("Reading MCP results")
      let processing = try await snapshot("tool")
      require(
        changedHeaderPixels(streaming, processing) < 15,
        "HTML Preview control appeared during a subsequent tool round in \(mode)")
      message.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
      let complete = try await snapshot("complete")
      require(
        changedHeaderPixels(processing, complete) > 30,
        "Completed HTML has no visible Code/Preview switch in \(mode)")
      message.fail("Interrupted")
      let failed = try await snapshot("interrupted")
      require(
        changedHeaderPixels(streaming, failed) < 15,
        "Interrupted HTML kept an executable Preview control in \(mode)")
    }
    let source = "const café = 'é 👩‍💻 e\u{301}';\n// 日本語 <tag> & literal\nconsole.log(café);\n"
    let codeHost = NSHostingView(
      rootView: CodeBlockView(language: "javascript", source: source, streaming: false)
        .environmentObject(studio).padding(24).background(StudioTheme.canvas).preferredColorScheme(
          .dark))
    let codeWindow = NSWindow(
      contentRect: NSRect(x: 40, y: 40, width: 800, height: 360), styleMask: [.titled],
      backing: .buffered, defer: false)
    codeWindow.contentView = codeHost
    codeWindow.orderFront(nil)
    defer {
      codeWindow.contentView = nil
      codeWindow.orderOut(nil)
    }
    var editor: NSTextView?
    for _ in 0..<40 {
      codeHost.layoutSubtreeIfNeeded()
      editor = nativeViews(codeHost).compactMap { $0 as? NSTextView }.first {
        $0.accessibilityIdentifier() == "studio.code-editor"
      }
      if editor != nil { break }
      try await Task.sleep(for: .milliseconds(50))
    }
    require(editor != nil, "Completed code has no native source viewer")
    require(
      editor!.string.utf8.elementsEqual(source.utf8), "Code viewer changed Unicode or line breaks")
    require(
      !editor!.isEditable && editor!.isSelectable, "Generated code is not read-only/selectable")
    let scroll = editor!.enclosingScrollView!
    require(
      scroll.rulersVisible && scroll.hasVerticalRuler && scroll.verticalRulerView != nil,
      "Completed code has no line numbers")
    require(
      scroll.clipsToBounds && scroll.layer?.masksToBounds == true,
      "Code gutter can bleed into the header")
    require(!nativeViews(codeHost).contains { $0 is WKWebView }, "Code-only block created WebKit")
    if let path = ProcessInfo.processInfo.environment["MLXL3_HTML_UI_SNAPSHOTS"] {
      let folder = URL(fileURLWithPath: path)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      let bitmap = codeHost.bitmapImageRepForCachingDisplay(in: codeHost.bounds)!
      codeHost.cacheDisplay(in: codeHost.bounds, to: bitmap)
      try bitmap.representation(using: .png, properties: [:])!.write(
        to: folder.appendingPathComponent("code-inline.png"))
    }
    print(
      "HTML/code controls passed: Chat/Agent header pixels and source-level lifecycle gates; exact Unicode read-only native source, clipped line numbers and no automatic WebKit"
    )
  }
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
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-scroll-" + UUID().uuidString)
    let suite = "io.mlxl3.scroll-check." + UUID().uuidString
    let prefs = UserDefaults(suiteName: suite)!
    defer {
      prefs.removePersistentDomain(forName: suite)
      try? FileManager.default.removeItem(at: root)
    }
    let studio = StudioModel(
      conversationFileURL: root.appendingPathComponent("history.json"), isPreview: true,
      preferences: prefs)
    let host = NSHostingView(
      rootView: MessagesView(messages: [message]).environmentObject(studio).background(
        StudioTheme.canvas
      ).preferredColorScheme(.dark))
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

    var visibleFrames = 0
    func requireVisible(_ label: String) throws {
      host.layoutSubtreeIfNeeded()
      guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
        preconditionFailure("No streaming viewport bitmap")
      }
      host.cacheDisplay(in: host.bounds, to: bitmap)
      var foreground = 0
      for y in stride(from: bitmap.pixelsHigh / 5, to: bitmap.pixelsHigh * 3 / 4, by: 3) {
        for x in stride(from: bitmap.pixelsWide / 8, to: bitmap.pixelsWide * 7 / 8, by: 3) {
          if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.deviceRGB),
            color.redComponent + color.greenComponent + color.blueComponent > 1.8
          {
            foreground += 1
          }
        }
      }
      if let folder = ProcessInfo.processInfo.environment["MLXL3_STREAM_UI_SNAPSHOTS"],
        foreground < 40 || label == "complete"
      {
        let directory = URL(fileURLWithPath: folder)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bitmap.representation(using: .png, properties: [:])!.write(
          to: directory.appendingPathComponent("stream-\(label).png"))
      }
      require(
        foreground >= 40,
        "Streamed text disappeared at \(label): pixels=\(foreground), bottom=\(distance()), height=\(document.bounds.height)"
      )
      visibleFrames += 1
    }
    for index in 0..<40 {
      let text =
        "Paragraph \(index): words remain visible while the response grows. **Bold** and `inline code` must stay painted.\n\n"
      message.append(String(repeating: text, count: 3), phase: "answer")
      try await Task.sleep(for: .milliseconds(150))
      try requireVisible("prose-\(index)")
    }
    message.append("\n```javascript\n", phase: "answer")
    for index in 0..<40 {
      message.append(
        String(repeating: "const visible\(index) = 'paint this growing source';\n", count: 8),
        phase: "answer")
      try await Task.sleep(for: .milliseconds(150))
      try requireVisible("code-\(index)")
    }
    message.append("```\n\nThe answer is complete and remains visible.\n\n", phase: "answer")
    message.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
    try await Task.sleep(for: .milliseconds(500))
    try requireVisible("complete")
    print(
      "Streaming visibility passed: \(visibleFrames) native viewport frames across long prose, growing code and completion"
    )

    var follow = StreamingScrollState()
    require(follow.shouldFollow, "Initial follow state")
    follow.proximityChanged(false)
    require(follow.shouldFollow, "Automatic content growth must not suspend following")
    follow.phaseChanged(.animating)
    follow.proximityChanged(false)
    require(follow.shouldFollow, "Programmatic scrolling must keep following new content")
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

  private static func recognizedCPU(_ image: CGImage) throws -> String {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.recognitionLanguages = ["en-US"]
    request.usesLanguageCorrection = false
    let stages = try request.supportedComputeStageDevices
    guard !stages.isEmpty else {
      throw NSError(
        domain: "MLXL3LongVisibility", code: 2,
        userInfo: [
          NSLocalizedDescriptionKey:
            "Vision supplied no compute stages; CPU-only OCR not established"
        ])
    }
    for (stage, devices) in stages {
      guard
        let cpu = devices.first(where: {
          if case .cpu = $0 { return true }
          return false
        })
      else {
        throw NSError(
          domain: "MLXL3LongVisibility", code: 2,
          userInfo: [NSLocalizedDescriptionKey: "OCR stage has no CPU device: \(stage)"])
      }
      request.setComputeDevice(cpu, for: stage)
    }
    try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
    return (request.results ?? []).compactMap { $0.topCandidates(1).first?.string }.joined(
      separator: " ")
  }

  @MainActor private static func checkLongVisibility() async throws {
    func report(_ text: String) {
      FileHandle.standardError.write(Data(("Long visibility: " + text + "\n").utf8))
    }
    func require(_ condition: Bool, _ message: String) throws {
      guard condition else {
        report("FAIL " + message)
        throw NSError(
          domain: "MLXL3LongVisibility", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
      }
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-long-visibility-\(UUID())")
    let suite = "io.mlxl3.long-visibility.\(UUID())"
    let prefs = UserDefaults(suiteName: suite)!
    defer {
      prefs.removePersistentDomain(forName: suite)
      try? FileManager.default.removeItem(at: root)
    }
    let studio = StudioModel(
      conversationFileURL: root.appendingPathComponent("history.json"),
      isPreview: true, preferences: prefs)
    let previous = ChatMessage(role: .assistant, content: "Previous answer remains in history.")
    let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
    let host = NSHostingView(
      rootView: MessagesView(messages: [previous, message])
        .environmentObject(studio).background(StudioTheme.canvas).preferredColorScheme(.dark))
    let window = NSWindow(
      contentRect: NSRect(x: 50, y: 50, width: 900, height: 700),
      styleMask: [.titled, .resizable], backing: .buffered, defer: false)
    window.level = .floating
    window.hidesOnDeactivate = false
    window.contentView = host
    NSApp.activate()
    window.makeKeyAndOrderFront(nil)
    defer {
      window.contentView = nil
      window.orderOut(nil)
    }
    func nativeScroll(_ view: NSView) -> NSScrollView? {
      if let scroll = view as? NSScrollView { return scroll }
      return view.subviews.lazy.compactMap { nativeScroll($0) }.first
    }
    func normalized(_ text: String) -> String { String(text.lowercased().filter(\.isLetter)) }
    func ensurePresented(_ label: String) async throws {
      guard !window.isVisible || !window.occlusionState.contains(.visible) else { return }
      report(
        "PRESENTATION RECOVERY \(label): visible=\(window.isVisible), occluded=\(!window.occlusionState.contains(.visible))"
      )
      NSApp.activate()
      window.makeKeyAndOrderFront(nil)
      for _ in 0..<25 {
        if window.isVisible && window.occlusionState.contains(.visible) {
          report("PRESENTATION READY \(label)")
          return
        }
        try await Task.sleep(for: .milliseconds(100))
      }
      try require(
        window.isVisible && window.occlusionState.contains(.visible),
        "Native stress window failed bounded presentation setup at \(label)")
    }
    try await ensurePresented("setup")
    var frames = 0
    var firstMisses = 0
    func checkFrame(_ label: String, marker: String, save: Bool = false) async throws {
      func capture() throws -> (
        bitmap: NSBitmapImageRep, image: CGImage, ink: Int, height: CGFloat, offset: CGFloat
      ) {
        host.layoutSubtreeIfNeeded()
        try require(
          window.isVisible && window.occlusionState.contains(.visible),
          "Native stress window is hidden/occluded at \(label)")
        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
          throw NSError(
            domain: "MLXL3LongVisibility", code: 3,
            userInfo: [NSLocalizedDescriptionKey: "No viewport bitmap at \(label)"])
        }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let outer = nativeScroll(host)
        let height = outer?.documentView?.bounds.height ?? 0
        let offset = outer?.contentView.bounds.minY ?? 0
        var ink = 0
        for y in stride(from: bitmap.pixelsHigh / 5, to: bitmap.pixelsHigh * 9 / 10, by: 3) {
          for x in stride(from: bitmap.pixelsWide / 12, to: bitmap.pixelsWide * 11 / 12, by: 3) {
            if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.sRGB),
              color.redComponent + color.greenComponent + color.blueComponent > 1.15
            {
              ink += 1
            }
          }
        }
        return (bitmap, bitmap.cgImage!, ink, height, offset)
      }
      func saveBitmap(_ bitmap: NSBitmapImageRep, suffix: String) throws {
        guard let folder = ProcessInfo.processInfo.environment["MLXL3_STREAM_UI_SNAPSHOTS"] else {
          return
        }
        let directory = URL(fileURLWithPath: folder)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bitmap.representation(using: .png, properties: [:])!.write(
          to: directory.appendingPathComponent("long-\(label)-\(suffix).png"))
      }
      try await ensurePresented(label)
      try await Task.sleep(for: .milliseconds(350))
      var snapshot = try capture()
      let firstImage = snapshot.image
      var text = try await Task.detached(priority: .utility) { try Self.recognizedCPU(firstImage) }
        .value
      var found = normalized(text).contains(normalized(marker))
      report(
        "\(label) first thinking=\(message.thinking.utf8.count)B answer=\(message.content.utf8.count)B ink=\(snapshot.ink) marker=\(found) documentHeight=\(snapshot.height) offset=\(snapshot.offset)"
      )
      if !found || snapshot.ink < 40 {
        firstMisses += 1
        report("FIRST MISS \(label): expected=\(marker); OCR=\(text)")
        try saveBitmap(snapshot.bitmap, suffix: "first-miss")
        // Let native layout and scroll animations run without the OCR observer blocking the main actor.
        try await Task.sleep(for: .seconds(2))
        snapshot = try capture()
        let settledImage = snapshot.image
        text = try await Task.detached(priority: .utility) { try Self.recognizedCPU(settledImage) }
          .value
        found = normalized(text).contains(normalized(marker))
        report(
          "\(label) settled ink=\(snapshot.ink) marker=\(found) documentHeight=\(snapshot.height) offset=\(snapshot.offset)"
        )
        try saveBitmap(
          snapshot.bitmap,
          suffix: found && snapshot.ink >= 40 ? "settled-recovery" : "settled-failure")
      } else if save {
        try saveBitmap(snapshot.bitmap, suffix: "first")
      }
      try require(
        window.occlusionState.contains(.visible), "Native stress window became occluded at \(label)"
      )
      try require(
        snapshot.ink >= 40 && found,
        "Visible tail persistently missing at \(label): expected=\(marker); OCR=\(text); ink=\(snapshot.ink), height=\(snapshot.height), offset=\(snapshot.offset)"
      )
      frames += 1
    }
    let checkpoints = ["ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL"]
    func repeated(_ line: String) -> String {
      String(repeating: line, count: max(1, 16_384 / line.utf8.count))
    }
    // Flat paragraphs deliberately cross the previous 16 KiB line/chunk boundary.
    for checkpoint in checkpoints {
      let marker = "VISIBLE THINKING \(checkpoint)"
      message.append(
        repeated("Reasoning through a long project, preserving every visible word. ") + marker
          + " ", phase: "thinking")
      try await checkFrame(
        "thinking-\(checkpoint.lowercased())", marker: marker, save: checkpoint == "HOTEL")
    }
    try require(
      message.thinking.utf8.count >= 120_000, "Thinking stress never reached the declared size")
    message.append("Beginning the answer. ", phase: "answer")
    for checkpoint in checkpoints {
      let marker = "VISIBLE ANSWER \(checkpoint)"
      message.append(
        repeated(
          "The normal answer keeps its complete explanation readable as the document becomes very long. "
        ) + marker + " ", phase: "answer")
      try await checkFrame(
        "answer-\(checkpoint.lowercased())", marker: marker, save: checkpoint == "HOTEL")
    }
    message.append("\n\n```javascript\n// ", phase: "answer")
    for checkpoint in checkpoints {
      let marker = "VISIBLE CODE \(checkpoint)"
      message.append(
        repeated("Long generated source remains painted with exact content and stable scrolling. ")
          + marker + " ", phase: "answer")
      try await checkFrame(
        "code-\(checkpoint.lowercased())", marker: marker, save: checkpoint == "HOTEL")
    }
    message.append("\n```\n\nVISIBLE BEFORE TOOL CHECKPOINT\n\n", phase: "answer")
    try await checkFrame("before-tool", marker: "VISIBLE BEFORE TOOL CHECKPOINT")
    message.startTool(
      id: "visibility-tool", serverName: "Disposable fixture", toolName: "VISIBLE TOOL CHECKPOINT")
    try await checkFrame("tool-running", marker: "VISIBLE TOOL CHECKPOINT")
    message.finishTool(id: "visibility-tool", result: "Fixture action complete", isError: false)
    message.processing("VISIBLE PROCESSING CHECKPOINT")
    try await checkFrame("processing", marker: "VISIBLE PROCESSING CHECKPOINT")
    message.append("VISIBLE FINAL CHECKPOINT\n\n", phase: "answer")
    message.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
    try await checkFrame("complete", marker: "VISIBLE FINAL CHECKPOINT", save: true)
    try require(
      message.content.utf8.count >= 250_000 && message.toolActivities.first?.state == .complete,
      "Long answer/tool fixture did not reach the declared boundary")
    window.setContentSize(NSSize(width: 640, height: 520))
    try await checkFrame("narrow", marker: "VISIBLE FINAL CHECKPOINT")
    window.setContentSize(NSSize(width: 900, height: 700))
    try await checkFrame("wide", marker: "VISIBLE FINAL CHECKPOINT")
    report(
      "PASS \(frames) actual viewport/OCR checks, initial misses=\(firstMisses) preserved, >120KiB thinking + >250KiB answer, tool/completion/resize, detached OCR CPU only"
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

  @MainActor static func main() {
    _ = NSApplication.shared
    NSApp.setActivationPolicy(.regular)
    NSApp.activate()
    Task { @MainActor in
      do {
        try await runChecks()
        NSApp.terminate(nil)
      } catch {
        FileHandle.standardError.write(Data((String(describing: error) + "\n").utf8))
        exit(1)
      }
    }
    NSApp.run()
  }

  @MainActor private static func runChecks() async throws {
    if CommandLine.arguments.contains("--check-long-visibility") {
      try await checkLongVisibility()
      return
    }
    if CommandLine.arguments.contains("--check-inline-preview") {
      try await checkInlinePreviewLifetime()
      return
    }
    checkFade()
    try await checkScroll()
    try await checkHTMLControls()
    try await checkInlinePreviewLifetime()
    try await checkLongVisibility()
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
      precondition(
        MTMathListBuilder.mathListToString(copied) == roundTrip,
        "Copy dropped the box or introduced invisible delimiters")
      let source = "Before.\n\n$$\n" + formula + "\n$$\n\nAfter."
      precondition(
        description(MarkdownParser.parse(source)) == [
          "0:paragraph:Before.", "1:math:" + formula, "2:paragraph:After.",
        ])
      for mode in [MTMathUILabelMode.text, .display] {
        let image = LatexImageCache.image(latex: formula, fontSize: 18, mode: mode)
        precondition(image != nil, "Production math image fell back to text")
        precondition(
          image!.size.width.isFinite && image!.size.height.isFinite && image!.size.width > 0
            && image!.size.height > 0)
        let bitmap = NSBitmapImageRep(data: image!.tiffRepresentation!)!
        precondition(bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0)
        precondition(
          (0..<bitmap.pixelsHigh).contains { y in
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
    precondition(
      boxed.size.width > plain.size.width && boxed.size.height > plain.size.height,
      "Box command was stripped instead of framed")
    let emptyBox = LatexImageCache.image(latex: #"\boxed{}"#, fontSize: 18, mode: .display)!
    let frame = NSBitmapImageRep(data: emptyBox.tiffRepresentation!)!
    let w = frame.pixelsWide
    let h = frame.pixelsHigh
    precondition(
      frame.colorAt(x: w / 2, y: h / 2)!.alphaComponent == 0,
      "Frame was filled over its content")
    precondition(
      (0..<min(2, h)).contains { y in
        frame.colorAt(x: w / 2, y: y)!.alphaComponent > 0
      }
        && (max(0, h - 2)..<h).contains { y in
          frame.colorAt(x: w / 2, y: y)!.alphaComponent > 0
        }, "Top or bottom frame stroke missing")
    precondition(
      (0..<min(2, w)).contains { x in
        frame.colorAt(x: x, y: h / 2)!.alphaComponent > 0
      }
        && (max(0, w - 2)..<w).contains { x in
          frame.colorAt(x: x, y: h / 2)!.alphaComponent > 0
        }, "Left or right frame stroke missing")
    for invalid in [
      #"\boxed{x"#, #"\notASupportedCommand{x}"#,
      String(repeating: "x", count: 4097),
    ] {
      precondition(LatexImageCache.image(latex: invalid, fontSize: 18, mode: .display) == nil)
    }
    let codeMath = "```latex\n" + formulas[1] + "\n```"
    precondition(description(MarkdownParser.parse(codeMath)) == ["0:code:latex:" + formulas[1]])
    print(
      "LaTeX rendering checks passed: 9 formulas, 2 styles, parse/copy/roundtrip, actual pixels/frame, invalid input"
    )
    let pasteboard = NSPasteboard.withUniqueName()
    defer { pasteboard.releaseGlobally() }
    let answer =
      "# Été 👩‍💻\n\nFirst **paragraph**.\n\n```swift\nlet x = 73\n```\n\n$x^2$\n"
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
    let host = NSHostingView(
      rootView: MessagesView(messages: studio.currentConversation!.messages).environmentObject(
        studio))
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
