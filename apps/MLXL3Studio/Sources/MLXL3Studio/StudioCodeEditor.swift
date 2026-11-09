import AppKit
import SwiftUI

struct StudioCodeEditor: NSViewRepresentable {
  @Binding var text: String
  var language: String
  var editable: Bool
  var documentID: String
  var diff = false
  var fontSize: CGFloat = 12.5

  func makeCoordinator() -> Coordinator { Coordinator(self) }

  func makeNSView(context: Context) -> NSScrollView {
    let scroll = NSScrollView()
    scroll.clipsToBounds = true
    scroll.wantsLayer = true
    scroll.layer?.masksToBounds = true
    scroll.hasVerticalScroller = true
    scroll.hasHorizontalScroller = true
    scroll.autohidesScrollers = true
    scroll.borderType = .noBorder
    scroll.backgroundColor = NSColor(StudioTheme.canvas)
    let editor = NSTextView(frame: .zero)
    editor.isRichText = false
    editor.allowsUndo = true
    editor.usesFindBar = true
    editor.isAutomaticQuoteSubstitutionEnabled = false
    editor.isAutomaticDashSubstitutionEnabled = false
    editor.isAutomaticTextReplacementEnabled = false
    editor.isAutomaticLinkDetectionEnabled = false
    editor.isContinuousSpellCheckingEnabled = false
    editor.isGrammarCheckingEnabled = false
    editor.backgroundColor = NSColor(StudioTheme.canvas)
    editor.insertionPointColor = NSColor(StudioTheme.ink)
    editor.textContainerInset = NSSize(width: 18, height: 16)
    editor.minSize = .zero
    editor.maxSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
    editor.isVerticallyResizable = true
    editor.isHorizontallyResizable = true
    editor.autoresizingMask = [.width]
    editor.textContainer?.containerSize = NSSize(
      width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
    editor.textContainer?.widthTracksTextView = false
    editor.delegate = context.coordinator
    editor.setAccessibilityIdentifier("studio.code-editor")
    scroll.documentView = editor
    let ruler = SourceLineRuler(scrollView: scroll, orientation: .verticalRuler)
    ruler.ruleThickness = 48
    ruler.clientView = editor
    scroll.verticalRulerView = ruler
    scroll.hasVerticalRuler = true
    scroll.rulersVisible = !diff
    updateNSView(scroll, context: context)
    return scroll
  }

  func updateNSView(_ scroll: NSScrollView, context: Context) {
    guard let editor = scroll.documentView as? NSTextView else { return }
    let changedDocument = context.coordinator.parent.documentID != documentID
    let changedFont = context.coordinator.parent.fontSize != fontSize
      || context.coordinator.parent.language != language || context.coordinator.parent.diff != diff
    context.coordinator.parent = self
    editor.isEditable = editable
    editor.isSelectable = true
    scroll.rulersVisible = !diff
    if editor.string != text || changedDocument {
      guard !editor.hasMarkedText() || changedDocument else { return }
      editor.string = text
      context.coordinator.resetLines(editor)
      editor.undoManager?.removeAllActions()
      if changedDocument { editor.scrollRangeToVisible(NSRange(location: 0, length: 0)) }
      context.coordinator.style(editor)
    } else if changedFont || editor.textStorage?.length == 0 {
      context.coordinator.resetLines(editor)
      context.coordinator.style(editor)
    }
  }

  @MainActor final class Coordinator: NSObject, NSTextViewDelegate {
    var parent: StudioCodeEditor
    private var task: Task<Void, Never>?
    private var styling: Task<Void, Never>?
    private var lexer: Task<EditorLexicalCache.Result, Never>?
    private var lexicalCache: EditorLexicalCache.Cache?
    private var revision = UUID()
    private var changedRange: NSRange?
    private var previouslyHighlighted = false

    func resetLines(_ editor: NSTextView) {
      invalidateStyle()
      lexicalCache = nil
      changedRange = nil
      if let ruler = editor.enclosingScrollView?.verticalRulerView as? SourceLineRuler {
        ruler.offsets = SourceLineOffsets.build(editor.string)
      }
    }

    func textView(_ textView: NSTextView, shouldChangeTextIn affectedCharRange: NSRange,
                  replacementString: String?) -> Bool {
      guard let replacementString else { return true }
      invalidateStyle()
      if let ruler = textView.enclosingScrollView?.verticalRulerView as? SourceLineRuler {
        ruler.offsets = SourceLineOffsets.replacing(ruler.offsets, range: affectedCharRange,
                                                   replacement: replacementString)
        ruler.needsDisplay = true
      }
      // Attribute inheritance already styles inserted text. Keep subsequent plain
      // large-document styling confined to the changed span, never the full buffer.
      let edit = NSRange(location: affectedCharRange.location, length: (replacementString as NSString).length)
      changedRange = changedRange.map { NSUnionRange($0, edit) } ?? edit
      return true
    }
    init(_ parent: StudioCodeEditor) { self.parent = parent }
    func textDidChange(_ notification: Notification) {
      guard let editor = notification.object as? NSTextView else { return }
      invalidateStyle()
      parent.text = editor.string
      task?.cancel()
      task = Task { [weak self, weak editor] in
        try? await Task.sleep(for: .milliseconds(120))
        guard !Task.isCancelled, let self, let editor else { return }
        style(editor)
      }
    }

    private func invalidateStyle() {
      revision = UUID()
      styling?.cancel()
      lexer?.cancel()
    }

    func waitForStyling() async { await styling?.value }

    func style(_ editor: NSTextView) {
      guard !editor.hasMarkedText(), let storage = editor.textStorage else { return }
      invalidateStyle()
      let ticket = revision
      let source = editor.string
      let highlight = source.utf8.count <= 256 * 1024
      let fullRange = NSRange(location: 0, length: storage.length)
      let requested = (!highlight && !previouslyHighlighted ? changedRange : nil) ?? fullRange
      let range = NSIntersectionRange(requested, fullRange)
      let language = parent.language, diff = parent.diff
      let previous = lexicalCache
      let work = Task.detached(priority: .utility) {
        let fragment = (source as NSString).substring(with: range)
        if highlight {
          return EditorLexicalCache.update(fragment, language: language, diff: diff, previous: previous)
        }
        return EditorLexicalCache.Result(cache: nil, tokens: [
          SyntaxHighlighter.Token(range: NSRange(location: 0, length: range.length), color: 0)
        ], scannedLines: 0, lexedChunks: 0, reusedChunks: 0)
      }
      lexer = work
      styling = Task { [weak self, weak editor] in
        let result = await work.value
        guard !Task.isCancelled, let self, let editor, self.revision == ticket,
              !editor.hasMarkedText(), editor.string == source else { return }
        self.lexicalCache = result.cache
        self.apply(result.tokens, range: range, highlight: highlight, editor: editor)
      }
    }

    private func apply(_ tokens: [SyntaxHighlighter.Token], range: NSRange,
                       highlight: Bool, editor: NSTextView) {
      guard let storage = editor.textStorage else { return }
      let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.monospacedSystemFont(ofSize: parent.fontSize, weight: .regular),
        .foregroundColor: NSColor(StudioTheme.ink),
      ]
      let paragraph = NSMutableParagraphStyle()
      paragraph.lineSpacing = 4
      paragraph.defaultTabInterval = 30
      let ranges = editor.selectedRanges
      let registration = editor.undoManager?.isUndoRegistrationEnabled == true
      if registration { editor.undoManager?.disableUndoRegistration() }
      storage.beginEditing()
      // Apply attributes only. Replacing the string here disrupts marked text,
      // undo coalescing and selections even when characters are identical.
      for token in tokens where token.range.length > 0 {
        var desired = attributes
        desired[.paragraphStyle] = paragraph
        if highlight {
          let color = SyntaxHighlighter.color(token.color)
          desired[.foregroundColor] = color
          if token.color >= 7 {
            desired[.foregroundColor] = color.blended(withFraction: 0.3, of: NSColor(StudioTheme.ink)) ?? color
            desired[.backgroundColor] = color.withAlphaComponent(0.08)
          }
        }
        let target = NSRange(location: range.location + token.range.location, length: token.range.length)
        var edits: [NSRange] = []
        storage.enumerateAttributes(in: target) { current, run, _ in
          if !NSDictionary(dictionary: current).isEqual(to: desired) { edits.append(run) }
        }
        for edit in edits { storage.setAttributes(desired, range: edit) }
      }
      storage.endEditing()
      changedRange = nil
      previouslyHighlighted = highlight
      if registration { editor.undoManager?.enableUndoRegistration() }
      editor.selectedRanges = ranges
      editor.typingAttributes = attributes
      if let ruler = editor.enclosingScrollView?.verticalRulerView as? SourceLineRuler {
        ruler.needsDisplay = true
      }
    }
  }
}

/// Incremental *lexical* work, with conservative checkpoints. The regex renderer
/// remains the color oracle: an uncertain boundary enlarges a chunk, never resets
/// its context. Splitting/comparing lines and assembling ranges are still O(n).
enum EditorLexicalCache {
  struct State: Sendable, Equatable {
    var close = ""
    var escapes = false
    var restrictedComment = false
    var markupComment = false
    var doctype = false
    // A failed regex candidate may expose another opener inside it. In that case
    // retain the rest as one chunk rather than guessing the regex's backtracking.
    var uncertain = false

    var neutral: Bool {
      close.isEmpty && !markupComment && !doctype && !uncertain
    }
  }

  struct Line: Sendable {
    let text: String
    let after: State
    let checkpoint: Bool
  }

  struct Chunk: Sendable {
    let text: String
    let tokens: [SyntaxHighlighter.Token]
  }

  struct Cache: Sendable {
    let language: String
    let diff: Bool
    let lines: [Line]
    let chunks: [Chunk]
  }

  struct Result: Sendable {
    let cache: Cache?
    let tokens: [SyntaxHighlighter.Token]
    let scannedLines: Int
    let lexedChunks: Int
    let reusedChunks: Int
  }

  nonisolated static func update(_ source: String, language raw: String, diff: Bool,
                                previous: Cache? = nil) -> Result {
    let language = normalized(raw)
    let old = previous.flatMap { $0.language == language && $0.diff == diff ? $0 : nil }
    let parts = source.components(separatedBy: "\n")
    let texts = parts.enumerated().map { $0.element + ($0.offset + 1 < parts.count ? "\n" : "") }
    let oldLines = old?.lines ?? []
    var prefix = 0
    while prefix < min(texts.count, oldLines.count), texts[prefix] == oldLines[prefix].text {
      prefix += 1
    }
    var suffix = 0
    while suffix < min(texts.count, oldLines.count) - prefix,
          texts[texts.count - 1 - suffix] == oldLines[oldLines.count - 1 - suffix].text {
      suffix += 1
    }
    var lines = Array(oldLines.prefix(prefix))
    var state = lines.last?.after ?? State()
    var index = prefix
    var scanned = 0
    while index < texts.count {
      if Task.isCancelled { return Result(cache: nil, tokens: [], scannedLines: scanned, lexedChunks: 0, reusedChunks: 0) }
      if index >= texts.count - suffix {
        let oldIndex = oldLines.count - (texts.count - index)
        let incoming = oldIndex == 0 ? State() : oldLines[oldIndex - 1].after
        if state == incoming {
          lines.append(contentsOf: oldLines[oldIndex...])
          break
        }
      }
      state = diff ? State() : advance(texts[index], from: state, language: language)
      lines.append(Line(text: texts[index], after: state,
                        checkpoint: diff || (state.neutral && safeEnd(texts[index]))))
      scanned += 1
      index += 1
    }

    // Each cached chunk starts at a neutral checkpoint. Identical text therefore
    // has identical colors even after edits shift its absolute UTF-16 location.
    var reusable: [String: [SyntaxHighlighter.Token]] = [:]
    for chunk in old?.chunks ?? [] { reusable[chunk.text] = chunk.tokens }
    var chunks: [Chunk] = []
    var pending = ""
    var lexed = 0, reused = 0
    for (lineIndex, line) in lines.enumerated() {
      pending += line.text
      guard line.checkpoint || lineIndex + 1 == lines.count else { continue }
      if Task.isCancelled { return Result(cache: nil, tokens: [], scannedLines: scanned, lexedChunks: lexed, reusedChunks: reused) }
      let tokens: [SyntaxHighlighter.Token]
      if let cached = reusable[pending] {
        tokens = cached
        reused += 1
      } else {
        tokens = SyntaxHighlighter.tokens(pending, language: language, diff: diff)
        lexed += 1
      }
      chunks.append(Chunk(text: pending, tokens: tokens))
      pending = ""
    }
    var tokens: [SyntaxHighlighter.Token] = []
    var offset = 0
    for chunk in chunks {
      for token in chunk.tokens where token.range.length > 0 {
        let range = NSRange(location: offset + token.range.location, length: token.range.length)
        if let last = tokens.last, last.color == token.color, NSMaxRange(last.range) == range.location {
          tokens[tokens.count - 1] = SyntaxHighlighter.Token(
            range: NSRange(location: last.range.location, length: last.range.length + range.length), color: token.color)
        } else {
          tokens.append(SyntaxHighlighter.Token(range: range, color: token.color))
        }
      }
      offset += chunk.text.utf16.count
    }
    return Result(cache: Cache(language: language, diff: diff, lines: lines, chunks: chunks),
                  tokens: tokens, scannedLines: scanned, lexedChunks: lexed, reusedChunks: reused)
  }

  /// An identifier at EOL could acquire a function/markup-attribute lookahead on
  /// a later line. Keep those lines in the same chunk, including intervening blanks.
  nonisolated private static func safeEnd(_ line: String) -> Bool {
    let units = Array(line.trimmingCharacters(in: .whitespacesAndNewlines).utf16)
    guard let last = units.last else { return false }
    if Array(";{}[]),>\"'`=+*%&!?".utf16).contains(last) { return true }
    guard (48...57).contains(last) else { return false }
    let word = units.reversed().prefix {
      (48...57).contains($0) || (65...90).contains($0) || (97...122).contains($0)
        || [95, 36, 58, 46, 45].contains(Int($0))
    }
    return word.last.map { (48...57).contains($0) } ?? false
  }

  nonisolated private static func advance(_ text: String, from incoming: State,
                                         language: String) -> State {
    var state = incoming
    let units = Array(text.utf16)
    let markup = ["html", "xml", "svg", "vue", "svelte"].contains(language)
    let hashComments = ["python", "bash", "yaml", "toml", "markdown"].contains(language)
    let restrictedComments = ["yaml", "toml", "markdown"].contains(language)
    let sql = language == "sql"
    let json = language == "json"
    let cComments = !markup && !hashComments && !sql && !json
    func matches(_ value: String, at index: Int) -> Bool {
      let pattern = Array(value.utf16)
      return index + pattern.count <= units.count && units[index..<(index + pattern.count)].elementsEqual(pattern)
    }
    // ICU treats these as line breaks too. Keep unusual line endings in the
    // remainder chunk so line-comment skipping cannot hide a subsequent opener.
    for (index, unit) in units.enumerated() {
      if unit == 0x85 || unit == 0x2028 || unit == 0x2029
        || (unit == 13 && (index + 1 == units.count || units[index + 1] != 10)) {
        state.uncertain = true
      }
    }
    // Markup uses overlapping regex passes. Its comment and DOCTYPE states must
    // advance independently of strings, including quotes inside comments/tags.
    if markup {
      var cursor = 0
      while cursor < units.count {
        if state.markupComment {
          if matches("-->", at: cursor) { state.markupComment = false; cursor += 3 } else { cursor += 1 }
        } else if matches("<!--", at: cursor) {
          state.markupComment = true; cursor += 4
        } else { cursor += 1 }
      }
      cursor = 0
      while cursor < units.count {
        if state.doctype {
          if units[cursor] == 62 { state.doctype = false }
          cursor += 1
        } else if matches("<!DOCTYPE", at: cursor) {
          state.doctype = true; cursor += 9
        } else { cursor += 1 }
      }
    }
    var index = 0
    while index < units.count {
      if !state.close.isEmpty {
        if matches(state.close, at: index) {
          index += state.close.utf16.count
          state.close = ""; state.escapes = false; state.restrictedComment = false
        } else if state.escapes && units[index] == 92 {
          if index + 1 == units.count || [10, 13, 0x85, 0x2028, 0x2029].contains(Int(units[index + 1])) {
            state.uncertain = true
          }
          index += min(2, units.count - index)
        } else {
          if state.restrictedComment && units[index] == 62 { state.uncertain = true }
          index += 1
        }
        continue
      }
      if !markup && ((hashComments && units[index] == 35)
        || (cComments && matches("//", at: index)) || (sql && matches("--", at: index))) {
        break
      }
      if (cComments || sql) && matches("/*", at: index) {
        state.close = "*/"; index += 2
      } else if restrictedComments && matches("<!--", at: index) {
        state.close = "-->"; state.restrictedComment = true; index += 4
      } else if ["python", "swift"].contains(language) && matches("\"\"\"", at: index) {
        state.close = "\"\"\""; index += 3
      } else if language == "python" && matches("'''", at: index) {
        state.close = "'''"; index += 3
      } else if units[index] == 34 {
        state.close = "\""; state.escapes = !sql; index += 1
      } else if units[index] == 39 && language != "swift" && !json {
        state.close = "'"
        state.escapes = !sql && language != "bash" && !restrictedComments
        index += 1
      } else if units[index] == 96 && cComments && language != "swift" {
        state.close = "`"; state.escapes = true; index += 1
      } else { index += 1 }
    }
    return state
  }

  nonisolated private static func normalized(_ raw: String) -> String {
    let language = raw.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
    return switch language {
    case "js", "jsx": "javascript"
    case "ts", "tsx": "typescript"
    case "py": "python"
    case "rs": "rust"
    case "sh", "zsh", "shell": "bash"
    case "yml": "yaml"
    case "md": "markdown"
    case "c++", "cc", "hpp": "cpp"
    case "cs": "csharp"
    default: language
    }
  }
}

private final class SourceLineRuler: NSRulerView {
  var offsets = [0]
  override var isFlipped: Bool { true }
  override func drawHashMarksAndLabels(in rect: NSRect) {
    NSColor(StudioTheme.sidebar).setFill()
    bounds.fill()
    guard let text = clientView as? NSTextView, let layout = text.layoutManager,
      let container = text.textContainer
    else { return }
    let visible = text.visibleRect.offsetBy(
      dx: -text.textContainerInset.width, dy: -text.textContainerInset.height)
    let glyphs = layout.glyphRange(forBoundingRect: visible, in: container)
    guard glyphs.length > 0 else { return }
    let first = layout.characterIndexForGlyph(at: glyphs.location)
    let last = NSMaxRange(layout.characterRange(forGlyphRange: glyphs, actualGlyphRange: nil))
    var low = 0
    var high = offsets.count
    while low < high {
      let mid = (low + high) / 2
      if offsets[mid] <= first { low = mid + 1 } else { high = mid }
    }
    for line in max(0, low - 1)..<offsets.count {
      let offset = offsets[line]
      if offset > last { break }
      let lineRect =
        offset < (text.string as NSString).length
        ? layout.lineFragmentRect(
          forGlyphAt: layout.glyphIndexForCharacter(at: offset), effectiveRange: nil)
        : layout.extraLineFragmentRect
      let label = NSAttributedString(
        string: "\(line + 1)",
        attributes: [
          .font: NSFont.monospacedDigitSystemFont(ofSize: 10, weight: .regular),
          .foregroundColor: NSColor(StudioTheme.quiet),
        ])
      label.draw(
        at: NSPoint(
          x: 35 - label.size().width,
          y: lineRect.minY + text.textContainerInset.height - text.visibleRect.minY + 2))
    }
  }
}

/// UTF-16 offsets match AppKit's edit ranges, including surrogate pairs.
enum SourceLineOffsets {
  static func build(_ text: String) -> [Int] {
    [0] + text.utf16.enumerated().compactMap { $0.element == 10 ? $0.offset + 1 : nil }
  }

  static func replacing(_ offsets: [Int], range: NSRange, replacement: String) -> [Int] {
    let end = NSMaxRange(range)
    let delta = replacement.utf16.count - range.length
    let before = offsets.prefix { $0 <= range.location }
    let inserted = replacement.utf16.enumerated().compactMap {
      $0.element == 10 ? range.location + $0.offset + 1 : nil
    }
    let after = offsets.drop { $0 <= end }.map { $0 + delta }
    return Array(before) + inserted + after
  }
}
