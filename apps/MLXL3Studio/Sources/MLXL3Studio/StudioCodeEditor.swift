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
    private var lexer: Task<[SyntaxHighlighter.Token], Never>?
    private var revision = UUID()
    private var changedRange: NSRange?
    private var previouslyHighlighted = false

    func resetLines(_ editor: NSTextView) {
      invalidateStyle()
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
      let work = Task.detached(priority: .utility) {
        let fragment = (source as NSString).substring(with: range)
        if highlight { return SyntaxHighlighter.tokens(fragment, language: language, diff: diff) }
        return [SyntaxHighlighter.Token(range: NSRange(location: 0, length: range.length), color: 0)]
      }
      lexer = work
      styling = Task { [weak self, weak editor] in
        let tokens = await work.value
        guard !Task.isCancelled, let self, let editor, self.revision == ticket,
              !editor.hasMarkedText(), editor.string == source else { return }
        self.apply(tokens, range: range, highlight: highlight, editor: editor)
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
