import AppKit
import SwiftUI

struct StudioCodeEditor: NSViewRepresentable {
  @Binding var text: String
  var language: String
  var editable: Bool
  var documentID: String
  var diff = false

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
    context.coordinator.parent = self
    editor.isEditable = editable
    editor.isSelectable = true
    scroll.rulersVisible = !diff
    if editor.string != text || changedDocument {
      editor.string = text
      editor.undoManager?.removeAllActions()
      if changedDocument { editor.scrollRangeToVisible(NSRange(location: 0, length: 0)) }
      context.coordinator.style(editor)
    } else if editor.textStorage?.length == 0 {
      context.coordinator.style(editor)
    }
  }

  @MainActor final class Coordinator: NSObject, NSTextViewDelegate {
    var parent: StudioCodeEditor
    private var task: Task<Void, Never>?
    init(_ parent: StudioCodeEditor) { self.parent = parent }
    func textDidChange(_ notification: Notification) {
      guard let editor = notification.object as? NSTextView else { return }
      parent.text = editor.string
      task?.cancel()
      task = Task { [weak self, weak editor] in
        try? await Task.sleep(for: .milliseconds(120))
        guard !Task.isCancelled, let self, let editor else { return }
        style(editor)
      }
    }
    func style(_ editor: NSTextView) {
      let source = editor.string
      let attributes: [NSAttributedString.Key: Any] = [
        .font: NSFont.monospacedSystemFont(ofSize: 12.5, weight: .regular),
        .foregroundColor: NSColor(StudioTheme.ink),
      ]
      let result = NSMutableAttributedString(string: source, attributes: attributes)
      // shortcut: files above 256 KiB stay plain; add visible-range lexing if larger files need colors.
      if source.utf8.count <= 256 * 1024, !parent.diff {
        result.setAttributedString(
          NSAttributedString(SyntaxHighlighter.highlight(source, language: parent.language)))
        result.addAttribute(
          .font, value: attributes[.font]!, range: NSRange(location: 0, length: result.length))
        if result.length > 0, result.attribute(.foregroundColor, at: 0, effectiveRange: nil) == nil
        {
          result.addAttribute(
            .foregroundColor, value: NSColor(StudioTheme.ink),
            range: NSRange(location: 0, length: result.length))
        }
      }
      if parent.diff {
        var offset = 0
        for line in source.components(separatedBy: "\n") {
          let length = (line as NSString).length
          let color: NSColor? =
            line.hasPrefix("+") && !line.hasPrefix("+++")
            ? NSColor.systemGreen
            : line.hasPrefix("-") && !line.hasPrefix("---")
              ? NSColor.systemRed : line.hasPrefix("@@") ? NSColor.systemTeal : nil
          if let color, length > 0 {
            result.addAttributes(
              [
                .foregroundColor: color.blended(withFraction: 0.3, of: NSColor(StudioTheme.ink))
                  ?? color, .backgroundColor: color.withAlphaComponent(0.08),
              ], range: NSRange(location: offset, length: length))
          }
          offset += length + 1
        }
      }
      let paragraph = NSMutableParagraphStyle()
      paragraph.lineSpacing = 4
      paragraph.defaultTabInterval = 30
      result.addAttribute(
        .paragraphStyle, value: paragraph, range: NSRange(location: 0, length: result.length))
      let ranges = editor.selectedRanges
      let registration = editor.undoManager?.isUndoRegistrationEnabled == true
      if registration { editor.undoManager?.disableUndoRegistration() }
      editor.textStorage?.setAttributedString(result)
      if registration { editor.undoManager?.enableUndoRegistration() }
      editor.selectedRanges = ranges
      editor.typingAttributes = attributes
      if let ruler = editor.enclosingScrollView?.verticalRulerView as? SourceLineRuler {
        ruler.offsets =
          [0] + source.utf16.enumerated().compactMap { $0.element == 10 ? $0.offset + 1 : nil }
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
