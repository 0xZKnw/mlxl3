import AppKit
import SwiftUI

enum MessageClipboard {
  @MainActor @discardableResult
  static func copy(_ message: ChatMessage, to pasteboard: NSPasteboard = .general) -> Bool {
    guard !message.content.isEmpty else { return false }
    pasteboard.clearContents()
    return pasteboard.setString(message.content, forType: .string)
  }
}

struct MessageCopyButton: View {
  @ObservedObject var message: ChatMessage
  var pasteboard: NSPasteboard = .general
  @State private var copied = false

  var body: some View {
    Button {
      copied = MessageClipboard.copy(message, to: pasteboard)
    } label: {
      Image(systemName: copied ? "checkmark" : "doc.on.doc")
        .font(.system(size: 11.5, weight: .medium))
        .foregroundStyle(copied ? StudioTheme.ink : StudioTheme.quiet)
        .contentTransition(.symbolEffect(.replace))
        .frame(width: 26, height: 26)
    }
    .buttonStyle(StudioControlStyle())
    .disabled(message.content.isEmpty)
    .help(L("Copier le message entier", "Copy entire message"))
    .accessibilityLabel(copied ? L("Copié", "Copied") : L("Copier le message entier", "Copy entire message"))
    .animation(.easeOut(duration: 0.15), value: copied)
    .task(id: copied) {
      guard copied else { return }
      try? await Task.sleep(for: .seconds(1.4))
      guard !Task.isCancelled else { return }
      copied = false
    }
  }
}
