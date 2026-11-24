import AppKit
import SwiftUI

/// Shows one code block or HTML preview in its own full-screen window. Opening another one
/// replaces it. Closing it (Esc, ⌘W, the close button or leaving full screen) removes its
/// content, so an HTML preview's WebKit page stops with the window.
@MainActor
final class CodeFullscreenPresenter: NSObject, NSWindowDelegate {
  static let shared = CodeFullscreenPresenter()

  /// Native full screen opens a Space of its own; checks use a plain window instead.
  var usesNativeFullScreen = true
  private(set) var window: NSWindow?
  private var onClose: (() -> Void)?

  func present(title: String, content: some View, onClose: @escaping () -> Void) {
    close()
    let screen = NSApp.keyWindow?.screen ?? NSScreen.main
    let frame = screen?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1280, height: 820)
    let window = NSWindow(
      contentRect: frame, styleMask: [.titled, .closable, .miniaturizable, .resizable],
      backing: .buffered, defer: false)
    window.title = title
    window.isReleasedWhenClosed = false
    window.collectionBehavior.insert(.fullScreenPrimary)
    window.delegate = self
    window.contentView = NSHostingView(rootView: content)
    self.window = window
    self.onClose = onClose
    window.makeKeyAndOrderFront(nil)
    if usesNativeFullScreen { window.toggleFullScreen(nil) }
  }

  func close() { window?.close() }

  func windowWillClose(_ notification: Notification) {
    guard let closing = notification.object as? NSWindow, closing === window else { return }
    closing.delegate = nil
    closing.contentView = nil
    window = nil
    let callback = onClose
    onClose = nil
    callback?()
  }

  /// Leaving full screen (green button, ⌃⌘F) returns to the conversation.
  func windowDidExitFullScreen(_ notification: Notification) {
    guard (notification.object as? NSWindow) === window else { return }
    close()
  }
}

/// Content of the full-screen window: the same Code / Preview switch as the inline block,
/// at full size.
struct CodeFullscreenView: View {
  let language: String?
  let source: String
  let canPreview: Bool
  var workspace: URL? = nil
  var file: URL? = nil
  var textScale: CGFloat = 1
  @State var showingPreview: Bool
  let onClose: () -> Void

  var body: some View {
    VStack(spacing: 0) {
      HStack(spacing: 10) {
        Image(systemName: "chevron.left.forwardslash.chevron.right")
          .foregroundStyle(StudioTheme.quiet)
        Text(file?.lastPathComponent ?? language?.uppercased() ?? "CODE")
          .font(.system(size: 12, weight: .medium, design: .monospaced))
          .lineLimit(1).truncationMode(.middle)
        Spacer(minLength: 8)
        if canPreview {
          HStack(spacing: 2) {
            mode("Code", icon: "chevron.left.forwardslash.chevron.right", preview: false)
            mode(L("Aperçu", "Preview"), icon: "play", preview: true)
          }
          .padding(3).background(StudioTheme.contrast.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
        }
        Button(action: onClose) {
          Label(L("Quitter le plein écran", "Exit full screen"),
                systemImage: "arrow.down.right.and.arrow.up.left")
            .font(.system(size: 11)).padding(.horizontal, 11).frame(height: 28)
        }
        .buttonStyle(StudioControlStyle())
        .keyboardShortcut(.cancelAction)
        .accessibilityIdentifier("code.fullscreen.exit")
        .help(L("Quitter le plein écran (Échap)", "Exit full screen (Esc)"))
      }
      .padding(.horizontal, 18).frame(height: 52)
      .background(StudioTheme.contrast.opacity(0.025))
      Rectangle().fill(StudioTheme.edge).frame(height: 1)
      if showingPreview && canPreview {
        InlineHTMLPreview(source: source, workspace: workspace, file: file)
      } else {
        StudioCodeEditor(
          text: .constant(source), language: language ?? "text", editable: false,
          documentID: "fullscreen-code", fontSize: 14 * textScale)
      }
    }
    .frame(minWidth: 480, minHeight: 320)
    .background(StudioTheme.canvas)
  }

  private func mode(_ label: String, icon: String, preview: Bool) -> some View {
    Button { showingPreview = preview } label: {
      Label(label, systemImage: icon)
        .font(.system(size: 10, weight: .medium))
        .padding(.horizontal, 9).frame(height: 26)
        .foregroundStyle(showingPreview == preview ? StudioTheme.ink : StudioTheme.quiet)
        .background(showingPreview == preview ? StudioTheme.contrast.opacity(0.085) : .clear,
                    in: RoundedRectangle(cornerRadius: 6))
    }
    .buttonStyle(.plain)
    .accessibilityIdentifier(preview ? "code.fullscreen.preview" : "code.fullscreen.source")
    .accessibilityAddTraits(showingPreview == preview ? .isSelected : [])
  }
}

extension StudioModel {
  /// Opens `source` full screen with the app's theme; `onClose` runs when the window goes.
  func presentCodeFullscreen(
    language: String?, source: String, canPreview: Bool, showingPreview: Bool,
    workspace: URL? = nil, file: URL? = nil, textScale: CGFloat = 1,
    onClose: @escaping () -> Void = {}
  ) {
    let presenter = CodeFullscreenPresenter.shared
    let title = file?.lastPathComponent
      ?? (showingPreview && canPreview ? L("Aperçu", "Preview") : (language?.uppercased() ?? "Code"))
    presenter.present(
      title: title,
      content: CodeFullscreenView(
        language: language, source: source, canPreview: canPreview, workspace: workspace,
        file: file, textScale: textScale, showingPreview: showingPreview,
        onClose: { presenter.close() }
      )
      .environmentObject(self)
      .studioAppearance(self),
      onClose: onClose)
  }
}
