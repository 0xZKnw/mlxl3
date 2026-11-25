import AppKit
import SwiftUI

/// Shared building blocks for the full-window pages (Settings, Lab, Models, MCP,
/// Projects, Images, Agent Studio) and the standalone modes (Server, Decision).
/// One set of metrics, the same as the main sidebar, so every page reads as the same app.
enum StudioPageMetrics {
    /// Rows sit this far from the sidebar edge, their content `content` further in.
    static let inset: CGFloat = 8
    static let content: CGFloat = 10
    static let row: CGFloat = 28
    static let radius: CGFloat = 7
    /// Reading column of a page.
    static let column: CGFloat = 760
    static let card: CGFloat = 12
    static let control: CGFloat = 30
}

extension StudioTheme {
    /// Something needs attention but nothing failed: at least 4.5:1 on the canvas.
    static let warning = Color(nsColor: adaptiveColor(light: rgb(0x8F5411), dark: rgb(0xE5A562)))
    /// A confirmed good state (loaded, online, verified).
    static let success = Color(nsColor: adaptiveColor(light: rgb(0x2C6A48), dark: rgb(0x7CC498)))
}

// MARK: - Sidebar

/// Hover, press and selection for a page sidebar row: the main sidebar's fills.
struct StudioRowStyle: ButtonStyle {
    var selected = false
    var radius = StudioPageMetrics.radius

    func makeBody(configuration: Configuration) -> some View {
        Surface(configuration: configuration, selected: selected, radius: radius)
    }

    private struct Surface: View {
        let configuration: ButtonStyle.Configuration
        let selected: Bool
        let radius: CGFloat
        @Environment(\.isEnabled) private var enabled
        @Environment(\.studioStyle) private var style
        @State private var hovered = false

        var body: some View {
            let fill = !enabled ? (selected ? 0.05 : 0)
                : configuration.isPressed ? 0.11
                : selected ? (style.isGraphite ? 0.085 : 0.075)
                : hovered ? 0.045 : 0
            configuration.label
                .background(StudioTheme.contrast.opacity(fill),
                            in: RoundedRectangle(cornerRadius: radius, style: .continuous))
                .contentShape(RoundedRectangle(cornerRadius: radius, style: .continuous))
                .opacity(enabled ? 1 : selected ? 0.8 : 0.45)
                .onHover { hovered = $0 }
        }
    }
}

/// A destination in a page sidebar: icon, label, optional trailing count.
struct StudioNavRow: View {
    let icon: String
    let title: String
    let selected: Bool
    var badge: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 9) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: selected ? .medium : .regular))
                    .frame(width: 18)
                    .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.quiet)
                Text(title)
                    .font(.system(size: 13, weight: selected ? .medium : .regular))
                    .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.secondary)
                    .lineLimit(1)
                Spacer(minLength: 4)
                if let badge {
                    Text(badge).font(.system(size: 11)).monospacedDigit()
                        .foregroundStyle(StudioTheme.quiet)
                }
            }
            .padding(.leading, 9)
            .padding(.trailing, StudioPageMetrics.content)
            .frame(height: StudioPageMetrics.row)
            .contentShape(Rectangle())
        }
        .buttonStyle(StudioRowStyle(selected: selected))
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// A page's sidebar: its name, its destinations, an optional footer, and Back in the
/// bottom row, where the main sidebar keeps the button that opened the page.
struct StudioPageSidebar<Content: View, Footer: View>: View {
    @EnvironmentObject private var studio: StudioModel
    let title: String
    var fixedWidth = true
    let onBack: () -> Void
    @ViewBuilder var content: Content
    @ViewBuilder var footer: Footer

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            StudioSidebarHeader(title: title)
            content
            Spacer(minLength: 0)
            footer
            StudioBackButton(action: onBack)
                .padding(.horizontal, StudioPageMetrics.inset)
                .padding(.top, 4)
                .padding(.bottom, 10)
        }
        .frame(width: fixedWidth ? studio.sidebarWidth.points : nil, alignment: .leading)
        .frame(maxHeight: .infinity, alignment: .top)
        .modifier(StudioSidebarSurface())
    }
}

extension StudioPageSidebar where Footer == EmptyView {
    init(title: String, fixedWidth: Bool = true, onBack: @escaping () -> Void, @ViewBuilder content: () -> Content) {
        self.init(title: title, fixedWidth: fixedWidth, onBack: onBack, content: content, footer: { EmptyView() })
    }
}

/// The destinations of a page sidebar, spaced like the main sidebar's modes.
struct StudioNavList<Content: View>: View {
    @ViewBuilder var content: Content

    var body: some View {
        VStack(spacing: 1) { content }
            .padding(.horizontal, StudioPageMetrics.inset)
    }
}

// MARK: - Page

/// The scrolling reading column of a page.
struct StudioPageColumn<Content: View>: View {
    var maxWidth = StudioPageMetrics.column
    var spacing: CGFloat = 24
    /// 40 under the window's title area; less under a workspace toolbar.
    var top: CGFloat = 40
    @ViewBuilder var content: Content

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: spacing) { content }
                .padding(.top, top).padding(.bottom, 36)
                .frame(maxWidth: maxWidth, alignment: .leading)
                .padding(.horizontal, 32)
                .frame(maxWidth: .infinity)
        }
    }
}

/// A page title with its one-line purpose and, on the right, the page's actions.
struct StudioPageHeader<Trailing: View>: View {
    let title: String
    var detail: String? = nil
    @ViewBuilder var trailing: Trailing

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            VStack(alignment: .leading, spacing: 6) {
                Text(title).font(StudioTheme.display(39)).tracking(StudioTheme.displayTracking(-1.2))
                    .foregroundStyle(StudioTheme.ink)
                    .accessibilityAddTraits(.isHeader)
                if let detail, !detail.isEmpty {
                    Text(detail).font(.system(size: 13.5)).foregroundStyle(StudioTheme.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Spacer(minLength: 0)
            trailing
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

extension StudioPageHeader where Trailing == EmptyView {
    init(title: String, detail: String? = nil) {
        self.init(title: title, detail: detail, trailing: { EmptyView() })
    }
}

/// A group title above a card or a list.
struct StudioSectionTitle: View {
    let title: String
    var icon: String? = nil

    var body: some View {
        HStack(spacing: 7) {
            if let icon {
                Image(systemName: icon).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                    .frame(width: 16).accessibilityHidden(true)
            }
            Text(title).font(.system(size: 13, weight: .semibold)).foregroundStyle(StudioTheme.ink)
        }
        .padding(.horizontal, 2)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}

// MARK: - Surfaces

private struct StudioCardSurface: ViewModifier {
    var padding: CGFloat
    var radius: CGFloat

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(StudioTheme.panel, in: shape)
            .overlay { shape.strokeBorder(StudioTheme.edge.opacity(0.7), lineWidth: 0.5).allowsHitTesting(false) }
    }
}

extension View {
    /// The flat page surface: panel fill and a hairline.
    func studioCard(padding: CGFloat = 16, radius: CGFloat = StudioPageMetrics.card) -> some View {
        modifier(StudioCardSurface(padding: padding, radius: radius))
    }
}

/// Rows separated by hairlines inside one card, like a settings list.
struct StudioRowList<Content: View>: View {
    var rowPadding: CGFloat = 12
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Group(subviews: content) { subviews in
                ForEach(Array(subviews.enumerated()), id: \.element.id) { index, subview in
                    if index > 0 {
                        Rectangle().fill(StudioTheme.edge.opacity(0.7)).frame(height: 0.5)
                            .accessibilityHidden(true)
                    }
                    subview
                        .padding(.vertical, rowPadding)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }
}

/// A titled list of setting rows in one card, with an optional note under it.
struct SettingSection<Content: View>: View {
    let title: String
    var icon: String? = nil
    var footer: String? = nil
    @ViewBuilder var content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            StudioSectionTitle(title: title, icon: icon)
            StudioRowList { content }
                .padding(.horizontal, 16)
                .studioCard(padding: 0)
            if let footer {
                Text(footer).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                    .fixedSize(horizontal: false, vertical: true)
                    .padding(.horizontal, 2)
            }
        }
        .font(.system(size: 13))
    }
}

/// One setting: what it is and what it does on the left, its control on the right.
struct SettingRow<Control: View>: View {
    let title: String
    var detail: String? = nil
    var icon: String? = nil
    @ViewBuilder var control: Control

    var body: some View {
        HStack(alignment: .center, spacing: 16) {
            if let icon {
                Image(systemName: icon).font(.system(size: 13)).foregroundStyle(StudioTheme.quiet)
                    .frame(width: 18).accessibilityHidden(true)
            }
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 13, weight: .medium)).foregroundStyle(StudioTheme.ink)
                if let detail, !detail.isEmpty {
                    Text(detail).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            control
        }
    }
}

extension SettingRow where Control == EmptyView {
    init(title: String, detail: String? = nil, icon: String? = nil) {
        self.init(title: title, detail: detail, icon: icon, control: { EmptyView() })
    }
}

/// A switch with its explanation: the whole row names the switch for VoiceOver.
struct SettingToggleRow: View {
    let title: String
    var detail: String? = nil
    @Binding var isOn: Bool

    var body: some View {
        SettingRow(title: title, detail: detail) {
            Toggle(title, isOn: $isOn).labelsHidden().toggleStyle(.switch).controlSize(.small)
        }
    }
}

/// A label and its value on one line, the value selectable.
struct StudioReading: View {
    let label: String
    let value: String
    var monospaced = false

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 18) {
            Text(label).foregroundStyle(StudioTheme.secondary)
            Spacer(minLength: 0)
            Text(value)
                .font(monospaced ? .system(size: 12, design: .monospaced) : .system(size: 13))
                .monospacedDigit().multilineTextAlignment(.trailing)
                .foregroundStyle(StudioTheme.ink).textSelection(.enabled)
        }
        .font(.system(size: 13))
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Controls

/// The page button: `secondary` for ordinary actions, `primary` for the one that moves
/// the page forward, `destructive` for removals, `quiet` for low-key links.
struct StudioButtonStyle: ButtonStyle {
    enum Kind { case secondary, primary, destructive, quiet }
    var kind = Kind.secondary
    var height = StudioPageMetrics.control

    func makeBody(configuration: Configuration) -> some View {
        Surface(configuration: configuration, kind: kind, height: height)
    }

    private struct Surface: View {
        let configuration: ButtonStyle.Configuration
        let kind: Kind
        let height: CGFloat
        @Environment(\.isEnabled) private var enabled
        @Environment(\.accessibilityReduceMotion) private var reduceMotion
        @State private var hovered = false

        private var foreground: Color {
            switch kind {
            case .primary: StudioTheme.onAccent.opacity(0.92)
            case .destructive: StudioTheme.failure
            case .secondary: StudioTheme.ink
            case .quiet: StudioTheme.secondary
            }
        }

        private var fill: Color {
            let pressed = configuration.isPressed
            switch kind {
            case .primary: return StudioTheme.ink.opacity(pressed ? 0.72 : hovered ? 0.88 : 1)
            case .destructive: return StudioTheme.failure.opacity(pressed ? 0.2 : hovered ? 0.15 : 0.09)
            case .secondary: return StudioTheme.contrast.opacity(pressed ? 0.13 : hovered ? 0.095 : 0.06)
            case .quiet: return StudioTheme.contrast.opacity(pressed ? 0.11 : hovered ? 0.07 : 0)
            }
        }

        var body: some View {
            let shape = RoundedRectangle(cornerRadius: 8, style: .continuous)
            configuration.label
                .font(.system(size: 12.5, weight: .medium))
                .lineLimit(1)
                .foregroundStyle(foreground)
                .padding(.horizontal, 12)
                .frame(height: height)
                .background(fill, in: shape)
                .overlay {
                    if kind == .secondary {
                        shape.strokeBorder(StudioTheme.edge.opacity(0.6), lineWidth: 0.5).allowsHitTesting(false)
                    }
                }
                .contentShape(shape)
                .opacity(enabled ? 1 : 0.38)
                .onHover { hovered = $0 }
                .animation(reduceMotion ? nil : .easeOut(duration: 0.14), value: hovered)
        }
    }
}

extension ButtonStyle where Self == StudioButtonStyle {
    static var studio: StudioButtonStyle { StudioButtonStyle() }
    static func studio(_ kind: StudioButtonStyle.Kind, height: CGFloat = StudioPageMetrics.control) -> StudioButtonStyle {
        StudioButtonStyle(kind: kind, height: height)
    }
}

/// A square icon button for a row's secondary actions (reveal, remove, refresh).
struct StudioIconButton: View {
    let icon: String
    let label: String
    var destructive = false
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: icon).font(.system(size: 12.5))
                .foregroundStyle(destructive ? StudioTheme.failure : StudioTheme.secondary)
                .frame(width: 28, height: 28).contentShape(Rectangle())
        }
        .buttonStyle(StudioRowStyle())
        .help(label)
        .accessibilityLabel(label)
    }
}

private struct StudioFieldSurface: ViewModifier {
    var monospaced: Bool
    @FocusState private var focused: Bool

    func body(content: Content) -> some View {
        let shape = RoundedRectangle(cornerRadius: 8, style: .continuous)
        content
            .textFieldStyle(.plain)
            .font(monospaced ? .system(size: 12.5, design: .monospaced) : .system(size: 13))
            .focused($focused)
            .padding(.horizontal, 10).padding(.vertical, 6)
            .frame(minHeight: StudioPageMetrics.control)
            .background(StudioTheme.contrast.opacity(focused ? 0.065 : 0.045), in: shape)
            .overlay {
                shape.strokeBorder(focused ? StudioTheme.contrast.opacity(0.22) : StudioTheme.edge, lineWidth: focused ? 1 : 0.5)
                    .allowsHitTesting(false)
            }
    }
}

/// The page field as a style, for a group of fields: `.textFieldStyle(StudioTextFieldStyle())`.
@MainActor
struct StudioTextFieldStyle: @preconcurrency TextFieldStyle {
    var monospaced = false

    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration.studioField(monospaced: monospaced)
    }
}

extension View {
    /// The page text field: flat, hairline, a visible focus ring.
    func studioField(monospaced: Bool = false) -> some View {
        modifier(StudioFieldSurface(monospaced: monospaced))
    }
}

// MARK: - Feedback

/// An inline message: what happened, in the colour of its severity, selectable.
struct StudioNotice: View {
    enum Kind { case info, success, warning, error }
    var kind = Kind.info
    let text: String

    private var icon: String {
        switch kind {
        case .info: "info.circle"
        case .success: "checkmark.circle"
        case .warning: "exclamationmark.triangle"
        case .error: "exclamationmark.octagon"
        }
    }

    private var color: Color {
        switch kind {
        case .info: StudioTheme.secondary
        case .success: StudioTheme.success
        case .warning: StudioTheme.warning
        case .error: StudioTheme.failure
        }
    }

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: 9) {
            Image(systemName: icon).font(.system(size: 12, weight: .medium)).foregroundStyle(color)
                .accessibilityHidden(true)
            Text(text).font(.system(size: 12.5)).foregroundStyle(kind == .info ? StudioTheme.secondary : StudioTheme.ink)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12).padding(.vertical, 10)
        .background(color.opacity(kind == .info ? 0.06 : 0.1), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
    }
}

/// A state with a dot: "Online", "Loaded", "Stopped".
struct StudioStatusPill: View {
    let text: String
    var color: Color = StudioTheme.quiet
    var busy = false

    var body: some View {
        HStack(spacing: 7) {
            if busy {
                ProgressView().controlSize(.mini)
            } else {
                Circle().fill(color).frame(width: 7, height: 7).accessibilityHidden(true)
            }
            Text(text).font(.system(size: 12, weight: .medium)).foregroundStyle(StudioTheme.secondary)
        }
        .padding(.horizontal, 10).frame(height: 24)
        .background(StudioTheme.contrast.opacity(0.05), in: Capsule())
        .accessibilityElement(children: .combine)
    }
}

/// Nothing here yet: what this place is for and the one thing to do next.
struct StudioEmptyState<Actions: View>: View {
    let icon: String
    let title: String
    var detail: String? = nil
    @ViewBuilder var actions: Actions

    var body: some View {
        VStack(spacing: 0) {
            Image(systemName: icon).font(.system(size: 19, weight: .regular))
                .foregroundStyle(StudioTheme.secondary)
                .frame(width: 44, height: 44)
                .background(StudioTheme.contrast.opacity(0.055), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                .accessibilityHidden(true)
            Text(title).font(.system(size: 15, weight: .semibold)).foregroundStyle(StudioTheme.ink)
                .multilineTextAlignment(.center)
                .padding(.top, 14)
            if let detail {
                Text(detail).font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
                    .multilineTextAlignment(.center)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: 360)
                    .padding(.top, 5)
            }
            HStack(spacing: 8) { actions }.padding(.top, 16)
        }
        .padding(.horizontal, 24).padding(.vertical, 34)
        .frame(maxWidth: .infinity)
    }
}

extension StudioEmptyState where Actions == EmptyView {
    init(icon: String, title: String, detail: String? = nil) {
        self.init(icon: icon, title: title, detail: detail, actions: { EmptyView() })
    }
}

/// A value to copy (URL, key, command): shown in full or masked, with a button that
/// confirms the copy for a moment.
struct StudioCopyField: View {
    let label: String
    let value: String
    var secret = false
    /// What to print instead of `value` (a command with its key masked); the copy stays `value`.
    var shown: String? = nil
    @State private var copied = false
    @State private var revealed = false

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 4) {
                Text(label).font(.system(size: 11.5, weight: .medium)).foregroundStyle(StudioTheme.secondary)
                Text(secret && !revealed ? String(repeating: "•", count: min(28, max(8, value.count))) : shown ?? value)
                    .font(.system(size: 12.5, design: .monospaced))
                    .foregroundStyle(StudioTheme.ink)
                    .lineLimit(secret ? 1 : 3).truncationMode(.middle)
                    .textSelection(.enabled)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if secret {
                StudioIconButton(icon: revealed ? "eye.slash" : "eye",
                                 label: revealed ? L("Masquer", "Hide") : L("Afficher", "Show")) { revealed.toggle() }
            }
            Button {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(value, forType: .string)
                copied = true
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(1.4))
                    copied = false
                }
            } label: {
                Label(copied ? L("Copié", "Copied") : L("Copier", "Copy"), systemImage: copied ? "checkmark" : "doc.on.doc")
            }
            .buttonStyle(.studio)
            .accessibilityLabel(L("Copier ", "Copy ") + label)
        }
    }
}
