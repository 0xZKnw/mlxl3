import SwiftUI

enum StudioTheme {
    static func adaptiveColor(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
        }
    }
    private static func adaptive(light: Double, dark: Double) -> Color {
        Color(nsColor: adaptiveColor(light: NSColor(white: light, alpha: 1),
                                     dark: NSColor(white: dark, alpha: 1)))
    }
    static let canvas = adaptive(light: 0.97, dark: 0.071)
    static let sidebar = adaptive(light: 0.93, dark: 0.047)
    static let composer = adaptive(light: 1, dark: 0.107)
    static let contrast = adaptive(light: 0, dark: 1)
    static let onAccent = adaptive(light: 1, dark: 0)
    static let panel = contrast.opacity(0.035)
    static let edge = contrast.opacity(0.085)
    static let ink = adaptive(light: 0.10, dark: 0.91)
    static let quiet = contrast.opacity(0.58)
    static let secondary = contrast.opacity(0.72)
    static let accent = adaptive(light: 0.18, dark: 0.86)
    static let thinking = secondary

    static let blueAccent = Color(red: 0.31, green: 0.66, blue: 1)
    static func canvas(for theme: AppTheme) -> Color {
        switch theme {
        case .midnightBlue: Color(red: 0.10, green: 0.145, blue: 0.205)
        case .graphite: Color(red: 0.14, green: 0.15, blue: 0.17)
        case .forest: Color(red: 0.08, green: 0.14, blue: 0.12)
        case .plum: Color(red: 0.15, green: 0.10, blue: 0.16)
        case .paper: Color(red: 0.97, green: 0.95, blue: 0.90)
        case .ocean: Color(red: 0.07, green: 0.15, blue: 0.17)
        case .coffee: Color(red: 0.17, green: 0.12, blue: 0.10)
        case .sand: Color(red: 0.97, green: 0.94, blue: 0.90)
        case .lavender: Color(red: 0.96, green: 0.94, blue: 0.98)
        default: canvas
        }
    }
    static func sidebar(for theme: AppTheme) -> Color {
        switch theme {
        case .midnightBlue: Color(red: 0.065, green: 0.095, blue: 0.14)
        case .graphite: Color(red: 0.10, green: 0.11, blue: 0.13)
        case .forest: Color(red: 0.05, green: 0.10, blue: 0.08)
        case .plum: Color(red: 0.10, green: 0.065, blue: 0.11)
        case .paper: Color(red: 0.92, green: 0.89, blue: 0.82)
        case .ocean: Color(red: 0.045, green: 0.10, blue: 0.12)
        case .coffee: Color(red: 0.12, green: 0.08, blue: 0.065)
        case .sand: Color(red: 0.92, green: 0.88, blue: 0.82)
        case .lavender: Color(red: 0.91, green: 0.88, blue: 0.95)
        default: sidebar
        }
    }
    static func composer(for theme: AppTheme) -> Color {
        switch theme {
        case .midnightBlue: Color(red: 0.13, green: 0.19, blue: 0.27)
        case .graphite: Color(red: 0.19, green: 0.20, blue: 0.23)
        case .forest: Color(red: 0.12, green: 0.19, blue: 0.16)
        case .plum: Color(red: 0.21, green: 0.15, blue: 0.22)
        case .paper: Color(red: 1, green: 0.99, blue: 0.96)
        case .ocean: Color(red: 0.11, green: 0.20, blue: 0.22)
        case .coffee: Color(red: 0.23, green: 0.17, blue: 0.14)
        case .sand: Color(red: 1, green: 0.98, blue: 0.95)
        case .lavender: Color(red: 0.99, green: 0.98, blue: 1)
        default: composer
        }
    }

}

private struct StudioAppThemeKey: EnvironmentKey {
    static let defaultValue: AppTheme = .dark
}

extension EnvironmentValues {
    var studioAppTheme: AppTheme {
        get { self[StudioAppThemeKey.self] }
        set { self[StudioAppThemeKey.self] = newValue }
    }
}

extension View {
    func studioAppearance(_ theme: AppTheme, accent: AppAccent = .neutral, customAccent: Color = .blue) -> some View {
        environment(\.studioAppTheme, theme)
            .preferredColorScheme(theme.colorScheme)
            .tint(accent == .custom ? customAccent : accent == .neutral && theme == .midnightBlue ? StudioTheme.blueAccent : accent.color)
    }
}

struct StudioSidebarSurface: ViewModifier {
    @Environment(\.studioAppTheme) private var theme

    func body(content: Content) -> some View {
        content
            .background {
                let base = StudioTheme.sidebar(for: theme)
                LinearGradient(colors: [base.opacity(0.20), base.opacity(0.78)],
                               startPoint: .top, endPoint: .center)
            }
            .overlay(alignment: .trailing) {
                Rectangle().fill(StudioTheme.edge).frame(width: 1).allowsHitTesting(false)
            }
    }
}

struct StudioControlStyle: ButtonStyle {
    var emphasized = false
    var cornerRadius: CGFloat = 6

    func makeBody(configuration: Configuration) -> some View {
        ControlSurface(configuration: configuration, emphasized: emphasized, radius: cornerRadius)
    }

    private struct ControlSurface: View {
        let configuration: ButtonStyle.Configuration
        let emphasized: Bool
        let radius: CGFloat
        @Environment(\.isEnabled) private var enabled
        @Environment(\.accessibilityReduceMotion) private var reduceMotion
        @State private var hovered = false

        var body: some View {
            configuration.label
                .foregroundStyle(emphasized ? StudioTheme.ink : StudioTheme.secondary)
                .background(StudioTheme.contrast.opacity(!enabled ? 0 : configuration.isPressed ? 0.11 : (hovered ? 0.075 : (emphasized ? 0.055 : 0))),
                            in: RoundedRectangle(cornerRadius: radius))
                .contentShape(RoundedRectangle(cornerRadius: radius))
                .opacity(enabled ? 1 : 0.32)
                .onHover { hovered = $0 }
                .animation(reduceMotion ? nil : .easeOut(duration: 0.14), value: hovered)
        }
    }
}

/// Capsule feedback stays inside the shared toolbar glass.
struct WorkspaceControlStyle: ButtonStyle {
    @Environment(\.isEnabled) private var enabled
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    var selected = false

    func makeBody(configuration: Configuration) -> some View {
        Surface(configuration: configuration, selected: selected, enabled: enabled, reduceMotion: reduceMotion)
    }

    private struct Surface: View {
        let configuration: ButtonStyle.Configuration
        let selected: Bool
        let enabled: Bool
        let reduceMotion: Bool
        @State private var hovered = false

        var body: some View {
            configuration.label
                .foregroundStyle(StudioTheme.ink)
                .background(StudioTheme.contrast.opacity(
                    !enabled ? 0 : configuration.isPressed ? 0.14 : hovered ? 0.09 : selected ? 0.055 : 0
                ), in: Capsule())
                .contentShape(Capsule())
                .opacity(enabled ? 1 : 0.32)
                .onHover { hovered = $0 }
                .animation(reduceMotion ? nil : .easeOut(duration: 0.14), value: hovered)
                .animation(reduceMotion ? nil : .easeOut(duration: 0.10), value: configuration.isPressed)
        }
    }
}

@MainActor
enum StudioBrand {
    static let logoURL: URL? = {
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appending(path: "Resources/AppIcon.png")
        return [Bundle.main.url(forResource: "AppIcon", withExtension: "png"), source]
            .compactMap { $0 }.first { FileManager.default.fileExists(atPath: $0.path) }
    }()
    static let logo: NSImage? = {
        logoURL.flatMap { NSImage(contentsOf: $0) }
    }()
}

/// Keep the transparent artwork legible in both light and dark themes.
struct MonogramMark: View {
    @Environment(\.colorScheme) private var colorScheme
    var size: CGFloat = 20

    var body: some View {
        Group {
            if let logo = StudioBrand.logo {
                Image(nsImage: logo)
                    .renderingMode(colorScheme == .light ? .template : .original)
                    .resizable().scaledToFit()
                    .foregroundStyle(StudioTheme.ink)
            } else {
                Image(systemName: "cpu").resizable().scaledToFit().foregroundStyle(StudioTheme.ink)
            }
        }
        .frame(width: size, height: size)
        .accessibilityHidden(true)
    }
}

struct PremiumGlass: ViewModifier {
    @Environment(\.studioAppTheme) private var theme
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    let radius: CGFloat
    let tint: Color

    func body(content: Content) -> some View {
        content.background {
            let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
            if reduceTransparency {
                shape.fill(StudioTheme.composer(for: theme))
            } else {
                shape.fill(.ultraThinMaterial)
                    .overlay { shape.fill(tint).allowsHitTesting(false) }
            }
        }
    }
}

extension View {
    func premiumGlass(radius: CGFloat = 18, tint: Color = StudioTheme.contrast.opacity(0.045)) -> some View {
        modifier(PremiumGlass(radius: radius, tint: tint))
    }
}

// Compatibility names keep auxiliary panels on the same design system.
struct RoundGlassButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var bright = false

    @ViewBuilder func makeBody(configuration: Configuration) -> some View {
        if bright {
            configuration.label
                .foregroundStyle(StudioTheme.onAccent.opacity(0.9))
                .background(StudioTheme.ink.opacity(configuration.isPressed ? 0.7 : 1),
                            in: RoundedRectangle(cornerRadius: 8))
                .opacity(isEnabled ? 1 : 0.3)
        } else {
            StudioControlStyle().makeBody(configuration: configuration)
        }
    }
}

struct GlassPillButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        StudioControlStyle(emphasized: true).makeBody(configuration: configuration)
    }
}

struct PrimaryGlassButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(StudioTheme.onAccent.opacity(0.88))
            .background(StudioTheme.ink.opacity(configuration.isPressed ? 0.7 : 1),
                        in: RoundedRectangle(cornerRadius: 7))
            .opacity(isEnabled ? 1 : 0.35)
    }
}

struct StatusDot: View {
    let state: EngineState

    private var color: Color {
        switch state {
        case .ready: Color(red: 0.42, green: 1.0, blue: 0.69)
        case .generating: StudioTheme.accent
        case .loading: Color(red: 1.0, green: 0.79, blue: 0.38)
        case .failed: Color(red: 1.0, green: 0.36, blue: 0.39)
        case .idle: StudioTheme.quiet
        }
    }

    var body: some View {
        Circle()
            .fill(color)
            .frame(width: 7, height: 7)
            .accessibilityLabel(state.label)
    }
}

/// Explicit sizing avoids the system glass button's extra padding and accent fill.
struct StudioToolbarStyle: ButtonStyle {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    func makeBody(configuration: Configuration) -> some View {
        StudioControlStyle(emphasized: true, cornerRadius: 9).makeBody(configuration: configuration)
            .premiumGlass(radius: 9)
            .opacity(configuration.isPressed ? 0.75 : 1)
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.97 : 1)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.14), value: configuration.isPressed)
    }
}

struct StudioBackButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(L("Retour", "Back"), systemImage: "arrow.left")
                .font(.system(size: 13, weight: .medium))
                .padding(.horizontal, 12).frame(height: 36)
        }
        .buttonStyle(StudioToolbarStyle())
        .keyboardShortcut(.cancelAction)
        .help(L("Revenir à l’écran précédent · Échap", "Return to the previous screen · Escape"))
    }
}

struct StudioPanelHeader: View {
    let title: String
    let detail: String
    let close: () -> Void

    var body: some View {
        HStack(alignment: .top, spacing: 20) {
            VStack(alignment: .leading, spacing: 7) {
                Text(title).font(.system(size: 32, weight: .regular, design: .serif))
                    .contentTransition(.opacity)
                Text(detail).font(.system(size: 13)).foregroundStyle(StudioTheme.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
            StudioBackButton(action: close)
        }
        .padding(.horizontal, 28).padding(.top, 38).padding(.bottom, 22)
        .background(alignment: .top) { StudioAurora().scaleEffect(y: 2, anchor: .top).opacity(0.55) }
        .clipped()
    }
}
