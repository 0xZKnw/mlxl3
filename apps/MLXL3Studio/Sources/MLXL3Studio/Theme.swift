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
        theme == .midnightBlue ? Color(red: 0.10, green: 0.145, blue: 0.205) : canvas
    }
    static func sidebar(for theme: AppTheme) -> Color {
        theme == .midnightBlue ? Color(red: 0.065, green: 0.095, blue: 0.14) : sidebar
    }
    static func composer(for theme: AppTheme) -> Color {
        theme == .midnightBlue ? Color(red: 0.13, green: 0.19, blue: 0.27) : composer
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
    func studioAppearance(_ theme: AppTheme) -> some View {
        environment(\.studioAppTheme, theme)
            .preferredColorScheme(theme.colorScheme)
            .tint(theme == .midnightBlue ? StudioTheme.blueAccent : nil)
    }
}

/// A single, restrained interaction surface for navigation and toolbars.
struct StudioControlStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var emphasized = false

    func makeBody(configuration: Configuration) -> some View {
        ControlSurface(configuration: configuration, emphasized: emphasized, enabled: isEnabled)
    }

    private struct ControlSurface: View {
        let configuration: ButtonStyle.Configuration
        let emphasized: Bool
        let enabled: Bool
        @State private var hovered = false

        var body: some View {
            configuration.label
                .foregroundStyle(emphasized ? StudioTheme.ink : StudioTheme.secondary)
                .background(StudioTheme.contrast.opacity(configuration.isPressed ? 0.11 : (hovered ? 0.075 : (emphasized ? 0.055 : 0))),
                            in: RoundedRectangle(cornerRadius: 6))
                .contentShape(RoundedRectangle(cornerRadius: 6))
                .opacity(enabled ? 1 : 0.32)
                .onHover { hovered = $0 }
        }
    }
}

@MainActor
enum StudioBrand {
    static let logoURL: URL? = {
        let source = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appending(path: "Resources/AppIcon.png")
        return [Bundle.main.url(forResource: "AppIcon", withExtension: "icns"), source]
            .compactMap { $0 }.first { FileManager.default.fileExists(atPath: $0.path) }
    }()
    static let logo: NSImage? = {
        guard let original = logoURL.flatMap({ NSImage(contentsOf: $0) }) else { return nil }
        let renderer = ImageRenderer(content: LogoArtwork(image: original))
        renderer.scale = 2
        return renderer.nsImage
    }()
}

private struct LogoArtwork: View {
    let image: NSImage
    var body: some View {
        Image(nsImage: image).resizable().scaledToFit()
            .mask {
                Image(nsImage: image).resizable().scaledToFit()
                    .brightness(0.30).contrast(16).luminanceToAlpha()
            }
            .frame(width: 256, height: 256)
    }
}

/// Keep the real artwork and mask its dark icon background in the view.
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
    let radius: CGFloat
    let tint: Color

    func body(content: Content) -> some View {
        content
            .background(
                Color.black.opacity(0.18),
                in: RoundedRectangle(cornerRadius: radius, style: .continuous)
            )
            .glassEffect(
                .clear.tint(tint),
                in: RoundedRectangle(cornerRadius: radius, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: radius, style: .continuous)
                    .stroke(
                        LinearGradient(
                            stops: [
                                .init(color: StudioTheme.contrast.opacity(0.14), location: 0),
                                .init(color: StudioTheme.contrast.opacity(0.055), location: 0.42),
                                .init(color: StudioTheme.contrast.opacity(0.10), location: 1),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.7
                    )
                    .allowsHitTesting(false)
            }
            .shadow(color: Color.black.opacity(0.18), radius: 12, y: 4)
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
                .glassEffect(.clear.interactive(isEnabled), in: RoundedRectangle(cornerRadius: 8))
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
