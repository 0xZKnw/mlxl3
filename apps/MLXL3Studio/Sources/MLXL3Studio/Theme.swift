import SwiftUI

enum StudioTheme {
    static let canvas = Color(red: 0.070, green: 0.072, blue: 0.072)
    static let sidebar = Color(red: 0.046, green: 0.048, blue: 0.049)
    static let panel = Color.white.opacity(0.035)
    static let edge = Color.white.opacity(0.085)
    static let ink = Color(red: 0.91, green: 0.90, blue: 0.87)
    static let quiet = Color.white.opacity(0.48)
    static let secondary = Color.white.opacity(0.68)
    static let accent = Color(red: 0.84, green: 0.87, blue: 0.86)
    static let thinking = Color.white.opacity(0.68)
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
                .background(Color.white.opacity(configuration.isPressed ? 0.11 : (hovered ? 0.075 : (emphasized ? 0.055 : 0))),
                            in: RoundedRectangle(cornerRadius: 6))
                .contentShape(RoundedRectangle(cornerRadius: 6))
                .opacity(enabled ? 1 : 0.32)
                .onHover { hovered = $0 }
        }
    }
}

/// Monochrome signature shared by the interface and menu bar.
struct MonogramMark: View {
    var size: CGFloat = 20

    var body: some View {
        ZStack {
            Circle().trim(from: 0.08, to: 0.78)
                .stroke(StudioTheme.ink, style: StrokeStyle(lineWidth: size * 0.1, lineCap: .round))
                .rotationEffect(.degrees(-38))
                .padding(size * 0.1)
            Circle().fill(StudioTheme.ink).frame(width: size * 0.17, height: size * 0.17)
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
                                .init(color: Color.white.opacity(0.14), location: 0),
                                .init(color: Color.white.opacity(0.055), location: 0.42),
                                .init(color: Color.white.opacity(0.10), location: 1),
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
    func premiumGlass(radius: CGFloat = 18, tint: Color = Color.white.opacity(0.045)) -> some View {
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
                .foregroundStyle(Color.black.opacity(0.9))
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
            .foregroundStyle(Color.black.opacity(0.88))
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
