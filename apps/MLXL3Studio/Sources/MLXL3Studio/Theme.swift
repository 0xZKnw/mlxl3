import SwiftUI

enum StudioTheme {
    static let canvas = Color(red: 0.018, green: 0.020, blue: 0.026)
    static let sidebar = Color(red: 0.027, green: 0.030, blue: 0.038)
    static let panel = Color.white.opacity(0.035)
    static let edge = Color.white.opacity(0.105)
    static let quiet = Color.white.opacity(0.48)
    static let secondary = Color.white.opacity(0.68)
    static let accent = Color(red: 0.76, green: 0.90, blue: 1.0)
    static let thinking = Color.white.opacity(0.68)
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
                                .init(color: Color.white.opacity(0.24), location: 0),
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
            .overlay(alignment: .top) {
                LinearGradient(
                    colors: [.clear, Color.white.opacity(0.26), .clear],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .frame(height: 0.7)
                .padding(.horizontal, radius)
                .allowsHitTesting(false)
            }
            .shadow(color: Color.black.opacity(0.34), radius: 18, y: 9)
    }
}

extension View {
    func premiumGlass(radius: CGFloat = 18, tint: Color = Color.white.opacity(0.045)) -> some View {
        modifier(PremiumGlass(radius: radius, tint: tint))
    }
}

struct RoundGlassButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var bright = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(bright ? Color.black : Color.white.opacity(0.88))
            .background(
                configuration.isPressed
                    ? Color.white.opacity(bright ? 0.72 : 0.12)
                    : Color.black.opacity(bright ? 0.04 : 0.16),
                in: Circle()
            )
            .glassEffect(
                (bright ? Glass.regular : Glass.clear)
                    .tint(bright ? Color.white : Color.white.opacity(0.035))
                    .interactive(isEnabled),
                in: Circle()
            )
            .overlay {
                Circle()
                    .stroke(
                        LinearGradient(
                            colors: [Color.white.opacity(0.28), Color.white.opacity(0.06)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: 0.65
                    )
            }
            .shadow(color: Color.black.opacity(0.32), radius: 9, y: 4)
            .scaleEffect(configuration.isPressed ? 0.94 : 1)
            .opacity(isEnabled ? 1 : 0.32)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct GlassPillButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color.white.opacity(0.9))
            .background(
                Color.black.opacity(configuration.isPressed ? 0.05 : 0.18),
                in: Capsule()
            )
            .glassEffect(
                .clear.tint(Color.white.opacity(configuration.isPressed ? 0.07 : 0.03))
                    .interactive(isEnabled),
                in: Capsule()
            )
            .overlay {
                Capsule()
                    .stroke(
                        LinearGradient(
                            colors: [Color.white.opacity(0.23), Color.white.opacity(0.055)],
                            startPoint: .top,
                            endPoint: .bottom
                        ),
                        lineWidth: 0.65
                    )
            }
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .opacity(isEnabled ? 1 : 0.38)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct PrimaryGlassButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(Color.black.opacity(0.88))
            .background(Color.white.opacity(configuration.isPressed ? 0.64 : 0.82), in: Capsule())
            .glassEffect(
                .regular.tint(Color.white).interactive(isEnabled),
                in: Capsule()
            )
            .overlay {
                Capsule()
                    .stroke(Color.white.opacity(0.68), lineWidth: 0.7)
            }
            .shadow(color: Color.white.opacity(0.10), radius: 11, y: 2)
            .scaleEffect(configuration.isPressed ? 0.985 : 1)
            .opacity(isEnabled ? 1 : 0.35)
            .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
    }
}

struct LogoMark: View {
    var size: CGFloat = 32

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: size * 0.31, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [Color.white, Color(red: 0.55, green: 0.72, blue: 0.88)],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            Circle()
                .trim(from: 0.08, to: 0.78)
                .stroke(Color.black.opacity(0.9), style: StrokeStyle(lineWidth: size * 0.105, lineCap: .round))
                .rotationEffect(.degrees(-38))
                .padding(size * 0.22)
            Circle()
                .fill(StudioTheme.canvas)
                .frame(width: size * 0.16, height: size * 0.16)
        }
        .frame(width: size, height: size)
        .shadow(color: Color.white.opacity(0.16), radius: 12, y: 2)
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
            .shadow(color: color.opacity(0.8), radius: 5)
    }
}
