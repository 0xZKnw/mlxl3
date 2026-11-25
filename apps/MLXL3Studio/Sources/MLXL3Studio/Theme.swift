import SwiftUI

/// Design tokens. Each one resolves for light/dark and for the visual style:
/// the Aurora styles keep their neutral greys; Graphite uses warm neutrals taken
/// from the logo's off-white, never pure black or white, with depth from
/// luminance steps (sidebar < canvas < raised) instead of glows.
enum StudioTheme {
    /// Read when a colour resolves. Set by `StudioModel.setStyle`; the window root is
    /// re-identified on change so every view resolves again.
    nonisolated(unsafe) static var style = AppStyle.auroraASCII

    static func adaptiveColor(light: NSColor, dark: NSColor) -> NSColor {
        NSColor(name: nil) { appearance in
            appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua ? dark : light
        }
    }
    static func rgb(_ value: UInt32) -> NSColor {
        NSColor(srgbRed: CGFloat(value >> 16) / 255, green: CGFloat((value >> 8) & 255) / 255,
                blue: CGFloat(value & 255) / 255, alpha: 1)
    }
    /// Aurora greys (light, dark white levels) beside Graphite sRGB values.
    private static func token(_ light: Double, _ dark: Double, graphite: (UInt32, UInt32), alpha: CGFloat = 1) -> Color {
        let classic = (NSColor(white: light, alpha: alpha), NSColor(white: dark, alpha: alpha))
        let warm = (rgb(graphite.0).withAlphaComponent(alpha), rgb(graphite.1).withAlphaComponent(alpha))
        return Color(nsColor: NSColor(name: nil) { appearance in
            let dark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            let pair = style.isGraphite ? warm : classic
            return dark ? pair.1 : pair.0
        })
    }
    /// Warm white/black at an alpha; Graphite may differ per scheme to hold 4.5:1.
    private static func contrastToken(classic: CGFloat, graphite: (light: CGFloat, dark: CGFloat)) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let dark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            if style.isGraphite { return rgb(dark ? 0xF4EFE7 : 0x1A1816).withAlphaComponent(dark ? graphite.dark : graphite.light) }
            return NSColor(white: dark ? 1 : 0, alpha: classic)
        })
    }
    static let canvas = token(0.97, 0.071, graphite: (0xF9F8F6, 0x1B1A19))
    /// Graphite: a few notches dimmer than the canvas so the conversation reads as the focus.
    static let sidebar = token(0.93, 0.047, graphite: (0xF1EFEC, 0x151413))
    /// The single raised surface (composer, popovers).
    static let composer = token(1, 0.107, graphite: (0xFFFFFF, 0x252321))
    /// Every translucent fill and hairline derives from it (warm in Graphite).
    static let contrast = token(0, 1, graphite: (0x1A1816, 0xF4EFE7))
    static let onAccent = token(1, 0, graphite: (0xFFFFFF, 0x161514))
    static let panel = contrastToken(classic: 0.035, graphite: (0.04, 0.04))
    static let edge = contrastToken(classic: 0.085, graphite: (0.085, 0.085))
    /// Graphite text tops out below pure white in dark mode (APCA Lc ≈ 90) to avoid halation.
    static let ink = token(0.10, 0.91, graphite: (0x1C1B19, 0xEAE6DF))
    /// Metadata text: at least 4.5:1 on the canvas in both schemes.
    static let quiet = contrastToken(classic: 0.58, graphite: (0.62, 0.56))
    static let secondary = contrastToken(classic: 0.72, graphite: (0.74, 0.72))
    static let accent = token(0.18, 0.86, graphite: (0x2B2926, 0xDDD8D0))
    static let thinking = secondary
    /// Something is running right now: the accent grey in Aurora, one warm signal in Graphite.
    static let live = Color(nsColor: NSColor(name: nil) { appearance in
        let dark = appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
        if style.isGraphite { return rgb(dark ? 0xE3A25F : 0xB4661C) }
        return NSColor(white: dark ? 0.86 : 0.18, alpha: 1)
    })
    /// A failed step: muted red, at least 4.5:1 on the canvas in both schemes.
    static let failure = Color(nsColor: NSColor(name: nil) { appearance in
        appearance.bestMatch(from: [.aqua, .darkAqua]) == .darkAqua
            ? rgb(0xEC7A72) : rgb(0xB8342C)
    })

    /// Display titles: serif in the Aurora styles, SF Pro semibold in Graphite
    /// (about three quarters of the serif size, which reads larger).
    static func display(_ size: CGFloat) -> Font {
        style.isGraphite ? .system(size: (size * 0.68).rounded(), weight: .semibold)
            : .system(size: size, weight: .regular, design: .serif)
    }
    static func displayTracking(_ tracking: CGFloat) -> CGFloat {
        style.isGraphite ? tracking * 0.4 : tracking
    }
}

extension View {
    func studioAppearance(_ theme: AppTheme, accent: AppAccent = .violet, customAccent: Color = AppAccent.custom.color) -> some View {
        preferredColorScheme(theme.colorScheme)
            .tint(accent == .custom ? customAccent : accent.color)
    }

    /// Theme, accent and style together; popovers and extra windows do not
    /// inherit the root environment, so each one applies this again.
    func studioAppearance(_ studio: StudioModel) -> some View {
        studioAppearance(studio.theme, accent: studio.accent, customAccent: studio.customAccentColor)
            .environment(\.studioStyle, studio.style)
    }
}

/// A section caption: spaced capitals in the Aurora styles, sentence case in Graphite
/// (tracked capitals read as template chrome). `colored: false` keeps the caller's colour.
struct StudioCaption: View {
    let french: String
    let english: String
    var size: CGFloat = 12
    var tracking: CGFloat = 1.4
    var colored = true

    var body: some View {
        let text = L(french, english)
        if StudioTheme.style.isGraphite {
            Text(text).font(.system(size: max(11, size + 1), weight: .semibold))
                .foregroundStyle(colored ? AnyShapeStyle(StudioTheme.secondary) : AnyShapeStyle(.foreground))
        } else {
            Text(text.uppercased()).font(.system(size: size, weight: .medium)).tracking(tracking)
                .foregroundStyle(colored ? AnyShapeStyle(StudioTheme.quiet) : AnyShapeStyle(.foreground))
        }
    }
}

/// Live status text. While `active`, a soft highlight sweeps across the words: the label says
/// what is running, the sweep says it is still running. It replaces pulsing dots, which
/// carried no information and kept blinking beside text that already moved.
/// Reduce Motion, or `active == false`, shows the plain label in `color`.
struct StudioShimmerText: View {
    let text: String
    var font: Font = .system(size: 13, weight: .medium)
    var active = true
    var color: Color = StudioTheme.secondary
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    /// Seconds for one sweep; the band rests outside the text for part of it.
    static let period = 2.2

    var body: some View {
        let label = Text(text).font(font)
        if active && !reduceMotion {
            TimelineView(.animation(minimumInterval: 1 / 30)) { context in
                label.foregroundStyle(Self.sweep(at: context.date))
            }
        } else {
            label.foregroundStyle(color)
        }
    }

    /// The highlight centre moves from beyond the leading edge to beyond the trailing one;
    /// outside its band the gradient pads with the resting colour.
    static func sweep(at date: Date) -> LinearGradient {
        let progress = date.timeIntervalSinceReferenceDate.truncatingRemainder(dividingBy: period) / period
        let centre = -0.45 + 1.9 * progress
        return LinearGradient(
            stops: [
                .init(color: StudioTheme.quiet, location: 0),
                .init(color: StudioTheme.ink, location: 0.5),
                .init(color: StudioTheme.quiet, location: 1),
            ],
            startPoint: UnitPoint(x: centre - 0.32, y: 0.5),
            endPoint: UnitPoint(x: centre + 0.32, y: 0.5))
    }
}

/// Mask for scrolling content: text fades out near the chrome instead of
/// stopping on a hard edge under the toolbar or above the composer.
struct StudioEdgeFade: View {
    var top: CGFloat = 0
    var bottom: CGFloat = 0

    var body: some View {
        VStack(spacing: 0) {
            LinearGradient(colors: [.clear, .black], startPoint: .top, endPoint: .bottom)
                .frame(height: max(0, top))
            Color.black
            LinearGradient(colors: [.black, .clear], startPoint: .top, endPoint: .bottom)
                .frame(height: max(0, bottom))
        }
        .allowsHitTesting(false)
    }
}

struct StudioSidebarSurface: ViewModifier {
    @Environment(\.studioStyle) private var style

    func body(content: Content) -> some View {
        content
            .background {
                let base = StudioTheme.sidebar
                if style.isGraphite {
                    base
                } else {
                    // The light shows through the history.
                    LinearGradient(colors: [base.opacity(0.20), base.opacity(0.78)],
                                   startPoint: .top, endPoint: .center)
                }
            }
            .overlay(alignment: .trailing) {
                Rectangle().fill(StudioTheme.edge.opacity(style.isGraphite ? 0.7 : 1))
                    .frame(width: style.isGraphite ? 0.5 : 1).allowsHitTesting(false)
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
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    let radius: CGFloat
    let tint: Color
    var materialOpacity = 1.0

    func body(content: Content) -> some View {
        content.background {
            let shape = RoundedRectangle(cornerRadius: radius, style: .continuous)
            if reduceTransparency {
                shape.fill(StudioTheme.composer)
            } else {
                shape.fill(.ultraThinMaterial).opacity(materialOpacity)
                    .overlay { shape.fill(tint).allowsHitTesting(false) }
            }
        }
    }
}

extension View {
    func premiumGlass(radius: CGFloat = 18, tint: Color = StudioTheme.contrast.opacity(0.045), materialOpacity: Double = 1) -> some View {
        modifier(PremiumGlass(radius: radius, tint: tint, materialOpacity: materialOpacity))
    }
}

// Compatibility names keep auxiliary panels on the same design system.
struct RoundGlassButtonStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var bright = false

    @ViewBuilder func makeBody(configuration: Configuration) -> some View {
        if bright {
            // Primary round action (send/stop): a solid disc with a soft lift.
            configuration.label
                .foregroundStyle(StudioTheme.onAccent.opacity(0.92))
                .background {
                    Circle().fill(StudioTheme.ink.opacity(configuration.isPressed ? 0.72 : 1))
                        .shadow(color: .black.opacity(isEnabled ? 0.18 : 0), radius: 4, y: 2)
                }
                .contentShape(Circle())
                .scaleEffect(configuration.isPressed ? 0.94 : 1)
                .opacity(isEnabled ? 1 : 0.3)
                .animation(.easeOut(duration: 0.12), value: configuration.isPressed)
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
        if StudioTheme.style.isGraphite {
            // Muted signals: the dot informs, it does not glow.
            switch state {
            case .ready: return Color(red: 0.42, green: 0.72, blue: 0.52)
            case .generating: return StudioTheme.live
            case .loading: return StudioTheme.live.opacity(0.7)
            case .failed: return Color(red: 0.90, green: 0.42, blue: 0.40)
            case .idle: return StudioTheme.quiet
            }
        }
        switch state {
        case .ready: return Color(red: 0.42, green: 1.0, blue: 0.69)
        case .generating: return StudioTheme.accent
        case .loading: return Color(red: 1.0, green: 0.79, blue: 0.38)
        case .failed: return Color(red: 1.0, green: 0.36, blue: 0.39)
        case .idle: return StudioTheme.quiet
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
    var materialOpacity = 1.0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.studioStyle) private var style

    func makeBody(configuration: Configuration) -> some View {
        StudioControlStyle(emphasized: true, cornerRadius: 9).makeBody(configuration: configuration)
            .background {
                // Glass needs something behind it: Graphite has a plain canvas.
                if style.isGraphite {
                    RoundedRectangle(cornerRadius: 9, style: .continuous).fill(StudioTheme.contrast.opacity(0.05))
                } else {
                    Color.clear.premiumGlass(radius: 9, materialOpacity: materialOpacity)
                }
            }
            .opacity(configuration.isPressed ? 0.75 : 1)
            .scaleEffect(configuration.isPressed && !reduceMotion ? 0.97 : 1)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.14), value: configuration.isPressed)
    }
}

/// Back as the sidebar's bottom row, where the main sidebar keeps the button that opened
/// the page: the pointer is already there. Escape does the same.
struct StudioBackButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 9) {
                Image(systemName: "chevron.left")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 18)
                    .foregroundStyle(StudioTheme.quiet)
                Text(L("Retour", "Back"))
                    .font(.system(size: 13))
                    .foregroundStyle(StudioTheme.secondary)
                Spacer(minLength: 4)
                Text("esc")
                    .font(.system(size: 11))
                    .foregroundStyle(StudioTheme.quiet)
                    .accessibilityHidden(true)
            }
            .padding(.leading, 9).padding(.trailing, StudioPageMetrics.content)
            .frame(height: 30)
            .contentShape(Rectangle())
        }
        .buttonStyle(StudioRowStyle())
        .keyboardShortcut(.cancelAction)
        .help(L("Revenir à l’écran précédent · Échap", "Return to the previous screen · Escape"))
        .accessibilityLabel(L("Retour", "Back"))
    }
}

/// A page's name at the top of its sidebar, on the main sidebar's left edge (x = 18).
struct StudioSidebarHeader: View {
    @Environment(\.studioStyle) private var style
    let title: String
    var body: some View {
        let graphite = style.isGraphite
        HStack(spacing: 8) {
            MonogramMark(size: graphite ? 24 : 32)
            Text(title).font(.system(size: graphite ? 15 : 17, weight: .semibold)).tracking(graphite ? -0.2 : -0.5)
                .lineLimit(1)
                .accessibilityAddTraits(.isHeader)
        }
        .padding(.horizontal, 17).padding(.top, graphite ? 42 : 38).padding(.bottom, 16)
    }
}

struct StudioPanelHeader: View {
    let title: String
    let detail: String

    var body: some View {
        StudioPageHeader(title: title, detail: detail)
    }
}
