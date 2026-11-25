import SwiftUI

struct AppearanceSettings: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            SettingCard(title: L("Style", "Style"), icon: "rectangle.stack") {
                Text(
                    L(
                        "Chaque style apporte ses couleurs, sa typographie et son fond. L’accent reste modifiable.",
                        "Each style brings its colours, type and background. The accent can still be changed.")
                )
                .font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 150), spacing: 10)], spacing: 10
                ) {
                    ForEach(AppStyle.allCases, id: \.self) { style in
                        Button {
                            studio.setStyle(style)
                        } label: {
                            StylePreview(style: style, selected: studio.style == style)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(style.title)
                        .accessibilityHint(style.detail)
                        .accessibilityIdentifier("settings.style.\(style.rawValue)")
                        .accessibilityAddTraits(studio.style == style ? .isSelected : [])
                    }
                }
                // A style change rebuilds the window; Agent Studio's open editor must not be rebuilt.
                .disabled(studio.studioDocument != nil)
                if studio.studioDocument != nil {
                    StudioNotice(text: L("Fermez Agent Studio pour changer de style.", "Close Agent Studio to change the style."))
                }
            }
            SettingCard(title: L("Thème de l’interface", "Interface theme"), icon: "circle.lefthalf.filled") {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 132), spacing: 10)], spacing: 10
                ) {
                    ForEach(AppTheme.allCases, id: \.self) { theme in
                        Button {
                            studio.setTheme(theme)
                        } label: {
                            ThemePreview(theme: theme, selected: studio.theme == theme)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(theme.title)
                        .accessibilityIdentifier("settings.theme.\(theme.rawValue)")
                        .accessibilityAddTraits(studio.theme == theme ? .isSelected : [])
                    }
                }
            }
            SettingSection(
                title: L("Couleur d’accent", "Accent color"), icon: "paintpalette",
                footer: L(
                    "Appliquée aux sélecteurs, aux interrupteurs et aux contrôles de l’app.",
                    "Applied to the app’s selectors, switches and controls.")
            ) {
                SettingRow(title: L("Accent", "Accent"), detail: studio.accent.title) {
                    HStack(spacing: 8) {
                        ForEach(AppAccent.allCases.filter { $0 != .custom }, id: \.self) { accent in
                            Button {
                                studio.setAccent(accent)
                            } label: {
                                AccentSwatch(color: accent.color, selected: studio.accent == accent)
                            }
                            .buttonStyle(.plain)
                            .help(accent.title)
                            .accessibilityLabel(accent.title)
                            .accessibilityAddTraits(studio.accent == accent ? .isSelected : [])
                        }
                    }
                }
                SettingRow(title: L("Couleur personnalisée", "Custom color"),
                           detail: studio.accent == .custom ? L("Utilisée comme accent.", "Used as the accent.") : nil) {
                    ColorPicker(L("Couleur personnalisée", "Custom color"), selection: Binding(
                        get: { studio.customAccentColor }, set: { studio.setCustomAccent($0) }
                    ), supportsOpacity: false)
                    .labelsHidden()
                    if studio.accent != .custom {
                        Button(L("Utiliser", "Use")) { studio.setAccent(.custom) }
                            .buttonStyle(.studio)
                    } else {
                        Image(systemName: "checkmark").font(.system(size: 12, weight: .semibold))
                            .foregroundStyle(StudioTheme.secondary)
                            .accessibilityLabel(L("Sélectionnée", "Selected"))
                    }
                }
            }
            SettingCard(title: L("Lecture & conversations", "Reading & conversations"), icon: "textformat.size") {
                ConversationDisplaySettings(showsTitle: false)
            }
            SettingSection(title: L("Messages", "Messages"), icon: "bubble.left") {
                SettingRow(title: L("Espacement des messages", "Message spacing")) {
                    Picker(
                        L("Espacement des messages", "Message spacing"),
                        selection: Binding(
                            get: { studio.conversationDensity }, set: { studio.setConversationDensity($0) }
                        )
                    ) {
                        ForEach(ConversationDensity.allCases, id: \.self) { density in
                            Text(density.title).tag(density)
                        }
                    }
                    .pickerStyle(.menu).labelsHidden().fixedSize()
                }
                SettingRow(title: L("Police des messages", "Message font")) {
                    Picker(L("Police des messages", "Message font"), selection: Binding(
                        get: { studio.conversationFont }, set: { studio.setConversationFont($0) }
                    )) {
                        ForEach(ConversationFont.allCases, id: \.self) { font in
                            Text(font.title).tag(font)
                        }
                    }
                    .labelsHidden().fixedSize()
                }
                SettingRow(title: L("Bulles de vos messages", "Your message bubbles")) {
                    Picker(L("Bulles de vos messages", "Your message bubbles"), selection: Binding(
                        get: { studio.messageBubbleStyle }, set: { studio.setMessageBubbleStyle($0) }
                    )) {
                        ForEach(MessageBubbleStyle.allCases, id: \.self) { style in
                            Text(style.title).tag(style)
                        }
                    }
                    .labelsHidden().fixedSize()
                }
            }
            SettingSection(title: L("Barre latérale", "Sidebar"), icon: "sidebar.left") {
                SettingRow(title: L("Largeur", "Width")) {
                    Picker(L("Largeur", "Width"), selection: Binding(
                        get: { studio.sidebarWidth }, set: { studio.setSidebarWidth($0) }
                    )) {
                        ForEach(SidebarWidth.allCases, id: \.self) { width in
                            Text(width.title).tag(width)
                        }
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize()
                }
            }
            Button(L("Rétablir l’apparence par défaut", "Restore default appearance"), action: studio.resetAppearance)
                .buttonStyle(.studio)
        }
    }
}

/// An accent as a disc; the selected one carries a ring of its own colour and a check.
private struct AccentSwatch: View {
    let color: Color
    let selected: Bool

    var body: some View {
        Circle().fill(color).frame(width: 22, height: 22)
            .overlay {
                if selected {
                    Image(systemName: "checkmark").font(.system(size: 10, weight: .bold))
                        .foregroundStyle(StudioTheme.onAccent)
                }
            }
            .padding(3)
            .overlay { Circle().strokeBorder(selected ? color : .clear, lineWidth: 1.5) }
            .contentShape(Circle())
    }
}

/// Explicit swatches: a thumbnail must show its own theme and style whatever the
/// app's current appearance (dynamic colours resolve against the window).
private struct PreviewPalette {
    let canvas, sidebar, raised, ink, line: Color

    static func make(dark: Bool, graphite: Bool) -> PreviewPalette {
        func c(_ value: UInt32) -> Color { Color(nsColor: StudioTheme.rgb(value)) }
        func w(_ white: Double) -> Color { Color(white: white) }
        switch (graphite, dark) {
        case (true, true):
            return .init(canvas: c(0x1B1A19), sidebar: c(0x151413), raised: c(0x252321), ink: c(0xEAE6DF), line: c(0x34322F))
        case (true, false):
            return .init(canvas: c(0xF9F8F6), sidebar: c(0xF1EFEC), raised: c(0xFFFFFF), ink: c(0x1C1B19), line: c(0xE2DFDA))
        case (false, true):
            return .init(canvas: w(0.071), sidebar: w(0.047), raised: w(0.107), ink: w(0.91), line: w(0.18))
        case (false, false):
            return .init(canvas: w(0.97), sidebar: w(0.93), raised: w(1), ink: w(0.10), line: w(0.86))
        }
    }
}

/// A miniature window: sidebar, title, composer, and the style's background.
private struct MiniWindow: View {
    let palette: PreviewPalette
    var serif = false
    var background: AnyView? = nil

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 4) {
                RoundedRectangle(cornerRadius: 1.5).fill(palette.ink).frame(width: 9, height: 6)
                Capsule().fill(palette.ink.opacity(0.35)).frame(width: 12, height: 2)
                Capsule().fill(palette.ink.opacity(0.35)).frame(width: 16, height: 2)
                Spacer()
            }
            .padding(7).frame(width: 30, alignment: .leading)
            .background(palette.sidebar)
            VStack(spacing: 5) {
                Spacer()
                Text("Aa").font(serif ? .system(size: 18, design: .serif) : .system(size: 16, weight: .semibold))
                    .foregroundStyle(palette.ink)
                RoundedRectangle(cornerRadius: 4)
                    .fill(palette.raised)
                    .overlay { RoundedRectangle(cornerRadius: 4).strokeBorder(palette.line, lineWidth: 0.5) }
                    .frame(height: 11)
                Spacer(minLength: 0)
            }
            .padding(.horizontal, 7).frame(maxWidth: .infinity)
            // Clip the column itself: an oversized background is centred on it and
            // would otherwise spill over the sidebar.
            .background { ZStack { palette.canvas; background } }
            .clipped()
        }
    }
}

/// Rows of the aurora's glyphs, for the thumbnails.
private struct GlyphRows: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(0..<9, id: \.self) { row in
                Text(String(repeating: [":-=+xX+=-:", "-=+xXx+=-:", ".:-=+x+=-:"][row % 3], count: 6))
                    .font(.system(size: 6, design: .monospaced)).lineLimit(1).fixedSize()
            }
        }
    }
}

private struct StylePreview: View {
    @Environment(\.colorScheme) private var colorScheme
    let style: AppStyle
    let selected: Bool

    private var background: AnyView? {
        let dark = colorScheme == .dark
        switch style {
        case .graphite:
            return AnyView(GlyphRows().foregroundStyle(PreviewPalette.make(dark: dark, graphite: true).ink.opacity(0.28))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomLeading)
                .mask(LinearGradient(colors: [.clear, .black], startPoint: .center, endPoint: .bottom)).clipped())
        case .minimal:
            return nil
        case .auroraASCII:
            return AnyView(LinearGradient(colors: [Color(red: 0.55, green: 0.35, blue: 0.95), Color(red: 0.93, green: 0.42, blue: 0.70),
                                                   Color(red: 0.98, green: 0.62, blue: 0.35)], startPoint: .top, endPoint: .bottom)
                .mask(GlyphRows().frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing))
                .opacity(dark ? 0.75 : 0.5).clipped())
        case .aurora:
            return AnyView(ZStack(alignment: .top) {
                EllipticalGradient(colors: [Color(red: 0.47, green: 0.55, blue: 0.95).opacity(0.8), .clear])
                    .frame(width: 80, height: 40).offset(x: -30, y: -12)
                EllipticalGradient(colors: [Color(red: 0.85, green: 0.45, blue: 0.70).opacity(0.75), .clear])
                    .frame(width: 80, height: 40).offset(x: 10, y: -10)
                EllipticalGradient(colors: [Color(red: 0.98, green: 0.62, blue: 0.38).opacity(0.7), .clear])
                    .frame(width: 70, height: 36).offset(x: 45, y: -12)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .opacity(dark ? 0.8 : 0.9).clipped())
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            MiniWindow(palette: .make(dark: colorScheme == .dark, graphite: style.isGraphite),
                       serif: !style.isGraphite, background: background)
                .frame(height: 68)
                .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
                .overlay { RoundedRectangle(cornerRadius: 7, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5) }
                .accessibilityHidden(true)
            HStack {
                Text(style.title).font(.system(size: 13, weight: selected ? .semibold : .regular))
                Spacer()
                if selected { Image(systemName: "checkmark.circle.fill") }
            }
            .foregroundStyle(StudioTheme.ink)
            Text(style.detail).font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                .lineLimit(3).fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(9)
        .frame(maxHeight: .infinity, alignment: .top)
        .overlay(
            RoundedRectangle(cornerRadius: 11, style: .continuous).strokeBorder(
                selected ? StudioTheme.ink.opacity(0.85) : StudioTheme.edge,
                lineWidth: selected ? 1.5 : 0.5))
        .background(StudioTheme.contrast.opacity(selected ? 0.04 : 0), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
        .contentShape(RoundedRectangle(cornerRadius: 12))
    }
}

private struct ThemePreview: View {
    @Environment(\.studioStyle) private var style
    let theme: AppTheme
    let selected: Bool

    private func window(dark: Bool) -> some View {
        MiniWindow(palette: .make(dark: dark, graphite: style.isGraphite), serif: !style.isGraphite)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Group {
                if theme == .system {
                    // Light on the left, dark on the right, cut on a diagonal.
                    ZStack {
                        window(dark: false)
                        window(dark: true).mask(
                            LinearGradient(stops: [.init(color: .clear, location: 0.5), .init(color: .black, location: 0.5)],
                                           startPoint: .topLeading, endPoint: .bottomTrailing))
                    }
                } else {
                    window(dark: theme == .dark)
                }
            }
            .frame(height: 68)
            .clipShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 7, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5) }
            .accessibilityHidden(true)
            HStack {
                Text(theme.title).font(.system(size: 13, weight: selected ? .semibold : .regular))
                Spacer()
                if selected { Image(systemName: "checkmark.circle.fill") }
            }
            .foregroundStyle(StudioTheme.ink)
        }
        .padding(9)
        .overlay(
            RoundedRectangle(cornerRadius: 11, style: .continuous).strokeBorder(
                selected ? StudioTheme.ink.opacity(0.85) : StudioTheme.edge,
                lineWidth: selected ? 1.5 : 0.5))
        .background(StudioTheme.contrast.opacity(selected ? 0.04 : 0), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
    }
}
