import SwiftUI

struct AppearanceSettings: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            SettingCard(title: L("Thème de l’interface", "Interface theme"), icon: "paintpalette") {
                Text(
                    L(
                        "Le changement s’applique immédiatement à toute l’app.",
                        "Changes apply immediately throughout the app.")
                )
                .foregroundStyle(StudioTheme.secondary)
                LazyVGrid(
                    columns: Array(repeating: GridItem(.flexible(minimum: 100), spacing: 12), count: 4), spacing: 12
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
            SettingCard(title: L("Couleur d’accent", "Accent color"), icon: "circle.lefthalf.filled") {
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 88))], spacing: 10) {
                    ForEach(AppAccent.allCases, id: \.self) { accent in
                        Button {
                            studio.setAccent(accent)
                        } label: {
                            VStack(spacing: 8) {
                                Circle().fill(accent.color).frame(width: 25, height: 25)
                                    .overlay {
                                        if studio.accent == accent {
                                            Image(systemName: "checkmark").font(.system(size: 11, weight: .bold))
                                                .foregroundStyle(accent == .neutral ? StudioTheme.onAccent : .white)
                                        }
                                    }
                                Text(accent.title).font(.system(size: 12))
                            }
                            .frame(maxWidth: .infinity).padding(.vertical, 10)
                            .background(
                                studio.accent == accent ? StudioTheme.panel : .clear,
                                in: RoundedRectangle(cornerRadius: 9))
                        }
                        .buttonStyle(StudioControlStyle())
                        .accessibilityLabel(accent.title)
                        .accessibilityAddTraits(studio.accent == accent ? .isSelected : [])
                    }
                }
                Text(
                    L(
                        "Appliquée aux sélecteurs, aux interrupteurs et aux contrôles de l’app.",
                        "Applied to the app’s selectors, switches and controls.")
                )
                .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
            }
            SettingCard(title: L("Lecture & conversations", "Reading & conversations"), icon: "textformat.size") {
                ConversationDisplaySettings(showsTitle: false)
                Divider().overlay(StudioTheme.edge)
                Text(L("Espacement des messages", "Message spacing")).fontWeight(.medium)
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
                .labelsHidden().pickerStyle(.segmented)
            }
            Button(L("Rétablir l’apparence par défaut", "Restore default appearance"), action: studio.resetAppearance)
                .buttonStyle(GlassPillButtonStyle()).padding(.vertical, 6)
        }
    }
}

private struct ThemePreview: View {
    @Environment(\.colorScheme) private var colorScheme
    let theme: AppTheme
    let selected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 7) {
                    RoundedRectangle(cornerRadius: 2).fill(StudioTheme.secondary).frame(width: 20, height: 4)
                    RoundedRectangle(cornerRadius: 2).fill(StudioTheme.quiet).frame(width: 28, height: 3)
                    RoundedRectangle(cornerRadius: 2).fill(StudioTheme.quiet).frame(width: 23, height: 3)
                    Spacer()
                }
                .padding(10).frame(width: 48).background(StudioTheme.sidebar(for: theme))
                VStack(alignment: .leading, spacing: 7) {
                    Text("Aa").font(.system(size: 19, design: .serif))
                    RoundedRectangle(cornerRadius: 2).fill(StudioTheme.quiet).frame(height: 3)
                    RoundedRectangle(cornerRadius: 2).fill(StudioTheme.quiet).frame(width: 40, height: 3)
                    Spacer(minLength: 0)
                    RoundedRectangle(cornerRadius: 4).fill(StudioTheme.composer(for: theme)).frame(height: 14)
                }
                .padding(10).frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(height: 100)
            .background(StudioTheme.canvas(for: theme))
            .foregroundStyle(StudioTheme.ink)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .environment(\.colorScheme, theme.colorScheme ?? colorScheme)
            HStack {
                Text(theme.title).font(.system(size: 13, weight: selected ? .semibold : .regular))
                Spacer()
                if selected { Image(systemName: "checkmark.circle.fill") }
            }
            .foregroundStyle(StudioTheme.ink)
        }
        .padding(9)
        .overlay(
            RoundedRectangle(cornerRadius: 12).stroke(
                selected ? StudioTheme.ink : StudioTheme.edge,
                lineWidth: selected ? 1.5 : 0.7))
    }
}
