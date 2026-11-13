import SwiftUI

struct ConversationDisplayControl: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var presented = false

    var body: some View {
        Button { presented.toggle() } label: {
            Image(systemName: studio.readingWidth == .wide ? "arrow.left.and.right" : "text.alignleft")
                .frame(width: 30, height: 30)
        }
        .buttonStyle(WorkspaceControlStyle(selected: presented))
        .accessibilityLabel(L("Affichage des réponses", "Response appearance"))
        .help(L("Largeur et taille du texte", "Width and text size"))
        .popover(isPresented: $presented, arrowEdge: .bottom) {
            ConversationDisplaySettings()
                .padding(20).frame(width: 320)
                .background {
                    StudioAurora().opacity(0.5)
                        .background(StudioTheme.canvas(for: studio.theme).opacity(0.85))
                }
                .premiumGlass(radius: 18)
                .studioAppearance(studio.theme, accent: studio.accent, customAccent: studio.customAccentColor)
                .environmentObject(studio)
        }
    }
}

struct ConversationDisplaySettings: View {
    var showsTitle = true
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            if showsTitle {
                Text(L("Affichage", "Appearance"))
                    .font(.system(size: 25, weight: .regular, design: .serif)).tracking(-0.5)
            }
            VStack(alignment: .leading, spacing: 9) {
                Text(L("Largeur des réponses", "Response width")).font(.system(size: 13, weight: .medium))
                HStack(spacing: 5) {
                    ForEach(ConversationReadingWidth.allCases, id: \.self) { width in
                        Button { studio.setReadingWidth(width) } label: {
                            Text(width.title)
                                .font(.system(size: 12, weight: studio.readingWidth == width ? .semibold : .regular))
                                .frame(maxWidth: .infinity).frame(height: 32)
                        }
                        .buttonStyle(StudioControlStyle(emphasized: studio.readingWidth == width, cornerRadius: 8))
                        .accessibilityAddTraits(studio.readingWidth == width ? .isSelected : [])
                    }
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(L("Taille du texte", "Text size"))
                    Spacer()
                    Text("\(Int(studio.responseTextSize)) pt").monospacedDigit()
                        .foregroundStyle(StudioTheme.secondary)
                }.font(.system(size: 13, weight: .medium))
                HStack(spacing: 12) {
                    Text("A").font(.system(size: 11)).accessibilityHidden(true)
                    Slider(value: Binding(get: { studio.responseTextSize }, set: { studio.setResponseTextSize($0) }),
                           in: ResponseTextSize.range)
                        .tint(studio.accentColor)
                        .accessibilityLabel(L("Taille du texte", "Text size"))
                        .accessibilityValue("\(Int(studio.responseTextSize)) pt")
                    Text("A").font(.system(size: 21)).accessibilityHidden(true)
                }
                Text(L("Exemple de texte", "Sample text"))
                    .font(.system(size: studio.responseTextSize))
                    .frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
                    .padding(.vertical, 4)
                Button(L("Réinitialiser la taille", "Reset text size")) {
                    studio.setResponseTextSize(ResponseTextSize.defaultValue)
                }
                .buttonStyle(StudioControlStyle(cornerRadius: 8)).font(.system(size: 11))
                .padding(.vertical, 4)
                .foregroundStyle(StudioTheme.secondary)
                .disabled(studio.responseTextSize == ResponseTextSize.defaultValue)
            }
        }
        .foregroundStyle(StudioTheme.ink)
    }
}
