import SwiftUI

struct ConversationDisplayControl: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var presented = false

    var body: some View {
        Button { presented.toggle() } label: {
            Image(systemName: studio.readingWidth == .wide ? "arrow.left.and.right" : "text.alignleft")
                .frame(width: 30, height: 30)
        }
        .buttonStyle(StudioControlStyle(emphasized: presented))
        .accessibilityLabel(L("Affichage des réponses", "Response appearance"))
        .help(L("Largeur et taille du texte", "Width and text size"))
        .popover(isPresented: $presented, arrowEdge: .bottom) {
            ConversationDisplaySettings()
                .padding(20).frame(width: 300)
                .background(StudioTheme.canvas(for: studio.theme))
                .studioAppearance(studio.theme)
                .environmentObject(studio)
        }
    }
}

struct ConversationDisplaySettings: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(L("Affichage", "Appearance")).font(.system(size: 17, weight: .semibold))
            VStack(alignment: .leading, spacing: 9) {
                Text(L("Largeur des réponses", "Response width")).font(.system(size: 11, weight: .medium))
                Picker(L("Largeur des réponses", "Response width"), selection: Binding(
                    get: { studio.readingWidth }, set: { studio.setReadingWidth($0) }
                )) {
                    ForEach(ConversationReadingWidth.allCases, id: \.self) { width in
                        Text(width.title).tag(width)
                    }
                }
                .labelsHidden().pickerStyle(.segmented)
            }
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(L("Taille du texte", "Text size"))
                    Spacer()
                    Text("\(Int(studio.responseTextSize)) pt").monospacedDigit()
                        .foregroundStyle(StudioTheme.secondary)
                }.font(.system(size: 11, weight: .medium))
                HStack(spacing: 12) {
                    Text("A").font(.system(size: 11)).accessibilityHidden(true)
                    Slider(value: Binding(get: { studio.responseTextSize }, set: { studio.setResponseTextSize($0) }),
                           in: ResponseTextSize.range, step: 1)
                        .tint(StudioTheme.blueAccent)
                        .accessibilityLabel(L("Taille du texte", "Text size"))
                        .accessibilityValue("\(Int(studio.responseTextSize)) pt")
                    Text("A").font(.system(size: 21)).accessibilityHidden(true)
                }
                Text(L("Un texte à votre taille.", "Text that suits you."))
                    .font(.system(size: studio.responseTextSize))
                    .frame(maxWidth: .infinity, minHeight: 52, alignment: .leading)
                    .padding(.horizontal, 12)
                    .background(StudioTheme.contrast.opacity(0.035), in: RoundedRectangle(cornerRadius: 9))
                Button(L("Réinitialiser la taille", "Reset text size")) {
                    studio.setResponseTextSize(ResponseTextSize.defaultValue)
                }
                .buttonStyle(.plain).font(.system(size: 10.5))
                .foregroundStyle(StudioTheme.secondary)
                .disabled(studio.responseTextSize == ResponseTextSize.defaultValue)
            }
        }
        .foregroundStyle(StudioTheme.ink)
    }
}
