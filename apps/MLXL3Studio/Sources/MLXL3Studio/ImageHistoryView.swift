import AppKit
import ImageIO
import SwiftUI

struct ImageHistoryView: View {
    @ObservedObject var engine: ImageGeneration
    @Binding var selection: URL?
    let openGenerator: () -> Void

    @State private var refreshID = 0
    @Environment(\.locale) private var locale

    var body: some View {
        GeometryReader { geometry in
            let inset: CGFloat = geometry.size.width < 600 ? 24 : 36
            let width = max(0, min(960, geometry.size.width - 2 * inset))
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    if let error = engine.historyError {
                        StudioNotice(kind: .error, text: error)
                    }
                    if let item = engine.history.first(where: { $0.id == selection }) {
                        detail(item, width: width, viewportHeight: geometry.size.height)
                    } else {
                        library(width: width)
                    }
                }
                .frame(width: width, alignment: .leading)
                .padding(.top, 40).padding(.bottom, 36)
                .frame(maxWidth: .infinity)
            }
            .id(selection)
        }
        .task(id: refreshID) { await engine.refreshHistory() }
        .onChange(of: engine.history) {
            if let selection, !engine.history.contains(where: { $0.id == selection }) { self.selection = nil }
        }
    }

    @ViewBuilder private func library(width: CGFloat) -> some View {
        StudioPageHeader(title: L("Historique des images", "Image history"),
                         detail: L("Les images générées sur ce Mac, les plus récentes d’abord.",
                                   "Pictures generated on this Mac, newest first.")) {
            StudioIconButton(icon: "arrow.clockwise", label: L("Actualiser l’historique", "Refresh history")) { refreshID += 1 }
                .disabled(engine.isLoadingHistory)
                .accessibilityIdentifier("images.history.refresh")
        }
        if engine.history.isEmpty {
            if engine.isLoadingHistory {
                HStack(spacing: 10) {
                    ProgressView().controlSize(.small)
                    Text(L("Chargement des images…", "Loading images…")).font(.system(size: 12.5))
                        .foregroundStyle(StudioTheme.secondary)
                }
            } else if engine.historyError == nil {
                StudioEmptyState(icon: "photo.on.rectangle", title: L("Aucune image générée", "No images yet"),
                                 detail: L("Les images que vous générez apparaîtront ici avec leurs réglages.",
                                           "The pictures you generate will appear here with their settings.")) {
                    Button(L("Créer une image", "Create an image"), action: openGenerator).buttonStyle(.studio(.primary))
                }
                .studioCard(padding: 0)
            }
        } else {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: min(width, 168), maximum: 260),
                                          spacing: 18, alignment: .topLeading)],
                      alignment: .leading, spacing: 24) {
                ForEach(engine.history) { item in
                    Button { selection = item.id } label: {
                        VStack(alignment: .leading, spacing: 5) {
                            Color.clear.aspectRatio(1, contentMode: .fit)
                                .overlay { SavedImagePreview(url: item.fileURL, thumbnail: true) }
                                .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                                .overlay { RoundedRectangle(cornerRadius: 10, style: .continuous).strokeBorder(StudioTheme.edge.opacity(0.7), lineWidth: 0.5) }
                                .padding(.bottom, 3)
                            Text(title(item)).font(.system(size: 13, weight: .medium))
                                .lineLimit(2).multilineTextAlignment(.leading)
                            Text(date(item)).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                        }
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(title(item)).accessibilityValue(date(item))
                    .accessibilityIdentifier("images.history.item." + item.fileURL.lastPathComponent)
                }
            }
            .id(refreshID)
        }
    }

    @ViewBuilder private func detail(_ item: GeneratedImageRecord, width: CGFloat, viewportHeight: CGFloat) -> some View {
        Button { selection = nil } label: {
            Label(L("Liste des images", "All images"), systemImage: "chevron.left")
        }
        .buttonStyle(.studio(.quiet, height: 26)).accessibilityIdentifier("images.history.back")
        StudioPageHeader(title: L("Image générée", "Generated image"), detail: date(item))
        let wide = width >= 640
        let infoWidth = min(width * 0.34, 285)
        let artworkWidth = wide ? width - infoWidth - 24 : width
        let nativeRatioHeight = artworkWidth * CGFloat(item.height) / CGFloat(max(1, item.width))
        let heightLimit = wide ? max(140, viewportHeight - 175) : max(150, viewportHeight * 0.30)
        let layout = wide ? AnyLayout(HStackLayout(alignment: .top, spacing: 24))
                          : AnyLayout(VStackLayout(alignment: .leading, spacing: 20))
        layout {
            Color.clear.frame(maxWidth: .infinity)
                .frame(height: max(80, min(nativeRatioHeight, heightLimit)))
                .overlay { SavedImagePreview(url: item.fileURL, thumbnail: false) }
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(StudioTheme.edge.opacity(0.7), lineWidth: 0.5) }
                .accessibilityLabel(L("Aperçu de l’image générée", "Generated image preview"))
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    if let prompt = item.prompt {
                        StudioSectionTitle(title: L("Prompt", "Prompt"))
                        Text(prompt).font(.system(size: 13)).textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    } else {
                        Text(L("Prompt non enregistré", "Prompt unavailable")).font(.system(size: 13))
                            .foregroundStyle(StudioTheme.secondary)
                    }
                }
                Rectangle().fill(StudioTheme.edge).frame(height: 0.5).accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 8) {
                    StudioSectionTitle(title: L("Réglages", "Settings"))
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 7) {
                        if let model = item.modelName { setting(L("Modèle", "Model"), model) }
                        if let variant = item.variant { setting(L("Version", "Variant"), variant.title) }
                        setting(L("Dimensions", "Dimensions"), "\(item.width) × \(item.height) px")
                        if let steps = item.steps { setting(L("Étapes", "Steps"), String(steps)) }
                        if let seed = item.seed { setting(L("Graine", "Seed"), String(seed)) }
                    }
                }
                if let prompt = item.prompt, let variant = item.variant, let steps = item.steps, let seed = item.seed,
                   item.width == item.height, [512, 768, 1024].contains(item.width), variant.stepRange.contains(steps) {
                    Button {
                        engine.selectVariant(variant)
                        engine.prompt = prompt
                        engine.size = item.width
                        if variant == .base { engine.baseSteps = steps }
                        engine.seed = seed
                        selection = nil
                        openGenerator()
                    } label: {
                        Label(L("Réutiliser les réglages", "Reuse settings"), systemImage: "arrow.counterclockwise")
                    }
                    .buttonStyle(.studio(.primary)).disabled(engine.running)
                    .accessibilityIdentifier("images.history.reuse")
                }
                Button { NSWorkspace.shared.activateFileViewerSelecting([item.fileURL]) } label: {
                    Label(L("Afficher le PNG", "Show PNG"), systemImage: "folder")
                }
                .buttonStyle(.studio).accessibilityIdentifier("images.history.reveal")
            }
            // The card padding stays inside the column the layout reserved.
            .frame(width: wide ? infoWidth - 32 : nil, alignment: .leading)
            .frame(maxWidth: wide ? nil : .infinity, alignment: .leading)
            .padding(wide ? 16 : 0)
            .background(wide ? StudioTheme.panel : .clear, in: RoundedRectangle(cornerRadius: StudioPageMetrics.card, style: .continuous))
        }
    }

    private func setting(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(StudioTheme.secondary)
            Text(value).monospacedDigit().textSelection(.enabled)
        }
        .font(.system(size: 12.5)).accessibilityElement(children: .combine)
    }

    private func date(_ item: GeneratedImageRecord) -> String {
        item.createdAt.formatted(Date.FormatStyle(date: .abbreviated, time: .shortened, locale: locale))
    }

    private func title(_ item: GeneratedImageRecord) -> String {
        item.prompt ?? L("Image générée", "Generated image")
    }
}

private struct SavedImagePreview: View {
    let url: URL
    let thumbnail: Bool
    @State private var image: CGImage?
    @State private var failed = false

    var body: some View {
        ZStack {
            StudioTheme.panel
            if let image {
                Image(decorative: image, scale: 1).resizable()
                    .aspectRatio(contentMode: thumbnail ? .fill : .fit)
            } else if failed {
                Label(L("Aperçu indisponible", "Preview unavailable"), systemImage: "photo")
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
            } else {
                ProgressView().controlSize(.small)
            }
        }
        .clipped()
        .task(id: url) {
            image = nil
            failed = false
            let sourceURL = url
            let limit = thumbnail ? 520 : 1600
            let load = Task.detached(priority: .utility) {
                guard !Task.isCancelled,
                      let source = CGImageSourceCreateWithURL(sourceURL as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary)
                else { return nil as CGImage? }
                let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true,
                                                kCGImageSourceCreateThumbnailWithTransform: true,
                                                kCGImageSourceThumbnailMaxPixelSize: limit,
                                                kCGImageSourceShouldCacheImmediately: true]
                return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
            }
            let decoded = await withTaskCancellationHandler { await load.value } onCancel: { load.cancel() }
            guard !Task.isCancelled else { return }
            image = decoded
            failed = decoded == nil
        }
        .onDisappear { image = nil }
    }
}
