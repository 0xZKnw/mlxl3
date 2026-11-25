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
                        Label(error, systemImage: "exclamationmark.circle")
                            .foregroundStyle(StudioTheme.failure).textSelection(.enabled)
                    }
                    if let item = engine.history.first(where: { $0.id == selection }) {
                        detail(item, width: width, viewportHeight: geometry.size.height)
                    } else {
                        library(width: width)
                    }
                }
                .frame(width: width, alignment: .leading)
                .padding(.vertical, 36)
                .frame(maxWidth: .infinity)
            }
            .id(selection)
        }
        .task(id: refreshID) { await engine.refreshHistory() }
        .onChange(of: engine.history) {
            if let selection, !engine.history.contains(where: { $0.id == selection }) { self.selection = nil }
        }
    }

    private func heading(_ text: String) -> some View {
        Text(text).font(StudioTheme.display(39)).tracking(StudioTheme.displayTracking(-1.2))
            .accessibilityAddTraits(.isHeader)
    }

    @ViewBuilder private func library(width: CGFloat) -> some View {
        HStack {
            heading(L("Historique des images", "Image history"))
            Spacer()
            Button { refreshID += 1 } label: {
                Image(systemName: "arrow.clockwise").frame(width: 28, height: 28)
            }
            .buttonStyle(StudioControlStyle()).disabled(engine.isLoadingHistory)
            .help(L("Actualiser", "Refresh"))
            .accessibilityLabel(L("Actualiser l’historique", "Refresh history"))
            .accessibilityIdentifier("images.history.refresh")
        }
        if engine.history.isEmpty {
            if engine.isLoadingHistory {
                ProgressView(L("Chargement des images…", "Loading images…")).controlSize(.small)
            } else if engine.historyError == nil {
                VStack(alignment: .leading, spacing: 10) {
                    Text(L("Aucune image générée", "No images yet")).font(.headline)
                    Button(L("Créer une image", "Create an image"), action: openGenerator).buttonStyle(.link)
                }
                .padding(.vertical, 12)
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
                                .clipShape(RoundedRectangle(cornerRadius: 9))
                                .overlay { RoundedRectangle(cornerRadius: 9).strokeBorder(StudioTheme.edge) }
                                .padding(.bottom, 3)
                            Text(title(item)).font(.body.weight(.medium))
                                .lineLimit(2).multilineTextAlignment(.leading)
                            Text(date(item)).font(.caption).foregroundStyle(StudioTheme.secondary)
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
        .buttonStyle(.link).accessibilityIdentifier("images.history.back")
        heading(L("Image générée", "Generated image"))
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
                .clipShape(RoundedRectangle(cornerRadius: 10))
                .overlay { RoundedRectangle(cornerRadius: 10).strokeBorder(StudioTheme.edge) }
                .accessibilityLabel(L("Aperçu de l’image générée", "Generated image preview"))
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(date(item)).font(.callout).foregroundStyle(StudioTheme.secondary)
                    if let prompt = item.prompt {
                        Text(L("Prompt", "Prompt")).font(.headline)
                        Text(prompt).textSelection(.enabled)
                    } else {
                        Text(L("Prompt non enregistré", "Prompt unavailable")).foregroundStyle(StudioTheme.secondary)
                    }
                }
                Divider()
                VStack(alignment: .leading, spacing: 8) {
                    Text(L("Réglages", "Settings")).font(.headline)
                    Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 7) {
                        if let model = item.modelName { setting(L("Modèle", "Model"), model) }
                        setting(L("Dimensions", "Dimensions"), "\(item.width) × \(item.height) px")
                        if let steps = item.steps { setting(L("Étapes", "Steps"), String(steps)) }
                        if let seed = item.seed { setting(L("Graine", "Seed"), String(seed)) }
                    }
                }
                if let prompt = item.prompt, let steps = item.steps, let seed = item.seed,
                   item.width == item.height, [512, 768, 1024].contains(item.width), (2...50).contains(steps) {
                    Button {
                        engine.prompt = prompt
                        engine.size = item.width
                        engine.steps = steps
                        engine.seed = seed
                        selection = nil
                        openGenerator()
                    } label: {
                        Label(L("Réutiliser les réglages", "Reuse settings"), systemImage: "arrow.counterclockwise")
                            .padding(.horizontal, 10).frame(height: 30)
                    }
                    .buttonStyle(StudioToolbarStyle()).disabled(engine.running)
                    .accessibilityIdentifier("images.history.reuse")
                }
                Button { NSWorkspace.shared.activateFileViewerSelecting([item.fileURL]) } label: {
                    Label(L("Afficher le PNG", "Show PNG"), systemImage: "folder")
                }
                .buttonStyle(.link).accessibilityIdentifier("images.history.reveal")
            }
            .frame(width: wide ? infoWidth : nil, alignment: .leading)
            .frame(maxWidth: wide ? nil : .infinity, alignment: .leading)
        }
    }

    private func setting(_ label: String, _ value: String) -> some View {
        GridRow {
            Text(label).foregroundStyle(StudioTheme.secondary)
            Text(value).textSelection(.enabled)
        }
        .font(.callout).accessibilityElement(children: .combine)
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
                    .font(.caption).foregroundStyle(StudioTheme.secondary)
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
