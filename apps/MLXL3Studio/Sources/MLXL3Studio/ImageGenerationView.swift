import AppKit
import SwiftUI
import UniformTypeIdentifiers

@MainActor
func presentImageSavePanel(filename: String, directory: URL? = nil, export: @escaping (URL) async throws -> Void) {
    let panel = NSSavePanel()
    panel.allowedContentTypes = [.png]
    panel.canCreateDirectories = true
    panel.nameFieldStringValue = filename
    if let directory { panel.directoryURL = directory }
    guard panel.runModal() == .OK, let destination = panel.url else { return }
    Task {
        do { try await export(destination) }
        catch { NSAlert(error: error).runModal() }
    }
}

private extension ImageGeneration {
    // Swift 6.3.3 crashes when it emits the isolated method adapter for
    // Binding(get:set:). A writable key path keeps the same guarded selection.
    var pickerVariant: Image21Variant {
        get { variant }
        set { selectVariant(newValue) }
    }
}

struct ImageGenerationView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var engine: ImageGeneration
    let close: () -> Void
    @State private var previewImage: NSImage?
    @State private var referencePreviewImage: NSImage?
    @State private var referenceDropTargeted = false
    @State private var showsHistory = false
    @State private var selectedImage: URL?

    private static let example = Bundle.main.url(forResource: "studio-image21-m5", withExtension: "png",
                                                  subdirectory: "Image21").flatMap(NSImage.init(contentsOf:))

    var body: some View {
        HStack(spacing: 0) {
            StudioPageSidebar(title: L("Images", "Images"), onBack: goBack) {
                StudioNavList {
                    navigation(L("Générer", "Generate"), icon: "wand.and.stars", history: false)
                    navigation(L("Historique", "History"), icon: "clock", history: true)
                }
                .padding(.bottom, 16)
                if !showsHistory {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 20) {
                            modelPicker
                            Rectangle().fill(StudioTheme.edge).frame(height: 0.5).accessibilityHidden(true)
                            settings
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 18).padding(.bottom, 12)
                    }
                    .scrollIndicators(.hidden)
                }
            }

            if showsHistory {
                ImageHistoryView(engine: engine, selection: $selectedImage) { showsHistory = false }
            } else {
                VStack(alignment: .leading, spacing: 16) {
                    StudioPageHeader(title: engine.editing ? L("Retoucher une image", "Edit an image")
                                                            : L("Générer une image", "Generate an image"),
                                     detail: L("Qwen-Image-2.1, entièrement sur ce Mac.", "Qwen-Image-2.1, entirely on this Mac."))
                    preview
                    composer
                }
                .padding(.horizontal, 32).padding(.top, 40).padding(.bottom, 24)
                .frame(maxWidth: 960)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .contentShape(Rectangle())
                .dropDestination(for: URL.self) { urls, _ in
                    acceptReferenceDrop(urls)
                } isTargeted: {
                    referenceDropTargeted = $0 && canImportReference
                }
            }
        }
        .font(.system(size: 13))
        .foregroundStyle(StudioTheme.ink)
        .background { StudioAurora(animated: false).background(StudioTheme.canvas) }
        .onExitCommand(perform: goBack)
        .task(id: engine.variant) { await engine.refreshModelDirectory() }
        .task(id: engine.result?.id) { previewImage = engine.result.flatMap { NSImage(data: $0.png) } }
        .task(id: engine.referenceImage?.png) {
            referencePreviewImage = engine.referenceImage.flatMap { NSImage(data: $0.png) }
        }
    }

    private var canImportReference: Bool {
        engine.variant == .turbo && !engine.running && !engine.isDownloading && !engine.referenceImporting
    }

    @discardableResult
    func acceptReferenceDrop(_ urls: [URL]) -> Bool {
        guard canImportReference, urls.count == 1, let url = urls.first, url.isFileURL else { return false }
        Task { await engine.importReference(url) }
        return true
    }

    private func navigation(_ title: String, icon: String, history: Bool) -> some View {
        StudioNavRow(icon: icon, title: title, selected: history == showsHistory,
                     badge: !history && engine.running ? "…" : nil) {
            selectedImage = nil
            showsHistory = history
        }
        .accessibilityIdentifier(history ? "images.history" : "images.create")
    }

    /// Back leaves an opened image first, then the page.
    private func goBack() {
        if showsHistory && selectedImage != nil { selectedImage = nil } else { close() }
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L("Modèle", "Model")).font(.system(size: 12.5, weight: .semibold)).foregroundStyle(StudioTheme.secondary)
            Text("Qwen-Image-2.1").font(.system(size: 14, weight: .semibold))
            Picker(L("Version du modèle", "Model variant"), selection: $engine.pickerVariant) {
                Text("Turbo").tag(Image21Variant.turbo)
                Text("Base").tag(Image21Variant.base)
            }
            .pickerStyle(.segmented).labelsHidden()
            .disabled(engine.running || engine.isDownloading || engine.referenceImporting)
            .accessibilityIdentifier("images.variant")
            Text(engine.variant == .turbo
                 ? L("Recommandé pour générer rapidement, en 8 étapes.", "Recommended for fast generation in 8 steps.")
                 : L("De 2 à 50 étapes pour ajuster le rendu ou recréer une image Base.",
                     "Choose 2 to 50 steps to adjust the result or recreate a Base image."))
                .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                .fixedSize(horizontal: false, vertical: true)
            modelInstallation
            if let error = engine.folderError {
                Text(error).font(.caption).foregroundStyle(StudioTheme.warning)
            }
            if !engine.isDownloading && engine.modelAvailability != .checking {
                Button {
                    let panel = NSOpenPanel()
                    panel.canChooseFiles = false
                    panel.canChooseDirectories = true
                    if panel.runModal() == .OK, let url = panel.url {
                        Task { await engine.selectModelDirectory(url) }
                    }
                } label: {
                    Label(L("Choisir un dossier local…", "Choose a local folder…"), systemImage: "folder")
                        .font(.system(size: 12))
                }
                .buttonStyle(.plain).foregroundStyle(StudioTheme.secondary)
                .disabled(engine.running).accessibilityIdentifier("images.model")
            }
        }
    }

    @ViewBuilder private var modelInstallation: some View {
        switch engine.modelAvailability {
        case .checking:
            Text(L("Recherche sur ce Mac…", "Checking this Mac…")).font(.caption).foregroundStyle(StudioTheme.secondary)
        case let .installed(url):
            Label(L("Installé sur ce Mac", "Installed on this Mac"), systemImage: "checkmark.circle.fill")
                .font(.system(size: 12, weight: .medium)).foregroundStyle(StudioTheme.success)
                .help(url.path).accessibilityIdentifier("images.installed")
        case let .downloading(completed, total):
            VStack(alignment: .leading, spacing: 5) {
                HStack {
                    Text(L("Téléchargement", "Downloading"))
                    Spacer()
                    Button(L("Annuler", "Cancel"), action: engine.cancelModelDownload)
                        .buttonStyle(.plain).foregroundStyle(StudioTheme.accent)
                        .accessibilityIdentifier("images.cancelDownload")
                }
                ProgressView(value: Double(completed), total: Double(max(1, total))).tint(StudioTheme.accent)
                if total > 0 {
                    Text(Self.storage(completed) + " / " + Self.storage(total)).monospacedDigit()
                }
            }
            .font(.caption).foregroundStyle(StudioTheme.secondary)
        case let .missing(bytes):
            downloadButton(L("Télécharger", "Download"), bytes: bytes)
        case let .paused(bytes):
            downloadButton(L("Reprendre", "Resume"), bytes: bytes)
        case let .failed(message, bytes):
            Text(message).font(.caption).foregroundStyle(StudioTheme.warning).textSelection(.enabled)
            downloadButton(L("Réessayer", "Retry"), bytes: bytes)
        }
    }

    private static func storage(_ bytes: Int64) -> String {
        if bytes < 1_000_000 { return bytes.formatted() + L(" o", " B") }
        if bytes < 1_000_000_000 {
            return (Double(bytes) / 1_000_000).formatted(.number.precision(.fractionLength(1))) + L(" Mo", " MB")
        }
        let value = (Double(bytes) / 1_000_000_000).formatted(.number.precision(.fractionLength(2)))
        return value + L(" Go", " GB")
    }

    private func downloadButton(_ title: String, bytes: Int64) -> some View {
        Button(action: engine.downloadModel) {
            Label(bytes > 0 ? title + " · " + Self.storage(bytes) : title, systemImage: "arrow.down.circle")
        }
        .buttonStyle(.studio(.primary)).disabled(engine.running)
        .accessibilityLabel(title + " " + engine.variant.title)
        .accessibilityIdentifier("images.download")
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L("Réglages", "Settings")).font(.system(size: 12.5, weight: .semibold)).foregroundStyle(StudioTheme.secondary)
            Text(L("Taille de l’image (px)", "Image size (px)")).font(.system(size: 12.5))
            if engine.editing {
                Label("\(ImageEditReference.size) × \(ImageEditReference.size) px", systemImage: "lock")
                    .font(.system(size: 12).monospacedDigit())
                    .foregroundStyle(StudioTheme.secondary)
                    .accessibilityIdentifier("images.editSize")
            } else {
                // Bare numbers: three "… px" segments are wider than a narrow sidebar.
                Picker(L("Taille de l’image", "Image size"), selection: $engine.size) {
                    ForEach([512, 768, 1024], id: \.self) { Text(String($0)).tag($0) }
                }
                .pickerStyle(.segmented).labelsHidden()
            }
            if engine.variant == .base && !engine.editing {
                Stepper(value: $engine.baseSteps, in: 2...50) {
                    Text(L("\(engine.steps) étapes", "\(engine.steps) steps")).font(.system(size: 12.5)).monospacedDigit()
                }
                .padding(.vertical, 4)
            }
            Text(L("Graine", "Seed")).font(.system(size: 12.5))
            HStack(spacing: 6) {
                TextField(L("Graine", "Seed"), value: $engine.seed, format: .number.grouping(.never))
                    .studioField(monospaced: true)
                StudioIconButton(icon: "dice", label: L("Nouvelle graine", "New seed")) {
                    engine.seed = .random(in: 0...UInt32.max)
                }
            }
        }
        .disabled(engine.running)
    }

    private var previewTitle: String {
        if engine.running {
            return engine.editing ? L("Retouche en cours", "Editing") : L("Génération en cours", "Generating")
        }
        if let result = engine.result {
            return result.isEdit ? L("Image retouchée", "Edited image") : L("Image générée", "Generated image")
        }
        if engine.referenceImage != nil { return L("Image d’origine", "Original image") }
        return engine.editing ? L("Aperçu", "Preview") : L("Exemple du modèle Base", "Base model example")
    }

    private var displayedPreview: NSImage? {
        if engine.result != nil { return previewImage }
        if engine.editing { return referencePreviewImage }
        return Self.example
    }

    private var preview: some View {
        VStack(spacing: 10) {
            HStack {
                Text(previewTitle)
                    .font(.system(size: 12.5, weight: .medium)).foregroundStyle(StudioTheme.secondary)
                Spacer()
                if !engine.running, let url = engine.result?.savedURL {
                    Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                        Label(L("Afficher le PNG", "Show PNG"), systemImage: "folder")
                    }.buttonStyle(.studio(.secondary, height: 26))
                } else if !engine.running, let output = engine.result {
                    Button {
                        presentImageSavePanel(filename: "Image-\(output.seed).png", export: output.export)
                    } label: {
                        Label(L("Enregistrer…", "Save…"), systemImage: "square.and.arrow.down")
                    }
                    .buttonStyle(.studio(.secondary, height: 26))
                    .accessibilityIdentifier("images.savePNG")
                }
            }
            ZStack {
                if engine.running {
                    VStack(spacing: 12) {
                        ProgressView().controlSize(.small)
                        Text(engine.status).font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
                    }
                } else if let image = displayedPreview {
                    Image(nsImage: image).resizable().interpolation(.high).scaledToFit()
                        .accessibilityLabel(engine.result != nil || engine.editing ? previewTitle
                            : L("Exemple généré avec Qwen Image, une théière bleue", "Qwen Image example of a blue teapot"))
                } else {
                    Image(systemName: "photo").font(.largeTitle).foregroundStyle(StudioTheme.quiet)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(StudioTheme.panel)
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(StudioTheme.edge.opacity(0.7), lineWidth: 0.5) }
        }
        .frame(maxHeight: .infinity)
    }

    private var referenceTile: some View {
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 9) {
                ZStack {
                    RoundedRectangle(cornerRadius: 6).fill(StudioTheme.panel)
                    if engine.referenceImporting {
                        ProgressView().controlSize(.small)
                    } else if let referencePreviewImage {
                        Image(nsImage: referencePreviewImage).resizable().scaledToFit()
                    } else {
                        Image(systemName: "photo.badge.plus").font(.system(size: 15))
                            .foregroundStyle(StudioTheme.secondary)
                    }
                }
                .frame(width: 32, height: 32)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(engine.referenceImage?.name ?? L("Image de référence", "Reference image"))
                        .font(.system(size: 12, weight: .semibold)).lineLimit(1).truncationMode(.middle)
                    Text(referenceHint).font(.system(size: 11.5)).foregroundStyle(StudioTheme.secondary)
                        .lineLimit(2)
                }
                Spacer(minLength: 0)
                if engine.variant == .base {
                    Button(L("Utiliser Turbo", "Use Turbo")) { engine.selectVariant(.turbo) }
                        .buttonStyle(.studio(.secondary, height: 26))
                        .disabled(engine.running || engine.isDownloading)
                        .accessibilityIdentifier("images.referenceTurbo")
                } else {
                    Button(engine.referenceImage == nil ? L("Choisir…", "Choose…") : L("Remplacer…", "Replace…"),
                           action: chooseReference)
                        .buttonStyle(.studio(.secondary, height: 26)).disabled(!canImportReference)
                        .help(L("Choisir une image de référence", "Choose a reference image"))
                        .accessibilityLabel(engine.referenceImage == nil
                            ? L("Choisir une image de référence", "Choose a reference image")
                            : L("Remplacer l’image de référence", "Replace the reference image"))
                        .accessibilityIdentifier("images.referenceChoose")
                }
                if engine.editing {
                    StudioIconButton(icon: "xmark", label: L("Retirer la référence", "Remove reference")) {
                        engine.removeReference()
                    }
                    .disabled(engine.running)
                    .accessibilityIdentifier("images.referenceRemove")
                }
            }
            if let error = engine.referenceError {
                Text(error).font(.caption).foregroundStyle(StudioTheme.failure)
                    .textSelection(.enabled)
            }
        }
        .padding(.horizontal, 9).padding(.vertical, 5)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 9))
        .overlay {
            RoundedRectangle(cornerRadius: 9)
                .strokeBorder(referenceDropTargeted ? studio.accentColor : StudioTheme.edge,
                              lineWidth: referenceDropTargeted ? 1.5 : 0.5)
                .allowsHitTesting(false)
        }
    }

    private var referenceHint: String {
        if engine.referenceImporting { return L("Importation…", "Importing…") }
        if engine.variant == .base {
            return L("Retouche à 512 px avec Turbo.", "512 px edits use Turbo.")
        }
        if engine.referenceImage != nil { return L("Retouche · 512 px", "Edit · 512 px") }
        return L("Glisse une image pour la retoucher.", "Drop an image to edit it.")
    }

    private func chooseReference() {
        guard canImportReference else { return }
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            acceptReferenceDrop([url])
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            referenceTile
            TextField(engine.editing ? L("Décris ce qui doit changer…", "Describe what to change…")
                                     : L("Décris l’image à générer…", "Describe the image to generate…"),
                      text: $engine.prompt, axis: .vertical)
                .textFieldStyle(.plain).font(.system(size: 14)).lineLimit(2...4).disabled(engine.running)
                .accessibilityIdentifier("images.prompt")
            if let error = engine.error {
                Text(error).font(.system(size: 12.5)).foregroundStyle(StudioTheme.failure)
                    .textSelection(.enabled).lineLimit(3).help(error)
            } else if !engine.running && !engine.status.isEmpty {
                Text(engine.status).font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
            }
            HStack(spacing: 12) {
                Text(engine.editing
                     ? L("Les composants de vision sont téléchargés à la première retouche.",
                         "Vision components download on the first edit.")
                     : L("Environ 5 Go de composants partagés au premier lancement.",
                         "First use downloads about 5 GB of shared components."))
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                Spacer()
                if engine.running {
                    Button(L("Annuler", "Cancel"), action: engine.cancel)
                        .buttonStyle(.studio).disabled(engine.finalizing)
                } else {
                    Button {
                        guard let model = engine.modelDirectory else { return }
                        engine.start(model: model, prompt: engine.prompt,
                                     size: engine.editing ? ImageEditReference.size : engine.size, steps: engine.steps, seed: engine.seed)
                    } label: {
                        Label(engine.editing ? L("Retoucher", "Edit") : L("Générer", "Generate"),
                              systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.studio(.primary)).keyboardShortcut(.return, modifiers: .command)
                    .help(engine.editing ? L("Retoucher · ⌘↩", "Edit · ⌘↩") : L("Générer · ⌘↩", "Generate · ⌘↩"))
                    .disabled(engine.modelDirectory == nil || engine.referenceImporting
                        || (engine.editing && (engine.referenceImage == nil || engine.variant != .turbo))
                        || engine.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("images.generate")
                }
            }
        }
        .padding(14)
        .background(StudioTheme.composer.opacity(0.9), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5).allowsHitTesting(false) }
    }
}
