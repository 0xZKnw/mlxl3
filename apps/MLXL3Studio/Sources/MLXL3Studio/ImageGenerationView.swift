import AppKit
import SwiftUI

struct ImageGenerationView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var engine: ImageGeneration
    let close: () -> Void
    @AppStorage("image21ModelDirectory") private var modelPath = ""
    @State private var previewImage: NSImage?

    private static let example = Bundle.main.url(forResource: "studio-image21-m5", withExtension: "png",
                                                  subdirectory: "Image21").flatMap(NSImage.init(contentsOf:))

    var body: some View {
        HStack(spacing: 0) {
            StudioPageSidebar(title: L("Images", "Images"), onBack: close) {
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

            VStack(alignment: .leading, spacing: 16) {
                StudioPageHeader(title: L("Générer une image", "Generate an image"),
                                 detail: L("Qwen-Image-2.1, entièrement sur ce Mac.", "Qwen-Image-2.1, entirely on this Mac."))
                preview
                composer
            }
            .padding(.horizontal, 32).padding(.top, 40).padding(.bottom, 24)
            .frame(maxWidth: 960)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .font(.system(size: 13))
        .foregroundStyle(StudioTheme.ink)
        .background { StudioAurora(animated: false).background(StudioTheme.canvas) }
        .onExitCommand(perform: close)
        .task(id: engine.result) { previewImage = engine.result.flatMap(NSImage.init(contentsOf:)) }
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L("Modèle", "Model")).font(.system(size: 12.5, weight: .semibold)).foregroundStyle(StudioTheme.secondary)
            Text("Qwen-Image-2.1").font(.system(size: 14, weight: .semibold))
            Text(modelPath.isEmpty ? L("Choisis le dossier EXL3 pour commencer.", "Choose the EXL3 folder to start.")
                 : URL(fileURLWithPath: modelPath).lastPathComponent)
                .font(.system(size: 12)).foregroundStyle(modelPath.isEmpty ? StudioTheme.warning : StudioTheme.secondary)
                .lineLimit(2).truncationMode(.middle).help(modelPath)
            Button {
                let panel = NSOpenPanel()
                panel.canChooseFiles = false
                panel.canChooseDirectories = true
                if panel.runModal() == .OK, let url = panel.url { modelPath = url.path }
            } label: {
                Label(modelPath.isEmpty ? L("Choisir le dossier…", "Choose folder…") : L("Changer de dossier…", "Change folder…"),
                      systemImage: "folder")
            }
            .buttonStyle(.studio(modelPath.isEmpty ? .primary : .secondary)).disabled(engine.running)
            .accessibilityIdentifier("images.model")
        }
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(L("Réglages", "Settings")).font(.system(size: 12.5, weight: .semibold)).foregroundStyle(StudioTheme.secondary)
            Text(L("Taille de l’image (px)", "Image size (px)")).font(.system(size: 12.5))
            // Bare numbers: three "… px" segments are wider than a narrow sidebar.
            Picker(L("Taille de l’image", "Image size"), selection: $engine.size) {
                ForEach([512, 768, 1024], id: \.self) { Text(String($0)).tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden()
            Stepper(value: $engine.steps, in: 2...50) {
                Text(L("\(engine.steps) étapes", "\(engine.steps) steps")).font(.system(size: 12.5)).monospacedDigit()
            }
            .padding(.vertical, 4)
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

    private var preview: some View {
        VStack(spacing: 10) {
            HStack {
                Text(engine.running ? L("Génération en cours", "Generating")
                     : engine.result == nil ? L("Exemple du modèle", "Model example") : L("Image générée", "Generated image"))
                    .font(.system(size: 12.5, weight: .medium)).foregroundStyle(StudioTheme.secondary)
                Spacer()
                if let url = engine.result {
                    Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                        Label(L("Afficher le PNG", "Show PNG"), systemImage: "folder")
                    }.buttonStyle(.studio(.secondary, height: 26))
                }
            }
            ZStack {
                if engine.running {
                    VStack(spacing: 12) {
                        ProgressView().controlSize(.small)
                        Text(engine.status).font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
                    }
                } else if let image = engine.result == nil ? Self.example : previewImage {
                    Image(nsImage: image).resizable().interpolation(.high).scaledToFit()
                        .accessibilityLabel(engine.result == nil
                            ? L("Exemple généré avec Qwen Image, une théière bleue", "Qwen Image example of a blue teapot")
                            : L("Image générée", "Generated image"))
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

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField(L("Décris l’image à générer…", "Describe the image to generate…"),
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
                Text(L("Premier lancement : environ 5 Go à télécharger.", "First use downloads about 5 GB."))
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                Spacer()
                if engine.running {
                    Button(L("Annuler", "Cancel"), action: engine.cancel).buttonStyle(.studio)
                } else {
                    Button {
                        engine.start(model: URL(fileURLWithPath: modelPath), prompt: engine.prompt,
                                     size: engine.size, steps: engine.steps, seed: engine.seed)
                    } label: {
                        Label(L("Générer", "Generate"), systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.studio(.primary)).keyboardShortcut(.return, modifiers: .command)
                    .help(L("Générer · ⌘↩", "Generate · ⌘↩"))
                    .disabled(modelPath.isEmpty || engine.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("images.generate")
                }
            }
        }
        .padding(14)
        .background(StudioTheme.composer.opacity(0.9), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
        .overlay { RoundedRectangle(cornerRadius: 14, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5).allowsHitTesting(false) }
    }
}
