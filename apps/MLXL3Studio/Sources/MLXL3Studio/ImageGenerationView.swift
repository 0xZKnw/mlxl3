import AppKit
import SwiftUI

struct ImageGenerationView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var engine: ImageGeneration
    let close: () -> Void
    @AppStorage("image21ModelDirectory") private var modelPath = ""
    @State private var previewImage: NSImage?
    @State private var showsHistory = false
    @State private var selectedImage: URL?

    private static let example = Bundle.main.url(forResource: "studio-image21-m5", withExtension: "png",
                                                  subdirectory: "Image21").flatMap(NSImage.init(contentsOf:))

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                StudioSidebarHeader(title: L("Images", "Images"))
                VStack(spacing: 3) {
                    navigation(L("Générer", "Generate"), icon: "wand.and.stars", history: false)
                    navigation(L("Historique", "History"), icon: "clock", history: true)
                }
                .padding(.horizontal, 10).padding(.bottom, 20)
                if showsHistory {
                    Spacer()
                } else {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 22) {
                            modelPicker
                            Divider()
                            settings
                        }
                        .padding(.horizontal, 20)
                    }
                }
                StudioBackButton(action: goBack)
                    .padding(.horizontal, 18).padding(.vertical, 16)
            }
            .frame(width: studio.sidebarWidth.points)
            .modifier(StudioSidebarSurface())

            if showsHistory {
                ImageHistoryView(engine: engine, selection: $selectedImage) { showsHistory = false }
            } else {
                VStack(alignment: .leading, spacing: 16) {
                    Text(L("Générer une image", "Generate an image"))
                        .font(StudioTheme.display(39)).tracking(StudioTheme.displayTracking(-1.2))
                        .accessibilityAddTraits(.isHeader)
                    preview
                    composer
                }
                .padding(.horizontal, 28).padding(.top, 38).padding(.bottom, 24)
                .frame(maxWidth: 960)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .foregroundStyle(StudioTheme.ink)
        .background { StudioAurora(animated: false).background(StudioTheme.canvas) }
        .onExitCommand(perform: goBack)
        .task(id: engine.result) { previewImage = engine.result.flatMap(NSImage.init(contentsOf:)) }
    }

    private func navigation(_ title: String, icon: String, history: Bool) -> some View {
        let selected = history == showsHistory
        return Button {
            selectedImage = nil
            showsHistory = history
        } label: {
            HStack(spacing: 11) {
                Image(systemName: icon).frame(width: 18)
                Text(title)
                Spacer()
                if !history && engine.running { ProgressView().controlSize(.mini) }
            }
            .font(.system(size: 13, weight: selected ? .semibold : .regular))
            .padding(.horizontal, 12).frame(height: 36)
        }
        .buttonStyle(StudioControlStyle(emphasized: selected))
        .accessibilityAddTraits(selected ? .isSelected : [])
        .accessibilityIdentifier(history ? "images.history" : "images.create")
    }

    private func goBack() {
        if showsHistory && selectedImage != nil { selectedImage = nil } else { close() }
    }

    private var modelPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            StudioCaption(french: "Modèle", english: "Model")
            Text("Qwen-Image-2.1").font(.headline)
            Text(modelPath.isEmpty ? L("Choisis le dossier EXL3 pour commencer.", "Choose the EXL3 folder to start.")
                 : URL(fileURLWithPath: modelPath).lastPathComponent)
                .font(.callout).foregroundStyle(StudioTheme.secondary)
                .lineLimit(2).truncationMode(.middle).help(modelPath)
            Button {
                let panel = NSOpenPanel()
                panel.canChooseFiles = false
                panel.canChooseDirectories = true
                if panel.runModal() == .OK, let url = panel.url { modelPath = url.path }
            } label: {
                Label(modelPath.isEmpty ? L("Choisir le dossier…", "Choose folder…") : L("Changer de dossier…", "Change folder…"),
                      systemImage: "folder").padding(.horizontal, 10).frame(height: 30)
            }
            .buttonStyle(StudioToolbarStyle()).disabled(engine.running)
            .accessibilityIdentifier("images.model")
        }
    }

    private var settings: some View {
        VStack(alignment: .leading, spacing: 10) {
            StudioCaption(french: "Réglages", english: "Settings")
            Text(L("Taille de l’image", "Image size")).font(.callout)
            Picker(L("Taille de l’image", "Image size"), selection: $engine.size) {
                ForEach([512, 768, 1024], id: \.self) { Text("\($0) px").tag($0) }
            }
            .pickerStyle(.segmented).labelsHidden()
            Stepper(value: $engine.steps, in: 2...50) {
                Text(L("\(engine.steps) étapes", "\(engine.steps) steps")).monospacedDigit()
            }
            .padding(.vertical, 4)
            Text(L("Graine", "Seed")).font(.callout)
            HStack {
                TextField(L("Graine", "Seed"), value: $engine.seed, format: .number.grouping(.never))
                    .monospacedDigit()
                Button { engine.seed = .random(in: 0...UInt32.max) } label: {
                    Image(systemName: "dice").frame(width: 28, height: 26)
                }
                .buttonStyle(StudioControlStyle()).accessibilityLabel(L("Nouvelle graine", "New seed"))
                .help(L("Nouvelle graine", "New seed"))
            }
        }
        .disabled(engine.running)
    }

    private var preview: some View {
        VStack(spacing: 10) {
            HStack {
                Text(engine.running ? L("Génération en cours", "Generating")
                     : engine.result == nil ? L("Exemple du modèle", "Model example") : L("Image générée", "Generated image"))
                    .font(.callout.weight(.medium)).foregroundStyle(StudioTheme.secondary)
                Spacer()
                if let url = engine.result {
                    Button { NSWorkspace.shared.activateFileViewerSelecting([url]) } label: {
                        Label(L("Afficher le PNG", "Show PNG"), systemImage: "folder")
                    }.buttonStyle(.link)
                }
            }
            ZStack {
                if engine.running {
                    VStack(spacing: 12) {
                        ProgressView().controlSize(.small)
                        Text(engine.status).font(.callout)
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
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(StudioTheme.edge) }
        }
        .frame(maxHeight: .infinity)
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField(L("Décris l’image à générer…", "Describe the image to generate…"),
                      text: $engine.prompt, axis: .vertical)
                .textFieldStyle(.plain).lineLimit(2...4).disabled(engine.running)
                .accessibilityIdentifier("images.prompt")
            if let error = engine.error {
                Text(error).font(.callout).foregroundStyle(StudioTheme.failure)
                    .textSelection(.enabled).lineLimit(3).help(error)
            } else if !engine.running && !engine.status.isEmpty {
                Text(engine.status).font(.callout).foregroundStyle(StudioTheme.secondary)
            }
            HStack(spacing: 12) {
                Text(L("Premier lancement : environ 5 Go à télécharger.", "First use downloads about 5 GB."))
                    .font(.caption).foregroundStyle(StudioTheme.quiet)
                Spacer()
                if engine.running {
                    Button(L("Annuler", "Cancel"), action: engine.cancel)
                } else {
                    Button {
                        engine.start(model: URL(fileURLWithPath: modelPath), prompt: engine.prompt,
                                     size: engine.size, steps: engine.steps, seed: engine.seed)
                    } label: {
                        Label(L("Générer", "Generate"), systemImage: "wand.and.stars")
                    }
                    .buttonStyle(.borderedProminent).keyboardShortcut(.return, modifiers: .command)
                    .disabled(modelPath.isEmpty || engine.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    .accessibilityIdentifier("images.generate")
                }
            }
        }
        .padding(14)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(StudioTheme.edge).allowsHitTesting(false) }
    }
}
