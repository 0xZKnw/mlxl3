import AppKit
import SwiftUI

/// Unload button state beside the model selector: hidden with nothing to
/// release, disabled while work owns the engine, otherwise enabled.
enum ModelUnloadControl: Equatable {
    case hidden, disabled, enabled

    init(canEject: Bool, busy: Bool) {
        self = !canEject ? .hidden : busy ? .disabled : .enabled
    }
}

struct ModelSelectionMenu: View {
    @EnvironmentObject private var studio: StudioModel

    private var unloadHelp: String {
        if studio.modelUnloadControl != .enabled {
            return L("Arrêtez la tâche en cours pour décharger le modèle", "Stop the current task to unload the model")
        }
        if case .loading = studio.engineState {
            return L("Annuler le chargement et libérer la mémoire", "Cancel loading and free memory")
        }
        return L("Décharger le modèle et libérer la mémoire", "Unload the model and free memory")
    }

    var body: some View {
        HStack(spacing: 3) {
            Menu {
                ForEach(studio.chatModels) { model in
                    Button { studio.selectModel(model.name) } label: {
                        if model.name == studio.selectedModelName {
                            Label(model.name, systemImage: "checkmark")
                        } else { Text(model.name) }
                    }
                }
                if !studio.chatModels.isEmpty { Divider() }
                Button(action: studio.openModelManager) {
                    Label(L("Ajouter un modèle…", "Add a model…"), systemImage: "square.and.arrow.down")
                }
                Button(action: studio.ejectModel) {
                    Label(L("Décharger le modèle", "Unload model"), systemImage: "eject")
                }.disabled(!studio.canEject)
            } label: {
                HStack(spacing: 7) {
                    StatusDot(state: studio.engineState)
                    Text(studio.selectedModelName ?? L("Choisir un modèle", "Choose a model"))
                        .font(.system(size: StudioTheme.style.isGraphite ? 12.5 : 11, weight: .medium))
                        .lineLimit(1).truncationMode(.middle)
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
                        .foregroundStyle(StudioTheme.quiet)
                }
                .padding(.horizontal, 10).frame(height: 30).frame(maxWidth: 210)
            }
            .menuStyle(.button).buttonStyle(WorkspaceControlStyle()).menuIndicator(.hidden)
            .accessibilityIdentifier("toolbar.model")
            .fixedSize(horizontal: false, vertical: true)
            .disabled(studio.composerBusy || studio.localServer.active)
            .help(studio.engineState.label)
            .accessibilityLabel(L("Modèle : \(studio.selectedModelName ?? "aucun")", "Model: \(studio.selectedModelName ?? "none")"))

            if studio.modelUnloadControl != .hidden {
                Button(action: studio.unloadModelFromToolbar) {
                    Image(systemName: "eject")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 30, height: 30)
                }
                .buttonStyle(WorkspaceControlStyle())
                .disabled(studio.modelUnloadControl != .enabled)
                .accessibilityIdentifier("toolbar.unload")
                .help(unloadHelp)
                .accessibilityLabel(L("Décharger le modèle", "Unload model"))
                .accessibilityHint(unloadHelp)
            }
        }
        .foregroundStyle(StudioTheme.ink)
        .padding(StudioTheme.style.isGraphite ? 0 : 4)
        // Graphite: no glass pill on a plain canvas.
        .background { if !StudioTheme.style.isGraphite { Color.clear.premiumGlass(radius: 100, materialOpacity: 0.76) } }
    }
}

struct ModelManagerView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var studio: StudioModel
    private enum Page { case library, hub, quantize, storage }
    @State private var page: Page = .library
    @State private var storageFocus: ModelStorageFocus?

    var body: some View {
        HStack(spacing: 0) {
            StudioPageSidebar(title: L("Modèles", "Models"), onBack: { studio.showModelManager = false }) {
                StudioNavList {
                    StudioNavRow(icon: "square.stack", title: L("Ma bibliothèque", "My library"), selected: page == .library,
                                 badge: studio.models.isEmpty ? nil : "\(studio.models.count)") { page = .library }
                    StudioNavRow(icon: "magnifyingglass", title: "Hugging Face", selected: page == .hub) { page = .hub }
                    StudioNavRow(icon: "slider.horizontal.3", title: "Quantization Studio", selected: page == .quantize) {
                        page = .quantize
                    }
                }
            } footer: {
                // The space the registered models take is the way into its management page.
                StudioNavList {
                    StudioNavRow(icon: "internaldrive", title: L("Gérer le stockage", "Manage storage"), selected: page == .storage,
                                 badge: studio.models.isEmpty ? nil
                                     : ByteCountFormatter.string(fromByteCount: studio.models.reduce(0) { $0 + $1.sizeBytes }, countStyle: .file)) {
                        openStorage()
                    }
                    .accessibilityIdentifier("models.storage")
                }
            }
            .overlay(alignment: .topTrailing) {
                // Lives in the header row instead of floating under the navigation.
                DownloadNotificationButton(library: studio.modelLibrary)
                    .padding(.trailing, 10).padding(.top, 38)
            }
            Group {
                switch page {
                case .quantize: QuantizationStudioView(quantizer: studio.quantizer)
                case .hub: HubBrowserView(library: studio.modelLibrary)
                case .library: installed
                case .storage: ModelStorageView(library: studio.modelLibrary, comparison: studio.labComparison, initialFocus: storageFocus)
                }
            }
            .frame(maxWidth: 880)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .id(page)
            .transition(.opacity)
        }
        .foregroundStyle(StudioTheme.ink)
        .background { StudioAurora(animated: true).background(StudioTheme.canvas) }
        .onAppear {
            if studio.quantizer.converting { page = .quantize } else { showHubIfRequested() }
        }
        .onChange(of: studio.modelManagerShowsHub) { showHubIfRequested() }
        .task { if !studio.isPreview { studio.modelLibrary.refreshPending() } }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onExitCommand { studio.showModelManager = false }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: page)
    }

    private func showHubIfRequested() {
        guard studio.modelManagerShowsHub else { return }
        studio.modelManagerShowsHub = false
        page = .hub
    }

    private var working: Bool { studio.isGenerating || studio.modelInstallState.isWorking }

    /// Removal is reviewed on the storage page: what the model takes, what it shares, what comes back.
    private var removalBlocked: Bool {
        studio.storageHasActiveWork || studio.modelInstallState.isWorking || studio.modelLibrary.discardingStorageID != nil
    }

    private func openStorage(_ focus: ModelStorageFocus? = nil) {
        storageFocus = focus
        page = .storage
    }

    private var installed: some View {
        VStack(alignment: .leading, spacing: 18) {
            StudioPageHeader(title: L("Ma bibliothèque", "My library"), detail: L("Modèles EXL3 sur ce Mac.", "EXL3 models on this Mac.")) {
                Button(action: studio.importModelFolder) {
                    Label(L("Importer un dossier", "Import folder"), systemImage: "folder.badge.plus")
                }
                .buttonStyle(.studio)
                .disabled(working)
                StudioIconButton(icon: "arrow.clockwise", label: L("Actualiser", "Refresh")) {
                    studio.refreshModels(autoLoad: false)
                }
            }
            switch studio.modelInstallState {
            case .idle: EmptyView()
            case .working(let text): StudioNotice(text: text)
            case .succeeded(let text): StudioNotice(kind: .success, text: text)
            case .failed(let text): StudioNotice(kind: .warning, text: text)
            }
            ScrollView {
                LazyVStack(spacing: 10) {
                    PendingDownloadsView(library: studio.modelLibrary) { openStorage(.download($0)) }
                    if studio.models.isEmpty {
                        StudioEmptyState(
                            icon: "square.stack",
                            title: L("Aucun modèle installé", "No models installed"),
                            detail: L(
                                "Aucun modèle installé. Téléchargez un modèle EXL3 ou importez un dossier.",
                                "No models installed. Download an EXL3 model or import a folder.")
                        ) {
                            Button(L("Parcourir Hugging Face", "Browse Hugging Face")) { page = .hub }
                            .buttonStyle(.studio(.primary))
                            Button(L("Importer un dossier", "Import folder"), action: studio.importModelFolder)
                                .buttonStyle(.studio).disabled(working)
                        }
                        .studioCard(padding: 0)
                    }
                    ForEach(studio.models) { model in
                        modelRow(model)
                    }
                }
            }
        }.padding(.horizontal, 32).padding(.top, 40).padding(.bottom, 24).frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func modelRow(_ model: LocalModel) -> some View {
        let loaded = studio.loadedModel?.name == model.name
        let decision = studio.isDecisionModel(model)
        let prefix = decision ? L("DÉCISION · ", "DECISION · ") : ""
        let bits = model.bits.map { String(format: "%.2f BPW", $0) } ?? "—"
        let details = "\(prefix)\(model.architectureLabel) · EXL3 · \(bits) · \(model.size)"
        return HStack(alignment: .center, spacing: 13) {
            Image(systemName: decision ? "checklist.checked" : "cube.transparent")
                .font(.system(size: 17, weight: .light))
                .foregroundStyle(loaded ? StudioTheme.ink : StudioTheme.secondary)
                .frame(width: 38, height: 38)
                .background(StudioTheme.contrast.opacity(0.055), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text(model.name).font(.system(size: 14, weight: .medium)).lineLimit(1).truncationMode(.middle)
                        .textSelection(.enabled)
                    if loaded {
                        StudioStatusPill(text: L("Chargé", "Loaded"), color: StudioTheme.success)
                    }
                }
                Text(verbatim: details)
                    .font(.system(size: 12)).monospacedDigit().foregroundStyle(StudioTheme.quiet).lineLimit(1)
            }
            Spacer(minLength: 8)
            if decision {
                Button { studio.openInDecision(model) } label: {
                    Label(L("Ouvrir dans Décision", "Open in Decision"), systemImage: "checklist.checked")
                }
                .buttonStyle(.studio)
                .disabled(working)
                .accessibilityIdentifier("library.openDecision")
            } else if !loaded {
                Button(L("Charger", "Load")) { studio.selectModel(model.name) }
                    .buttonStyle(.studio)
                    .disabled(working)
            }
            Menu {
                Button(L("Afficher dans Finder", "Show in Finder"), systemImage: "folder") {
                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)])
                }
                Button(L("Relocaliser", "Locate folder"), systemImage: "arrow.triangle.branch") { studio.relocateModel(model) }
                    .disabled(working)
                Divider()
                Button(L("Examiner le retrait du modèle…", "Review model removal…"), systemImage: "trash") {
                    openStorage(.model(model.name))
                }
                .disabled(removalBlocked)
            } label: {
                Image(systemName: "ellipsis").font(.system(size: 13, weight: .medium))
                    .frame(width: 28, height: 28).contentShape(Rectangle())
            }
            .menuStyle(.button).buttonStyle(StudioRowStyle()).menuIndicator(.hidden).fixedSize()
            .help(L("Actions du modèle", "Model actions"))
            .accessibilityLabel(L("Actions pour ", "Actions for ") + model.name)
        }
        .studioCard(padding: 14)
        .contextMenu {
            Button(L("Afficher dans Finder", "Show in Finder"), systemImage: "folder") {
                NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)])
            }
            Button(L("Examiner le retrait du modèle…", "Review model removal…"), systemImage: "trash") {
                openStorage(.model(model.name))
            }
            .disabled(removalBlocked)
        }
    }
}

private struct HubBrowserView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var library: ModelLibrary

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let detail = library.detail {
                details(detail)
            } else {
                StudioPageHeader(title: "Hugging Face", detail: L("Modèles EXL3 et téléchargements.", "EXL3 models and downloads.")) {
                    StudioIconButton(icon: "arrow.clockwise", label: L("Actualiser", "Refresh"), action: library.refresh)
                }
                HStack(spacing: 9) {
                    Image(systemName: "magnifyingglass").font(.system(size: 13)).foregroundStyle(StudioTheme.quiet)
                    TextField(L("Rechercher sur Hugging Face…", "Search Hugging Face…"), text: $library.query)
                        .textFieldStyle(.plain).onSubmit { library.search() }
                    if library.searching { ProgressView().controlSize(.small) }
                    Text("EXL3").font(.system(size: 11, weight: .semibold)).foregroundStyle(StudioTheme.quiet)
                        .padding(.horizontal, 6).frame(height: 20)
                        .background(StudioTheme.contrast.opacity(0.07), in: RoundedRectangle(cornerRadius: 5, style: .continuous))
                }
                .studioField()
                .help(L(
                    "Recherche EXL3 · affinez par nom ou auteur/dépôt.",
                    "EXL3 search · refine by name or owner/repository."))
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        if library.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            recommendedCards.padding(.bottom, 24)
                            if !library.results.isEmpty {
                                StudioSectionTitle(title: L("Tous les modèles EXL3", "All EXL3 models"))
                                    .padding(.bottom, 9)
                            }
                        }
                        LazyVStack(alignment: .leading, spacing: 0) {
                            ForEach(Array(library.results.enumerated()), id: \.element.id) { index, model in
                                if index > 0 {
                                    Rectangle().fill(StudioTheme.edge.opacity(0.7)).frame(height: 0.5).accessibilityHidden(true)
                                }
                                Button {
                                    library.open(model.id)
                                } label: {
                                    HStack(spacing: 14) {
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text(model.id.split(separator: "/").last.map(String.init) ?? model.id).font(
                                                .system(size: 13, weight: .medium)
                                            ).lineLimit(2).multilineTextAlignment(.leading)
                                            Text(model.id.split(separator: "/").first.map(String.init) ?? "").font(
                                                .system(size: 12)
                                            ).foregroundStyle(StudioTheme.quiet)
                                        }
                                        Spacer()
                                        if model.gated {
                                            Image(systemName: "lock").font(.system(size: 11)).foregroundStyle(StudioTheme.quiet)
                                                .help(L("Accès restreint", "Gated repository"))
                                        }
                                        Label(model.downloads.formatted(), systemImage: "arrow.down").font(
                                            .system(size: 12)
                                        ).monospacedDigit().foregroundStyle(StudioTheme.quiet)
                                        Image(systemName: "chevron.right").font(.system(size: 10, weight: .semibold))
                                            .foregroundStyle(StudioTheme.quiet)
                                    }.padding(.vertical, 12).padding(.horizontal, 14).frame(
                                        maxWidth: .infinity, alignment: .leading
                                    ).contentShape(Rectangle())
                                }.buttonStyle(StudioRowStyle(radius: 0))
                            }
                        }
                        .clipShape(RoundedRectangle(cornerRadius: StudioPageMetrics.card, style: .continuous))
                        .studioCard(padding: 0)
                        .opacity(library.results.isEmpty ? 0 : 1)
                        if library.results.isEmpty && !library.searching && library.error == nil {
                            StudioEmptyState(
                                icon: "magnifyingglass",
                                title: L("Aucun résultat", "No results"),
                                detail: L(
                                    "Aucun résultat. Essayez un autre nom ou un dépôt exact.",
                                    "No results. Try another name or an exact repository."))
                        }
                        if library.results.count >= 60 {
                            Button(L("Afficher plus de résultats", "Show more results"), action: library.moreResults)
                                .buttonStyle(.studio)
                                .disabled(library.searching || library.results.count >= 600)
                                .frame(maxWidth: .infinity).padding(.top, 12)
                        }
                    }
                }
            }
            if let error = library.error { StudioNotice(kind: .warning, text: error) }
        }.padding(.horizontal, 32).padding(.top, 40).padding(.bottom, 24).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .overlay {
                if library.loadingDetail {
                    ProgressView().padding(20).background(
                        StudioTheme.canvas, in: RoundedRectangle(cornerRadius: 12))
                }
            }
            .onChange(of: library.query) { library.search() }
            .onAppear { if library.results.isEmpty { library.search() } }
    }

    private var recommendedCards: some View {
        VStack(alignment: .leading, spacing: 12) {
            StudioSectionTitle(title: L("Recommandés", "Recommended"), icon: "sparkles")
            ForEach(ModelLibrary.recommendedFamilies) { family in
              VStack(alignment: .leading, spacing: 9) {
                Text(family.id).font(.system(size: 12, weight: .medium)).foregroundStyle(StudioTheme.quiet)
                LazyVGrid(columns: [GridItem(.adaptive(minimum: 170), spacing: 12)], spacing: 12) {
                  ForEach(family.models) { model in
                    Button {
                        library.open(model.id, revision: model.revision, preferredVariant: model.preferredVariant)
                    } label: {
                        VStack(alignment: .leading, spacing: 9) {
                            HStack {
                                Image(systemName: model.symbol).font(.system(size: 21, weight: .light)).foregroundStyle(
                                    StudioTheme.accent)
                                Spacer()
                                Text("EXL3").font(.system(size: 11, weight: .semibold)).foregroundStyle(
                                    StudioTheme.quiet)
                            }
                            Spacer(minLength: 6)
                            Text(model.title).font(StudioTheme.display(25))
                            Text(model.parameters).font(.system(size: 13, weight: .medium))
                            Text(model.quantization).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                            HStack {
                                Text(L("Voir le modèle", "View model")).font(.system(size: 12))
                                Spacer()
                                Image(systemName: "arrow.up.right").font(.system(size: 12))
                            }.foregroundStyle(StudioTheme.quiet).padding(.top, 3)
                        }.padding(16).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                            .aspectRatio(1, contentMode: .fit)
                            .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(StudioTheme.edge.opacity(0.7), lineWidth: 0.5) }
                    }.buttonStyle(StudioControlStyle(cornerRadius: 12)).help(model.id)
                  }
                }
              }
              .padding(.top, 8)
            }
        }
    }

    private func details(_ detail: HubDetails) -> some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Button(action: library.back) { Label(L("Recherche", "Search"), systemImage: "chevron.left") }
                    .buttonStyle(.studio(.quiet, height: 26))
                Spacer()
                Link("Hugging Face ↗", destination: URL(string: "https://huggingface.co/\(detail.id)")!).font(
                    .system(size: 12, weight: .medium)
                ).foregroundStyle(StudioTheme.secondary)
            }
            Text(detail.id).font(StudioTheme.display(23)).lineLimit(2).textSelection(
                .enabled)
            HStack(alignment: .top, spacing: 14) {
                VStack(alignment: .leading, spacing: 6) {
                    StudioCaption(french: "Version", english: "Version", size: 12, tracking: 1, colored: false)
                    Picker(
                        L("Révision ou branche", "Revision or branch"),
                        selection: Binding(get: { detail.revision }, set: { library.open(detail.id, revision: $0) })
                    ) {
                        ForEach([detail.revision] + detail.branches.filter { $0 != detail.revision }, id: \.self) {
                            Text($0.count == 40 ? L("Révision ", "Revision ") + String($0.prefix(8)) : $0).tag($0)
                        }
                    }.labelsHidden()
                    let choices = ModelLibrary.recommendedChoices(for: detail.id)
                    if choices.count > 1 {
                        Picker(L("Quantification", "Quantization"), selection: Binding(
                            get: { detail.id },
                            set: { library.selectRecommendedChoice($0, from: detail.id) }
                        )) {
                            ForEach(choices) { Text($0.label).tag($0.id) }
                        }
                    }
                    if detail.variants.count > 1 {
                        Picker(L("Variante", "Variant"), selection: $library.variantID) {
                            ForEach(detail.variants) { Text($0.label).tag($0.id) }
                        }
                    }
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    if let variant = library.selectedVariant {
                        Text(ByteCountFormatter.string(fromByteCount: variant.sizeBytes, countStyle: .file)).font(
                            StudioTheme.display(24))
                        Text(L("à télécharger", "to download")).font(.system(size: 12)).foregroundStyle(
                            StudioTheme.quiet)
                    }
                }
                Button {
                    library.download(studio: studio)
                } label: {
                    Label(L("Télécharger", "Download"), systemImage: "arrow.down.to.line")
                }.buttonStyle(.studio(.primary, height: 36)).disabled(
                    library.selectedVariant == nil || library.downloading != nil || library.loadingDetail)
            }.studioCard(padding: 14)
            if let variant = library.selectedVariant {
                if variant.preview?.modelType == "qwen4_exp" {
                    Text(
                        L(
                            "Flash Next utilise le streaming SSD : la taille du téléchargement n’est pas la RAM requise. Le moteur charge les experts à la demande dans un cache borné ; l’empreinte dépend du contexte et de la mémoire disponible.",
                            "Flash Next uses SSD streaming: download size is not the required RAM. The engine loads experts on demand into a bounded cache; memory use depends on context and available memory."
                        )
                    )
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                } else {
                    ModelAdviceView(sizeBytes: variant.sizeBytes, preview: variant.preview)
                }
            }
            if detail.variants.isEmpty {
                StudioNotice(kind: .warning, text: L(
                    "Pas de checkpoint EXL3 complet identifié ici. Essayez une branche BPW. Les poids ambigus partageant un même descripteur ne sont pas mélangés.",
                    "No complete EXL3 checkpoint identified here. Try a BPW branch. Ambiguous weights sharing one descriptor are not mixed."
                ))
            }
            if detail.gated {
                StudioNotice(kind: .warning, text: L(
                    "Dépôt à accès restreint : acceptez sa licence sur Hugging Face et connectez votre compte dans les Réglages.",
                    "Gated repository: accept its license on Hugging Face and sign in in Settings."))
            }
            StudioSectionTitle(title: L("Fiche du modèle", "Model card"), icon: "doc.text")
            ModelCardView(
                source: detail.readme.isEmpty ? L("Aucun README disponible.", "No README available.") : detail.readme,
                repo: detail.id, commit: detail.commit
            ).frame(maxWidth: .infinity, maxHeight: .infinity).layoutPriority(1)
            Text(
                L(
                    "EXL3 indique le format, pas une garantie de compatibilité de l’architecture.",
                    "EXL3 describes the format, not a guarantee of architecture compatibility.")
            )
            .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
        }
    }
}

private struct PendingDownloadsView: View {
    @ObservedObject var library: ModelLibrary
    @EnvironmentObject private var studio: StudioModel
    let reviewRemoval: (String) -> Void
    var body: some View {
        ForEach(library.pending) { job in
            HStack(alignment: .center, spacing: 12) {
                Image(systemName: "pause.circle").font(.system(size: 17, weight: .light)).foregroundStyle(StudioTheme.warning)
                    .frame(width: 38, height: 38)
                    .background(StudioTheme.contrast.opacity(0.055), in: RoundedRectangle(cornerRadius: 10, style: .continuous))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 4) {
                    Text(job.repo).font(.system(size: 13, weight: .medium)).lineLimit(2)
                    Text(
                        ByteCountFormatter.string(fromByteCount: job.retained_bytes, countStyle: .file) + " · "
                            + L("interrompu", "interrupted"))
                        .font(.system(size: 12)).monospacedDigit().foregroundStyle(StudioTheme.quiet)
                }
                Spacer(minLength: 8)
                Button(L("Reprendre", "Resume")) { library.resume(job, studio: studio) }
                    .buttonStyle(.studio(.primary)).disabled(library.downloading != nil)
                StudioIconButton(icon: "trash", label: L("Examiner les fichiers partiels…", "Review partial files…")) {
                    reviewRemoval(job.id)
                }
                .disabled(studio.storageHasActiveWork || studio.modelInstallState.isWorking || library.discardingStorageID != nil)
            }.studioCard(padding: 14)
        }
    }
}
