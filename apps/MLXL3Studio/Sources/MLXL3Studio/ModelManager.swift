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
            VStack(alignment: .leading, spacing: 0) {
                StudioSidebarHeader(title: L("Modèles", "Models"))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .overlay(alignment: .bottomTrailing) {
                        // Lives in the header row instead of floating under the navigation.
                        DownloadNotificationButton(library: studio.modelLibrary)
                            .padding(.trailing, 12).padding(.bottom, 22)
                    }
                VStack(spacing: 3) {
                    navigation(L("Ma bibliothèque", "My library"), icon: "square.stack", active: page == .library) {
                        page = .library
                    }
                    navigation("Hugging Face", icon: "magnifyingglass", active: page == .hub) { page = .hub }
                    navigation("Quantization Studio", icon: "slider.horizontal.3", active: page == .quantize) { page = .quantize }
                }.padding(.horizontal, 10)
                Spacer()
                navigation(L("Gérer le stockage", "Manage storage"), icon: "internaldrive", active: page == .storage) {
                    openStorage()
                }
                .accessibilityIdentifier("models.storage")
                .padding(.horizontal, 10).padding(.bottom, 10)
                StudioBackButton(action: { studio.showModelManager = false })
                    .padding(.horizontal, 18).padding(.bottom, 18)
            }
            .frame(width: studio.sidebarWidth.points, alignment: .leading)
            .modifier(StudioSidebarSurface())
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

    private func openStorage(_ focus: ModelStorageFocus? = nil) {
        storageFocus = focus
        page = .storage
    }

    private func showHubIfRequested() {
        guard studio.modelManagerShowsHub else { return }
        studio.modelManagerShowsHub = false
        page = .hub
    }

    private func navigation(_ title: String, icon: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            HStack(spacing: 11) {
                Image(systemName: icon).frame(width: 18)
                Text(title)
                Spacer(minLength: 0)
            }
            .font(.system(size: 13, weight: active ? .semibold : .regular))
            .padding(.horizontal, 12).frame(height: 36)

        }.buttonStyle(StudioControlStyle(emphasized: active))
            .accessibilityAddTraits(active ? .isSelected : [])
    }

    private var installed: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    StudioPanelHeader(title: L("Ma bibliothèque", "My library"), detail: L("Modèles EXL3 sur ce Mac.", "EXL3 models on this Mac."))
                }
                Spacer()
                Button(action: studio.importModelFolder) {
                    Label(L("Importer un dossier", "Import folder"), systemImage: "folder.badge.plus")
                        .padding(10)
                }
                .buttonStyle(GlassPillButtonStyle())
                .disabled(studio.modelInstallState.isWorking || studio.isGenerating)
                Button {
                    studio.refreshModels(autoLoad: false)
                } label: {
                    Image(systemName: "arrow.clockwise").frame(width: 30, height: 30)
                }
                .buttonStyle(RoundGlassButtonStyle()).help(L("Actualiser", "Refresh"))
                .accessibilityLabel(L("Actualiser", "Refresh"))
            }
            ScrollView {
                LazyVStack(spacing: 10) {
                    PendingDownloadsView(library: studio.modelLibrary) { openStorage(.download($0)) }
                    if studio.models.isEmpty {
                        Text(
                            L(
                                "Aucun modèle installé. Téléchargez un modèle EXL3 ou importez un dossier.",
                                "No models installed. Download an EXL3 model or import a folder.")
                        )
                        .foregroundStyle(StudioTheme.quiet).padding(.vertical, 50)
                    }
                    ForEach(studio.models) { model in
                        VStack(alignment: .leading, spacing: 13) {
                            HStack(alignment: .top, spacing: 12) {
                                Image(systemName: "cube.transparent").font(.system(size: 24, weight: .ultraLight))
                                    .foregroundStyle(StudioTheme.quiet).frame(width: 34, height: 34)
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(model.name).font(.system(size: 14, weight: .medium)).textSelection(.enabled)
                                    Text(
                                        (studio.isDecisionModel(model) ? L("DÉCISION · ", "DECISION · ") : "")
                                            + "\(model.architectureLabel) · EXL3 · "
                                            + (model.bits.map { String(format: "%.2f BPW", $0) } ?? "—")
                                    )
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                }
                                Spacer()
                                Text(model.size).font(StudioTheme.display(18))
                            }
                            HStack(spacing: 10) {
                                if studio.isDecisionModel(model) {
                                    Button { studio.openInDecision(model) } label: {
                                        Label(L("Ouvrir dans Décision", "Open in Decision"), systemImage: "checklist.checked")
                                            .font(.system(size: 12, weight: .medium))
                                            .padding(.horizontal, 12).frame(height: 26)
                                    }
                                    .buttonStyle(GlassPillButtonStyle())
                                    .disabled(studio.isGenerating || studio.modelInstallState.isWorking)
                                    .accessibilityIdentifier("library.openDecision")
                                } else if studio.loadedModel?.name == model.name {
                                    Label(L("Chargé", "Loaded"), systemImage: "checkmark.circle").font(
                                        .system(size: 12)
                                    ).foregroundStyle(StudioTheme.secondary)
                                } else {
                                    Button { studio.selectModel(model.name) } label: {
                                        Text(L("Charger", "Load")).font(.system(size: 12, weight: .medium))
                                            .padding(.horizontal, 12).frame(height: 26)
                                    }
                                    .buttonStyle(GlassPillButtonStyle())
                                        .disabled(studio.isGenerating || studio.modelInstallState.isWorking)
                                }
                                Spacer()
                                Button { studio.relocateModel(model) } label: {
                                    Text(L("Relocaliser", "Locate folder")).font(.system(size: 12))
                                        .padding(.horizontal, 10).frame(height: 26)
                                }
                                    .buttonStyle(StudioControlStyle()).disabled(
                                        studio.isGenerating || studio.modelInstallState.isWorking)
                                Button {
                                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)])
                                } label: {
                                    Image(systemName: "folder").frame(width: 28, height: 25)
                                }
                                .buttonStyle(RoundGlassButtonStyle()).help(L("Afficher dans Finder", "Show in Finder"))
                                .accessibilityLabel(L("Afficher dans Finder", "Show in Finder"))
                                Button {
                                    openStorage(.model(model.name))
                                } label: {
                                    Image(systemName: "trash").frame(width: 28, height: 25)
                                }
                                .buttonStyle(RoundGlassButtonStyle()).help(L("Examiner le retrait du modèle", "Review model removal"))
                                .accessibilityLabel(L("Examiner le retrait du modèle", "Review model removal"))
                                .disabled(studio.storageHasActiveWork || studio.modelInstallState.isWorking
                                          || studio.modelLibrary.discardingStorageID != nil)
                            }
                        }.padding(16).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                            .overlay { RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5) }
                    }
                }
            }
            switch studio.modelInstallState {
            case .idle: EmptyView()
            case .working(let text), .succeeded(let text):
                Text(text).font(.caption).foregroundStyle(StudioTheme.secondary)
            case .failed(let text): Text(text).font(.caption).foregroundStyle(.orange).textSelection(.enabled)
            }
        }.padding(.horizontal, 28).padding(.vertical, 36).frame(maxWidth: .infinity, maxHeight: .infinity)
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
                StudioPanelHeader(title: "Hugging Face", detail: L("Modèles EXL3 et téléchargements.", "EXL3 models and downloads."))
                HStack(spacing: 10) {
                    Image(systemName: "magnifyingglass").foregroundStyle(StudioTheme.quiet)
                    TextField(L("Rechercher sur Hugging Face…", "Search Hugging Face…"), text: $library.query)
                        .textFieldStyle(.plain).onSubmit { library.search() }
                    if library.searching { ProgressView().controlSize(.regular) }
                    Text("EXL3").font(.system(size: 12, weight: .medium)).tracking(1).foregroundStyle(StudioTheme.quiet)
                }.padding(12).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
                HStack {
                    Text(
                        L(
                            "Recherche EXL3 · affinez par nom ou auteur/dépôt.",
                            "EXL3 search · refine by name or owner/repository.")
                    )
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                    Spacer()
                    Button(action: library.refresh) { Image(systemName: "arrow.clockwise") }.help(
                        L("Actualiser", "Refresh"))
                }
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        if library.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            recommendedCards.padding(.bottom, 24)
                        }
                        ForEach(library.results) { model in
                            Button {
                                library.open(model.id)
                            } label: {
                                HStack(spacing: 14) {
                                    VStack(alignment: .leading, spacing: 5) {
                                        Text(model.id.split(separator: "/").last.map(String.init) ?? model.id).font(
                                            .system(size: 13, weight: .medium)
                                        ).lineLimit(2)
                                        Text(model.id.split(separator: "/").first.map(String.init) ?? "").font(
                                            .system(size: 12)
                                        ).foregroundStyle(StudioTheme.quiet)
                                    }
                                    Spacer()
                                    if model.gated { Image(systemName: "lock").font(.caption) }
                                    Label(model.downloads.formatted(), systemImage: "arrow.down").font(
                                        .system(size: 12)
                                    ).foregroundStyle(StudioTheme.quiet)
                                    Image(systemName: "chevron.right").font(.system(size: 12))
                                }.padding(.vertical, 16).padding(.horizontal, 10).frame(
                                    maxWidth: .infinity, alignment: .leading
                                ).contentShape(Rectangle())
                            }.buttonStyle(StudioControlStyle())
                            Divider().overlay(StudioTheme.edge)
                        }
                        if library.results.isEmpty && !library.searching && library.error == nil {
                            Text(
                                L(
                                    "Aucun résultat. Essayez un autre nom ou un dépôt exact.",
                                    "No results. Try another name or an exact repository.")
                            )
                            .font(.callout).foregroundStyle(StudioTheme.quiet).padding(.vertical, 40)
                        }
                        if library.results.count >= 60 {
                            Button(L("Afficher plus de résultats", "Show more results"), action: library.moreResults)
                                .disabled(library.searching || library.results.count >= 600)
                        }
                    }
                }
            }
            if let error = library.error { Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
        }.padding(.horizontal, 28).padding(.vertical, 36).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
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
            StudioCaption(french: "Recommandés", english: "Recommended")
            ForEach(ModelLibrary.recommendedFamilies) { family in
              VStack(alignment: .leading, spacing: 10) {
                Text(family.id).font(.system(size: 14, weight: .semibold)).foregroundStyle(StudioTheme.secondary)
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
                                Text("EXL3").font(.system(size: 12, weight: .medium)).tracking(1).foregroundStyle(
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
                            .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 14))
                            .overlay { RoundedRectangle(cornerRadius: 14).stroke(StudioTheme.edge, lineWidth: 0.5) }
                    }.buttonStyle(StudioControlStyle()).help(model.id)
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
                Button(action: library.back) { Label(L("Recherche", "Search"), systemImage: "arrow.left") }.buttonStyle(
                    .plain
                ).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                Spacer()
                Link("Hugging Face ↗", destination: URL(string: "https://huggingface.co/\(detail.id)")!).font(
                    .system(size: 12)
                ).foregroundStyle(StudioTheme.quiet)
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
                    Label(L("Télécharger", "Download"), systemImage: "arrow.down.to.line").font(
                        .system(size: 12, weight: .medium)
                    ).padding(.horizontal, 15).padding(.vertical, 12)
                }.buttonStyle(PrimaryGlassButtonStyle()).disabled(
                    library.selectedVariant == nil || library.downloading != nil || library.loadingDetail)
            }.padding(14).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
            if let variant = library.selectedVariant {
                if variant.preview?.modelType == "qwen4_exp" {
                    Text(
                        L(
                            "Flash Next utilise le streaming SSD : la taille du téléchargement n’est pas la RAM requise. Le moteur charge les experts à la demande dans un cache borné ; l’empreinte dépend du contexte et de la mémoire disponible.",
                            "Flash Next uses SSD streaming: download size is not the required RAM. The engine loads experts on demand into a bounded cache; memory use depends on context and available memory."
                        )
                    )
                    .font(.caption).foregroundStyle(StudioTheme.secondary)
                } else {
                    ModelAdviceView(sizeBytes: variant.sizeBytes, preview: variant.preview)
                }
            }
            if detail.variants.isEmpty {
                Text(
                    L(
                        "Pas de checkpoint EXL3 complet identifié ici. Essayez une branche BPW. Les poids ambigus partageant un même descripteur ne sont pas mélangés.",
                        "No complete EXL3 checkpoint identified here. Try a BPW branch. Ambiguous weights sharing one descriptor are not mixed."
                    )
                )
                .font(.caption).foregroundStyle(.orange)
            }
            if detail.gated {
                Text(
                    L(
                        "Dépôt à accès restreint : acceptez sa licence sur Hugging Face et connectez votre compte dans les Réglages.",
                        "Gated repository: accept its license on Hugging Face and sign in in Settings.")
                )
                .font(.caption).foregroundStyle(.orange)
            }
            StudioCaption(french: "Fiche du modèle", english: "Model card", colored: false)
                .foregroundStyle(StudioTheme.quiet)
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
            VStack(alignment: .leading, spacing: 8) {
                Text(job.repo).font(.system(size: 12, weight: .medium)).lineLimit(2)
                HStack {
                    Text(
                        ByteCountFormatter.string(fromByteCount: job.retained_bytes, countStyle: .file) + " · "
                            + L("interrompu", "interrupted"))
                    Spacer()
                    Button(L("Reprendre", "Resume")) { library.resume(job, studio: studio) }.disabled(
                        library.downloading != nil)
                    Button(L("Examiner les fichiers partiels…", "Review partial files…")) { reviewRemoval(job.id) }
                        .disabled(studio.storageHasActiveWork || studio.modelInstallState.isWorking || library.discardingStorageID != nil)
                }.font(.system(size: 12))
            }.padding(14).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
        }
    }
}
