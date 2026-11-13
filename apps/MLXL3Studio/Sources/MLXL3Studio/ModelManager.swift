import AppKit
import SwiftUI

struct ModelSelectionMenu: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        Menu {
            ForEach(studio.models) { model in
                Button { studio.selectModel(model.name) } label: {
                    if model.name == studio.selectedModelName {
                        Label(model.name, systemImage: "checkmark")
                    } else { Text(model.name) }
                }
            }
            if !studio.models.isEmpty { Divider() }
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
                    .font(.system(size: 11, weight: .medium))
                    .lineLimit(1).truncationMode(.middle)
                Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
                    .foregroundStyle(StudioTheme.quiet)
            }
            .padding(.horizontal, 12).frame(height: 32).frame(maxWidth: 210)
        }
        .menuStyle(.button).buttonStyle(WorkspaceControlStyle()).menuIndicator(.hidden)
        .foregroundStyle(StudioTheme.ink)
        .premiumGlass(radius: 100, materialOpacity: 0.76)
        .accessibilityIdentifier("toolbar.model")
        .fixedSize(horizontal: false, vertical: true)
        .disabled(studio.isGenerating || studio.localServer.active)
        .help(studio.engineState.label)
        .accessibilityLabel(L("Modèle : \(studio.selectedModelName ?? "aucun")", "Model: \(studio.selectedModelName ?? "none")"))
    }
}

struct ModelManagerView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var studio: StudioModel
    @State private var discover = false
    @State private var quantize = false
    @State private var removal: LocalModel?

    var body: some View {
        VStack(spacing: 0) {
            StudioPanelHeader(
                title: L("Modèles", "Models"),
                detail: L(
                    "Bibliothèque, téléchargements et quantification.",
                    "Library, downloads and quantization."),
                close: { studio.showModelManager = false })
            Divider().overlay(StudioTheme.edge)
            HStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 8) {
                    DownloadNotificationButton(library: studio.modelLibrary)
                        .padding(.bottom, 8)
                    navigation(L("Ma bibliothèque", "My library"), icon: "square.stack", active: !discover && !quantize)
                    {
                        discover = false
                        quantize = false
                    }
                    navigation("Hugging Face", icon: "magnifyingglass", active: discover && !quantize) {
                        discover = true
                        quantize = false
                    }
                    navigation("Quantization Studio", icon: "slider.horizontal.3", active: quantize) {
                        quantize = true
                        discover = false
                    }
                    Spacer()
                    Text(L("Stockage local", "Local storage")).font(.system(size: 12, weight: .medium))
                        .foregroundStyle(StudioTheme.quiet)
                    Text(
                        ByteCountFormatter.string(
                            fromByteCount: studio.models.reduce(0) { $0 + $1.sizeBytes }, countStyle: .file)
                    )
                    .font(.system(size: 27, weight: .regular, design: .serif))
                    Text(L("\(studio.models.count) modèles enregistrés", "\(studio.models.count) registered models"))
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                }.padding(20).frame(width: 222).background(StudioTheme.sidebar)
                Divider().overlay(StudioTheme.edge)
                Group {
                    if quantize {
                        QuantizationStudioView(quantizer: studio.quantizer)
                    } else if discover {
                        HubBrowserView(library: studio.modelLibrary)
                    } else {
                        installed
                    }
                }
                .id(quantize ? "quantize" : discover ? "discover" : "library")
                .transition(.opacity)
            }
        }
        .foregroundStyle(StudioTheme.ink).background(StudioTheme.canvas)
        .onAppear { if studio.quantizer.converting { quantize = true } }
        .task { if !studio.isPreview { studio.modelLibrary.refreshPending() } }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onExitCommand { studio.showModelManager = false }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: [discover, quantize])
        .confirmationDialog(
            L("Supprimer ce modèle ?", "Remove this model?"),
            isPresented: Binding(
                get: { removal != nil }, set: { if !$0 { removal = nil } }
            ), titleVisibility: .visible
        ) {
            if let model = removal {
                Button(L("Placer le dossier dans la Corbeille", "Move folder to Trash"), role: .destructive) {
                    studio.removeLocalModel(model, trashFiles: true)
                }
                Button(L("Retirer de la bibliothèque uniquement", "Remove from library only")) {
                    studio.removeLocalModel(model, trashFiles: false)
                }
                Button(L("Annuler", "Cancel"), role: .cancel) { removal = nil }
            }
        } message: {
            if let model = removal {
                Text(
                    "\(model.name) · \(model.size)\n\(model.path)\n\n"
                        + L(
                            "La Corbeille conserve les fichiers jusqu’à ce que tu la vides.",
                            "Trash keeps the files until you empty it."))
            }
        }
    }

    private func navigation(_ title: String, icon: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: icon).font(.system(size: 12, weight: .medium))
                .frame(maxWidth: .infinity, alignment: .leading).padding(11)

        }.buttonStyle(StudioControlStyle(emphasized: active))
            .accessibilityAddTraits(active ? .isSelected : [])
    }

    private var installed: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    Text(L("Modèles installés", "Installed models")).font(
                        .system(size: 29, weight: .regular, design: .serif))
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
            }
            ScrollView {
                LazyVStack(spacing: 10) {
                    PendingDownloadsView(library: studio.modelLibrary)
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
                                        "\(model.architectureLabel) · EXL3 · "
                                            + (model.bits.map { String(format: "%.2f BPW", $0) } ?? "—")
                                    )
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                }
                                Spacer()
                                Text(model.size).font(.system(size: 18, weight: .regular, design: .serif))
                            }
                            HStack(spacing: 10) {
                                if studio.loadedModel?.name == model.name {
                                    Label(L("Chargé", "Loaded"), systemImage: "checkmark.circle").font(
                                        .system(size: 12)
                                    ).foregroundStyle(StudioTheme.secondary)
                                } else {
                                    Button(L("Charger", "Load")) { studio.selectModel(model.name) }.buttonStyle(
                                        .bordered
                                    ).controlSize(.regular)
                                        .disabled(studio.isGenerating || studio.modelInstallState.isWorking)
                                }
                                Spacer()
                                Button(L("Relocaliser", "Locate folder")) { studio.relocateModel(model) }
                                    .controlSize(.regular).disabled(
                                        studio.isGenerating || studio.modelInstallState.isWorking)
                                Button {
                                    NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)])
                                } label: {
                                    Image(systemName: "folder").frame(width: 28, height: 25)
                                }
                                .buttonStyle(RoundGlassButtonStyle()).help(L("Afficher dans Finder", "Show in Finder"))
                                Button {
                                    removal = model
                                } label: {
                                    Image(systemName: "trash").frame(width: 28, height: 25)
                                }
                                .buttonStyle(RoundGlassButtonStyle()).help(L("Supprimer le modèle", "Remove model"))
                                .disabled(
                                    studio.isGenerating || studio.modelInstallState.isWorking
                                        || studio.modelLibrary.downloading != nil)
                            }
                        }.padding(16).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
                            .overlay { RoundedRectangle(cornerRadius: 10).stroke(StudioTheme.edge, lineWidth: 0.5) }
                    }
                }
            }
            switch studio.modelInstallState {
            case .idle: EmptyView()
            case .working(let text), .succeeded(let text):
                Text(text).font(.caption).foregroundStyle(StudioTheme.secondary)
            case .failed(let text): Text(text).font(.caption).foregroundStyle(.orange).textSelection(.enabled)
            }
        }.padding(26).frame(maxWidth: .infinity, maxHeight: .infinity)
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
                Text(L("Modèles Hugging Face", "Hugging Face models")).font(
                    .system(size: 29, weight: .regular, design: .serif))
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
        }.padding(26).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
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
            Text(L("RECOMMANDÉS", "RECOMMENDED"))
                .font(.system(size: 12, weight: .medium)).tracking(1.4).foregroundStyle(StudioTheme.quiet)
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
                            Text(model.title).font(.system(size: 25, weight: .regular, design: .serif))
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
            Text(detail.id).font(.system(size: 23, weight: .regular, design: .serif)).lineLimit(2).textSelection(
                .enabled)
            HStack(alignment: .top, spacing: 14) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L("VERSION", "VERSION")).font(
                        .system(size: 12, weight: .medium)
                    ).tracking(1)
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
                            .system(size: 24, weight: .regular, design: .serif))
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
            Text(L("FICHE DU MODÈLE", "MODEL CARD")).font(.system(size: 12, weight: .medium)).tracking(1.4)
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
    @State private var removal: PendingDownload?
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
                    Button(L("Supprimer les fichiers partiels", "Delete partial files")) { removal = job }.disabled(
                        library.downloading != nil)
                }.font(.system(size: 12))
            }.padding(14).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
        }
        .confirmationDialog(
            L(
                "Supprimer le téléchargement incomplet ? Les octets devront être téléchargés à nouveau.",
                "Delete the incomplete download? These bytes will need to be downloaded again."),
            isPresented: Binding(get: { removal != nil }, set: { if !$0 { removal = nil } })
        ) {
            Button(L("Supprimer", "Delete"), role: .destructive) { if let removal { library.discard(removal) } }
        }
    }
}
