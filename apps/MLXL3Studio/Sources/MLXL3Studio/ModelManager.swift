import AppKit
import SwiftUI

struct ModelManagerView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.dismiss) private var dismiss
    @State private var discover = false
    @State private var removal: LocalModel?

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 18) {
                MonogramMark(size: 28)
                Text(L("Modèles", "Models")).font(.system(size: 36, weight: .regular, design: .serif))
                Spacer()
                Button(action: studio.importModelFolder) {
                    Label(L("Importer un dossier", "Import folder"), systemImage: "folder.badge.plus")
                        .padding(.horizontal, 12).padding(.vertical, 9)
                }.buttonStyle(GlassPillButtonStyle()).disabled(studio.modelInstallState.isWorking)
                Button(action: dismiss.callAsFunction) { Image(systemName: "xmark").frame(width: 32, height: 32) }
                    .buttonStyle(RoundGlassButtonStyle()).accessibilityLabel(L("Fermer", "Close"))
            }.padding(.horizontal, 28).padding(.vertical, 22)
            Divider().overlay(StudioTheme.edge)
            HStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 8) {
                    navigation(L("Ma bibliothèque", "My library"), icon: "square.stack", active: !discover) { discover = false }
                    navigation(L("Découvrir", "Discover"), icon: "magnifyingglass", active: discover) { discover = true }
                    Spacer()
                    Text(L("SUR CE MAC", "ON THIS MAC")).font(.system(size: 9, weight: .medium)).tracking(1.5).foregroundStyle(StudioTheme.quiet)
                    Text(ByteCountFormatter.string(fromByteCount: studio.models.reduce(0) { $0 + $1.sizeBytes }, countStyle: .file))
                        .font(.system(size: 27, weight: .regular, design: .serif))
                    Text(L("\(studio.models.count) modèles enregistrés", "\(studio.models.count) registered models"))
                        .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                }.padding(18).frame(width: 180).background(StudioTheme.sidebar)
                Divider().overlay(StudioTheme.edge)
                if discover { HubBrowserView(library: studio.modelLibrary) } else { installed }
            }
            LibraryDownloadView(library: studio.modelLibrary)
        }
        .foregroundStyle(StudioTheme.ink).background(StudioTheme.canvas)
        .frame(width: min(1040, (NSScreen.main?.visibleFrame.width ?? 1200) - 60),
               height: min(740, (NSScreen.main?.visibleFrame.height ?? 900) - 90))
        .confirmationDialog(L("Supprimer ce modèle ?", "Remove this model?"), isPresented: Binding(
            get: { removal != nil }, set: { if !$0 { removal = nil } }
        ), titleVisibility: .visible) {
            if let model = removal {
                Button(L("Placer le dossier dans la Corbeille", "Move folder to Trash"), role: .destructive) { studio.removeLocalModel(model, trashFiles: true) }
                Button(L("Retirer de la bibliothèque uniquement", "Remove from library only")) { studio.removeLocalModel(model, trashFiles: false) }
                Button(L("Annuler", "Cancel"), role: .cancel) { removal = nil }
            }
        } message: {
            if let model = removal {
                Text("\(model.name) · \(model.size)\n\(model.path)\n\n" + L("La Corbeille conserve les fichiers jusqu’à ce que tu la vides.", "Trash keeps the files until you empty it."))
            }
        }
    }

    private func navigation(_ title: String, icon: String, active: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: icon).font(.system(size: 12, weight: .medium))
                .frame(maxWidth: .infinity, alignment: .leading).padding(11)
                .background(active ? Color.white.opacity(0.065) : .clear, in: RoundedRectangle(cornerRadius: 7))
        }.buttonStyle(StudioControlStyle())
    }

    private var installed: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                VStack(alignment: .leading, spacing: 5) {
                    Text(L("Votre collection.", "Your collection.")).font(.system(size: 29, weight: .regular, design: .serif))
                    Text(L("Des modèles locaux, prêts à vous répondre.", "Local models, ready when you are."))
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                }
                Spacer()
                Button { studio.refreshModels(autoLoad: false) } label: { Image(systemName: "arrow.clockwise").frame(width: 30, height: 30) }
                    .buttonStyle(RoundGlassButtonStyle()).help(L("Actualiser", "Refresh"))
            }
            ScrollView {
                LazyVStack(spacing: 10) {
                    if studio.models.isEmpty {
                        Text(L("Aucun modèle installé. Découvrez les checkpoints EXL3 ou importez un dossier.", "No models installed. Discover EXL3 checkpoints or import a folder."))
                            .foregroundStyle(StudioTheme.quiet).padding(.vertical, 50)
                    }
                    ForEach(studio.models) { model in
                        VStack(alignment: .leading, spacing: 13) {
                            HStack(alignment: .top, spacing: 12) {
                                Image(systemName: "cube.transparent").font(.system(size: 24, weight: .ultraLight)).foregroundStyle(StudioTheme.quiet).frame(width: 34, height: 34)
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(model.name).font(.system(size: 14, weight: .medium)).textSelection(.enabled)
                                    Text("\(model.architectureLabel) · EXL3 · " + (model.bits.map { String(format: "%.2f BPW", $0) } ?? "—"))
                                        .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
                                }
                                Spacer()
                                Text(model.size).font(.system(size: 18, weight: .regular, design: .serif))
                            }
                            HStack(spacing: 10) {
                                if studio.loadedModel?.name == model.name {
                                    Label(L("Chargé", "Loaded"), systemImage: "checkmark.circle").font(.system(size: 10)).foregroundStyle(StudioTheme.secondary)
                                } else {
                                    Button(L("Charger", "Load")) { studio.selectModel(model.name) }.buttonStyle(.bordered).controlSize(.small)
                                        .disabled(studio.isGenerating || studio.modelInstallState.isWorking)
                                }
                                Spacer()
                                Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.path)]) } label: { Image(systemName: "folder").frame(width: 28, height: 25) }
                                    .buttonStyle(RoundGlassButtonStyle()).help(L("Afficher dans Finder", "Show in Finder"))
                                Button { removal = model } label: { Image(systemName: "trash").frame(width: 28, height: 25) }
                                    .buttonStyle(RoundGlassButtonStyle()).help(L("Supprimer le modèle", "Remove model"))
                                    .disabled(studio.isGenerating || studio.modelInstallState.isWorking || studio.modelLibrary.downloading != nil)
                            }
                        }.padding(16).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
                            .overlay { RoundedRectangle(cornerRadius: 10).stroke(StudioTheme.edge, lineWidth: 0.5) }
                    }
                }
            }
            switch studio.modelInstallState {
            case .idle: EmptyView()
            case let .working(text), let .succeeded(text): Text(text).font(.caption).foregroundStyle(StudioTheme.secondary)
            case let .failed(text): Text(text).font(.caption).foregroundStyle(.orange).textSelection(.enabled)
            }
        }.padding(26).frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct HubBrowserView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var library: ModelLibrary

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let detail = library.detail { details(detail) }
            else {
                Text(L("Trouvez votre prochain modèle.", "Find your next model.")).font(.system(size: 29, weight: .regular, design: .serif))
                HStack(spacing: 10) {
                    Image(systemName: "magnifyingglass").foregroundStyle(StudioTheme.quiet)
                    TextField(L("Rechercher sur Hugging Face…", "Search Hugging Face…"), text: $library.query).textFieldStyle(.plain).onSubmit { library.search() }
                    if library.searching { ProgressView().controlSize(.small) }
                    Text("EXL3").font(.system(size: 9, weight: .medium)).tracking(1).foregroundStyle(StudioTheme.quiet)
                }.padding(12).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
                Text(L("Recherche EXL3 · popularité · 60 résultats maximum. Affinez avec un nom ou collez auteur/dépôt.", "EXL3 search · popularity · up to 60 results. Refine by name or paste owner/repository."))
                    .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(library.results) { model in
                            Button { library.open(model.id) } label: {
                                HStack(spacing: 14) {
                                    VStack(alignment: .leading, spacing: 5) {
                                        Text(model.id.split(separator: "/").last.map(String.init) ?? model.id).font(.system(size: 13, weight: .medium)).lineLimit(2)
                                        Text(model.id.split(separator: "/").first.map(String.init) ?? "").font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                                    }
                                    Spacer()
                                    if model.gated { Image(systemName: "lock").font(.caption) }
                                    Label(model.downloads.formatted(), systemImage: "arrow.down").font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                                    Image(systemName: "chevron.right").font(.system(size: 9))
                                }.padding(.vertical, 16).padding(.horizontal, 10).frame(maxWidth: .infinity, alignment: .leading).contentShape(Rectangle())
                            }.buttonStyle(StudioControlStyle())
                            Divider().overlay(StudioTheme.edge)
                        }
                        if library.results.isEmpty && !library.searching {
                            Text(L("Aucun résultat. Essayez un autre nom ou un dépôt exact.", "No results. Try another name or an exact repository."))
                                .font(.callout).foregroundStyle(StudioTheme.quiet).padding(.vertical, 40)
                        }
                    }
                }
            }
            if let error = library.error { Text(error).font(.caption).foregroundStyle(.orange).textSelection(.enabled) }
        }.padding(26).frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
            .overlay { if library.loadingDetail { ProgressView().padding(20).background(StudioTheme.canvas, in: RoundedRectangle(cornerRadius: 12)) } }
            .onChange(of: library.query) { library.search() }
            .onAppear { if library.results.isEmpty { library.search() } }
    }

    private func details(_ detail: HubDetails) -> some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Button(action: library.back) { Label(L("Recherche", "Search"), systemImage: "arrow.left") }.buttonStyle(.plain).font(.system(size: 11)).foregroundStyle(StudioTheme.quiet)
                Spacer()
                Link("Hugging Face ↗", destination: URL(string: "https://huggingface.co/\(detail.id)")!).font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
            }
            Text(detail.id).font(.system(size: 23, weight: .regular, design: .serif)).lineLimit(2).textSelection(.enabled)
            HStack(alignment: .top, spacing: 14) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(L("BRANCHE / QUANTIFICATION", "BRANCH / QUANTIZATION")).font(.system(size: 8, weight: .medium)).tracking(1)
                    Picker(L("Branche ou quantification", "Branch or quantization"), selection: Binding(get: { detail.revision }, set: { library.open(detail.id, revision: $0) })) {
                        ForEach(detail.branches, id: \.self) { Text($0).tag($0) }
                    }.labelsHidden()
                    if detail.variants.count > 1 {
                        Picker(L("Variante", "Variant"), selection: $library.variantID) { ForEach(detail.variants) { Text($0.label).tag($0.id) } }
                    }
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    if let variant = library.selectedVariant {
                        Text(ByteCountFormatter.string(fromByteCount: variant.sizeBytes, countStyle: .file)).font(.system(size: 24, weight: .regular, design: .serif))
                        Text(L("à télécharger", "to download")).font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
                    }
                }
                Button { library.download(studio: studio) } label: {
                    Label(L("Télécharger", "Download"), systemImage: "arrow.down.to.line").font(.system(size: 11, weight: .medium)).padding(.horizontal, 15).padding(.vertical, 12)
                }.buttonStyle(PrimaryGlassButtonStyle()).disabled(library.selectedVariant == nil || library.downloading != nil || library.loadingDetail)
            }.padding(14).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
            if detail.variants.isEmpty {
                Text(L("Pas de checkpoint EXL3 complet identifié ici. Essayez une branche BPW. Les poids ambigus partageant un même descripteur ne sont pas mélangés.", "No complete EXL3 checkpoint identified here. Try a BPW branch. Ambiguous weights sharing one descriptor are not mixed."))
                    .font(.caption).foregroundStyle(.orange)
            }
            if detail.gated {
                Text(L("Dépôt à accès restreint : acceptez sa licence sur Hugging Face et connectez votre compte avec hf auth login.", "Gated repository: accept its license on Hugging Face and sign in with hf auth login."))
                    .font(.caption).foregroundStyle(.orange)
            }
            Text(L("FICHE DU MODÈLE", "MODEL CARD")).font(.system(size: 9, weight: .medium)).tracking(1.4).foregroundStyle(StudioTheme.quiet)
            ScrollView {
                MarkdownResponseView(detail.readme.isEmpty ? L("Aucun README disponible.", "No README available.") : detail.readme).frame(maxWidth: .infinity, alignment: .leading)
            }.environment(\.openURL, OpenURLAction { url in
                guard ["https", "http"].contains(url.scheme?.lowercased() ?? "") else { return .discarded }
                return .systemAction
            })
            Text(L("EXL3 indique le format, pas une garantie de compatibilité de l’architecture.", "EXL3 describes the format, not a guarantee of architecture compatibility."))
                .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
        }
    }
}

private struct LibraryDownloadView: View {
    @ObservedObject var library: ModelLibrary
    var body: some View {
        if library.downloading != nil || library.downloadMessage != nil {
            VStack(alignment: .leading, spacing: 8) {
                Divider().overlay(StudioTheme.edge)
                if let name = library.downloading {
                    HStack {
                        Text(name).lineLimit(1)
                        Spacer()
                        Text("\(ByteCountFormatter.string(fromByteCount: Int64(library.completed), countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: Int64(library.total), countStyle: .file))").monospacedDigit()
                        Button(L("Suspendre", "Pause"), action: library.cancelDownload).buttonStyle(.plain)
                    }
                    ProgressView(value: min(library.completed, max(library.total, 1)), total: max(library.total, 1)).tint(StudioTheme.accent)
                } else if let message = library.downloadMessage { Text(message).textSelection(.enabled).lineLimit(3) }
            }.font(.system(size: 11)).padding(.horizontal, 25).padding(.bottom, 17)
        }
    }
}
