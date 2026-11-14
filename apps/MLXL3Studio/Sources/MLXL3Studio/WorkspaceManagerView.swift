import AppKit
import PDFKit
import SwiftUI

struct WorkspaceManagerView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var projectName = ""
    @State private var profileName = ""
    @State private var query = ""
    @State private var selectedTab = 0
    @State private var source: DocumentExcerpt?

    var body: some View {
        let titles = [L("Projets", "Projects"), L("Documents", "Documents"), L("Profils", "Profiles")]
        let icons = ["folder", "doc.text.magnifyingglass", "slider.horizontal.3"]
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                StudioSidebarHeader(title: L("Projets", "Projects"))
                VStack(spacing: 3) {
                    ForEach(0..<3) { index in
                        Button { selectedTab = index } label: {
                            HStack(spacing: 11) {
                                Image(systemName: icons[index]).frame(width: 18)
                                Text(titles[index])
                                Spacer(minLength: 0)
                            }
                            .font(.system(size: 13, weight: selectedTab == index ? .semibold : .regular))
                            .padding(.horizontal, 12).frame(height: 36)
                        }
                        .buttonStyle(StudioControlStyle(emphasized: selectedTab == index))
                        .accessibilityAddTraits(selectedTab == index ? .isSelected : [])
                    }
                }.padding(.horizontal, 10)
                Spacer()
                StudioBackButton(action: { studio.showWorkspace = false })
                    .padding(.horizontal, 18).padding(.bottom, 18)
            }
            .frame(width: studio.sidebarWidth.points, alignment: .leading)
            .modifier(StudioSidebarSurface())
            VStack(alignment: .leading, spacing: 25) {
                StudioPanelHeader(title: titles[selectedTab], detail: selectedTab == 1
                    ? L("Documents du projet.", "Project documents.")
                    : selectedTab == 2 ? L("Réglages de génération enregistrés.", "Saved generation settings.")
                    : L("Instructions et documents par projet.", "Instructions and documents by project."))
                Group {
                    switch selectedTab {
                    case 1: library
                    case 2: profiles
                    default: project
                    }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .id(selectedTab)
                .transition(.opacity)
                .disabled(studio.isGenerating)
                if studio.isGenerating {
                    Text(
                        L(
                            "Les modifications seront disponibles à la fin de la génération.",
                            "Editing becomes available when generation finishes.")
                    )
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary).padding(16)
                }
            }
            .padding(.vertical, 36).frame(maxWidth: 760, alignment: .leading)
            .padding(.horizontal, 28).frame(maxWidth: .infinity)
        }
        .font(.system(size: 13)).controlSize(.large)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background { StudioAurora(animated: true).background(StudioTheme.canvas) }.foregroundStyle(StudioTheme.ink)
        .onExitCommand { studio.showWorkspace = false }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: selectedTab)
        .sheet(item: $source) { DocumentSourceView(source: $0) }
    }

    private var project: some View {
        ScrollView {
          VStack(alignment: .leading, spacing: 20) {
            SettingCard(title: L("Projet actif", "Active project"), icon: "folder") {
            Picker(
                L("Projet actif", "Active project"),
                selection: Binding(get: { studio.selectedProjectID }, set: { studio.chooseProject($0) })
            ) {
                Text(L("Sans projet", "No project")).tag(UUID?.none)
                ForEach(studio.projects) { Text($0.name).tag(Optional($0.id)) }
            }
            HStack {
                TextField(L("Nom du nouveau projet", "New project name"), text: $projectName)
                Button(L("Créer", "Create")) {
                    studio.addProject(projectName)
                    projectName = ""
                }.disabled(projectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            }
            if let project = studio.currentProject {
              SettingCard(title: L("Instructions et profil", "Instructions and profile"), icon: "text.alignleft") {
                TextField(L("Nom", "Name"), text: projectBinding(\.name))
                Text(L("Instructions propres à ce projet", "Instructions for this project"))
                TextEditor(text: projectBinding(\.instructions)).scrollContentBackground(.hidden)
                    .padding(8).frame(minHeight: 120)
                    .background(StudioTheme.composer, in: RoundedRectangle(cornerRadius: 8))
                Picker(L("Profil à l’ouverture", "Profile on opening"), selection: projectBinding(\.profileID)) {
                    Text(L("Réglages actuels", "Current settings")).tag(UUID?.none)
                    ForEach(studio.profiles) { Text($0.name).tag(Optional($0.id)) }
                }
                Text(
                    L(
                        "Retirer le projet conserve ses conversations dans « Sans projet ». Les extraits déjà cités restent dans l’historique.",
                        "Removing a project keeps its conversations under No project. Previously cited excerpts remain in history."
                    )
                ).font(.caption).foregroundStyle(.secondary)
                Button(L("Retirer ce projet", "Remove this project"), role: .destructive) {
                    studio.deleteProject(project.id)
                }
              }
            }
          }.textFieldStyle(.roundedBorder)
        }
    }

    private func projectBinding<Value>(_ key: WritableKeyPath<WorkspaceProject, Value>) -> Binding<Value> {
        let fallback = studio.currentProject![keyPath: key]
        return Binding(
            get: { studio.currentProject?[keyPath: key] ?? fallback },
            set: { value in
                guard var project = studio.currentProject else { return }
                project[keyPath: key] = value
                studio.updateProject(project)
            })
    }

    private var profiles: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(
                L(
                    "Enregistrez le modèle, le contexte, l’instruction, l’échantillonnage et le schéma JSON actuels.",
                    "Save the current model, context, instruction, sampling settings and JSON schema.")
            ).font(.callout)
            HStack {
                TextField(L("Nom du profil", "Profile name"), text: $profileName)
                Button(L("Enregistrer", "Save")) {
                    studio.saveProfile(profileName)
                    profileName = ""
                }.disabled(profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            if studio.profiles.isEmpty {
                ContentUnavailableView(
                    L("Aucun profil enregistré", "No saved profiles"), systemImage: "slider.horizontal.3",
                    description: Text(
                        L(
                            "Nommez vos réglages actuels ci-dessus pour les réutiliser dans vos conversations.",
                            "Name your current settings above to reuse them in conversations.")))
            }
            List(studio.profiles) { profile in
                HStack {
                    VStack(alignment: .leading) {
                        Text(profile.name).font(.headline)
                        Text(profile.model ?? L("Modèle actuel", "Current model")).font(.caption).foregroundStyle(
                            .secondary)
                    }
                    Spacer()
                    Button(L("Appliquer", "Apply")) { studio.applyProfile(profile) }
                    Button(role: .destructive) {
                        studio.deleteProfile(profile.id)
                    } label: {
                        Image(systemName: "trash")
                    }.help(L("Supprimer le profil", "Delete profile"))
                }
            }.scrollContentBackground(.hidden).listStyle(.plain)
        }.textFieldStyle(.roundedBorder)
    }

    private var library: some View {
        VStack(alignment: .leading, spacing: 12) {
            if let projectID = studio.selectedProjectID {
                HStack {
                    Button(L("Ajouter des fichiers", "Add files")) { studio.chooseLibraryFiles(folder: false) }
                    Button(L("Ajouter un dossier", "Add folder")) { studio.chooseLibraryFiles(folder: true) }
                    if studio.libraryImporting { ProgressView().controlSize(.small) }
                    Spacer()
                    Text("\(studio.documents.filter { $0.projectID == projectID }.count) / 128").font(.caption)
                }.disabled(studio.libraryImporting)
                TextField(L("Rechercher un passage…", "Find a passage…"), text: $query)
                Text(
                    L(
                        "Recherche locale par mots. Les passages pertinents sont joints à chaque question ; vérifiez les citations dans la réponse.",
                        "Local keyword search. Relevant passages accompany each question; verify the citations in the answer."
                    )
                ).font(.caption).foregroundStyle(.secondary)
                if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    List(studio.documents.filter { $0.projectID == projectID }) { document in
                        HStack {
                            Label(document.fileName, systemImage: "doc.text")
                            Spacer()
                            Button(role: .destructive) {
                                studio.removeLibraryDocument(document.id)
                            } label: {
                                Image(systemName: "trash")
                            }.help(L("Retirer de la bibliothèque", "Remove from library"))
                        }
                    }.scrollContentBackground(.hidden).listStyle(.plain)
                } else {
                    let results = DocumentSearch.search(
                        query, documents: studio.documents.filter { $0.projectID == projectID }, limit: 8)
                    List(results) { result in
                        Button {
                            source = result
                        } label: {
                            VStack(alignment: .leading, spacing: 5) {
                                Text(result.label).bold()
                                Text(result.text).lineLimit(4)
                            }.frame(maxWidth: .infinity, alignment: .leading)
                        }.buttonStyle(.plain)
                    }.scrollContentBackground(.hidden).listStyle(.plain)
                    if results.isEmpty { Text(L("Aucun passage correspondant.", "No matching passage.")) }
                }
            } else {
                ContentUnavailableView {
                    Label(L("Choisissez un projet", "Choose a project"), systemImage: "folder")
                } description: {
                    Text(
                        L(
                            "Les documents restent associés à leur projet.",
                            "Documents remain associated with their project."))
                } actions: {
                    Button(L("Voir les projets", "View projects")) { selectedTab = 0 }
                }
            }
        }.textFieldStyle(.roundedBorder)
    }
}

struct DocumentSourceView: View {
    let source: DocumentExcerpt
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(source.label).font(.headline)
                Spacer()
                Button(L("Fermer", "Close")) { dismiss() }.keyboardShortcut(.cancelAction)
            }
            if source.originalPath.lowercased().hasSuffix(".pdf"),
                FileManager.default.fileExists(atPath: source.originalPath)
            {
                SourcePDFView(path: source.originalPath, page: source.page).frame(minHeight: 250)
            }
            ScrollView { Text(source.text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            Text(
                L(
                    "Extrait conservé au moment de la réponse. Le fichier original peut avoir changé.",
                    "Excerpt saved with the answer. The original file may have changed.")
            ).font(.caption).foregroundStyle(.secondary)
        }.padding(24).frame(width: 720, height: 560)
            .background(StudioTheme.canvas).foregroundStyle(StudioTheme.ink)
    }
}

private struct SourcePDFView: NSViewRepresentable {
    let path: String
    let page: Int?
    func makeNSView(context: Context) -> PDFView { PDFView() }
    func updateNSView(_ view: PDFView, context: Context) {
        view.autoScales = true
        view.document = PDFDocument(url: URL(fileURLWithPath: path))
        if let page, page > 0, let target = view.document?.page(at: page - 1) { view.go(to: target) }
    }
}

struct ModelAdviceView: View {
    let sizeBytes: Int64
    var preview: HubModelPreview? = nil
    @State private var context = 4096
    @State private var expanded = false
    var body: some View {
        let tokens = min(UInt64(context), preview?.maximumContext ?? UInt64(context))
        let cache = preview?.kvBytesPerToken.map { Double($0) * Double(tokens) + Double(preview?.recurrentBytes ?? 0) }
        let estimate = Double(max(0, sizeBytes)) + (cache ?? 0)
        let advice = ModelAdvice(
            weights: sizeBytes > 0 ? Int64(min(estimate, Double(Int64.max / 2))) : 0,
            memory: ProcessInfo.processInfo.physicalMemory)
        DisclosureGroup(isExpanded: $expanded) {
          VStack(alignment: .leading, spacing: 7) {
            Stepper(
                L("Contexte prévu : \(tokens) tokens", "Planned context: \(tokens) tokens"), value: $context,
                in: 512...131072, step: 512)
            if let cache {
                if sizeBytes > 0 {
                    Text(
                        L("Poids + cache théorique : ", "Weights + theoretical cache: ")
                            + ByteCountFormatter.string(
                                fromByteCount: Int64(min(Double(Int64.max / 2), estimate)), countStyle: .file))
                }
                Text(
                    L("Cache KV FP16 et état récurrent : ", "FP16 KV cache and recurrent state: ")
                        + ByteCountFormatter.string(
                            fromByteCount: Int64(min(Double(Int64.max / 2), cache)), countStyle: .file))
            } else {
                Text(
                    L(
                        "Cache pour cette architecture : non estimé avant chargement.",
                        "Cache for this architecture: not estimated before loading."))
            }
            Label(advice.message, systemImage: advice.fit == .comfortable ? "checkmark.circle" : "info.circle")
            Text(
                L(
                    "Estimation, pas une garantie : buffers, cache de préfixe, MTP, arrondis d’allocation et macOS s’ajoutent. La disposition exacte des poids sera vérifiée au chargement. Comparez la taille et le BPW des variantes ; leur qualité n’est pas mesurée ici.",
                    "Estimate, not a guarantee: buffers, prefix cache, MTP, allocation rounding and macOS add memory. Exact weight layout is checked on load. Compare variant sizes and BPW; quality is not measured here."
                )
            )
            .foregroundStyle(.secondary)
          }.padding(.top, 10)
        } label: {
            HStack(spacing: 10) {
                Label(preview?.modelType ?? L("Architecture inconnue", "Unknown architecture"),
                      systemImage: preview.map { $0.architectureSupported ? "checkmark.circle" : "exclamationmark.triangle" } ?? "info.circle")
                Text(preview.map { $0.architectureSupported ? L("prise en charge", "supported") : L("non prise en charge", "unsupported") } ?? L("à vérifier", "check support"))
                    .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Text(L("Estimation mémoire", "Memory estimate")).foregroundStyle(.secondary)
            }
        }.font(.caption).padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }
}
