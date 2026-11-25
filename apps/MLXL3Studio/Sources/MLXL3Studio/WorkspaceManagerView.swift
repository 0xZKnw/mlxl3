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
            StudioPageSidebar(title: L("Projets", "Projects"), onBack: { studio.showWorkspace = false }) {
                StudioNavList {
                    ForEach(0..<3) { index in
                        StudioNavRow(icon: icons[index], title: titles[index], selected: selectedTab == index,
                                     badge: index == 0 && !studio.projects.isEmpty ? "\(studio.projects.count)"
                                         : index == 2 && !studio.profiles.isEmpty ? "\(studio.profiles.count)" : nil) {
                            selectedTab = index
                        }
                    }
                }
            }
            StudioPageColumn(spacing: 22) {
                StudioPageHeader(title: titles[selectedTab], detail: selectedTab == 1
                    ? L("Documents du projet.", "Project documents.")
                    : selectedTab == 2 ? L("Réglages de génération enregistrés.", "Saved generation settings.")
                    : L("Instructions et documents par projet.", "Instructions and documents by project."))
                if studio.isGenerating {
                    StudioNotice(text: L(
                        "Les modifications seront disponibles à la fin de la génération.",
                        "Editing becomes available when generation finishes."))
                }
                Group {
                    switch selectedTab {
                    case 1: library
                    case 2: profiles
                    default: project
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .transition(.opacity)
                .disabled(studio.isGenerating)
            }
            .id(selectedTab)
        }
        .font(.system(size: 13))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background { StudioAurora(animated: true).background(StudioTheme.canvas) }.foregroundStyle(StudioTheme.ink)
        .onExitCommand { studio.showWorkspace = false }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: selectedTab)
        .sheet(item: $source) { DocumentSourceView(source: $0) }
    }

    private var project: some View {
        VStack(alignment: .leading, spacing: 22) {
            SettingSection(title: L("Projet actif", "Active project"), icon: "folder") {
                SettingRow(title: L("Projet actif", "Active project"),
                           detail: L("Les nouvelles conversations y sont rangées.", "New conversations are filed under it.")) {
                    Picker(
                        L("Projet actif", "Active project"),
                        selection: Binding(get: { studio.selectedProjectID }, set: { studio.chooseProject($0) })
                    ) {
                        Text(L("Sans projet", "No project")).tag(UUID?.none)
                        ForEach(studio.projects) { Text($0.name).tag(Optional($0.id)) }
                    }
                    .labelsHidden().frame(maxWidth: 240, alignment: .trailing)
                }
                SettingRow(title: L("Nouveau projet", "New project")) {
                    TextField(L("Nom du nouveau projet", "New project name"), text: $projectName)
                        .studioField().frame(width: 220)
                        .onSubmit(createProject)
                    Button(L("Créer", "Create"), action: createProject)
                        .buttonStyle(.studio(.primary))
                        .disabled(projectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            if let project = studio.currentProject {
                SettingSection(
                    title: L("Instructions et profil", "Instructions and profile"), icon: "text.alignleft",
                    footer: L(
                        "Retirer le projet conserve ses conversations dans « Sans projet ». Les extraits déjà cités restent dans l’historique.",
                        "Removing a project keeps its conversations under No project. Previously cited excerpts remain in history.")
                ) {
                    SettingRow(title: L("Nom", "Name")) {
                        TextField(L("Nom", "Name"), text: projectBinding(\.name))
                            .studioField().frame(width: 280)
                    }
                    VStack(alignment: .leading, spacing: 8) {
                        Text(L("Instructions propres à ce projet", "Instructions for this project"))
                            .font(.system(size: 13, weight: .medium))
                        TextEditor(text: projectBinding(\.instructions)).scrollContentBackground(.hidden)
                            .font(.system(size: 13))
                            .padding(8).frame(minHeight: 120)
                            .background(StudioTheme.contrast.opacity(0.045), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                            .overlay { RoundedRectangle(cornerRadius: 8, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5).allowsHitTesting(false) }
                            .accessibilityLabel(L("Instructions propres à ce projet", "Instructions for this project"))
                    }
                    SettingRow(title: L("Profil à l’ouverture", "Profile on opening")) {
                        Picker(L("Profil à l’ouverture", "Profile on opening"), selection: projectBinding(\.profileID)) {
                            Text(L("Réglages actuels", "Current settings")).tag(UUID?.none)
                            ForEach(studio.profiles) { Text($0.name).tag(Optional($0.id)) }
                        }
                        .labelsHidden().frame(maxWidth: 240, alignment: .trailing)
                    }
                    SettingRow(title: L("Retirer ce projet", "Remove this project")) {
                        Button(L("Retirer", "Remove"), role: .destructive) {
                            studio.deleteProject(project.id)
                        }
                        .buttonStyle(.studio(.destructive))
                        .accessibilityLabel(L("Retirer ce projet", "Remove this project"))
                    }
                }
            } else if studio.projects.isEmpty {
                StudioEmptyState(
                    icon: "folder",
                    title: L("Aucun projet", "No projects yet"),
                    detail: L("Un projet regroupe des conversations, des instructions et des documents.",
                              "A project groups conversations, instructions and documents."))
                    .studioCard(padding: 0)
            }
        }
    }

    private func createProject() {
        guard !projectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        studio.addProject(projectName)
        projectName = ""
    }

    private func saveProfile() {
        guard !profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        studio.saveProfile(profileName)
        profileName = ""
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
        VStack(alignment: .leading, spacing: 22) {
            SettingSection(
                title: L("Nouveau profil", "New profile"), icon: "plus",
                footer: L(
                    "Enregistrez le modèle, le contexte, l’instruction, l’échantillonnage et le schéma JSON actuels.",
                    "Save the current model, context, instruction, sampling settings and JSON schema.")
            ) {
                SettingRow(title: L("Réglages actuels", "Current settings")) {
                    TextField(L("Nom du profil", "Profile name"), text: $profileName)
                        .studioField().frame(width: 220)
                        .onSubmit(saveProfile)
                    Button(L("Enregistrer", "Save"), action: saveProfile)
                        .buttonStyle(.studio(.primary))
                        .disabled(profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }
            if studio.profiles.isEmpty {
                StudioEmptyState(
                    icon: "slider.horizontal.3",
                    title: L("Aucun profil enregistré", "No saved profiles"),
                    detail: L(
                        "Nommez vos réglages actuels ci-dessus pour les réutiliser dans vos conversations.",
                        "Name your current settings above to reuse them in conversations."))
                    .studioCard(padding: 0)
            } else {
                SettingSection(title: L("Profils enregistrés", "Saved profiles"), icon: "slider.horizontal.3") {
                    ForEach(studio.profiles) { profile in
                        SettingRow(title: profile.name, detail: profile.model ?? L("Modèle actuel", "Current model")) {
                            Button(L("Appliquer", "Apply")) { studio.applyProfile(profile) }
                                .buttonStyle(.studio)
                            StudioIconButton(icon: "trash", label: L("Supprimer le profil", "Delete profile"), destructive: true) {
                                studio.deleteProfile(profile.id)
                            }
                        }
                    }
                }
            }
        }
    }

    private var library: some View {
        VStack(alignment: .leading, spacing: 16) {
            if let projectID = studio.selectedProjectID {
                let documents = studio.documents.filter { $0.projectID == projectID }
                HStack(spacing: 8) {
                    Button { studio.chooseLibraryFiles(folder: false) } label: {
                        Label(L("Ajouter des fichiers", "Add files"), systemImage: "doc.badge.plus")
                    }
                    .buttonStyle(.studio(.primary))
                    Button { studio.chooseLibraryFiles(folder: true) } label: {
                        Label(L("Ajouter un dossier", "Add folder"), systemImage: "folder.badge.plus")
                    }
                    .buttonStyle(.studio)
                    if studio.libraryImporting { ProgressView().controlSize(.small) }
                    Spacer()
                    Text("\(documents.count) / 128").font(.system(size: 12)).monospacedDigit()
                        .foregroundStyle(StudioTheme.quiet)
                        .help(L("Documents du projet, sur 128 au maximum", "Project documents, out of 128 at most"))
                }.disabled(studio.libraryImporting)
                HStack(spacing: 7) {
                    Image(systemName: "magnifyingglass").font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                    TextField(L("Rechercher un passage…", "Find a passage…"), text: $query).textFieldStyle(.plain)
                }
                .studioField()
                Text(
                    L(
                        "Recherche locale par mots. Les passages pertinents sont joints à chaque question ; vérifiez les citations dans la réponse.",
                        "Local keyword search. Relevant passages accompany each question; verify the citations in the answer."
                    )
                ).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet).fixedSize(horizontal: false, vertical: true)
                if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    if documents.isEmpty {
                        StudioEmptyState(
                            icon: "doc.text",
                            title: L("Aucun document", "No documents yet"),
                            detail: L("Ajoutez des fichiers pour que leurs passages accompagnent vos questions.",
                                      "Add files so their passages accompany your questions."))
                            .studioCard(padding: 0)
                    } else {
                        StudioRowList(rowPadding: 6) {
                            ForEach(documents) { document in
                                HStack(spacing: 9) {
                                    Image(systemName: "doc.text").font(.system(size: 13)).foregroundStyle(StudioTheme.quiet)
                                        .frame(width: 18)
                                    Text(document.fileName).lineLimit(1).truncationMode(.middle)
                                    Spacer()
                                    StudioIconButton(icon: "trash", label: L("Retirer de la bibliothèque", "Remove from library"),
                                                     destructive: true) {
                                        studio.removeLibraryDocument(document.id)
                                    }
                                }
                            }
                        }
                        .padding(.horizontal, 14)
                        .studioCard(padding: 0)
                    }
                } else {
                    let results = DocumentSearch.search(query, documents: documents, limit: 8)
                    if results.isEmpty {
                        StudioEmptyState(icon: "magnifyingglass", title: L("Aucun passage correspondant.", "No matching passage."))
                            .studioCard(padding: 0)
                    } else {
                        StudioRowList(rowPadding: 0) {
                            ForEach(results) { result in
                                Button {
                                    source = result
                                } label: {
                                    VStack(alignment: .leading, spacing: 5) {
                                        Text(result.label).font(.system(size: 13, weight: .semibold))
                                        Text(result.text).font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
                                            .lineLimit(4).multilineTextAlignment(.leading)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .padding(.horizontal, 14).padding(.vertical, 12)
                                    .contentShape(Rectangle())
                                }
                                .buttonStyle(StudioRowStyle(radius: 0))
                            }
                        }
                        .clipShape(RoundedRectangle(cornerRadius: StudioPageMetrics.card, style: .continuous))
                        .studioCard(padding: 0)
                    }
                }
            } else {
                StudioEmptyState(
                    icon: "folder",
                    title: L("Choisissez un projet", "Choose a project"),
                    detail: L(
                        "Les documents restent associés à leur projet.",
                        "Documents remain associated with their project.")
                ) {
                    Button(L("Voir les projets", "View projects")) { selectedTab = 0 }
                        .buttonStyle(.studio(.primary))
                }
                .studioCard(padding: 0)
            }
        }
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
                Button(L("Fermer", "Close")) { dismiss() }.buttonStyle(.studio).keyboardShortcut(.cancelAction)
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
            ).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
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
            .foregroundStyle(StudioTheme.secondary)
          }.padding(.top, 10)
        } label: {
            HStack(spacing: 10) {
                Label(preview?.modelType ?? L("Architecture inconnue", "Unknown architecture"),
                      systemImage: preview.map { $0.architectureSupported ? "checkmark.circle" : "exclamationmark.triangle" } ?? "info.circle")
                Text(preview.map { $0.architectureSupported ? L("prise en charge", "supported") : L("non prise en charge", "unsupported") } ?? L("à vérifier", "check support"))
                    .foregroundStyle(StudioTheme.secondary)
                Spacer(minLength: 8)
                Text(L("Estimation mémoire", "Memory estimate")).foregroundStyle(StudioTheme.secondary)
            }
        }.font(.caption).padding(10)
            .background(StudioTheme.contrast.opacity(0.045), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}
