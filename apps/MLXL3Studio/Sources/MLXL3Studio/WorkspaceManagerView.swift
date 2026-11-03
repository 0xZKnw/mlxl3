import AppKit
import PDFKit
import SwiftUI

struct WorkspaceManagerView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.dismiss) private var dismiss
    @State private var projectName = ""
    @State private var profileName = ""
    @State private var query = ""
    @State private var selectedTab = 0
    @State private var source: DocumentExcerpt?

    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text(L("Projets et profils", "Projects and profiles")).font(.title2)
                Spacer()
                Button(L("Fermer", "Close")) { dismiss() }.keyboardShortcut(.cancelAction)
            }
            HStack(spacing: 8) {
                let titles = [L("Projets", "Projects"), L("Documents", "Documents"), L("Profils", "Profiles")]
                let icons = ["folder", "doc.text.magnifyingglass", "slider.horizontal.3"]
                ForEach(0..<3) { index in
                    Button { selectedTab = index } label: {
                        Label(titles[index], systemImage: icons[index]).frame(maxWidth: .infinity).padding(.vertical, 10)
                    }
                    .buttonStyle(StudioControlStyle(emphasized: selectedTab == index))
                    .accessibilityAddTraits(selectedTab == index ? .isSelected : [])
                }
            }
            Group {
                switch selectedTab {
                case 1: library
                case 2: profiles
                default: project
                }
            }.frame(maxWidth: .infinity, maxHeight: .infinity).disabled(studio.isGenerating)
        }
        .padding(24).frame(width: 740, height: 590)
        .background(StudioTheme.canvas).foregroundStyle(StudioTheme.ink)
        .sheet(item: $source) { DocumentSourceView(source: $0) }
    }

    private var project: some View {
        Form {
            Picker(L("Projet actif", "Active project"), selection: Binding(get: { studio.selectedProjectID }, set: { studio.chooseProject($0) })) {
                Text(L("Sans projet", "No project")).tag(UUID?.none)
                ForEach(studio.projects) { Text($0.name).tag(Optional($0.id)) }
            }
            HStack {
                TextField(L("Nom du nouveau projet", "New project name"), text: $projectName)
                Button(L("Créer", "Create")) { studio.addProject(projectName); projectName = "" }.disabled(projectName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            if let project = studio.currentProject {
                TextField(L("Nom", "Name"), text: projectBinding(\.name))
                Text(L("Instructions propres à ce projet", "Instructions for this project"))
                TextEditor(text: projectBinding(\.instructions)).frame(minHeight: 110)
                Picker(L("Profil à l’ouverture", "Profile on opening"), selection: projectBinding(\.profileID)) {
                    Text(L("Réglages actuels", "Current settings")).tag(UUID?.none)
                    ForEach(studio.profiles) { Text($0.name).tag(Optional($0.id)) }
                }
                Text(L("Retirer le projet conserve ses conversations dans « Sans projet ». Les extraits déjà cités restent dans l’historique.", "Removing a project keeps its conversations under No project. Previously cited excerpts remain in history.")).font(.caption).foregroundStyle(.secondary)
                Button(L("Retirer ce projet", "Remove this project"), role: .destructive) { studio.deleteProject(project.id) }
            }
        }.formStyle(.grouped)
    }

    private func projectBinding<Value>(_ key: WritableKeyPath<WorkspaceProject, Value>) -> Binding<Value> {
        let fallback = studio.currentProject![keyPath: key]
        return Binding(get: { studio.currentProject?[keyPath: key] ?? fallback }, set: { value in
            guard var project = studio.currentProject else { return }
            project[keyPath: key] = value; studio.updateProject(project)
        })
    }

    private var profiles: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(L("Enregistrez le modèle, le contexte, l’instruction, l’échantillonnage et le schéma JSON actuels.", "Save the current model, context, instruction, sampling settings and JSON schema.")).font(.callout)
            HStack {
                TextField(L("Nom du profil", "Profile name"), text: $profileName)
                Button(L("Enregistrer", "Save")) { studio.saveProfile(profileName); profileName = "" }.disabled(profileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            List(studio.profiles) { profile in
                HStack {
                    VStack(alignment: .leading) {
                        Text(profile.name).font(.headline)
                        Text(profile.model ?? L("Modèle actuel", "Current model")).font(.caption).foregroundStyle(.secondary)
                    }
                    Spacer()
                    Button(L("Appliquer", "Apply")) { studio.applyProfile(profile) }
                    Button(role: .destructive) { studio.deleteProfile(profile.id) } label: { Image(systemName: "trash") }.help(L("Supprimer le profil", "Delete profile"))
                }
            }
        }.padding(16)
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
                Text(L("Recherche locale par mots. Les passages pertinents sont joints à chaque question ; vérifiez les citations dans la réponse.", "Local keyword search. Relevant passages accompany each question; verify the citations in the answer.")).font(.caption).foregroundStyle(.secondary)
                if query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    List(studio.documents.filter { $0.projectID == projectID }) { document in
                        HStack {
                            Label(document.fileName, systemImage: "doc.text")
                            Spacer()
                            Button(role: .destructive) { studio.removeLibraryDocument(document.id) } label: { Image(systemName: "trash") }.help(L("Retirer de la bibliothèque", "Remove from library"))
                        }
                    }
                } else {
                    let results = DocumentSearch.search(query, documents: studio.documents.filter { $0.projectID == projectID }, limit: 8)
                    List(results) { result in
                        Button { source = result } label: {
                            VStack(alignment: .leading, spacing: 5) { Text(result.label).bold(); Text(result.text).lineLimit(4) }.frame(maxWidth: .infinity, alignment: .leading)
                        }.buttonStyle(.plain)
                    }
                    if results.isEmpty { Text(L("Aucun passage correspondant.", "No matching passage.")) }
                }
            } else {
                ContentUnavailableView(L("Choisissez un projet", "Choose a project"), systemImage: "folder", description: Text(L("Les documents restent associés à leur projet.", "Documents remain associated with their project.")))
            }
        }.padding(16)
    }
}

struct DocumentSourceView: View {
    let source: DocumentExcerpt
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack { Text(source.label).font(.headline); Spacer(); Button(L("Fermer", "Close")) { dismiss() }.keyboardShortcut(.cancelAction) }
            if source.originalPath.lowercased().hasSuffix(".pdf"), FileManager.default.fileExists(atPath: source.originalPath) {
                SourcePDFView(path: source.originalPath, page: source.page).frame(minHeight: 250)
            }
            ScrollView { Text(source.text).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }
            Text(L("Extrait conservé au moment de la réponse. Le fichier original peut avoir changé.", "Excerpt saved with the answer. The original file may have changed.")).font(.caption).foregroundStyle(.secondary)
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
    var body: some View {
        let tokens = min(UInt64(context), preview?.maximumContext ?? UInt64(context))
        let cache = preview?.kvBytesPerToken.map { Double($0) * Double(tokens) + Double(preview?.recurrentBytes ?? 0) }
        let estimate = Double(max(0, sizeBytes)) + (cache ?? 0)
        let advice = ModelAdvice(weights: sizeBytes > 0 ? Int64(min(estimate, Double(Int64.max / 2))) : 0, memory: ProcessInfo.processInfo.physicalMemory)
        VStack(alignment: .leading, spacing: 7) {
            if let preview {
                Label((preview.modelType ?? L("Architecture inconnue", "Unknown architecture")) + " · " + (preview.architectureSupported ? L("famille prise en charge", "supported family") : L("famille non prise en charge", "unsupported family")), systemImage: preview.architectureSupported ? "checkmark.circle" : "exclamationmark.triangle")
            }
            Stepper(L("Contexte prévu : \(tokens) tokens", "Planned context: \(tokens) tokens"), value: $context, in: 512...131072, step: 512)
            if let cache {
                if sizeBytes > 0 {
                    Text(L("Poids + cache théorique : ", "Weights + theoretical cache: ") + ByteCountFormatter.string(fromByteCount: Int64(min(Double(Int64.max / 2), estimate)), countStyle: .file))
                }
                Text(L("Cache KV FP16 et état récurrent : ", "FP16 KV cache and recurrent state: ") + ByteCountFormatter.string(fromByteCount: Int64(min(Double(Int64.max / 2), cache)), countStyle: .file))
            } else { Text(L("Cache pour cette architecture : non estimé avant chargement.", "Cache for this architecture: not estimated before loading.")) }
            Label(advice.message, systemImage: advice.fit == .comfortable ? "checkmark.circle" : "info.circle")
            Text(L("Estimation, pas une garantie : buffers, cache de préfixe, MTP, arrondis d’allocation et macOS s’ajoutent. La disposition exacte des poids sera vérifiée au chargement. Comparez la taille et le BPW des variantes ; leur qualité n’est pas mesurée ici.", "Estimate, not a guarantee: buffers, prefix cache, MTP, allocation rounding and macOS add memory. Exact weight layout is checked on load. Compare variant sizes and BPW; quality is not measured here."))
                .foregroundStyle(.secondary)
        }.font(.caption).padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
    }
}
