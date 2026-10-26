import SwiftUI

/// Window composition. The transcript owns its scroll state; chrome never
/// observes token-by-token text updates.
struct StudioView: View {
    @EnvironmentObject private var studio: StudioModel
    @AppStorage("studio.sidebarVisible") private var sidebarVisible = true

    var body: some View {
        GeometryReader { geometry in
            HStack(spacing: 0) {
                if sidebarVisible {
                    SidebarView().frame(width: 232)
                }
                ChatWorkspaceView(sidebarVisible: $sidebarVisible)
                    .background(StudioTheme.canvas)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay {
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(StudioTheme.edge, lineWidth: 0.5)
                            .allowsHitTesting(false)
                    }
                    .padding(.top, 8)
                    .padding(.bottom, 8)
                    .padding(.leading, sidebarVisible ? 0 : 8)
                    .padding(.trailing, 8)

                if studio.showInspector && (studio.selectedMode == .chat || studio.selectedMode == .agent) && geometry.size.width >= 1220 {
                    GenerationInspector().frame(width: 286)
                }
            }
            .overlay(alignment: .trailing) {
                if studio.showInspector && (studio.selectedMode == .chat || studio.selectedMode == .agent) && geometry.size.width < 1220 {
                  ZStack(alignment: .trailing) {
                    Color.black.opacity(0.28)
                        .contentShape(Rectangle())
                        .onTapGesture { studio.showInspector = false }
                    GenerationInspector()
                        .frame(width: 286)
                        .overlay(alignment: .leading) {
                            Rectangle().fill(StudioTheme.edge).frame(width: 0.5)
                        }
                        .shadow(color: .black.opacity(0.3), radius: 20, x: -8)
                  }
                  .onExitCommand { studio.showInspector = false }
                }
            }
        }
        .id(studio.language)
        .environment(\.locale, Locale(identifier: studio.language.rawValue))
        .background(StudioTheme.sidebar)
        .ignoresSafeArea()
        .frame(minWidth: 820, minHeight: 600)
        .sheet(isPresented: $studio.showModelManager) {
            ModelManagerView().environmentObject(studio)
        }
        .sheet(isPresented: $studio.showWorkspace) { WorkspaceManagerView().environmentObject(studio) }
        .sheet(isPresented: $studio.showMCPManager) { MCPManagerView().environmentObject(studio) }
        .alert(L("MLXL3", "MLXL3"), isPresented: Binding(get: { studio.featureError != nil }, set: { if !$0 { studio.featureError = nil } })) {
            Button("OK") { studio.featureError = nil }
        } message: { Text(studio.featureError ?? "") }
        .sheet(isPresented: $studio.showAppSettings) {
            AppSettingsView().environmentObject(studio)
        }
        .alert(L("Sauvegarde des conversations", "Conversation storage"), isPresented: Binding(
            get: { studio.storageError != nil }, set: { if !$0 { studio.storageError = nil } }
        )) {
            Button(L("Afficher les fichiers", "Show files")) { studio.revealConversations() }
            Button(L("Récupérer une sauvegarde", "Recover backup")) { studio.recoverConversations() }
            Button(L("Fermer", "Close"), role: .cancel) { }
        } message: { Text(studio.storageError ?? "") }
    }
}

private struct SidebarView: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var search = ""

    private var filteredConversations: [Conversation] {
        let query = search.trimmingCharacters(in: .whitespacesAndNewlines)
        return studio.conversations.filter {
            $0.agent == (studio.selectedMode == .agent) && $0.projectID == studio.selectedProjectID && (query.isEmpty || $0.title.localizedCaseInsensitiveContains(query)
                || $0.messages.contains { $0.content.localizedCaseInsensitiveContains(query) })
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 9) {
                MonogramMark(size: 21)
                Text("MLXL3")
                    .font(.system(size: 14, weight: .semibold))
                    .tracking(1)
                Spacer()
                Text("Desktop")
                    .font(.system(size: 10))
                    .foregroundStyle(StudioTheme.quiet)
            }
            .padding(.horizontal, 22)
            .padding(.top, 51)
            .padding(.bottom, 22)

            HStack(spacing: 2) {
                ForEach(StudioMode.allCases) { mode in
                    Button { studio.selectMode(mode) } label: {
                        Text(mode.title).font(.system(size: 11, weight: .medium))
                            .frame(maxWidth: .infinity).frame(height: 30)
                    }
                    .buttonStyle(StudioControlStyle(emphasized: studio.selectedMode == mode))
                    .accessibilityLabel(mode.title)
                    .accessibilityAddTraits(studio.selectedMode == mode ? .isSelected : [])
                }
            }
            .padding(3).background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal, 12).padding(.bottom, 16).disabled(studio.isGenerating)

            Button(action: studio.newConversation) {
                HStack(spacing: 10) {
                    Image(systemName: "square.and.pencil").font(.system(size: 14))
                    Text(studio.selectedMode == .agent ? L("Nouvelle tâche", "New task") : L("Nouvelle conversation", "New conversation")).font(.system(size: 12, weight: .medium))
                    Spacer(minLength: 0)
                }
                .padding(.horizontal, 11)
                .frame(height: 38)
            }
            .buttonStyle(StudioControlStyle(emphasized: true))
            .padding(.horizontal, 12)
            .help(L("Nouvelle conversation · ⌘N", "New conversation · ⌘N"))

            Picker(L("Projet", "Project"), selection: Binding(get: { studio.selectedProjectID }, set: { studio.chooseProject($0) })) {
                Text(L("Sans projet", "No project")).tag(UUID?.none)
                ForEach(studio.projects) { Text($0.name).tag(Optional($0.id)) }
            }.labelsHidden().padding(.horizontal, 16).padding(.top, 10).disabled(studio.isGenerating)

            HStack(spacing: 9) {
                Image(systemName: "magnifyingglass").font(.system(size: 12))
                TextField(L("Rechercher", "Search"), text: $search)
                    .textFieldStyle(.plain)
                    .accessibilityLabel(L("Rechercher dans les titres des conversations", "Search conversation titles"))
                if !search.isEmpty {
                    Button { search = "" } label: { Image(systemName: "xmark.circle.fill") }
                        .buttonStyle(.plain)
                        .help(L("Effacer la recherche", "Clear search"))
                }
            }
            .font(.system(size: 12))
            .foregroundStyle(StudioTheme.secondary)
            .padding(.horizontal, 12)
            .frame(height: 38)
            .padding(.horizontal, 12)
            .padding(.top, 7)

            HStack {
                Text("Conversations")
                Spacer()
                Text("\(filteredConversations.count)").monospacedDigit()
            }
            .font(.system(size: 10, weight: .medium))
            .foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 23)
            .padding(.top, 25)
            .padding(.bottom, 11)

            ScrollView {
                LazyVStack(spacing: 2) {
                    ForEach(filteredConversations) { conversation in
                        Button { studio.selectConversation(conversation.id) } label: {
                            ConversationRow(
                                conversation: conversation,
                                selected: conversation.id == studio.selectedConversationID
                            )
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button(L("Exporter en Markdown", "Export Markdown")) { studio.selectConversation(conversation.id); studio.exportMarkdown() }
                            Button(L("Supprimer", "Delete"), role: .destructive) {
                                studio.deleteConversation(conversation.id)
                            }
                        }
                    }
                    if filteredConversations.isEmpty {
                        Text(search.isEmpty ? L("Vos conversations apparaîtront ici.", "Your conversations will appear here.") : L("Aucune conversation trouvée.", "No conversations found."))
                            .font(.system(size: 12))
                            .foregroundStyle(StudioTheme.quiet)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(12)
                    }
                }
                .padding(.horizontal, 12)
            }
            .scrollIndicators(.hidden)

            VStack(spacing: 2) {
                Rectangle().fill(StudioTheme.edge).frame(height: 0.5)
                    .padding(.horizontal, 10).padding(.bottom, 12)
                utilityButton(L("Projets et profils", "Projects and profiles"), icon: "folder", action: { studio.showWorkspace = true })
                utilityButton("MCP", icon: "shippingbox", action: { studio.showMCPManager = true })
                utilityButton(L("Modèles", "Models"), icon: "square.stack", action: studio.openModelManager)
                HStack(spacing: 0) {
                    utilityButton(L("Réglages", "Settings"), icon: "gearshape", action: studio.openAppSettings)
                    SidebarUpdateBadge(updater: studio.updateManager, action: studio.openAppSettings)
                        .padding(.trailing, 8)
                }
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 17)
        }
    }

    private func utilityButton(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .font(.system(size: 12))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 11)
                .frame(height: 35)
        }
        .buttonStyle(StudioControlStyle())
    }
}

private struct ConversationRow: View {
    let conversation: Conversation
    let selected: Bool
    @State private var hovered = false

    var body: some View {
        HStack(spacing: 9) {
            RoundedRectangle(cornerRadius: 1)
                .fill(selected ? StudioTheme.ink : .clear)
                .frame(width: 2, height: 13)
            Text(conversation.messages.isEmpty && ["Nouvelle conversation", "New conversation"].contains(conversation.title)
                 ? L("Nouvelle conversation", "New conversation") : conversation.title)
                .font(.system(size: 13, weight: selected ? .medium : .regular))
                .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.secondary)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .frame(height: 36)
        .background(Color.white.opacity(selected ? 0.055 : (hovered ? 0.03 : 0)),
                    in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
        .onHover { hovered = $0 }
        .help(conversation.title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

private struct ChatWorkspaceView: View {
    @EnvironmentObject private var studio: StudioModel
    @Binding var sidebarVisible: Bool

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceHeader(sidebarVisible: $sidebarVisible)
            if studio.selectedMode == .server {
                ServerWorkspaceView(server: studio.localServer)
            } else if studio.selectedMode == .lab {
                LabWorkspaceView()
            } else if let conversation = studio.currentConversation, !conversation.messages.isEmpty {
                if studio.selectedMode == .agent { AgentWorkspaceBar() }
                MessagesView(messages: conversation.messages)
                ComposerView()
                    .padding(.horizontal, 32)
                    .padding(.top, 10)
                    .padding(.bottom, 22)
            } else if studio.selectedMode == .agent {
                AgentWelcomeView()
            } else {
                WelcomeView()
            }
        }
    }
}

private struct WorkspaceHeader: View {
    @EnvironmentObject private var studio: StudioModel
    @Binding var sidebarVisible: Bool

    var body: some View {
        HStack(spacing: 10) {
            Button { sidebarVisible.toggle() } label: {
                Image(systemName: "sidebar.left").frame(width: 30, height: 30)
            }
            .buttonStyle(StudioControlStyle())
            .help(sidebarVisible ? L("Masquer l’historique", "Hide history") : L("Afficher l’historique", "Show history"))
            .accessibilityLabel(L("Afficher ou masquer l’historique", "Toggle history"))
            .keyboardShortcut("s", modifiers: [.command, .control])

            Text(studio.selectedMode == .server || studio.selectedMode == .lab ? studio.selectedMode.title
                 : studio.currentConversation?.messages.isEmpty == false ? (studio.currentConversation?.title ?? "Conversation") : studio.selectedMode.title)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(StudioTheme.secondary)
                .lineLimit(1)
                .layoutPriority(-1)

            Spacer(minLength: 12)

            Menu {
                ForEach(studio.models) { model in
                    Button { studio.selectModel(model.name) } label: {
                        if model.name == studio.selectedModelName {
                            Label(model.name, systemImage: "checkmark")
                        } else {
                            Text(model.name)
                        }
                    }
                }
                if !studio.models.isEmpty { Divider() }
                Button(action: studio.openModelManager) {
                    Label(L("Ajouter un modèle…", "Add a model…"), systemImage: "square.and.arrow.down")
                }
            } label: {
                HStack(spacing: 8) {
                    StatusDot(state: studio.engineState)
                    Text(studio.selectedModelName ?? L("Choisir un modèle", "Choose a model"))
                        .font(.system(size: 11, weight: .medium))
                        .lineLimit(1).truncationMode(.middle)
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
                        .foregroundStyle(StudioTheme.quiet)
                }
                .padding(.horizontal, 11)
                .frame(height: 30)
                .frame(maxWidth: 260)
            }
            .menuStyle(.button)
            .buttonStyle(StudioControlStyle())
            .menuIndicator(.hidden)
            .fixedSize(horizontal: false, vertical: true)
            .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 6))
            .disabled(studio.isGenerating || studio.localServer.active)
            .help(studio.engineState.label)
            .accessibilityLabel(L("Modèle : \(studio.selectedModelName ?? "aucun")", "Model: \(studio.selectedModelName ?? "none")"))

            Button(action: studio.ejectModel) {
                Image(systemName: "eject").frame(width: 30, height: 30)
            }
            .buttonStyle(StudioControlStyle())
            .disabled(!studio.canEject)
            .help(L("Éjecter le modèle et libérer la mémoire Metal", "Eject the model and free Metal memory"))
            .accessibilityLabel(L("Éjecter le modèle", "Eject model"))

            MTPTuningToolbarControl()
            if studio.dflash2Available {
                MTPTuningToolbarControl(dflash: true)
            }

            Button { studio.showInspector.toggle() } label: {
                Image(systemName: "slider.horizontal.3").frame(width: 30, height: 30)
            }
            .buttonStyle(StudioControlStyle(emphasized: studio.showInspector))
            .help(L("Réglages de génération", "Generation settings"))
            .accessibilityLabel(L("Réglages de génération", "Generation settings"))
        }
        .font(.system(size: 13))
        .padding(.horizontal, 19)
        .padding(.top, sidebarVisible ? 14 : 30)
        .padding(.bottom, 14)
        .overlay(alignment: .bottom) {
            Rectangle().fill(StudioTheme.edge).frame(height: 0.5)
        }
    }
}

private struct AgentWorkspaceBar: View {
    @EnvironmentObject private var studio: StudioModel
    var body: some View {
        HStack(spacing: 18) {
            Button(action: studio.openAgentWorkspace) { Label(L("Dossier de travail", "Workspace folder"), systemImage: "folder") }
                .buttonStyle(.plain).help(L("Ouvrir le dossier isolé de cette tâche dans le Finder", "Open this task’s isolated folder in Finder"))
            Label(L("Ce Mac", "This Mac"), systemImage: "desktopcomputer")
            Spacer()
            Label(L("Sandbox · hors ligne", "Sandbox · offline"), systemImage: "lock.shield")
        }
        .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
        .padding(.horizontal, 22).padding(.vertical, 13)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 9))
    }
}

private struct AgentWelcomeView: View {
    @EnvironmentObject private var studio: StudioModel
    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 35)
            MonogramMark(size: 80).frame(height: 48).clipped()
            Text(L("Que voulez-vous construire ?", "What would you like to build?"))
                .font(.system(size: 32, weight: .regular, design: .serif)).tracking(-0.7)
                .multilineTextAlignment(.center).padding(.top, 20)
            Text(L("Votre modèle local, un espace de travail isolé.", "Your local model, an isolated workspace."))
                .font(.system(size: 13)).foregroundStyle(StudioTheme.quiet).padding(.top, 12)
            Spacer(minLength: 45)
            VStack(spacing: 0) {
                AgentWorkspaceBar().padding(.horizontal, 12).padding(.bottom, -7)
                ComposerView()
            }
            if !studio.mcpErrors.isEmpty {
                Text(studio.mcpErrors.values.sorted().joined(separator: "\n")).font(.caption).foregroundStyle(.red).textSelection(.enabled).padding(.top, 12)
            }
            if case let .failed(message) = studio.engineState {
                Text(message).font(.caption).foregroundStyle(.red).textSelection(.enabled).lineLimit(5).padding(.top, 12)
            }
            Text(L("Seuls les fichiers de cette tâche sont accessibles. Réseau bloqué ; commandes limitées à 5 s. Ouvrez le dossier pour y déposer les fichiers à modifier.", "Only this task’s files are accessible. Network blocked; commands limited to 5 s. Open the folder to add files to edit."))
                .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet).padding(.top, 13)
            Spacer().frame(height: 28)
        }
        .frame(maxWidth: 760).padding(.horizontal, 40).frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct WelcomeView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Spacer(minLength: 32)
            VStack(alignment: .leading, spacing: 14) {
                Text(L("Nouvelle conversation", "New conversation"))
                    .font(.system(size: 34, weight: .regular, design: .serif))
                    .tracking(-0.8)
                    .foregroundStyle(StudioTheme.ink)
                Text(studio.models.isEmpty
                     ? L("Ajoutez un modèle pour commencer.", "Add a model to get started.")
                     : L("Échangez avec un modèle exécuté sur ce Mac.", "Chat with a model running on this Mac."))
                    .font(.system(size: 13))
                    .foregroundStyle(StudioTheme.quiet)
            }
            .padding(.horizontal, 4)
            .padding(.bottom, 30)

            ComposerView()

            if case let .failed(message) = studio.engineState {
                DisclosureGroup(L("Le modèle n’a pas pu être chargé", "The model could not be loaded")) {
                    Text(message)
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxHeight: 150)
                }
                .font(.system(size: 12))
                .foregroundStyle(Color(red: 0.94, green: 0.57, blue: 0.5))
                .padding(.top, 18)
            }

            HStack(spacing: 18) {
                Button(action: studio.openModelManager) {
                    Label(studio.models.isEmpty ? L("Ajouter un modèle", "Add a model") : L("Gérer les modèles", "Manage models"), systemImage: "square.stack")
                }
                .buttonStyle(.plain)
                .help(L("Importer ou télécharger un modèle EXL3", "Import or download an EXL3 model"))
                Spacer()
                Text(L("⌘N  Nouvelle conversation", "⌘N  New conversation"))
            }
            .font(.system(size: 10))
            .foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 4)
            .padding(.top, 20)
            Spacer(minLength: 32)
            Spacer().frame(maxHeight: 70)
        }
        .frame(maxWidth: 680)
        .padding(.horizontal, 40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
