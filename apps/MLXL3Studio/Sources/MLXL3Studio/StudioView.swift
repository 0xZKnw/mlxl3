import SwiftUI

/// Window composition. The transcript owns its scroll state; chrome never
/// observes token-by-token text updates.
struct StudioView: View {
    @Environment(\.studioAppTheme) private var appTheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var studio: StudioModel
    @AppStorage("studio.sidebarVisible") private var sidebarVisible = true

    var body: some View {
        GeometryReader { geometry in
            HStack(spacing: 0) {
                if sidebarVisible {
                    SidebarView()
                        .frame(width: studio.sidebarWidth.points)
                        .modifier(StudioSidebarSurface())
                }
                ChatWorkspaceView(sidebarVisible: $sidebarVisible)

                if studio.showInspector && (studio.selectedMode == .chat || studio.selectedMode == .agent) && geometry.size.width >= 1280 {
                    GenerationInspector().frame(width: 340).padding(8)
                }
            }
            .background {
                StudioAurora(busy: studio.isGenerating, animated: !studio.showAppSettings && !studio.showWorkspace && !studio.showModelManager && !studio.showMCPManager && studio.studioDocument == nil)
                    .background(StudioTheme.canvas(for: appTheme))
            }
            .overlay(alignment: .trailing) {
                if studio.showInspector && (studio.selectedMode == .chat || studio.selectedMode == .agent) && geometry.size.width < 1280 {
                  ZStack(alignment: .trailing) {
                    Color.black.opacity(0.28)
                        .contentShape(Rectangle())
                        .onTapGesture { studio.showInspector = false }
                    GenerationInspector()
                        .frame(width: 340)
                        .shadow(color: .black.opacity(0.14), radius: 16, x: -3)
                        .padding(8)
                  }
                  .onExitCommand { studio.showInspector = false }
                }
            }
        }
        .environment(\.conversationDensity, studio.conversationDensity)
        .environment(\.conversationFont, studio.conversationFont)
        .environment(\.messageBubbleStyle, studio.messageBubbleStyle)
        .environment(\.conversationReadingWidth, studio.readingWidth)
        .environment(\.conversationTextScale, CGFloat(studio.responseTextSize / ResponseTextSize.defaultValue))
        .id(studio.language)
        .disabled(studio.studioDocument != nil)
        .accessibilityHidden(studio.studioDocument != nil)
        .overlay {
            if let document = studio.studioDocument {
                AgentStudioView(workspace: document.workspace, file: document.file, onClose: { studio.studioDocument = nil })
                    .id(document.id)
                    .environmentObject(studio)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(StudioTheme.canvas)
            }
        }
        .environment(\.locale, Locale(identifier: studio.language.rawValue))
        .disabled(studio.showAppSettings)
        .allowsHitTesting(!studio.showAppSettings)
        .accessibilityHidden(studio.showAppSettings)
        .overlay {
            if studio.showAppSettings {
                AppSettingsView().environmentObject(studio).transition(.opacity)
            }
        }
        .disabled(studio.showModelManager || studio.showWorkspace || studio.showMCPManager)
        .allowsHitTesting(!studio.showModelManager && !studio.showWorkspace && !studio.showMCPManager)
        .accessibilityHidden(studio.showModelManager || studio.showWorkspace || studio.showMCPManager)
        .overlay {
            if studio.showMCPManager {
                MCPManagerView().environmentObject(studio).transition(.opacity)
            } else if studio.showWorkspace {
                WorkspaceManagerView().environmentObject(studio).transition(.opacity)
            } else if studio.showModelManager {
                ModelManagerView().environmentObject(studio).transition(.opacity)
            }
        }
        .background(StudioTheme.sidebar(for: appTheme))
        .ignoresSafeArea()
        .frame(minWidth: 820, minHeight: 600)
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2),
                   value: [studio.showAppSettings, studio.showModelManager, studio.showWorkspace, studio.showMCPManager])
        .alert(L("MLXL3", "MLXL3"), isPresented: Binding(get: { studio.featureError != nil }, set: { if !$0 { studio.featureError = nil } })) {
            Button("OK") { studio.featureError = nil }
        } message: { Text(studio.featureError ?? "") }
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
        studio.sidebarConversations(matching: search)
    }

    var body: some View {
        let visibleConversations = filteredConversations
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 8) {
                MonogramMark(size: 35)
                Text("MLXL3")
                    .font(.system(size: 17, weight: .semibold))
                    .tracking(-0.7)
                Spacer(minLength: 0)
                DownloadNotificationButton(library: studio.modelLibrary)
            }
            .foregroundStyle(StudioTheme.ink)
            .padding(.horizontal, 18)
            .padding(.top, 37).padding(.bottom, 16)

            VStack(spacing: 2) {
                ForEach(StudioMode.allCases) { mode in
                    Button { studio.selectMode(mode) } label: {
                        HStack(spacing: 11) {
                            Image(systemName: mode.icon).frame(width: 18)
                            Text(mode.title)
                            Spacer()
                        }
                        .font(.system(size: 13, weight: studio.selectedMode == mode ? .semibold : .regular))
                        .padding(.horizontal, 12).frame(height: 32)
                    }
                    .buttonStyle(StudioControlStyle(emphasized: studio.selectedMode == mode))
                    .accessibilityLabel(mode.title)
                    .accessibilityAddTraits(studio.selectedMode == mode ? .isSelected : [])
                }
            }
            .padding(.horizontal, 10).disabled(studio.isGenerating)

            Picker(L("Projet", "Project"), selection: Binding(get: { studio.selectedProjectID }, set: { studio.chooseProject($0) })) {
                Text(L("Sans projet", "No project")).tag(UUID?.none)
                ForEach(studio.projects) { Text($0.name).tag(Optional($0.id)) }
            }.labelsHidden().padding(.horizontal, 19).padding(.top, 16).disabled(studio.isGenerating)

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
            .frame(height: 32)
            .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal, 12)
            .padding(.top, 10)

            Button(action: studio.newConversation) {
                HStack(spacing: 9) {
                    Image(systemName: "square.and.pencil")
                    Text(studio.selectedMode == .agent
                         ? L("Nouvelle tâche", "New task") : L("Nouvelle conversation", "New conversation"))
                        .lineLimit(1)
                    Spacer(minLength: 0)
                }
                .font(.system(size: 12, weight: .medium))
                .padding(.horizontal, 12).frame(height: 36)
            }
            .buttonStyle(StudioToolbarStyle())
            .accessibilityIdentifier("sidebar.newConversation")
            .help(L("Nouvelle conversation · ⌘N", "New conversation · ⌘N"))
            .padding(.horizontal, 12).padding(.top, 12)

            HStack {
                Text("Conversations")
                Text("\(visibleConversations.count)").monospacedDigit()
                Spacer()
            }
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 21)
            .padding(.top, 18)
            .padding(.bottom, 8)

            ScrollView {
                let conversations = visibleConversations
                let pinnedID = conversations.first(where: \.isPinned)?.id
                let recentID = pinnedID == nil ? nil : conversations.first(where: { !$0.isPinned })?.id
                LazyVStack(spacing: 2) {
                    ForEach(conversations) { conversation in
                        if conversation.id == pinnedID {
                            Text(L("ÉPINGLÉES", "PINNED"))
                                .font(.system(size: 9, weight: .semibold)).foregroundStyle(StudioTheme.quiet)
                                .frame(maxWidth: .infinity, alignment: .leading).padding(.leading, 10).padding(.vertical, 8)
                        }
                        if conversation.id == recentID {
                            Text(L("RÉCENTES", "RECENT"))
                                .font(.system(size: 9, weight: .semibold)).foregroundStyle(StudioTheme.quiet)
                                .frame(maxWidth: .infinity, alignment: .leading).padding(.leading, 10).padding(.vertical, 8)
                        }
                        SidebarConversationItem(conversation: conversation)
                    }
                    if conversations.isEmpty {
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
                    SidebarUpdateBadge(updater: studio.updateManager, action: studio.openUpdateSettings)
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

private struct SidebarConversationItem: View {
    @EnvironmentObject private var studio: StudioModel
    let conversation: Conversation
    @State private var hovered = false

    var body: some View {
        HStack(spacing: 0) {
            Button { studio.selectConversation(conversation.id) } label: {
                ConversationRow(conversation: conversation, selected: conversation.id == studio.selectedConversationID)
            }
            .buttonStyle(.plain)
            Button { studio.setConversationPinned(conversation.id, !conversation.isPinned) } label: {
                Image(systemName: conversation.isPinned ? "pin.fill" : "pin")
                    .font(.system(size: 11)).frame(width: 26, height: 32)
                    .foregroundStyle(conversation.isPinned ? StudioTheme.ink : StudioTheme.quiet)
            }
            .buttonStyle(.plain)
            .opacity(conversation.isPinned || hovered ? 1 : 0.45)
            .help(conversation.isPinned ? L("Désépingler", "Unpin") : L("Épingler", "Pin"))
            .accessibilityLabel((conversation.isPinned ? L("Désépingler ", "Unpin ") : L("Épingler ", "Pin ")) + conversation.title)
        }
        .onHover { hovered = $0 }
        .contextMenu {
            Button(conversation.isPinned ? L("Désépingler", "Unpin") : L("Épingler", "Pin"),
                   systemImage: conversation.isPinned ? "pin.slash" : "pin") {
                studio.setConversationPinned(conversation.id, !conversation.isPinned)
            }
            Button(L("Supprimer", "Delete"), role: .destructive) { studio.deleteConversation(conversation.id) }
        }
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
        .background(StudioTheme.contrast.opacity(selected ? 0.055 : (hovered ? 0.03 : 0)),
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
            if case let .failed(message) = studio.engineState {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: "exclamationmark.triangle")
                    Text(message).font(.caption).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Button(L("Réessayer", "Retry"), action: studio.retrySelectedModel)
                        .disabled(studio.selectedModel == nil)
                }
                .foregroundStyle(StudioTheme.ink)
                .padding(12)
                .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10))
                .padding(.horizontal, 24)
            }
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

            if !sidebarVisible {
                DownloadNotificationButton(library: studio.modelLibrary)
            }

            Text(studio.selectedMode == .server || studio.selectedMode == .lab ? studio.selectedMode.title
                 : studio.currentConversation?.messages.isEmpty == false ? (studio.currentConversation?.title ?? "Conversation") : studio.selectedMode.title)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(StudioTheme.secondary)
                .lineLimit(1)
                .layoutPriority(-1)

            Spacer(minLength: 12)

            ModelSelectionMenu()

            HStack(spacing: 3) {
                ConversationDisplayControl()
                MTPToolbarControl()
                if studio.dflash2Available {
                    MTPTuningToolbarControl(dflash: true)
                }
                if studio.selectedMode == .chat || studio.selectedMode == .agent {
                    Button { studio.showInspector.toggle() } label: {
                        Image(systemName: "slider.horizontal.3").frame(width: 30, height: 30)
                    }
                    .buttonStyle(WorkspaceControlStyle(selected: studio.showInspector))
                    .accessibilityIdentifier("toolbar.generation")
                    .help(L("Réglages de génération", "Generation settings"))
                    .accessibilityLabel(L("Réglages de génération", "Generation settings"))
                }
            }
            .padding(4)
            .premiumGlass(radius: 100)
        }
        .font(.system(size: 13))
        .padding(.leading, sidebarVisible ? 22 : 82)
        .padding(.trailing, 18)
        .padding(.vertical, 14)
    }
}

private struct AgentWorkspaceBar: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var htmlFiles: [URL] = []
    @State private var previewError: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
        HStack(spacing: 18) {
            Button { if let workspace = studio.agentWorkspaceURL { studio.studioDocument = StudioDocument(workspace: workspace) } } label: { Label("Agent Studio", systemImage: "sidebar.right") }.buttonStyle(.bordered)
                .disabled(studio.isGenerating || studio.agentStudioBusy)
            Button(action: studio.openAgentWorkspace) { Label(L("Dossier de travail", "Workspace folder"), systemImage: "folder") }
                .buttonStyle(.plain).help(L("Ouvrir le dossier isolé de cette tâche dans le Finder", "Open this task’s isolated folder in Finder"))
            Label(L("Ce Mac", "This Mac"), systemImage: "desktopcomputer")
            Spacer()
            Label(L("Outils sans réseau · aperçu avec imports distants possibles", "Tools offline · preview may load remote imports"), systemImage: "lock.shield")
        }
        if !studio.isGenerating && !htmlFiles.isEmpty {
            ScrollView(.horizontal) {
                HStack(spacing: 10) {
                    ForEach(htmlFiles, id: \.self) { file in
                        Button {
                            guard let workspace = studio.agentWorkspaceURL else { return }
                            do {
                                _ = try HTMLPreview.source(at: file, in: workspace)
                                studio.studioDocument = StudioDocument(workspace: workspace, file: file)
                            }
                            catch { previewError = error.localizedDescription }
                        } label: { Label(file.lastPathComponent, systemImage: "play.rectangle") }
                        .buttonStyle(.bordered)
                        .help(L("Exécuter ce fichier HTML dans l’app", "Run this HTML file in the app"))
                    }
                }
            }
            .scrollIndicators(.hidden)
        }
        }
        .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
        .padding(.horizontal, 22).padding(.vertical, 13)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 9))
        .task(id: "\(studio.selectedConversationID?.uuidString ?? "")/\(studio.isGenerating)") {
            htmlFiles = []
            guard !studio.isGenerating, let workspace = studio.agentWorkspaceURL else { return }
            let files = await Task.detached { HTMLPreview.artifactURLs(in: workspace) }.value
            guard !Task.isCancelled else { return }
            htmlFiles = files
        }
        .alert(L("Aperçu HTML impossible", "HTML preview unavailable"), isPresented: Binding(
            get: { previewError != nil }, set: { if !$0 { previewError = nil } }
        )) { Button("OK") { previewError = nil } } message: { Text(previewError ?? "") }
    }
}

private struct WelcomeView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        let agent = studio.selectedMode == .agent
        ViewThatFits(in: .vertical) {
            content(agent: agent)
            ScrollView { content(agent: agent) }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func content(agent: Bool) -> some View {
        VStack(spacing: 0) {
            MonogramMark(size: 76)
                .frame(height: 49)
            Text(agent ? L("Nouvelle tâche", "New task") : L("Nouvelle conversation", "New conversation"))
                .font(.system(size: 44, weight: .regular, design: .serif))
                .tracking(-1.5)
                .multilineTextAlignment(.center)
                .foregroundStyle(StudioTheme.ink)
                .padding(.top, 15)
                .padding(.bottom, 32)

            if agent {
                AgentWorkspaceBar()
                    .padding(.bottom, 8)
            }
            ComposerView()
            if agent {
                if !studio.mcpErrors.isEmpty {
                    Text(studio.mcpErrors.values.sorted().joined(separator: "\n"))
                        .font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
                Text(L("Seuls les fichiers de cette tâche sont accessibles. Réseau bloqué ; commandes limitées à 5 s. Ouvrez le dossier pour y déposer les fichiers à modifier.", "Only this task’s files are accessible. Network blocked; commands limited to 5 s. Open the folder to add files to edit."))
                    .font(.caption).foregroundStyle(StudioTheme.secondary)
                    .multilineTextAlignment(.center).padding(.top, 12)
            } else {
                HStack(spacing: 12) {
                    Button(action: studio.openModelManager) {
                        Label(studio.models.isEmpty ? L("Ajouter un modèle", "Add a model") : L("Gérer les modèles", "Manage models"), systemImage: "square.stack")
                            .padding(.horizontal, 8).frame(height: 28)
                    }
                    .buttonStyle(StudioControlStyle())
                    .help(L("Importer ou télécharger un modèle EXL3", "Import or download an EXL3 model"))
                    Spacer()
                    if let model = studio.selectedModel {
                        Text([model.format, model.size].filter { !$0.isEmpty }.joined(separator: " · "))
                            .font(.system(size: 11, design: .monospaced))
                    }
                }
                .font(.system(size: 12))
                .foregroundStyle(StudioTheme.secondary)
                .padding(.horizontal, 4).padding(.top, 8)
            }
        }
        .padding(.vertical, 16)
        .frame(maxWidth: 660)
        .padding(.horizontal, 28)
    }
}
