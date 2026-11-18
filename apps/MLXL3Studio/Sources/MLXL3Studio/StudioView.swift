import SwiftUI

/// Window composition. The transcript owns its scroll state; chrome never
/// observes token-by-token text updates.
struct StudioView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var studio: StudioModel
    @AppStorage("studio.sidebarVisible") private var sidebarVisible = true

    private var coveredByPage: Bool {
        studio.showAppSettings || studio.showModelManager || studio.showWorkspace || studio.showMCPManager
    }

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
                StudioAurora(busy: studio.isGenerating, animated: !studio.showAppSettings && !studio.showWorkspace && !studio.showModelManager && !studio.showMCPManager && studio.studioDocument == nil,
                             fullHeight: true)
                    .background(StudioTheme.canvas)
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
        // Full-screen pages fade through instead of double-exposing the chat behind them.
        .opacity(coveredByPage ? 0 : 1)
        .scaleEffect(coveredByPage && !reduceMotion ? 0.985 : 1)
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
        .background(StudioTheme.sidebar)
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
                // The monogram alone names the app; the wordmark repeated it.
                MonogramMark(size: 35)
                    .accessibilityElement()
                    .accessibilityLabel("MLXL3")
                Spacer(minLength: 0)
                DownloadNotificationButton(library: studio.modelLibrary)
            }
            .foregroundStyle(StudioTheme.ink)
            .padding(.horizontal, 18)
            .padding(.top, 37).padding(.bottom, 16)

            SidebarModePicker()
                .padding(.horizontal, 10)
                .disabled(studio.isGenerating)

            SidebarProjectMenu()
                .padding(.horizontal, 10).padding(.top, 8)
                .disabled(studio.isGenerating)

            // Search and the new-conversation action share one row so the
            // conversation list keeps the vertical space.
            HStack(spacing: 6) {
                HStack(spacing: 8) {
                    Image(systemName: "magnifyingglass").font(.system(size: 11.5))
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
                .padding(.horizontal, 10)
                .frame(height: 30)
                .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(StudioTheme.edge, lineWidth: 0.5).allowsHitTesting(false)
                }

                Button(action: studio.newConversation) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 13, weight: .medium))
                        .frame(width: 30, height: 30)
                        .contentShape(Rectangle())
                }
                .buttonStyle(StudioToolbarStyle(materialOpacity: 0.76))
                .accessibilityIdentifier("sidebar.newConversation")
                .accessibilityLabel(newConversationTitle)
                .help(newConversationTitle + " · ⌘N")
            }
            .padding(.horizontal, 10)
            .padding(.top, 6)

            HStack(spacing: 7) {
                Text("Conversations")
                Text("\(visibleConversations.count)").monospacedDigit()
                    .font(.system(size: 10, weight: .semibold))
                    .padding(.horizontal, 6).frame(height: 16)
                    .background(StudioTheme.contrast.opacity(0.07), in: Capsule())
                Spacer()
            }
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 22)
            .padding(.top, 12)
            .padding(.bottom, 4)

            ScrollView {
                let conversations = visibleConversations
                let pinnedID = conversations.first(where: \.isPinned)?.id
                let recentID = pinnedID == nil ? nil : conversations.first(where: { !$0.isPinned })?.id
                LazyVStack(spacing: 2) {
                    ForEach(conversations) { conversation in
                        if conversation.id == pinnedID {
                            sectionLabel(L("ÉPINGLÉES", "PINNED"))
                        }
                        if conversation.id == recentID {
                            sectionLabel(L("RÉCENTES", "RECENT"))
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
                .padding(.horizontal, 10)
            }
            .scrollIndicators(.hidden)
            .mask(StudioEdgeFade(top: 10, bottom: 18))

            // Projects, MCP, Models and Settings as one icon row (named in the
            // tooltip and accessibility label) instead of four full-width rows.
            VStack(spacing: 0) {
                Rectangle().fill(StudioTheme.edge).frame(height: 0.5)
                    .padding(.horizontal, 12).padding(.bottom, 8)
                HStack(spacing: 2) {
                    utilityButton(L("Projets et profils", "Projects and profiles"), icon: "folder", action: { studio.showWorkspace = true })
                    utilityButton("MCP", icon: "shippingbox", action: { studio.showMCPManager = true })
                    utilityButton(L("Modèles", "Models"), icon: "square.stack", action: studio.openModelManager)
                    utilityButton(L("Réglages", "Settings"), icon: "gearshape", action: studio.openAppSettings)
                    SidebarUpdateBadge(updater: studio.updateManager, action: studio.openUpdateSettings)
                }
            }
            .padding(.horizontal, 10)
            .padding(.bottom, 12)
        }
    }

    private var newConversationTitle: String {
        studio.selectedMode == .agent ? L("Nouvelle tâche", "New task") : L("Nouvelle conversation", "New conversation")
    }

    private func sectionLabel(_ title: String) -> some View {
        Text(title)
            .font(.system(size: 9, weight: .semibold)).tracking(0.8)
            .foregroundStyle(StudioTheme.quiet)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.leading, 12).padding(.top, 10).padding(.bottom, 5)
    }

    private func utilityButton(_ title: String, icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 13.5))
                .frame(maxWidth: .infinity).frame(height: 32)
                .contentShape(Rectangle())
        }
        .buttonStyle(StudioControlStyle(cornerRadius: 8))
        .help(title)
        .accessibilityLabel(title)
    }
}

/// Chat, Agent, Server and Lab as one segmented row: four full-width rows used
/// to take the space of four conversations.
private struct SidebarModePicker: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Namespace private var selection

    var body: some View {
        HStack(spacing: 2) {
            ForEach(StudioMode.allCases) { mode in
                let selected = studio.selectedMode == mode
                Button { studio.selectMode(mode) } label: {
                    VStack(spacing: 3) {
                        Image(systemName: mode.icon).font(.system(size: 12.5, weight: selected ? .semibold : .regular))
                            .frame(height: 15)
                        Text(mode.title)
                            .font(.system(size: 10.5, weight: selected ? .semibold : .medium))
                            .lineLimit(1).minimumScaleFactor(0.8)
                    }
                    .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.quiet)
                    .frame(maxWidth: .infinity).frame(height: 40)
                    .background {
                        if selected {
                            RoundedRectangle(cornerRadius: 7, style: .continuous)
                                .fill(StudioTheme.contrast.opacity(0.09))
                                .overlay {
                                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                                        .strokeBorder(StudioTheme.edge, lineWidth: 0.5)
                                }
                                .matchedGeometryEffect(id: "mode", in: selection)
                        }
                    }
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(mode.title)
                .accessibilityLabel(mode.title)
                .accessibilityAddTraits(selected ? .isSelected : [])
            }
        }
        .padding(3)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 10, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 10, style: .continuous)
                .strokeBorder(StudioTheme.edge, lineWidth: 0.5).allowsHitTesting(false)
        }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.16), value: studio.selectedMode)
    }
}

/// The project choice shares the navigation rows' inset and surface instead of
/// the system pop-up chrome.
private struct SidebarProjectMenu: View {
    @EnvironmentObject private var studio: StudioModel

    private var projectName: String {
        studio.projects.first { $0.id == studio.selectedProjectID }?.name ?? L("Sans projet", "No project")
    }

    var body: some View {
        Menu {
            Button { studio.chooseProject(nil) } label: {
                if studio.selectedProjectID == nil {
                    Label(L("Sans projet", "No project"), systemImage: "checkmark")
                } else { Text(L("Sans projet", "No project")) }
            }
            if !studio.projects.isEmpty { Divider() }
            ForEach(studio.projects) { project in
                Button { studio.chooseProject(project.id) } label: {
                    if studio.selectedProjectID == project.id {
                        Label(project.name, systemImage: "checkmark")
                    } else { Text(project.name) }
                }
            }
        } label: {
            HStack(spacing: 11) {
                Image(systemName: studio.selectedProjectID == nil ? "folder" : "folder.fill").frame(width: 18)
                Text(projectName).lineLimit(1).truncationMode(.tail)
                Spacer(minLength: 4)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(StudioTheme.quiet)
            }
            .font(.system(size: 12, weight: .medium))
            .padding(.horizontal, 12)
            .frame(height: 30)
            .frame(maxWidth: .infinity)
            .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .strokeBorder(StudioTheme.edge, lineWidth: 0.5)
            }
        }
        .menuStyle(.button).buttonStyle(StudioControlStyle(cornerRadius: 8)).menuIndicator(.hidden)
        .accessibilityLabel(L("Projet", "Project"))
        .accessibilityValue(projectName)
        .help(L("Projet de la conversation", "Conversation project"))
    }
}

/// One surface per row: the selection and hover fill also cover the pin, and
/// an unpinned row only reserves room for the pin while it is hovered.
private struct SidebarConversationItem: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let conversation: Conversation
    @State private var hovered = false

    var body: some View {
        let selected = conversation.id == studio.selectedConversationID
        let showsPin = conversation.isPinned || hovered
        Button { studio.selectConversation(conversation.id) } label: {
            ConversationRow(conversation: conversation, selected: selected)
                .padding(.trailing, showsPin ? 26 : 0)
        }
        .buttonStyle(.plain)
        .overlay(alignment: .trailing) {
            Button { studio.setConversationPinned(conversation.id, !conversation.isPinned) } label: {
                Image(systemName: conversation.isPinned ? "pin.fill" : "pin")
                    .font(.system(size: 10.5)).frame(width: 26, height: 30)
                    .foregroundStyle(conversation.isPinned ? StudioTheme.secondary : StudioTheme.quiet)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .opacity(showsPin ? 1 : 0)
            .allowsHitTesting(showsPin)
            .padding(.trailing, 2)
            .help(conversation.isPinned ? L("Désépingler", "Unpin") : L("Épingler", "Pin"))
            .accessibilityLabel((conversation.isPinned ? L("Désépingler ", "Unpin ") : L("Épingler ", "Pin ")) + conversation.title)
        }
        .background(StudioTheme.contrast.opacity(selected ? 0.075 : (hovered ? 0.035 : 0)),
                    in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(alignment: .leading) {
            if selected {
                Capsule().fill(studio.accentColor)
                    .frame(width: 3, height: 14)
                    .offset(x: 1)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
        .onHover { hovered = $0 }
        .animation(reduceMotion ? nil : .easeOut(duration: 0.14), value: hovered)
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

    var body: some View {
        HStack(spacing: 0) {
            Text(conversation.messages.isEmpty && ["Nouvelle conversation", "New conversation"].contains(conversation.title)
                 ? L("Nouvelle conversation", "New conversation") : conversation.title)
                .font(.system(size: 13, weight: selected ? .medium : .regular))
                .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.secondary)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .frame(height: 30)
        .contentShape(Rectangle())
        .help(conversation.title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

private struct ChatWorkspaceView: View {
    @Namespace private var composerMotion
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
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
            if studio.selectedMode == .decision {
                DecisionWorkspaceView(decision: studio.decision)
            } else if studio.selectedMode == .server {
                ServerWorkspaceView(server: studio.localServer)
            } else if studio.selectedMode == .lab {
                LabWorkspaceView()
            } else if let conversation = studio.currentConversation, !conversation.messages.isEmpty {
                if studio.selectedMode == .agent { AgentWorkspaceBar() }
                MessagesView(messages: conversation.messages)
                ComposerView()
                    .matchedGeometryEffect(id: "composer", in: composerMotion, properties: .position)
                    .padding(.horizontal, 32)
                    .padding(.top, 10)
                    .padding(.bottom, 22)
            } else {
                WelcomeView(composerMotion: composerMotion)
                    .transition(.opacity)
            }
        }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.28), value: studio.lastSubmittedMessageID)
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

            Text(studio.selectedMode.standalone ? studio.selectedMode == .decision ? L("Décision", "Decision") : studio.selectedMode.title
                 : studio.currentConversation?.messages.isEmpty == false ? (studio.currentConversation?.title ?? "Conversation") : studio.selectedMode.title)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(StudioTheme.secondary)
                .lineLimit(1)
                .layoutPriority(-1)

            Spacer(minLength: 12)

            if studio.selectedMode != .decision {
                ModelSelectionMenu()
            }

            HStack(spacing: 3) {
                if studio.selectedMode != .decision {
                    ConversationDisplayControl()
                    MTPToolbarControl()
                }
                if studio.dflash2Available && studio.selectedMode != .decision {
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
            .premiumGlass(radius: 100, materialOpacity: 0.76)
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
    let composerMotion: Namespace.ID
    @EnvironmentObject private var studio: StudioModel

    /// Stable while a conversation stays open; a new conversation draws again.
    private var greeting: String {
        let seed = studio.selectedConversationID.map { UInt64(truncatingIfNeeded: $0.hashValue) } ?? 0
        let phrase = WelcomeGreeting.pick(hour: Calendar.current.component(.hour, from: .now), seed: seed)
        return L(phrase.french, phrase.english)
    }

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
            Text(agent ? L("Nouvelle tâche", "New task") : greeting)
                .font(.system(size: 44, weight: .regular, design: .serif))
                .tracking(-1.5)
                .multilineTextAlignment(.center)
                .foregroundStyle(StudioTheme.ink)
                .padding(.top, 15)
                .padding(.bottom, 32)
                .accessibilityAddTraits(.isHeader)

            if agent {
                AgentWorkspaceBar()
                    .padding(.bottom, 8)
            }
            ComposerView()
                .matchedGeometryEffect(id: "composer", in: composerMotion, properties: .position)
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

/// Welcome titles: mostly tied to the time of day, sometimes one that fits any hour.
enum WelcomeGreeting {
    struct Phrase: Equatable {
        let french: String
        let english: String
    }

    static let morning = [
        Phrase(french: "Bonjour", english: "Good morning"),
        Phrase(french: "Bien réveillé ?", english: "Awake yet?"),
        Phrase(french: "Un café, une idée ?", english: "Coffee and an idea?"),
        Phrase(french: "On commence par quoi ?", english: "Where do we start?"),
    ]
    static let afternoon = [
        Phrase(french: "Bon après-midi", english: "Good afternoon"),
        Phrase(french: "On s’y remet ?", english: "Back at it?"),
        Phrase(french: "Qu’est-ce qu’on construit ?", english: "What are we building?"),
    ]
    static let evening = [
        Phrase(french: "Bonsoir", english: "Good evening"),
        Phrase(french: "Encore une idée ?", english: "One more idea?"),
        Phrase(french: "Une dernière pour ce soir ?", english: "One last one tonight?"),
    ]
    static let night = [
        Phrase(french: "Encore debout ?", english: "Still up?"),
        Phrase(french: "Les idées viennent la nuit", english: "Ideas come at night"),
        Phrase(french: "Nuit blanche ?", english: "Pulling an all-nighter?"),
    ]
    static let anytime = [
        Phrase(french: "De quoi parle-t-on ?", english: "What’s on your mind?"),
        Phrase(french: "Par où commencer ?", english: "Where to begin?"),
        Phrase(french: "Je vous écoute", english: "I’m listening"),
    ]

    static func pool(hour: Int) -> [Phrase] {
        switch hour {
        case 5..<12: morning
        case 12..<18: afternoon
        case 18..<23: evening
        default: night
        }
    }

    /// One draw in four uses the any-hour pool; out-of-range hours fall back to night.
    static func pick(hour: Int, seed: UInt64) -> Phrase {
        let timed = pool(hour: hour)
        if seed % 4 == 3 { return anytime[Int((seed / 4) % UInt64(anytime.count))] }
        return timed[Int((seed / 4) % UInt64(timed.count))]
    }
}
