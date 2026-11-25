import SwiftUI

/// Window composition. The transcript owns its scroll state; chrome never
/// observes token-by-token text updates.
struct StudioView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @EnvironmentObject private var studio: StudioModel
    @AppStorage("studio.sidebarVisible") private var sidebarVisible = true

    private var coveredByPage: Bool {
        studio.showImages || studio.showAppSettings || studio.showModelManager || studio.showWorkspace || studio.showMCPManager
    }

    var body: some View {
        GeometryReader { geometry in
            HStack(spacing: 0) {
                if sidebarVisible && studio.selectedMode != .lab {
                    SidebarView()
                        .frame(width: studio.sidebarWidth.points)
                        .modifier(StudioSidebarSurface())
                }
                if studio.selectedMode == .lab {
                    LabWorkspaceView(imageGeneration: studio.imageGeneration, onBack: studio.leaveLab)
                } else {
                    ChatWorkspaceView(sidebarVisible: $sidebarVisible)
                }

                if studio.showInspector && (studio.selectedMode == .chat || studio.selectedMode == .agent) && geometry.size.width >= 1280 {
                    GenerationInspector().frame(width: 340).padding(8)
                }
            }
            .background {
                StudioAurora(busy: studio.isGenerating, animated: !coveredByPage && studio.studioDocument == nil,
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
                AgentStudioView(workspace: document.workspace, file: document.file, showPermissions: document.showPermissions, turnResult: document.turnResult, onClose: { studio.studioDocument = nil })
                    .id(document.id)
                    .environmentObject(studio)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .background(StudioTheme.canvas)
            }
        }
        .disabled(studio.showImages)
        .accessibilityHidden(studio.showImages)
        .allowsHitTesting(!studio.showImages)
        .overlay {
            if studio.showImages && Image21Hardware.available {
                ImageGenerationView(engine: studio.imageGeneration, close: { studio.showImages = false })
                    .transition(.opacity)
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
                   value: [studio.showImages, studio.showAppSettings, studio.showModelManager, studio.showWorkspace, studio.showMCPManager])
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

/// One left edge for everything: rows sit 8 pt from the sidebar edge and their
/// content 10 pt further in, so icons, titles and section headings share x = 18.
private enum SidebarMetrics {
    static let inset: CGFloat = 8
    static let content: CGFloat = 10
    static let row: CGFloat = 28
    static let radius: CGFloat = 7
}

/// Search and new conversation at the top, the modes and Image as labelled rows,
/// the history, and one footer row instead of four unlabelled icons.
private struct SidebarView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.studioStyle) private var style
    @State private var search = ""
    @FocusState private var searchFocused: Bool

    var body: some View {
        let conversations = studio.sidebarConversations(matching: search)
        VStack(alignment: .leading, spacing: 0) {
            // The window controls share this row; the actions line up with the workspace toolbar.
            HStack(spacing: 2) {
                Spacer(minLength: 0)
                DownloadNotificationButton(library: studio.modelLibrary)
                Button(action: studio.newConversation) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 14))
                        .frame(width: 30, height: 30)
                        .contentShape(Rectangle())
                }
                .buttonStyle(StudioControlStyle(cornerRadius: 8))
                .accessibilityIdentifier("sidebar.newConversation")
                .accessibilityLabel(newConversationTitle)
                .help(newConversationTitle + " · ⌘N")
            }
            .font(.system(size: 14))
            .frame(height: 30)
            .padding(.horizontal, SidebarMetrics.inset)
            .padding(.top, 14)

            searchField
                .padding(.horizontal, SidebarMetrics.inset)
                .padding(.top, 8)

            VStack(spacing: 1) {
                VStack(spacing: 1) {
                    // Lab lives in the footer "…"; the others keep the ⌘-numbers of the View menu.
                    ForEach(Array(StudioMode.allCases.enumerated()).filter { $0.element != .lab }, id: \.element) { index, mode in
                        SidebarNavRow(icon: mode.icon, title: mode.displayTitle, selected: studio.selectedMode == mode,
                                      shortcut: "⌘\(index + 1)") { studio.selectMode(mode) }
                    }
                }
                .disabled(studio.isGenerating)
                // Images open their own page, available while a chat reply streams (as before).
                if Image21Hardware.available {
                    SidebarNavRow(icon: "photo", title: L("Image", "Image"), selected: studio.showImages,
                                  detail: L("Générer une image", "Generate an image")) {
                        studio.showImages = true
                    }
                }
            }
            .padding(.horizontal, SidebarMetrics.inset)
            .padding(.top, 10)

            SidebarScopeMenu()
                .padding(.horizontal, SidebarMetrics.inset)
                .padding(.top, 16)
                .disabled(studio.isGenerating)

            ScrollView {
                // Only Chat and Agent open a conversation; elsewhere no row reads as current.
                let showsSelection = studio.selectedMode == .chat || studio.selectedMode == .agent
                let pinnedID = conversations.first(where: \.isPinned)?.id
                let recentID = pinnedID == nil ? nil : conversations.first(where: { !$0.isPinned })?.id
                LazyVStack(alignment: .leading, spacing: 1) {
                    ForEach(conversations) { conversation in
                        if conversation.id == pinnedID {
                            SidebarSectionHeader(title: L("Épinglées", "Pinned"), first: true)
                        }
                        if conversation.id == recentID {
                            SidebarSectionHeader(title: L("Récentes", "Recent"), first: false)
                        }
                        SidebarConversationItem(conversation: conversation, showsSelection: showsSelection)
                    }
                    if conversations.isEmpty { emptyState }
                }
                .padding(.horizontal, SidebarMetrics.inset)
                .padding(.bottom, 14)
            }
            .scrollIndicators(.hidden)
            .mask(StudioEdgeFade(top: 6, bottom: 22))

            SidebarFooter()
                .padding(.horizontal, SidebarMetrics.inset)
                .padding(.top, 4)
                .padding(.bottom, 10)
        }
    }

    private var newConversationTitle: String {
        studio.selectedMode == .agent ? L("Nouvelle tâche", "New task") : L("Nouvelle conversation", "New conversation")
    }

    private var searchField: some View {
        let graphite = style.isGraphite
        return HStack(spacing: 7) {
            Image(systemName: "magnifyingglass")
                .font(.system(size: 12))
                .foregroundStyle(StudioTheme.quiet)
            TextField(L("Rechercher", "Search"), text: $search)
                .textFieldStyle(.plain)
                .focused($searchFocused)
                .onExitCommand { search = ""; searchFocused = false }
                .accessibilityLabel(L("Rechercher dans les conversations", "Search conversations"))
            if !search.isEmpty {
                Button { search = "" } label: {
                    Image(systemName: "xmark.circle.fill").foregroundStyle(StudioTheme.quiet)
                }
                .buttonStyle(.plain)
                .help(L("Effacer la recherche", "Clear search"))
            } else if !searchFocused {
                Text("⌘K")
                    .font(.system(size: 11))
                    .foregroundStyle(StudioTheme.quiet)
                    .accessibilityHidden(true)
            }
        }
        .font(.system(size: 13))
        .foregroundStyle(StudioTheme.ink)
        .padding(.leading, SidebarMetrics.content)
        .padding(.trailing, 9)
        .frame(height: 30)
        .background(graphite ? StudioTheme.contrast.opacity(searchFocused ? 0.07 : 0.05) : StudioTheme.panel,
                    in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .strokeBorder(searchFocused ? StudioTheme.contrast.opacity(0.16) : StudioTheme.edge,
                              lineWidth: graphite && !searchFocused ? 0 : 0.5)
                .allowsHitTesting(false)
        }
        .background {
            // ⌘K focuses the search; Escape clears it and gives the focus back.
            Button("") { searchFocused = true }
                .keyboardShortcut("k", modifiers: .command)
                .opacity(0).allowsHitTesting(false).accessibilityHidden(true)
        }
    }

    private var emptyState: some View {
        VStack(alignment: .leading, spacing: 4) {
            if search.isEmpty {
                Text(studio.selectedMode == .agent ? L("Aucune tâche", "No tasks yet") : L("Aucune conversation", "No conversations yet"))
                    .font(.system(size: 12.5, weight: .medium))
                    .foregroundStyle(StudioTheme.secondary)
                Text(L("Commencez avec ⌘N.", "Start one with ⌘N."))
            } else {
                Text(L("Aucun résultat pour « \(search) »", "No results for “\(search)”"))
                    .lineLimit(2)
            }
        }
        .font(.system(size: 12))
        .foregroundStyle(StudioTheme.quiet)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, SidebarMetrics.content)
        .padding(.top, 10)
    }
}

/// Hover, press and selection share one fill so every sidebar row answers alike.
/// Hover is instant: rows are passed over all day and should not shimmer.
private struct SidebarRowStyle: ButtonStyle {
    var selected = false
    var hovered = false

    func makeBody(configuration: Configuration) -> some View {
        Surface(configuration: configuration, selected: selected, hovered: hovered)
    }

    private struct Surface: View {
        let configuration: ButtonStyle.Configuration
        let selected: Bool
        let hovered: Bool
        @Environment(\.isEnabled) private var enabled
        @Environment(\.studioStyle) private var style

        var body: some View {
            let fill = !enabled ? (selected ? 0.05 : 0)
                : configuration.isPressed ? 0.11
                : selected ? (style.isGraphite ? 0.085 : 0.075)
                : hovered ? 0.045 : 0
            configuration.label
                .background(StudioTheme.contrast.opacity(fill),
                            in: RoundedRectangle(cornerRadius: SidebarMetrics.radius, style: .continuous))
                .contentShape(RoundedRectangle(cornerRadius: SidebarMetrics.radius, style: .continuous))
                .opacity(enabled ? 1 : selected ? 0.8 : 0.45)
        }
    }
}

extension StudioMode {
    var displayTitle: String { self == .decision ? L("Décision", "Decision") : title }
}

/// A place as a labelled row: the icon alone did not say what Decision or Lab were.
/// A mode's ⌘-number appears on hover, next to the place it opens.
private struct SidebarNavRow: View {
    @Environment(\.isEnabled) private var enabled
    let icon: String
    let title: String
    let selected: Bool
    var shortcut: String? = nil
    var detail: String? = nil
    let action: () -> Void
    @State private var hovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 9) {
                Image(systemName: icon)
                    .font(.system(size: 13, weight: selected ? .medium : .regular))
                    .frame(width: 18)
                    .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.quiet)
                Text(title)
                    .font(.system(size: 13, weight: selected ? .medium : .regular))
                    .foregroundStyle(selected ? StudioTheme.ink : StudioTheme.secondary)
                Spacer(minLength: 4)
                if hovered && enabled, let shortcut {
                    Text(shortcut)
                        .font(.system(size: 11))
                        .monospacedDigit()
                        .foregroundStyle(StudioTheme.quiet)
                        .accessibilityHidden(true)
                }
            }
            .padding(.leading, 9)
            .padding(.trailing, SidebarMetrics.content)
            .frame(height: SidebarMetrics.row)
            .contentShape(Rectangle())
        }
        .buttonStyle(SidebarRowStyle(selected: selected, hovered: hovered))
        .onHover { hovered = $0 }
        .help(detail ?? shortcut.map { "\(title) · \($0)" } ?? title)
        .accessibilityLabel(title)
        .accessibilityAddTraits(selected ? .isSelected : [])
    }
}

/// The list's title is also its project filter: "Conversations" (or the project's
/// name) with a menu, instead of a "No project" row that read like an error.
private struct SidebarScopeMenu: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var hovered = false

    private var project: WorkspaceProject? {
        studio.projects.first { $0.id == studio.selectedProjectID }
    }

    var body: some View {
        let title = project?.name ?? (studio.selectedMode == .agent ? L("Tâches", "Tasks") : L("Conversations", "Conversations"))
        HStack(spacing: 0) {
            Menu {
                Section(L("Projet", "Project")) {
                    Button { studio.chooseProject(nil) } label: {
                        if studio.selectedProjectID == nil {
                            Label(L("Sans projet", "No project"), systemImage: "checkmark")
                        } else { Text(L("Sans projet", "No project")) }
                    }
                    ForEach(studio.projects) { project in
                        Button { studio.chooseProject(project.id) } label: {
                            if studio.selectedProjectID == project.id {
                                Label(project.name, systemImage: "checkmark")
                            } else { Text(project.name) }
                        }
                    }
                }
                Divider()
                Button { SidebarDestination.projects.open(studio) } label: {
                    Label(L("Gérer les projets…", "Manage projects…"), systemImage: "folder.badge.gearshape")
                }
            } label: {
                HStack(spacing: 6) {
                    if project != nil {
                        Image(systemName: "folder.fill").font(.system(size: 11))
                    }
                    Text(title).lineLimit(1).truncationMode(.tail)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 8, weight: .bold))
                        .foregroundStyle(StudioTheme.quiet)
                }
                .font(.system(size: 12.5, weight: .semibold))
                .foregroundStyle(StudioTheme.secondary)
                .padding(.horizontal, SidebarMetrics.content)
                .frame(height: 26)
                .contentShape(Rectangle())
            }
            .menuStyle(.button)
            .buttonStyle(SidebarRowStyle(hovered: hovered))
            .menuIndicator(.hidden)
            .fixedSize(horizontal: true, vertical: false)
            .onHover { hovered = $0 }
            .accessibilityLabel(L("Projet", "Project"))
            .accessibilityValue(project?.name ?? L("Sans projet", "No project"))
            .help(L("Choisir le projet de la conversation", "Choose the conversation project"))
            Spacer(minLength: 0)
        }
    }
}

private struct SidebarSectionHeader: View {
    @Environment(\.studioStyle) private var style
    let title: String
    let first: Bool

    var body: some View {
        Group {
            if style.isGraphite {
                Text(title).font(.system(size: 11, weight: .semibold))
            } else {
                Text(title.uppercased()).font(.system(size: 9.5, weight: .semibold)).tracking(0.8)
            }
        }
        .foregroundStyle(StudioTheme.quiet)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.leading, SidebarMetrics.content)
        .padding(.top, first ? 4 : 14)
        .padding(.bottom, 4)
        .accessibilityAddTraits(.isHeader)
    }
}

/// A conversation title fades out at the edge instead of ending on "…"; short titles
/// never reach the fade. Its actions sit behind "…" on hover and in the context menu.
private struct SidebarConversationItem: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.studioStyle) private var style
    let conversation: Conversation
    let showsSelection: Bool
    @State private var hovered = false

    var body: some View {
        let selected = showsSelection && conversation.id == studio.selectedConversationID
        let empty = conversation.messages.isEmpty
        Button { studio.selectConversation(conversation.id) } label: {
            Text(empty && ["Nouvelle conversation", "New conversation"].contains(conversation.title)
                 ? L("Nouvelle conversation", "New conversation") : conversation.title)
                .font(.system(size: 13, weight: selected && !style.isGraphite ? .medium : .regular))
                // An untouched draft recedes so it does not read as one more chat.
                .foregroundStyle(selected ? StudioTheme.ink : empty ? StudioTheme.quiet : StudioTheme.secondary)
                .lineLimit(1)
                .fixedSize(horizontal: true, vertical: false)
                // minWidth 0: the frame takes the row's width even when the title is longer.
                .frame(minWidth: 0, maxWidth: .infinity, alignment: .leading)
                .mask {
                    HStack(spacing: 0) {
                        Color.black
                        LinearGradient(colors: [.black, .clear], startPoint: .leading, endPoint: .trailing)
                            .frame(width: 22)
                    }
                }
                .padding(.leading, SidebarMetrics.content)
                .padding(.trailing, hovered ? 30 : 6)
                .frame(height: SidebarMetrics.row)
                .contentShape(Rectangle())
        }
        .buttonStyle(SidebarRowStyle(selected: selected, hovered: hovered))
        .overlay(alignment: .trailing) {
            if hovered { actionsMenu.padding(.trailing, 3) }
        }
        .overlay(alignment: .leading) {
            // Aurora keeps its accent mark; Graphite selects by luminance alone.
            if selected && !style.isGraphite {
                Capsule().fill(studio.accentColor)
                    .frame(width: 3, height: 14)
                    .offset(x: 1)
                    .allowsHitTesting(false)
                    .accessibilityHidden(true)
            }
        }
        .onHover { hovered = $0 }
        .help(conversation.title)
        .accessibilityLabel(conversation.title)
        .accessibilityAddTraits(selected ? .isSelected : [])
        .contextMenu { actions }
    }

    @ViewBuilder private var actions: some View {
        Button(conversation.isPinned ? L("Désépingler", "Unpin") : L("Épingler", "Pin"),
               systemImage: conversation.isPinned ? "pin.slash" : "pin") {
            studio.setConversationPinned(conversation.id, !conversation.isPinned)
        }
        Divider()
        Button(L("Supprimer", "Delete"), systemImage: "trash", role: .destructive) {
            studio.deleteConversation(conversation.id)
        }
    }

    private var actionsMenu: some View {
        Menu { actions } label: {
            Image(systemName: "ellipsis")
                .font(.system(size: 12, weight: .semibold))
                .frame(width: 24, height: 22)
                .contentShape(Rectangle())
        }
        .menuStyle(.button)
        .buttonStyle(StudioControlStyle(cornerRadius: 6))
        .menuIndicator(.hidden)
        .fixedSize()
        .help(L("Actions", "Actions"))
        .accessibilityLabel(L("Actions pour ", "Actions for ") + conversation.title)
    }
}

/// The pages the sidebar footer and project menu open. Shared with the checks so
/// a destination cannot be wired in one place and forgotten in the other.
enum SidebarDestination: CaseIterable {
    case models, lab, projects, mcp, settings

    var title: String {
        switch self {
        case .models: L("Modèles", "Models")
        case .lab: "Lab"
        case .projects: L("Projets et profils", "Projects and profiles")
        case .mcp: L("Serveurs MCP", "MCP servers")
        case .settings: L("Réglages…", "Settings…")
        }
    }

    var icon: String {
        switch self {
        case .models: "square.stack"
        case .lab: StudioMode.lab.icon
        case .projects: "folder"
        case .mcp: "shippingbox"
        case .settings: "gearshape"
        }
    }

    @MainActor func open(_ studio: StudioModel) {
        switch self {
        case .models: studio.openModelManager()
        case .lab: studio.selectMode(.lab)
        case .projects: studio.showWorkspace = true
        case .mcp: studio.showMCPManager = true
        case .settings: studio.openAppSettings()
        }
    }
}

/// Models as a labelled row (the page people open most), Lab, projects and MCP
/// behind "…", and the gear for Settings: no unlabelled icon row, no account-like brand row.
private struct SidebarFooter: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var hovered = false

    var body: some View {
        HStack(spacing: 2) {
            Button { SidebarDestination.models.open(studio) } label: {
                HStack(spacing: 9) {
                    Image(systemName: SidebarDestination.models.icon)
                        .font(.system(size: 13))
                        .frame(width: 18)
                        .foregroundStyle(StudioTheme.quiet)
                    Text(SidebarDestination.models.title)
                        .font(.system(size: 13))
                        .foregroundStyle(StudioTheme.secondary)
                    Spacer(minLength: 0)
                }
                .padding(.leading, 9)
                .padding(.trailing, SidebarMetrics.content)
                .frame(height: 30)
                .contentShape(Rectangle())
            }
            .buttonStyle(SidebarRowStyle(hovered: hovered))
            .onHover { hovered = $0 }
            .help(L("Gérer et télécharger des modèles", "Manage and download models"))

            SidebarUpdateBadge(updater: studio.updateManager, action: studio.openUpdateSettings)

            Menu {
                Button { SidebarDestination.lab.open(studio) } label: {
                    if studio.selectedMode == .lab {
                        Label(SidebarDestination.lab.title, systemImage: "checkmark")
                    } else {
                        Label(SidebarDestination.lab.title, systemImage: SidebarDestination.lab.icon)
                    }
                }
                .disabled(studio.isGenerating)
                Divider()
                Button { SidebarDestination.projects.open(studio) } label: {
                    Label(SidebarDestination.projects.title, systemImage: SidebarDestination.projects.icon)
                }
                Button { SidebarDestination.mcp.open(studio) } label: {
                    Label(SidebarDestination.mcp.title, systemImage: SidebarDestination.mcp.icon)
                }
            } label: {
                Image(systemName: "ellipsis")
                    .font(.system(size: 13, weight: .medium))
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .menuStyle(.button)
            .buttonStyle(StudioControlStyle(cornerRadius: 8))
            .menuIndicator(.hidden)
            .fixedSize()
            .help(L("Lab, projets et serveurs MCP", "Lab, projects and MCP servers"))
            .accessibilityLabel(L("Plus", "More"))

            Button { SidebarDestination.settings.open(studio) } label: {
                Image(systemName: "gearshape")
                    .font(.system(size: 14))
                    .frame(width: 30, height: 30)
                    .contentShape(Rectangle())
            }
            .buttonStyle(StudioControlStyle(cornerRadius: 8))
            .help(L("Réglages · ⌘,", "Settings · ⌘,"))
            .accessibilityLabel(L("Réglages", "Settings"))
        }
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
            } else if let conversation = studio.currentConversation, !conversation.messages.isEmpty {
                MessagesView(messages: conversation.messages,
                    agentWorkspace: studio.selectedMode == .agent && conversation.agent ? studio.agentWorkspaceURL : nil,
                    savingAgentTurn: studio.agentStudioBusy)
                ComposerView()
                    .matchedGeometryEffect(id: "composer", in: composerMotion, properties: .position)
                    .padding(.horizontal, 32)
                    .padding(.top, 10)
                    .padding(.bottom, studio.selectedMode == .agent ? 8 : 22)
                if studio.selectedMode == .agent {
                    AgentTaskBar().padding(.horizontal, 36).padding(.bottom, 16)
                }
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
    @Environment(\.studioStyle) private var style
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
                .font(.system(size: style.isGraphite ? 13 : 12, weight: .medium))
                .foregroundStyle(style.isGraphite ? StudioTheme.ink : StudioTheme.secondary)
                .lineLimit(1)
                .layoutPriority(-1)

            Spacer(minLength: 12)

            if studio.selectedMode != .decision {
                ModelSelectionMenu()
            }

            HStack(spacing: 3) {
                if studio.selectedMode != .decision {
                    ConversationDisplayControl()
                    if studio.mtpToolbarVisible {
                        MTPToolbarControl()
                    }
                }
                if studio.dsparkToolbarVisible && studio.selectedMode != .decision {
                    DSparkToolbarControl()
                }
                if studio.dflash2Available && studio.selectedMode != .decision {
                    DFlashToolbarControl()
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
            .padding(style.isGraphite ? 0 : 4)
            // Graphite: the controls sit on the canvas; glass with nothing under it reads as a pill.
            .background { if !style.isGraphite { Color.clear.premiumGlass(radius: 100, materialOpacity: 0.76) } }
        }
        .font(.system(size: 13))
        .padding(.leading, sidebarVisible ? 22 : 82)
        .padding(.trailing, 18)
        .padding(.vertical, 14)
    }
}

/// Agent controls under the composer: the task's studio and its runnable HTML files. The
/// folder and the Internet switch sit in the composer; everything else lives in Agent Studio.
private struct AgentTaskBar: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var htmlFiles: [URL] = []
    @State private var previewError: String?

    var body: some View {
        HStack(spacing: 8) {
            Button {
                if let workspace = studio.agentWorkspaceURL { studio.studioDocument = StudioDocument(workspace: workspace) }
            } label: {
                Label("Agent Studio", systemImage: "sidebar.right").padding(.horizontal, 8).frame(height: 28)
            }
            .buttonStyle(StudioControlStyle())
            .disabled(studio.isGenerating || studio.agentStudioBusy)
            .help(L("Fichiers, changements à valider, versions et permissions de la tâche", "The task’s files, changes to review, versions and permissions"))
            if !studio.isGenerating, let first = htmlFiles.first {
                if htmlFiles.count == 1 {
                    Button { preview(first) } label: {
                        Label(first.lastPathComponent, systemImage: "play.rectangle").lineLimit(1)
                            .padding(.horizontal, 8).frame(height: 28)
                    }
                    .buttonStyle(StudioControlStyle())
                    .help(L("Exécuter ce fichier HTML dans l’app", "Run this HTML file in the app"))
                } else {
                    Menu {
                        ForEach(htmlFiles, id: \.self) { file in Button(file.lastPathComponent) { preview(file) } }
                    } label: { Label(L("Aperçu", "Preview"), systemImage: "play.rectangle") }
                    .menuStyle(.borderlessButton).fixedSize()
                    .help(L("Exécuter un fichier HTML de la tâche dans l’app", "Run one of the task’s HTML files in the app"))
                }
            }
            Spacer(minLength: 8)
        }
        .font(.system(size: 12))
        .foregroundStyle(StudioTheme.secondary)
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

    private func preview(_ file: URL) {
        guard let workspace = studio.agentWorkspaceURL else { return }
        do {
            _ = try HTMLPreview.source(at: file, in: workspace)
            studio.studioDocument = StudioDocument(workspace: workspace, file: file)
        } catch { previewError = error.localizedDescription }
    }
}

private struct WelcomeView: View {
    let composerMotion: Namespace.ID
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.studioStyle) private var style

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
        .background {
            // Graphite's home texture (it keeps the top half clear itself); the other
            // styles draw nothing here.
            StudioAurora(animated: true, fullHeight: true, placement: .home)
                .ignoresSafeArea()
        }
    }

    private func content(agent: Bool) -> some View {
        VStack(spacing: 0) {
            // Graphite: the greeting alone; the mark already sits in the sidebar.
            if !style.isGraphite {
                MonogramMark(size: 76)
                    .frame(height: 49)
            }
            Text(agent ? L("Nouvelle tâche", "New task") : greeting)
                .font(style.isGraphite ? .system(size: 30, weight: .semibold) : .system(size: 44, weight: .regular, design: .serif))
                .tracking(style.isGraphite ? -0.7 : -1.5)
                .multilineTextAlignment(.center)
                .foregroundStyle(StudioTheme.ink)
                .padding(.top, style.isGraphite ? 0 : 15)
                .padding(.bottom, style.isGraphite ? 26 : 32)
                .accessibilityAddTraits(.isHeader)

            ComposerView()
                .matchedGeometryEffect(id: "composer", in: composerMotion, properties: .position)
            if agent {
                if !studio.mcpErrors.isEmpty {
                    Text(studio.mcpErrors.values.sorted().joined(separator: "\n"))
                        .font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
                AgentTaskBar().padding(.horizontal, 4).padding(.top, 8)
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
                            .font(style.isGraphite ? .system(size: 11.5).monospacedDigit() : .system(size: 11, design: .monospaced))
                            .foregroundStyle(style.isGraphite ? StudioTheme.quiet : StudioTheme.secondary)
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
