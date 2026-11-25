import SwiftUI

enum SettingsPage: String, CaseIterable, Identifiable {
    case general, appearance, models, conversations, connections, privacy, updates
    var id: Self { self }

    var title: String {
        switch self {
        case .general: L("Général", "General")
        case .appearance: L("Apparence", "Appearance")
        case .models: L("Modèles & mémoire", "Models & memory")
        case .conversations: L("Conversations", "Conversations")
        case .connections: L("Connexions", "Connections")
        case .privacy: L("Confidentialité", "Privacy")
        case .updates: L("Mises à jour", "Updates")
        }
    }

    var icon: String {
        switch self {
        case .general: "slider.horizontal.3"
        case .appearance: "paintpalette"
        case .models: "cpu"
        case .conversations: "bubble.left.and.bubble.right"
        case .connections: "link"
        case .privacy: "lock.shield"
        case .updates: "arrow.triangle.2.circlepath"
        }
    }

    var detail: String {
        switch self {
        case .general: L("Langue et notifications.", "Language and notifications.")
        case .appearance: L("Thème, couleur et taille du texte.", "Theme, color and text size.")
        case .models:
            L("Démarrage automatique et gestion de la mémoire.", "Automatic startup and memory management.")
        case .conversations:
            L("Importer et exporter votre historique.", "Import and export your history.")
        case .connections:
            L("Comptes et outils connectés.", "Connected accounts and tools.")
        case .privacy: L("Où vos données sont traitées et envoyées.", "Where your data is processed and sent.")
        case .updates: L("Versions et canaux de mise à jour.", "Versions and update channels.")
        }
    }
}

struct AppSettingsView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    private var selection: SettingsPage { studio.appSettingsPage }
    @AppStorage(QwenRuntimeProfile.preferenceKey) private var requestedQwenProfile = "reference"

    var body: some View {
        HStack(spacing: 0) {
            StudioPageSidebar(title: L("Réglages", "Settings"), onBack: { studio.showAppSettings = false }) {
                StudioNavList {
                    ForEach(SettingsPage.allCases) { page in
                        StudioNavRow(icon: page.icon, title: page.title, selected: selection == page,
                                     badge: page == .updates && studio.updateManager.hasAvailableUpdate ? "•" : nil) {
                            studio.appSettingsPage = page
                        }
                        .accessibilityIdentifier("settings.\(page.rawValue)")
                    }
                }
            }

            StudioPageColumn(spacing: 24) {
                StudioPageHeader(title: selection.title, detail: selection.detail)
                pageContent
            }
            .id(selection)
            .transition(.opacity)
        }
        .font(.system(size: 13))
        .foregroundStyle(StudioTheme.ink)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background { StudioAurora(animated: true).background(StudioTheme.canvas) }
        .studioAppearance(studio)
        .environment(\.locale, Locale(identifier: studio.language.rawValue))
        .onExitCommand { studio.showAppSettings = false }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: selection)
    }

    private var startupModelMissing: Bool {
        !studio.startupModelPath.isEmpty && !studio.models.contains(where: { $0.path == studio.startupModelPath })
    }

    @ViewBuilder private var pageContent: some View {
        VStack(alignment: .leading, spacing: 22) {
            switch selection {
            case .general:
                SettingSection(title: L("Langue", "Language"), icon: "globe") {
                    SettingRow(title: L("Langue de l’app", "App language"),
                               detail: L("S’applique immédiatement à toute l’interface.", "Applies immediately to the whole interface.")) {
                        Picker(
                            L("Langue de l’app", "App language"),
                            selection: Binding(
                                get: { studio.language }, set: { studio.setLanguage($0) }
                            )
                        ) {
                            ForEach(AppLanguage.allCases, id: \.self) { language in
                                Text(language.title).tag(language)
                            }
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .fixedSize()
                    }
                }
                SettingSection(title: L("Notifications", "Notifications"), icon: "bell") {
                    SettingToggleRow(
                        title: L("Notification de fin de génération", "Generation completion notification"),
                        detail: L(
                            "Préviens-moi lorsque la réponse est terminée et que MLXL3 est en arrière-plan. Le contenu de la conversation n’apparaît pas dans la notification.",
                            "Notify me when the response is complete and MLXL3 is in the background. Conversation content is not included in the notification."
                        ),
                        isOn: Binding(
                            get: { studio.completionNotificationsEnabled },
                            set: { studio.setCompletionNotificationsEnabled($0) }
                        ))
                    if let status = studio.notificationStatus {
                        Text(status).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                if EngineRuntimeStore.nativeEngineEnabled {
                    SettingSection(title: L("Moteur", "Engine"), icon: "cpu") {
                        SettingRow(title: L("Profil Qwen natif", "Native Qwen profile"),
                                   detail: L("Rechargez le modèle pour appliquer le profil. Le profil optimisé est expérimental ; les variables du moteur restent prioritaires.", "Reload the model to apply the profile. The optimized profile is experimental; engine environment variables take precedence.")) {
                            Picker(L("Profil demandé", "Requested profile"), selection: Binding(
                                get: { QwenRuntimeProfile.requested(requestedQwenProfile).rawValue },
                                set: { requestedQwenProfile = $0 }
                            )) {
                                Text(L("Référence", "Reference")).tag("reference")
                                Text(L("Optimisé (expérimental)", "Optimized (experimental)")).tag("optimized")
                            }
                            .labelsHidden().fixedSize()
                        }
                        DisclosureGroup(L("Détails du moteur", "Engine details")) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(L("Profil actif : ", "Active profile: ") + (studio.runtimeQwenProfile ?? "—"))
                                if let identity = studio.runtimeIdentity {
                                    Text(identity).textSelection(.enabled)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading).padding(.top, 6)
                        }
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                    }
                }
            case .appearance:
                AppearanceSettings()
            case .models:
                SettingSection(title: L("Au démarrage", "At startup"), icon: "power") {
                    SettingToggleRow(
                        title: L("Charger le modèle au démarrage", "Load model on startup"),
                        detail: L("Le modèle est prêt dès l’ouverture de l’app.", "The model is ready as soon as the app opens."),
                        isOn: Binding(
                            get: { studio.loadModelOnStartup },
                            set: { studio.setLoadModelOnStartup($0) }
                        ))
                    SettingRow(title: L("Modèle de démarrage", "Startup model"),
                               detail: startupModelMissing
                                   ? L(
                                       "Le modèle choisi n’est plus disponible. Choisis un autre modèle pour le prochain démarrage.",
                                       "The selected model is no longer available. Choose another model for the next launch.")
                                   : nil) {
                        Picker(
                            L("Modèle de démarrage", "Startup model"),
                            selection: Binding(
                                get: { studio.startupModelPath }, set: { studio.setStartupModelPath($0) }
                            )
                        ) {
                            Text(L("Dernier modèle sélectionné", "Last selected model")).tag("")
                            ForEach(studio.chatModels) { model in
                                Text(model.name).tag(model.path)
                            }
                            if startupModelMissing {
                                Text(L("Modèle indisponible", "Model unavailable")).tag(studio.startupModelPath)
                            }
                        }
                        .labelsHidden().frame(maxWidth: 260, alignment: .trailing)
                        .disabled(!studio.loadModelOnStartup)
                    }
                }
                SettingSection(title: L("Mémoire", "Memory"), icon: "memorychip") {
                    SettingRow(title: L("Déchargement automatique du modèle", "Automatic model unloading"),
                               detail: L("Le modèle est déchargé après ce délai d’inactivité. La génération et le tuning ne sont pas interrompus.", "The model unloads after this idle delay. Generation and tuning are not interrupted.")) {
                        Picker(
                            L("Après une période d’inactivité", "After inactivity"),
                            selection: Binding(
                                get: { studio.modelIdleUnloadDelay },
                                set: { studio.setModelIdleUnloadDelay($0) }
                            )
                        ) {
                            ForEach(ModelIdleUnloadDelay.allCases, id: \.self) { delay in
                                Text(delay.title).tag(delay)
                            }
                        }
                        .pickerStyle(.menu).labelsHidden().fixedSize()
                        .accessibilityLabel(L("Délai de déchargement automatique", "Automatic unload delay"))
                    }
                    SettingRow(title: L("Économie de mémoire", "Memory saving"),
                               detail: (studio.engineState.isReady && !studio.memorySaverSupported
                                   ? L("Mets à jour le moteur pour utiliser cette option.", "Update the engine to use this option.") + " "
                                   : "")
                                   + L(
                                       "Libère le cache de contexte après chaque réponse sans décharger le modèle. Le prochain message peut démarrer plus lentement.",
                                       "Frees the context cache after each response while keeping the model loaded. The next message may start more slowly."
                                   )) {
                        Toggle(
                            L("Économie de mémoire", "Memory saving"),
                            isOn: Binding(
                                get: { studio.memorySaverEnabled },
                                set: { studio.setMemorySaverEnabled($0) }
                            )
                        )
                        .labelsHidden().toggleStyle(.switch).controlSize(.small)
                        .disabled(
                            studio.isGenerating || studio.isTuningMTP
                                || (studio.engineState.isReady && !studio.memorySaverSupported))
                    }
                }
            case .conversations:
                SettingSection(
                    title: L("Historique", "History"), icon: "bubble.left.and.bubble.right",
                    footer: L("Les conversations restent sur ce Mac. Une sauvegarde est un fichier que vous gardez où vous voulez.",
                              "Conversations stay on this Mac. A backup is a file you keep wherever you like.")
                ) {
                    SettingRow(title: L("Exporter les conversations", "Export conversations"),
                               detail: L("Enregistre tout l’historique dans un fichier.", "Saves the whole history to a file.")) {
                        Button(action: studio.exportConversations) {
                            Label(L("Exporter…", "Export…"), systemImage: "square.and.arrow.up")
                        }
                        .buttonStyle(.studio)
                        .accessibilityLabel(L("Exporter les conversations…", "Export conversations…"))
                    }
                    SettingRow(title: L("Importer une sauvegarde", "Import a backup"),
                               detail: L("Ajoute les conversations d’un fichier exporté.", "Adds the conversations of an exported file.")) {
                        Button(action: studio.importConversations) {
                            Label(L("Importer…", "Import…"), systemImage: "square.and.arrow.down")
                        }
                        .buttonStyle(.studio)
                        .accessibilityLabel(L("Importer une sauvegarde…", "Import a backup…"))
                    }
                }
            case .connections:
                SettingCard(title: "Hugging Face", icon: "key") {
                    HuggingFaceAccountView(library: studio.modelLibrary)
                }
                SettingCard(title: L("Outils MCP", "MCP tools"), icon: "shippingbox") {
                    MCPSettingsControls()
                }
            case .privacy:
                SettingSection(title: L("Sur ce Mac", "On this Mac"), icon: "desktopcomputer") {
                    SettingRow(title: L("Inférence et kernels Metal", "Inference and Metal kernels"),
                               detail: L("Exécutés localement sur votre Mac.", "Run locally on your Mac."), icon: "cpu") {
                        StudioStatusPill(text: L("Local", "Local"), color: StudioTheme.success)
                    }
                    SettingRow(title: L("Conversations et documents", "Conversations and documents"),
                               detail: L("Enregistrés dans le dossier de l’app, jamais envoyés.", "Stored in the app’s folder, never sent."),
                               icon: "internaldrive") {
                        StudioStatusPill(text: L("Local", "Local"), color: StudioTheme.success)
                    }
                }
                SettingSection(
                    title: L("Accès au réseau", "Network access"), icon: "network",
                    footer: L(
                        "Un serveur MCP local peut lui-même utiliser Internet. Activez uniquement des outils de confiance.",
                        "A local MCP server can itself use the Internet. Enable only tools you trust.")
                ) {
                    SettingRow(title: L("Modèles et mises à jour", "Models and updates"),
                               detail: "Hugging Face · GitHub", icon: "arrow.down.circle") {
                        StudioStatusPill(text: L("À la demande", "On demand"))
                    }
                    SettingRow(title: "MCP",
                               detail: studio.mcpEnabled
                                   ? L("Activé · des données peuvent quitter ce Mac.", "Enabled · data may leave this Mac.")
                                   : L("Désactivé", "Disabled"), icon: "link") {
                        StudioStatusPill(text: studio.mcpEnabled ? L("Activé", "On") : L("Désactivé", "Off"),
                                         color: studio.mcpEnabled ? StudioTheme.warning : StudioTheme.quiet)
                    }
                }
            case .updates:
                UpdateSettingsCard(
                    updater: studio.updateManager, canInstall: !studio.isGenerating,
                    install: studio.installUpdateAndRestart)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }
}

private struct HuggingFaceAccountView: View {
    @ObservedObject var library: ModelLibrary
    @State private var token = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(
                L(
                    "Accédez aux modèles privés ou à accès restreint avec votre token Hugging Face.",
                    "Access private or gated models with your Hugging Face token.")
            )
            .foregroundStyle(StudioTheme.secondary)
            HStack(spacing: 8) {
                SecureField(L("Token Hugging Face · hf_…", "Hugging Face token · hf_…"), text: $token)
                    .studioField(monospaced: true)
                    .accessibilityLabel(L("Token Hugging Face", "Hugging Face token"))
                    .onSubmit { signIn() }
                Button(L("Se connecter", "Sign in"), action: signIn)
                    .buttonStyle(.studio(.primary)).disabled(!validToken)
                Button(L("Se déconnecter", "Sign out")) {
                    library.authenticate(token: nil)
                    token = ""
                }
                .buttonStyle(.studio)
            }
            if let message = library.authMessage {
                StudioNotice(text: message)
            }
            Text(
                L(
                    "Le token est enregistré localement par Hugging Face. Pour un modèle à accès restreint, acceptez aussi sa licence sur son site.",
                    "Hugging Face stores your token locally. For a gated model, also accept its license on the website."
                )
            )
            .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var validToken: Bool { token.hasPrefix("hf_") && token.count <= 4000 }
    private func signIn() {
        guard validToken else { return }
        library.authenticate(token: token)
        token = ""
    }
}

struct MCPSettingsControls: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            SettingRow(
                title: L("Activer les outils MCP", "Enable MCP tools"),
                detail: studio.mcpUpdating
                    ? L("Connexion aux serveurs…", "Connecting to servers…")
                    : studio.mcpEnabled
                        ? L(
                            "\(studio.mcpServerCount) serveur(s) connecté(s) · \(studio.mcpToolCount) outil(s)",
                            "\(studio.mcpServerCount) connected server(s) · \(studio.mcpToolCount) tool(s)")
                        : L("Les outils externes sont désactivés.", "External tools are disabled.")
            ) {
                if studio.mcpUpdating { ProgressView().controlSize(.small) }
                Toggle(
                    L("Activer les outils MCP", "Enable MCP tools"),
                    isOn: Binding(
                        get: { studio.mcpEnabled }, set: { studio.setMCPEnabled($0) }
                    )
                )
                .labelsHidden().toggleStyle(.switch).controlSize(.small)
                .disabled(studio.isGenerating || studio.mcpUpdating)
            }
            ForEach(studio.mcpErrors.keys.sorted(), id: \.self) { key in
                StudioNotice(kind: .warning, text: "\(key) · \(studio.mcpErrors[key] ?? "")")
            }
            HStack(spacing: 8) {
                Button(action: { studio.showMCPManager = true }) {
                    Label(L("Configurer…", "Configure…"), systemImage: "slider.horizontal.3")
                }
                .buttonStyle(.studio)
                Button(action: studio.reloadMCPServers) {
                    Label(L("Recharger", "Reload"), systemImage: "arrow.clockwise")
                }
                .buttonStyle(.studio)
                .disabled(studio.isGenerating || studio.mcpUpdating || !studio.mcpEnabled)
                .help(L("Recharger les serveurs MCP", "Reload MCP servers"))
                .accessibilityLabel(L("Recharger les serveurs MCP", "Reload MCP servers"))
            }
            Text(
                L(
                    "Les serveurs MCP peuvent recevoir vos requêtes et utiliser Internet, même s’ils sont lancés localement.",
                    "MCP servers may receive your requests and access the Internet, even when launched locally."
                )
            )
            .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
            .fixedSize(horizontal: false, vertical: true)
        }
    }
}

private struct UpdateSettingsCard: View {
    @ObservedObject var updater: UpdateManager
    let canInstall: Bool
    let install: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 22) {
            // The state first: what is installed, whether anything is waiting, the one action.
            VStack(alignment: .leading, spacing: 14) {
                HStack(alignment: .center, spacing: 13) {
                    stateIcon
                        .font(.system(size: 17, weight: .medium))
                        .frame(width: 40, height: 40)
                        .background(StudioTheme.contrast.opacity(0.06), in: RoundedRectangle(cornerRadius: 11, style: .continuous))
                    VStack(alignment: .leading, spacing: 3) {
                        Text(statusTitle).font(.system(size: 14, weight: .semibold))
                        Text(statusDetail).font(.system(size: 12.5)).foregroundStyle(StudioTheme.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .textSelection(.enabled)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    updateAction
                }
                if case .downloading = updater.state, let progress = updater.downloadProgress {
                    UpdateTransferView(progress: progress)
                }
                if updater.hasReadyUpdate && !canInstall {
                    StudioNotice(kind: .warning, text: L(
                        "Termine ou arrête la génération avant de redémarrer pour installer.",
                        "Finish or stop generation before restarting to install."))
                }
            }
            .studioCard(padding: 16, radius: 13)

            if let release = updater.latestRelease,
                updater.hasAppUpdate,
                !release.notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                VStack(alignment: .leading, spacing: 9) {
                    HStack {
                        StudioSectionTitle(title: L("Nouveautés de \(release.tag)", "What’s new in \(release.tag)"), icon: "sparkles")
                        Spacer()
                        Button(L("Voir la release", "View release")) { updater.openReleasePage() }
                            .buttonStyle(.studio(.quiet, height: 24))
                    }
                    MarkdownResponseView(release.notes, textScale: 13.0 / 15.0, htmlPreviewEnabled: false)
                        .equatable()
                        .accessibilityIdentifier("updates.release-notes")
                        .studioCard(padding: 16)
                }
            }

            SettingSection(
                title: L("Versions", "Versions"), icon: "shippingbox",
                footer: L(
                    "L’app et le moteur peuvent être mis à jour séparément. Les téléchargements sont vérifiés avant installation.",
                    "The app and engine can update separately. Downloads are verified before installation.")
            ) {
                SettingRow(title: "MLXL3 Desktop", detail: L("L’interface.", "The interface.")) {
                    Text(updater.currentVersion).font(.system(size: 12.5, design: .monospaced)).foregroundStyle(StudioTheme.secondary)
                        .textSelection(.enabled)
                }
                VStack(alignment: .leading, spacing: 8) {
                    SettingRow(title: L("Moteur", "Engine"), detail: engineDetail) {
                        Text(updater.currentEngineVersion).font(.system(size: 12.5, design: .monospaced))
                            .foregroundStyle(StudioTheme.secondary).textSelection(.enabled)
                    }
                    if case .downloading = updater.engineState, let progress = updater.engineDownloadProgress {
                        UpdateTransferView(progress: progress)
                    }
                }
                SettingRow(title: L("Canal des mises à jour", "Update channel"),
                           detail: L(
                               "Utilise Rechercher pour vérifier les versions du canal choisi.",
                               "Use Search to check versions from the selected channel.")) {
                    Picker(
                        L("Canal des mises à jour", "Update channel"),
                        selection: Binding(
                            get: { updater.updateChannel }, set: { updater.setUpdateChannel($0) }
                        )
                    ) {
                        Text("Stable").tag(UpdateChannel.stable)
                        Text(L("Bêta", "Beta")).tag(UpdateChannel.beta)
                    }
                    .pickerStyle(.segmented).labelsHidden().fixedSize()
                    .disabled(updater.isInstalling)
                }
            }
        }
    }

    @ViewBuilder
    private var updateAction: some View {
        if updater.hasReadyUpdate && !updater.isBusy {
            Button(action: install) {
                Label(
                    updater.state.readyRelease != nil
                        ? L("Redémarrer et installer", "Restart and install")
                        : L("Installer le moteur", "Install engine"), systemImage: "arrow.clockwise"
                )
            }
            .buttonStyle(.studio(.primary))
            .disabled(!canInstall)
        } else if updater.isBusy {
            ProgressView()
                .controlSize(.small)
                .frame(width: 30, height: 30)
        } else {
            Button(L("Rechercher", "Search"), action: updater.checkForUpdates)
                .buttonStyle(.studio)
        }
    }

    @ViewBuilder
    private var stateIcon: some View {
        switch updater.state {
        case .checking, .downloading, .installing:
            ProgressView().controlSize(.small)
        case .ready:
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(StudioTheme.ink)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(StudioTheme.warning)
        case .upToDate:
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(StudioTheme.success)
        case .idle:
            Image(systemName: "arrow.triangle.2.circlepath")
                .foregroundStyle(StudioTheme.secondary)
        }
    }

    private var engineDetail: String {
        switch updater.engineState {
        case .idle: L("Le moteur se met à jour séparément de l’app.", "The engine updates separately from the app.")
        case .checking: L("Recherche d’un nouveau moteur…", "Checking for a new engine…")
        case .upToDate:
            L(
                "Moteur à jour. L’app et tes modèles sont conservés.",
                "Engine up to date. Your app and models are preserved.")
        case .downloading(let release):
            L("Téléchargement du moteur \(release.version)…", "Downloading engine \(release.version)…")
        case .ready(let release, _):
            L(
                "Moteur \(release.version) vérifié et prêt à installer.",
                "Engine \(release.version) verified and ready to install.")
        case .installing: L("Validation et activation du moteur…", "Validating and activating engine…")
        case .failed(let message): message
        }
    }

    private var statusTitle: String {
        switch updater.state {
        case .idle: L("Prêt à vérifier", "Ready to check")
        case .checking: L("Recherche sur GitHub…", "Checking GitHub…")
        case .upToDate(let date):
            L(
                "MLXL3 est à jour · \(date.formatted(date: .omitted, time: .shortened))",
                "MLXL3 is up to date · \(date.formatted(date: .omitted, time: .shortened))")
        case .downloading(let release): L("Téléchargement de \(release.tag)…", "Downloading \(release.tag)…")
        case .ready(let release, _): L("\(release.tag) est prête", "\(release.tag) is ready")
        case .installing(let release): L("Préparation de \(release.tag)…", "Preparing \(release.tag)…")
        case .failed: L("Mise à jour impossible", "Update failed")
        }
    }

    private var statusDetail: String {
        switch updater.state {
        case .idle:
            L(
                "Recherche les nouvelles versions du moteur et de l’interface.",
                "Check for new engine and interface versions.")
        case .checking:
            L(
                "Lecture de la dernière release publique de 0xZKnw/mlxl3.",
                "Reading the latest public release of 0xZKnw/mlxl3.")
        case .upToDate:
            L(
                "Le moteur et l’interface utilisent la dernière version publiée.",
                "The engine and interface are up to date.")
        case .downloading(let release):
            L(
                "Le DMG de \(ByteCountFormatter.string(fromByteCount: release.asset.size, countStyle: .file)) est téléchargé en arrière-plan.",
                "The \(ByteCountFormatter.string(fromByteCount: release.asset.size, countStyle: .file)) DMG is downloading in the background."
            )
        case .ready:
            L("Le téléchargement et l’empreinte SHA-256 sont validés.", "Download and SHA-256 checksum verified.")
        case .installing:
            L(
                "MLXL3 va se fermer, remplacer l’app puis se relancer automatiquement.",
                "MLXL3 will close, replace the app, then restart automatically.")
        case .failed(let message):
            message
        }
    }
}

struct UpdateTransferView: View {
    let progress: UpdateDownloadProgress

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            ProgressView(value: progress.transfer.fraction ?? 0)
                .tint(StudioTheme.accent)
                .accessibilityLabel(L("Téléchargement", "Download"))
            TimelineView(.periodic(from: .now, by: 1)) { _ in
                HStack {
                    if progress.isVerifying {
                        Text(L("Vérification SHA-256…", "Verifying SHA-256…"))
                    } else {
                        Text("\(Int((progress.transfer.fraction ?? 0) * 100)) %")
                        Text("\(bytes(progress.transfer.completed)) / \(bytes(progress.transfer.total))")
                        Spacer(minLength: 5)
                        let rate =
                            ProcessInfo.processInfo.systemUptime - progress.observedAt > 2
                            ? 0 : progress.transfer.bytesPerSecond
                        Text(rate.map { bytes($0) + "/s" } ?? L("Calcul du débit…", "Measuring speed…"))
                    }
                }
                .font(.system(size: 13, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(StudioTheme.secondary)
            }
        }
        .padding(.top, 5)
        .frame(maxWidth: .infinity)
    }

    private func bytes(_ value: Double) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(max(0, min(Double(Int64.max / 2), value))), countStyle: .file)
    }
}

struct UpdateStatusButton: View {
    @ObservedObject var updater: UpdateManager
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            ZStack(alignment: .topTrailing) {
                Image(systemName: "gearshape.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(width: 30, height: 30)
                if updater.hasAvailableUpdate {
                    Circle()
                        .fill(StudioTheme.accent)
                        .frame(width: 7, height: 7)
                        .overlay(Circle().stroke(StudioTheme.sidebar, lineWidth: 1.5))
                        .offset(x: 1, y: -1)
                }
            }
        }
        .buttonStyle(RoundGlassButtonStyle())
        .help(updater.hasAvailableUpdate ? L("Mise à jour prête", "Update ready") : L("Réglages", "Settings"))
    }
}
