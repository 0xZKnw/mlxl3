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
    @State private var selection: SettingsPage = .general
    @AppStorage(QwenRuntimeProfile.preferenceKey) private var requestedQwenProfile = "reference"

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 16) {
                MonogramMark(size: 44)
                    .padding(.horizontal, 10)
                VStack(spacing: 5) {
                    ForEach(SettingsPage.allCases) { page in
                        Button {
                            selection = page
                        } label: {
                            HStack(spacing: 12) {
                                Image(systemName: page.icon).frame(width: 20)
                                Text(page.title)
                                Spacer(minLength: 0)
                            }
                            .font(.system(size: 13, weight: selection == page ? .semibold : .regular))
                            .padding(.horizontal, 12).frame(height: 44)

                        }
                        .buttonStyle(StudioControlStyle(emphasized: selection == page))
                        .accessibilityIdentifier("settings.\(page.rawValue)")
                        .accessibilityAddTraits(selection == page ? .isSelected : [])
                    }
                }
                Spacer()
            }
            .padding(20).padding(.top, 16).padding(.bottom, 12)
            .frame(width: 222)
            .background(StudioTheme.sidebar(for: studio.theme))

            Rectangle().fill(StudioTheme.edge).frame(width: 1)

            VStack(spacing: 0) {
                StudioPanelHeader(
                    title: selection.title, detail: selection.detail,
                    close: { studio.showAppSettings = false })
                Rectangle().fill(StudioTheme.edge).frame(height: 1)
                ScrollView {
                    pageContent
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(28)
                }
                .id(selection)
                .transition(.opacity)
            }
        }
        .font(.system(size: 13))
        .foregroundStyle(StudioTheme.ink)
        .controlSize(.large)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(StudioTheme.canvas(for: studio.theme))
        .studioAppearance(studio.theme, accent: studio.accent)
        .environment(\.locale, Locale(identifier: studio.language.rawValue))
        .onExitCommand { studio.showAppSettings = false }
        .animation(reduceMotion ? nil : .easeInOut(duration: 0.2), value: selection)
    }

    @ViewBuilder private var pageContent: some View {
        VStack(alignment: .leading, spacing: 20) {
            switch selection {
            case .general:
                if EngineRuntimeStore.nativeEngineEnabled {
                    SettingCard(title: L("Profil Qwen natif", "Native Qwen profile"), icon: "cpu") {
                        Picker(L("Profil demandé", "Requested profile"), selection: Binding(
                            get: { QwenRuntimeProfile.requested(requestedQwenProfile).rawValue },
                            set: { requestedQwenProfile = $0 }
                        )) {
                            Text(L("Référence", "Reference")).tag("reference")
                            Text(L("Optimisé (expérimental)", "Optimized (experimental)")).tag("optimized")
                        }
                        .help(L("Rechargez le modèle pour appliquer le profil. Le profil optimisé est expérimental ; les variables du moteur restent prioritaires.", "Reload the model to apply the profile. The optimized profile is experimental; engine environment variables take precedence."))
                        DisclosureGroup(L("Détails du moteur", "Engine details")) {
                            Text(L("Profil actif : ", "Active profile: ") + (studio.runtimeQwenProfile ?? "—"))
                            if let identity = studio.runtimeIdentity {
                                Text(identity).textSelection(.enabled)
                            }
                        }
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                    }
                }
                SettingCard(title: L("Langue de l’app", "App language"), icon: "globe") {
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
                }
                Label(
                    L(
                        "Les changements sont enregistrés automatiquement.", "Changes are saved automatically."),
                    systemImage: "checkmark.circle"
                )
                .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)

                SettingCard(title: L("Notifications", "Notifications"), icon: "bell") {
                    VStack(alignment: .leading, spacing: 10) {
                        Toggle(
                            L("Notification de fin de génération", "Generation completion notification"),
                            isOn: Binding(
                                get: { studio.completionNotificationsEnabled },
                                set: { studio.setCompletionNotificationsEnabled($0) }
                            ))
                        Text(
                            L(
                                "Préviens-moi lorsque la réponse est terminée et que MLXL3 est en arrière-plan. Le contenu de la conversation n’apparaît pas dans la notification.",
                                "Notify me when the response is complete and MLXL3 is in the background. Conversation content is not included in the notification."
                            )
                        )
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                        if let status = studio.notificationStatus {
                            Text(status).font(.caption).foregroundStyle(StudioTheme.secondary)
                        }
                    }
                }
            case .appearance:
                AppearanceSettings()
            case .models:
                SettingCard(title: L("Au démarrage", "At startup"), icon: "power") {
                    VStack(alignment: .leading, spacing: 10) {
                        Toggle(
                            L("Charger le modèle au démarrage", "Load model on startup"),
                            isOn: Binding(
                                get: { studio.loadModelOnStartup },
                                set: { studio.setLoadModelOnStartup($0) }
                            ))
                        Picker(
                            L("Modèle de démarrage", "Startup model"),
                            selection: Binding(
                                get: { studio.startupModelPath }, set: { studio.setStartupModelPath($0) }
                            )
                        ) {
                            Text(L("Dernier modèle sélectionné", "Last selected model")).tag("")
                            ForEach(studio.models) { model in
                                Text(model.name).tag(model.path)
                            }
                            if !studio.startupModelPath.isEmpty
                                && !studio.models.contains(where: { $0.path == studio.startupModelPath })
                            {
                                Text(L("Modèle indisponible", "Model unavailable")).tag(studio.startupModelPath)
                            }
                        }
                        .disabled(!studio.loadModelOnStartup)
                        if !studio.startupModelPath.isEmpty
                            && !studio.models.contains(where: { $0.path == studio.startupModelPath })
                        {
                            Text(
                                L(
                                    "Le modèle choisi n’est plus disponible. Choisis un autre modèle pour le prochain démarrage.",
                                    "The selected model is no longer available. Choose another model for the next launch."
                                )
                            )
                            .font(.caption).foregroundStyle(StudioTheme.secondary)
                        }
                    }

                }
                SettingCard(title: L("Mémoire", "Memory"), icon: "memorychip") {
                    VStack(alignment: .leading, spacing: 10) {
                        Label(
                            L("Déchargement automatique du modèle", "Automatic model unloading"),
                            systemImage: "moon.zzz"
                        )
                        .font(.system(size: 13, weight: .semibold))
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
                        .pickerStyle(.menu)
                        .accessibilityLabel(L("Délai de déchargement automatique", "Automatic unload delay"))
                        .help(L("Le modèle est déchargé après ce délai d’inactivité. La génération et le tuning ne sont pas interrompus.", "The model unloads after this idle delay. Generation and tuning are not interrupted."))
                    }

                    Divider().overlay(StudioTheme.edge)
                    VStack(alignment: .leading, spacing: 10) {
                        Toggle(
                            L("Économie de mémoire", "Memory saving"),
                            isOn: Binding(
                                get: { studio.memorySaverEnabled },
                                set: { studio.setMemorySaverEnabled($0) }
                            )
                        )
                        .disabled(
                            studio.isGenerating || studio.isTuningMTP
                                || (studio.engineState.isReady && !studio.memorySaverSupported))
                        if studio.engineState.isReady && !studio.memorySaverSupported {
                            Text(
                                L(
                                    "Mets à jour le moteur pour utiliser cette option.",
                                    "Update the engine to use this option.")
                            )
                            .font(.system(size: 12))
                            .foregroundStyle(StudioTheme.secondary)
                        }
                        Text(
                            L(
                                "Libère le cache de contexte après chaque réponse sans décharger le modèle. Le prochain message peut démarrer plus lentement.",
                                "Frees the context cache after each response while keeping the model loaded. The next message may start more slowly."
                            )
                        )
                        .font(.system(size: 12))
                        .foregroundStyle(StudioTheme.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                    }

                }
            case .conversations:
                SettingCard(
                    title: L("Historique", "History"), icon: "bubble.left.and.bubble.right"
                ) {
                    Button(action: studio.exportConversations) {
                        Label(
                            L("Exporter les conversations…", "Export conversations…"),
                            systemImage: "square.and.arrow.up"
                        )
                        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                    }.buttonStyle(GlassPillButtonStyle())
                    Button(action: studio.importConversations) {
                        Label(
                            L("Importer une sauvegarde…", "Import a backup…"),
                            systemImage: "square.and.arrow.down"
                        )
                        .frame(maxWidth: .infinity, alignment: .leading).padding(12)
                    }.buttonStyle(GlassPillButtonStyle())
                }
            case .connections:
                SettingCard(title: "Hugging Face", icon: "key") {
                    HuggingFaceAccountView(library: studio.modelLibrary)
                }
                SettingCard(title: L("Outils MCP", "MCP tools"), icon: "shippingbox") {
                    MCPSettingsControls()
                }
            case .privacy:
                SettingCard(title: L("Sur ce Mac", "On this Mac"), icon: "desktopcomputer") {
                    dataFlowRow(
                        L("Inférence et kernels Metal", "Inference and Metal kernels"),
                        L("Exécutés localement sur votre Mac.", "Run locally on your Mac."), "cpu")
                }
                SettingCard(title: L("Accès au réseau", "Network access"), icon: "network") {
                    dataFlowRow(
                        L("Modèles et mises à jour", "Models and updates"),
                        "Hugging Face · GitHub", "arrow.down.circle")
                    Divider().overlay(StudioTheme.edge)
                    dataFlowRow(
                        "MCP",
                        studio.mcpEnabled
                            ? L(
                                "Activé · des données peuvent quitter ce Mac.", "Enabled · data may leave this Mac."
                            )
                            : L("Désactivé", "Disabled"), "link")
                    Text(
                        L(
                            "Un serveur MCP local peut lui-même utiliser Internet. Activez uniquement des outils de confiance.",
                            "A local MCP server can itself use the Internet. Enable only tools you trust.")
                    )
                    .foregroundStyle(StudioTheme.secondary)
                }
            case .updates:
                SettingCard(title: "MLXL3 Desktop", icon: "arrow.triangle.2.circlepath") {
                    UpdateSettingsCard(
                        updater: studio.updateManager, canInstall: !studio.isGenerating,
                        install: studio.installUpdateAndRestart)
                }
                Text(
                    L(
                        "L’app et le moteur peuvent être mis à jour séparément. Les téléchargements sont vérifiés avant installation.",
                        "The app and engine can update separately. Downloads are verified before installation.")
                )
                .foregroundStyle(StudioTheme.secondary)
            }
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func dataFlowRow(_ title: String, _ detail: String, _ icon: String) -> some View {
        HStack(alignment: .top, spacing: 13) {
            Image(systemName: icon).frame(width: 20).foregroundStyle(StudioTheme.secondary)
            VStack(alignment: .leading, spacing: 6) {
                Text(title).fontWeight(.medium)
                Text(detail).foregroundStyle(StudioTheme.secondary)
            }
        }
    }
}

private struct HuggingFaceAccountView: View {
    @ObservedObject var library: ModelLibrary
    @State private var token = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(
                L(
                    "Accédez aux modèles privés ou à accès restreint avec votre token Hugging Face.",
                    "Access private or gated models with your Hugging Face token.")
            )
            .foregroundStyle(StudioTheme.secondary)
            SecureField(L("Token Hugging Face · hf_…", "Hugging Face token · hf_…"), text: $token)
                .textFieldStyle(.plain).padding(12)
                .background(Color.black.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(StudioTheme.edge))
                .accessibilityLabel(L("Token Hugging Face", "Hugging Face token"))
                .onSubmit { signIn() }
            HStack(spacing: 12) {
                Button(action: signIn) {
                    Text(L("Se connecter", "Sign in")).padding(.horizontal, 16).padding(.vertical, 11)
                }
                .buttonStyle(PrimaryGlassButtonStyle()).disabled(!validToken)
                Button {
                    library.authenticate(token: nil)
                    token = ""
                } label: {
                    Text(L("Se déconnecter", "Sign out")).padding(.horizontal, 12).padding(.vertical, 11)
                }.buttonStyle(GlassPillButtonStyle())
            }
            Text(
                L(
                    "Le token est enregistré localement par Hugging Face. Pour un modèle à accès restreint, acceptez aussi sa licence sur son site.",
                    "Hugging Face stores your token locally. For a gated model, also accept its license on the website."
                )
            )
            .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
            if let message = library.authMessage {
                Text(message).textSelection(.enabled)
            }
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
        VStack(alignment: .leading, spacing: 14) {
            Toggle(
                L("Activer les outils MCP", "Enable MCP tools"),
                isOn: Binding(
                    get: { studio.mcpEnabled }, set: { studio.setMCPEnabled($0) }
                )
            )
            .toggleStyle(.switch)
            .disabled(studio.isGenerating || studio.mcpUpdating)
            Text(
                studio.mcpUpdating
                    ? L("Connexion aux serveurs…", "Connecting to servers…")
                    : studio.mcpEnabled
                        ? L(
                            "\(studio.mcpServerCount) serveur(s) connecté(s) · \(studio.mcpToolCount) outil(s)",
                            "\(studio.mcpServerCount) connected server(s) · \(studio.mcpToolCount) tool(s)")
                        : L("Les outils externes sont désactivés.", "External tools are disabled.")
            )
            .foregroundStyle(StudioTheme.secondary)
            ForEach(studio.mcpErrors.keys.sorted(), id: \.self) { key in
                Text("\(key) · \(studio.mcpErrors[key] ?? "")")
                    .foregroundStyle(.orange).textSelection(.enabled)
            }
            HStack(spacing: 10) {
                Button(action: { studio.showMCPManager = true }) {
                    Label(L("Configurer…", "Configure…"), systemImage: "doc.text")
                        .padding(10)
                }.buttonStyle(GlassPillButtonStyle())
                Button(action: studio.reloadMCPServers) {
                    Image(systemName: "arrow.clockwise").frame(width: 36, height: 36)
                }
                .buttonStyle(RoundGlassButtonStyle())
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
            .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
        }
    }
}

private struct UpdateSettingsCard: View {
    @ObservedObject var updater: UpdateManager
    let canInstall: Bool
    let install: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .top, spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 11, style: .continuous)
                        .fill(StudioTheme.contrast.opacity(0.07))
                    Image(systemName: "arrow.triangle.2.circlepath")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(StudioTheme.accent)
                }
                .frame(width: 40, height: 40)

                VStack(alignment: .leading, spacing: 4) {
                    Text(L("Mises à jour", "Updates"))
                        .font(.system(size: 15, weight: .bold))
                    Text(
                        L(
                            "App \(updater.currentVersion) · Moteur \(updater.currentEngineVersion)",
                            "App \(updater.currentVersion) · Engine \(updater.currentEngineVersion)")
                    )
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(StudioTheme.quiet)
                }
                Spacer()
                updateAction
            }

            Picker(
                L("Canal des mises à jour", "Update channel"),
                selection: Binding(
                    get: { updater.updateChannel }, set: { updater.setUpdateChannel($0) }
                )
            ) {
                Text("Stable").tag(UpdateChannel.stable)
                Text(L("Bêta", "Beta")).tag(UpdateChannel.beta)
            }
            .pickerStyle(.segmented)
            .disabled(updater.isInstalling)
            Text(
                L(
                    "Utilise Rechercher pour vérifier les versions du canal choisi.",
                    "Use Search to check versions from the selected channel.")
            )
            .font(.system(size: 13)).foregroundStyle(StudioTheme.quiet)

            HStack(spacing: 10) {
                stateIcon
                VStack(alignment: .leading, spacing: 3) {
                    Text(statusTitle)
                        .font(.system(size: 12.5, weight: .semibold))
                    Text(statusDetail)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(StudioTheme.quiet)
                        .fixedSize(horizontal: false, vertical: true)
                    if case .downloading = updater.state, let progress = updater.downloadProgress {
                        UpdateTransferView(progress: progress)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(13)
            .background(Color.black.opacity(0.20), in: RoundedRectangle(cornerRadius: 12))
            .overlay {
                RoundedRectangle(cornerRadius: 12)
                    .stroke(StudioTheme.contrast.opacity(0.07), lineWidth: 0.7)
            }

            HStack(spacing: 10) {
                Image(systemName: "cpu")
                VStack(alignment: .leading, spacing: 3) {
                    Text(L("Moteur indépendant", "Independent engine"))
                        .font(.system(size: 13, weight: .semibold))
                    Text(engineDetail)
                        .font(.system(size: 13))
                        .foregroundStyle(StudioTheme.quiet)
                    if case .downloading = updater.engineState, let progress = updater.engineDownloadProgress {
                        UpdateTransferView(progress: progress)
                    }
                }
                Spacer()
            }

            if let release = updater.latestRelease,
                updater.hasAppUpdate,
                !release.notes.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            {
                VStack(alignment: .leading, spacing: 7) {
                    HStack {
                        Text(L("NOUVEAUTÉS · \(release.tag.uppercased())", "WHAT’S NEW · \(release.tag.uppercased())"))
                            .font(.system(size: 13, weight: .bold))
                            .tracking(1.1)
                            .foregroundStyle(StudioTheme.quiet)
                        Spacer()
                        Button(L("Voir la release", "View release")) { updater.openReleasePage() }
                            .buttonStyle(.plain)
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(StudioTheme.accent)
                    }
                    Text(release.notes)
                        .font(.system(size: 13, weight: .regular))
                        .foregroundStyle(StudioTheme.secondary)
                        .lineSpacing(3)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                }
            }

            if updater.hasReadyUpdate && !canInstall {
                Text(
                    L(
                        "Termine ou arrête la génération avant de redémarrer pour installer.",
                        "Finish or stop generation before restarting to install.")
                )
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(Color.orange.opacity(0.82))
            }
        }
        .padding(.vertical, 12)
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
                .font(.system(size: 13, weight: .bold))
                .padding(.horizontal, 13)
                .frame(height: 34)
            }
            .buttonStyle(PrimaryGlassButtonStyle())
            .disabled(!canInstall)
        } else if updater.isBusy {
            ProgressView()
                .controlSize(.small)
                .frame(width: 34, height: 34)
        } else {
            Button(action: updater.checkForUpdates) {
                Text(L("Rechercher", "Search"))
                    .font(.system(size: 13, weight: .bold))
                    .padding(.horizontal, 15)
                    .frame(height: 34)
            }
            .buttonStyle(GlassPillButtonStyle())
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

    @ViewBuilder
    private var stateIcon: some View {
        switch updater.state {
        case .checking, .downloading, .installing:
            ProgressView().controlSize(.small)
        case .ready:
            Image(systemName: "arrow.down.circle.fill")
                .foregroundStyle(StudioTheme.accent)
        case .failed:
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(Color.orange.opacity(0.9))
        case .upToDate:
            Image(systemName: "checkmark.seal.fill")
                .foregroundStyle(Color(red: 0.42, green: 1.0, blue: 0.69))
        case .idle:
            Image(systemName: "circle.dotted")
                .foregroundStyle(StudioTheme.quiet)
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
    @Environment(\.studioAppTheme) private var appTheme
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
                        .overlay(Circle().stroke(StudioTheme.sidebar(for: appTheme), lineWidth: 1.5))
                        .offset(x: 1, y: -1)
                }
            }
        }
        .buttonStyle(RoundGlassButtonStyle())
        .help(updater.hasAvailableUpdate ? L("Mise à jour prête", "Update ready") : L("Réglages", "Settings"))
    }
}
