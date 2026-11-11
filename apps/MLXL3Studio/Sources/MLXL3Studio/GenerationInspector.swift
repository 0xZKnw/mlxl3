import SwiftUI

struct GenerationInspector: View {
    @Environment(\.studioAppTheme) private var appTheme
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(L("Paramètres", "Settings"))
                        .font(.system(size: 27, weight: .regular, design: .serif))
                    Text(studio.selectedModel?.name ?? L("Aucun modèle sélectionné", "No model selected"))
                        .lineLimit(2)
                        .font(.system(size: 13))
                        .foregroundStyle(StudioTheme.quiet)
                }
                Spacer()
                Button {
                    studio.showInspector = false
                } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 13, weight: .bold))
                        .frame(width: 34, height: 34)
                }
                .buttonStyle(RoundGlassButtonStyle())
                .accessibilityLabel(L("Fermer les paramètres du modèle", "Close model settings"))
            }
            .padding(.top, 25)
            .padding(.horizontal, 18)
            .padding(.bottom, 22)

            ScrollViewReader { proxy in
                ScrollView {
                    VStack(spacing: 16) {
                        SettingCard(title: L("Contexte", "Context"), icon: "chart.bar") {
                            Text(studio.contextLabel + " tokens")
                                .font(.system(size: 13, weight: .medium)).monospacedDigit()
                            if let limit = studio.activeContextLimit, limit > 0 {
                                ProgressView(value: min(Double(studio.contextUsed ?? 0) / Double(limit), 1))
                                    .tint(StudioTheme.accent)
                            }
                            ContextLimitField(value: $studio.contextLengthDraft, maximum: studio.modelContextLimit)
                                .disabled(studio.isGenerating || studio.modelContextLimit == nil)
                            if let cache = studio.draftContextBytes, let model = studio.modelResidentBytes {
                                VStack(alignment: .leading, spacing: 7) {
                                    HStack {
                                        Text(L("Modèle + contexte", "Model + context"))
                                        Spacer()
                                        Text(memory(model + cache)).monospacedDigit()
                                    }
                                    .font(.system(size: 13, weight: .semibold))
                                    Text(L("Estimation à contexte plein", "Full-context estimate"))
                                        .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                        .help(L("Modèle \(memory(model)) + cache \(memory(cache)). Hors buffers de calcul, autres caches et macOS.", "Model \(memory(model)) + cache \(memory(cache)). Excludes compute buffers, other caches and macOS."))
                                    if model + cache > Double(ProcessInfo.processInfo.physicalMemory) * 0.85 {
                                        Label(
                                            L(
                                                "Peu de marge mémoire pour le calcul et macOS.",
                                                "Little memory headroom for compute and macOS."),
                                            systemImage: "exclamationmark.triangle"
                                        )
                                        .font(.system(size: 12)).foregroundStyle(.orange)
                                    }
                                }
                                .padding(10)
                                .background(StudioTheme.contrast.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                            } else {
                                Text(L("Estimation mémoire indisponible", "Memory estimate unavailable"))
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                            }
                            Menu(L("Préréglages", "Presets")) {
                                Button(L("Automatique", "Automatic")) { studio.contextLengthDraft = 0 }
                                ForEach(
                                    [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144].filter {
                                        $0 <= (studio.modelContextLimit ?? 0)
                                    }, id: \.self
                                ) { length in
                                    Button(length.formatted()) { studio.contextLengthDraft = length }
                                }
                            }
                            .disabled(studio.isGenerating || studio.modelContextLimit == nil)
                            if let advised = studio.recommendedContextLength {
                                Button(
                                    L(
                                        "Utiliser le contexte conseillé : \(advised.formatted()) tokens",
                                        "Use recommended context: \(advised.formatted()) tokens")
                                ) {
                                    studio.contextLengthDraft = advised
                                }
                                .font(.system(size: 12))
                                .disabled(studio.isGenerating || studio.isTuningMTP)
                                .help(L("Estimation avec marge mémoire ; les autres applications peuvent réduire la marge disponible.", "Estimate includes memory headroom; other apps may reduce available memory."))
                            }
                            if let maximum = studio.modelContextLimit {
                                Text(
                                    L(
                                        "Maximum du modèle : \(maximum.formatted()) tokens",
                                        "Model maximum: \(maximum.formatted()) tokens")
                                )
                                .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                            }
                            Button(action: studio.saveContextAndReload) {
                                Text(L("Enregistrer et recharger le modèle", "Save and reload model"))
                                    .font(.system(size: 13, weight: .medium))
                                    .frame(maxWidth: .infinity).padding(.vertical, 10)
                            }
                            .buttonStyle(GlassPillButtonStyle())
                            .disabled(!studio.canSaveContext)
                            .help(L("0 utilise le maximum du modèle. Recharger vide le cache et conserve l’historique.", "0 uses the model maximum. Reloading clears the cache and keeps history."))
                        }
                        SettingCard(title: L("Échantillonnage", "Sampling"), icon: "dial.medium") {
                            VStack(spacing: 16) {
                                SliderSetting(
                                    label: L("Température", "Temperature"),
                                    value: Binding(
                                        get: { studio.samplingTemperature }, set: { studio.temperature = $0 }),
                                    range: 0...1.5,
                                    format: "%.2f"
                                )
                                Divider().overlay(StudioTheme.contrast.opacity(0.08))
                                VStack(spacing: 9) {
                                    HStack {
                                        Text("Top K")
                                        Spacer()
                                        Text("\(studio.samplingTopK)").monospacedDigit()
                                    }
                                    .font(.system(size: 13, weight: .medium))
                                    .foregroundStyle(StudioTheme.secondary)
                                    // Keep integer values without macOS drawing 201 overlapping tick marks.
                                    Slider(
                                        value: Binding(
                                            get: { Double(studio.samplingTopK) },
                                            set: { studio.topK = Int($0.rounded()) }
                                        ),
                                        in: 0...200
                                    )
                                    .tint(StudioTheme.accent)
                                    .accessibilityLabel("Top K")
                                }
                                Divider().overlay(StudioTheme.contrast.opacity(0.08))
                                SliderSetting(
                                    label: L("Répétition", "Repetition"),
                                    value: Binding(
                                        get: { studio.samplingRepetitionPenalty },
                                        set: { studio.repetitionPenalty = $0 }),
                                    range: 1...1.3,
                                    format: "%.2f"
                                )
                            }
                            .disabled(studio.samplingLocked)
                            if studio.samplingLocked {
                                let mode = studio.dflash2Applicable ? "DFlash2" : "MTP"
                                Label(
                                    L(
                                        "\(mode) utilise le mode greedy. Désactive \(mode) pour modifier ces paramètres.",
                                        "\(mode) uses greedy sampling. Disable \(mode) to change these settings."),
                                    systemImage: "lock"
                                )
                                .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                            }
                        }

                        if studio.mtpAvailable || studio.mtpRememberedEnabled || studio.mtpDownloading {
                            SettingCard(title: L("Accélération MTP", "MTP acceleration"), icon: "bolt.fill") {
                                MTPSettingsContent().id("mtp-tuning")
                            }
                        }

                        if studio.dflash2Available || studio.dflash2Enabled || studio.dflashDownloading {
                            SettingCard(title: "DFlash2 · Qwen3.6 / Qwen3.8", icon: "bolt.fill") {
                                Toggle(
                                    studio.dflash2Available
                                        ? L("Accélérer avec DFlash2", "Accelerate with DFlash2")
                                        : L(
                                            "Préférence DFlash2 (inactive pour ce modèle)",
                                            "DFlash2 preference (inactive for this model)"),
                                    isOn: Binding(
                                        get: { studio.dflash2Enabled || studio.dflashDownloading },
                                        set: { studio.setDFlash2Enabled($0) }
                                    )
                                ).font(.system(size: 13))
                                    .disabled(
                                        studio.isGenerating
                                            || (!(studio.dflash2Enabled || studio.dflashDownloading)
                                                && (!studio.dflash2Available || !studio.engineState.isReady))
                                    )
                                if !studio.dflash2Available {
                                    Text(
                                        L(
                                            "Préférence DFlash2 conservée ; indisponible pour ce modèle. Tu peux la désactiver ici.",
                                            "DFlash2 preference saved; unavailable for this model. You can turn it off here."
                                        )
                                    )
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                }
                                MTPTuningControls(dflash: true).disabled(!studio.dflash2Available)
                                if studio.dflashDownloading {
                                    ProgressView(
                                        value: studio.dflashDownloadTotal > 0
                                            ? studio.dflashDownloadCompleted / studio.dflashDownloadTotal : 0)
                                    Text(
                                        L(
                                            "Téléchargement et préparation du draft…",
                                            "Downloading and preparing the draft…")
                                    )
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                    Button(L("Annuler la préparation", "Cancel preparation")) {
                                        studio.setDFlash2Enabled(false)
                                    }
                                    .buttonStyle(GlassPillButtonStyle())
                                }
                                if let error = studio.dflashDownloadError {
                                    Text(error).font(.system(size: 12)).foregroundStyle(.red)
                                }
                                if studio.selectedModel?.modelType == "qwen4_exp" {
                                    Button(
                                        L("Télécharger le draft compatible", "Download the compatible draft"),
                                        action: studio.downloadCompatibleDFlashDraft
                                    )
                                    .buttonStyle(GlassPillButtonStyle())
                                    .disabled(
                                        !studio.dflash2Available || !studio.engineState.isReady || studio.isGenerating
                                            || studio.dflashDownloading)
                                    Text(
                                        L(
                                            "Draft Flash Next : environ 1,2 Go. Chaque token est vérifié par la cible.",
                                            "Flash Next draft: about 1.2 GB. The target verifies every token.")
                                    )
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                } else {
                                    Text(
                                        L(
                                            "Un clic prépare le draft adapté au modèle actif. Chaque token est vérifié par la cible. Tune choisit le meilleur mode sur ce Mac.",
                                            "One click prepares the draft matching the active model. The target verifies every token. Tune selects the best mode on this Mac."
                                        )
                                    )
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                                }
                                Button(
                                    L("Choisir un draft local…", "Choose a local draft…"),
                                    action: studio.chooseDFlashDraft
                                )
                                .buttonStyle(GlassPillButtonStyle()).disabled(
                                    !studio.dflash2Available || studio.isGenerating || studio.dflashDownloading)
                            }
                            .id("dflash-tuning")
                        }

                        if let runtime = studio.runtimeIdentity, !runtime.isEmpty {
                            DisclosureGroup(L("Détails du moteur", "Engine details")) {
                                Text(runtime).textSelection(.enabled)
                            }.font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                        }

                        Button(L("Projets et profils enregistrés", "Projects and saved profiles")) {
                            studio.showWorkspace = true
                        }
                        if studio.selectedMode != .agent {
                            SettingCard(title: L("JSON validé", "Validated JSON"), icon: "curlybraces") {
                                TextEditor(text: $studio.responseSchema).font(.system(.caption, design: .monospaced))
                                    .frame(minHeight: 80)
                                Text(
                                    L(
                                        "Schéma JSON optionnel. Une réponse non conforme échoue explicitement ; le texte est livré après validation. MCP est désactivé pour cette requête. Vide = texte libre.",
                                        "Optional JSON schema. Invalid output fails explicitly; text is delivered after validation. MCP is disabled for this request. Empty = free text."
                                    )
                                ).font(.system(size: 12)).foregroundStyle(.secondary)
                                Button(L("Revenir au texte libre", "Use free text")) { studio.responseSchema = "" }
                            }
                        }
                        SettingCard(title: L("Instruction système", "System instruction"), icon: "command") {
                            TextEditor(text: $studio.systemPrompt)
                                .font(.system(size: 13))
                                .scrollContentBackground(.hidden)
                                .frame(height: 150)
                                .accessibilityLabel(L("Instruction système", "System instruction"))
                                .padding(9)
                                .background(Color.black.opacity(0.22), in: RoundedRectangle(cornerRadius: 10))
                                .overlay {
                                    RoundedRectangle(cornerRadius: 10)
                                        .stroke(StudioTheme.contrast.opacity(0.07), lineWidth: 0.7)
                                }
                                .overlay(alignment: .topLeading) {
                                    if studio.systemPrompt.isEmpty {
                                        Text(L("Comportement optionnel du modèle…", "Optional model behavior…"))
                                            .font(.system(size: 13))
                                            .foregroundStyle(StudioTheme.quiet)
                                            .padding(.horizontal, 14)
                                            .padding(.vertical, 15)
                                            .allowsHitTesting(false)
                                    }
                                }
                        }

                        if studio.selectedMode == .agent {
                            SettingCard(title: L("Sandbox native", "Native sandbox"), icon: "lock.shield") {
                                Text(
                                    L(
                                        "Dossier privé par tâche · réseau bloqué · aucun serveur MCP externe.",
                                        "Private task folder · network blocked · no external MCP servers."))
                                Text(
                                    L(
                                        "5 s par commande · 32 Kio de sortie · outils système macOS.",
                                        "5 s per command · 32 KiB captured output · macOS system tools."))
                                Text(
                                    L("\(studio.mcpToolCount) outil connecté", "\(studio.mcpToolCount) connected tool"))
                                ForEach(studio.mcpErrors.keys.sorted(), id: \.self) { key in
                                    Text(studio.mcpErrors[key] ?? "").foregroundStyle(.red)
                                }
                            }.font(.system(size: 12))
                        } else {
                            SettingCard(title: "MCP", icon: "shippingbox") {
                                HStack {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(
                                            L(
                                                "\(studio.mcpServerCount) serveur\(studio.mcpServerCount == 1 ? "" : "s") connecté\(studio.mcpServerCount == 1 ? "" : "s")",
                                                "\(studio.mcpServerCount) connected server\(studio.mcpServerCount == 1 ? "" : "s")"
                                            )
                                        )
                                        .font(.system(size: 13, weight: .semibold))
                                        Text(
                                            L(
                                                "\(studio.mcpToolCount) outil\(studio.mcpToolCount == 1 ? "" : "s") disponible\(studio.mcpToolCount == 1 ? "" : "s")",
                                                "\(studio.mcpToolCount) available tool\(studio.mcpToolCount == 1 ? "" : "s")"
                                            )
                                        )
                                        .font(.system(size: 13, weight: .medium))
                                        .foregroundStyle(StudioTheme.quiet)
                                    }
                                    Spacer()
                                    Circle()
                                        .fill(studio.mcpServerCount > 0 ? Color.green : StudioTheme.quiet)
                                        .frame(width: 7, height: 7)
                                }

                                if let error = studio.mcpErrors.sorted(by: { $0.key < $1.key }).first {
                                    Text("\(error.key) · \(error.value)")
                                        .font(.system(size: 13, weight: .medium))
                                        .foregroundStyle(Color(red: 1, green: 0.56, blue: 0.56))
                                        .lineLimit(3)
                                }

                                HStack(spacing: 8) {
                                    Button(action: { studio.showMCPManager = true }) {
                                        Label(L("Configurer", "Configure"), systemImage: "doc.text")
                                            .frame(maxWidth: .infinity)
                                            .frame(height: 29)
                                    }
                                    .buttonStyle(GlassPillButtonStyle())

                                    Button(action: studio.reloadMCPServers) {
                                        Image(systemName: "arrow.clockwise")
                                            .frame(width: 29, height: 29)
                                    }
                                    .buttonStyle(RoundGlassButtonStyle())
                                    .disabled(studio.isGenerating)
                                    .help(L("Recharger les serveurs MCP", "Reload MCP servers"))
                                }

                                Text(
                                    L(
                                        "Les outils MCP locaux restent sur ce Mac. Exa et les serveurs distants peuvent recevoir ta requête et leur résultat.",
                                        "Local MCP tools stay on this Mac. Exa and remote servers may receive your request and return data."
                                    )
                                )
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(StudioTheme.quiet)
                                .fixedSize(horizontal: false, vertical: true)
                            }
                        }

                        if let model = studio.selectedModel {
                            SettingCard(title: L("Modèle actif", "Active model"), icon: "cpu") {
                                InspectorValue(label: "Architecture", value: model.architectureLabel)
                                InspectorValue(
                                    label: L("Quantification", "Quantization"),
                                    value: model.bits.map { String(format: "EXL3 %.2f BPW", $0) } ?? "EXL3")
                                InspectorValue(label: L("Poids", "Weights"), value: model.size)
                                InspectorValue(label: "Modules", value: "\(model.modules)")
                            }
                        }
                    }
                    .padding(.horizontal, 16)
                    .padding(.bottom, 20)
                }
                .onChange(of: studio.isTuningMTP, initial: true) { _, running in
                    if studio.isTuningDFlash
                        || (!running && (studio.dflashDownloadError != nil || !studio.dflashTuneRows.isEmpty))
                    {
                        proxy.scrollTo("dflash-tuning", anchor: .top)
                    } else if running || studio.mtpError != nil || !studio.mtpTuneRows.isEmpty {
                        proxy.scrollTo("mtp-tuning", anchor: .top)
                    }
                }
            }
        }
        .background(StudioTheme.sidebar(for: appTheme))
        .onChange(of: studio.temperature) { studio.settingsDidChange() }
        .onChange(of: studio.topK) { studio.settingsDidChange() }
        .onChange(of: studio.repetitionPenalty) { studio.settingsDidChange() }
        .onChange(of: studio.systemPrompt) { studio.settingsDidChange() }
    }

    private func memory(_ bytes: Double) -> String {
        String(format: "%.2f %@", bytes / 1e9, L("Go", "GB"))
    }
}

struct SettingCard<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    init(title: String, icon: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            Label(title, systemImage: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(StudioTheme.ink)
            content
        }
        .font(.system(size: 13))
        .padding(16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(StudioTheme.edge, lineWidth: 0.7))
    }
}

private struct SliderSetting: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        VStack(spacing: 9) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: format, value)).monospacedDigit()
            }
            .font(.system(size: 13, weight: .medium))
            .foregroundStyle(StudioTheme.secondary)
            Slider(value: $value, in: range)
                .accessibilityLabel(label)
                .tint(StudioTheme.contrast.opacity(0.88))
        }
    }
}

private struct InspectorValue: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(StudioTheme.quiet)
            Spacer()
            Text(value)
                .foregroundStyle(StudioTheme.contrast.opacity(0.78))
                .multilineTextAlignment(.trailing)
        }
        .font(.system(size: 13, weight: .medium))
    }
}

/// Keep invalid input visible and invalidate the draft instead of reusing a stale valid number.
struct ContextLimitField: View {
    @Binding var value: Int
    let maximum: Int?
    @State private var text = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(L("Limite en tokens", "Token limit")).fontWeight(.medium)
            TextField(L("0 = automatique", "0 = automatic"), text: $text)
                .textFieldStyle(.plain).monospacedDigit().padding(11)
                .background(Color.black.opacity(0.2), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(StudioTheme.edge))
                .accessibilityLabel(L("Limite en tokens", "Token limit"))
                .accessibilityIdentifier("settings.contextLimit")
                .onChange(of: text) { value = Int(text) ?? -1 }
            if let maximum, value < 0 || value > maximum {
                Text(
                    L(
                        "Saisissez un entier de 0 à \(maximum.formatted()).",
                        "Enter a whole number from 0 to \(maximum.formatted()).")
                )
                .font(.system(size: 12)).foregroundStyle(.orange)
            }
        }
        .onAppear { text = String(value) }
        .onChange(of: value) {
            if value >= 0 && Int(text) != value { text = String(value) }
        }
    }
}
