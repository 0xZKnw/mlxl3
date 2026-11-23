import SwiftUI

/// The DFlash2 toolbar entry, built like `MTPToolbarControl`: a compact state badge that
/// opens the same settings as the inspector, with Tune inside instead of a bare Tune button.
struct DFlashToolbarControl: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var presented = false

    private var badge: String {
        if studio.dflashDownloadError != nil { return "!" }
        return studio.dflash2Enabled || studio.dflashDownloading ? "On" : "Off"
    }

    private var busy: Bool { studio.dflashDownloading || studio.isTuningDFlash }

    private var accessibleStatus: String {
        if studio.dflashDownloadError != nil { return L("Erreur", "Error") }
        if busy { return L("En cours", "In progress") }
        return studio.dflash2Enabled ? L("Activé", "On") : L("Désactivé", "Off")
    }

    var body: some View {
        Button { presented.toggle() } label: {
            HStack(spacing: 5) {
                if busy {
                    ProgressView().controlSize(.mini)
                }
                Text("DFlash2")
                Text(badge)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(studio.dflashDownloadError != nil ? Color.orange
                                     : studio.dflash2Enabled || studio.dflashDownloading ? studio.accentColor : StudioTheme.quiet)
            }
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 8).frame(height: 30)
        }
        .buttonStyle(WorkspaceControlStyle(selected: presented))
        .help(L("Accélération DFlash2 · réglage mémorisé par modèle", "DFlash2 acceleration · remembered per model"))
        .accessibilityLabel(L("Réglages DFlash2", "DFlash2 settings"))
        .accessibilityValue(accessibleStatus)
        .onChange(of: studio.isTuningDFlash) { _, active in
            if active { presented = false }
        }
        .popover(isPresented: $presented, arrowEdge: .bottom) {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    HStack {
                        Label(L("Accélération", "Acceleration"), systemImage: "bolt.fill")
                            .font(.system(size: 18, weight: .semibold))
                        Spacer()
                        Button { presented = false } label: { Image(systemName: "xmark").frame(width: 24, height: 24) }
                            .buttonStyle(StudioControlStyle()).accessibilityLabel(L("Fermer", "Close"))
                    }
                    DFlashSettingsContent()
                }.padding(20)
            }
            .frame(width: 360).frame(maxHeight: 540)
            .background(StudioTheme.canvas)
            .studioAppearance(studio)
            .environmentObject(studio)
        }
    }
}

/// DFlash2 settings shared by the toolbar popover and the inspector card, laid out like
/// `MTPSettingsContent`: switch, state, preparation, Tune, then the draft options.
struct DFlashSettingsContent: View {
    @EnvironmentObject private var studio: StudioModel

    private var flashNext: Bool { studio.selectedModel?.modelType == "qwen4_exp" }

    private var status: String {
        if studio.isTuningDFlash { return studio.dflashTuneStatus }
        if studio.dflashDownloading { return L("Préparation du draft…", "Preparing the draft…") }
        if studio.dflashDownloadError != nil {
            // A Tune error leaves an active draft untouched.
            return studio.dflash2Enabled && studio.dflashActive == true
                ? L("Actif · dernier Tune à vérifier", "Active · last Tune needs attention")
                : L("Activation à vérifier", "Activation needs attention")
        }
        if !studio.dflash2Available {
            return studio.dflash2Enabled
                ? L("Préférence conservée ; indisponible pour ce modèle", "Preference saved; unavailable for this model")
                : L("Indisponible pour ce modèle", "Unavailable for this model")
        }
        if studio.isGenerating {
            return studio.dflashActive == true
                ? L("Actif", "Active")
                : L("Décodage normal", "Ordinary decoding")
        }
        return studio.dflash2Enabled ? L("Prêt", "Ready") : L("Désactivé", "Off")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: Binding(
                get: { studio.dflash2Enabled || studio.dflashDownloading },
                set: { studio.setDFlash2Enabled($0) }
            )) {
                Text(L("Activer DFlash2", "Enable DFlash2")).font(.system(size: 13, weight: .semibold))
            }
            .toggleStyle(.switch)
            .disabled(studio.isGenerating
                      || (!(studio.dflash2Enabled || studio.dflashDownloading)
                          && (!studio.dflash2Available || !studio.engineState.isReady)))
            .help(L("Choix mémorisé. DFlash2 utilise le décodage déterministe et vérifie chaque proposition avec le modèle cible.", "Remembered. DFlash2 uses deterministic decoding and verifies every proposal with the target model."))
            Text(status).font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                .accessibilityIdentifier("dflash-state")
            if studio.dflashDownloading {
                if studio.dflashDownloadTotal > 0 {
                    ProgressView(value: studio.dflashDownloadCompleted, total: studio.dflashDownloadTotal)
                } else { ProgressView().controlSize(.small) }
                Button(L("Annuler la préparation", "Cancel preparation")) { studio.setDFlash2Enabled(false) }
                    .buttonStyle(GlassPillButtonStyle())
            }
            if let error = studio.dflashDownloadError {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                if !flashNext && !(studio.dflash2Enabled && studio.dflashActive == true) {
                    Button(L("Réessayer l’activation", "Retry activation")) { studio.setDFlash2Enabled(true) }
                        .disabled(studio.isGenerating || studio.dflashDownloading || !studio.dflash2Available || !studio.engineState.isReady)
                }
            }
            Divider()
            MTPTuningControls(dflash: true).disabled(!studio.dflash2Available)
            if flashNext {
                draftActions
            } else {
                DisclosureGroup(L("Draft DFlash2 · options avancées", "DFlash2 draft · advanced options")) {
                    VStack(alignment: .leading, spacing: 8) { draftActions }.padding(.top, 6)
                }.font(.system(size: 10.5))
            }
        }
    }

    @ViewBuilder private var draftActions: some View {
        DFlashDraftButton(
            title: !flashNext && studio.dflashDraftIsQwen38Dense
                ? L("Télécharger le draft Qwen3.8-27B", "Download the Qwen3.8-27B draft")
                : L("Télécharger le draft compatible", "Download the compatible draft"),
            icon: "arrow.down.circle", action: studio.downloadCompatibleDFlashDraft
        )
        .accessibilityIdentifier("dflash.download")
        .disabled(!studio.dflash2Available || !studio.engineState.isReady || studio.isGenerating || studio.dflashDownloading)
        Text(flashNext
             ? L("Draft Flash Next : environ 1,2 Go. Chaque token est vérifié par la cible.",
                 "Flash Next draft: about 1.2 GB. The target verifies every token.")
             : studio.dflashDraftIsQwen38Dense
             ? L("Draft incoai/Qwen3.8-27B-Splash : environ 1 Go, version épinglée et vérifiée. Chaque token est vérifié par la cible.",
                 "incoai/Qwen3.8-27B-Splash draft: about 1 GB, pinned and verified. The target verifies every token.")
             : L("Un clic prépare le draft adapté au modèle actif. Chaque token est vérifié par la cible.",
                 "One click prepares the draft matching the active model. The target verifies every token."))
            .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
        DFlashDraftButton(title: L("Choisir un draft local…", "Choose a local draft…"),
                          icon: "folder", action: studio.chooseDFlashDraft)
            .disabled(!studio.dflash2Available || studio.isGenerating || studio.dflashDownloading)
        if !studio.dflashDraftPath.isEmpty {
            Text(studio.dflashDraftPath).font(.caption2).foregroundStyle(StudioTheme.quiet).textSelection(.enabled)
        }
    }
}

/// The DFlash2 draft actions, styled like Tune's "Compare speeds" button.
private struct DFlashDraftButton: View {
    let title: String
    let icon: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Label(title, systemImage: icon)
                .font(.system(size: 12, weight: .medium))
                .padding(.horizontal, 12).frame(height: 32)
        }
        .buttonStyle(StudioToolbarStyle())
    }
}
