import SwiftUI

struct MTPToolbarControl: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var presented = false

    private var badge: String {
        if studio.mtpError != nil { return "!" }
        if !studio.mtpRememberedEnabled { return "Off" }
        return studio.mtpEnabled && !studio.mtpDownloading ? "On" : "Auto"
    }

    private var accessibleStatus: String {
        if studio.mtpError != nil { return L("Erreur", "Error") }
        if studio.mtpDownloading || studio.isTuningMTP { return L("En cours", "In progress") }
        if !studio.mtpRememberedEnabled { return L("Désactivé", "Off") }
        return studio.mtpEnabled ? L("Activé", "On") : L("En attente", "Pending")
    }

    var body: some View {
        Button { presented.toggle() } label: {
            HStack(spacing: 5) {
                if studio.mtpDownloading || studio.isTuningMTP {
                    ProgressView().controlSize(.mini)
                }
                Text("MTP")
                Text(badge)
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(studio.mtpError != nil ? Color.orange : studio.mtpRememberedEnabled ? StudioTheme.blueAccent : StudioTheme.quiet)
            }
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 8).frame(height: 30)
        }
        .buttonStyle(WorkspaceControlStyle(selected: presented))
        .disabled(!studio.mtpAvailable && !studio.isTuningMTP)
        .help(L("Accélération MTP · activation mémorisée par modèle", "MTP acceleration · remembered per model"))
        .accessibilityLabel(L("Réglages MTP", "MTP settings"))
        .accessibilityValue(accessibleStatus)
        .onChange(of: studio.isTuningMTP) { _, active in
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
                    MTPSettingsContent()
                }.padding(20)
            }
            .frame(width: 360).frame(maxHeight: 540)
            .background(StudioTheme.canvas(for: studio.theme))
            .studioAppearance(studio.theme, accent: studio.accent, customAccent: studio.customAccentColor)
            .environmentObject(studio)
        }
    }
}

struct MTPSettingsContent: View {
    @EnvironmentObject private var studio: StudioModel

    private var status: String {
        if studio.isTuningMTP { return studio.mtpTuneStatus }
        if studio.mtpDownloading { return L("Préparation en cours…", "Preparing…") }
        if studio.mtpError != nil { return L("Activation à vérifier", "Activation needs attention") }
        if studio.isGenerating {
            return studio.mtpActive == true
                ? L("Actif", "Active")
                : L("Décodage normal", "Ordinary decoding")
        }
        if studio.mtpEnabled { return L("Prêt", "Ready") }
        if studio.mtpRememberedEnabled { return L("Sera activé au chargement", "Will enable on model load") }
        return L("Désactivé", "Off")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: Binding(get: { studio.mtpRememberedEnabled }, set: { studio.setMTPEnabled($0) })) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(L("Activer MTP", "Enable MTP")).font(.system(size: 13, weight: .semibold))
                }
            }
            .toggleStyle(.switch)
            .disabled(studio.isGenerating || !studio.engineState.isReady)
            .help(L("Choix mémorisé par modèle. MTP utilise le décodage déterministe et vérifie les propositions avec le modèle cible.", "Remembered per model. MTP uses deterministic decoding and verifies proposals with the target model."))
            Text(status).font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                .accessibilityIdentifier("mtp-state")
            if studio.mtpDownloading {
                if studio.mtpDownloadTotal > 0 {
                    ProgressView(value: studio.mtpDownloadCompleted, total: studio.mtpDownloadTotal)
                } else { ProgressView().controlSize(.small) }
            }
            if let error = studio.mtpError {
                Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                Button(L("Réessayer l’activation", "Retry activation")) { studio.setMTPEnabled(true) }
                    .disabled(studio.isGenerating || studio.mtpDownloading || !studio.engineState.isReady)
            }
            Divider()
            MTPTuningControls()
            if !studio.mtpUsesModelCheckpoint {
                DisclosureGroup(L("Tête MTP · options avancées", "MTP head · advanced options")) {
                    Button(L("Choisir une tête compatible…", "Choose a matching head…"), action: studio.chooseMTPHead)
                        .disabled(studio.mtpDownloading || studio.isGenerating)
                    if !studio.mtpHeadPath.isEmpty {
                        Text(studio.mtpHeadPath).font(.caption2).foregroundStyle(StudioTheme.quiet).textSelection(.enabled)
                    }
                }.font(.system(size: 10.5))
            }
        }
    }
}
