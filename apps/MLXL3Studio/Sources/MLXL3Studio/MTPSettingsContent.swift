import SwiftUI

struct MTPToolbarControl: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var presented = false

    private var badge: String {
        if studio.mtpError != nil { return "!" }
        if !studio.mtpRememberedEnabled { return "OFF" }
        return studio.mtpEnabled && !studio.mtpDownloading ? "ON" : "AUTO"
    }

    var body: some View {
        Button { presented.toggle() } label: {
            HStack(spacing: 6) {
                if studio.mtpDownloading || studio.isTuningMTP {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: "bolt.fill")
                }
                Text("MTP")
                Text(badge)
                    .font(.system(size: 8, weight: .bold, design: .rounded))
                    .padding(.horizontal, 5).padding(.vertical, 3)
                    .background(StudioTheme.contrast.opacity(0.07), in: Capsule())
            }
            .font(.system(size: 11, weight: .semibold))
            .foregroundStyle(studio.mtpRememberedEnabled ? StudioTheme.blueAccent : StudioTheme.secondary)
            .padding(.horizontal, 11).frame(height: 32)
            .background(studio.mtpRememberedEnabled ? StudioTheme.blueAccent.opacity(0.10) : StudioTheme.panel,
                        in: RoundedRectangle(cornerRadius: 9))
            .overlay(RoundedRectangle(cornerRadius: 9).stroke(studio.mtpRememberedEnabled
                ? StudioTheme.blueAccent.opacity(0.30) : StudioTheme.edge, lineWidth: 0.7))
        }
        .buttonStyle(.plain)
        .disabled(!studio.mtpAvailable && !studio.isTuningMTP)
        .help(L("Accélération MTP · activation mémorisée par modèle", "MTP acceleration · remembered per model"))
        .accessibilityLabel(L("Réglages MTP", "MTP settings"))
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
            .studioAppearance(studio.theme)
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
                ? L("Actif · propositions vérifiées par le modèle", "Active · proposals verified by the target model")
                : L("Décodage normal", "Ordinary decoding")
        }
        if studio.mtpEnabled { return L("Prêt · propositions vérifiées par le modèle", "Ready · proposals verified by the target model") }
        if studio.mtpRememberedEnabled { return L("Sera activé au chargement", "Will enable on model load") }
        return L("Désactivé", "Off")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: Binding(get: { studio.mtpRememberedEnabled }, set: { studio.setMTPEnabled($0) })) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(L("Activer MTP", "Enable MTP")).font(.system(size: 13, weight: .semibold))
                    Text(L("Choix mémorisé pour ce modèle", "Remembered for this model"))
                        .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                }
            }
            .toggleStyle(.switch)
            .disabled(studio.isGenerating || !studio.engineState.isReady)
            Text(status).font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                .accessibilityIdentifier("mtp-state")
            Text(L("Après activation, MTP est préparé automatiquement aux prochains chargements, y compris au démarrage de l’app.", "Once enabled, MTP is prepared automatically on future model loads, including app startup."))
                .font(.system(size: 10.5)).foregroundStyle(StudioTheme.quiet)
                .fixedSize(horizontal: false, vertical: true)
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
            if studio.mtpTuneSupported {
                Button(action: studio.tuneMTP) {
                    Label(L("Optimiser les réglages", "Optimize settings"), systemImage: "sparkles")
                        .frame(maxWidth: .infinity).padding(.vertical, 10)
                }
                .buttonStyle(PrimaryGlassButtonStyle())
                .disabled(!studio.canTuneMTP)
            }
            if !studio.mtpUsesModelCheckpoint {
                DisclosureGroup(L("Tête MTP · options avancées", "MTP head · advanced options")) {
                    Button(L("Choisir une tête compatible…", "Choose a matching head…"), action: studio.chooseMTPHead)
                        .disabled(studio.mtpDownloading || studio.isGenerating)
                    if !studio.mtpHeadPath.isEmpty {
                        Text(studio.mtpHeadPath).font(.caption2).foregroundStyle(StudioTheme.quiet).textSelection(.enabled)
                    }
                }.font(.system(size: 10.5))
            }
            Text(L("MTP utilise le décodage déterministe. Le modèle cible vérifie les propositions ; aucune quantification supplémentaire n’est appliquée.", "MTP uses deterministic decoding. The target model verifies proposals; no additional quantization is applied."))
                .font(.caption).foregroundStyle(StudioTheme.quiet)
        }
    }
}
