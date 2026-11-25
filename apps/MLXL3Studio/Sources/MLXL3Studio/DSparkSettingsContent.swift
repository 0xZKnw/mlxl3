import SwiftUI

/// The DSpark toolbar entry, built like `DFlashToolbarControl`: a compact On/Off badge that opens the
/// same settings as the inspector card. Shown for models that ship a DSpark drafter (Inkling 2).
struct DSparkToolbarControl: View {
    @EnvironmentObject private var studio: StudioModel
    @State private var presented = false

    private var accessibleStatus: String {
        studio.dsparkEnabled ? L("Activé", "On") : L("Désactivé", "Off")
    }

    var body: some View {
        Button { presented.toggle() } label: {
            HStack(spacing: 5) {
                Text("DFlash2")
                Text(studio.dsparkEnabled ? "On" : "Off")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(studio.dsparkEnabled ? studio.accentColor : StudioTheme.quiet)
            }
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 8).frame(height: 30)
        }
        .buttonStyle(WorkspaceControlStyle(selected: presented))
        .help(L("Accélération DFlash2 · choix mémorisé", "DFlash2 acceleration · remembered"))
        .accessibilityIdentifier("toolbar.dspark")
        .accessibilityLabel(L("Réglages DFlash2", "DFlash2 settings"))
        .accessibilityValue(accessibleStatus)
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
                    DSparkSettingsContent()
                }.padding(20)
            }
            .frame(width: 360).frame(maxHeight: 540)
            .background(StudioTheme.canvas)
            .studioAppearance(studio)
            .environmentObject(studio)
        }
    }
}

/// DSpark settings shared by the toolbar popover and the inspector card, laid out like
/// `DFlashSettingsContent`: switch, state, the latest answer's acceptance, then what the drafter does.
struct DSparkSettingsContent: View {
    @EnvironmentObject private var studio: StudioModel

    private var status: String {
        if studio.isGenerating {
            return studio.dsparkEnabled ? L("Actif", "Active") : L("Décodage normal", "Ordinary decoding")
        }
        return studio.dsparkEnabled ? L("Prêt", "Ready") : L("Désactivé", "Off")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: Binding(get: { studio.dsparkEnabled || studio.dsparkDrafterDownloading },
                                 set: { studio.setDSparkEnabled($0) })) {
                Text(L("Activer DFlash2", "Enable DFlash2")).font(.system(size: 13, weight: .semibold))
            }
            .toggleStyle(.switch)
            .disabled(studio.isGenerating)
            .help(L("Choix mémorisé. Le modèle cible vérifie chaque proposition du brouillon : le texte suit exactement sa loi d'échantillonnage.",
                    "Remembered. The target model verifies every draft proposal, so the text follows its sampling law exactly."))
            Text(status).font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                .accessibilityIdentifier("dspark-state")
            if let latest = studio.latestDSparkAcceptance {
                Text(String(format: L("Dernière réponse : %d/%d propositions acceptées (%.0f %%) · %.2f tokens par bloc",
                                      "Latest answer: %d/%d proposals accepted (%.0f%%) · %.2f tokens per block"),
                            latest.accepted, latest.proposed,
                            100 * Double(latest.accepted) / Double(latest.proposed), latest.tokensPerBlock))
                    .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("dspark-latest")
            }
            if studio.dsparkDrafterDownloading {
                if studio.dsparkDrafterTotal > 0 {
                    ProgressView(value: studio.dsparkDrafterCompleted, total: studio.dsparkDrafterTotal)
                        .accessibilityIdentifier("dspark-drafter-progress")
                } else {
                    ProgressView().controlSize(.small)
                }
                Text(L("Téléchargement du brouillon DFlash2 (≈ 1 Go). Les réponses restent normales en attendant ; coupe l'interrupteur pour annuler.",
                       "Downloading the DFlash2 drafter (≈ 1 GB). Answers decode normally meanwhile; turn the switch off to cancel."))
                    .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let error = studio.dsparkDrafterError {
                Text(error).font(.system(size: 11)).foregroundStyle(Color.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("dspark-drafter-error")
                Button(L("Réessayer le téléchargement", "Retry the download")) { studio.retryDSparkDrafter() }
                    .buttonStyle(StudioControlStyle())
                    .disabled(studio.isGenerating || !studio.engineState.isReady)
            }
            Divider()
            Text(studio.dsparkDefault == true
                 ? L("Brouillon DFlash2 téléchargé une fois pour ce modèle. La largeur des blocs suit l'acceptation pour ne pas ralentir le décodage.",
                     "DFlash2 drafter downloaded once for this model. The block width follows the acceptance so decoding is not slowed down.")
                 : L("Le brouillon DFlash2 (≈ 1 Go) se télécharge une fois pour ce modèle ; l'accélération s'active dès qu'il est installé.",
                     "The DFlash2 drafter (≈ 1 GB) is downloaded once for this model; the acceleration turns on as soon as it is installed."))
                .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
