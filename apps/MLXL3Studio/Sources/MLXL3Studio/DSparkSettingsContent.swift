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
                Text("DSpark")
                Text(studio.dsparkEnabled ? "On" : "Off")
                    .font(.system(size: 10, weight: .medium))
                    .foregroundStyle(studio.dsparkEnabled ? studio.accentColor : StudioTheme.quiet)
            }
            .font(.system(size: 11, weight: .semibold))
            .padding(.horizontal, 8).frame(height: 30)
        }
        .buttonStyle(WorkspaceControlStyle(selected: presented))
        .help(L("Accélération DSpark · choix mémorisé", "DSpark acceleration · remembered"))
        .accessibilityIdentifier("toolbar.dspark")
        .accessibilityLabel(L("Réglages DSpark", "DSpark settings"))
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
            .studioAppearance(studio.theme, accent: studio.accent, customAccent: studio.customAccentColor)
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
            Toggle(isOn: Binding(get: { studio.dsparkEnabled }, set: { studio.setDSparkEnabled($0) })) {
                Text(L("Activer DSpark", "Enable DSpark")).font(.system(size: 13, weight: .semibold))
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
            Divider()
            Text(L("Brouillon fourni avec le modèle. La largeur des blocs suit l'acceptation pour ne pas ralentir le décodage.",
                   "Drafter shipped with the model. The block width follows the acceptance so decoding is not slowed down."))
                .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
