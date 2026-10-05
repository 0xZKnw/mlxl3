import SwiftUI

struct MTPTuningControls: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if studio.mtpEnabled {
                Picker(L("Profondeur", "Depth"), selection: Binding(
                    get: { studio.mtpDepth }, set: { studio.setMTPDepth($0) }
                )) {
                    ForEach(1...studio.mtpMaxDepth, id: \.self) { depth in
                        Text("MTP\(depth)").tag(depth)
                    }
                }
                .pickerStyle(.segmented).disabled(studio.isGenerating)
            }
            if studio.isTuningMTP {
                HStack {
                    Text(studio.mtpTuneStatus).font(.system(size: 11, weight: .medium))
                    Spacer()
                    Button(L("Arrêter", "Stop"), action: studio.cancelMTPTuning)
                        .buttonStyle(GlassPillButtonStyle())
                }
                ProgressView(value: studio.mtpTuneProgress).tint(StudioTheme.accent)
                Text(L("Baseline → MTP1 → MTP2 → MTP3 · puis ordre inversé", "Baseline → MTP1 → MTP2 → MTP3 · then reverse order"))
                    .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
            } else {
                Button(action: studio.tuneMTP) {
                    HStack {
                        Image(systemName: "slider.horizontal.3")
                        Text("Tune MTP")
                        Spacer()
                        Image(systemName: "sparkles").foregroundStyle(StudioTheme.accent)
                    }.frame(height: 29)
                }
                .buttonStyle(GlassPillButtonStyle()).disabled(!studio.canTuneMTP)
            }
            if !studio.mtpTuneRows.isEmpty && !studio.isTuningMTP {
                let winner = MTPTuning.winner(studio.mtpTuneRows)
                let maxTPS = studio.mtpTuneRows.map(\.decodeTPS).filter(\.isFinite).max() ?? 1
                VStack(spacing: 9) {
                    ForEach(studio.mtpTuneRows) { row in
                        VStack(spacing: 4) {
                            HStack {
                                Text(row.label).font(.system(size: 10, weight: .semibold))
                                if row.depth == winner {
                                    Text(L("MEILLEUR", "BEST")).font(.system(size: 8, weight: .bold))
                                        .foregroundStyle(StudioTheme.accent)
                                }
                                Spacer()
                                Text(String(format: "%.1f tok/s", row.decodeTPS))
                                    .font(.system(size: 10, weight: .medium)).monospacedDigit()
                                    .foregroundStyle(row.score == nil ? StudioTheme.quiet : StudioTheme.ink)
                            }
                            GeometryReader { geometry in
                                Capsule().fill(Color.white.opacity(0.05))
                                    .overlay(alignment: .leading) {
                                        Capsule().fill(row.depth == winner ? StudioTheme.accent : Color.white.opacity(0.22))
                                            .frame(width: geometry.size.width * max(0, min(1, row.decodeTPS / max(1, maxTPS))))
                                    }
                            }.frame(height: 3)
                            if let reason = row.reason {
                                Text(reason == "zero_acceptance" ? L("Aucune proposition acceptée", "No proposals accepted")
                                     : reason == "target_mismatch" ? L("Sortie différente de la baseline", "Output differs from baseline")
                                     : L("Mesure insuffisante", "Insufficient measurement"))
                                    .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet).frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                }
                .padding(12).background(Color.white.opacity(0.025), in: RoundedRectangle(cornerRadius: 10))
                Text(L("Réglage enregistré : ", "Saved setting: ") + (studio.mtpEnabled ? "MTP\(studio.mtpDepth)" : "Baseline"))
                    .font(.system(size: 10, weight: .medium)).foregroundStyle(StudioTheme.accent)
            }
            Text(studio.mtpTuneSupported
                 ? studio.mtpHeadPath.isEmpty
                    ? L("Active MTP une fois pour préparer la tête, puis lance le test.", "Enable MTP once to prepare the head, then run the test.")
                    : L("Deux prompts locaux, sans outils. Le meilleur mode est enregistré pour ce modèle et ce Mac. Baseline si le gain ne dépasse pas 3 %.", "Two local prompts, without tools. The best mode is saved for this model and Mac. Baseline if the gain is within 3%.")
                 : L("MTP2/MTP3 et Tune MTP nécessitent le moteur 1.3.0. Mets-le à jour puis recharge le modèle.", "MTP2/MTP3 and Tune MTP require engine 1.3.0. Update it, then reload the model."))
                .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
        }
    }
}
