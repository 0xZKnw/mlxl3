import SwiftUI

/// Keep the transcript visible; opening detailed results is an explicit action.
struct MTPTuningToolbarControl: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Button {
                if studio.isTuningMTP { studio.cancelMTPTuning() }
                else { studio.tuneMTP() }
            } label: {
                HStack(spacing: 6) {
                    if studio.isTuningMTP {
                        ProgressView().progressViewStyle(.circular).controlSize(.mini)
                            .tint(StudioTheme.accent).frame(width: 12, height: 12)
                            .accessibilityHidden(true)
                    } else {
                        Image(systemName: "sparkles")
                    }
                    Text("Tune MTP")
                    if studio.isTuningMTP {
                        Text("\(Int(studio.mtpTuneProgress * 100)) %").monospacedDigit()
                        Image(systemName: "stop.fill").font(.system(size: 9))
                    }
                }
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 10).frame(height: 30)
            }
            .buttonStyle(StudioControlStyle(emphasized: true))
            .disabled(!studio.isTuningMTP && !studio.canTuneMTP)
            .help(studio.isTuningMTP
                  ? studio.mtpTuneStatus + L(" · Cliquer pour arrêter", " · Click to stop")
                  : L("Comparer les modes MTP et enregistrer le meilleur réglage", "Compare MTP modes and save the best setting"))
            .accessibilityLabel(studio.isTuningMTP ? L("Arrêter le test MTP", "Stop MTP test") : "Tune MTP")
            .accessibilityValue(studio.isTuningMTP ? studio.mtpTuneStatus + " · \(Int(studio.mtpTuneProgress * 100)) %" : "")

            if studio.isTuningMTP {
                Text(studio.mtpTuneStatus)
                    .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
                    .frame(maxWidth: 180, alignment: .leading).lineLimit(1)
            } else if let message = studio.mtpError {
                Button(L("Message MTP · Détails", "MTP message · Details")) { studio.showInspector = true }
                    .buttonStyle(.plain).font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
                    .help(message)
            } else if let winner = MTPTuning.winner(studio.mtpTuneRows) {
                Button(L("Résultat : ", "Result: ") + (winner == 0 ? "Baseline" : "MTP\(winner)") + L(" · Détails", " · Details")) {
                    studio.showInspector = true
                }
                .buttonStyle(.plain).font(.system(size: 9)).foregroundStyle(StudioTheme.accent)
            }
        }
        .fixedSize()
    }
}

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
                if winner == 0 {
                    Text(L("Baseline retenue : aucun gain MTP supérieur à 3 % mesuré sur ce test. Le résultat peut varier selon le prompt et les conditions du Mac.", "Baseline retained: no MTP gain above 3% measured in this test. Results can vary with the prompt and Mac conditions."))
                        .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
                }
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
