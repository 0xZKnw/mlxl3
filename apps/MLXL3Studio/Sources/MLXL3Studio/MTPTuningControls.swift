import SwiftUI

struct MTPTuningControls: View {
    @EnvironmentObject private var studio: StudioModel
    var dflash = false
    private var enabled: Bool { dflash ? studio.dflash2Enabled : studio.mtpEnabled }
    private var tuning: Bool { dflash ? studio.isTuningDFlash : studio.isTuningMTPDepth }
    private var rows: [MTPTuningRow] { dflash ? studio.dflashTuneRows : studio.mtpTuneRows }
    private var progress: Double { dflash ? studio.dflashTuneProgress : studio.mtpTuneProgress }
    private var status: String { dflash ? studio.dflashTuneStatus : studio.mtpTuneStatus }
    private var depth: Int { dflash ? studio.dflashMode : studio.mtpDepth }
    private var canTune: Bool { dflash ? studio.canTuneDFlash : studio.canTuneMTP }
    private var supported: Bool { dflash ? studio.dflashTuneSupported : studio.mtpTuneSupported }
    private var artifact: String { dflash ? studio.dflashDraftPath : studio.mtpHeadPath }
    private func label(_ mode: Int) -> String { dflash ? ["Baseline", "Auto", "2 tokens", "7 tokens"][min(3, max(0, mode))] : "MTP\(mode)" }
    private func setMode(_ mode: Int) { if dflash { studio.setDFlashMode(mode) } else { studio.setMTPDepth(mode) } }
    private func tune() { if dflash { studio.tuneDFlash() } else { studio.tuneMTP() } }
    private func cancel() { if dflash { studio.cancelDFlashTuning() } else { studio.cancelMTPTuning() } }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if enabled {
                VStack(alignment: .leading, spacing: 8) {
                    Text(L("Profondeur", "Depth"))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(StudioTheme.secondary)
                        .fixedSize(horizontal: true, vertical: false)
                    Picker(L("Profondeur", "Depth"), selection: Binding(
                        get: { depth }, set: { setMode($0) }
                    )) {
                        ForEach(1...(dflash ? 3 : studio.mtpMaxDepth), id: \.self) { depth in
                            Text(dflash ? label(depth) : depth == 1 ? L("1 token", "1 token") : L("\(depth) tokens", "\(depth) tokens")).tag(depth)
                        }
                    }
                    .labelsHidden()
                    .pickerStyle(.segmented)
                    .controlSize(.small)
                    .frame(maxWidth: .infinity)
                    .accessibilityLabel(L("Profondeur", "Depth"))
                    .disabled(studio.isGenerating)
                }
            }
            if tuning {
                HStack {
                    Text(status).font(.system(size: 11, weight: .medium))
                    Spacer()
                    Button(L("Arrêter", "Stop"), action: cancel)
                        .buttonStyle(GlassPillButtonStyle())
                }
                ProgressView(value: progress).tint(StudioTheme.accent)
                Text(dflash ? "Baseline → Auto → 2 → 7 · ↔" : L("Baseline → MTP1 → MTP2 → MTP3 · puis ordre inversé", "Baseline → MTP1 → MTP2 → MTP3 · then reverse order"))
                    .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
            } else if !dflash || studio.selectedModel?.modelType != "qwen4_exp" || supported {
                Button(action: tune) {
                    Label(L("Comparer les vitesses", "Compare speeds"), systemImage: "speedometer")
                        .font(.system(size: 12, weight: .medium))
                        .padding(.horizontal, 12).frame(height: 32)
                }
                .buttonStyle(StudioToolbarStyle()).disabled(!canTune)
            }
            if !rows.isEmpty && !tuning {
                let winner = MTPTuning.winner(rows)
                let maxTPS = rows.map(\.decodeTPS).filter(\.isFinite).max() ?? 1
                VStack(spacing: 9) {
                    ForEach(rows) { row in
                        VStack(spacing: 4) {
                            HStack {
                                Text(dflash ? row.dflashLabel : row.label).font(.system(size: 10, weight: .semibold))
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
                                Capsule().fill(StudioTheme.contrast.opacity(0.05))
                                    .overlay(alignment: .leading) {
                                        Capsule().fill(row.depth == winner ? StudioTheme.accent : StudioTheme.contrast.opacity(0.22))
                                            .frame(width: geometry.size.width * max(0, min(1, row.decodeTPS / max(1, maxTPS))))
                                    }
                            }.frame(height: 3)
                            if let tie = row.nearTies?.first, row.reason == nil {
                                Text(L("Quasi-égalité acceptée · écart \(String(format: "%.3f", tie.gap))",
                                       "Near tie accepted · gap \(String(format: "%.3f", tie.gap))"))
                                    .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet).frame(maxWidth: .infinity, alignment: .leading)
                            }
                            if let reason = row.reason {
                                Text(reason == "zero_acceptance" ? L("Aucune proposition acceptée", "No proposals accepted")
                                     : reason == "target_mismatch" ? L("Sortie différente de la baseline", "Output differs from baseline")
                                     : L("Mesure insuffisante", "Insufficient measurement"))
                                    .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet).frame(maxWidth: .infinity, alignment: .leading)
                            }
                        }
                    }
                }
                .padding(12).background(StudioTheme.contrast.opacity(0.025), in: RoundedRectangle(cornerRadius: 10))
                if rows.contains(where: { $0.sampled == true }) {
                    Text(L("Mesuré avec tes réglages d’échantillonnage.", "Measured with your sampling settings."))
                        .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
                }
                Text(L("Réglage enregistré : ", "Saved setting: ") + (enabled ? label(depth) : "Baseline"))
                    .font(.system(size: 10, weight: .medium)).foregroundStyle(StudioTheme.accent)
                if winner == 0 {
                    Text(L("Baseline retenue : aucun gain \(dflash ? "DFlash2" : "MTP") supérieur à 3 % mesuré sur ce test. Le résultat peut varier selon le prompt et les conditions du Mac.", "Baseline retained: no \(dflash ? "DFlash2" : "MTP") gain above 3% measured in this test. Results can vary with the prompt and Mac conditions."))
                        .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
                }
            }
            Text(dflash && studio.selectedModel?.modelType == "qwen4_exp" && !supported
                 ? L("Les modes Auto, 2 et 7 restent sélectionnables. Tune n’est pas disponible pour Flash Next.", "Auto, 2 and 7 remain selectable. Tune is not available for Flash Next.")
                 : !dflash && studio.mtpBuiltIn
                 ? L("Flash Next utilise MTP1. Chaque proposition est vérifiée par le modèle cible.", "Flash Next uses MTP1. The target verifies every proposal.")
                 : supported
                 ? artifact.isEmpty
                    ? L("Active le mode une fois pour préparer le draft, puis lance Tune.", "Enable the mode once to prepare the draft, then run Tune.")
                    : L("Deux prompts locaux, sans outils. Le meilleur mode est enregistré pour ce modèle et ce Mac. Baseline si le gain ne dépasse pas 3 %.", "Two local prompts, without tools. The best mode is saved for this model and Mac. Baseline if the gain is within 3%.")
                 : L("Mets le moteur à jour puis recharge le modèle pour utiliser Tune.", "Update the engine and reload the model to use Tune."))
                .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
        }
    }
}
