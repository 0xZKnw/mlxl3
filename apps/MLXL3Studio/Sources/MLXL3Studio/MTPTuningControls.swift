import SwiftUI

/// Keep the transcript visible; opening detailed results is an explicit action.
struct MTPTuningToolbarControl: View {
    @EnvironmentObject private var studio: StudioModel
    var dflash = false
    private var tuning: Bool { dflash ? studio.isTuningDFlash : (studio.isTuningMTP && !studio.isTuningDFlash) }
    private var progress: Double { dflash ? studio.dflashTuneProgress : studio.mtpTuneProgress }
    private var status: String { dflash ? studio.dflashTuneStatus : studio.mtpTuneStatus }
    private var error: String? { dflash ? studio.dflashDownloadError : studio.mtpError }
    private var rows: [MTPTuningRow] { dflash ? studio.dflashTuneRows : studio.mtpTuneRows }
    private var canTune: Bool { dflash ? studio.canTuneDFlash : studio.canTuneMTP }
    private var title: String { dflash ? "Tune DFlash2" : "Tune MTP" }
    private func tune() { if dflash { studio.tuneDFlash() } else { studio.tuneMTP() } }
    private func cancel() { if dflash { studio.cancelDFlashTuning() } else { studio.cancelMTPTuning() } }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            Button {
                if tuning { cancel() }
                else { tune() }
            } label: {
                HStack(spacing: 6) {
                    if tuning {
                        ProgressView().progressViewStyle(.circular).controlSize(.mini)
                            .tint(StudioTheme.accent).frame(width: 12, height: 12)
                            .accessibilityHidden(true)
                    } else {
                        Image(systemName: "sparkles")
                    }
                    Text(title)
                    if tuning {
                        Text("\(Int(progress * 100)) %").monospacedDigit()
                        Image(systemName: "stop.fill").font(.system(size: 9))
                    }
                }
                .font(.system(size: 11, weight: .medium))
                .padding(.horizontal, 10).frame(height: 30)
            }
            .buttonStyle(StudioControlStyle(emphasized: true))
            .disabled(!tuning && !canTune)
            .help(tuning
                  ? status + L(" · Cliquer pour arrêter", " · Click to stop")
                  : L("Comparer les modes et enregistrer le meilleur réglage", "Compare modes and save the best setting"))
            .accessibilityLabel(tuning ? L("Arrêter : ", "Stop: ") + title : title)
            .accessibilityValue(tuning ? status + " · \(Int(progress * 100)) %" : "")

            if tuning {
                Text(status)
                    .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
                    .frame(maxWidth: 180, alignment: .leading).lineLimit(1)
            } else if let message = error {
                Button((dflash ? "DFlash2" : "MTP") + L(" · Détails", " · Details")) { studio.showInspector = true }
                    .buttonStyle(.plain).font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
                    .help(message)
            } else if let winner = MTPTuning.winner(rows) {
                Button(L("Résultat : ", "Result: ") + (winner == 0 ? "Baseline" : dflash ? ["", "Auto", "2 tokens", "7 tokens"][winner] : "MTP\(winner)") + L(" · Détails", " · Details")) {
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
    var dflash = false
    private var enabled: Bool { dflash ? studio.dflash2Enabled : studio.mtpEnabled }
    private var tuning: Bool { dflash ? studio.isTuningDFlash : (studio.isTuningMTP && !studio.isTuningDFlash) }
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
                Picker(L("Profondeur", "Depth"), selection: Binding(
                    get: { depth }, set: { setMode($0) }
                )) {
                    ForEach(1...(dflash ? 3 : studio.mtpMaxDepth), id: \.self) { depth in
                        Text(label(depth)).tag(depth)
                    }
                }
                .pickerStyle(.segmented).disabled(studio.isGenerating)
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
            } else {
                Button(action: tune) {
                    HStack {
                        Image(systemName: "slider.horizontal.3")
                        Text(dflash ? "Tune DFlash2" : "Tune MTP")
                        Spacer()
                        Image(systemName: "sparkles").foregroundStyle(StudioTheme.accent)
                    }.frame(height: 29)
                }
                .buttonStyle(GlassPillButtonStyle()).disabled(!canTune)
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
                Text(L("Réglage enregistré : ", "Saved setting: ") + (enabled ? label(depth) : "Baseline"))
                    .font(.system(size: 10, weight: .medium)).foregroundStyle(StudioTheme.accent)
                if winner == 0 {
                    Text(L("Baseline retenue : aucun gain \(dflash ? "DFlash2" : "MTP") supérieur à 3 % mesuré sur ce test. Le résultat peut varier selon le prompt et les conditions du Mac.", "Baseline retained: no \(dflash ? "DFlash2" : "MTP") gain above 3% measured in this test. Results can vary with the prompt and Mac conditions."))
                        .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
                }
            }
            Text(supported
                 ? artifact.isEmpty
                    ? L("Active le mode une fois pour préparer le draft, puis lance Tune.", "Enable the mode once to prepare the draft, then run Tune.")
                    : L("Deux prompts locaux, sans outils. Le meilleur mode est enregistré pour ce modèle et ce Mac. Baseline si le gain ne dépasse pas 3 %.", "Two local prompts, without tools. The best mode is saved for this model and Mac. Baseline if the gain is within 3%.")
                 : L("Mets le moteur à jour puis recharge le modèle pour utiliser Tune.", "Update the engine and reload the model to use Tune."))
                .font(.system(size: 9.5)).foregroundStyle(StudioTheme.quiet)
        }
    }
}
