import SwiftUI

struct LabComparisonDraft {
    var left = ""
    var right = ""
    var prompt = ""
    var budget = 1024
}

// The Hub appends eight hex digits to installation names. Keep the full name when two
// registered installations would otherwise have the same label.
private func comparisonLabel(_ name: String, among names: [String]) -> String {
    func short(_ text: String) -> String {
        guard let dash = text.lastIndex(of: "-") else { return text }
        let suffix = text[text.index(after: dash)...]
        guard suffix.count == 8, suffix.allSatisfy({ $0.isHexDigit }) else { return text }
        return String(text[..<dash])
    }
    let label = short(name)
    return names.contains(where: { $0 != name && short($0) == label }) ? name : label
}

struct LabComparisonView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var comparison: LabComparison
    @Binding var draft: LabComparisonDraft
    let width: CGFloat
    @State private var editing = false
    @State private var showingFullPrompt = false
    @FocusState private var writingPrompt: Bool

    private var models: [LocalModel] { studio.comparisonModels }
    private var needsSetup: Bool { comparison.runs.isEmpty || editing }
    private var canStart: Bool {
        !comparison.isRunning && studio.comparisonUnavailability == nil &&
            models.contains { $0.name == draft.left } && models.contains { $0.name == draft.right } &&
            draft.left != draft.right && LabComparison.validPrompt(draft.prompt)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            if needsSetup {
                if models.count < 2 {
                    StudioEmptyState(
                        icon: "rectangle.split.2x1",
                        title: L("Deux modèles de texte sont nécessaires.", "Two text models are required."),
                        detail: L("Choisis deux modèles compatibles dans la bibliothèque pour comparer leurs réponses sur ce Mac.",
                                  "Add two compatible text models to your library to compare their answers on this Mac.")
                    ) {
                        Button(action: studio.openModelManager) {
                            Label(L("Ouvrir les modèles", "Open Models"), systemImage: "square.stack")
                        }.buttonStyle(.studio(.primary))
                    }
                    .studioCard(padding: 0)
                    .accessibilityIdentifier("lab.compare.empty")
                } else {
                    setup
                }
            } else {
                submittedPrompt
            }

            if !comparison.runs.isEmpty && !editing {
                let layout = width >= 600 ? AnyLayout(HStackLayout(alignment: .top, spacing: 12))
                    : AnyLayout(VStackLayout(alignment: .leading, spacing: 12))
                layout {
                    ForEach(comparison.runs, id: \.model.name) { run in
                        LabComparisonAnswer(run: run, title: comparisonLabel(run.model.name, among: comparison.runs.map { $0.model.name }))
                            .frame(maxWidth: .infinity, alignment: .topLeading)
                    }
                }
            }
        }
        .onChange(of: models.map(\.name), initial: true) { _, names in
            if draft.left.isEmpty && draft.right.isEmpty, let first = comparison.runs.first,
               let last = comparison.runs.last {
                draft.left = first.model.name
                draft.right = last.model.name
                draft.prompt = comparison.prompt
                draft.budget = comparison.budget
            }
            if !names.contains(draft.left) {
                draft.left = names.first(where: { $0 != draft.right }) ?? names.first ?? ""
            }
            if !names.contains(draft.right) || draft.left == draft.right {
                draft.right = names.first(where: { $0 != draft.left }) ?? ""
            }
        }
    }

    private var setup: some View {
        VStack(alignment: .leading, spacing: 14) {
            let pair = width >= 520 ? AnyLayout(HStackLayout(spacing: 12)) : AnyLayout(VStackLayout(spacing: 10))
            pair {
                modelMenu(first: true)
                modelMenu(first: false)
            }
            VStack(alignment: .leading, spacing: 9) {
                HStack {
                    StudioSectionTitle(title: L("Prompt commun", "Shared prompt"))
                    Spacer()
                    if draft.prompt.isEmpty {
                        Button(L("Essayer un exemple", "Try an example")) {
                            draft.prompt = L("Explique pourquoi la Lune présente toujours la même face à la Terre, en trois phrases.",
                                             "In three sentences, explain why the Moon always shows the same face to Earth.")
                        }.buttonStyle(.studio(.quiet, height: 24))
                    }
                }
                TextField(L("Pose ta question aux deux modèles…", "Ask both models a question…"),
                          text: $draft.prompt, axis: .vertical)
                    .font(.system(size: 14)).textFieldStyle(.plain).lineLimit(3...8)
                    .focused($writingPrompt)
                    .accessibilityLabel(L("Prompt commun", "Shared prompt"))
                    .accessibilityIdentifier("lab.compare.prompt")
            }
            .padding(14)
            .background(StudioTheme.composer.opacity(0.9), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
            .overlay {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .strokeBorder(writingPrompt ? StudioTheme.contrast.opacity(0.22) : StudioTheme.edge, lineWidth: writingPrompt ? 1 : 0.5)
                    .allowsHitTesting(false)
            }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 14) { budgetPicker; Spacer(minLength: 12); startButton }
                VStack(alignment: .leading, spacing: 12) { budgetPicker; startButton }
            }
            Text(L("Les modèles répondent l’un après l’autre. Le raisonnement compte dans la limite de sortie.",
                   "The models run one at a time. Reasoning tokens count toward the output limit."))
                .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
            if draft.prompt.utf8.count > 65_536 {
                StudioNotice(kind: .error, text: L("Le prompt dépasse la limite de 64 Kio.", "The prompt exceeds the 64 KiB limit."))
            }
            if let reason = studio.comparisonUnavailability {
                StudioNotice(text: reason)
            }
            if editing && !comparison.runs.isEmpty {
                Button(L("Revenir aux réponses", "Back to answers")) { editing = false }
                    .buttonStyle(.studio)
            }
        }
    }

    private func modelMenu(first: Bool) -> some View {
        let selected = first ? draft.left : draft.right
        let label = first ? L("Premier modèle", "First model") : L("Second modèle", "Second model")
        return Menu {
            ForEach(models) { model in
                Button {
                    if first {
                        if draft.right == model.name { draft.right = draft.left }
                        draft.left = model.name
                    } else {
                        if draft.left == model.name { draft.left = draft.right }
                        draft.right = model.name
                    }
                } label: {
                    if selected == model.name { Label(model.name, systemImage: "checkmark") }
                    else { Text(model.name) }
                }
            }
        } label: {
            HStack(spacing: 12) {
                Image(systemName: first ? "rectangle.lefthalf.inset.filled" : "rectangle.righthalf.inset.filled")
                    .font(.system(size: 17)).foregroundStyle(StudioTheme.secondary)
                VStack(alignment: .leading, spacing: 3) {
                    Text(label).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                    Text(comparisonLabel(selected, among: models.map(\.name))).font(.system(size: 13.5, weight: .medium))
                        .foregroundStyle(StudioTheme.ink).lineLimit(1).truncationMode(.middle)
                }
                Spacer(minLength: 3)
                Image(systemName: "chevron.up.chevron.down").font(.system(size: 9, weight: .semibold))
                    .foregroundStyle(StudioTheme.quiet)
            }.padding(.horizontal, 13).frame(height: 54).frame(maxWidth: .infinity, alignment: .leading)
                .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: StudioPageMetrics.card, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: StudioPageMetrics.card, style: .continuous)
                        .strokeBorder(StudioTheme.edge.opacity(0.7), lineWidth: 0.5).allowsHitTesting(false)
                }
                .contentShape(Rectangle())
        }
        .menuStyle(.button).menuIndicator(.hidden).buttonStyle(StudioRowStyle(radius: StudioPageMetrics.card))
        .frame(maxWidth: .infinity, alignment: .leading)
        .help(selected).accessibilityLabel(label).accessibilityValue(selected)
        .accessibilityIdentifier(first ? "lab.compare.left" : "lab.compare.right")
    }

    private var budgetPicker: some View {
        HStack(spacing: 6) {
            Text(L("Limite de sortie", "Output limit")).foregroundStyle(StudioTheme.secondary)
            Picker(L("Limite de sortie", "Output limit"), selection: $draft.budget) {
                ForEach([128, 256, 512, 1024, 2048, 4096], id: \.self) { Text("\($0) tokens").tag($0) }
            }.labelsHidden().fixedSize()
        }
        .help(L("Chaque modèle peut terminer sa réponse avant cette limite.", "Either model can finish its answer before this limit."))
    }

    private var startButton: some View {
        Button {
            writingPrompt = false
            studio.startLabComparison(left: draft.left, right: draft.right, prompt: draft.prompt, budget: draft.budget)
            if comparison.isRunning {
                editing = false
                showingFullPrompt = false
            }
        } label: {
            Label(L("Comparer", "Compare"), systemImage: "play.fill")
        }
        .buttonStyle(.studio(.primary, height: 34)).disabled(!canStart)
        .help(L("Comparer · ⌘↩", "Compare · ⌘↩"))
        .keyboardShortcut(.return, modifiers: [.command])
        .accessibilityIdentifier("lab.compare.start")
    }

    private var submittedPrompt: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                StudioSectionTitle(title: L("Prompt commun", "Shared prompt"), icon: "text.quote")
                Spacer()
                if comparison.isRunning {
                    Button(L("Arrêter", "Stop")) { comparison.cancel() }.buttonStyle(.studio(.secondary, height: 26))
                        .accessibilityIdentifier("lab.compare.stop")
                } else {
                    Button(L("Modifier", "Edit")) { editing = true }.buttonStyle(.studio(.secondary, height: 26))
                        .accessibilityIdentifier("lab.compare.edit")
                }
            }
            let prompt = comparison.prompt
            let needsDisclosure = prompt.count > 220 || prompt.prefix(220).filter(\.isNewline).count > 2
            Text(prompt).font(.system(size: 13.5)).textSelection(.enabled)
                .lineLimit(needsDisclosure && !showingFullPrompt ? 3 : nil)
            if needsDisclosure {
                Button(showingFullPrompt ? L("Réduire le prompt", "Collapse prompt") : L("Voir le prompt entier", "Show full prompt")) {
                    showingFullPrompt.toggle()
                }.buttonStyle(.studio(.quiet, height: 24))
            }
            Text(L("Jusqu’à \(comparison.budget) tokens par modèle", "Up to \(comparison.budget) tokens per model"))
                .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
        }
        .studioCard(padding: 16)
        .accessibilityIdentifier("lab.compare.submitted")
    }
}

private struct LabComparisonAnswer: View {
    let run: LabComparison.Run
    let title: String
    @Environment(\.locale) private var locale

    private var status: String {
        switch run.phase {
        case .waiting: L("En attente", "Waiting")
        case .loading: L("Chargement du modèle…", "Loading model…")
        case .generating: run.isThinking ? L("Réflexion…", "Thinking…") : L("Génération…", "Generating…")
        case .complete: L("Terminée", "Complete")
        case .failed: L("Échec", "Failed")
        case .cancelled: L("Arrêtée", "Stopped")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Text(title).font(.system(size: 15, weight: .semibold))
                    .lineLimit(2).help(run.model.name).accessibilityAddTraits(.isHeader)
                Spacer(minLength: 6)
                if run.phase != .complete {
                    StudioStatusPill(text: status,
                                     color: run.phase == .failed ? StudioTheme.failure : StudioTheme.quiet,
                                     busy: run.phase == .loading || run.phase == .generating)
                }
            }
            HStack(alignment: .top, spacing: 12) {
                reading(L("Premier texte", "First text"), seconds: run.firstTextSeconds)
                    .help(L("Délai jusqu’au premier texte visible, après le chargement du modèle.",
                            "Time to the first visible answer text after the model has loaded."))
                VStack(alignment: .leading, spacing: 3) {
                    Text(L("Décodage", "Decode")).font(.system(size: 12, weight: .medium)).foregroundStyle(StudioTheme.secondary)
                    Text(run.decodeTps.map { number($0, precision: 1) + " tok/s" } ?? "—")
                        .font(.system(size: 17, weight: .semibold)).monospacedDigit()
                        .foregroundStyle(run.decodeTps == nil ? StudioTheme.secondary : LabWorkspaceView.outputColor)
                }.frame(maxWidth: .infinity, alignment: .leading).accessibilityElement(children: .combine)
            }
            Rectangle().fill(StudioTheme.edge).frame(height: 0.5).accessibilityHidden(true)
            if !run.answer.isEmpty {
                MarkdownResponseView(run.answer, streaming: run.phase == .generating, htmlPreviewEnabled: false)
                    .equatable().frame(maxWidth: .infinity, alignment: .leading)
            }
            if let error = run.error {
                StudioNotice(kind: .error, text: error)
            }
            if run.phase == .complete {
                let details = [run.generatedTokens.map { L("\($0) tokens générés", "\($0) tokens generated") },
                               run.elapsedSeconds.map(duration),
                               run.hitTokenLimit ? L("limite atteinte", "limit reached") : nil]
                Text(details.compactMap { $0 }.joined(separator: " · "))
                    .font(.system(size: 12)).monospacedDigit().foregroundStyle(StudioTheme.quiet)
                if run.answer.isEmpty {
                    StudioNotice(text: L("Le modèle a atteint la limite avant de répondre. Essaie un budget plus élevé.",
                                         "The model reached the limit before answering. Try a larger output budget."))
                }
            }
        }
        .studioCard(padding: 16)
    }

    private func reading(_ title: String, seconds: Double?) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.system(size: 12, weight: .medium)).foregroundStyle(StudioTheme.secondary)
            Text(seconds.map(duration) ?? "—").font(.system(size: 17, weight: .semibold)).monospacedDigit()
        }.frame(maxWidth: .infinity, alignment: .leading).accessibilityElement(children: .combine)
    }

    private func duration(_ seconds: Double) -> String {
        seconds < 1 ? number(seconds * 1_000, precision: 0) + " ms" : number(seconds, precision: 2) + " s"
    }

    private func number(_ value: Double, precision: Int) -> String {
        value.formatted(.number.locale(locale).precision(.fractionLength(0...precision)))
    }
}
