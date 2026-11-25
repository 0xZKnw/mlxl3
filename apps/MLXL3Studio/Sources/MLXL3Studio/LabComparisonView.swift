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
        VStack(alignment: .leading, spacing: 24) {
            if needsSetup {
                if models.count < 2 {
                    VStack(alignment: .leading, spacing: 12) {
                        Text(L("Deux modèles de texte sont nécessaires.", "Two text models are required."))
                            .font(.headline)
                        Text(L("Choisis deux modèles compatibles dans la bibliothèque pour comparer leurs réponses sur ce Mac.",
                               "Add two compatible text models to your library to compare their answers on this Mac."))
                            .foregroundStyle(StudioTheme.secondary)
                        Button(action: studio.openModelManager) {
                            Label(L("Ouvrir les modèles", "Open Models"), systemImage: "square.stack")
                                .padding(.horizontal, 10).frame(height: 30)
                        }.buttonStyle(StudioToolbarStyle())
                    }.accessibilityIdentifier("lab.compare.empty")
                } else {
                    setup
                }
            } else {
                submittedPrompt
            }

            if !comparison.runs.isEmpty && !editing {
                let layout = width >= 600 ? AnyLayout(HStackLayout(alignment: .top, spacing: 18))
                    : AnyLayout(VStackLayout(alignment: .leading, spacing: 18))
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
        VStack(alignment: .leading, spacing: 18) {
            let pair = width >= 520 ? AnyLayout(HStackLayout(spacing: 16)) : AnyLayout(VStackLayout(spacing: 10))
            pair {
                modelMenu(first: true)
                modelMenu(first: false)
            }
            VStack(alignment: .leading, spacing: 9) {
                HStack {
                    Text(L("Prompt commun", "Shared prompt")).font(.headline)
                    Spacer()
                    if draft.prompt.isEmpty {
                        Button(L("Essayer un exemple", "Try an example")) {
                            draft.prompt = L("Explique pourquoi la Lune présente toujours la même face à la Terre, en trois phrases.",
                                             "In three sentences, explain why the Moon always shows the same face to Earth.")
                        }.buttonStyle(.link)
                    }
                }
                TextField(L("Pose ta question aux deux modèles…", "Ask both models a question…"),
                          text: $draft.prompt, axis: .vertical)
                    .font(.body).textFieldStyle(.plain).lineLimit(3...8)
                    .focused($writingPrompt)
                    .accessibilityLabel(L("Prompt commun", "Shared prompt"))
                    .accessibilityIdentifier("lab.compare.prompt")
            }
            .padding(16)
            .background(StudioTheme.composer, in: RoundedRectangle(cornerRadius: 11))
            .overlay { RoundedRectangle(cornerRadius: 11).strokeBorder(writingPrompt ? StudioTheme.quiet : StudioTheme.edge) }

            ViewThatFits(in: .horizontal) {
                HStack(spacing: 14) { budgetPicker; Spacer(minLength: 12); startButton }
                VStack(alignment: .leading, spacing: 12) { budgetPicker; startButton }
            }
            Text(L("Les modèles répondent l’un après l’autre. Le raisonnement compte dans la limite de sortie.",
                   "The models run one at a time. Reasoning tokens count toward the output limit."))
                .font(.caption).foregroundStyle(StudioTheme.secondary)
            if draft.prompt.utf8.count > 65_536 {
                Text(L("Le prompt dépasse la limite de 64 Kio.", "The prompt exceeds the 64 KiB limit."))
                    .font(.callout).foregroundStyle(StudioTheme.failure)
            }
            if let reason = studio.comparisonUnavailability {
                Text(reason).font(.callout).foregroundStyle(StudioTheme.secondary)
            }
            if editing && !comparison.runs.isEmpty {
                Button(L("Revenir aux réponses", "Back to answers")) { editing = false }
                    .buttonStyle(.link)
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
                    Text(label).font(.caption).foregroundStyle(StudioTheme.secondary)
                    Text(comparisonLabel(selected, among: models.map(\.name))).font(.body.weight(.medium)).lineLimit(1).truncationMode(.middle)
                }
                Spacer(minLength: 3)
                Image(systemName: "chevron.up.chevron.down").font(.caption2)
            }.padding(.horizontal, 13).frame(height: 54).contentShape(Rectangle())
        }
        .menuStyle(.button).menuIndicator(.hidden).buttonStyle(StudioToolbarStyle())
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
                .font(.body.weight(.semibold)).padding(.horizontal, 13).frame(height: 33)
        }
        .buttonStyle(PrimaryGlassButtonStyle()).disabled(!canStart)
        .keyboardShortcut(.return, modifiers: [.command])
        .accessibilityIdentifier("lab.compare.start")
    }

    private var submittedPrompt: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label(L("Prompt commun", "Shared prompt"), systemImage: "text.quote")
                    .font(.headline)
                Spacer()
                if comparison.isRunning {
                    Button(L("Arrêter", "Stop")) { comparison.cancel() }.buttonStyle(.link)
                        .accessibilityIdentifier("lab.compare.stop")
                } else {
                    Button(L("Modifier", "Edit")) { editing = true }.buttonStyle(.link)
                        .accessibilityIdentifier("lab.compare.edit")
                }
            }
            let prompt = comparison.prompt
            let needsDisclosure = prompt.count > 220 || prompt.prefix(220).filter(\.isNewline).count > 2
            Text(prompt).font(.body).textSelection(.enabled)
                .lineLimit(needsDisclosure && !showingFullPrompt ? 3 : nil)
            if needsDisclosure {
                Button(showingFullPrompt ? L("Réduire le prompt", "Collapse prompt") : L("Voir le prompt entier", "Show full prompt")) {
                    showingFullPrompt.toggle()
                }.buttonStyle(.link)
            }
            Text(L("Jusqu’à \(comparison.budget) tokens par modèle", "Up to \(comparison.budget) tokens per model"))
                .font(.caption).foregroundStyle(StudioTheme.secondary)
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 11))
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
            Text(title).font(.title3.weight(.semibold))
                .lineLimit(2).help(run.model.name).accessibilityAddTraits(.isHeader)
            HStack(alignment: .top, spacing: 12) {
                reading(L("Premier texte", "First text"), seconds: run.firstTextSeconds)
                    .help(L("Délai jusqu’au premier texte visible, après le chargement du modèle.",
                            "Time to the first visible answer text after the model has loaded."))
                VStack(alignment: .leading, spacing: 3) {
                    Text(L("Décodage", "Decode")).font(.caption).foregroundStyle(StudioTheme.secondary)
                    Text(run.decodeTps.map { number($0, precision: 1) + " tok/s" } ?? "—")
                        .font(.headline.monospacedDigit())
                        .foregroundStyle(run.decodeTps == nil ? StudioTheme.secondary : LabWorkspaceView.outputColor)
                }.frame(maxWidth: .infinity, alignment: .leading).accessibilityElement(children: .combine)
            }
            Divider()
            if !run.answer.isEmpty {
                MarkdownResponseView(run.answer, streaming: run.phase == .generating, htmlPreviewEnabled: false)
                    .equatable().frame(maxWidth: .infinity, alignment: .leading)
            }
            if let error = run.error {
                Text(error).font(.callout).foregroundStyle(StudioTheme.failure).textSelection(.enabled)
            }
            if run.phase == .complete {
                let details = [run.generatedTokens.map { L("\($0) tokens générés", "\($0) tokens generated") },
                               run.elapsedSeconds.map(duration),
                               run.hitTokenLimit ? L("limite atteinte", "limit reached") : nil]
                Text(details.compactMap { $0 }.joined(separator: " · "))
                    .font(.caption).foregroundStyle(StudioTheme.secondary)
                if run.answer.isEmpty {
                    Text(L("Le modèle a atteint la limite avant de répondre. Essaie un budget plus élevé.",
                           "The model reached the limit before answering. Try a larger output budget."))
                        .font(.callout).foregroundStyle(StudioTheme.secondary)
                }
            } else {
                HStack(spacing: 7) {
                    if run.phase == .loading || run.phase == .generating { ProgressView().controlSize(.small) }
                    Text(status)
                }.font(.callout).foregroundStyle(StudioTheme.secondary)
            }
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .topLeading)
        .background(StudioTheme.composer, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).strokeBorder(StudioTheme.edge) }
    }

    private func reading(_ title: String, seconds: Double?) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.caption).foregroundStyle(StudioTheme.secondary)
            Text(seconds.map(duration) ?? "—").font(.headline.monospacedDigit())
        }.frame(maxWidth: .infinity, alignment: .leading).accessibilityElement(children: .combine)
    }

    private func duration(_ seconds: Double) -> String {
        seconds < 1 ? number(seconds * 1_000, precision: 0) + " ms" : number(seconds, precision: 2) + " s"
    }

    private func number(_ value: Double, precision: Int) -> String {
        value.formatted(.number.locale(locale).precision(.fractionLength(0...precision)))
    }
}
