import SwiftUI

struct GenerationInspector: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(L("Génération", "Generation"))
                        .font(.system(size: 23, weight: .regular, design: .serif))
                    Text(L("Paramètres du modèle", "Model settings"))
                        .font(.system(size: 11))
                        .foregroundStyle(StudioTheme.quiet)
                }
                Spacer()
                Button { studio.showInspector = false } label: {
                    Image(systemName: "xmark")
                        .font(.system(size: 10, weight: .bold))
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(RoundGlassButtonStyle())
            }
            .padding(.top, 25)
            .padding(.horizontal, 18)
            .padding(.bottom, 22)

            ScrollView {
                VStack(spacing: 28) {
                    SettingCard(title: L("Contexte", "Context"), icon: "chart.bar") {
                        Text(studio.contextLabel + " tokens")
                            .font(.system(size: 12, weight: .medium)).monospacedDigit()
                        if let limit = studio.activeContextLimit, limit > 0 {
                            ProgressView(value: min(Double(studio.contextUsed ?? 0) / Double(limit), 1))
                                .tint(StudioTheme.accent)
                        }
                        TextField(L("Limite en tokens (0 = auto)", "Token limit (0 = auto)"),
                                  value: $studio.contextLengthDraft, format: .number.grouping(.never))
                            .textFieldStyle(.roundedBorder)
                        .disabled(studio.isGenerating || studio.modelContextLimit == nil)
                        if let cache = studio.draftContextBytes, let model = studio.modelResidentBytes {
                            VStack(alignment: .leading, spacing: 7) {
                                HStack {
                                    Text(L("Modèle + contexte", "Model + context"))
                                    Spacer()
                                    Text(memory(model + cache)).monospacedDigit()
                                }
                                .font(.system(size: 12, weight: .semibold))
                                Text(L("Modèle \(memory(model)) + cache \(memory(cache))", "Model \(memory(model)) + cache \(memory(cache))"))
                                    .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                                    .monospacedDigit()
                                Text(L("Estimation à contexte plein · 1 conversation. Hors buffers de calcul, autres caches et macOS.", "Estimate at full context · 1 conversation. Excludes compute buffers, other caches and macOS."))
                                    .font(.caption).foregroundStyle(StudioTheme.quiet)
                                if model + cache > Double(ProcessInfo.processInfo.physicalMemory) * 0.85 {
                                    Label(L("Peu de marge mémoire pour le calcul et macOS.", "Little memory headroom for compute and macOS."), systemImage: "exclamationmark.triangle")
                                        .font(.caption).foregroundStyle(.orange)
                                }
                            }
                            .padding(10)
                            .background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                        } else {
                            Text(L("Estimation mémoire indisponible", "Memory estimate unavailable"))
                                .font(.caption).foregroundStyle(StudioTheme.quiet)
                        }
                        Menu(L("Préréglages", "Presets")) {
                            Button(L("Automatique", "Automatic")) { studio.contextLengthDraft = 0 }
                            ForEach([2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144].filter {
                                $0 <= (studio.modelContextLimit ?? 0)
                            }, id: \.self) { length in
                                Button(length.formatted()) { studio.contextLengthDraft = length }
                            }
                        }
                        .disabled(studio.isGenerating || studio.modelContextLimit == nil)
                        if let maximum = studio.modelContextLimit {
                            Text(L("Maximum du modèle : \(maximum.formatted()) tokens", "Model maximum: \(maximum.formatted()) tokens"))
                                .font(.caption).foregroundStyle(StudioTheme.quiet)
                        }
                        Button(action: studio.saveContextAndReload) {
                            Text(L("Enregistrer et recharger le modèle", "Save and reload model"))
                                .font(.system(size: 11, weight: .medium))
                                .frame(maxWidth: .infinity).padding(.vertical, 10)
                        }
                        .buttonStyle(GlassPillButtonStyle())
                        .disabled(!studio.canSaveContext)
                        Text(L("0 = maximum du modèle. Le rechargement libère le cache, pas l’historique. La mémoire augmente avec le contexte réellement utilisé.", "0 = model maximum. Reloading clears the cache, not your history. Memory grows with the context actually used."))
                            .font(.caption).foregroundStyle(StudioTheme.quiet)
                    }
                    SettingCard(title: L("Échantillonnage", "Sampling"), icon: "dial.medium") {
                        SliderSetting(
                            label: L("Température", "Temperature"),
                            value: $studio.temperature,
                            range: 0...1.5,
                            format: "%.2f"
                        )
                        Divider().overlay(Color.white.opacity(0.08))
                        VStack(spacing: 9) {
                            HStack {
                                Text("Top K")
                                Spacer()
                                Text("\(studio.topK)").monospacedDigit()
                            }
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(StudioTheme.secondary)
                            Slider(
                                value: Binding(
                                    get: { Double(studio.topK) },
                                    set: { studio.topK = Int($0.rounded()) }
                                ),
                                in: 0...200,
                                step: 1
                            )
                            .tint(Color.white.opacity(0.88))
                        }
                        Divider().overlay(Color.white.opacity(0.08))
                        SliderSetting(
                            label: L("Répétition", "Repetition"),
                            value: $studio.repetitionPenalty,
                            range: 1...1.3,
                            format: "%.2f"
                        )
                    }

                    SettingCard(title: L("Instruction système", "System instruction"), icon: "command") {
                        TextEditor(text: $studio.systemPrompt)
                            .font(.system(size: 11.5))
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: 88)
                            .padding(9)
                            .background(Color.black.opacity(0.22), in: RoundedRectangle(cornerRadius: 10))
                            .overlay {
                                RoundedRectangle(cornerRadius: 10)
                                    .stroke(Color.white.opacity(0.07), lineWidth: 0.7)
                            }
                            .overlay(alignment: .topLeading) {
                                if studio.systemPrompt.isEmpty {
                                    Text(L("Comportement optionnel du modèle…", "Optional model behavior…"))
                                        .font(.system(size: 11.5))
                                        .foregroundStyle(StudioTheme.quiet)
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 15)
                                        .allowsHitTesting(false)
                                }
                            }
                    }

                    SettingCard(title: "MCP", icon: "shippingbox") {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(L("\(studio.mcpServerCount) serveur\(studio.mcpServerCount == 1 ? "" : "s") connecté\(studio.mcpServerCount == 1 ? "" : "s")", "\(studio.mcpServerCount) connected server\(studio.mcpServerCount == 1 ? "" : "s")"))
                                    .font(.system(size: 11, weight: .semibold))
                                Text(L("\(studio.mcpToolCount) outil\(studio.mcpToolCount == 1 ? "" : "s") disponible\(studio.mcpToolCount == 1 ? "" : "s")", "\(studio.mcpToolCount) available tool\(studio.mcpToolCount == 1 ? "" : "s")"))
                                    .font(.system(size: 9.5, weight: .medium))
                                    .foregroundStyle(StudioTheme.quiet)
                            }
                            Spacer()
                            Circle()
                                .fill(studio.mcpServerCount > 0 ? Color.green : StudioTheme.quiet)
                                .frame(width: 7, height: 7)
                        }

                        if let error = studio.mcpErrors.sorted(by: { $0.key < $1.key }).first {
                            Text("\(error.key) · \(error.value)")
                                .font(.system(size: 9, weight: .medium))
                                .foregroundStyle(Color(red: 1, green: 0.56, blue: 0.56))
                                .lineLimit(3)
                        }

                        HStack(spacing: 8) {
                            Button(action: studio.openMCPConfiguration) {
                                Label(L("Configurer", "Configure"), systemImage: "doc.text")
                                    .frame(maxWidth: .infinity)
                                    .frame(height: 29)
                            }
                            .buttonStyle(GlassPillButtonStyle())

                            Button(action: studio.reloadMCPServers) {
                                Image(systemName: "arrow.clockwise")
                                    .frame(width: 29, height: 29)
                            }
                            .buttonStyle(RoundGlassButtonStyle())
                            .disabled(studio.isGenerating)
                            .help(L("Recharger les serveurs MCP", "Reload MCP servers"))
                        }

                        Text(L("Les outils des serveurs activés sont proposés au modèle et exécutés localement.", "Tools from enabled servers are offered to the model and executed locally."))
                            .font(.system(size: 8.5, weight: .medium))
                            .foregroundStyle(StudioTheme.quiet)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if let model = studio.selectedModel {
                        SettingCard(title: L("Modèle actif", "Active model"), icon: "cpu") {
                            InspectorValue(label: "Architecture", value: model.architectureLabel)
                            InspectorValue(label: L("Quantification", "Quantization"), value: model.bits.map { String(format: "EXL3 %.2f BPW", $0) } ?? "EXL3")
                            InspectorValue(label: L("Poids", "Weights"), value: model.size)
                            InspectorValue(label: "Modules", value: "\(model.modules)")
                        }
                    }
                }
                .padding(.horizontal, 13)
                .padding(.bottom, 20)
            }
        }
        .background(StudioTheme.sidebar)
        .onChange(of: studio.temperature) { studio.settingsDidChange() }
        .onChange(of: studio.topK) { studio.settingsDidChange() }
        .onChange(of: studio.repetitionPenalty) { studio.settingsDidChange() }
        .onChange(of: studio.systemPrompt) { studio.settingsDidChange() }
    }

    private func memory(_ bytes: Double) -> String {
        String(format: "%.2f %@", bytes / 1e9, L("Go", "GB"))
    }
}

private struct SettingCard<Content: View>: View {
    let title: String
    let icon: String
    @ViewBuilder let content: Content

    init(title: String, icon: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.icon = icon
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            Text(title)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(StudioTheme.ink)
            content
        }
        .padding(.horizontal, 8)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct SliderSetting: View {
    let label: String
    @Binding var value: Double
    let range: ClosedRange<Double>
    let format: String

    var body: some View {
        VStack(spacing: 9) {
            HStack {
                Text(label)
                Spacer()
                Text(String(format: format, value)).monospacedDigit()
            }
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(StudioTheme.secondary)
            Slider(value: $value, in: range)
                .tint(Color.white.opacity(0.88))
        }
    }
}

private struct InspectorValue: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .foregroundStyle(StudioTheme.quiet)
            Spacer()
            Text(value)
                .foregroundStyle(Color.white.opacity(0.78))
                .multilineTextAlignment(.trailing)
        }
        .font(.system(size: 10, weight: .medium))
    }
}
