import SwiftUI

struct GenerationInspector: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Génération")
                        .font(.system(size: 15, weight: .semibold))
                    Text("PARAMÈTRES DU MODÈLE")
                        .font(.system(size: 8, weight: .bold, design: .rounded))
                        .tracking(1.25)
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
            .padding(.top, 42)
            .padding(.horizontal, 18)
            .padding(.bottom, 22)

            ScrollView {
                VStack(spacing: 14) {
                    SettingCard(title: "Échantillonnage", icon: "dial.medium") {
                        SliderSetting(
                            label: "Température",
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
                            label: "Répétition",
                            value: $studio.repetitionPenalty,
                            range: 1...1.3,
                            format: "%.2f"
                        )
                    }

                    SettingCard(title: "Instruction système", icon: "command") {
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
                                    Text("Comportement optionnel du modèle…")
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
                                Text("\(studio.mcpServerCount) serveur\(studio.mcpServerCount == 1 ? "" : "s") connecté\(studio.mcpServerCount == 1 ? "" : "s")")
                                    .font(.system(size: 11, weight: .semibold))
                                Text("\(studio.mcpToolCount) outil\(studio.mcpToolCount == 1 ? "" : "s") disponible\(studio.mcpToolCount == 1 ? "" : "s")")
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
                                .font(.system(size: 9, weight: .medium, design: .rounded))
                                .foregroundStyle(Color(red: 1, green: 0.56, blue: 0.56))
                                .lineLimit(3)
                        }

                        HStack(spacing: 8) {
                            Button(action: studio.openMCPConfiguration) {
                                Label("Configurer", systemImage: "doc.text")
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
                            .help("Recharger les serveurs MCP")
                        }

                        Text("Les outils des serveurs activés sont proposés au modèle et exécutés localement.")
                            .font(.system(size: 8.5, weight: .medium))
                            .foregroundStyle(StudioTheme.quiet)
                            .fixedSize(horizontal: false, vertical: true)
                    }

                    if let model = studio.selectedModel {
                        SettingCard(title: "Modèle actif", icon: "cpu") {
                            InspectorValue(label: "Architecture", value: model.architectureLabel)
                            InspectorValue(label: "Quantification", value: model.bits.map { String(format: "EXL3 %.2f BPW", $0) } ?? "EXL3")
                            InspectorValue(label: "Poids", value: model.size)
                            InspectorValue(label: "Modules", value: "\(model.modules)")
                        }
                    }
                }
                .padding(.horizontal, 13)
                .padding(.bottom, 20)
            }
        }
        .background(StudioTheme.sidebar.opacity(0.95))
        .onChange(of: studio.temperature) { studio.settingsDidChange() }
        .onChange(of: studio.topK) { studio.settingsDidChange() }
        .onChange(of: studio.repetitionPenalty) { studio.settingsDidChange() }
        .onChange(of: studio.systemPrompt) { studio.settingsDidChange() }
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
            Label(title, systemImage: icon)
                .font(.system(size: 10, weight: .bold, design: .rounded))
                .foregroundStyle(Color.white.opacity(0.78))
            content
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 15, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 15, style: .continuous)
                .stroke(Color.white.opacity(0.075), lineWidth: 0.7)
        }
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
        .font(.system(size: 10, weight: .medium, design: .rounded))
    }
}
