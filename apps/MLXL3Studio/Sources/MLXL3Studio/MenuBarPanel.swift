import AppKit
import SwiftUI

struct MenuBarPanel: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            let engine = studio.engineResidentMemoryBytes
            let interface = studio.interfaceResidentMemoryBytes
            VStack(alignment: .leading, spacing: 21) {
                HStack(spacing: 12) {
                    MonogramMark(size: 25)
                    Text("MLXL3").font(.system(size: 27, weight: .regular, design: .serif))
                    Spacer()
                    StatusDot(state: studio.engineState)
                    Text(studio.isGenerating ? L("Génération", "Generating") : studio.canEject ? L("Prêt", "Ready") : L("Au repos", "Idle"))
                        .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                }
                Divider().overlay(StudioTheme.edge)
                VStack(alignment: .leading, spacing: 8) {
                    caption(L("MÉMOIRE DE L’APP", "APP MEMORY"))
                    HStack(alignment: .firstTextBaseline) {
                        Text(memory(engine + interface)).font(.system(size: 38, weight: .regular, design: .serif)).monospacedDigit()
                        Spacer()
                        Text(L("sur \(memory(ProcessInfo.processInfo.physicalMemory))", "of \(memory(ProcessInfo.processInfo.physicalMemory))"))
                            .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                    }
                    ProgressView(value: min(Double(engine + interface) / Double(ProcessInfo.processInfo.physicalMemory), 1))
                        .tint(StudioTheme.ink).scaleEffect(y: 0.65)
                    HStack {
                        Text(L("Moteur \(memory(engine))", "Engine \(memory(engine))"))
                        Spacer()
                        Text("UI \(memory(interface))")
                    }.font(.system(size: 10)).foregroundStyle(StudioTheme.quiet).monospacedDigit()
                }
                VStack(alignment: .leading, spacing: 10) {
                    caption(L("MODÈLE ACTIF", "ACTIVE MODEL"))
                    if let model = studio.loadedModel {
                        Text(model.name).font(.system(size: 14, weight: .medium)).lineLimit(2)
                        Text(model.architectureLabel + " · EXL3" + (model.bits.map { String(format: " · %.2f BPW", $0) } ?? ""))
                            .font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
                        HStack {
                            Text(L("Contexte", "Context"))
                            Spacer()
                            Text(studio.contextLabel).monospacedDigit()
                        }.font(.system(size: 10)).foregroundStyle(StudioTheme.secondary)
                    } else {
                        Text(L("Aucun modèle chargé", "No model loaded"))
                            .font(.system(size: 20, weight: .regular, design: .serif)).foregroundStyle(StudioTheme.secondary)
                    }
                }
                .padding(15).frame(maxWidth: .infinity, alignment: .leading)
                .background(StudioTheme.panel, in: RoundedRectangle(cornerRadius: 9))
                HStack(spacing: 0) {
                    metric("DECODE", studio.latestGenerationStats.map { String(format: "%.1f", $0.decodeTps) }, "tok/s")
                    Divider().frame(height: 30)
                    metric("PREFILL", studio.latestGenerationStats.map { String(format: "%.1f", $0.prefillTps) }, "tok/s")
                    Divider().frame(height: 30)
                    metric("TTFT", studio.latestGenerationStats.map { String(format: "%.0f", $0.ttftSeconds * 1000) }, "ms")
                }
                Divider().overlay(StudioTheme.edge)
                HStack(spacing: 8) {
                    Button {
                        if let window = NSApplication.shared.windows.first(where: { $0.canBecomeMain && !($0 is NSPanel) }) {
                            window.makeKeyAndOrderFront(nil)
                        } else { openWindow(id: "studio") }
                        NSApplication.shared.activate()
                    } label: {
                        Label(L("Ouvrir", "Open"), systemImage: "arrow.up.forward.app").frame(maxWidth: .infinity).frame(height: 33)
                    }.buttonStyle(PrimaryGlassButtonStyle())
                    Button(action: studio.ejectModel) {
                        Label(L("Décharger", "Unload"), systemImage: "eject").frame(maxWidth: .infinity).frame(height: 33)
                    }.buttonStyle(GlassPillButtonStyle()).disabled(!studio.canEject)
                    Button { NSApplication.shared.terminate(nil) } label: {
                        Image(systemName: "power").frame(width: 30, height: 33)
                    }.buttonStyle(RoundGlassButtonStyle()).help(L("Quitter MLXL3", "Quit MLXL3"))
                        .accessibilityLabel(L("Quitter MLXL3", "Quit MLXL3"))
                }.font(.system(size: 11, weight: .medium))
            }.padding(23)
        }
        .id(studio.language)
        .frame(width: 360)
        .fixedSize(horizontal: false, vertical: true)
        .foregroundStyle(StudioTheme.ink)
        .background(StudioTheme.canvas)
        .preferredColorScheme(.dark)
    }

    private func caption(_ text: String) -> some View {
        Text(text).font(.system(size: 8, weight: .medium)).tracking(1.3).foregroundStyle(StudioTheme.quiet)
    }

    private func metric(_ title: String, _ value: String?, _ unit: String) -> some View {
        VStack(spacing: 5) {
            caption(title)
            Text(value ?? "—").font(.system(size: 23, weight: .regular, design: .serif)).monospacedDigit()
            Text(unit).font(.system(size: 9)).foregroundStyle(StudioTheme.quiet)
        }.frame(maxWidth: .infinity)
    }

    private func memory(_ bytes: UInt64) -> String {
        String(format: "%.2f %@", Double(bytes) / 1e9, L("Go", "GB"))
    }
}
