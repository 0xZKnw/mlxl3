import AppKit
import SwiftUI

struct MenuBarPanel: View {
    var sizeChanged: ((CGSize) -> Void)? = nil
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            let engine = studio.engineMemoryFootprintBytes
            let interface = studio.interfaceMemoryFootprintBytes
            let total = engine.flatMap { engine in interface.map { engine + $0 } }
            VStack(alignment: .leading, spacing: 18) {
                HStack(spacing: 9) {
                    MonogramMark(size: 32).accessibilityHidden(false).accessibilityLabel("MLXL3")
                    Spacer()
                    StatusDot(state: studio.engineState)
                    Text(studio.engineState.label)
                        .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                        .lineLimit(1).help(studio.engineState.label)
                }

                VStack(alignment: .leading, spacing: 7) {
                    Text(L("Modèle actif", "Active model"))
                        .font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                    Text(studio.loadedModel?.name ?? (studio.canEject ? studio.selectedModelName : nil)
                         ?? L("Aucun modèle chargé", "No model loaded"))
                        .font(.system(size: 23, weight: .regular, design: .serif))
                        .tracking(-0.5).lineLimit(2).fixedSize(horizontal: false, vertical: true)
                    if let model = studio.loadedModel {
                        HStack {
                            Text(model.architectureLabel)
                            Spacer()
                            Text(studio.contextLabel).monospacedDigit()
                        }.font(.system(size: 11)).foregroundStyle(StudioTheme.secondary)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                Divider()

                HStack(spacing: 12) {
                    metric("Decode", studio.latestGenerationStats.map { String(format: "%.1f", $0.decodeTps) }, "tok/s")
                    metric("Prefill", studio.latestGenerationStats.map { String(format: "%.1f", $0.prefillTps) }, "tok/s")
                    metric("TTFT", studio.latestGenerationStats.map { String(format: "%.0f", $0.ttftSeconds * 1000) }, "ms")
                }

                Divider()

                VStack(alignment: .leading, spacing: 9) {
                    HStack(alignment: .firstTextBaseline) {
                        Text(L("Mémoire de l’app", "App memory")).font(.system(size: 12, weight: .medium))
                        Spacer()
                        Text(memory(total)).font(.system(size: 15, weight: .medium)).monospacedDigit()
                    }
                    if let total {
                        ProgressView(value: min(Double(total) / Double(ProcessInfo.processInfo.physicalMemory), 1))
                            .tint(StudioTheme.ink)
                    } else {
                        Text(L("Mesure indisponible", "Measurement unavailable"))
                            .font(.caption).foregroundStyle(StudioTheme.quiet)
                    }
                    HStack {
                        Text(L("Moteur \(memory(engine))", "Engine \(memory(engine))"))
                        Spacer()
                        Text("UI \(memory(interface))")
                    }.font(.system(size: 10)).foregroundStyle(StudioTheme.secondary).monospacedDigit()
                }
                .help(L("Empreinte mémoire macOS du moteur et de l’interface sur \(memory(ProcessInfo.processInfo.physicalMemory)).",
                        "macOS memory footprint of the engine and interface, out of \(memory(ProcessInfo.processInfo.physicalMemory))."))

                HStack(spacing: 8) {
                    Button {
                        if let window = NSApplication.shared.windows.first(where: { $0.canBecomeMain && !($0 is NSPanel) }) {
                            window.makeKeyAndOrderFront(nil)
                        } else { openWindow(id: "studio") }
                        NSApplication.shared.activate()
                    } label: {
                        Label(L("Ouvrir", "Open"), systemImage: "arrow.up.forward.app")
                            .frame(maxWidth: .infinity).frame(height: 32)
                    }.buttonStyle(StudioToolbarStyle())
                    Button(action: studio.ejectModel) {
                        Label(L("Décharger", "Unload"), systemImage: "eject")
                            .frame(maxWidth: .infinity).frame(height: 32)
                    }.buttonStyle(StudioControlStyle()).disabled(!studio.canEject)
                    Button { NSApplication.shared.terminate(nil) } label: {
                        Image(systemName: "power").frame(width: 28, height: 32)
                    }.buttonStyle(WorkspaceControlStyle())
                        .help(L("Quitter MLXL3", "Quit MLXL3"))
                        .accessibilityLabel(L("Quitter MLXL3", "Quit MLXL3"))
                }.font(.system(size: 12, weight: .medium))
            }
            .padding(20)
        }
        .id(studio.language)
        .frame(width: 340)
        .fixedSize(horizontal: false, vertical: true)
        .onGeometryChange(for: CGSize.self) { $0.size } action: { sizeChanged?($0) }
        .foregroundStyle(StudioTheme.ink)
        .background {
            StudioAurora(busy: studio.isGenerating, animated: true, fullHeight: true)
                .background(StudioTheme.canvas)
        }
        .premiumGlass(radius: 18)
        .studioAppearance(studio.theme, accent: studio.accent, customAccent: studio.customAccentColor)
    }

    private func metric(_ title: String, _ value: String?, _ unit: String) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title).font(.system(size: 10)).foregroundStyle(StudioTheme.secondary)
            Text(value ?? "—").font(.system(size: 21, weight: .medium)).monospacedDigit()
            Text(unit).font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }

    private func memory(_ bytes: UInt64?) -> String {
        guard let bytes else { return "—" }
        return String(format: "%.2f %@", Double(bytes) / 1e9, L("Go", "GB"))
    }
}
