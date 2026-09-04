import AppKit
import SwiftUI

struct MenuBarPanel: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1)) { _ in
            panel(
                memoryBytes: studio.residentMemoryBytes,
                engineMemoryBytes: studio.engineResidentMemoryBytes,
                interfaceMemoryBytes: studio.interfaceResidentMemoryBytes
            )
        }
        .frame(width: 342)
        .preferredColorScheme(.dark)
        .onAppear { studio.start() }
    }

    private func panel(
        memoryBytes: UInt64,
        engineMemoryBytes: UInt64,
        interfaceMemoryBytes: UInt64
    ) -> some View {
        ZStack {
            StudioTheme.canvas
            RadialGradient(
                colors: [StudioTheme.accent.opacity(0.11), .clear],
                center: .topLeading,
                startRadius: 0,
                endRadius: 280
            )

            VStack(spacing: 13) {
                header
                memoryCard(
                    memoryBytes: memoryBytes,
                    engineMemoryBytes: engineMemoryBytes,
                    interfaceMemoryBytes: interfaceMemoryBytes
                )
                modelCard
                performanceCard(studio.latestGenerationStats)
                actions
            }
            .padding(16)
        }
    }

    private var header: some View {
        HStack(spacing: 11) {
            LogoMark(size: 36)
            VStack(alignment: .leading, spacing: 2) {
                Text("MLXL3 Desktop")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .tracking(0.25)
                Text("EXL3 · APPLE METAL")
                    .font(.system(size: 8, weight: .bold, design: .rounded))
                    .tracking(1.25)
                    .foregroundStyle(StudioTheme.quiet)
            }
            Spacer()
            HStack(spacing: 6) {
                StatusDot(state: studio.engineState)
                Text(statusLabel)
                    .font(.system(size: 9, weight: .semibold, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.66))
            }
            .padding(.horizontal, 9)
            .frame(height: 25)
            .background(Color.white.opacity(0.045), in: Capsule())
            .overlay { Capsule().stroke(Color.white.opacity(0.08), lineWidth: 0.6) }
        }
    }

    private func memoryCard(
        memoryBytes: UInt64,
        engineMemoryBytes: UInt64,
        interfaceMemoryBytes: UInt64
    ) -> some View {
        let physicalMemory = ProcessInfo.processInfo.physicalMemory
        let fraction = min(1, Double(memoryBytes) / Double(max(physicalMemory, 1)))

        return VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .firstTextBaseline) {
                Label("MÉMOIRE UNIFIÉE", systemImage: "memorychip")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .tracking(0.85)
                    .foregroundStyle(StudioTheme.quiet)
                Spacer()
                Text(memory(memoryBytes))
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .monospacedDigit()
            }

            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    Capsule().fill(Color.white.opacity(0.065))
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [Color.white.opacity(0.92), StudioTheme.accent.opacity(0.72)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: max(5, geometry.size.width * fraction))
                        .shadow(color: StudioTheme.accent.opacity(0.24), radius: 6)
                }
            }
            .frame(height: 5)

            HStack {
                Text("Moteur \(memory(engineMemoryBytes))")
                Spacer()
                Text("Interface \(memory(interfaceMemoryBytes))")
                    .monospacedDigit()
            }
            .font(.system(size: 9.5, weight: .medium, design: .rounded))
            .foregroundStyle(StudioTheme.quiet)

            Text("\(Int(fraction * 100)) % de \(memory(physicalMemory))")
                .font(.system(size: 8.5, weight: .medium, design: .rounded))
                .foregroundStyle(StudioTheme.quiet.opacity(0.82))
                .monospacedDigit()
        }
        .padding(13)
        .premiumGlass(radius: 17, tint: StudioTheme.accent.opacity(0.035))
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Label("MODÈLE", systemImage: "cpu")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .tracking(0.85)
                    .foregroundStyle(StudioTheme.quiet)
                Spacer()
                Text(studio.engineState.label.uppercased())
                    .font(.system(size: 8, weight: .bold, design: .rounded))
                    .tracking(0.65)
                    .foregroundStyle(StudioTheme.secondary)
            }

            if let model = studio.loadedModel {
                Text(model.name)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .lineLimit(1)
                HStack(spacing: 6) {
                    modelTag(model.format)
                    if let bits = model.bits {
                        modelTag(String(format: "%.2f BPW", bits))
                    }
                    modelTag(model.size)
                    Spacer(minLength: 0)
                }
                Text(model.architectureLabel)
                    .font(.system(size: 8.5, weight: .medium, design: .rounded))
                    .tracking(0.7)
                    .foregroundStyle(StudioTheme.quiet)
            } else {
                Text("Aucun modèle chargé")
                    .font(.system(size: 12.5, weight: .medium, design: .rounded))
                    .foregroundStyle(StudioTheme.secondary)
            }
        }
        .padding(13)
        .premiumGlass(radius: 17, tint: Color.white.opacity(0.018))
    }

    private func performanceCard(_ stats: GenerationStats?) -> some View {
        HStack(spacing: 0) {
            compactMetric("DECODE", stats.map { String(format: "%.1f", $0.decodeTps) } ?? "—", "tok/s")
            metricDivider
            compactMetric("PREFILL", stats.map { String(format: "%.1f", $0.prefillTps) } ?? "—", "tok/s")
            metricDivider
            compactMetric("TTFT", stats.map { String(format: "%.0f", $0.ttftSeconds * 1000) } ?? "—", "ms")
            metricDivider
            compactMetric("CACHE", stats.map { String(format: "%.0f", $0.cacheHitPercent) } ?? "—", "%")
        }
        .padding(.vertical, 11)
        .premiumGlass(radius: 17, tint: Color.white.opacity(0.014))
    }

    private var actions: some View {
        HStack(spacing: 9) {
            Button(action: studio.ejectModel) {
                Label("Décharger", systemImage: "eject.fill")
                    .frame(maxWidth: .infinity)
                    .frame(height: 34)
            }
            .buttonStyle(MenuBarActionStyle())
            .disabled(!studio.canEject)

            Button {
                NSApplication.shared.terminate(nil)
            } label: {
                Label("Quitter", systemImage: "power")
                    .frame(maxWidth: .infinity)
                    .frame(height: 34)
            }
            .buttonStyle(MenuBarActionStyle(destructive: true))
        }
        .font(.system(size: 11, weight: .semibold, design: .rounded))
    }

    private var statusLabel: String {
        switch studio.engineState {
        case .idle: "INACTIF"
        case .loading: "CHARGEMENT"
        case .ready: "PRÊT"
        case .generating: "GÉNÈRE"
        case .failed: "ERREUR"
        }
    }

    private func modelTag(_ value: String) -> some View {
        Text(value.uppercased())
            .font(.system(size: 8.5, weight: .bold, design: .rounded))
            .tracking(0.45)
            .foregroundStyle(Color.white.opacity(0.66))
            .padding(.horizontal, 7)
            .frame(height: 21)
            .background(Color.white.opacity(0.045), in: Capsule())
            .overlay { Capsule().stroke(Color.white.opacity(0.075), lineWidth: 0.6) }
    }

    private func compactMetric(_ title: String, _ value: String, _ unit: String) -> some View {
        VStack(spacing: 3) {
            Text(title)
                .font(.system(size: 8, weight: .bold, design: .rounded))
                .tracking(0.75)
                .foregroundStyle(StudioTheme.quiet)
            HStack(alignment: .firstTextBaseline, spacing: 3) {
                Text(value)
                    .font(.system(size: 13, weight: .semibold, design: .rounded))
                    .monospacedDigit()
                Text(unit)
                    .font(.system(size: 8, weight: .medium, design: .rounded))
                    .foregroundStyle(StudioTheme.quiet)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var metricDivider: some View {
        Rectangle()
            .fill(Color.white.opacity(0.075))
            .frame(width: 0.7, height: 29)
    }

    private func memory(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .memory)
    }
}

private struct MenuBarActionStyle: ButtonStyle {
    @Environment(\.isEnabled) private var isEnabled
    var destructive = false

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .foregroundStyle(
                destructive
                    ? Color(red: 1, green: 0.62, blue: 0.62)
                    : Color.white.opacity(0.9)
            )
            .background(
                destructive
                    ? Color.red.opacity(configuration.isPressed ? 0.14 : 0.075)
                    : Color.white.opacity(configuration.isPressed ? 0.10 : 0.045),
                in: RoundedRectangle(cornerRadius: 11, style: .continuous)
            )
            .glassEffect(
                .clear.tint(
                    destructive ? Color.red.opacity(0.035) : Color.white.opacity(0.025)
                ).interactive(isEnabled),
                in: RoundedRectangle(cornerRadius: 11, style: .continuous)
            )
            .overlay {
                RoundedRectangle(cornerRadius: 11, style: .continuous)
                    .stroke(
                        destructive ? Color.red.opacity(0.16) : Color.white.opacity(0.10),
                        lineWidth: 0.65
                    )
            }
            .scaleEffect(configuration.isPressed ? 0.98 : 1)
            .opacity(isEnabled ? 1 : 0.34)
            .animation(.easeOut(duration: 0.11), value: configuration.isPressed)
    }
}
