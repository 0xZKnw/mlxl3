import SwiftUI

struct LabGenerationsView: View {
    let period: LabUsagePeriod
    let now: Date

    @EnvironmentObject private var studio: StudioModel
    @State private var model: String?
    @State private var limit = 12

    var body: some View {
        let recent = studio.labSamples.filter { $0.valid && period.contains($0, now: now) }
            .sorted { $0.date > $1.date }
        let models = Set(recent.map(\.model)).sorted()
        let samples = model.map { name in recent.filter { $0.model == name } } ?? recent

        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                if !models.isEmpty {
                    Picker(L("Modèle", "Model"), selection: $model) {
                        Text(L("Tous les modèles", "All models")).tag(String?.none)
                        ForEach(models, id: \.self) { Text($0).tag(Optional($0)) }
                    }
                    .labelsHidden().fixedSize()
                    .accessibilityIdentifier("lab.history.model")
                    Text(samples.count == 1 ? L("1 génération", "1 generation")
                         : L("\(samples.count) générations", "\(samples.count) generations"))
                        .font(.system(size: 12)).monospacedDigit().foregroundStyle(StudioTheme.quiet)
                }
                Spacer(minLength: 0)
                Button(action: studio.exportLab) {
                    Label(L("Exporter…", "Export…"), systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.studio)
                .disabled(studio.labSamples.isEmpty)
                .help(L("Exporter en JSON toutes les générations détaillées conservées, toutes périodes et tous modèles confondus.",
                        "Export all retained generation details as JSON, across every model and period."))
                .accessibilityIdentifier("lab.history.export")
            }

            if samples.isEmpty {
                StudioEmptyState(icon: "clock",
                                 title: L("Aucun détail de génération", "No generation details"),
                                 detail: L("Aucun détail de génération disponible pour cette période.",
                                           "No generation details available for this period."))
                    .studioCard(padding: 0)
                    .accessibilityIdentifier("lab.history.empty")
            } else {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(Array(samples.prefix(limit).enumerated()), id: \.element.id) { index, sample in
                        if index > 0 {
                            Rectangle().fill(StudioTheme.edge.opacity(0.7)).frame(height: 0.5)
                                .accessibilityHidden(true)
                        }
                        LabGenerationRow(sample: sample)
                    }
                }
                .studioCard(padding: 0)
                if samples.count > limit {
                    Button(L("Afficher davantage", "Show more")) { limit += 12 }
                        .buttonStyle(.studio)
                        .frame(maxWidth: .infinity)
                        .accessibilityIdentifier("lab.history.more")
                }
            }
        }
        .onChange(of: period) { _, _ in model = nil; limit = 12 }
        .onChange(of: model) { _, _ in limit = 12 }
        .onChange(of: models, initial: true) { _, updated in
            if let model, !updated.contains(model) { self.model = nil }
        }
    }
}

/// One reply: the model, when, and its speed on the closed row; every measurement once open.
private struct LabGenerationRow: View {
    let sample: LabSample
    @Environment(\.locale) private var locale
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var expanded = false
    @State private var hovered = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            Button {
                if reduceMotion { expanded.toggle() } else {
                    withAnimation(.easeOut(duration: 0.16)) { expanded.toggle() }
                }
            } label: {
                HStack(alignment: .center, spacing: 11) {
                    Image(systemName: "chevron.right")
                        .font(.system(size: 9.5, weight: .semibold))
                        .foregroundStyle(StudioTheme.quiet)
                        .rotationEffect(.degrees(expanded ? 90 : 0))
                        .frame(width: 12)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(sample.model).font(.system(size: 13, weight: .medium)).lineLimit(1)
                            .truncationMode(.middle).help(sample.model)
                        let date = sample.date.formatted(.dateTime.year().month(.abbreviated).day().hour().minute().locale(locale))
                        Text("\(date) · \(sample.mode)")
                            .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                    }
                    Spacer(minLength: 8)
                    let tokens = sample.stats.generatedTokens.formatted(.number.locale(locale))
                    Text(L("\(tokens) tokens", "\(tokens) tokens"))
                        .font(.system(size: 12)).monospacedDigit().foregroundStyle(StudioTheme.quiet)
                    if let rate = LabDecodeUsage.measurement(sample.stats) {
                        let speed = rate.average.formatted(.number.precision(.fractionLength(1)).locale(locale))
                        Text("\(speed) tok/s").font(.system(size: 13, weight: .medium)).monospacedDigit()
                            .frame(minWidth: 84, alignment: .trailing)
                            .accessibilityLabel(L("Décodage à \(speed) tokens par seconde", "Decode at \(speed) tokens per second"))
                    } else {
                        Text("—").foregroundStyle(StudioTheme.quiet)
                            .frame(minWidth: 84, alignment: .trailing)
                            .accessibilityLabel(L("Débit de décodage indisponible", "Decode speed unavailable"))
                    }
                }
                .padding(.horizontal, 16).padding(.vertical, 11)
                .contentShape(Rectangle())
                .background(StudioTheme.contrast.opacity(hovered ? 0.03 : 0))
            }
            .buttonStyle(.plain)
            .onHover { hovered = $0 }
            .accessibilityHint(expanded ? L("Masquer les mesures", "Hide the measurements") : L("Afficher les mesures", "Show the measurements"))
            if expanded {
                LabGenerationDetails(sample: sample)
                    .padding(.leading, 39).padding(.trailing, 16).padding(.bottom, 14).padding(.top, 2)
                    .transition(.opacity)
            }
        }
        .accessibilityIdentifier("lab.generation.\(sample.id)")
    }
}

private struct LabGenerationDetails: View {
    let sample: LabSample
    @Environment(\.locale) private var locale

    var body: some View {
        let stats = sample.stats
        let tokens = [stats.promptTokens, stats.generatedTokens].map { $0.formatted(.number.locale(locale)) }.joined(separator: " / ")
        let on = L("Activé", "On"), off = L("Désactivé", "Off")
        // Two columns: what was measured on the left, how the model was set up on the right.
        HStack(alignment: .top, spacing: 28) {
            VStack(alignment: .leading, spacing: 7) {
                caption(L("Mesures", "Measurements"))
                StudioReading(label: L("Tokens en entrée / sortie", "Input / output tokens"), value: tokens)
                StudioReading(label: L("Tokens en cache", "Cached tokens"),
                              value: stats.cachedPromptTokens?.formatted(.number.locale(locale)) ?? "—")
                StudioReading(label: L("Préremplissage", "Prefill"), value: metric(stats.prefillTps, "tok/s"))
                StudioReading(label: L("Premier token (moteur)", "First token (engine)"), value: metric(stats.ttftSeconds * 1_000, "ms"))
                if let endToEnd = stats.endToEndTTFTSeconds {
                    StudioReading(label: L("Premier token (bout en bout)", "First token (end to end)"), value: metric(endToEnd * 1_000, "ms"))
                }
                if let elapsed = stats.elapsedSeconds {
                    StudioReading(label: L("Durée", "Duration"), value: metric(elapsed, "s"))
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
            VStack(alignment: .leading, spacing: 7) {
                caption(L("Configuration", "Setup"))
                if let bits = sample.bits { StudioReading(label: L("Quantification", "Quantization"), value: metric(bits, "bpw")) }
                StudioReading(label: L("Contexte", "Context"), value: sample.context.formatted(.number.locale(locale)))
                StudioReading(label: L("Température / Top K", "Temperature / Top K"),
                              value: "\(sample.temperature.formatted(.number.locale(locale))) / \(sample.topK.formatted(.number.locale(locale)))")
                StudioReading(label: "MTP / DFlash", value: "\(sample.mtp ? on : off) / \(sample.dflash ? on : off)")
                if let memory = stats.memory {
                    if let metal = memory.nativeMetalAllocatedBytes {
                        StudioReading(label: L("Allocation Metal", "Metal allocation"), value: labBytes(metal, locale: locale))
                    } else {
                        StudioReading(label: L("Pic MLX", "MLX peak"), value: labBytes(memory.mlxPeakBytes, locale: locale))
                    }
                    StudioReading(label: L("Mémoire du processus", "Process memory"), value: labBytes(memory.processFootprintBytes, locale: locale))
                }
            }
            .frame(maxWidth: .infinity, alignment: .topLeading)
        }
    }

    private func caption(_ text: String) -> some View {
        Text(text).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(StudioTheme.quiet)
            .padding(.bottom, 1).accessibilityAddTraits(.isHeader)
    }

    private func metric(_ value: Double, _ unit: String) -> String {
        guard value.isFinite, value >= 0 else { return "—" }
        return "\(value.formatted(.number.precision(.fractionLength(1)).locale(locale))) \(unit)"
    }
}

struct LabHardwareView: View {
    @Environment(\.locale) private var locale
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        let info = ProcessInfo.processInfo
        VStack(alignment: .leading, spacing: 22) {
            TimelineView(.periodic(from: .now, by: 2)) { _ in
                memory(physical: info.physicalMemory, interface: studio.interfaceMemoryFootprintBytes,
                       engine: studio.engineMemoryFootprintBytes)
            }
            SettingSection(title: L("Matériel", "Hardware"), icon: "cpu") {
                StudioReading(label: L("Puce", "Chip"), value: LabHardware.chip)
                StudioReading(label: L("Identifiant du Mac", "Mac identifier"), value: LabHardware.machine)
                StudioReading(label: L("Cœurs CPU", "CPU cores"), value: info.processorCount.formatted())
                StudioReading(label: L("Mémoire unifiée", "Unified memory"), value: labBytes(info.physicalMemory, locale: locale))
                StudioReading(label: "GPU Metal", value: LabHardware.gpu?.name ?? "—")
                StudioReading(label: L("Budget GPU conseillé", "Recommended GPU budget"),
                              value: labBytes(LabHardware.gpu?.recommendedBytes, locale: locale))
            }
            SettingSection(title: L("Logiciels", "Software"), icon: "shippingbox") {
                StudioReading(label: "macOS", value: info.operatingSystemVersionString)
                StudioReading(label: "MLXL3 Studio",
                              value: Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev")
                StudioReading(label: L("Moteur", "Runtime"), value: studio.runtimeIdentity ?? "—")
            }
            TimelineView(.periodic(from: .now, by: 2)) { _ in
                SettingSection(title: L("État actuel", "Current status"), icon: "waveform.path.ecg") {
                    StudioReading(label: L("État thermique", "Thermal state"), value: thermalDescription(info.thermalState))
                    StudioReading(label: L("Mode économie d’énergie", "Low power mode"),
                                  value: info.isLowPowerModeEnabled ? L("Activé", "On") : L("Désactivé", "Off"))
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    /// The app's own share of the unified memory: the engine (the model) and the interface.
    /// What other apps use is not measured here, so the rest is not called "free".
    private func memory(physical: UInt64, interface: UInt64?, engine: UInt64?) -> some View {
        let total = Double(max(1, physical))
        let engineShare = min(1, Double(engine ?? 0) / total)
        let interfaceShare = min(1 - engineShare, Double(interface ?? 0) / total)
        return VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline) {
                Text(L("Mémoire utilisée par MLXL3", "Memory used by MLXL3")).font(.system(size: 12, weight: .medium))
                    .foregroundStyle(StudioTheme.secondary)
                Spacer(minLength: 8)
                Text(L("sur \(labBytes(physical, locale: locale)) de mémoire unifiée",
                       "of \(labBytes(physical, locale: locale)) unified memory"))
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
            }
            GeometryReader { geometry in
                HStack(spacing: 2) {
                    if engineShare > 0 {
                        Rectangle().fill(StudioTheme.ink).frame(width: max(3, geometry.size.width * engineShare))
                    }
                    if interfaceShare > 0 {
                        Rectangle().fill(StudioTheme.secondary).frame(width: max(3, geometry.size.width * interfaceShare))
                    }
                    Rectangle().fill(StudioTheme.contrast.opacity(0.08))
                }
                .clipShape(Capsule())
            }
            .frame(height: 8).accessibilityHidden(true)
            HStack(spacing: 22) {
                share(L("Moteur", "Engine"),
                      (engine ?? 0) == 0 ? L("aucun modèle chargé", "no model loaded") : labBytes(engine, locale: locale),
                      StudioTheme.ink)
                    .accessibilityLabel(L("Mémoire du moteur", "Engine memory"))
                share("Studio", labBytes(interface, locale: locale), StudioTheme.secondary)
                    .accessibilityLabel(L("Mémoire de Studio", "Studio memory"))
                Spacer(minLength: 0)
            }
        }
        .studioCard(padding: 18, radius: 13)
    }

    private func share(_ title: String, _ value: String, _ color: Color) -> some View {
        HStack(spacing: 7) {
            Circle().fill(color).frame(width: 7, height: 7).accessibilityHidden(true)
            Text(title).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
            Text(value).font(.system(size: 13, weight: .medium)).monospacedDigit()
        }
        .accessibilityElement(children: .combine)
    }

    private func thermalDescription(_ state: ProcessInfo.ThermalState) -> String {
        switch state {
        case .nominal: L("Normal", "Nominal")
        case .fair: L("Modéré", "Fair")
        case .serious: L("Élevé", "Serious")
        case .critical: L("Critique", "Critical")
        @unknown default: "—"
        }
    }
}

private func labBytes(_ value: UInt64?, locale: Locale) -> String {
    value.map { Int64(clamping: $0).formatted(.byteCount(style: .memory, spellsOutZero: false).locale(locale)) } ?? "—"
}
