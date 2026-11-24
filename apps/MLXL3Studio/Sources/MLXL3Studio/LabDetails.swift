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

        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 16) {
                if !models.isEmpty {
                    Picker(L("Modèle", "Model"), selection: $model) {
                        Text(L("Tous les modèles", "All models")).tag(String?.none)
                        ForEach(models, id: \.self) { Text($0).tag(Optional($0)) }
                    }
                    .labelsHidden().fixedSize(horizontal: false, vertical: true)
                    .frame(idealWidth: 180, maxWidth: 230, alignment: .leading)
                    .accessibilityIdentifier("lab.history.model")
                }
                Spacer(minLength: 0)
                Button(action: studio.exportLab) {
                    Label(L("Exporter…", "Export…"), systemImage: "square.and.arrow.up")
                }
                .disabled(studio.labSamples.isEmpty)
                .help(L("Exporter en JSON toutes les générations détaillées conservées, toutes périodes et tous modèles confondus.",
                        "Export all retained generation details as JSON, across every model and period."))
                .accessibilityIdentifier("lab.history.export")
            }

            if samples.isEmpty {
                Text(L("Aucun détail de génération disponible pour cette période.", "No generation details available for this period."))
                    .font(.body).foregroundStyle(StudioTheme.secondary).padding(.vertical, 12)
                    .accessibilityIdentifier("lab.history.empty")
            } else {
                LazyVStack(alignment: .leading, spacing: 0) {
                    ForEach(samples.prefix(limit)) { sample in
                        Divider()
                        LabGenerationRow(sample: sample)
                    }
                }
                if samples.count > limit {
                    Button(L("Afficher davantage", "Show more")) { limit += 12 }
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

private struct LabGenerationRow: View {
    let sample: LabSample
    @Environment(\.locale) private var locale
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            LabGenerationDetails(sample: sample)
                .padding(.leading, 3).padding(.vertical, 10)
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text(sample.model).font(.body.weight(.medium)).lineLimit(1)
                        .truncationMode(.middle).help(sample.model)
                    let date = sample.date.formatted(.dateTime.year().month(.abbreviated).day().hour().minute().locale(locale))
                    Text("\(date) · \(sample.mode)")
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                }
                Spacer(minLength: 0)
                if let rate = LabDecodeUsage.measurement(sample.stats) {
                    let speed = rate.average.formatted(.number.precision(.fractionLength(1)).locale(locale))
                    Text("\(speed) tok/s").font(.body.monospacedDigit())
                        .accessibilityLabel(L("Décodage à \(speed) tokens par seconde", "Decode at \(speed) tokens per second"))
                } else {
                    Text("—").foregroundStyle(StudioTheme.quiet)
                        .accessibilityLabel(L("Débit de décodage indisponible", "Decode speed unavailable"))
                }
            }
            .contentShape(Rectangle())
        }
        .padding(.vertical, 11)
        .accessibilityIdentifier("lab.generation.\(sample.id)")
    }
}

private struct LabGenerationDetails: View {
    let sample: LabSample
    @Environment(\.locale) private var locale

    var body: some View {
        let stats = sample.stats
        let tokens = [stats.promptTokens, stats.generatedTokens].map { $0.formatted(.number.locale(locale)) }.joined(separator: " / ")
        VStack(alignment: .leading, spacing: 7) {
            reading(L("Tokens en entrée / sortie", "Input / output tokens"), tokens)
            reading(L("Tokens en cache", "Cached tokens"), stats.cachedPromptTokens?.formatted(.number.locale(locale)) ?? "—")
            reading(L("Préremplissage", "Prefill"), metric(stats.prefillTps, "tok/s"))
            reading(L("Premier token (moteur)", "First token (engine)"), metric(stats.ttftSeconds * 1_000, "ms"))
            if let endToEnd = stats.endToEndTTFTSeconds {
                reading(L("Premier token (bout en bout)", "First token (end to end)"), metric(endToEnd * 1_000, "ms"))
            }
            if let elapsed = stats.elapsedSeconds {
                reading(L("Durée", "Duration"), metric(elapsed, "s"))
            }
            Divider().padding(.vertical, 3)
            if let bits = sample.bits { reading(L("Quantification", "Quantization"), metric(bits, "bpw")) }
            reading(L("Contexte", "Context"), sample.context.formatted(.number.locale(locale)))
            reading(L("Température / Top K", "Temperature / Top K"),
                    "\(sample.temperature.formatted(.number.locale(locale))) / \(sample.topK.formatted(.number.locale(locale)))")
            let on = L("Activé", "On"), off = L("Désactivé", "Off")
            reading("MTP / DFlash", "\(sample.mtp ? on : off) / \(sample.dflash ? on : off)")
            if let memory = stats.memory {
                if let metal = memory.nativeMetalAllocatedBytes {
                    reading(L("Allocation Metal", "Metal allocation"), labBytes(metal, locale: locale))
                } else {
                    reading(L("Pic MLX", "MLX peak"), labBytes(memory.mlxPeakBytes, locale: locale))
                }
                reading(L("Mémoire du processus", "Process memory"), labBytes(memory.processFootprintBytes, locale: locale))
            }
        }
        .font(.callout)
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
        VStack(alignment: .leading, spacing: 26) {
            VStack(alignment: .leading, spacing: 12) {
                StudioCaption(french: "Matériel", english: "Hardware").accessibilityAddTraits(.isHeader)
                Divider()
                reading(L("Puce", "Chip"), LabHardware.chip)
                reading(L("Identifiant du Mac", "Mac identifier"), LabHardware.machine)
                reading(L("Cœurs CPU", "CPU cores"), info.processorCount.formatted())
                reading(L("Mémoire unifiée", "Unified memory"), labBytes(info.physicalMemory, locale: locale))
                reading("GPU Metal", LabHardware.gpu?.name ?? "—")
                reading(L("Budget GPU conseillé", "Recommended GPU budget"), labBytes(LabHardware.gpu?.recommendedBytes, locale: locale))
            }
            VStack(alignment: .leading, spacing: 12) {
                StudioCaption(french: "Logiciels", english: "Software").accessibilityAddTraits(.isHeader)
                Divider()
                reading("macOS", info.operatingSystemVersionString)
                reading("MLXL3 Studio", Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev")
                reading(L("Moteur", "Runtime"), studio.runtimeIdentity ?? "—")
            }
            VStack(alignment: .leading, spacing: 12) {
                StudioCaption(french: "État actuel", english: "Current status").accessibilityAddTraits(.isHeader)
                Divider()
                TimelineView(.periodic(from: .now, by: 2)) { _ in
                    VStack(alignment: .leading, spacing: 12) {
                        reading(L("Mémoire de Studio", "Studio memory"), labBytes(studio.interfaceMemoryFootprintBytes, locale: locale))
                        reading(L("Mémoire du moteur", "Engine memory"), labBytes(studio.engineMemoryFootprintBytes, locale: locale))
                        reading(L("État thermique", "Thermal state"), thermalDescription(info.thermalState))
                        reading(L("Mode économie d’énergie", "Low power mode"), info.isLowPowerModeEnabled ? L("Activé", "On") : L("Désactivé", "Off"))
                    }
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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

private func reading(_ label: String, _ value: String) -> some View {
    HStack(alignment: .firstTextBaseline, spacing: 18) {
        Text(label).foregroundStyle(StudioTheme.secondary)
        Spacer(minLength: 0)
        Text(value).monospacedDigit().multilineTextAlignment(.trailing)
            .foregroundStyle(StudioTheme.ink).textSelection(.enabled)
    }
    .accessibilityElement(children: .combine)
}

private func labBytes(_ value: UInt64?, locale: Locale) -> String {
    value.map { Int64(clamping: $0).formatted(.byteCount(style: .memory, spellsOutZero: false).locale(locale)) } ?? "—"
}
