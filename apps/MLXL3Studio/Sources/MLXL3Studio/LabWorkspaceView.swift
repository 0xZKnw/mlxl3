import SwiftUI

private extension LabUsagePeriod {
    var title: String {
        switch self {
        case .today: L("Aujourd’hui", "Today")
        case .sevenDays: L("7 jours", "7 days")
        case .thirtyDays: L("30 jours", "30 days")
        case .allTime: L("Depuis le début", "All time")
        }
    }
}

private enum LabPage: String, CaseIterable, Identifiable {
    case usage, images, compare, generations, hardware
    var id: Self { self }

    var title: String {
        switch self {
        case .usage: L("Utilisation", "Usage")
        case .images: L("Images", "Images")
        case .compare: L("Comparer", "Compare")
        case .generations: L("Générations", "Generations")
        case .hardware: L("Ce Mac", "This Mac")
        }
    }

    /// The page's own heading; the sidebar keeps the short name.
    var heading: String {
        switch self {
        case .images: L("Images générées", "Generated images")
        case .compare: L("Comparer les modèles", "Compare models")
        default: title
        }
    }

    var detail: String {
        switch self {
        case .usage: L("Tokens, débit et modèles utilisés sur ce Mac.", "Tokens, speed and models used on this Mac.")
        case .images: "Qwen-Image-2.1"
        case .compare: L("Le même prompt sur deux modèles, l’un après l’autre.", "The same prompt on two models, one after the other.")
        case .generations: L("Le détail mesuré de chaque réponse récente.", "The measured detail of each recent reply.")
        case .hardware: L("Le matériel, les logiciels et la mémoire en ce moment.", "Hardware, software and memory right now.")
        }
    }

    var icon: String {
        switch self {
        case .usage: "chart.bar"
        case .images: "photo.on.rectangle"
        case .compare: "rectangle.split.2x1"
        case .generations: "clock"
        case .hardware: "desktopcomputer"
        }
    }
}

struct LabWorkspaceView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.locale) private var locale
    // `--ui-preview-lab-page=generations` (or hardware) opens that page in the isolated preview.
    @State private var page = ProcessInfo.processInfo.arguments.contains("--ui-preview")
        ? ProcessInfo.processInfo.arguments.compactMap { $0.hasPrefix("--ui-preview-lab-page=")
            ? LabPage(rawValue: String($0.dropFirst("--ui-preview-lab-page=".count))) : nil }.first ?? .usage
        : .usage
    @State private var period = LabUsagePeriod.thirtyDays
    @State private var comparisonDraft = LabComparisonDraft()
    let imageGeneration: ImageGeneration
    let onBack: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            StudioPageSidebar(title: "Lab", onBack: onBack) {
                StudioNavList {
                    ForEach(LabPage.allCases) { item in
                        StudioNavRow(icon: item.icon, title: item.title, selected: page == item) { page = item }
                            .accessibilityIdentifier("lab.\(item.rawValue)")
                    }
                }
            }

            GeometryReader { geometry in
                let inset: CGFloat = geometry.size.width < 620 ? 24 : 36
                let width = min(page == .compare ? 940 : 820, max(0, geometry.size.width - 2 * inset))
                ScrollView {
                    VStack(alignment: .leading, spacing: 24) {
                        heading(width: width)
                        switch page {
                        case .usage:
                            TimelineView(.everyMinute) { time in
                                usage(studio.labUsage.summary(for: period, now: time.date),
                                      days: studio.labUsage.daily(for: period, now: time.date),
                                      responses: LabResponseMeasurements(samples: studio.labSamples, period: period, now: time.date),
                                      width: width)
                            }
                        case .images:
                            LabImageUsageView(engine: imageGeneration)
                        case .compare:
                            LabComparisonView(comparison: studio.labComparison, draft: $comparisonDraft, width: width)
                        case .generations:
                            TimelineView(.everyMinute) { time in
                                LabGenerationsView(period: period, now: time.date)
                            }
                        case .hardware:
                            LabHardwareView()
                        }
                    }
                    .frame(width: width, alignment: .leading)
                    .padding(.top, 40).padding(.bottom, 36)
                    .frame(maxWidth: .infinity)
                }
                .id(page)
            }
        }
        .font(.system(size: 13))
        .foregroundStyle(StudioTheme.ink)
        .onExitCommand(perform: onBack)
    }

    private func heading(width: CGFloat) -> some View {
        let wide = width >= 640
        let layout = wide ? AnyLayout(HStackLayout(alignment: .center, spacing: 20))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 16))
        return layout {
            StudioPageHeader(title: page.heading, detail: page.detail)
            if page == .usage || page == .generations {
                Picker(L("Période", "Period"), selection: $period) {
                    ForEach(LabUsagePeriod.allCases) { Text($0.title).tag($0) }
                }
                .labelsHidden().pickerStyle(.segmented).fixedSize()
                .accessibilityIdentifier("lab.usage.period")
            }
        }
    }

    private func value(_ count: Int) -> String { count.formatted(.number.locale(locale)) }

    private func generationCount(_ count: Int) -> String {
        count == 1 ? L("1 génération", "1 generation") : L("\(value(count)) générations", "\(value(count)) generations")
    }

    // Sage identifies generated output in the legend, the token split and the daily bars.
    static let outputColor = Color(nsColor: StudioTheme.adaptiveColor(
        light: StudioTheme.rgb(0x426D57), dark: StudioTheme.rgb(0x9ABCA5)))

    static func formattedShare(_ fraction: Double, locale: Locale) -> String {
        let style = FloatingPointFormatStyle<Double>.Percent(locale: locale).precision(.fractionLength(0...1))
        if fraction > 0 && fraction < 0.001 { return "< " + 0.001.formatted(style) }
        if fraction > 0.999 && fraction < 1 { return "> " + 0.999.formatted(style) }
        return fraction.formatted(style)
    }

    /// The change against the preceding period, as a signed share: nil when there is
    /// nothing to compare with (no earlier usage, or all time).
    static func change(current: Int, previous: Int?) -> Double? {
        guard let previous, previous > 0, current >= 0 else { return nil }
        return (Double(current) - Double(previous)) / Double(previous)
    }

    private func usage(_ summary: LabUsageSummary, days: [LabUsageDay], responses: LabResponseMeasurements, width: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            if !summary.models.isEmpty {
                overview(summary, width: width)
                if period != .today, days.count > 1 {
                    // The chart and the three figures share a row, so the first models stay
                    // above the fold in a 660-point window.
                    HStack(alignment: .top, spacing: 12) {
                        LabDailyChart(days: days, outputColor: Self.outputColor)
                        activity(summary, responses: responses, width: width, stacked: true)
                            .frame(width: min(236, max(168, width * 0.31)))
                    }
                    .fixedSize(horizontal: false, vertical: true)
                } else {
                    activity(summary, responses: responses, width: width, stacked: false)
                }
                VStack(alignment: .leading, spacing: 9) {
                    StudioSectionTitle(title: L("Par modèle", "By model"))
                    modelTable(summary.models, width: width)
                }
                .padding(.top, 6)
            } else if period != .allTime && studio.labUsage.hasActivity {
                overview(summary, width: width)
                StudioEmptyState(icon: "calendar",
                                 title: L("Aucune génération enregistrée sur cette période", "No generations recorded in this period"),
                                 detail: L("Des mesures plus anciennes existent.", "Older measurements exist.")) {
                    Button(L("Afficher tout l’historique", "Show all time")) { period = .allTime }
                        .buttonStyle(.studio)
                }
                .studioCard(padding: 0)
            } else {
                StudioEmptyState(icon: "chart.bar",
                                 title: L("Aucune génération enregistrée", "No generations recorded"),
                                 detail: L("Les statistiques apparaîtront après une première réponse dans Chat ou Agent sur ce Mac.",
                                           "Your stats will appear after your first reply in Chat or Agent on this Mac.")) {
                    Button(L("Ouvrir une conversation", "Open a conversation"), action: onBack)
                        .buttonStyle(.studio(.primary))
                }
                .studioCard(padding: 0)
            }
        }
    }

    private func overview(_ summary: LabUsageSummary, width: CGFloat) -> some View {
        let totals = summary.totals
        let compact = width < 600
        return HStack(alignment: .center, spacing: compact ? 18 : 34) {
            VStack(alignment: .leading, spacing: 5) {
                HStack(alignment: .center, spacing: 8) {
                    Text(L("Tokens utilisés", "Tokens used")).font(.system(size: 12, weight: .medium))
                        .foregroundStyle(StudioTheme.secondary)
                    if let change = Self.change(current: totals.totalTokens, previous: summary.previousPeriodTokens) {
                        trend(change)
                    }
                }
                Text(value(totals.totalTokens))
                    .font(.system(size: compact ? 36 : 42, weight: .semibold))
                    .tracking(-0.8)
                    .monospacedDigit().lineLimit(1).minimumScaleFactor(0.45)
                    .accessibilityIdentifier("lab.usage.total")
                let days = summary.activeDays == 1 ? L("1 jour actif", "1 active day")
                    : L("\(value(summary.activeDays)) jours actifs", "\(value(summary.activeDays)) active days")
                Text(generationCount(totals.generations) + " · " + days)
                    .foregroundStyle(StudioTheme.secondary)
                if let prior = summary.previousPeriodTokens, let description = comparison(prior) {
                    Text(description).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .leading, spacing: 11) {
                HStack(alignment: .top, spacing: 12) {
                    tokenCount(L("Entrée", "Input"), count: totals.inputTokens)
                    Spacer(minLength: 0)
                    tokenCount(L("Sortie", "Output"), count: totals.outputTokens, isOutput: true)
                }
                GeometryReader { geometry in
                    let total = Double(totals.inputTokens) + Double(totals.outputTokens)
                    let outputShare = total > 0 ? Double(totals.outputTokens) / total : 0
                    ZStack(alignment: .trailing) {
                        (total > 0 ? StudioTheme.secondary : StudioTheme.edge)
                        Rectangle().fill(Self.outputColor)
                            .frame(width: geometry.size.width * outputShare)
                        if geometry.size.width * min(outputShare, 1 - outputShare) > 2 {
                            Rectangle().fill(StudioTheme.canvas)
                                .frame(width: 2, height: geometry.size.height)
                                .position(x: geometry.size.width * (1 - outputShare), y: geometry.size.height / 2)
                        }
                    }
                    .clipShape(Capsule())
                }
                .frame(height: 6).accessibilityHidden(true)
                if let cached = totals.cachedInputTokens, totals.inputTokens > 0 {
                    let fraction = Self.formattedShare(Double(cached) / Double(totals.inputTokens), locale: locale)
                    Text(L("\(value(cached)) en cache · \(fraction) de l’entrée",
                           "\(value(cached)) cached · \(fraction) of input"))
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                }
            }
            .frame(width: min(304, width * (compact ? 0.40 : 0.39)))
        }
        .studioCard(padding: compact ? 16 : 18, radius: 13)
    }

    /// A signed share against the previous period; the arrow carries the direction for
    /// people who do not tell the two greys apart.
    private func trend(_ change: Double) -> some View {
        let share = Self.formattedShare(min(abs(change), 99.99), locale: locale)
        let up = change > 0, flat = change == 0
        return Label(flat ? share : (up ? "+" : "−") + share,
                     systemImage: flat ? "arrow.right" : up ? "arrow.up.right" : "arrow.down.right")
            .font(.system(size: 11.5, weight: .medium)).monospacedDigit()
            .foregroundStyle(StudioTheme.secondary)
            .padding(.horizontal, 7).frame(height: 19)
            .background(StudioTheme.contrast.opacity(0.06), in: Capsule())
            .help(L("Par rapport à la période précédente", "Compared with the previous period"))
            .accessibilityLabel(L("Évolution ", "Change ") + (flat ? share : (up ? "+" : "−") + share))
    }

    private func comparison(_ previous: Int) -> String? {
        let n = value(previous)
        switch period {
        case .today: return L("\(n) tokens hier", "\(n) tokens yesterday")
        case .sevenDays: return L("\(n) tokens sur les 7 jours précédents", "\(n) tokens in the previous 7 days")
        case .thirtyDays: return L("\(n) tokens sur les 30 jours précédents", "\(n) tokens in the previous 30 days")
        case .allTime: return nil
        }
    }

    private func tokenCount(_ title: String, count: Int, isOutput: Bool = false) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 5) {
                Circle().fill(isOutput ? Self.outputColor : StudioTheme.secondary)
                    .frame(width: 7, height: 7).accessibilityHidden(true)
                Text(title).font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
            }
            Text(value(count)).monospacedDigit().font(.system(size: 18, weight: .medium))
                .foregroundStyle(isOutput ? Self.outputColor : StudioTheme.ink)
                .lineLimit(1).minimumScaleFactor(0.55)
        }
        .accessibilityElement(children: .combine)
    }

    /// Average reply, median first text and busiest day: three tiles in a row, or one card
    /// of three lines beside the chart (`stacked`).
    @ViewBuilder
    private func activity(_ summary: LabUsageSummary, responses: LabResponseMeasurements, width: CGFloat, stacked: Bool) -> some View {
        let total = summary.totals
        let average = (Double(total.outputTokens) / Double(total.generations))
            .formatted(.number.precision(.fractionLength(0...1)).locale(locale))
        let measured = responses.measuredCount
        let timingNote = measured == 1 ? L("Médiane · 1 mesure", "Median · 1 measurement")
            : L("Médiane · \(value(measured)) mesures", "Median · \(value(measured)) measurements")
        let noTiming = L("Délai non mesuré", "Time not measured")
        let timingHelp = L("Délai médian du moteur avant le premier texte. \(value(measured)) mesures parmi \(value(responses.retainedCount)) générations détaillées conservées sur cette période.",
                           "Median time from the engine receiving a request to first text. \(value(measured)) measurements across \(value(responses.retainedCount)) retained generation details for this period.")
        let narrow = width < 600
        let figures: [(name: String, number: String, note: String, color: Color, help: String?)] = [
            (L("Réponse moyenne", "Avg. response"), "\(average) tokens", L("par génération", "per generation"), Self.outputColor, nil),
            (L("Premier texte", "First text"), responses.medianFirstTokenSeconds.map(duration) ?? "—",
             measured > 0 ? timingNote : noTiming, StudioTheme.ink, timingHelp),
            (L("Jour le plus actif", "Busiest day"), summary.busiestDay?.formatted(locale: locale) ?? "—",
             summary.busiestDay.map { "\(value($0.tokens)) tokens" } ?? "", StudioTheme.ink, nil),
        ]
        if stacked {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(Array(figures.enumerated()), id: \.offset) { index, figure in
                    if index > 0 {
                        Rectangle().fill(StudioTheme.edge.opacity(0.7)).frame(height: 0.5).accessibilityHidden(true)
                    }
                    VStack(alignment: .leading, spacing: 2) {
                        Text(figure.name).font(.system(size: 11.5, weight: .medium)).foregroundStyle(StudioTheme.secondary)
                        Text(figure.number).font(.system(size: 15, weight: .semibold)).monospacedDigit()
                            .foregroundStyle(figure.color).lineLimit(1).minimumScaleFactor(0.7)
                        Text(figure.note).font(.system(size: 11.5)).foregroundStyle(StudioTheme.quiet)
                            .lineLimit(1).minimumScaleFactor(0.8)
                    }
                    .padding(.horizontal, 14).padding(.vertical, 9)
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                    .help(figure.help ?? "")
                    .accessibilityElement(children: .combine)
                }
            }
            .frame(maxHeight: .infinity)
            .studioCard(padding: 0)
        } else {
            HStack(alignment: .top, spacing: narrow ? 8 : 12) {
                ForEach(Array(figures.enumerated()), id: \.offset) { _, figure in
                    activityValue(figure.name, figure.number, note: figure.note, compact: narrow, color: figure.color)
                        .help(figure.help ?? "")
                }
            }
            .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func activityValue(_ name: String, _ number: String, note: String, compact: Bool, color: Color = StudioTheme.ink) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(name).font(.system(size: 12, weight: .medium)).foregroundStyle(StudioTheme.secondary)
            Text(number).font(.system(size: compact ? 17 : 20, weight: .semibold)).monospacedDigit()
                .foregroundStyle(color)
                .lineLimit(1).minimumScaleFactor(0.65)
            Text(note).font(.system(size: 12)).foregroundStyle(StudioTheme.quiet)
                .lineLimit(1).minimumScaleFactor(0.8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .studioCard(padding: compact ? 12 : 15)
        .accessibilityElement(children: .combine)
    }

    private func duration(_ seconds: Double) -> String {
        if seconds < 1 {
            let milliseconds = (seconds * 1_000).formatted(.number.precision(.fractionLength(0)).locale(locale))
            return seconds < 0.001 ? "< 1 ms" : "\(milliseconds) ms"
        }
        return seconds.formatted(.number.precision(.fractionLength(0...2)).locale(locale)) + " s"
    }

    private func modelTable(_ models: [LabModelUsage], width: CGFloat) -> some View {
        let column = min(104, max(60, width * 0.13))
        let total = models.reduce(0.0) { $0 + Double($1.totals.inputTokens) + Double($1.totals.outputTokens) }
        let metricsHelp = L("Débit de décodage en tokens par seconde. Chaque génération mesurée compte une fois dans la moyenne.",
                            "Decode speed in tokens per second. Each measured generation counts once toward the average.")
        return VStack(alignment: .leading, spacing: 0) {
            HStack(alignment: .firstTextBaseline, spacing: 10) {
                Text(L("Modèle", "Model")).frame(maxWidth: .infinity, alignment: .leading)
                Text("Tokens").frame(width: column, alignment: .trailing)
                Text(L("Moy. tok/s", "Avg. tok/s")).frame(width: column, alignment: .trailing)
                Text(L("Max. tok/s", "Max. tok/s")).frame(width: column, alignment: .trailing)
            }
            .font(.system(size: 11.5, weight: .medium)).foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 16).padding(.vertical, 10).help(metricsHelp)
            ForEach(models) { model in
                Rectangle().fill(StudioTheme.edge.opacity(0.7)).frame(height: 0.5).accessibilityHidden(true)
                let stats = model.totals
                let fraction = total > 0 ? (Double(stats.inputTokens) + Double(stats.outputTokens)) / total : 0
                let share = Self.formattedShare(fraction, locale: locale)
                VStack(alignment: .leading, spacing: 7) {
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(model.model).fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                            .help(model.model).frame(maxWidth: .infinity, alignment: .leading)
                        Text(stats.totalTokens, format: .number).fontWeight(.semibold)
                            .monospacedDigit().lineLimit(1).minimumScaleFactor(0.65)
                            .frame(width: column, alignment: .trailing)
                        rate(stats.decode?.average).frame(width: column, alignment: .trailing)
                        rate(stats.decode?.maximum).frame(width: column, alignment: .trailing)
                    }
                    .font(.system(size: 13.5))
                    let input = value(stats.inputTokens), output = value(stats.outputTokens)
                    let detail = generationCount(stats.generations) + " · " +
                        L("\(input) en entrée · \(output) en sortie", "\(input) input · \(output) output")
                    HStack(alignment: .center, spacing: 10) {
                        Text(detail).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
                        Text(share).monospacedDigit().frame(width: column, alignment: .trailing)
                            .help(L("Part des tokens sur la période", "Share of tokens in this period"))
                        // The same share as a bar, so the ranking reads without the numbers.
                        GeometryReader { geometry in
                            ZStack(alignment: .leading) {
                                Capsule().fill(StudioTheme.contrast.opacity(0.07))
                                Capsule().fill(StudioTheme.secondary)
                                    .frame(width: max(fraction > 0 ? 3 : 0, geometry.size.width * fraction))
                            }
                        }
                        .frame(width: 2 * column + 10, height: 4)
                        .accessibilityHidden(true)
                    }
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                }
                .padding(.horizontal, 16).padding(.vertical, 12)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(model.model)
                .accessibilityValue(spokenStats(stats, share: share))
                .help(model.model + " · " + L("Débit mesuré sur \(generationCount(stats.decode?.count ?? 0)).",
                                              "Speed measured over \(generationCount(stats.decode?.count ?? 0))."))
            }
        }
        .studioCard(padding: 0)
    }

    private func speed(_ rate: Double?) -> String? {
        rate?.formatted(.number.locale(locale).precision(.fractionLength(1)))
    }

    private func rate(_ rate: Double?) -> some View {
        Text(speed(rate) ?? "—").monospacedDigit().lineLimit(1).minimumScaleFactor(0.65)
            .foregroundStyle(rate == nil ? StudioTheme.secondary : StudioTheme.ink)
    }

    private func spokenStats(_ stats: LabUsageTotals, share: String) -> String {
        let counts = L("\(value(stats.totalTokens)) tokens, \(value(stats.inputTokens)) en entrée, \(value(stats.outputTokens)) en sortie",
                       "\(value(stats.totalTokens)) tokens, \(value(stats.inputTokens)) input, \(value(stats.outputTokens)) output")
        let speeds: String
        if let average = speed(stats.decode?.average), let maximum = speed(stats.decode?.maximum) {
            speeds = L("Débit moyen \(average), maximum \(maximum) tokens par seconde",
                       "Average decode speed \(average), maximum \(maximum) tokens per second")
        } else {
            speeds = L("Débit non mesuré", "Speed not measured")
        }
        return [generationCount(stats.generations), counts, L("\(share) du total", "\(share) of total"), speeds].joined(separator: ", ")
    }
}

/// Tokens per civil day: input and output stacked, on a regular time axis. Pointing at a
/// day reads its date and counts in the header, so no bar needs a label.
private struct LabDailyChart: View {
    let days: [LabUsageDay]
    let outputColor: Color
    @Environment(\.locale) private var locale
    @State private var hovered: Int?

    /// The least the bars get; the card grows with the figures beside it.
    private static let height: CGFloat = 86

    private func value(_ count: Int) -> String { count.formatted(.number.locale(locale)) }

    var body: some View {
        let peak = max(1, days.map(\.tokens).max() ?? 1)
        let shown = hovered.flatMap { days.indices.contains($0) ? days[$0] : nil }
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .firstTextBaseline, spacing: 12) {
                Text(L("Activité quotidienne", "Daily activity")).font(.system(size: 12, weight: .medium))
                    .foregroundStyle(StudioTheme.secondary)
                Spacer(minLength: 0)
                Group {
                    if let shown {
                        Text(shown.formatted(locale: locale) + " · "
                             + L("\(value(shown.inputTokens)) en entrée · \(value(shown.outputTokens)) en sortie",
                                 "\(value(shown.inputTokens)) input · \(value(shown.outputTokens)) output"))
                            .foregroundStyle(StudioTheme.ink)
                    } else {
                        Text(L("Pic : \(value(peak)) tokens", "Peak: \(value(peak)) tokens"))
                            .foregroundStyle(StudioTheme.quiet)
                    }
                }
                .font(.system(size: 12)).monospacedDigit().lineLimit(1)
            }
            GeometryReader { geometry in
                let count = CGFloat(days.count)
                let gap: CGFloat = days.count > 45 ? 1 : days.count > 14 ? 2 : 5
                let bar = max(1, (geometry.size.width - gap * (count - 1)) / count)
                let full = max(Self.height, geometry.size.height)
                HStack(alignment: .bottom, spacing: gap) {
                    ForEach(Array(days.enumerated()), id: \.offset) { index, day in
                        let total = full * CGFloat(Double(day.tokens) / Double(peak))
                        let output = day.tokens > 0 ? total * CGFloat(Double(day.outputTokens) / Double(day.tokens)) : 0
                        VStack(spacing: 0) {
                            Spacer(minLength: 0)
                            if day.tokens > 0 {
                                // An active day never rounds down to nothing.
                                Rectangle().fill(outputColor).frame(height: max(0, output))
                                Rectangle().fill(StudioTheme.secondary).frame(height: max(day.outputTokens == 0 ? 2 : 0, total - output))
                            } else {
                                Rectangle().fill(StudioTheme.contrast.opacity(0.09)).frame(height: 1)
                            }
                        }
                        .frame(width: bar, height: full)
                        .clipShape(UnevenRoundedRectangle(topLeadingRadius: min(3, bar / 2), topTrailingRadius: min(3, bar / 2)))
                        .opacity(hovered == nil || hovered == index ? 1 : 0.45)
                    }
                }
                .contentShape(Rectangle())
                .onContinuousHover { phase in
                    switch phase {
                    case .active(let point):
                        let index = Int((point.x / max(1, geometry.size.width)) * count)
                        hovered = min(days.count - 1, max(0, index))
                    case .ended:
                        hovered = nil
                    }
                }
            }
            .frame(minHeight: Self.height, maxHeight: .infinity)
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(L("Activité quotidienne", "Daily activity"))
            .accessibilityValue(L("\(days.count) jours, pic de \(value(peak)) tokens", "\(days.count) days, peak of \(value(peak)) tokens"))
            HStack {
                Text(days.first?.formatted(locale: locale) ?? "")
                Spacer(minLength: 8)
                HStack(spacing: 12) {
                    legend(L("Entrée", "Input"), StudioTheme.secondary)
                    legend(L("Sortie", "Output"), outputColor)
                }
                Spacer(minLength: 8)
                Text(days.last?.formatted(locale: locale) ?? "")
            }
            .font(.system(size: 11)).foregroundStyle(StudioTheme.quiet)
        }
        .frame(maxHeight: .infinity)
        .studioCard(padding: 16)
    }

    private func legend(_ title: String, _ color: Color) -> some View {
        HStack(spacing: 5) {
            RoundedRectangle(cornerRadius: 1.5).fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
            Text(title)
        }
    }
}
