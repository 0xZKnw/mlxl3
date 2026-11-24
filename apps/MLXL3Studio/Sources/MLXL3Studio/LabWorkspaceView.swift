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
    case usage, generations, hardware
    var id: Self { self }

    var title: String {
        switch self {
        case .usage: L("Utilisation", "Usage")
        case .generations: L("Générations", "Generations")
        case .hardware: L("Ce Mac", "This Mac")
        }
    }

    var icon: String {
        switch self {
        case .usage: "chart.bar"
        case .generations: "clock"
        case .hardware: "desktopcomputer"
        }
    }
}

struct LabWorkspaceView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.locale) private var locale
    @State private var page = LabPage.usage
    @State private var period = LabUsagePeriod.thirtyDays
    let onBack: () -> Void

    var body: some View {
        HStack(spacing: 0) {
            VStack(alignment: .leading, spacing: 0) {
                StudioSidebarHeader(title: "Lab")
                VStack(spacing: 3) {
                    ForEach(LabPage.allCases) { item in
                        Button { page = item } label: {
                            HStack(spacing: 11) {
                                Image(systemName: item.icon).frame(width: 18)
                                Text(item.title)
                                Spacer(minLength: 0)
                            }
                            .font(.system(size: 13, weight: page == item ? .semibold : .regular))
                            .padding(.horizontal, 12).frame(height: 36)
                        }
                        .buttonStyle(StudioControlStyle(emphasized: page == item))
                        .accessibilityAddTraits(page == item ? .isSelected : [])
                        .accessibilityIdentifier("lab.\(item.rawValue)")
                    }
                }
                .padding(.horizontal, 10)
                Spacer()
                StudioBackButton(action: onBack).padding(.horizontal, 18).padding(.bottom, 18)
            }
            .frame(width: studio.sidebarWidth.points, alignment: .leading)
            .modifier(StudioSidebarSurface())

            GeometryReader { geometry in
                let inset: CGFloat = geometry.size.width < 620 ? 24 : 36
                let width = min(820, max(0, geometry.size.width - 2 * inset))
                ScrollView {
                    VStack(alignment: .leading, spacing: 30) {
                        heading(width: width)
                        switch page {
                        case .usage:
                            TimelineView(.everyMinute) { time in
                                usage(studio.labUsage.summary(for: period, now: time.date),
                                      responses: LabResponseMeasurements(samples: studio.labSamples, period: period, now: time.date),
                                      width: width)
                            }
                        case .generations:
                            TimelineView(.everyMinute) { time in
                                LabGenerationsView(period: period, now: time.date)
                            }
                        case .hardware:
                            LabHardwareView()
                        }
                    }
                    .frame(width: width, alignment: .leading)
                    .padding(.vertical, 36)
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
        let wide = width >= 725
        let layout = wide ? AnyLayout(HStackLayout(alignment: .center, spacing: 20))
            : AnyLayout(VStackLayout(alignment: .leading, spacing: 18))
        return layout {
            Text(page.title).font(StudioTheme.display(39)).tracking(StudioTheme.displayTracking(-1.2))
                .accessibilityAddTraits(.isHeader)
            if page != .hardware {
                if wide { Spacer(minLength: 0) }
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

    // Sage identifies generated output in both the legend and the token split.
    private static let outputColor = Color(nsColor: StudioTheme.adaptiveColor(
        light: StudioTheme.rgb(0x426D57), dark: StudioTheme.rgb(0x9ABCA5)))

    static func formattedShare(_ fraction: Double, locale: Locale) -> String {
        let style = FloatingPointFormatStyle<Double>.Percent(locale: locale).precision(.fractionLength(0...1))
        if fraction > 0 && fraction < 0.001 { return "< " + 0.001.formatted(style) }
        if fraction > 0.999 && fraction < 1 { return "> " + 0.999.formatted(style) }
        return fraction.formatted(style)
    }

    private func usage(_ summary: LabUsageSummary, responses: LabResponseMeasurements, width: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 26) {
            if !summary.models.isEmpty {
                overview(summary, width: width)
                activity(summary, responses: responses, width: width)
                Divider()
                VStack(alignment: .leading, spacing: 9) {
                    StudioCaption(french: "Par modèle", english: "By model", colored: false)
                        .accessibilityAddTraits(.isHeader)
                    modelTable(summary.models, width: width)
                }
            } else if period != .allTime && studio.labUsage.hasActivity {
                overview(summary, width: width)
                VStack(alignment: .leading, spacing: 10) {
                    Text(L("Aucune génération enregistrée sur cette période.", "No generations recorded in this period."))
                        .foregroundStyle(StudioTheme.secondary)
                    Button(L("Afficher tout l’historique", "Show all time")) { period = .allTime }
                }
            } else {
                VStack(alignment: .leading, spacing: 9) {
                    Text(L("Aucune génération enregistrée", "No generations recorded")).font(.headline)
                    Text(L("Les statistiques apparaîtront après une première réponse dans Chat ou Agent sur ce Mac.",
                           "Your stats will appear after your first reply in Chat or Agent on this Mac."))
                        .foregroundStyle(StudioTheme.secondary)
                }
                .padding(.vertical, 12)
            }
        }
    }

    private func overview(_ summary: LabUsageSummary, width: CGFloat) -> some View {
        let totals = summary.totals
        let compact = width < 600
        let shape = RoundedRectangle(cornerRadius: 13, style: .continuous)
        return HStack(alignment: .center, spacing: compact ? 18 : 34) {
            VStack(alignment: .leading, spacing: 5) {
                Text(L("Tokens utilisés", "Tokens used")).fontWeight(.medium)
                Text(value(totals.totalTokens))
                    .font(.system(size: compact ? 41 : 54, weight: .medium))
                    .monospacedDigit().lineLimit(1).minimumScaleFactor(0.45)
                    .accessibilityIdentifier("lab.usage.total")
                let days = summary.activeDays == 1 ? L("1 jour actif", "1 active day")
                    : L("\(value(summary.activeDays)) jours actifs", "\(value(summary.activeDays)) active days")
                Text(generationCount(totals.generations) + " · " + days)
                    .foregroundStyle(StudioTheme.secondary)
                if let prior = summary.previousPeriodTokens, let description = comparison(prior) {
                    Text(description).font(.caption).foregroundStyle(StudioTheme.secondary)
                        .padding(.top, 3)
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
                        .font(.caption).foregroundStyle(StudioTheme.secondary)
                }
            }
            .frame(width: min(304, width * (compact ? 0.40 : 0.39)))
        }
        .padding(.horizontal, compact ? 17 : 24).padding(.vertical, compact ? 16 : 22)
        .background(StudioTheme.panel, in: shape)
        .overlay { shape.strokeBorder(StudioTheme.edge) }
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

    private func activity(_ summary: LabUsageSummary, responses: LabResponseMeasurements, width: CGFloat) -> some View {
        let total = summary.totals
        let average = (Double(total.outputTokens) / Double(total.generations))
            .formatted(.number.precision(.fractionLength(0...1)).locale(locale))
        let measured = responses.measuredCount
        let timingNote = measured == 1 ? L("Médiane · 1 mesure", "Median · 1 measurement")
            : L("Médiane · \(value(measured)) mesures", "Median · \(value(measured)) measurements")
        let noTiming = L("Délai non mesuré", "Time not measured")
        let gap: CGFloat = width < 600 ? 10 : 22
        return HStack(alignment: .top, spacing: gap) {
            activityValue(L("Réponse moyenne", "Avg. response"), "\(average) tokens",
                          note: L("par génération", "per generation"), compact: width < 600, color: Self.outputColor)
            Divider()
            activityValue(L("Premier texte", "First text"), responses.medianFirstTokenSeconds.map(duration) ?? "—",
                          note: measured > 0 ? timingNote : noTiming, compact: width < 600)
                .help(L("Délai médian du moteur avant le premier texte. \(value(measured)) mesures parmi \(value(responses.retainedCount)) générations détaillées conservées sur cette période.",
                        "Median time from the engine receiving a request to first text. \(value(measured)) measurements across \(value(responses.retainedCount)) retained generation details for this period."))
            Divider()
            activityValue(L("Jour le plus actif", "Busiest day"), summary.busiestDay?.formatted(locale: locale) ?? "—",
                          note: summary.busiestDay.map { "\(value($0.tokens)) tokens" } ?? "",
                          compact: width < 600)
        }
        .fixedSize(horizontal: false, vertical: true)
    }

    private func activityValue(_ name: String, _ number: String, note: String, compact: Bool, color: Color = StudioTheme.ink) -> some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(name).font(.system(size: 12, weight: .medium)).foregroundStyle(StudioTheme.secondary)
            Text(number).font(.system(size: compact ? 18 : 22, weight: .medium)).monospacedDigit()
                .foregroundStyle(color)
                .lineLimit(1).minimumScaleFactor(0.65)
            Text(note).font(.caption).foregroundStyle(StudioTheme.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
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
        let column = min(110, max(64, width * 0.145))
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
            .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
            .padding(.vertical, 9).help(metricsHelp)
            ForEach(models) { model in
                Divider()
                let stats = model.totals
                let share = Self.formattedShare((Double(stats.inputTokens) + Double(stats.outputTokens)) / total, locale: locale)
                VStack(alignment: .leading, spacing: 5) {
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(model.model).fontWeight(.medium).lineLimit(1).truncationMode(.middle)
                            .help(model.model).frame(maxWidth: .infinity, alignment: .leading)
                        Text(stats.totalTokens, format: .number).fontWeight(.semibold)
                            .monospacedDigit().lineLimit(1).minimumScaleFactor(0.65)
                            .frame(width: column, alignment: .trailing)
                        rate(stats.decode?.average).frame(width: column, alignment: .trailing)
                        rate(stats.decode?.maximum).frame(width: column, alignment: .trailing)
                    }
                    .font(.system(size: 14))
                    let input = value(stats.inputTokens), output = value(stats.outputTokens)
                    let detail = generationCount(stats.generations) + " · " +
                        L("\(input) en entrée · \(output) en sortie", "\(input) input · \(output) output")
                    HStack(alignment: .firstTextBaseline, spacing: 10) {
                        Text(detail).frame(maxWidth: .infinity, alignment: .leading)
                        Text(share).monospacedDigit().frame(width: column, alignment: .trailing)
                            .help(L("Part des tokens sur la période", "Share of tokens in this period"))
                        Color.clear.frame(width: 2 * column + 10, height: 0)
                    }
                    .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                }
                .padding(.vertical, 12)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(model.model)
                .accessibilityValue(spokenStats(stats, share: share))
                .help(model.model + " · " + L("Débit mesuré sur \(generationCount(stats.decode?.count ?? 0)).",
                                              "Speed measured over \(generationCount(stats.decode?.count ?? 0))."))
            }
        }
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
