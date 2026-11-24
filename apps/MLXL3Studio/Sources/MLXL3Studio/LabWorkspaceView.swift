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
                                usage(studio.labUsage.summary(for: period, now: time.date), width: width)
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

    private func usage(_ summary: LabUsageSummary, width: CGFloat) -> some View {
        let total = summary.totals
        return VStack(alignment: .leading, spacing: 28) {
            HStack(alignment: .center, spacing: 24) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(L("Tokens utilisés", "Tokens used"))
                        .font(.system(size: 13, weight: .medium)).foregroundStyle(StudioTheme.secondary)
                    Text(value(total.totalTokens))
                        .font(.system(size: width < 555 ? 42 : 54, weight: .medium))
                        .monospacedDigit().lineLimit(1).minimumScaleFactor(0.46)
                        .accessibilityIdentifier("lab.usage.total")
                    Text(generationCount(total.generations))
                        .font(.system(size: 13)).foregroundStyle(StudioTheme.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                Grid(alignment: .leading, horizontalSpacing: 18, verticalSpacing: 8) {
                    tokenLine(L("Entrée", "Input"), count: total.inputTokens)
                    tokenLine(L("Sortie", "Output"), count: total.outputTokens)
                    if let cached = total.cachedInputTokens, cached > 0 {
                        tokenLine(L("Entrée en cache", "Cached input"), count: cached)
                    }
                }
                .accessibilityElement(children: .contain)
            }
            Divider()
            VStack(alignment: .leading, spacing: 12) {
                StudioCaption(french: "Par modèle", english: "By model", colored: false)
                    .accessibilityAddTraits(.isHeader)
                if summary.models.isEmpty {
                    Text(L("Aucune génération enregistrée sur cette période.", "No generations recorded in this period."))
                        .font(.callout).foregroundStyle(StudioTheme.secondary)
                        .padding(.vertical, 6)
                } else {
                    modelTable(summary.models, width: width)
                }
            }
        }
    }

    private func tokenLine(_ label: String, count: Int) -> some View {
        GridRow {
            Text(label).foregroundStyle(StudioTheme.secondary)
            Text(count, format: .number).monospacedDigit().fontWeight(.medium).gridColumnAlignment(.trailing)
        }
    }

    private func modelTable(_ models: [LabModelUsage], width: CGFloat) -> some View {
        let column = min(110, max(64, width * 0.145))
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
                    Text(generationCount(stats.generations) + "  ·  " +
                         L("\(input) en entrée · \(output) en sortie", "\(input) input · \(output) output"))
                        .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                }
                .padding(.vertical, 12)
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(model.model)
                .accessibilityValue(spokenStats(stats))
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

    private func spokenStats(_ stats: LabUsageTotals) -> String {
        let counts = L("\(value(stats.totalTokens)) tokens, \(value(stats.inputTokens)) en entrée, \(value(stats.outputTokens)) en sortie",
                       "\(value(stats.totalTokens)) tokens, \(value(stats.inputTokens)) input, \(value(stats.outputTokens)) output")
        let speeds: String
        if let average = speed(stats.decode?.average), let maximum = speed(stats.decode?.maximum) {
            speeds = L("Débit moyen \(average), maximum \(maximum) tokens par seconde",
                       "Average decode speed \(average), maximum \(maximum) tokens per second")
        } else {
            speeds = L("Débit non mesuré", "Speed not measured")
        }
        return [generationCount(stats.generations), counts, speeds].joined(separator: ", ")
    }
}
