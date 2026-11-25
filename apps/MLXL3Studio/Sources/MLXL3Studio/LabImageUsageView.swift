import SwiftUI

struct LabImageUsageView: View {
    @ObservedObject var engine: ImageGeneration
    @Environment(\.locale) private var locale
    @State private var summary: ImageUsageSummary?
    @State private var failed = false
    @State private var attempt = 0

    private struct Query: Equatable {
        let day: Int?
        let revision: Int
        let attempt: Int
    }

    var body: some View {
        TimelineView(.everyMinute) { time in
            let now = time.date
            let calendar = Calendar.current
            let query = Query(day: LabUsagePeriod.day(now, calendar: calendar),
                              revision: engine.usageRevision, attempt: attempt)
            Group {
                if failed {
                    HStack {
                        Text(L("Statistiques d’images indisponibles.", "Image statistics unavailable."))
                        Spacer()
                        retryButton
                    }
                } else if let summary {
                    VStack(alignment: .leading, spacing: 16) {
                        counts(summary)
                        if summary.historyUnavailable {
                            HStack(spacing: 12) {
                                Text(L("L’historique est inaccessible. Certaines anciennes images peuvent manquer dans ces totaux.",
                                       "History is unavailable. Some older images may be missing from these totals."))
                                    .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                                    .fixedSize(horizontal: false, vertical: true)
                                Spacer(minLength: 0)
                                retryButton
                            }
                        } else if summary.unclassifiedHistoryImages > 0 {
                            let count = summary.unclassifiedHistoryImages.formatted(.number.locale(locale))
                            Text(summary.unclassifiedHistoryImages == 1
                                 ? L("Le mode utilisé est inconnu pour une image de l’historique.",
                                     "The mode used for one image in your history is unknown.")
                                 : L("Le mode utilisé est inconnu pour \(count) images de l’historique.",
                                     "The mode used for \(count) images in your history is unknown."))
                                .font(.system(size: 12)).foregroundStyle(StudioTheme.secondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                } else {
                    ProgressView(L("Lecture des statistiques…", "Loading statistics…")).controlSize(.small)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .studioCard(padding: 18)
            .task(id: query) {
                let store = engine.usageStore
                do {
                    let read = Task.detached(priority: .utility) { try store.summary(now: now, calendar: calendar) }
                    let result = try await withTaskCancellationHandler {
                        try await read.value
                    } onCancel: { read.cancel() }
                    guard !Task.isCancelled else { return }
                    summary = result
                    failed = false
                } catch {
                    guard !Task.isCancelled else { return }
                    failed = true
                }
            }
        }
    }

    private var retryButton: some View {
        Button(L("Réessayer", "Retry")) { attempt += 1 }.buttonStyle(.studio)
    }

    private func counts(_ summary: ImageUsageSummary) -> some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 14) {
            GridRow(alignment: .lastTextBaseline) {
                Text(L("Mode", "Mode")).frame(maxWidth: .infinity, alignment: .leading)
                Text(L("30 derniers jours", "Last 30 days")).gridColumnAlignment(.trailing)
                Text(L("Depuis le début", "All time")).gridColumnAlignment(.trailing)
            }
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(StudioTheme.secondary)
            .multilineTextAlignment(.trailing)
            Divider().gridCellUnsizedAxes(.horizontal)
            row(.base, recent: summary.base.last30Days, total: summary.base.allTime)
            Divider().gridCellUnsizedAxes(.horizontal)
            row(.turbo, recent: summary.turbo.last30Days, total: summary.turbo.allTime)
        }
    }

    private func row(_ variant: Image21Variant, recent: Int, total: Int) -> some View {
        let recentCount = recent.formatted(.number.locale(locale))
        let totalCount = total.formatted(.number.locale(locale))
        return GridRow(alignment: .firstTextBaseline) {
            Text(variant.title).font(.system(size: 14, weight: .semibold))
            Text(recentCount).foregroundStyle(LabWorkspaceView.outputColor)
            Text(totalCount)
        }
        .font(.system(size: 18, weight: .medium)).monospacedDigit()
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(variant.title)
        .accessibilityValue(L("\(recentCount) sur les 30 derniers jours, \(totalCount) depuis le début",
                              "\(recentCount) in the last 30 days, \(totalCount) all time"))
    }
}
