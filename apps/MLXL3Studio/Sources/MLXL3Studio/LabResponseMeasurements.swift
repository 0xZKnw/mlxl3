import Foundation

/// The bridge reports the delay until first visible text for a completed turn.
/// Older engines may report only a model-level first-token time.
struct LabResponseMeasurements {
    let medianFirstTokenSeconds: Double?
    let measuredCount: Int
    let retainedCount: Int

    init(samples: [LabSample], period: LabUsagePeriod, now: Date = .now, calendar: Calendar = .current) {
        var seen = Set<UUID>()
        var values: [Double] = []
        var retained = 0
        for sample in samples where sample.valid && seen.insert(sample.id).inserted {
            guard period.contains(sample, now: now, calendar: calendar) else { continue }
            retained += 1
            if let seconds = sample.stats.endToEndTTFTSeconds, seconds.isFinite, seconds > 0 {
                values.append(seconds)
            }
        }

        measuredCount = values.count
        retainedCount = retained
        guard !values.isEmpty else {
            medianFirstTokenSeconds = nil
            return
        }
        values.sort()
        let middle = values.count / 2
        if values.count.isMultiple(of: 2) {
            let lower = values[middle - 1], upper = values[middle]
            // Sum first for subnormal values; halve first if the sum would overflow.
            let sum = lower + upper
            medianFirstTokenSeconds = sum.isFinite ? sum / 2 : lower / 2 + upper / 2
        } else {
            medianFirstTokenSeconds = values[middle]
        }
    }
}
