import Foundation

enum LabUsagePeriod: String, CaseIterable, Identifiable {
    case today, sevenDays, thirtyDays, allTime
    var id: Self { self }

    // A local civil day keeps its date even if the user later changes time zones.
    // Applying the offset at the event handles both transitions of daylight saving time.
    static func day(_ date: Date, calendar: Calendar) -> Int? {
        guard date.timeIntervalSince1970.isFinite else { return nil }
        let localSeconds = date.timeIntervalSince1970 + Double(calendar.timeZone.secondsFromGMT(for: date))
        return Int(exactly: floor(localSeconds / 86_400))
    }

    fileprivate func days(endingOn last: Int) -> ClosedRange<Int> {
        let previousDays: Int
        switch self {
        case .allTime: return Int.min...Int.max
        case .today: previousDays = 0
        case .sevenDays: previousDays = 6
        case .thirtyDays: previousDays = 29
        }
        let (first, overflow) = last.subtractingReportingOverflow(previousDays)
        return (overflow ? Int.min : first)...last
    }

    fileprivate func days(at now: Date, calendar: Calendar) -> ClosedRange<Int>? {
        if self == .allTime { return Int.min...Int.max }
        return Self.day(now, calendar: calendar).map { days(endingOn: $0) }
    }

    func contains(_ sample: LabSample, now: Date = .now, calendar: Calendar = .current) -> Bool {
        guard let day = sample.localDay ?? Self.day(sample.date, calendar: calendar) else { return false }
        return days(at: now, calendar: calendar)?.contains(day) == true
    }
}

struct LabDecodeUsage: Codable, Equatable, Sendable {
    let average: Double
    let maximum: Double
    let count: Int

    static func measurement(_ stats: GenerationStats) -> Self? {
        // The engine measures tokens after the first one. A one-token reply has no decode interval.
        guard stats.generatedTokens > 1, stats.decodeTps.isFinite, stats.decodeTps > 0 else { return nil }
        return Self(average: stats.decodeTps, maximum: stats.decodeTps, count: 1)
    }

    fileprivate var valid: Bool {
        count > 0 && average.isFinite && maximum.isFinite && average > 0 && average <= maximum
    }

    fileprivate func combined(with other: Self) -> Self {
        // Combine means by measured generation count. Interpolation avoids overflowing a sum of speeds.
        let (low, high) = average <= other.average ? (self, other) : (other, self)
        let weight = Double(high.count) / (Double(low.count) + Double(high.count))
        let mean = low.average.addingProduct(high.average - low.average, weight)
        return Self(average: min(mean, high.average), maximum: max(maximum, other.maximum),
                    count: LabUsageTotals.sum(count, other.count))
    }
}

struct LabUsageTotals: Codable, Equatable, Sendable {
    var inputTokens = 0
    var outputTokens = 0
    var generations = 0
    var cachedInputTokens: Int? = 0
    var decode: LabDecodeUsage? = nil

    var totalTokens: Int { Self.sum(inputTokens, outputTokens) }

    fileprivate var valid: Bool {
        inputTokens >= 0 && outputTokens >= 0 && (inputTokens > 0 || outputTokens > 0) && generations > 0
            && (cachedInputTokens.map { $0 >= 0 && $0 <= inputTokens } ?? true)
            && (decode.map { $0.valid && $0.count <= generations } ?? true)
    }

    fileprivate mutating func addDecode(_ rate: LabDecodeUsage) {
        decode = decode?.combined(with: rate) ?? rate
    }

    fileprivate mutating func add(_ other: Self) {
        if let rate = other.decode { addDecode(rate) }
        inputTokens = Self.sum(inputTokens, other.inputTokens)
        outputTokens = Self.sum(outputTokens, other.outputTokens)
        generations = Self.sum(generations, other.generations)
        // Report a cache count only if the engine supplied the split for every generation.
        cachedInputTokens = cachedInputTokens.flatMap { left in
            other.cachedInputTokens.map { Self.sum(left, $0) }
        }
    }

    fileprivate static func sum(_ lhs: Int, _ rhs: Int) -> Int {
        let (value, overflow) = lhs.addingReportingOverflow(rhs)
        return overflow ? .max : value
    }
}

struct LabModelUsage: Identifiable {
    let model: String
    let totals: LabUsageTotals
    var id: String { model }
}

struct LabUsageDay {
    // A ledger key is a civil date, even if Studio is later opened in another time zone.
    // Represent it at UTC midnight and keep that zone when formatting the label.
    let date: Date
    let tokens: Int

    fileprivate init(day: Int, tokens: Int) {
        self.date = Date(timeIntervalSince1970: Double(day) * 86_400)
        self.tokens = tokens
    }

    func formatted(locale: Locale) -> String {
        date.formatted(Date.FormatStyle(date: .abbreviated, time: .omitted, locale: locale, timeZone: .gmt))
    }
}

struct LabUsageSummary {
    let totals: LabUsageTotals
    let models: [LabModelUsage]
    let activeDays: Int
    let busiestDay: LabUsageDay?
    let previousPeriodTokens: Int?
}

/// Daily counts live in the same atomic snapshot as the recent, detailed measurements.
/// A turn adds one generation, even when its engine counters cover several tool rounds.
struct LabUsageLedger: Codable, Sendable {
    private var days: [Int: [String: LabUsageTotals]] = [:]
    // Missing in the first token-only ledger. Its retained details can supply speeds once.
    private var includesDecodeRates: Bool? = true

    init(samples: [LabSample] = [], calendar: Calendar = .current) {
        var seen = Set<UUID>()
        for sample in samples where seen.insert(sample.id).inserted {
            record(sample, calendar: calendar)
        }
    }

    var hasActivity: Bool { days.values.contains { !$0.isEmpty } }

    var valid: Bool {
        days.values.allSatisfy { models in
            models.allSatisfy { !$0.key.isEmpty && $0.value.valid }
        }
    }

    mutating func backfillDecodeIfNeeded(from samples: [LabSample], calendar: Calendar = .current) {
        guard includesDecodeRates != true else { return }
        var seen = Set<UUID>()
        for sample in samples where seen.insert(sample.id).inserted {
            guard sample.stats.promptTokens >= 0, let rate = LabDecodeUsage.measurement(sample.stats),
                  let day = sample.localDay ?? LabUsagePeriod.day(sample.date, calendar: calendar),
                  let counts = days[day]?[sample.model],
                  (counts.decode?.count ?? 0) < counts.generations else { continue }
            days[day]?[sample.model]?.addDecode(rate)
        }
        includesDecodeRates = true
    }

    mutating func record(_ sample: LabSample, calendar: Calendar = .current) {
        let stats = sample.stats
        let input = stats.promptTokens
        let output = stats.generatedTokens
        guard !sample.model.isEmpty, input >= 0, output >= 0, input > 0 || output > 0,
              let day = sample.localDay ?? LabUsagePeriod.day(sample.date, calendar: calendar) else { return }
        // prompt_tokens already includes the cached prefix. Some older engines only
        // give the evaluated count. Either counter determines the split if consistent.
        let cached = stats.cachedPromptTokens
        let evaluated = stats.evaluatedPromptTokens
        var cacheCount: Int?
        if (cached.map { (0...input).contains($0) } ?? true),
           (evaluated.map { (0...input).contains($0) } ?? true) {
            let fromEvaluated = evaluated.map { input - $0 }
            if cached == nil || fromEvaluated == nil || cached == fromEvaluated {
                cacheCount = cached ?? fromEvaluated
            }
        }
        let tokens = LabUsageTotals(inputTokens: input, outputTokens: output, generations: 1,
                                   cachedInputTokens: cacheCount, decode: LabDecodeUsage.measurement(stats))
        days[day, default: [:]][sample.model, default: LabUsageTotals()].add(tokens)
    }

    func summary(for period: LabUsagePeriod, now: Date = .now, calendar: Calendar = .current) -> LabUsageSummary {
        var total = LabUsageTotals()
        var models: [String: LabUsageTotals] = [:]
        var activeDays = 0
        var busiest: (tokens: Int, day: Int)?
        var previousPeriodTokens: Int?
        if let range = period.days(at: now, calendar: calendar) {
            let previous = period != .allTime && range.lowerBound > Int.min
                ? period.days(endingOn: range.lowerBound - 1) : nil
            for (day, buckets) in days where !buckets.isEmpty {
                if range.contains(day) {
                    var tokens = 0
                    for (model, counts) in buckets {
                        models[model, default: LabUsageTotals()].add(counts)
                        total.add(counts)
                        tokens = LabUsageTotals.sum(tokens, counts.totalTokens)
                    }
                    activeDays += 1
                    // For the same usage prefer the most recent day, regardless of dictionary order.
                    if busiest.map({ (tokens, day) > $0 }) ?? true { busiest = (tokens, day) }
                } else if previous?.contains(day) == true {
                    for counts in buckets.values {
                        previousPeriodTokens = LabUsageTotals.sum(previousPeriodTokens ?? 0, counts.totalTokens)
                    }
                }
            }
        }
        let rows = models.map { LabModelUsage(model: $0.key, totals: $0.value) }.sorted {
            if $0.totals.totalTokens == $1.totals.totalTokens { return $0.model < $1.model }
            return $0.totals.totalTokens > $1.totals.totalTokens
        }
        return LabUsageSummary(totals: total, models: rows, activeDays: activeDays,
                               busiestDay: busiest.map { LabUsageDay(day: $0.day, tokens: $0.tokens) },
                               previousPeriodTokens: previousPeriodTokens)
    }
}
