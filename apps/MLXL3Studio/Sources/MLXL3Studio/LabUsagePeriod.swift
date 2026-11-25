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

    func days(endingOn last: Int) -> ClosedRange<Int> {
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

    func days(at now: Date, calendar: Calendar) -> ClosedRange<Int>? {
        if self == .allTime { return Int.min...Int.max }
        return Self.day(now, calendar: calendar).map { days(endingOn: $0) }
    }
}
