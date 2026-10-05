import Foundation

@main struct DownloadProgressCheck {
    static func main() {
        var progress = ModelDownloadProgress(total: 12_000_000_000)
        precondition(progress.completed == 0 && progress.bytesPerSecond == nil && progress.fraction == 0)
        // Initial retained bytes are a baseline, never a spike in transfer rate.
        progress.update(completed: 4_000_000_000, total: 12_000_000_000, at: 100)
        precondition(progress.bytesPerSecond == nil)
        progress.update(completed: 4_050_000_000, total: 12_000_000_000, at: 100.25)
        precondition(progress.bytesPerSecond == nil)
        progress.update(completed: 4_100_000_000, total: 12_000_000_000, at: 100.5)
        precondition(progress.bytesPerSecond == 200_000_000)
        progress.update(completed: 4_100_000_000, total: 12_000_000_000, at: 101)
        precondition(progress.bytesPerSecond == 0, "A stalled transfer still reports speed")
        progress.update(completed: 100, total: 100, at: 102)
        precondition(progress.bytesPerSecond == nil && progress.fraction == 1, "Counter reset counted retained data")
        progress.update(completed: 200, total: 100, at: 103)
        precondition(progress.fraction == 1)
        progress.update(completed: 210, total: 300, at: 1)
        precondition(progress.bytesPerSecond == nil, "Backwards time produced a rate")
        for value in [-1.0, Double.nan, .infinity, -.infinity] {
            progress.update(completed: value, total: 300, at: 2)
            progress.update(completed: 250, total: value, at: 2)
            progress.update(completed: 250, total: 300, at: value)
            precondition(progress.completed == 210 && progress.total == 300 && progress.bytesPerSecond == nil)
        }
        var unknown = ModelDownloadProgress(total: .nan)
        precondition(unknown.total == 0 && unknown.fraction == nil)
        unknown.update(completed: 10, total: 0, at: 1)
        precondition(unknown.fraction == nil)
        unknown.update(completed: 20, total: 100, at: 2)
        precondition(unknown.fraction == 0.2 && unknown.bytesPerSecond == 10)
        var huge = ModelDownloadProgress(total: Double.greatestFiniteMagnitude)
        huge.update(completed: 0, total: Double.greatestFiniteMagnitude, at: 0)
        huge.update(completed: Double.greatestFiniteMagnitude, total: Double.greatestFiniteMagnitude, at: 0.5)
        precondition(huge.fraction == 1 && huge.bytesPerSecond == nil, "Overflow displayed an infinite rate")
        var cases = 0
        for initial in [0.0, 1, 1_000_000, 12_000_000_000] {
            for received in [0.0, 1, 500_000, 6_000_000_000, 12_000_000_001] {
                for total in [0.0, 1, 12_000_000_000] {
                    for elapsed in [0.0, 0.25, 0.5, 1, 10] {
                        var value = ModelDownloadProgress(total: total)
                        value.update(completed: initial, total: total, at: 10)
                        value.update(completed: received, total: total, at: 10 + elapsed)
                        precondition(value.completed == received && value.total == total)
                        if let fraction = value.fraction { precondition(fraction.isFinite && (0...1).contains(fraction)) }
                        if let rate = value.bytesPerSecond { precondition(rate.isFinite && rate >= 0) }
                        cases += 1
                    }
                }
            }
        }
        print("Download progress checks passed: rate/resume/stall/reset/unknown/invalid/overflow and \(cases) finite cases")
    }
}
