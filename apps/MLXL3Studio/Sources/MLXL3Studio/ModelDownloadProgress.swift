import Foundation

enum ModelDownloadStatus { case idle, transferring, paused, complete, failed }

/// Transfer rate uses monotonic time and counts only newly received bytes.
struct ModelDownloadProgress {
    private(set) var completed = 0.0
    private(set) var total: Double
    private(set) var bytesPerSecond: Double?
    private var sample: (bytes: Double, time: Double)?

    init(total: Double = 0) {
        self.total = total.isFinite ? max(0, total) : 0
    }

    var fraction: Double? { total > 0 ? min(1, completed / total) : nil }

    mutating func update(completed: Double, total: Double, at time: Double) {
        guard completed.isFinite, completed >= 0, total.isFinite, total >= 0,
              time.isFinite, time >= 0 else { return }
        self.completed = completed
        self.total = total
        guard let previous = sample else {
            sample = (completed, time)
            return
        }
        if completed < previous.bytes || time < previous.time {
            bytesPerSecond = nil
            sample = (completed, time)
        } else if time - previous.time >= 0.5 {
            let rate = (completed - previous.bytes) / (time - previous.time)
            bytesPerSecond = rate.isFinite ? rate : nil
            sample = (completed, time)
        }
    }
}
