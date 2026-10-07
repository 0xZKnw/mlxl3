import Foundation

enum ModelIdleUnloadDelay: String, CaseIterable, Sendable {
    case never = "never"
    case oneMinute = "1m"
    case fiveMinutes = "5m"
    case tenMinutes = "10m"
    case fifteenMinutes = "15m"
    case thirtyMinutes = "30m"
    case oneHour = "1h"
    case twoHours = "2h"

    static let preferenceKey = "studio.modelIdleUnloadDelay"
    static let defaultValue = Self.fifteenMinutes

    static func load(from preferences: UserDefaults) -> Self {
        Self(rawValue: preferences.string(forKey: preferenceKey) ?? "") ?? defaultValue
    }

    var title: String {
        switch self {
        case .never: L("Jamais", "Never")
        case .oneMinute: L("1 minute", "1 minute")
        case .fiveMinutes: L("5 minutes", "5 minutes")
        case .tenMinutes: L("10 minutes", "10 minutes")
        case .fifteenMinutes: L("15 minutes", "15 minutes")
        case .thirtyMinutes: L("30 minutes", "30 minutes")
        case .oneHour: L("1 heure", "1 hour")
        case .twoHours: L("2 heures", "2 hours")
        }
    }

    var duration: Duration? {
        switch self {
        case .never: nil
        case .oneMinute: .seconds(60)
        case .fiveMinutes: .seconds(300)
        case .tenMinutes: .seconds(600)
        case .fifteenMinutes: .seconds(900)
        case .thirtyMinutes: .seconds(1800)
        case .oneHour: .seconds(3600)
        case .twoHours: .seconds(7200)
        }
    }
}

/// A single cancellable deadline. ContinuousClock also counts time in sleep.
/// The injected clock lets lifecycle checks advance time without waiting minutes.
@MainActor
final class ModelIdleUnloader {
    private let now: () -> ContinuousClock.Instant
    private let unload: () -> Void
    private var task: Task<Void, Never>?
    private var deadline: ContinuousClock.Instant?
    private var delay: Duration?
    private var ticket = UUID()

    init(now: @escaping () -> ContinuousClock.Instant = { .now }, unload: @escaping () -> Void) {
        self.now = now
        self.unload = unload
    }

    deinit { task?.cancel() }

    func update(isIdle: Bool, after delay: Duration?) {
        guard isIdle, let delay, delay > .zero else { cancel(); return }
        // Duplicate ready/status events must not keep an unused model alive.
        guard self.delay != delay || deadline == nil else { return }
        cancel()
        self.delay = delay
        let deadline = now().advanced(by: delay)
        self.deadline = deadline
        let ticket = self.ticket
        task = Task { @MainActor [weak self] in
            do { try await Task.sleep(until: deadline, clock: .continuous) }
            catch { return }
            guard !Task.isCancelled, let self, self.ticket == ticket else { return }
            self.checkForExpiry()
        }
    }

    func cancel() {
        ticket = UUID()
        task?.cancel()
        task = nil
        deadline = nil
        delay = nil
    }

    func checkForExpiry() {
        guard let deadline, now() >= deadline else { return }
        cancel()
        unload()
    }
}
