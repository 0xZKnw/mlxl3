import Foundation

@main
struct ImageGenerationUsageCheck {
    enum Failure: Error { case unexpected(String) }

    static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw Failure.unexpected(message) }
    }

    static func main() async {
        do { try await exercise() } catch {
            FileHandle.standardError.write(Data((String(describing: error) + "\n").utf8))
            exit(EXIT_FAILURE)
        }
    }

    static func exercise() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent("mlxl3-image-usage-" + UUID().uuidString)
        defer { try? fm.removeItem(at: root) }
        let gallery = root.appendingPathComponent("Images")
        let store = ImageGenerationUsage(historyDirectory: gallery)
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "Europe/Paris")!
        func date(_ month: Int, _ day: Int, _ hour: Int = 0, _ minute: Int = 0) -> Date {
            calendar.date(from: DateComponents(year: 2026, month: month, day: day, hour: hour, minute: minute))!
        }
        // The March DST transition falls inside this range. Thirty local days end April 1.
        let now = date(4, 1, 0, 1)
        let initial = try store.summary(now: now, calendar: calendar)
        try require(initial == ImageUsageSummary(), "empty library")
        try store.record(id: UUID(), variant: .base, at: date(3, 2, 23, 59), calendar: calendar)
        try store.record(id: UUID(), variant: .base, at: date(3, 3), calendar: calendar)
        let preview = UUID()
        try store.record(id: preview, variant: .turbo, at: now, calendar: calendar)
        try store.record(id: preview, variant: .turbo, at: now, calendar: calendar)
        let initialCounts = try store.summary(now: now, calendar: calendar)
        try require(initialCounts.base == ImageUsageCounts(last30Days: 1, allTime: 2)
                    && initialCounts.turbo == ImageUsageCounts(last30Days: 1, allTime: 1), "inclusive cutoff or retry")
        try require(!fm.fileExists(atPath: gallery.path), "usage-only generations created an image directory")
        let shifted = try store.summary(now: date(4, 2), calendar: calendar)
        try require(shifted.base == ImageUsageCounts(last30Days: 0, allTime: 2), "rolling window did not expire at midnight")

        // Events retain the day where they were generated even if the user later changes time zones.
        var california = calendar
        california.timeZone = TimeZone(identifier: "America/Los_Angeles")!
        try require(try store.summary(now: now, calendar: california).turbo == ImageUsageCounts(last30Days: 0, allTime: 1),
                    "a local-day event crossed a time zone after recording")

        let prompt = "A private test prompt that does not belong in usage statistics."
        let png = URL(fileURLWithPath: "docs/images/studio-image21-m5.png")
        let history = GeneratedImageHistory(directory: gallery)
        func save(_ id: UUID = UUID(), _ variant: Image21Variant, _ at: Date) throws -> GeneratedImageRecord {
            let temporary = root.appendingPathComponent(UUID().uuidString + ".png")
            try fm.copyItem(at: png, to: temporary)
            return try history.save(png: temporary, prompt: prompt, modelName: "Fixture", variant: variant,
                                    size: 512, steps: 8, seed: 23, id: id, createdAt: at)
        }
        func metadata(_ record: GeneratedImageRecord) -> URL {
            gallery.appendingPathComponent("." + record.fileURL.lastPathComponent + ".json")
        }
        _ = try save(preview, .turbo, now) // later saving an already counted preview
        let legacy = try save(UUID(), .base, date(2, 1))
        let legacyMeta = metadata(legacy)
        var fields = try JSONSerialization.jsonObject(with: Data(contentsOf: legacyMeta)) as! [String: Any]
        fields["version"] = 1
        fields.removeValue(forKey: "variant")
        try JSONSerialization.data(withJSONObject: fields).write(to: legacyMeta)
        _ = try save(UUID(), .turbo, date(3, 9))
        let unknown = try save(UUID(), .turbo, now)
        try fm.removeItem(at: metadata(unknown))
        let bad = try save(UUID(), .base, now)
        var ambiguous = try JSONSerialization.jsonObject(with: Data(contentsOf: metadata(bad))) as! [String: Any]
        ambiguous["variant"] = "unknown-model"
        try JSONSerialization.data(withJSONObject: ambiguous).write(to: metadata(bad))
        try fm.copyItem(at: png, to: gallery.appendingPathComponent("my-own-photo.png"))

        let migrated = try store.summary(now: now, calendar: calendar)
        try require(migrated.base == ImageUsageCounts(last30Days: 1, allTime: 3)
                    && migrated.turbo == ImageUsageCounts(last30Days: 2, allTime: 2)
                    && migrated.unclassifiedHistoryImages == 2,
                    "history backfill, preview later saved, or invented attribution")
        try require(try store.summary(now: now, calendar: calendar) == migrated, "reopening the Lab counted the gallery twice")
        try fm.removeItem(at: gallery)
        let reopened = ImageGenerationUsage(historyDirectory: gallery)
        let afterRemoval = try reopened.summary(now: now, calendar: calendar)
        try require(afterRemoval.base == migrated.base && afterRemoval.turbo == migrated.turbo
                    && afterRemoval.unclassifiedHistoryImages == 0, "persisted counts or live gallery count after deletion")
        try Data().write(to: gallery)
        let inaccessible = try reopened.summary(now: now, calendar: calendar)
        try require(inaccessible.base == migrated.base && inaccessible.turbo == migrated.turbo
                    && inaccessible.historyUnavailable, "an inaccessible gallery hid recorded usage")
        do {
            _ = try await Task { [calendar] in
                withUnsafeCurrentTask { $0?.cancel() }
                return try reopened.summary(now: now, calendar: calendar)
            }.value
            throw Failure.unexpected("cancelled summary completed")
        } catch is CancellationError {}
        try fm.removeItem(at: gallery)
        try require(try reopened.summary(now: now, calendar: calendar) == afterRemoval,
                    "gallery recovery left a stale warning")
        let lastFiles = try fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)
        try require(lastFiles.count == 1, "persistent image or metadata remains alongside the ledger")
        let dbData = try Data(contentsOf: lastFiles[0])
        let sampleData = try Data(contentsOf: png)
        try require(dbData.range(of: Data(prompt.utf8)) == nil && dbData.range(of: sampleData.prefix(32)) == nil,
                    "image data or prompt was written to usage storage")

        let shared = UUID()
        let eventDate = now
        let localCalendar = calendar
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<12 {
                group.addTask {
                    try reopened.record(id: shared, variant: .base, at: eventDate, calendar: localCalendar)
                    try reopened.record(id: UUID(), variant: .turbo, at: eventDate, calendar: localCalendar)
                    _ = try reopened.summary(now: eventDate, calendar: localCalendar)
                }
            }
            try await group.waitForAll()
        }
        let concurrentCounts = try reopened.summary(now: now, calendar: calendar)
        try require(concurrentCounts.base.allTime == 4 && concurrentCounts.turbo.allTime == 14,
                    "concurrent inserts lost or duplicated a generation")

        do {
            try store.record(id: UUID(), variant: .base, at: Date(timeIntervalSince1970: .nan))
            throw Failure.unexpected("non-finite time accepted")
        } catch let failure as Failure { throw failure } catch {}
        try require(try reopened.summary(now: now, calendar: calendar) == concurrentCounts, "invalid timestamp mutated counts")
        try Data("damaged database".utf8).write(to: lastFiles[0])
        do {
            _ = try reopened.summary()
            throw Failure.unexpected("corrupt ledger was reported as empty")
        } catch let failure as Failure { throw failure } catch {}
        try require(try Data(contentsOf: lastFiles[0]) == Data("damaged database".utf8), "damaged ledger overwritten")
        print("Image usage passed: two variants across DST and midnight, time-zone change, durable previews, idempotent history migration, unknown PNGs, concurrent writes and corrupt data")
    }
}
