import Foundation
import SQLite3

struct ImageUsageCounts: Equatable, Sendable {
    var last30Days = 0
    var allTime = 0
}

struct ImageUsageSummary: Equatable, Sendable {
    var base = ImageUsageCounts()
    var turbo = ImageUsageCounts()
    var unclassifiedHistoryImages = 0
    var historyUnavailable = false
}

/// The ledger lives beside the gallery so removing PNGs never removes their counts.
/// Only the generation UUID, local civil day, and model variant are recorded.
struct ImageGenerationUsage: Sendable {
    let historyDirectory: URL

    private var databaseURL: URL {
        historyDirectory.deletingLastPathComponent()
            .appendingPathComponent(historyDirectory.lastPathComponent + ".usage.sqlite3")
    }

    func record(id: UUID, variant: Image21Variant, at date: Date = .now, calendar: Calendar = .current) throws {
        guard let day = LabUsagePeriod.day(date, calendar: calendar) else { throw CocoaError(.coderInvalidValue) }
        let db = try Database(url: databaseURL)
        try db.withInsert { insert in try insert(id, variant, day) }
    }

    func summary(now: Date = .now, calendar: Calendar = .current) throws -> ImageUsageSummary {
        try Task.checkCancellation()
        guard let range = LabUsagePeriod.thirtyDays.days(at: now, calendar: calendar) else {
            throw CocoaError(.coderInvalidValue)
        }
        let db = try Database(url: databaseURL)
        let history = GeneratedImageHistory(directory: historyDirectory)
        // Skip decoding already-counted pictures and sidecars. Reconcile on each read so a
        // completed save is recoverable after an interruption between the save and the count.
        let known = try db.knownIDs()
        let records: [GeneratedImageRecord]
        do {
            records = try history.records(excluding: known)
        } catch is CancellationError {
            throw CancellationError()
        } catch {
            try Task.checkCancellation()
            var summary = try db.counts(in: range)
            summary.historyUnavailable = true
            return summary
        }
        if records.contains(where: { $0.variant != nil }) {
            try db.execute("BEGIN IMMEDIATE")
            do {
                try db.withInsert { insert in
                    for record in records {
                        try Task.checkCancellation()
                        guard let variant = record.variant, let identity = GeneratedImageHistory.filenameIdentity(record.fileURL),
                              let day = LabUsagePeriod.day(record.createdAt, calendar: calendar) else { continue }
                        try insert(identity.id, variant, day)
                    }
                }
                try db.execute("COMMIT")
            } catch {
                try? db.execute("ROLLBACK")
                throw error
            }
        }
        var summary = try db.counts(in: range)
        // This is the number of unclassified Studio pictures still in the gallery.
        // A missing sidecar can appear later, so do not persist a guess in the ledger.
        summary.unclassifiedHistoryImages = records.count {
            $0.variant == nil && GeneratedImageHistory.filenameIdentity($0.fileURL) != nil
        }
        return summary
    }

    private final class Database {
        private let handle: OpaquePointer

        init(url: URL) throws {
            let parent = url.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
            // macOS temporary and home directories may contain system symlinks. Resolve
            // the existing parent, then let SQLite reject symlinks at the ledger itself.
            guard let resolved = realpath(parent.path, nil) else { throw POSIXError(.init(rawValue: errno) ?? .EIO) }
            defer { free(resolved) }
            let path = String(cString: resolved) + "/" + url.lastPathComponent
            var connection: OpaquePointer?
            let opened = sqlite3_open_v2(path, &connection, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOFOLLOW, nil)
            guard opened == SQLITE_OK, let connection else {
                let description = connection.map { String(cString: sqlite3_errmsg($0)) } ?? String(cString: sqlite3_errstr(opened))
                sqlite3_close(connection)
                throw NSError(domain: "MLXL3.ImageUsage", code: Int(opened),
                              userInfo: [NSLocalizedDescriptionKey: description, NSFilePathErrorKey: path])
            }
            handle = connection
            try check(sqlite3_busy_timeout(handle, 5_000))
            try execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    variant TEXT NOT NULL CHECK (variant IN ('base','turbo')),
                    day INTEGER NOT NULL
                ) WITHOUT ROWID
                """)
        }

        deinit { sqlite3_close(handle) }

        private func check(_ result: Int32) throws {
            guard result == SQLITE_OK else {
                throw NSError(domain: "MLXL3.ImageUsage", code: Int(result),
                              userInfo: [NSLocalizedDescriptionKey: String(cString: sqlite3_errmsg(handle))])
            }
        }

        func execute(_ sql: String) throws {
            try check(sqlite3_exec(handle, sql, nil, nil, nil))
        }

        private func statement<T>(_ sql: String, _ body: (OpaquePointer) throws -> T) throws -> T {
            var statement: OpaquePointer?
            try check(sqlite3_prepare_v2(handle, sql, -1, &statement, nil))
            let prepared = statement! // A fixed, non-empty SQL statement prepared successfully.
            defer { sqlite3_finalize(prepared) }
            return try body(prepared)
        }

        func withInsert(_ body: ((UUID, Image21Variant, Int) throws -> Void) throws -> Void) throws {
            try statement("INSERT INTO generations(id,variant,day) VALUES (?,?,?) ON CONFLICT(id) DO NOTHING") { stmt in
                try body { id, variant, day in
                    let copyText = unsafeBitCast(-1, to: sqlite3_destructor_type.self)
                    try self.check(sqlite3_bind_text(stmt, 1, id.uuidString, -1, copyText))
                    try self.check(sqlite3_bind_text(stmt, 2, variant.rawValue, -1, copyText))
                    try self.check(sqlite3_bind_int64(stmt, 3, Int64(day)))
                    let result = sqlite3_step(stmt)
                    if result != SQLITE_DONE { try self.check(result) }
                    try self.check(sqlite3_reset(stmt))
                }
            }
        }

        func knownIDs() throws -> Set<UUID> {
            try statement("SELECT id FROM generations") { stmt in
                var ids = Set<UUID>()
                while try next(stmt) {
                    guard let text = sqlite3_column_text(stmt, 0), let id = UUID(uuidString: String(cString: text)) else {
                        throw CocoaError(.coderReadCorrupt)
                    }
                    ids.insert(id)
                }
                return ids
            }
        }

        private func next(_ stmt: OpaquePointer) throws -> Bool {
            let result = sqlite3_step(stmt)
            if result == SQLITE_ROW { return true }
            if result != SQLITE_DONE { try check(result) }
            return false
        }

        func counts(in range: ClosedRange<Int>) throws -> ImageUsageSummary {
            try statement("SELECT variant, SUM(day BETWEEN ? AND ?), COUNT(*) FROM generations GROUP BY variant") { stmt in
                try check(sqlite3_bind_int64(stmt, 1, Int64(range.lowerBound)))
                try check(sqlite3_bind_int64(stmt, 2, Int64(range.upperBound)))
                var summary = ImageUsageSummary()
                while try next(stmt) {
                    guard let text = sqlite3_column_text(stmt, 0), let variant = Image21Variant(rawValue: String(cString: text)) else {
                        throw CocoaError(.coderReadCorrupt)
                    }
                    let counts = ImageUsageCounts(last30Days: Int(sqlite3_column_int64(stmt, 1)),
                                                  allTime: Int(sqlite3_column_int64(stmt, 2)))
                    switch variant {
                    case .base: summary.base = counts
                    case .turbo: summary.turbo = counts
                    }
                }
                return summary
            }
        }
    }
}
