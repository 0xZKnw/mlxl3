import Foundation

@main struct MTPTuningCheck {
    static func row(_ depth: Int, _ rate: Double, accepted: Int = 40, proposed: Int = 80,
                    eligible: Bool = true, hashes: [String] = ["a", "b"], tokens: Int = 190,
                    seconds: Double? = nil) -> MTPTuningRow {
        MTPTuningRow(depth: depth, decodeTPS: rate, decodeTokens: tokens,
            decodeSeconds: seconds ?? Double(tokens) / rate,
            acceptedTokens: depth == 0 ? 0 : accepted, proposedTokens: depth == 0 ? 0 : proposed,
            eligible: eligible, reason: nil, tokenHashes: hashes)
    }
    static func main() throws {
        for a in [20.0, 20.6, 21.0, 30.0] {
            for b in [20.0, 20.6, 21.0, 30.0] {
                for c in [20.0, 20.6, 21.0, 30.0] {
                    let rates = [20.0, a, b, c]
                    let rows = rates.enumerated().map { row($0.offset, $0.element) }
                    let candidates = (1...3).filter { rates[$0] > 20.6 }
                    let expected = candidates.sorted {
                        rates[$0] == rates[$1] ? $0 < $1 : rates[$0] > rates[$1]
                    }.first ?? 0
                    precondition(MTPTuning.winner(rows) == expected)
                    precondition(MTPTuning.winner(rows.reversed()) == expected)
                }
            }
        }
        let valid = [row(0, 50), row(1, 60), row(2, 75), row(3, 100)]
        precondition(MTPTuning.winner(valid) == 3)
        for bad in [row(3, 100, accepted: 0), row(3, 100, accepted: 90, proposed: 80),
                    row(3, 100, accepted: -1), row(3, 100, eligible: false),
                    row(3, 100, hashes: ["wrong", "b"]), row(3, 100, tokens: 61),
                    row(3, 100, seconds: 2), row(3, .infinity), row(3, .nan),
                    row(3, -1), row(3, 0), row(3, Double(UInt64.max))] {
            precondition(MTPTuning.winner(Array(valid.prefix(3)) + [bad]) == 2)
        }
        precondition(MTPTuning.winner(Array(valid.prefix(3))) == nil)
        precondition(MTPTuning.winner([valid[0], valid[1], valid[2], valid[2]]) == nil)
        precondition(MTPTuning.winner([row(0, 0)] + Array(valid.suffix(3))) == nil)
        precondition(MTPTuning.winner([row(0, 50, hashes: [])] + Array(valid.suffix(3))) == nil)
        let nearMax = row(3, (Double(UInt64.max) / 1000).nextDown)
        _ = nearMax.score // must not trap in Double -> UInt64 conversion
        let encoded = try JSONEncoder().encode(valid)
        let decoded = try JSONDecoder().decode([MTPTuningRow].self, from: encoded)
        precondition(MTPTuning.winner(decoded) == 3)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("dflash-profile-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root.appendingPathComponent("draft"), withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "io.mlxl3.dflash-profile." + UUID().uuidString
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        let weight = root.appendingPathComponent("draft/model.bin")
        try Data([1, 2]).write(to: weight)
        let key = MTPConfigurationKey(modelPath: root.path, headPath: root.path, runtime: "dflash", packedDraft: true)
        MTPTuning.save(MTPSelection(key: key, depth: 2, rows: decoded, tunedAt: nil), preferences: preferences, storageKey: "studio.dflashSelections")
        precondition(MTPTuning.load(key: key, preferences: preferences) == nil, "Draft profiles leaked into MTP")
        precondition(MTPTuning.load(key: key, preferences: preferences, storageKey: "studio.dflashSelections")?.depth == 2)
        try Data([1, 2, 3]).write(to: weight)
        let changed = MTPConfigurationKey(modelPath: root.path, headPath: root.path, runtime: "dflash", packedDraft: true)
        precondition(changed != key && MTPTuning.load(key: changed, preferences: preferences, storageKey: "studio.dflashSelections") == nil, "Replaced packed draft reused stale tuning")
        precondition(MTPTuning.load(key: key, preferences: preferences, storageKey: "studio.dflashSelections")?.depth == 2)
        precondition(decoded.map(\.dflashLabel) == ["Baseline", "Auto", "2 tokens", "7 tokens"])
        print("MTP tuning checks passed: 64 grids/order/ties, noise boundary, collapsed acceptance, invalid rates/counts/hashes, decoding")
        checkAutoTuneLedger(valid: valid)
    }

    /// Finite exhaustive check (MTP-AUTOTUNE-20261115): every history of up to five outcomes over
    /// two keys, then every key/stored-record/eligibility combination. The oracle folds the whole
    /// history instead of reusing the ledger's incremental state.
    static func checkAutoTuneLedger(valid: [MTPTuningRow]) {
        let keys = [MTPConfigurationKey(modelPath: "/nonexistent/model-a", headPath: "/nonexistent/head", runtime: "rt:mtppostnorm-v1=1:mtpevict-v1=1"),
                    MTPConfigurationKey(modelPath: "/nonexistent/model-a", headPath: "/nonexistent/head", runtime: "rt:mtppostnorm-v1=0:mtpevict-v1=1")]
        precondition(keys[0] != keys[1], "Runtime option change must change the configuration key")
        let outcomes: [MTPAutoTuneLedger.Outcome] = [.succeeded, .failed, .interrupted]
        let operations = keys.indices.flatMap { key in outcomes.map { (key, $0) } }
        let tunedAt = Date(timeIntervalSince1970: 1_800_000_000)
        func stored(_ key: MTPConfigurationKey) -> [(MTPSelection?, Bool)] {
            [(nil, false),
             (MTPSelection(key: key, depth: 2, rows: valid, tunedAt: tunedAt), true),
             (MTPSelection(key: key, depth: 0, rows: valid, tunedAt: tunedAt), true),   // baseline won
             (MTPSelection(key: key, depth: 3, rows: valid, tunedAt: tunedAt), true),   // later manual choice
             (MTPSelection(key: key, depth: 1, rows: [], tunedAt: nil), false),          // manual depth only
             (MTPSelection(key: key, depth: 2, rows: valid, tunedAt: nil), false),       // rows without a Tune date
             (MTPSelection(key: key, depth: 2, rows: Array(valid.prefix(3)), tunedAt: tunedAt), false),
             (MTPSelection(key: key, depth: 4, rows: valid, tunedAt: tunedAt), false)]
        }
        var histories: [[(Int, MTPAutoTuneLedger.Outcome)]] = [[]]
        var frontier = histories
        for _ in 0..<5 {
            frontier = frontier.flatMap { history in operations.map { history + [$0] } }
            histories += frontier
        }
        var decisions = 0, tunes = 0, reuses = 0, skips = 0, cappedSkips = 0
        for history in histories {
            var ledger = MTPAutoTuneLedger()
            for (key, outcome) in history { ledger.record(outcome, for: keys[key]) }
            for queried in [nil] + keys.indices.map(Optional.some) {
                let otherKey = keys[(queried ?? 0) == 0 ? 1 : 0]
                let candidates = queried.map { stored(keys[$0]) } ?? [(nil, false)]
                for (saved, tuned) in candidates + [(MTPSelection(key: otherKey, depth: 2, rows: valid, tunedAt: tunedAt), false)] {
                    for eligible in [false, true] {
                        let expected: MTPAutoTuneLedger.Decision
                        if let queried {
                            let lastSuccess = history.lastIndex { $0.0 == queried && $0.1 == .succeeded }
                            let mine = history[(lastSuccess.map { $0 + 1 } ?? 0)...].filter { $0.0 == queried }
                            let failed = mine.contains { $0.1 == .failed }
                            let interrupted = mine.filter { $0.1 == .interrupted }.count
                            if tuned { expected = .reuse }
                            else if !eligible || failed || interrupted >= MTPAutoTuneLedger.interruptionLimit { expected = .skip }
                            else { expected = .tune }
                            if tuned == false && eligible && !failed && interrupted >= MTPAutoTuneLedger.interruptionLimit { cappedSkips += 1 }
                        } else {
                            expected = .skip
                        }
                        let actual = ledger.decide(key: queried.map { keys[$0] }, saved: saved, eligible: eligible)
                        precondition(actual == expected, "Ledger \(history) key \(String(describing: queried)) saved \(String(describing: saved?.depth)) eligible \(eligible): \(actual) != \(expected)")
                        decisions += 1
                        switch actual { case .tune: tunes += 1; case .reuse: reuses += 1; case .skip: skips += 1 }
                    }
                }
            }
        }
        precondition(histories.count == 9331 && tunes > 0 && reuses > 0 && skips > 0 && cappedSkips > 0,
                     "Ledger exploration must reach every decision and the interruption cap")
        print("MTP auto-tune ledger checks passed: \(histories.count) histories, \(decisions) decisions (tune \(tunes), reuse \(reuses), skip \(skips), capped \(cappedSkips))")
    }
}
