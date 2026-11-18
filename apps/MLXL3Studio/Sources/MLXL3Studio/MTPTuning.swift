import Foundation

struct MTPTuningRow: Codable, Sendable, Identifiable {
    let depth: Int
    let decodeTPS: Double
    let decodeTokens: Int
    let decodeSeconds: Double
    let acceptedTokens: Int
    let proposedTokens: Int
    let eligible: Bool
    let reason: String?
    let tokenHashes: [String]
    /// Divergences from the baseline accepted as near ties (engine 2.0.19+).
    var nearTies: [MTPNearTie]? = nil
    /// Measured with the chat sampling settings instead of greedy decoding (engine 2.0.19+).
    var sampled: Bool? = nil
    var id: Int { depth }
    var label: String { depth == 0 ? "Baseline" : "MTP\(depth)" }
    var dflashLabel: String { ["Baseline", "Auto", "2 tokens", "7 tokens"][min(3, max(0, depth))] }
    enum CodingKeys: String, CodingKey {
        case depth, eligible, reason
        case decodeTPS = "decode_tps", decodeTokens = "decode_tokens", decodeSeconds = "decode_seconds"
        case acceptedTokens = "accepted_tokens", proposedTokens = "proposed_tokens", tokenHashes = "token_hashes"
        case nearTies = "near_ties", sampled
    }

    var score: UInt64? {
        guard eligible, (0...3).contains(depth), decodeTPS.isFinite, decodeTPS > 0,
              decodeTPS < Double(UInt64.max) / 1000, decodeTokens >= 62,
              decodeSeconds.isFinite, decodeSeconds > 0,
              abs(decodeTPS - Double(decodeTokens) / decodeSeconds) < max(1e-6, decodeTPS * 1e-6),
              acceptedTokens >= 0, proposedTokens >= acceptedTokens,
              depth == 0 || acceptedTokens > 0 else { return nil }
        let scaled = decodeTPS * 1000
        guard scaled < Double(UInt64.max) else { return nil }
        let score = UInt64(scaled)
        return score > 0 ? score : nil
    }
}

struct MTPNearTie: Codable, Sendable, Equatable {
    let prompt: Int
    let position: Int
    let gap: Double
}

struct MTPConfigurationKey: Codable, Hashable, Sendable {
    let modelPath: String
    let headPath: String
    let headRevision: String
    let runtime: String

    init(modelPath: String, headPath: String, runtime: String, packedDraft: Bool = false) {
        self.modelPath = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath().path
        let head = URL(fileURLWithPath: headPath).resolvingSymlinksInPath()
        self.headPath = head.path
        self.runtime = runtime
        let files = (try? FileManager.default.contentsOfDirectory(at: packedDraft ? head.appendingPathComponent("draft") : head, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])) ?? []
        self.headRevision = files.filter { ["json", "safetensors", "bin"].contains($0.pathExtension) }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }.map { file in
                let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
                return "\(file.lastPathComponent):\(values?.fileSize ?? -1):\(values?.contentModificationDate?.timeIntervalSince1970 ?? -1)"
            }.joined(separator: "|")
    }
}

struct MTPSelection: Codable, Sendable {
    let key: MTPConfigurationKey
    let depth: Int // 0 = baseline; 1..3 = enabled MTP
    let rows: [MTPTuningRow]
    let tunedAt: Date?
}

enum MTPTuning {
    static func winner(_ rows: [MTPTuningRow]) -> Int? {
        guard rows.count == 4, Set(rows.map(\.depth)) == Set(0...3),
              let baseline = rows.first(where: { $0.depth == 0 }), let base = baseline.score,
              baseline.tokenHashes.count == 2, baseline.tokenHashes.allSatisfy({ !$0.isEmpty }) else { return nil }
        var best = 0
        var fastest = base
        for row in rows.sorted(by: { $0.depth < $1.depth }) where row.depth > 0 {
            // Engines 2.0.19+ flag the rows whose tokens differ by design: a sampled measurement
            // (random draws) or a divergence they verified as a near tie. Others must match.
            let divergenceAllowed = row.sampled == true || !(row.nearTies ?? []).isEmpty
            guard row.tokenHashes == baseline.tokenHashes || divergenceAllowed,
                  let score = row.score else { continue }
            let candidate = score.multipliedFullWidth(by: 100)
            let threshold = base.multipliedFullWidth(by: 103)
            if score > fastest && (candidate.high > threshold.high
                || (candidate.high == threshold.high && candidate.low > threshold.low)) {
                best = row.depth; fastest = score
            }
        }
        return best
    }

    static func load(key: MTPConfigurationKey, preferences: UserDefaults, storageKey: String = "studio.mtpSelections") -> MTPSelection? {
        guard let data = preferences.data(forKey: storageKey),
              let records = try? JSONDecoder().decode([MTPSelection].self, from: data) else { return nil }
        return records.first { record in
            record.key == key && (0...3).contains(record.depth)
                && (record.rows.isEmpty || winner(record.rows) != nil)
        }
    }

    static func save(_ selection: MTPSelection, preferences: UserDefaults, storageKey: String = "studio.mtpSelections") {
        var records = preferences.data(forKey: storageKey)
            .flatMap { try? JSONDecoder().decode([MTPSelection].self, from: $0) } ?? []
        records.removeAll { $0.key == selection.key }
        records.append(selection)
        if let data = try? JSONEncoder().encode(Array(records.suffix(64))) {
            preferences.set(data, forKey: storageKey)
        }
    }
}

/// Automatic Tune bookkeeping for one app run (MTP-AUTOTUNE-20261115).
/// Results persist through `MTPTuning`; failures and interruptions stay in memory,
/// so a later app run, engine build or head revision gets a fresh attempt.
struct MTPAutoTuneLedger: Sendable {
    enum Decision: Equatable, Sendable { case skip, reuse, tune }
    enum Outcome: Equatable, Sendable { case succeeded, failed, interrupted }
    /// Chat, model switches and unloads may preempt a run; stop retrying after this many.
    static let interruptionLimit = 3
    private(set) var failed: Set<MTPConfigurationKey> = []
    private(set) var interruptions: [MTPConfigurationKey: Int] = [:]

    /// A stored Tune for exactly this key: measured rows that still elect a winner.
    /// A later manual depth choice keeps the record tuned and is respected.
    static func isTuned(_ selection: MTPSelection?, for key: MTPConfigurationKey) -> Bool {
        guard let selection, selection.key == key, selection.tunedAt != nil,
              (0...3).contains(selection.depth) else { return false }
        return MTPTuning.winner(selection.rows) != nil
    }

    func decide(key: MTPConfigurationKey?, saved: MTPSelection?, eligible: Bool) -> Decision {
        guard let key else { return .skip }
        if Self.isTuned(saved, for: key) { return .reuse }
        guard eligible, !failed.contains(key),
              interruptions[key, default: 0] < Self.interruptionLimit else { return .skip }
        return .tune
    }

    mutating func record(_ outcome: Outcome, for key: MTPConfigurationKey) {
        switch outcome {
        case .succeeded:
            failed.remove(key); interruptions[key] = nil
        case .failed:
            failed.insert(key)
        case .interrupted:
            interruptions[key] = min(Self.interruptionLimit, interruptions[key, default: 0] + 1)
        }
    }
}
