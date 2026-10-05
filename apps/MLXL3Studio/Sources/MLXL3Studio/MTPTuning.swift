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
    var id: Int { depth }
    var label: String { depth == 0 ? "Baseline" : "MTP\(depth)" }
    enum CodingKeys: String, CodingKey {
        case depth, eligible, reason
        case decodeTPS = "decode_tps", decodeTokens = "decode_tokens", decodeSeconds = "decode_seconds"
        case acceptedTokens = "accepted_tokens", proposedTokens = "proposed_tokens", tokenHashes = "token_hashes"
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

struct MTPConfigurationKey: Codable, Equatable, Sendable {
    let modelPath: String
    let headPath: String
    let headRevision: String
    let runtime: String

    init(modelPath: String, headPath: String, runtime: String) {
        self.modelPath = URL(fileURLWithPath: modelPath).resolvingSymlinksInPath().path
        let head = URL(fileURLWithPath: headPath).resolvingSymlinksInPath()
        self.headPath = head.path
        self.runtime = runtime
        let files = (try? FileManager.default.contentsOfDirectory(at: head, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])) ?? []
        self.headRevision = files.filter { ["json", "safetensors"].contains($0.pathExtension) }
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
            guard row.tokenHashes == baseline.tokenHashes, let score = row.score else { continue }
            let candidate = score.multipliedFullWidth(by: 100)
            let threshold = base.multipliedFullWidth(by: 103)
            if score > fastest && (candidate.high > threshold.high
                || (candidate.high == threshold.high && candidate.low > threshold.low)) {
                best = row.depth; fastest = score
            }
        }
        return best
    }

    static func load(key: MTPConfigurationKey, preferences: UserDefaults) -> MTPSelection? {
        guard let data = preferences.data(forKey: "studio.mtpSelections"),
              let records = try? JSONDecoder().decode([MTPSelection].self, from: data) else { return nil }
        return records.first { record in
            record.key == key && (0...3).contains(record.depth)
                && (record.rows.isEmpty || winner(record.rows) != nil)
        }
    }

    static func save(_ selection: MTPSelection, preferences: UserDefaults) {
        var records = preferences.data(forKey: "studio.mtpSelections")
            .flatMap { try? JSONDecoder().decode([MTPSelection].self, from: $0) } ?? []
        records.removeAll { $0.key == selection.key }
        records.append(selection)
        if let data = try? JSONEncoder().encode(Array(records.suffix(64))) {
            preferences.set(data, forKey: "studio.mtpSelections")
        }
    }
}
