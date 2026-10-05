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
        print("MTP tuning checks passed: 64 grids/order/ties, noise boundary, collapsed acceptance, invalid rates/counts/hashes, decoding")
    }
}
