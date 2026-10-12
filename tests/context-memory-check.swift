import Foundation

@main struct ContextMemoryCheck {
    static func main() throws {
        let gib = 1_073_741_824.0
        var checks = 0
        for step in [1, 7, 256] {
            for cap in [nil, 0, 31, 100] as [Int?] {
                let profile = ContextMemoryProfile(layers: [.init(bytesPerToken: 1_048_576, maxTokens: cap, step: step)], fixedBytes: 200)
                for maximum in [1, 7, 31, 257] {
                    for available in [0.0, 1.0, 16.0, 64.0, 500.0] {
                        // Independent linear scan, including step rounding.
                        let expected = (1...maximum).filter { tokens in
                            var allocated = 0
                            while allocated < tokens { allocated += step }
                            return 200 + Double(min(allocated, cap ?? allocated)) * 1_048_576 <= available * 1_048_576
                        }.last
                        let physical = 8 * gib
                        let resident = physical - 4 * gib - 512 * 1_048_576 - available * 1_048_576
                        precondition(profile.recommendedTokens(maximum: maximum, physicalBytes: physical, residentBytes: resident) == expected)
                        checks += 1
                    }
                }
            }
        }
        let draft = ContextMemoryProfile(layers: [.init(bytesPerToken: 1_048_576, maxTokens: nil, step: 1)], fixedBytes: 0, draftBytesPerToken: 1_048_576)
        precondition(draft.recommendedTokens(maximum: 100, physicalBytes: 8 * gib, residentBytes: 3 * gib) == min(256, 100))
        precondition(draft.recommendedTokens(maximum: 1000, physicalBytes: 8 * gib, residentBytes: 3 * gib) == 256)
        let invalidDraft = ContextMemoryProfile(layers: [], fixedBytes: 0, draftBytesPerToken: -1)
        precondition(invalidDraft.recommendedTokens(maximum: 1, physicalBytes: 8 * gib, residentBytes: 0) == nil)
        let huge = ContextMemoryProfile(layers: [.init(bytesPerToken: 1, maxTokens: nil, step: Int.max)], fixedBytes: 0)
        precondition(huge.bytes(tokens: Int.max) == Double(Int.max))
        precondition(huge.bytes(tokens: 1) == Double(Int.max))
        precondition(huge.recommendedTokens(maximum: Int.max, physicalBytes: 24 * gib, residentBytes: 1) == nil)
        let zero = ContextMemoryProfile(layers: [], fixedBytes: 0)
        precondition(zero.recommendedTokens(maximum: Int.max, physicalBytes: 8 * gib, residentBytes: 0) == Int.max)
        for value in [Double.nan, .infinity, -1, 0] {
            precondition(zero.recommendedTokens(maximum: 1024, physicalBytes: value, residentBytes: 0) == nil)
        }
        for value in [Double.nan, .infinity, -1] {
            precondition(zero.recommendedTokens(maximum: 1024, physicalBytes: 8 * gib, residentBytes: value) == nil)
        }
        for invalid in [ContextMemoryProfile(layers: [], fixedBytes: -1),
                        ContextMemoryProfile(layers: [.init(bytesPerToken: -1, maxTokens: nil, step: 1)], fixedBytes: 0),
                        ContextMemoryProfile(layers: [.init(bytesPerToken: 1, maxTokens: -1, step: 1)], fixedBytes: 0)] {
            precondition(invalid.recommendedTokens(maximum: 1024, physicalBytes: 8 * gib, residentBytes: 0) == nil)
        }
        let data = Data(#"{"layers":[{"bytes_per_token":32768,"step":256}],"fixed_bytes":4096}"#.utf8)
        let decoded = try JSONDecoder().decode(ContextMemoryProfile.self, from: data)
        precondition(decoded.draftBytesPerToken == nil)
        let withDraft = try JSONDecoder().decode(ContextMemoryProfile.self, from: Data(#"{"layers":[],"fixed_bytes":0,"draft_bytes_per_token":4096}"#.utf8))
        precondition(withDraft.draftBytesPerToken == 4096)
        precondition(decoded.bytes(tokens: 257) == 4096 + 512 * 32768)
        print("Context memory checks passed: \(checks) oracle cases, overflow/invalid/zero/maximum and bridge decoding")
    }
}
