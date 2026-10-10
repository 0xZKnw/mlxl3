import Foundation

struct ContextMemoryProfile: Decodable {
    struct Layer: Decodable {
        let bytesPerToken: Int
        let maxTokens: Int?
        let step: Int
        enum CodingKeys: String, CodingKey {
            case bytesPerToken = "bytes_per_token"
            case maxTokens = "max_tokens"
            case step
        }
    }
    let layers: [Layer]
    let fixedBytes: Int
    var draftBytesPerToken: Int? = nil
    enum CodingKeys: String, CodingKey {
        case layers
        case fixedBytes = "fixed_bytes"
        case draftBytesPerToken = "draft_bytes_per_token"
    }
    func bytes(tokens: Int) -> Double {
        guard tokens > 0 else { return 0 }
        guard fixedBytes >= 0 else { return .infinity }
        return layers.reduce(Double(fixedBytes)) { total, layer in
            guard layer.bytesPerToken >= 0, layer.maxTokens == nil || layer.maxTokens! >= 0 else { return .infinity }
            let step = max(layer.step, 1)
            let blocks = tokens / step + (tokens % step == 0 ? 0 : 1)
            let allocated = Double(blocks) * Double(step)
            return total + min(allocated, layer.maxTokens.map(Double.init) ?? allocated) * Double(layer.bytesPerToken)
        }
    }

    /// Conservative physical-RAM budget, not a promise about free RAM or swap.
    /// Observed engine footprint includes loaded draft and existing buffers.
    func recommendedTokens(maximum: Int, physicalBytes: Double, residentBytes: Double) -> Int? {
        guard maximum > 0, physicalBytes.isFinite, physicalBytes > 0,
              residentBytes.isFinite, residentBytes >= 0,
              draftBytesPerToken == nil || draftBytesPerToken! >= 0 else { return nil }
        let reserve = max(4 * 1_073_741_824.0, physicalBytes * 0.25)
        let budget = physicalBytes - reserve - residentBytes - 512 * 1_048_576.0
        guard budget >= 0, bytes(tokens: 1).isFinite else { return nil }
        var low = 0
        var high = maximum
        while low < high {
            let delta = high - low
            let mid = low + delta / 2 + delta % 2
            let required = bytes(tokens: mid) + Double(mid) * Double(draftBytesPerToken ?? 0)
            if required <= budget { low = mid } else { high = mid - 1 }
        }
        return low > 0 ? low : nil
    }
}
