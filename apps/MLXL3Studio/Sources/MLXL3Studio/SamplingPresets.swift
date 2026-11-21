import Foundation

/// Official sampling settings published on the Hugging Face card (or the
/// `generation_config.json`) of a recommended catalog model. Models outside the
/// catalog intentionally have none.
struct SamplingPreset: Identifiable, Equatable, Sendable {
    let id: String
    let french: String
    let english: String
    let temperature: Double
    let topK: Int
    var topP: Double = 1
    var repetitionPenalty: Double = 1
    var presencePenalty: Double = 0

    var title: String { L(french, english) }

    var valid: Bool {
        temperature.isFinite && (0...2).contains(temperature) && (0...10000).contains(topK)
            && topP.isFinite && topP > 0 && topP <= 1
            && repetitionPenalty.isFinite && (0.1...4).contains(repetitionPenalty)
            && presencePenalty.isFinite && (-2...2).contains(presencePenalty)
    }
}

enum SamplingPresets {
    // Qwen3.6 model card: thinking general / precise coding / instruct.
    private static let qwen36: [SamplingPreset] = [
        .init(id: "thinking", french: "Réflexion · général", english: "Thinking · general",
              temperature: 1.0, topK: 20, topP: 0.95, presencePenalty: 1.5),
        .init(id: "thinking-coding", french: "Réflexion · code précis", english: "Thinking · precise coding",
              temperature: 0.6, topK: 20, topP: 0.95),
        .init(id: "instruct", french: "Instruct · sans réflexion", english: "Instruct · non-thinking",
              temperature: 0.7, topK: 20, topP: 0.8, presencePenalty: 1.5),
    ]
    // Qwen3.8 model cards (27B and Flash Next share the same guidance).
    private static let qwen38: [SamplingPreset] = [
        .init(id: "thinking", french: "Réflexion", english: "Thinking",
              temperature: 1.0, topK: 20, topP: 0.95),
        .init(id: "instruct", french: "Instruct · sans réflexion", english: "Instruct · non-thinking",
              temperature: 0.7, topK: 20, topP: 0.8, presencePenalty: 1.5),
    ]
    private static let lfm3: [SamplingPreset] = [
        .init(id: "general", french: "Général", english: "General",
              temperature: 0.1, topK: 50, repetitionPenalty: 1.1),
        .init(id: "math", french: "Maths et sciences", english: "Math & STEM",
              temperature: 1.0, topK: 20, topP: 0.95),
    ]
    private static let lfm3Heretic: [SamplingPreset] = [
        .init(id: "general", french: "Recommandé", english: "Recommended",
              temperature: 0.1, topK: 50, repetitionPenalty: 1.1),
    ]
    private static let lfm3VL: [SamplingPreset] = [
        .init(id: "general", french: "Recommandé", english: "Recommended",
              temperature: 0.1, topK: 50, repetitionPenalty: 1.1),
    ]
    private static let lfm25: [SamplingPreset] = [
        .init(id: "general", french: "Recommandé", english: "Recommended",
              temperature: 0.1, topK: 50, repetitionPenalty: 1.05),
    ]

    // Inkling 2 Mini generation_config.json: sampling at temperature 1 over the whole vocabulary.
    private static let inkling2: [SamplingPreset] = [
        .init(id: "general", french: "Recommandé", english: "Recommended",
              temperature: 1.0, topK: 0),
    ]

    /// Every catalog repository, including the alternative quantizations.
    static let byRepository: [String: [SamplingPreset]] = [
        "0xzknw/LFM3-1.7B-Heretic-NX-EXL3-8bpw": lfm3Heretic,
        "0xzknw/LFM3-1.7B-EXL3-8bpw": lfm3,
        "0xzknw/LFM3-1.7B-EXL3-4.49bpw": lfm3,
        "0xzknw/LFM3-1.7B-EXL3-4bpw": lfm3,
        "0xzknw/LFM3-VL-2B-EXL3-8bpw": lfm3VL,
        "0xzknw/LFM3-VL-2B-EXL3-4.49bpw": lfm3VL,
        "0xzknw/LFM3-VL-2B-EXL3-4bpw": lfm3VL,
        "0xzknw/LFM2.5-24B-A2B-EXL3-4bpw": lfm25,
        "UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw": qwen36,
        "turboderp/Qwen3.8-27B-exl3": qwen38,
        "turboderp/Qwen3.8-Flash-Next-exl3": qwen38,
        "0xzknw/Inkling-2-Mini-EXL3-3.33bpw": inkling2,
    ]

    /// The catalog repository a local model was downloaded from. The engine
    /// names downloads `<repo name>-<8-char id>` or `<repo name>--<variant>`
    /// (older multi-variant catalog repos), so after collapsing dash runs the
    /// repository name must equal, or be followed by `-` in, the local name
    /// or folder.
    static func repository(name: String, path: String) -> String? {
        let candidates = [name, URL(fileURLWithPath: path).lastPathComponent].map(normalized)
        // Longest match first so a name never resolves to a shorter sibling.
        return byRepository.keys.sorted { $0.count > $1.count }.first { repository in
            let stem = normalized(repository.split(separator: "/").last.map(String.init) ?? repository)
            return candidates.contains { $0 == stem || $0.hasPrefix(stem + "-") }
        }
    }

    private static func normalized(_ value: String) -> String {
        var result = ""
        for character in value.lowercased() where !(character == "-" && result.last == "-") {
            result.append(character)
        }
        return result
    }

    static func presets(name: String, path: String) -> [SamplingPreset] {
        repository(name: name, path: path).flatMap { byRepository[$0] } ?? []
    }

    static func presets(for model: LocalModel?) -> [SamplingPreset] {
        guard let model else { return [] }
        return presets(name: model.name, path: model.path)
    }
}
