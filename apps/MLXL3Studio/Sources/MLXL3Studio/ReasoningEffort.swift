import Foundation
import SwiftUI

/// Reasoning levels the engine passes to the chat template as `reasoning_effort`
/// (`none` turns thinking off). Each template supports its own subset.
enum ReasoningEffort: String, CaseIterable, Codable, Identifiable, Sendable {
  case none, low, medium, high, xhigh, max

  var id: String { rawValue }

  var title: String {
    switch self {
    case .none: L("Sans réflexion", "No thinking")
    case .low: L("Faible", "Low")
    case .medium: L("Moyen", "Medium")
    case .high: L("Élevé", "High")
    case .xhigh: L("Très élevé", "Extra high")
    case .max: "Max"
    }
  }

  /// Composer label: the level reads as a kind of thinking.
  var label: String {
    switch self {
    case .none: L("Sans réflexion", "No thinking")
    case .low: L("Réflexion faible", "Low thinking")
    case .medium: L("Réflexion moyenne", "Medium thinking")
    case .high: L("Réflexion élevée", "High thinking")
    case .xhigh: L("Réflexion très élevée", "Extra high thinking")
    case .max: L("Réflexion max", "Max thinking")
    }
  }

  /// Maximum generated thinking tokens per step when the model supports a bounded chat.
  var budget: Int? {
    switch self {
    case .none, .max: nil
    case .low: 256
    case .medium: 1_024
    case .high: 4_096
    case .xhigh: 8_192
    }
  }

  /// Gauge needle that rises with the level.
  var symbol: String {
    switch self {
    case .none: "gauge.with.dots.needle.0percent"
    case .low: "gauge.with.dots.needle.33percent"
    case .medium: "gauge.with.dots.needle.50percent"
    case .high: "gauge.with.dots.needle.67percent"
    case .xhigh, .max: "gauge.with.dots.needle.100percent"
    }
  }
}

/// The model template controls the native levels. A compatible chat engine can
/// additionally distinguish levels with a thinking-token budget.
struct ReasoningSupport: Equatable, Sendable {
  let levels: [ReasoningEffort]
  let defaultLevel: ReasoningEffort

  var toggleOnly = false
  // Some templates assign two names to one native effort. An explicit chat budget
  // can distinguish them, while the native-only picker needs one representative.
  var nativeAliases: [ReasoningEffort: ReasoningEffort] = [:]

  func choices(bounded: Bool) -> [ReasoningEffort] {
    if toggleOnly && bounded { return [.none, .low, .medium, .high, .max] }
    return bounded ? levels : levels.filter { nativeAliases[$0] == nil }
  }

  func title(for level: ReasoningEffort) -> String {
    if toggleOnly && level == .max { return L("Réflexion activée", "Thinking on") }
    return level.title
  }

  static func detect(template: String) -> ReasoningSupport? {
    guard template.contains("reasoning_effort") else {
      // The Qwen switch emits an already closed thinking block when disabled.
      guard template.contains("<think>"), template.contains("</think>"),
            template.range(of: #"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+false"#,
                           options: .regularExpression) != nil else { return nil }
      return ReasoningSupport(levels: [.none, .max], defaultLevel: .max, toggleOnly: true)
    }
    func named(_ value: String) -> Bool {
      template.contains("'\(value)'") || template.contains("\"\(value)\"")
    }
    // A template may normalize an alias before checking its list of supported levels.
    let accepted = template.firstMatch(of: /\b\w*reasoning_effort\s+not\s+in\s*[\[(]([^\])]+)[\])]/)?.1
    var levels = ReasoningEffort.allCases.filter { level in
      guard level != .none else { return false }
      guard let accepted else { return named(level.rawValue) }
      return accepted.contains("'\(level.rawValue)'") || accepted.contains("\"\(level.rawValue)\"")
    }
    guard !levels.isEmpty else { return nil }
    if named("none") || template.contains("enable_thinking") { levels.insert(.none, at: 0) }
    guard levels.count >= 2 else { return nil }
    let fallback = defaultLevel(in: template, levels: levels)
    var aliases: [ReasoningEffort: ReasoningEffort] = [:]
    var numericLevels: [Double: [ReasoningEffort]] = [:]
    for match in template.matches(of: /['"](low|medium|high|xhigh|max)['"]\s*:\s*([0-9]+(?:\.[0-9]+)?)/) {
      if let level = ReasoningEffort(rawValue: String(match.1)), let value = Double(match.2) {
        numericLevels[value, default: []].append(level)
      }
    }
    for group in numericLevels.values where group.count > 1 {
      // Keep the default, or the highest supported name, for the unbounded selector.
      guard let representative = group.contains(fallback) ? fallback : levels.last(where: group.contains) else { continue }
      for alias in group where alias != representative { aliases[alias] = representative }
    }
    return ReasoningSupport(levels: levels, defaultLevel: fallback, nativeAliases: aliases)
  }

  /// `reasoning_effort|default('xhigh')` (Qwen3.8) or a numeric fallback mapped through the
  /// template's own level table (Inkling 2: `else 0.9` with `"high": 0.9`).
  private static func defaultLevel(in template: String, levels: [ReasoningEffort])
    -> ReasoningEffort
  {
    if let match = template.firstMatch(of: /reasoning_effort\s*\|\s*default\(\s*['"](\w+)['"]/),
      let level = ReasoningEffort(rawValue: String(match.1)), levels.contains(level)
    {
      return level
    }
    if let number = template.firstMatch(of: /reasoning_effort[^\n]*?else\s+([0-9]+(?:\.[0-9]+)?)/)?.1 {
      for level in levels where level != .none {
        let pattern = "[\"']\(level.rawValue)[\"']\\s*:\\s*\(NSRegularExpression.escapedPattern(for: String(number)))\\b"
        if template.range(of: pattern, options: .regularExpression) != nil { return level }
      }
    }
    return levels.last { $0 != .none } ?? levels[0]
  }

  /// The chat template of a model folder: `chat_template.jinja`, else `tokenizer_config.json`.
  static func detect(modelPath: String) -> ReasoningSupport? {
    let folder = URL(fileURLWithPath: modelPath, isDirectory: true)
    let jinja = folder.appendingPathComponent("chat_template.jinja")
    if FileManager.default.fileExists(atPath: jinja.path) {
      guard let data = try? Data(contentsOf: jinja, options: .mappedIfSafe), data.count <= 2_000_000 else { return nil }
      return detect(template: String(decoding: data, as: UTF8.self))
    }
    let config = folder.appendingPathComponent("tokenizer_config.json")
    guard let data = try? Data(contentsOf: config), data.count <= 8_000_000,
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    if let template = object["chat_template"] as? String { return detect(template: template) }
    return nil
  }
}
