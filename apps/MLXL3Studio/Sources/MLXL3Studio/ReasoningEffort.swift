import Foundation
import SwiftUI

/// Reasoning levels the engine passes to the chat template as `reasoning_effort`
/// Each model template defines the meaning of its supported levels.
enum ReasoningEffort: String, CaseIterable, Codable, Identifiable, Sendable {
  case none, minimal, low, medium, high, xhigh, max

  /// The previous engine protocol had no advertised list and no minimal effort.
  static let legacyLevels: Set<Self> = [.none, .low, .medium, .high, .xhigh, .max]

  var id: String { rawValue }

  var title: String {
    switch self {
    case .none: L("Sans réflexion", "No thinking")
    case .minimal: L("Minimal", "Minimal")
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
    case .minimal: L("Réflexion minimale", "Minimal thinking")
    case .low: L("Réflexion faible", "Low thinking")
    case .medium: L("Réflexion moyenne", "Medium thinking")
    case .high: L("Réflexion élevée", "High thinking")
    case .xhigh: L("Réflexion très élevée", "Extra high thinking")
    case .max: L("Réflexion max", "Max thinking")
    }
  }

  /// Gauge needle that rises with the level.
  var symbol: String {
    switch self {
    case .none: "gauge.with.dots.needle.0percent"
    case .minimal, .low: "gauge.with.dots.needle.33percent"
    case .medium: "gauge.with.dots.needle.50percent"
    case .high: "gauge.with.dots.needle.67percent"
    case .xhigh, .max: "gauge.with.dots.needle.100percent"
    }
  }
}

/// The model template supplies the native controls. The engine may understand fewer names.
struct ReasoningSupport: Equatable, Sendable {
  let levels: [ReasoningEffort]
  let defaultLevel: ReasoningEffort

  var toggleOnly = false
  var noneIsZeroEffort = false
  // Some templates assign two names to the same native effort.
  var nativeAliases: [ReasoningEffort: ReasoningEffort] = [:]

  func choices(acceptedByEngine accepted: Set<ReasoningEffort>) -> [ReasoningEffort] {
    levels.filter { level in
      guard accepted.contains(level) else { return false }
      return nativeAliases[level].map { !accepted.contains($0) } ?? true
    }
  }

  func title(for level: ReasoningEffort, inComposer: Bool = false) -> String {
    if noneIsZeroEffort && level == .none { return L("Effort nul", "Zero effort") }
    if toggleOnly && level == .max { return L("Réflexion activée", "Thinking on") }
    return inComposer ? level.label : level.title
  }

  static func detect(template: String) -> ReasoningSupport? {
    guard template.contains("reasoning_effort") else {
      // The Qwen switch emits an already closed thinking block when disabled.
      guard template.contains("<think>"), template.contains("</think>"),
            template.range(of: #"enable_thinking\s+is\s+defined\s+and\s+enable_thinking\s+is\s+false"#,
                           options: .regularExpression) != nil else { return nil }
      return ReasoningSupport(levels: [.none, .max], defaultLevel: .max, toggleOnly: true)
    }
    // Detect names in the effort control, not unrelated strings in the template
    // (for example a tool schema containing "high" as a priority).
    let accepted = template.firstMatch(of: /\b\w*reasoning_effort\s+not\s+in\s*[\[(]([^\])]+)[\])]/)?.1
    let effortMap = template.firstMatch(of: /\bset\s+\w*effort\w*\s*=\s*\{([^}]+)\}/)?.1
    var numericLevels: [Double: [ReasoningEffort]] = [:]
    for match in (effortMap ?? "").matches(of: /['"](\w+)['"]\s*:\s*([0-9]+(?:\.[0-9]+)?)/) {
      if let level = ReasoningEffort(rawValue: String(match.1)), let value = Double(match.2) {
        numericLevels[value, default: []].append(level)
      }
    }
    let compared = template.matches(of: /\b\w*reasoning_effort\s*==\s*['"](\w+)['"]/)
      .compactMap { ReasoningEffort(rawValue: String($0.1)) }
    let fallbackName = template.firstMatch(of: /reasoning_effort\s*\|\s*default\(\s*['"](\w+)['"]/)
      .flatMap { ReasoningEffort(rawValue: String($0.1)) }
    let declaresSwitch = template.contains("enable_thinking")
    let levels = ReasoningEffort.allCases.filter { level in
      if level == .none && declaresSwitch { return true }
      if let accepted {
        return accepted.contains("'\(level.rawValue)'") || accepted.contains("\"\(level.rawValue)\"")
      }
      return numericLevels.values.contains(where: { $0.contains(level) })
        || compared.contains(level) || fallbackName == level
    }
    guard levels.count >= 2 else { return nil }
    let fallback = defaultLevel(named: fallbackName, in: template, levels: levels, numericLevels: numericLevels)
    var aliases: [ReasoningEffort: ReasoningEffort] = [:]
    for group in numericLevels.values where group.count > 1 {
      // Prefer the default name, then the highest supported name.
      guard let representative = group.contains(fallback) ? fallback : levels.last(where: group.contains) else { continue }
      for alias in group where alias != representative { aliases[alias] = representative }
    }
    return ReasoningSupport(levels: levels, defaultLevel: fallback,
                            noneIsZeroEffort: numericLevels[0]?.contains(.none) == true, nativeAliases: aliases)
  }

  /// `reasoning_effort|default('xhigh')` (Qwen3.8) or a numeric fallback mapped through the
  /// template's own level table (Inkling 2: `else 0.9` with `"high": 0.9`).
  private static func defaultLevel(named: ReasoningEffort?, in template: String, levels: [ReasoningEffort],
                                   numericLevels: [Double: [ReasoningEffort]]) -> ReasoningEffort {
    if let named, levels.contains(named) { return named }
    if let match = template.firstMatch(of: /reasoning_effort[^\n]*?else\s+([0-9]+(?:\.[0-9]+)?)/),
       let number = Double(match.1), let group = numericLevels[number],
       let level = levels.last(where: group.contains) { return level }
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
