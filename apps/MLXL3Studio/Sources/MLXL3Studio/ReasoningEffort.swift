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

  var detail: String {
    switch self {
    case .none: L("Répond directement", "Answers directly")
    case .low: L("Réflexion brève", "Brief thinking")
    case .medium: L("Équilibré", "Balanced")
    case .high: L("Réflexion approfondie", "Thorough thinking")
    case .xhigh: L("Vérifie ses hypothèses", "Checks its assumptions")
    case .max: L("Le plus long et le plus précis", "Longest and most careful")
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

/// Reasoning levels a model's chat template accepts, read from the template itself so a new
/// model needs no list: a level is offered when the template names it, `none` when it can
/// turn thinking off.
struct ReasoningSupport: Equatable, Sendable {
  let levels: [ReasoningEffort]
  let defaultLevel: ReasoningEffort

  static func detect(template: String) -> ReasoningSupport? {
    guard template.contains("reasoning_effort") else { return nil }
    func named(_ value: String) -> Bool {
      template.contains("'\(value)'") || template.contains("\"\(value)\"")
    }
    var levels = ReasoningEffort.allCases.filter { $0 != .none && named($0.rawValue) }
    guard !levels.isEmpty else { return nil }
    if named("none") || template.contains("enable_thinking") { levels.insert(.none, at: 0) }
    guard levels.count >= 2 else { return nil }
    return ReasoningSupport(levels: levels, defaultLevel: defaultLevel(in: template, levels: levels))
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
    if let data = try? Data(contentsOf: jinja), data.count <= 2_000_000 {
      return detect(template: String(decoding: data, as: UTF8.self))
    }
    let config = folder.appendingPathComponent("tokenizer_config.json")
    guard let data = try? Data(contentsOf: config), data.count <= 8_000_000,
      let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { return nil }
    if let template = object["chat_template"] as? String { return detect(template: template) }
    if let templates = object["chat_template"] as? [[String: Any]] {
      let text = templates.compactMap { $0["template"] as? String }.joined(separator: "\n")
      return detect(template: text)
    }
    return nil
  }
}
