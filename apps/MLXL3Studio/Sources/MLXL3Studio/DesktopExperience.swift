import Foundation
import CoreFoundation
import SwiftUI

enum ConversationReadingWidth: String, CaseIterable, Sendable {
    case centered, wide
    static let preferenceKey = "studio.readingWidth"
    var title: String { self == .centered ? L("Centrée", "Centered") : L("Large", "Wide") }
    var messageMaximum: CGFloat { self == .centered ? 820 : .infinity }
    var composerMaximum: CGFloat { self == .centered ? 790 : .infinity }
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .centered
    }
}

private struct ConversationReadingWidthKey: EnvironmentKey {
    static let defaultValue = ConversationReadingWidth.centered
}
extension EnvironmentValues {
    var conversationReadingWidth: ConversationReadingWidth {
        get { self[ConversationReadingWidthKey.self] }
        set { self[ConversationReadingWidthKey.self] = newValue }
    }
    var conversationTextScale: CGFloat {
        get { self[ConversationTextScaleKey.self] }
        set { self[ConversationTextScaleKey.self] = newValue }
    }
}

private struct ConversationTextScaleKey: EnvironmentKey {
    static let defaultValue: CGFloat = 1
}

enum ResponseTextSize {
    static let preferenceKey = "studio.responseTextSize"
    static let defaultValue: Double = 15
    static let range: ClosedRange<Double> = 12...24

    static func normalized(_ value: Double) -> Double {
        guard value.isFinite else { return defaultValue }
        return min(range.upperBound, max(range.lowerBound, value.rounded()))
    }

    static func load(from preferences: UserDefaults) -> Double {
        guard let number = preferences.object(forKey: preferenceKey) as? NSNumber,
              CFGetTypeID(number) != CFBooleanGetTypeID() else { return defaultValue }
        return normalized(number.doubleValue)
    }
}

/// ON/OFF is model intent; calibration remains scoped to head revision and runtime.
enum ModelMTPPreference {
    static let key = "studio.mtpEnabledByModel"
    static func load(modelPath: String, preferences: UserDefaults) -> Bool? {
        guard let value = preferences.dictionary(forKey: key)?[modelPath] as? NSNumber,
              CFGetTypeID(value) == CFBooleanGetTypeID() else { return nil }
        return value.boolValue
    }
    static func save(_ enabled: Bool, modelPath: String, preferences: UserDefaults) {
        var entries = preferences.dictionary(forKey: key) ?? [:]
        entries[modelPath] = enabled
        preferences.set(entries, forKey: key)
    }
}
