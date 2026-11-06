import CoreFoundation
import Foundation
import SwiftUI

enum AppTheme: String, CaseIterable, Sendable {
    case system, light, dark, midnightBlue, graphite, forest, plum, paper

    static let preferenceKey = "studio.theme"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .dark
    }
    var title: String {
        switch self {
        case .system: L("Système", "System")
        case .light: L("Clair", "Light")
        case .dark: L("Sombre", "Dark")
        case .midnightBlue: L("Bleu nuit", "Midnight blue")
        case .graphite: L("Ardoise", "Slate")
        case .forest: L("Forêt", "Forest")
        case .plum: L("Prune", "Plum")
        case .paper: L("Papier", "Paper")
        }
    }
    var colorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light, .paper: .light
        case .dark, .midnightBlue, .graphite, .forest, .plum: .dark
        }
    }
}

enum StartupModelPreference {
    static let key = "studio.loadModelOnStartup"
    // Preserve existing behavior for users who have never changed this setting.
    static func load(from preferences: UserDefaults) -> Bool {
        guard let value = preferences.object(forKey: key) as? NSNumber,
            CFGetTypeID(value) == CFBooleanGetTypeID()
        else { return true }
        return value.boolValue
    }
}

enum AppAccent: String, CaseIterable, Sendable {
    case neutral, blue, sage, amber, rose, violet
    static let preferenceKey = "studio.accent"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .neutral
    }
    var title: String {
        switch self {
        case .neutral: L("Neutre", "Neutral")
        case .blue: L("Bleu", "Blue")
        case .sage: L("Sauge", "Sage")
        case .amber: L("Ambre", "Amber")
        case .rose: L("Rose", "Rose")
        case .violet: L("Violet", "Violet")
        }
    }
    var color: Color {
        switch self {
        case .neutral: StudioTheme.accent
        case .blue: .blue
        case .sage: .green
        case .amber: .orange
        case .rose: .pink
        case .violet: .purple
        }
    }
}

enum ConversationDensity: String, CaseIterable, Sendable {
    case compact, comfortable, spacious
    static let preferenceKey = "studio.conversationDensity"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .comfortable
    }
    var title: String {
        switch self {
        case .compact: L("Compact", "Compact")
        case .comfortable: L("Confort", "Comfortable")
        case .spacious: L("Aéré", "Spacious")
        }
    }
    var spacing: CGFloat {
        switch self {
        case .compact: 22
        case .comfortable: 38
        case .spacious: 54
        }
    }
}

private struct ConversationDensityKey: EnvironmentKey {
    static let defaultValue = ConversationDensity.comfortable
}
extension EnvironmentValues {
    var conversationDensity: ConversationDensity {
        get { self[ConversationDensityKey.self] }
        set { self[ConversationDensityKey.self] = newValue }
    }
}
