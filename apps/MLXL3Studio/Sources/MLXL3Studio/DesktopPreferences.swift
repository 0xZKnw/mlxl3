import CoreFoundation
import Foundation
import SwiftUI

enum AppTheme: String, CaseIterable, Sendable {
    case system, light, dark, midnightBlue, graphite, forest, plum, paper, ocean, coffee, sand, lavender

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
        case .ocean: L("Océan", "Ocean")
        case .coffee: L("Café", "Coffee")
        case .sand: L("Sable", "Sand")
        case .lavender: L("Lavande", "Lavender")
        }
    }
    var colorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light, .paper, .sand, .lavender: .light
        case .dark, .midnightBlue, .graphite, .forest, .plum, .ocean, .coffee: .dark
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
    case neutral, blue, sage, amber, rose, violet, cyan, mint, coral, indigo, custom
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
        case .cyan: L("Cyan", "Cyan")
        case .mint: L("Menthe", "Mint")
        case .coral: L("Corail", "Coral")
        case .indigo: L("Indigo", "Indigo")
        case .custom: L("Personnalisée", "Custom")
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
        case .cyan: .cyan
        case .mint: .mint
        case .coral: Color(red: 0.91, green: 0.36, blue: 0.30)
        case .indigo: .indigo
        case .custom: .blue
        }
    }
}

enum CustomAccent {
    static let preferenceKey = "studio.customAccent"
    static let defaultHex = "4F9FEF"
    static func normalized(_ value: String?) -> String {
        guard let value, value.utf8.count == 6,
              value.utf8.allSatisfy({ (48...57).contains($0) || (65...70).contains($0) || (97...102).contains($0) })
        else { return defaultHex }
        return value.uppercased()
    }
    static func color(_ value: String) -> Color {
        let rgb = UInt32(normalized(value), radix: 16)!
        return Color(red: Double((rgb >> 16) & 255) / 255,
                     green: Double((rgb >> 8) & 255) / 255,
                     blue: Double(rgb & 255) / 255)
    }
    static func hex(_ color: Color) -> String? {
        guard let rgb = NSColor(color).usingColorSpace(.sRGB),
              [rgb.redComponent, rgb.greenComponent, rgb.blueComponent].allSatisfy(\.isFinite) else { return nil }
        let values = [rgb.redComponent, rgb.greenComponent, rgb.blueComponent].map {
            Int((min(1, max(0, $0)) * 255).rounded())
        }
        return String(format: "%02X%02X%02X", values[0], values[1], values[2])
    }
}

enum ConversationDensity: String, CaseIterable, Sendable {
    case tight, compact, comfortable, spacious, extraSpacious
    static let preferenceKey = "studio.conversationDensity"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .comfortable
    }
    var title: String {
        switch self {
        case .tight: L("Serré", "Tight")
        case .compact: L("Compact", "Compact")
        case .comfortable: L("Confort", "Comfortable")
        case .spacious: L("Aéré", "Spacious")
        case .extraSpacious: L("Très aéré", "Extra spacious")
        }
    }
    var spacing: CGFloat {
        switch self {
        case .tight: 12
        case .compact: 22
        case .comfortable: 38
        case .spacious: 54
        case .extraSpacious: 72
        }
    }
}

enum SidebarWidth: String, CaseIterable, Sendable {
    case compact, standard, wide
    static let preferenceKey = "studio.sidebarWidth"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .standard
    }
    var title: String {
        switch self {
        case .compact: L("Étroite", "Narrow")
        case .standard: L("Normale", "Standard")
        case .wide: L("Large", "Wide")
        }
    }
    var points: CGFloat {
        switch self {
        case .compact: 208
        case .standard: 232
        case .wide: 280
        }
    }
}

enum ConversationFont: String, CaseIterable, Sendable {
    case standard, rounded, serif, monospaced
    static let preferenceKey = "studio.conversationFont"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .standard
    }
    var title: String {
        switch self {
        case .standard: L("Standard", "Standard")
        case .rounded: L("Arrondie", "Rounded")
        case .serif: L("Avec empattements", "Serif")
        case .monospaced: L("Chasse fixe", "Monospaced")
        }
    }
    var design: Font.Design {
        switch self {
        case .standard: .default
        case .rounded: .rounded
        case .serif: .serif
        case .monospaced: .monospaced
        }
    }
}

enum MessageBubbleStyle: String, CaseIterable, Sendable {
    case subtle, outlined, tinted
    static let preferenceKey = "studio.messageBubbleStyle"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .subtle
    }
    var title: String {
        switch self {
        case .subtle: L("Discrète", "Subtle")
        case .outlined: L("Contour", "Outline")
        case .tinted: L("Colorée", "Tinted")
        }
    }
}

private struct ConversationFontKey: EnvironmentKey {
    static let defaultValue = ConversationFont.standard
}
private struct MessageBubbleStyleKey: EnvironmentKey {
    static let defaultValue = MessageBubbleStyle.subtle
}

private struct ConversationDensityKey: EnvironmentKey {
    static let defaultValue = ConversationDensity.comfortable
}
extension EnvironmentValues {
    var conversationFont: ConversationFont {
        get { self[ConversationFontKey.self] }
        set { self[ConversationFontKey.self] = newValue }
    }
    var messageBubbleStyle: MessageBubbleStyle {
        get { self[MessageBubbleStyleKey.self] }
        set { self[MessageBubbleStyleKey.self] = newValue }
    }
    var conversationDensity: ConversationDensity {
        get { self[ConversationDensityKey.self] }
        set { self[ConversationDensityKey.self] = newValue }
    }
}
