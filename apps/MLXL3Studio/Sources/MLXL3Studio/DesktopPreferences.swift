import CoreFoundation
import Foundation
import SwiftUI

enum AppTheme: String, CaseIterable, Sendable {
    case system, light, dark

    static let preferenceKey = "studio.theme"
    static func load(from preferences: UserDefaults) -> Self {
        let value = preferences.string(forKey: preferenceKey) ?? ""
        if ["paper", "sand", "lavender"].contains(value) { return .light }
        return Self(rawValue: value) ?? .dark
    }
    var title: String {
        switch self {
        case .system: L("Système", "System")
        case .light: L("Clair", "Light")
        case .dark: L("Sombre", "Dark")
        }
    }
    var colorScheme: ColorScheme? {
        switch self {
        case .system: nil
        case .light: .light
        case .dark: .dark
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
    case violet, rose, amber, neutral, custom
    static let preferenceKey = "studio.accent"
    static func load(from preferences: UserDefaults) -> Self {
        switch preferences.string(forKey: preferenceKey) {
        case "sage", "mint": .neutral
        case "coral": .rose
        case let value?: Self(rawValue: value) ?? .violet
        default: .violet
        }
    }

    var title: String {
        switch self {
        case .neutral: L("Neutre", "Neutral")
        case .amber: L("Ambre", "Amber")
        case .rose: L("Rose", "Rose")
        case .violet: L("Iris", "Iris")
        case .custom: L("Personnalisée", "Custom")
        }
    }

    var color: Color {
        func adaptive(_ light: UInt32, _ dark: UInt32) -> Color {
            func swatch(_ value: UInt32) -> NSColor {
                NSColor(srgbRed: CGFloat(value >> 16) / 255, green: CGFloat((value >> 8) & 255) / 255,
                        blue: CGFloat(value & 255) / 255, alpha: 1)
            }
            return Color(nsColor: StudioTheme.adaptiveColor(light: swatch(light), dark: swatch(dark)))
        }
        switch self {
        case .neutral: return StudioTheme.accent
        case .violet: return adaptive(0x6642A4, 0xBBA4F3)
        case .rose: return adaptive(0x97345D, 0xE89BB9)
        case .amber: return adaptive(0x8E4B18, 0xF0B782)
        case .custom: return CustomAccent.color(CustomAccent.defaultHex)
        }
    }
}

enum CustomAccent {
    static let preferenceKey = "studio.customAccent"
    static let defaultHex = "BBA4F3"
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
