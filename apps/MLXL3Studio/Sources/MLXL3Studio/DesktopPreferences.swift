import Foundation
import CoreFoundation
import SwiftUI

enum AppTheme: String, CaseIterable {
    case system, light, dark

    static let preferenceKey = "studio.theme"
    static func load(from preferences: UserDefaults) -> Self {
        preferences.string(forKey: preferenceKey).flatMap(Self.init(rawValue:)) ?? .dark
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
              CFGetTypeID(value) == CFBooleanGetTypeID() else { return true }
        return value.boolValue
    }
}
