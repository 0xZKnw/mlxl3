import Foundation

enum AppLanguage: String, CaseIterable {
    case fr, en
    var title: String { self == .fr ? "Français" : "English" }
}

enum AppLocalization {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var language = AppLanguage.fr

    static func set(_ value: AppLanguage) {
        lock.lock()
        defer { lock.unlock() }
        language = value
    }

    static func text(_ french: String, _ english: String) -> String {
        lock.lock()
        defer { lock.unlock() }
        return language == .fr ? french : english
    }
}

func L(_ french: String, _ english: String) -> String {
    AppLocalization.text(french, english)
}

// Only engine-authored progress labels go through this function, never model text.
func localizedProgress(_ label: String) -> String {
    let translations = [
        ("Préparation du contexte", "Preparing context"),
        ("Lecture des résultats MCP", "Reading MCP results"),
        ("Finalisation du contexte · lancement de la réponse", "Finalizing context · starting response"),
        ("Lecture du contexte", "Reading context"),
        ("Traitement du contexte", "Processing context"),
        ("nouveaux tokens", "new tokens"),
        ("en cache", "cached"),
        ("tokens traités", "tokens processed"),
    ]
    return translations.reduce(label) { text, pair in
        text.replacingOccurrences(of: pair.0, with: L(pair.0, pair.1))
    }
}
