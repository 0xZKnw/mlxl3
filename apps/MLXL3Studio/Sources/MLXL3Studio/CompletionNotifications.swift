import AppKit
import UserNotifications

/// Injected in lifecycle tests; constructing it never prompts or contacts macOS.
@MainActor
struct CompletionNotificationClient {
    var requestAuthorization: () async throws -> Bool
    var deliver: (_ requestID: String) async throws -> Void

    static let system = Self(
        requestAuthorization: {
            try await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound])
        },
        deliver: { requestID in
            let content = UNMutableNotificationContent()
            content.title = L("Réponse terminée", "Response complete")
            content.body = L("MLXL3 a terminé de générer sa réponse.", "MLXL3 has finished generating its response.")
            content.sound = .default
            // No prompt, response, model path or conversation title leaves the app.
            try await UNUserNotificationCenter.current().add(
                UNNotificationRequest(identifier: "generation-" + requestID, content: content, trigger: nil)
            )
        }
    )
}
