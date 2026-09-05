import Foundation

extension StudioModel {
    /// Visual QA only: no model is loaded, no personal history is read, and
    /// any fixture edits are persisted to an isolated temporary workspace.
    static func uiPreview() -> StudioModel {
        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("mlxl3-ui-preview-\(UUID().uuidString)")
            .appendingPathComponent("conversations.json")
        let preferences = UserDefaults(suiteName: "io.mlxl3.ui-preview")!
        let studio = StudioModel(conversationFileURL: file, isPreview: true, preferences: preferences)
        studio.models = [LocalModel(
            name: "Qwen3.6 · 35B A3B", path: "/nonexistent/ui-preview",
            modelType: "qwen3_5", format: "EXL3", bits: 2.49,
            sizeBytes: 12_000_000_000, modules: 0, addedAt: "", size: "12 GB"
        )]
        studio.selectedModelName = studio.models.first?.name
        let sample = Conversation(title: "Un outil plus simple, plus agréable", messages: [
            ChatMessage(role: .user, content: "Comment rendre une interface plus agréable au quotidien ?"),
            ChatMessage(role: .assistant, content: """
            Une bonne interface vous laisse vous concentrer sur **ce que vous voulez faire**.

            ### Commencer par l’essentiel
            - Une hiérarchie claire, avec une action principale identifiable.
            - Une typographie lisible et assez d’espace pour respirer.
            - Des détails discrets : un état de focus, un survol, un retour immédiat.

            Le plus important : garder les interactions prévisibles, même lorsque le contenu change.

            ```swift
            let priority = "La clarté avant la décoration"
            print(priority)
            ```
            """, thinking: "Je vais distinguer la lisibilité, la navigation et les retours d’interaction.")
        ])
        studio.conversations = [Conversation(), sample, Conversation(title: "Notes de lecture"), Conversation(title: "Idées pour un projet personnel")]
        studio.selectedConversationID = studio.conversations.first?.id
        return studio
    }
}
