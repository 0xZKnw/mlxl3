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
        studio.selectModel(studio.models[0].name)
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
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-tools") {
            let response = ChatMessage(role: .assistant, content: "", isStreaming: true)
            response.processing("Préparation du contexte")
            response.append("Je vais consulter la documentation officielle avec Exa.", phase: "thinking")
            response.startTool(id: "preview-exa", serverName: "exa", toolName: "web_search_exa")
            response.finishTool(id: "preview-exa", result: "Documentation MLX · cache de prompt et évaluation différée.", isError: false)
            response.processing("Lecture des résultats MCP · 3 737 nouveaux tokens · 950 en cache")
            let conversation = Conversation(title: "Recherche avec Exa", messages: [
                ChatMessage(role: .user, content: "Cherche la documentation du cache MLX avec Exa."), response
            ])
            studio.conversations = [conversation]
            studio.selectedConversationID = conversation.id
            // Dedicated visual QA fixture; no model, tool call or real history.
            if ProcessInfo.processInfo.arguments.contains("--ui-preview-tools-complete") {
                response.append("Les résultats sont disponibles. Je vais distinguer le cache de prompt et le cache d’allocations.", phase: "thinking")
                response.append("### Deux caches différents\n\nLe **cache de prompt** réutilise les calculs du contexte. Le cache d’allocations conserve des buffers disponibles.\n\n```python\nmx.clear_cache()\n```", phase: "answer")
                response.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
            }
        }
        return studio
    }
}
