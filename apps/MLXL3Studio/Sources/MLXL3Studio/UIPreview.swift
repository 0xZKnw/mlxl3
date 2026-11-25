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
        // Sidebar QA: a long history spread over every date group, with a pin and empty drafts.
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-history") {
            let now = Date()
            func past(_ title: String, days: Double, hours: Double = 0, pinned: Bool = false, empty: Bool = false) -> Conversation {
                let date = now.addingTimeInterval(-(days * 86_400 + hours * 3_600))
                var conversation = Conversation(
                    title: title, messages: empty ? [] : [ChatMessage(role: .user, content: title)],
                    createdAt: date, isPinned: pinned)
                conversation.updatedAt = empty ? nil : date
                return conversation
            }
            studio.conversations = [
                past("Plan de lancement du site", days: 9, pinned: true),
                past("Fais-moi un fichier Python qui trie mes photos par date", days: 0, hours: 0.2),
                past("Nouvelle conversation", days: 0, hours: 1, empty: true),
                past("Réécrire l’introduction du mémoire", days: 0, hours: 3),
                past("Salut, tu sais faire quoi ?", days: 1, hours: 2),
                past("Crée-moi un txt avec une blague", days: 1, hours: 5),
                past("Comparer deux offres de forfait mobile", days: 3),
                past("Fais un petit site web où tu présentes tes outils", days: 4),
                past("Tu as quoi comme tools ?", days: 6),
                past("Recette rapide avec ce qu’il reste au frigo", days: 12),
                past("Résumé du chapitre 4", days: 19),
                past("Script de sauvegarde pour le NAS", days: 41),
                past("Idées de cadeaux pour l’anniversaire de Léa", days: 75),
                past("Notes de lecture : Le Mythe de Sisyphe", days: 400),
            ]
            studio.selectedConversationID = studio.conversations[1].id
        }
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
        // Agent bar QA: a finished task, or an empty one with `--ui-preview-agent-new`. Its
        // workspace lives beside the temporary conversation file.
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-agent") {
            var task = Conversation(title: "Corriger les tests", messages: [
                ChatMessage(role: .user, content: "Le script test.sh échoue. Corrige le projet."),
                ChatMessage(role: .assistant, content: "J’ai corrigé `mean` et `median` dans `src/stats.pl`. `sh test.sh` affiche maintenant **PASS**."),
            ])
            if ProcessInfo.processInfo.arguments.contains("--ui-preview-agent-new") { task = Conversation() }
            task.agent = true
            studio.conversations = [task] + studio.conversations
            studio.selectMode(.agent)
        }
        // Agent run QA: reasoning, sandbox tools (one failing command) and a command still
        // running; `--ui-preview-agent-run-complete` lets it finish with an answer.
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-agent-run") {
            let run = ChatMessage(role: .assistant, content: "", isStreaming: true)
            run.processing("Préparation du contexte")
            run.append("Je commence par lister les fichiers du projet pour trouver le test.", phase: "thinking")
            run.startTool(id: "a1", serverName: "sandbox", toolName: "javascript",
                          arguments: "// Inventaire du projet\nconst files = list();\nreturn files.join(\"\\n\");")
            run.finishTool(id: "a1", result: "README.md\nsrc/stats.pl\ntest.sh", isError: false)
            run.append("Le test appelle `mean` et `median`. Je lis l’implémentation.", phase: "thinking")
            run.startTool(id: "a2", serverName: "sandbox", toolName: "read_file", arguments: #"{"path":"src/stats.pl"}"#)
            run.finishTool(id: "a2", result: "sub mean { my $s = 0; $s += $_ for @_; return $s / @_; }\nsub median { return $_[int(@_ / 2)]; }", isError: false)
            run.startTool(id: "a3", serverName: "sandbox", toolName: "run", arguments: #"{"command":"sh test.sh"}"#)
            run.finishTool(id: "a3", result: "FAIL median(4, 1, 3, 2): expected 2.5, got 3\nExit status: 1", isError: true)
            run.append("`median` ne trie pas les valeurs et ignore les tailles paires. Je corrige les deux.", phase: "thinking")
            run.startTool(id: "a4", serverName: "sandbox", toolName: "write_file",
                          arguments: #"{"content":"sub median {\n  my @v = sort { $a <=> $b } @_;\n  my $m = int(@v / 2);\n  return @v % 2 ? $v[$m] : ($v[$m - 1] + $v[$m]) / 2;\n}","path":"src/stats.pl"}"#)
            run.finishTool(id: "a4", result: "Wrote src/stats.pl", isError: false)
            run.startTool(id: "a5", serverName: "sandbox", toolName: "run", arguments: #"{"command":"sh test.sh"}"#)
            if ProcessInfo.processInfo.arguments.contains("--ui-preview-agent-run-complete") {
                run.finishTool(id: "a5", result: "PASS 6 tests\nExit status: 0", isError: false)
                run.append("J’ai corrigé `median` dans `src/stats.pl` : les valeurs sont triées et une liste de taille paire renvoie la moyenne des deux valeurs centrales. `sh test.sh` affiche maintenant **PASS**.", phase: "answer")
                run.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
            } else if ProcessInfo.processInfo.arguments.contains("--ui-preview-agent-run-thinking") {
                // Reasoning still streaming: shown live, under its sweeping title.
                run.finishTool(id: "a5", result: "PASS 6 tests\nExit status: 0", isError: false)
                run.append("Les six tests passent. Je vérifie encore le cas d’une liste vide : `median()` divise par zéro "
                           + "si `@v` est vide, et `mean` aussi. Le test ne le couvre pas, mais une valeur `undef` serait "
                           + "plus sûre qu’une erreur. Je regarde si d’autres fonctions de `src/stats.pl` appellent", phase: "thinking")
            }
            var task = Conversation(title: "Corriger les tests", messages: [
                ChatMessage(role: .user, content: "Le script test.sh échoue. Corrige le projet."), run,
            ])
            task.agent = true
            studio.conversations = [task]
            studio.selectMode(.agent)
            studio.selectedConversationID = task.id
        }
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-stress") {
            let response = ChatMessage(role: .assistant, content: "", isStreaming: true)
            let conversation = Conversation(title: "QA · long streaming", messages: [ChatMessage(role: .user, content: "Generate a large HTML file"), response])
            studio.conversations = [conversation]
            studio.selectedConversationID = conversation.id
            Task { @MainActor in
                response.append("Stress test: streaming a large file.\n", phase: "thinking")
                response.append("```html\n", phase: "answer")
                for _ in 0..<600 {
                    response.append(String(repeating: "<div>Hello 👋</div>\n", count: 30), phase: "answer")
                    try? await Task.sleep(for: .milliseconds(30))
                }
                response.append("```\n\nFinished.", phase: "answer")
                response.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
            }
        }
        // Page QA: `--ui-preview-settings` opens Appearance, `--ui-preview-models` the
        // model page (no refresh: the fixture model stays the only one).
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-settings") {
            studio.appSettingsPage = .appearance
            studio.showAppSettings = true
        }
        // `--ui-preview-settings-page=models` (any SettingsPage), `--ui-preview-mcp`, `--ui-preview-projects`.
        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix("--ui-preview-settings-page=") }),
           let page = SettingsPage(rawValue: String(argument.dropFirst("--ui-preview-settings-page=".count))) {
            studio.appSettingsPage = page
            studio.showAppSettings = true
        }
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-mcp") { studio.showMCPManager = true }
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-projects") { studio.showWorkspace = true }
        // Lab QA: five weeks of measurements over three models, deterministic (no clock-free
        // randomness: the same launch day gives the same page).
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-lab") {
            let models = [("Qwen3.6 · 35B A3B", 2.49, 61.0), ("LFM3 · 8B", 4.0, 118.0), ("Inkling 2 Mini", 3.5, 87.0)]
            var samples: [LabSample] = []
            for day in 0..<36 where day % 7 != 5 {
                for turn in 0..<(1 + (day * 7) % 4) {
                    let (name, bits, speed) = models[(day + turn) % models.count]
                    let input = 900 + ((day * 37 + turn * 911) % 5200)
                    let output = 180 + ((day * 53 + turn * 269) % 1400)
                    var sample = LabSample(
                        model: name, modelPath: "/nonexistent/ui-preview", bits: bits, mode: turn % 3 == 2 ? "agent" : "chat",
                        runtime: "preview", system: "macOS", hardware: LabHardware.identity, context: 16_384,
                        temperature: 0.7, topK: 40, mtp: turn % 2 == 0, dflash: false,
                        stats: GenerationStats(
                            ttftSeconds: 0.31 + Double(turn) * 0.07, prefillTps: 940, decodeTps: speed + Double((day * 13) % 17) - 8,
                            promptTokens: input, generatedTokens: output, peakMemoryGB: nil, modelSizeGB: nil,
                            cachedPromptTokens: input / 2, evaluatedPromptTokens: input - input / 2, contextUsed: input + output,
                            elapsedSeconds: Double(output) / speed + 0.6, endToEndTTFTSeconds: 0.42 + Double(turn) * 0.09))
                    sample.date = Date().addingTimeInterval(-Double(day) * 86_400 - Double(turn) * 1_500)
                    samples.append(sample)
                }
            }
            studio.seedLabPreview(samples)
        }
        // Agent Studio QA: a small project with two versions in a temporary workspace.
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-agent-studio") {
            let workspace = FileManager.default.temporaryDirectory
                .appendingPathComponent("mlxl3-ui-preview-studio-\(UUID().uuidString)/files")
            try? FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
            let files = [
                "README.md": "# Statistiques\n\nMoyenne et médiane en Perl, avec un test.\n",
                "src/stats.pl": "sub mean { my $s = 0; $s += $_ for @_; return $s / @_; }\nsub median { return $_[int(@_ / 2)]; }\n1;\n",
                "test.sh": "#!/bin/sh\nperl -Isrc t/stats.t\n",
                "t/stats.t": "require 'stats.pl';\nprint median(4, 1, 3, 2) == 2.5 ? \"PASS\\n\" : \"FAIL\\n\";\n",
            ]
            for (path, text) in files { try? AgentProjectStore.write(Data(text.utf8), path: path, root: workspace) }
            let store = AgentProjectStore(workspace: workspace)
            _ = try? store.checkpoint("Projet importé")
            try? AgentProjectStore.write(Data("""
                sub mean { my $s = 0; $s += $_ for @_; return $s / @_; }
                sub median {
                  my @v = sort { $a <=> $b } @_;
                  my $m = int(@v / 2);
                  return @v % 2 ? $v[$m] : ($v[$m - 1] + $v[$m]) / 2;
                }
                1;

                """.utf8), path: "src/stats.pl", root: workspace)
            _ = try? store.checkpoint("Corriger la médiane")
            studio.studioDocument = StudioDocument(workspace: workspace, file: workspace.appendingPathComponent("src/stats.pl"))
        }
        // `--ui-preview-mode=server` (or decision, lab): standalone screens.
        if let argument = ProcessInfo.processInfo.arguments.first(where: { $0.hasPrefix("--ui-preview-mode=") }),
           let mode = StudioMode(rawValue: String(argument.dropFirst("--ui-preview-mode=".count))) {
            studio.selectMode(mode)
        }
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-inspector") {
            studio.showInspector = true
        }
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-models") {
            studio.showModelManager = true
        }
        if ProcessInfo.processInfo.arguments.contains("--ui-preview-images") && Image21Hardware.available {
            studio.showImages = true
        }
        return studio
    }
}
