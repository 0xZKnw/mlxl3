# Code de l’application

- `MLXL3StudioApp.swift`, `AppDelegate.swift` et `StudioView.swift` : entrée SwiftUI et fenêtres principales.
- `StudioModel.swift`, `Domain.swift` et `ConversationStore.swift` : état, types et persistance des conversations.
- `MLXL3Bridge.swift` et `EngineRuntimeStore.swift` : dialogue avec le moteur et gestion de son runtime.
- `ModelLibrary.swift`, `ModelManager.swift` et les vues de téléchargement : modèles locaux et téléchargements.
- `ChatComponents.swift`, `MarkdownResponseView.swift` et `MessageActionsView.swift` : affichage et actions de conversation.
- `AgentStudio*.swift`, `Workspace*` et `MCP*` : projets, espaces de travail et connexions MCP.
- `AppSettingsView.swift`, `DesktopPreferences.swift`, `Theme.swift` et `Localization.swift` : préférences et présentation.

Les contrôles spécifiques sont dans [tests/](../../../../tests/README.md), les ressources dans [Resources/](../../Resources/README.md).
