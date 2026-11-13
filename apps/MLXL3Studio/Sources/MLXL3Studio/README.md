# Application code

- `MLXL3StudioApp.swift`, `AppDelegate.swift`, and `StudioView.swift`: SwiftUI entry point and main windows.
- `StudioModel.swift`, `Domain.swift`, and `ConversationStore.swift`: state, types, and conversation persistence.
- `MLXL3Bridge.swift` and `EngineRuntimeStore.swift`: engine communication and runtime management.
- `ModelLibrary.swift`, `ModelManager.swift`, and download views: local models and downloads.
- `ChatComponents.swift`, `MarkdownResponseView.swift`, and `MessageActionsView.swift`: conversation display and actions.
- `AgentStudio*.swift`, `Workspace*`, and `MCP*`: projects, workspaces, and MCP connections.
- `AppSettingsView.swift`, `DesktopPreferences.swift`, `Theme.swift`, and `Localization.swift`: preferences and presentation.

Related checks are in [tests/](../../../../tests/README.md), and resources are in [Resources/](../../Resources/README.md).
