import SwiftUI

@main
struct MLXL3StudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var studio: StudioModel
    private let isPreview = ProcessInfo.processInfo.arguments.contains("--ui-preview")

    init() {
        if ProcessInfo.processInfo.arguments.contains("--check-mcp-preferences") {
            do {
                try MCPPreferenceCheck.run()
                print("MCP checks passed: default off, persisted on/off, request flag, isolated preview")
                exit(0)
            } catch {
                print("MCP checks failed: \(error)")
                exit(1)
            }
        }
        let preview = ProcessInfo.processInfo.arguments.contains("--ui-preview")
        _studio = StateObject(wrappedValue: preview ? StudioModel.uiPreview() : StudioModel())
    }

    var body: some Scene {
        WindowGroup(isPreview ? "MLXL3 — Aperçu" : "MLXL3 Desktop") {
            StudioView()
                .environmentObject(studio)
                .preferredColorScheme(.dark)
                .onAppear {
                    if !isPreview {
                        studio.start()
                        appDelegate.configureMenuBar(with: studio)
                    }
                }
        }
        .windowStyle(.hiddenTitleBar)
        .windowToolbarStyle(.unifiedCompact(showsTitle: false))
        .defaultSize(width: 1280, height: 820)
        .defaultPosition(.center)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("Nouvelle conversation") { studio.newConversation() }
                    .keyboardShortcut("n", modifiers: .command)
            }
            CommandGroup(replacing: .appSettings) {
                Button("Réglages…") { studio.openAppSettings() }
                    .keyboardShortcut(",", modifiers: .command)
            }
        }

    }
}
