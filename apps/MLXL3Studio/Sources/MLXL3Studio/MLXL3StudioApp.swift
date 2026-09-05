import SwiftUI

@main
struct MLXL3StudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var studio: StudioModel
    // A QA bundle must stay isolated even when macOS relaunches it without args.
    private static var previewProcess: Bool {
        ProcessInfo.processInfo.arguments.contains("--ui-preview")
            || Bundle.main.bundleIdentifier == "io.mlxl3.designreview"
    }
    private let isPreview = Self.previewProcess

    init() {
        if ProcessInfo.processInfo.arguments.contains("--check-chat-timeline") {
            do {
                try ChatTimelineCheck.run()
                print("Chat timeline checks passed: ordering, progress, streaming, persistence, migration, cancellation")
                exit(0)
            } catch {
                print("Chat timeline checks failed: \(error)")
                exit(1)
            }
        }
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
        let preview = Self.previewProcess
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
