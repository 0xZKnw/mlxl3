import SwiftUI

@main
struct MLXL3StudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @Environment(\.openWindow) private var openWindow
    @StateObject private var studio: StudioModel
    // A QA bundle must stay isolated even when macOS relaunches it without args.
    private static var previewProcess: Bool {
        ProcessInfo.processInfo.arguments.contains("--ui-preview")
            || Bundle.main.bundleIdentifier == "io.mlxl3.designreview"
    }
    private let isPreview = Self.previewProcess

    init() {
        let arguments = ProcessInfo.processInfo.arguments
        if [4, 6, 7].contains(arguments.count), arguments[1] == "--agent-command" {
            AgentSandbox.commandMode(root: arguments[2], command: arguments[3], seconds: arguments.count >= 6 ? Double(arguments[4]) ?? 0 : 5, writable: arguments.count == 4 || arguments[5] == "write",
                                     mode: arguments.count == 7 && arguments[6].hasPrefix("tail") ? .headAndTail : .stop,
                                     network: arguments.count == 7 && arguments[6] == "tail+net")
        }
        if arguments.count == 10, arguments[1] == "--agent-js" {
            AgentJavaScript.supervisorMode(arguments: arguments)
        }
        if arguments.count == 4, arguments[1] == "--agent-js-kernel" {
            AgentJavaScript.kernelMode(arguments: arguments)
        }
        if (3...5).contains(arguments.count), arguments[1] == "--agent-tools" {
            AgentSandbox.serve(root: arguments[2], permissionsPath: arguments.count >= 4 ? arguments[3] : nil,
                               longCalls: arguments.count == 5 && arguments[4] == "long-calls")
        }
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
        WindowGroup(isPreview ? "MLXL3 — Aperçu" : "MLXL3 Desktop", id: "studio") {
            Group {
                if isPreview && ProcessInfo.processInfo.arguments.contains("--ui-preview-menu") {
                    MenuBarPanel()
                } else { StudioView() }
            }
                .environmentObject(studio)
                .studioAppearance(studio)
                // Tokens resolve the style when drawn: rebuild the window on a style change.
                .id(studio.style)
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
                Button(L("Nouvelle conversation", "New conversation")) { studio.newConversation() }
                    .keyboardShortcut("n", modifiers: .command)
                Button(L("Rétablir la conversation supprimée", "Restore deleted conversation")) { studio.undoDeleteConversation() }
            }
            CommandGroup(replacing: .appSettings) {
                Button(L("Réglages…", "Settings…")) { studio.openAppSettings() }
                    .keyboardShortcut(",", modifiers: .command)
            }
            // ⌘1…⌘5 switch modes, as the sidebar rows show on hover; they also work
            // with the sidebar hidden. selectMode keeps its own guards.
            CommandGroup(before: .sidebar) {
                ForEach(Array(StudioMode.allCases.enumerated()), id: \.element) { index, mode in
                    Button(mode.displayTitle) { studio.selectMode(mode) }
                        .keyboardShortcut(KeyEquivalent(Character(String(index + 1))), modifiers: .command)
                        .disabled(mode == .lab ? !studio.canOpenLab : studio.isGenerating)
                }
                Divider()
            }
            CommandGroup(after: .windowList) {
                if Image21Hardware.available {
                    Button(L("Générer une image", "Generate an image")) {
                        studio.showImages = true
                        if let window = NSApp.windows.first(where: { $0.canBecomeMain && !($0 is NSPanel) }) {
                            window.makeKeyAndOrderFront(nil)
                        } else { openWindow(id: "studio") }
                    }
                    .disabled(studio.labComparison.isRunning || studio.showAppSettings || studio.showModelManager || studio.showMCPManager || studio.showWorkspace || studio.studioDocument != nil)
                }
            }
        }
    }
}
