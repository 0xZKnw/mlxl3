import SwiftUI

@main
struct MLXL3StudioApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var appDelegate
    @StateObject private var studio = StudioModel()

    var body: some Scene {
        WindowGroup("MLXL3 Desktop") {
            StudioView()
                .environmentObject(studio)
                .preferredColorScheme(.dark)
                .onAppear {
                    studio.start()
                    appDelegate.configureMenuBar(with: studio)
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
        }

    }
}
