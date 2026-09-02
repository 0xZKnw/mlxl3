import SwiftUI

@main
struct MLXL3StudioApp: App {
    @StateObject private var studio = StudioModel()

    var body: some Scene {
        WindowGroup {
            StudioView()
                .environmentObject(studio)
                .preferredColorScheme(.dark)
                .onAppear { studio.start() }
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
