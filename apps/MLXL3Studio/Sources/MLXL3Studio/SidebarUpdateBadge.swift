import SwiftUI

/// Observe download progress here, not across the conversation layout.
struct SidebarUpdateBadge: View {
    @ObservedObject var updater: UpdateManager
    let action: () -> Void

    var body: some View {
        if updater.hasAvailableUpdate {
            Button(action: action) {
                Image(systemName: "arrow.down.circle")
                    .font(.system(size: 13))
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(StudioControlStyle())
            .help("Mise à jour disponible")
            .accessibilityLabel("Mise à jour disponible")
        }
    }
}
