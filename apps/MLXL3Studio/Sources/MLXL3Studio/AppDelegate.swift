import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate {
    private var statusItem: NSStatusItem?
    private let popover = NSPopover()
    private weak var studio: StudioModel?

    func applicationDidFinishLaunching(_ notification: Notification) {
        guard let url = Bundle.main.url(forResource: "MLXL3DockV2", withExtension: "icns"),
              let image = NSImage(contentsOf: url) else { return }
        NSApplication.shared.applicationIconImage = image
    }

    func applicationWillTerminate(_ notification: Notification) {
        studio?.modelLibrary.cancelDownload()
        studio?.persistNow()
        studio?.stopServices()
    }

    func configureMenuBar(with studio: StudioModel) {
        guard statusItem == nil else { return }
        self.studio = studio

        let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = item.button {
            button.image = Self.menuBarIcon()
            button.imagePosition = .imageOnly
            button.imageScaling = .scaleProportionallyDown
            button.toolTip = "MLXL3 Desktop"
            button.target = self
            button.action = #selector(togglePopover(_:))
            button.sendAction(on: [.leftMouseUp])
        }

        popover.contentSize = NSSize(width: 340, height: 390)
        popover.behavior = .transient
        popover.animates = true
        popover.delegate = self
        statusItem = item
    }

    @objc private func togglePopover(_ sender: NSStatusBarButton) {
        if popover.isShown {
            popover.performClose(sender)
            sender.state = .off
        } else {
            guard let studio else { return }
            let root = MenuBarPanel { [weak self] size in self?.popover.contentSize = size }
                .environmentObject(studio)
                .environment(\.scenePhase, .active)
            popover.contentViewController = NSHostingController(rootView: root)
            popover.show(relativeTo: sender.bounds, of: sender, preferredEdge: .minY)
            sender.state = .on
            popover.contentViewController?.view.window?.makeKey()
        }
    }

    func popoverDidClose(_ notification: Notification) {
        popover.contentViewController = nil
        statusItem?.button?.state = .off
    }

    static func menuBarIcon() -> NSImage? {
        guard let icon = StudioBrand.logo?.copy() as? NSImage else { return nil }
        icon.size = NSSize(width: 22, height: 22)
        icon.isTemplate = true
        icon.accessibilityDescription = "MLXL3 Desktop"
        return icon
    }
}
