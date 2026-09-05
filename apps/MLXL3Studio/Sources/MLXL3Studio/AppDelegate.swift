import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate, NSPopoverDelegate {
    private var statusItem: NSStatusItem?
    private let popover = NSPopover()
    private weak var studio: StudioModel?

    func applicationWillTerminate(_ notification: Notification) {
        studio?.modelLibrary.cancelDownload()
        studio?.persistNow()
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

        let root = MenuBarPanel().environmentObject(studio)
        popover.contentViewController = NSHostingController(rootView: root)
        popover.contentSize = NSSize(width: 360, height: 470)
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
            popover.show(relativeTo: sender.bounds, of: sender, preferredEdge: .minY)
            sender.state = .on
            popover.contentViewController?.view.window?.makeKey()
        }
    }

    func popoverDidClose(_ notification: Notification) {
        statusItem?.button?.state = .off
    }

    private static func menuBarIcon() -> NSImage {
        let image = NSImage(size: NSSize(width: 18, height: 18), flipped: false) { bounds in
            let arc = NSBezierPath()
            arc.appendArc(
                withCenter: NSPoint(x: bounds.midX, y: bounds.midY),
                radius: 6.2,
                startAngle: 44,
                endAngle: 322,
                clockwise: false
            )
            arc.lineWidth = 1.8
            arc.lineCapStyle = .round
            NSColor.white.setStroke()
            arc.stroke()

            NSColor.white.setFill()
            NSBezierPath(ovalIn: NSRect(x: bounds.midX - 1.25, y: bounds.midY - 1.25, width: 2.5, height: 2.5)).fill()
            return true
        }
        image.isTemplate = true
        image.accessibilityDescription = "MLXL3 Desktop"
        return image
    }
}
