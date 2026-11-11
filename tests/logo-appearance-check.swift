import AppKit
import SwiftUI

@main struct LogoAppearanceCheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        precondition(StudioBrand.logoURL != nil && StudioBrand.logo != nil, "Real artwork loads")
        var checks = 0
        func check(_ value: Bool, _ message: String) {
            checks += 1
            precondition(value, message)
        }
        if ProcessInfo.processInfo.environment["MLXL3_TEST_PACKAGED_ICON"] == "1" {
            check(StudioBrand.logoURL == Bundle.main.url(forResource: "AppIcon", withExtension: "png"),
                  "Packaged transparent artwork was not loaded")
            let iconURL = Bundle.main.url(forResource: "MLXL3DockV2", withExtension: "icns")!
            check(Bundle.main.object(forInfoDictionaryKey: "CFBundleIconFile") as? String == iconURL.lastPathComponent,
                  "Bundle must name the packaged Dock icon")
            let expected = NSImage(contentsOf: iconURL)!
            let sentinel = NSImage(size: NSSize(width: 64, height: 64), flipped: false) { bounds in
                NSColor.systemPink.setFill()
                bounds.fill()
                return true
            }
            NSApplication.shared.applicationIconImage = sentinel
            AppDelegate().applicationDidFinishLaunching(Notification(name: NSApplication.didFinishLaunchingNotification))
            let actual = NSApplication.shared.applicationIconImage!
            func sample(_ image: NSImage, _ x: CGFloat, _ y: CGFloat) -> NSColor {
                let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil)!
                let bitmap = NSBitmapImageRep(cgImage: cgImage)
                return bitmap.colorAt(x: Int(CGFloat(bitmap.pixelsWide - 1) * x),
                                      y: Int(CGFloat(bitmap.pixelsHigh - 1) * y))!
                    .usingColorSpace(.deviceRGB)!
            }
            for (x, y) in [(CGFloat(0.15), CGFloat(0.5)), (0.5, 0.15), (0.5, 0.5)] {
                let a = sample(actual, x, y), b = sample(expected, x, y)
                check(abs(a.redComponent - b.redComponent) < 0.06
                      && abs(a.greenComponent - b.greenComponent) < 0.06
                      && abs(a.blueComponent - b.blueComponent) < 0.06,
                      "Running app must use packaged icon rather than an old cached icon")
            }
            check(sample(actual, 0.15, 0.5).greenComponent > 0.8,
                  "Oracle rejects the pink sentinel")
        } else {
            check(Bundle.main.url(forResource: "MLXL3DockV2", withExtension: "icns") == nil,
                  "Unbundled test must not find the Dock icon")
            let before = NSApplication.shared.applicationIconImage!.tiffRepresentation!
            AppDelegate().applicationDidFinishLaunching(Notification(name: NSApplication.didFinishLaunchingNotification))
            check(NSApplication.shared.applicationIconImage!.tiffRepresentation == before,
                  "A missing icon must leave the current app icon unchanged")
        }
        func render<V: View>(_ view: V, theme: AppTheme) async throws -> NSBitmapImageRep {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 128, height: 128),
                                  styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            defer { window.close() }
            let host = NSHostingView(rootView: view.frame(width: 96, height: 96).padding(16)
                .background(StudioTheme.canvas(for: theme)).studioAppearance(theme))
            window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: 128, height: 128)
            try await Task.sleep(for: .milliseconds(250))
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
                preconditionFailure("Logo bitmap missing")
            }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            precondition(bitmap.pixelsWide >= 128 && bitmap.pixelsHigh >= 128)
            return bitmap
        }
        func pixel(_ bitmap: NSBitmapImageRep, _ x: Int, _ y: Int) -> NSColor {
            let scale = Double(bitmap.pixelsWide) / 128
            return bitmap.colorAt(x: Int(Double(x) * scale), y: Int(Double(y) * scale))!
                .usingColorSpace(.deviceRGB)!
        }
        func visibleHalves(_ bitmap: NSBitmapImageRep, theme: AppTheme) -> (Int, Int) {
            let background = pixel(bitmap, 2, 2).redComponent
            var counts = [0, 0]
            for y in stride(from: 16, to: 112, by: 2) {
                for x in stride(from: 16, to: 112, by: 2) {
                    let red = pixel(bitmap, x, y).redComponent
                    let contrast = theme == .light ? background - red : red - background
                    if contrast > (theme == .light ? 0.5 : 0.3) { counts[x < 64 ? 0 : 1] += 1 }
                }
            }
            return (counts[0], counts[1])
        }
        for theme in [AppTheme.light, .dark, .midnightBlue] {
            let bitmap = try await render(MonogramMark(size: 96), theme: theme)
            if let folder = ProcessInfo.processInfo.environment["MLXL3_LOGO_SNAPSHOTS"] {
                let output = URL(fileURLWithPath: folder).appendingPathComponent("logo-\(theme.rawValue).png")
                try bitmap.representation(using: .png, properties: [:])!.write(to: output)
            }
            let background = pixel(bitmap, 2, 2)
            for point in [(18, 18), (109, 18), (18, 109), (109, 109), (64, 18), (64, 109)] {
                let value = pixel(bitmap, point.0, point.1)
                check(abs(value.redComponent - background.redComponent) < 0.015
                      && abs(value.greenComponent - background.greenComponent) < 0.015
                      && abs(value.blueComponent - background.blueComponent) < 0.015,
                      "Logo background is transparent in " + theme.rawValue)
            }
            let halves = visibleHalves(bitmap, theme: theme)
            print("Logo contrast \(theme.rawValue): left=\(halves.0), right=\(halves.1)")
            check(halves.0 > 100 && halves.1 > 100, "Both logo halves contrast in " + theme.rawValue)
            let blank = try await render(Color.clear, theme: theme)
            let empty = visibleHalves(blank, theme: theme)
            check(empty.0 == 0 && empty.1 == 0, "Oracle rejects missing logo in " + theme.rawValue)
        }
        let old = try await render(Image(nsImage: StudioBrand.logo!).resizable().scaledToFit(), theme: .light)
        check(visibleHalves(old, theme: .light).0 <= 100, "Oracle detects the original untinted light logo")
        print("Logo appearance checks passed: \(checks), actual artwork, light/dark/blue, transparent corners and negative controls")
    }
}
