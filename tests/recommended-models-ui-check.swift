import AppKit
import SwiftUI

extension HubBrowserView {
    fileprivate var recommendedForCheck: some View { recommendedCards }
}

@main struct RecommendedModelsUICheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-recommendations-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "mlxl3.recommendations.\(UUID())"
        let preferences = UserDefaults(suiteName: suite)!
        defer { preferences.removePersistentDomain(forName: suite) }
        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("history.json"), isPreview: true, preferences: preferences)
        for width in [480, 840] {
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: 1100), styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false
            let host = NSHostingView(rootView: HubBrowserView(library: studio.modelLibrary).recommendedForCheck
                .padding(20).environmentObject(studio).studioAppearance(.midnightBlue)
                .foregroundStyle(StudioTheme.ink)
                .frame(width: CGFloat(width), height: 1100, alignment: .topLeading)
                .background(StudioTheme.canvas(for: .midnightBlue)))
            window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: width, height: 1100)
            try await Task.sleep(for: .milliseconds(200))
            host.layoutSubtreeIfNeeded()
            guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { fatalError("No recommendation bitmap") }
            host.cacheDisplay(in: host.bounds, to: bitmap)
            print("Recommendations \(width)pt: \(bitmap.pixelsWide)×\(bitmap.pixelsHigh) pixels")
            if let output = ProcessInfo.processInfo.environment["MLXL3_RECOMMENDATION_SNAPSHOTS"] {
                try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output).appendingPathComponent("recommendations-\(width).png"))
            }
            precondition(bitmap.pixelsWide >= width && bitmap.pixelsHigh >= 1100, "Recommendation view did not fill its test window")
            var brightPixels = 0
            for y in stride(from: 0, to: bitmap.pixelsHigh, by: 4) {
                for x in stride(from: 0, to: bitmap.pixelsWide, by: 4) {
                    if let color = bitmap.colorAt(x: x, y: y)?.usingColorSpace(.sRGB),
                       color.redComponent > 0.6, color.greenComponent > 0.6, color.blueComponent > 0.6 {
                        brightPixels += 1
                    }
                }
            }
            precondition(brightPixels > 80, "Recommendation labels did not render")
            window.close()
        }
        print("Recommended models UI checks passed: 480 and 840 point layouts")
    }
}
