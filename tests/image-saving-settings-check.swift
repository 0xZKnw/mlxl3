import AppKit
import SwiftUI

@MainActor
func checkImageSaveSettings(studio: StudioModel, window: NSWindow, preferences: UserDefaults) async throws {
    func failure(_ message: String) -> NSError {
        NSError(domain: "ImageSavingSettings", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    func findSwitch(in view: NSView) -> NSSwitch? {
        if let toggle = view as? NSSwitch { return toggle }
        for child in view.subviews {
            if let toggle = findSwitch(in: child) { return toggle }
        }
        return nil
    }

    guard studio.imageGeneration.usageStore.historyDirectory != GeneratedImageHistory.defaultDirectory else {
        throw failure("Opening the preview Lab must not read the personal image gallery")
    }
    let key = ImageGeneration.autoSavePreferenceKey
    preferences.removeObject(forKey: key)
    studio.appSettingsPage = .privacy
    let host = NSHostingView(rootView: AppSettingsView().environmentObject(studio).defaultAppStorage(preferences))
    window.contentView = host
    window.setContentSize(NSSize(width: 1000, height: 760))
    let deadline = ContinuousClock.now.advanced(by: .seconds(4))
    while findSwitch(in: host) == nil && ContinuousClock.now < deadline {
        try await Task.sleep(for: .milliseconds(20))
    }
    guard let toggle = findSwitch(in: host), toggle.state == .off, !preferences.bool(forKey: key) else {
        throw failure("Privacy settings must offer an initially disabled switch for saving generated images")
    }
    for expected in [true, false] {
        toggle.performClick(nil)
        let deadline = ContinuousClock.now.advanced(by: .seconds(2))
        while preferences.bool(forKey: key) != expected && ContinuousClock.now < deadline {
            try await Task.sleep(for: .milliseconds(20))
        }
        guard preferences.bool(forKey: key) == expected, (toggle.state == .on) == expected else {
            throw failure("The Privacy switch did not persist the selected image saving preference")
        }
    }
}
