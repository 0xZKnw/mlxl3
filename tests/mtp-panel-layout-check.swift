// Combined with StudioModel.swift so fixture setters do not ship in the app.
import AppKit
import SwiftUI

extension StudioModel {
    fileprivate func seedToolbar(_ state: String) {
        mtpSupported = true
        mtpRememberedEnabled = state != "off"
        mtpEnabled = state == "on"
        mtpError = state == "error" ? "Fixture error" : nil
    }

    fileprivate func seedDFlashToolbar(_ state: String) {
        dflashSupported = true
        dflash2Enabled = state == "on"
        dflashDownloading = state == "preparing"
        dflashDownloadError = state == "error" ? "Fixture error" : nil
    }

    fileprivate func seedDSpark(supported: Bool?, enabled: Bool, mtp: Bool?) {
        dsparkSupported = supported
        dsparkEnabled = enabled
        mtpSupported = mtp
    }

    fileprivate func seedDFlashLayout(results: Bool) {
        dflashSupported = true
        dflashDownloading = false
        dflashDownloadError = nil
        dflash2Enabled = true
        dflashTuneSupported = true
        dflashDraftPath = "/fixture/draft"
        dflashMode = 1
        dflashTuneRows = results ? [8.8, 21.1, 13.8, 20.4].enumerated().map { depth, rate in
            MTPTuningRow(depth: depth, decodeTPS: rate, decodeTokens: 100,
                         decodeSeconds: 100 / rate, acceptedTokens: depth == 0 ? 0 : 50,
                         proposedTokens: depth == 0 ? 0 : 100, eligible: depth != 1,
                         reason: depth == 1 ? "target_mismatch" : nil,
                         tokenHashes: ["fixture-a", "fixture-b"])
        } : []
    }

    fileprivate func seedMTPLayout(results: Bool) {
        mtpError = nil
        mtpEnabled = true
        mtpRememberedEnabled = true
        mtpMaxDepth = 3
        mtpSupported = true
        mtpTuneSupported = true
        mtpHeadPath = "/fixture/head"
        mtpDepth = 1
        mtpTuneRows = results ? [51.2, 71.8, 67.7, 66.6].enumerated().map { depth, rate in
            MTPTuningRow(depth: depth, decodeTPS: rate, decodeTokens: 100,
                         decodeSeconds: 100 / rate, acceptedTokens: depth == 0 ? 0 : 50,
                         proposedTokens: depth == 0 ? 0 : 100, eligible: true, reason: nil,
                         tokenHashes: ["fixture-a", "fixture-b"])
        } : []
    }
}

@main struct MTPPanelLayoutCheck {
    @MainActor static func main() async throws {
        _ = NSApplication.shared
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-mtp-layout-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let suite = "io.mlxl3.mtp-layout." + UUID().uuidString
        let prefs = UserDefaults(suiteName: suite)!
        defer { prefs.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("history.json"),
            updateManager: UpdateManager(releasesURL: root.appendingPathComponent("no-network.json")),
            isPreview: true, preferences: prefs)
        var failures: [String] = []
        var cases = 0
        for language in [AppLanguage.fr, .en] {
            AppLocalization.set(language)
            for state in ["off", "on", "pending", "error"] {
                studio.seedToolbar(state)
                let host = NSHostingView(rootView: MTPToolbarControl().environmentObject(studio)
                    .studioAppearance(.dark).background(StudioTheme.canvas))
                let size = host.fittingSize
                cases += 1
                if !size.width.isFinite || size.width > 95 || size.height > 30 || size.height <= 0 {
                    failures.append("\(language.rawValue)-\(state): MTP toolbar exceeds its compact bounds")
                }
                if let output = ProcessInfo.processInfo.environment["MLXL3_MTP_LAYOUT_SNAPSHOTS"] {
                    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 120, height: 50), styleMask: .borderless, backing: .buffered, defer: false)
                    window.isReleasedWhenClosed = false; window.contentView = host
                    host.frame = NSRect(x: 0, y: 0, width: 120, height: 50)
                    try await Task.sleep(for: .milliseconds(100)); host.layoutSubtreeIfNeeded()
                    let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                    host.cacheDisplay(in: host.bounds, to: bitmap)
                    try bitmap.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: output).appendingPathComponent("toolbar-\(language.rawValue)-\(state).png"))
                    window.close()
                }
            }
            for state in ["off", "on", "preparing", "error"] {
                studio.seedDFlashToolbar(state)
                let host = NSHostingView(rootView: DFlashToolbarControl().environmentObject(studio)
                    .studioAppearance(.dark).background(StudioTheme.canvas))
                let size = host.fittingSize
                cases += 1
                if !size.width.isFinite || size.width > 110 || size.height > 30 || size.height <= 0 {
                    failures.append("\(language.rawValue)-dflash-\(state): DFlash2 toolbar exceeds its compact bounds (\(size))")
                }
            }
            // DSpark: the toolbar shows it for a model with a DSpark drafter, in place of a disabled MTP control
            // when the model has no MTP head; other models keep the MTP control exactly as before.
            for (supported, mtp, wantsDSpark, wantsMTP) in [(Bool?.some(true), Bool?.some(false), true, false),
                                                           (true, true, true, true),
                                                           (nil, true, false, true), (false, false, false, true),
                                                           (nil, false, false, true)] {
                studio.seedDSpark(supported: supported, enabled: true, mtp: mtp)
                cases += 1
                if studio.dsparkToolbarVisible != wantsDSpark || studio.mtpToolbarVisible != wantsMTP {
                    failures.append("dspark=\(String(describing: supported)) mtp=\(String(describing: mtp)): toolbar shows DSpark \(studio.dsparkToolbarVisible), MTP \(studio.mtpToolbarVisible)")
                }
            }
            for enabled in [false, true] {
                studio.seedDSpark(supported: true, enabled: enabled, mtp: false)
                let host = NSHostingView(rootView: DSparkToolbarControl().environmentObject(studio)
                    .studioAppearance(.dark).background(StudioTheme.canvas))
                let size = host.fittingSize
                cases += 1
                if !size.width.isFinite || size.width > 110 || size.height > 30 || size.height <= 0 {
                    failures.append("\(language.rawValue)-dspark-\(enabled): DSpark toolbar exceeds its compact bounds (\(size))")
                }
                for width in [200, 244, 280, 320] {
                    let view = DSparkSettingsContent().environmentObject(studio)
                        .frame(width: CGFloat(width), alignment: .leading)
                        .padding(16).background(StudioTheme.canvas)
                        .studioAppearance(.dark)
                        .foregroundStyle(StudioTheme.ink)
                    let host = NSHostingView(rootView: view)
                    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width + 32, height: 600),
                        styleMask: .borderless, backing: .buffered, defer: false)
                    window.isReleasedWhenClosed = false; window.contentView = host
                    host.frame = NSRect(x: 0, y: 0, width: width + 32, height: 600)
                    try await Task.sleep(for: .milliseconds(100)); host.layoutSubtreeIfNeeded()
                    let height = host.fittingSize.height - 32
                    // Switch, state and the wrapped explanation.
                    let limit: CGFloat = 200
                    cases += 1
                    let label = "\(language.rawValue)-dspark-\(width)-enabled\(enabled)"
                    print("\(label): height=\(height), limit=\(limit)")
                    if !height.isFinite || height <= 0 || height > limit {
                        failures.append(label + ": DSpark panel consumed excessive vertical space")
                    }
                    if let output = ProcessInfo.processInfo.environment["MLXL3_MTP_LAYOUT_SNAPSHOTS"] {
                        let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                        host.cacheDisplay(in: host.bounds, to: bitmap)
                        try bitmap.representation(using: .png, properties: [:])!.write(
                            to: URL(fileURLWithPath: output).appendingPathComponent(label + ".png"))
                    }
                    window.close()
                }
            }
            studio.seedDSpark(supported: nil, enabled: true, mtp: true)
            for results in [false, true] {
                studio.seedDFlashLayout(results: results)
                for width in [200, 244, 280, 320] {
                    let view = DFlashSettingsContent().environmentObject(studio)
                        .frame(width: CGFloat(width), alignment: .leading)
                        .padding(16).background(StudioTheme.canvas)
                        .studioAppearance(.dark)
                        .foregroundStyle(StudioTheme.ink)
                    let host = NSHostingView(rootView: view)
                    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width + 32, height: 800),
                        styleMask: .borderless, backing: .buffered, defer: false)
                    window.isReleasedWhenClosed = false; window.contentView = host
                    host.frame = NSRect(x: 0, y: 0, width: width + 32, height: 800)
                    try await Task.sleep(for: .milliseconds(100)); host.layoutSubtreeIfNeeded()
                    let height = host.fittingSize.height - 32
                    // Switch, state, picker, Tune, the four result rows and the collapsed draft options.
                    let limit: CGFloat = results ? 560 : 330
                    cases += 1
                    let label = "\(language.rawValue)-dflash-\(width)-results\(results)"
                    print("\(label): height=\(height), limit=\(limit)")
                    if !height.isFinite || height <= 0 || height > limit {
                        failures.append(label + ": DFlash2 panel consumed excessive vertical space")
                    }
                    if let output = ProcessInfo.processInfo.environment["MLXL3_MTP_LAYOUT_SNAPSHOTS"] {
                        let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                        host.cacheDisplay(in: host.bounds, to: bitmap)
                        try bitmap.representation(using: .png, properties: [:])!.write(
                            to: URL(fileURLWithPath: output).appendingPathComponent(label + ".png"))
                    }
                    window.close()
                }
            }
            for results in [false, true] {
                studio.seedMTPLayout(results: results)
                for width in [200, 244, 280, 320] {
                    let view = MTPTuningControls().environmentObject(studio)
                        .frame(width: CGFloat(width), alignment: .leading)
                        .padding(16).background(StudioTheme.canvas)
                        .studioAppearance(.dark)
                        .foregroundStyle(StudioTheme.ink)
                    let host = NSHostingView(rootView: view)
                    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width + 32, height: 600),
                        styleMask: .borderless, backing: .buffered, defer: false)
                    window.isReleasedWhenClosed = false; window.contentView = host
                    host.frame = NSRect(x: 0, y: 0, width: width + 32, height: 600)
                    try await Task.sleep(for: .milliseconds(100)); host.layoutSubtreeIfNeeded()
                    let height = host.fittingSize.height - 32
                    // The current tuning guidance wraps below the picker at narrow widths.
                    let limit: CGFloat = results ? 370 : 160
                    cases += 1
                    let label = "\(language.rawValue)-\(width)-results\(results)"
                    print("\(label): height=\(height), limit=\(limit)")
                    if !height.isFinite || height <= 0 || height > limit {
                        failures.append(label + ": depth selector consumed excessive vertical space")
                    }
                    if let output = ProcessInfo.processInfo.environment["MLXL3_MTP_LAYOUT_SNAPSHOTS"] {
                        let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
                        host.cacheDisplay(in: host.bounds, to: bitmap)
                        try bitmap.representation(using: .png, properties: [:])!.write(
                            to: URL(fileURLWithPath: output).appendingPathComponent(label + ".png"))
                    }
                    window.close()
                }
            }
        }
        AppLocalization.set(.fr)
        if let output = ProcessInfo.processInfo.environment["MLXL3_MTP_LAYOUT_SNAPSHOTS"] {
            studio.seedMTPLayout(results: true)
            let host = NSHostingView(rootView: GenerationInspector().environmentObject(studio)
                .studioAppearance(.dark).foregroundStyle(StudioTheme.ink))
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 286, height: 800),
                styleMask: .borderless, backing: .buffered, defer: false)
            window.isReleasedWhenClosed = false; window.contentView = host
            host.frame = NSRect(x: 0, y: 0, width: 286, height: 800)
            try await Task.sleep(for: .milliseconds(300)); host.layoutSubtreeIfNeeded()
            let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds)!
            host.cacheDisplay(in: host.bounds, to: bitmap)
            try bitmap.representation(using: .png, properties: [:])!.write(
                to: URL(fileURLWithPath: output).appendingPathComponent("inspector-286.png"))
            window.close()
        }
        fflush(nil)
        guard failures.isEmpty else {
            throw NSError(domain: "MTPPanelLayoutCheck", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: failures.joined(separator: "; ")])
        }
        print("MTP, DFlash2 and DSpark layout checks passed: \(cases) toolbar and panel cases, FR/EN, 200–320pt, with/without results")
    }
}
