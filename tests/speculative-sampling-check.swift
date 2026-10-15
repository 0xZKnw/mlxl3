import AppKit
import Foundation
import SwiftUI

// Concatenated with StudioModel.swift to exercise its real private event path.
extension StudioModel {
  fileprivate func samplingFixtureEvent(_ json: String) throws {
    handle(try JSONDecoder().decode(BridgeEvent.self, from: Data(json.utf8)))
  }
  fileprivate func samplingFixtureIntent(mtp: Bool, dflash: Bool) {
    mtpEnabled = mtp
    dflash2Enabled = dflash
  }
}

@main struct SpeculativeSamplingCheck {
  @MainActor static func main() async throws {
    var checks = 0
    func check(_ condition: Bool, _ message: String) throws {
      checks += 1
      if !condition {
        throw NSError(
          domain: "SpeculativeSamplingCheck", code: checks,
          userInfo: [NSLocalizedDescriptionKey: message])
      }
    }
    func wait(_ description: String, until condition: () -> Bool) async throws {
      for _ in 0..<300 {
        if condition() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      throw NSError(
        domain: "SpeculativeSamplingCheck", code: -1,
        userInfo: [NSLocalizedDescriptionKey: "Deadline waiting for " + description])
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-sampling-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    let suite = "io.mlxl3.sampling-check." + UUID().uuidString
    let preferences = UserDefaults(suiteName: suite)!
    defer { preferences.removePersistentDomain(forName: suite) }
    setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
    setenv("MLXL3_HOME", root.path, 1)
    let requestsURL = root.appendingPathComponent("requests.jsonl")
    setenv("MLXL3_SAMPLING_CHECK_REQUESTS", requestsURL.path, 1)
    defer { unsetenv("MLXL3_SAMPLING_CHECK_REQUESTS") }
    let file = root.appendingPathComponent("workspace.json")
    let conversation = Conversation()
    try ConversationStore.write(
      WorkspaceSnapshot(
        selectedConversationID: conversation.id,
        selectedModelName: "lfm-moe", conversations: [conversation.snapshot],
        temperature: 0.73, topK: 37, repetitionPenalty: 1.12, systemPrompt: ""), to: file)
    preferences.set(true, forKey: "studio.mtpEnabled")
    let relaunched = StudioModel(
      conversationFileURL: file, isPreview: true, preferences: preferences)
    try check(relaunched.mtpEnabled, "Relaunch lost global ON")
    try check(
      relaunched.temperature == 0.73 && relaunched.topK == 37
        && relaunched.repetitionPenalty == 1.12,
      "Unsupported LFM relaunch must preserve the user's sampling settings")
    relaunched.setMTPEnabled(false)
    try check(
      !relaunched.mtpEnabled && !preferences.bool(forKey: "studio.mtpEnabled"),
      "MTP OFF must work before model readiness is known")
    preferences.set(true, forKey: "studio.dflash2Enabled")
    preferences.set("/fixture/draft", forKey: "studio.dflashDraftPath")
    let flashRelaunch = StudioModel(
      conversationFileURL: file, isPreview: true, preferences: preferences)
    try check(
      flashRelaunch.dflash2Enabled && !flashRelaunch.samplingLocked,
      "DFlash saved intent must not lock an unknown model")
    try check(
      flashRelaunch.temperature == 0.73 && flashRelaunch.topK == 37
        && flashRelaunch.repetitionPenalty == 1.12,
      "DFlash relaunch overwrote sampling")
    flashRelaunch.setDFlash2Enabled(false)
    try check(
      !flashRelaunch.dflash2Enabled && !preferences.bool(forKey: "studio.dflash2Enabled"),
      "DFlash OFF must work before model readiness is known")

    func local(_ name: String, _ type: String) -> LocalModel {
      LocalModel(
        name: name, path: root.appendingPathComponent(name).path, modelType: type,
        format: "EXL3", bits: 2.49, sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
    }
    // Exhaust every Boolean intent and tri-state capability on real computed properties.
    // The independent oracle keeps the documented legacy Qwen fallback explicit.
    let types = ["lfm2", "lfm2_moe", "unknown", "qwen3_5", "qwen3_5_moe"]
    let capabilities: [Bool?] = [nil, false, true]
    for type in types {
      for mtp in [false, true] {
        for flash in [false, true] {
          for mtpCapability in capabilities {
            for flashCapability in capabilities {
              let model = StudioModel(
                conversationFileURL: file, isPreview: true, preferences: preferences)
              model.models = [local("qwen3.6-35b-a3b", type)]
              model.selectedModelName = "qwen3.6-35b-a3b"
              var ready: [String: Any] = ["type": "ready"]
              if let mtpCapability { ready["mtp_supported"] = mtpCapability }
              if let flashCapability { ready["dflash_supported"] = flashCapability }
              let data = try JSONSerialization.data(withJSONObject: ready)
              try model.samplingFixtureEvent(String(decoding: data, as: UTF8.self))
              model.samplingFixtureIntent(mtp: mtp, dflash: flash)
              let wantsMTP = mtp && (mtpCapability ?? ["qwen3_5", "qwen3_5_moe"].contains(type))
              let wantsFlash = flash && (flashCapability ?? (type == "qwen3_5_moe"))
              let locked = wantsMTP || wantsFlash
              try check(
                model.mtpApplicable == wantsMTP && model.dflash2Applicable == wantsFlash,
                "Intent/capability mismatch for \(type), \(String(describing: mtpCapability)), \(String(describing: flashCapability))"
              )
              try check(
                model.samplingLocked == locked && model.samplingTemperature == (locked ? 0 : 0.73)
                  && model.samplingTopK == (locked ? 1 : 37)
                  && model.samplingRepetitionPenalty == (locked ? 1 : 1.12),
                "Displayed/request sampling disagrees with applicability")
              try check(
                model.temperature == 0.73 && model.topK == 37 && model.repetitionPenalty == 1.12,
                "Capability event changed saved sampling")
              try check(
                model.mtpActive == nil && model.dflashActive == nil,
                "Intent was incorrectly reported as observed engine activity")
            }
          }
        }
      }
    }

    let model = StudioModel(conversationFileURL: file, preferences: preferences)
    model.models = [
      local("auto-dense", "qwen3_5"), local("lfm-dense", "lfm2"),
      local("lfm-moe", "lfm2_moe"), local("plain", "qwen3_5"), local("auto-fail", "qwen3_5"),
      local("dflash-good", "audit"),
    ]
    defer { model.ejectModel() }
    func select(_ name: String) async throws {
      model.selectModel(name)
      try await wait("ready " + name) {
        model.engineState.isReady && !model.mtpDownloading && !model.dflashDownloading
      }
    }
    _ = NSApplication.shared
    model.setLanguage(.en)
    let host = NSHostingView(rootView: GenerationInspector().environmentObject(model))
    let window = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 400, height: 3000),
      styleMask: .borderless, backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    host.frame = NSRect(x: 0, y: 0, width: 400, height: 3000)
    window.orderFront(nil)
    defer { window.close() }
    // Native AX descendants are unavailable in headless NSHostingView (as in
    // response-ui-check). Verify production wiring explicitly, plus rendering.
    let inspectorSource = try String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8)
    for required in [
      "value: Binding(get: { studio.samplingTemperature }, set: { studio.temperature = $0 })",
      "get: { Double(studio.samplingTopK) }",
      "value: Binding(get: { studio.samplingRepetitionPenalty }, set: { studio.repetitionPenalty = $0 })",
      ".disabled(studio.samplingLocked)",
      "let mode = studio.dflash2Applicable ? \"DFlash2\" : \"MTP\"",
      "if studio.mtpAvailable || studio.mtpEnabled || studio.mtpDownloading {",
      "if studio.dflash2Available || studio.dflash2Enabled || studio.dflashDownloading {",
      "get: { studio.mtpEnabled || studio.mtpDownloading }",
      "set: { studio.setMTPEnabled($0) }",
      "get: { studio.dflash2Enabled || studio.dflashDownloading }",
      "set: { studio.setDFlash2Enabled($0) }",
      "Toggle(studio.mtpAvailable",
      "Toggle(studio.dflash2Available",
      "MTP preference (inactive for this model)",
      "DFlash2 preference (inactive for this model)",
      "MTPTuningControls().id(\"mtp-tuning\").disabled(!studio.mtpAvailable)",
      "MTPTuningControls(dflash: true).disabled(!studio.dflash2Available)",
      ".disabled(studio.isGenerating || (!(studio.mtpEnabled || studio.mtpDownloading)",
      ".disabled(studio.isGenerating || (!(studio.dflash2Enabled || studio.dflashDownloading)",
    ] {
      try check(
        inspectorSource.contains(required), "Missing inspector production wiring: " + required)
    }
    func checkInspector(greedy: Bool) async throws {
      host.layoutSubtreeIfNeeded()
      try await Task.sleep(for: .milliseconds(40))
      try check(host.fittingSize.height > 0, "Inspector failed native layout")
      guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
        throw NSError(
          domain: "SpeculativeSamplingCheck", code: -2,
          userInfo: [NSLocalizedDescriptionKey: "Inspector native bitmap unavailable"])
      }
      host.cacheDisplay(in: host.bounds, to: bitmap)
      try check(
        bitmap.pixelsWide > 0 && bitmap.pixelsHigh > 0, "Inspector rendered an empty bitmap")
      try check(model.samplingLocked == greedy, "Inspector and production sampling differ")
    }
    func send(mtp: Bool, flash: Bool, greedy: Bool) async throws {
      model.draft = "sampling wire check"
      try check(model.canSend, "Fixture cannot send")
      model.send()
      try await wait("generation") { !model.isGenerating }
      let rows = try String(contentsOf: requestsURL, encoding: .utf8).split(separator: "\n")
        .map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
      let request = rows.compactMap { $0["request"] as? [String: Any] }.last {
        $0["type"] as? String == "generate"
      }!
      try check(
        request["mtp"] as? Bool == mtp && request["dflash2"] as? Bool == flash,
        "Production bridge sent an inapplicable mode")
      try check(
        request["temperature"] as? Double == (greedy ? 0 : 0.73)
          && request["top_k"] as? Int == (greedy ? 1 : 37)
          && request["repetition_penalty"] as? Double == (greedy ? 1 : 1.12),
        "Production bridge sampling mismatch")
      try check(
        model.temperature == 0.73 && model.topK == 37 && model.repetitionPenalty == 1.12,
        "Generation changed saved sampling")
    }
    preferences.set(true, forKey: "studio.mtpEnabled")
    try await select("auto-dense")
    try check(
      model.mtpActive == true && model.samplingLocked, "Compatible Qwen did not prepare MTP")
    try await send(mtp: true, flash: false, greedy: true)
    try await checkInspector(greedy: true)
    for name in ["lfm-dense", "lfm-moe", "plain"] {
      try await select(name)
      try check(
        model.mtpEnabled && preferences.bool(forKey: "studio.mtpEnabled") && !model.samplingLocked,
        "Unsupported model must preserve ON intent and unlock sampling")
      try await send(mtp: false, flash: false, greedy: false)
      try await checkInspector(greedy: false)
    }
    try check(model.persistNow(), "Could not persist ordinary sampling")
    let persisted = try ConversationStore.load(from: file)!
    try check(
      persisted.temperature == 0.73 && persisted.topK == 37 && persisted.repetitionPenalty == 1.12,
      "Workspace persisted forced greedy sampling")
    try await select("auto-fail")
    try check(
      model.mtpError != nil && !model.mtpEnabled && !model.samplingLocked
        && preferences.bool(forKey: "studio.mtpEnabled"),
      "Failed MTP preparation must unlock without discarding saved ON")
    try await send(mtp: false, flash: false, greedy: false)
    try await select("auto-dense")
    try await send(mtp: true, flash: false, greedy: true)
    try await select("lfm-moe")
    try await checkInspector(greedy: false)
    // Source guards cover the OFF toggle binding; invoke its production action.
    model.setMTPEnabled(false)
    try check(
      !model.mtpEnabled && !preferences.bool(forKey: "studio.mtpEnabled"),
      "Unsupported MTP OFF was ignored")
    try await select("auto-dense")
    try await send(mtp: false, flash: false, greedy: false)
    try await select("dflash-good")
    model.setDFlash2Enabled(true)
    try await wait("DFlash preparation") { !model.dflashDownloading }
    try check(model.dflashActive == true && model.samplingLocked, "DFlash did not prepare")
    try await send(mtp: false, flash: true, greedy: true)
    try await checkInspector(greedy: true)
    try await select("lfm-dense")
    try check(
      model.dflash2Enabled && !model.samplingLocked, "Unsupported DFlash preference locked LFM")
    try await send(mtp: false, flash: false, greedy: false)
    try await select("dflash-good")
    try await send(mtp: false, flash: true, greedy: true)
    try await select("lfm-dense")
    try await checkInspector(greedy: false)
    model.setDFlash2Enabled(false)
    try check(
      !model.dflash2Enabled && !preferences.bool(forKey: "studio.dflash2Enabled"),
      "Unsupported DFlash OFF was ignored")
    try await select("dflash-good")
    try await send(mtp: false, flash: false, greedy: false)
    print(
      "Speculative sampling checks passed: \(checks) assertions, 180 capability/intent states, relaunch, Qwen/LFM dense/MoE, OFF/return, DFlash, persisted settings and real bridge requests"
    )
  }
}
