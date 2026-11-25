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
    // APP-F2: the DSpark counters of a complete event are decoded and summarized (accepted drafts plus
    // the target token per block); missing or inconsistent counters give no summary.
    func decodedStats(_ extra: String) throws -> GenerationStats? {
      try JSONDecoder().decode(BridgeEvent.self, from: Data(
        (#"{"type":"complete","request_id":"r","stats":{"ttft_seconds":1,"prefill_tps":1,"decode_tps":1,"prompt_tokens":3,"generated_tokens":104"#
          + extra + "}}").utf8)).stats
    }
    let dspark = try decodedStats(#","dspark_proposed_tokens":120,"dspark_accepted_tokens":64,"dspark_blocks":40"#)?.dsparkAcceptance
    try check(dspark?.accepted == 64 && dspark?.proposed == 120
                && dspark.map { abs($0.tokensPerBlock - 2.6) < 1e-12 } == true,
              "DSpark counters not decoded: \(String(describing: dspark))")
    for extra in ["", #","dspark_proposed_tokens":0,"dspark_accepted_tokens":0,"dspark_blocks":0"#,
                  #","dspark_proposed_tokens":10,"dspark_accepted_tokens":11,"dspark_blocks":4"#,
                  #","dspark_proposed_tokens":10,"dspark_accepted_tokens":4"#,
                  #","mtp_proposed_tokens":10,"mtp_accepted_tokens":4,"mtp_blocks":3"#] {
      try check(try decodedStats(extra)?.dsparkAcceptance == nil, "DSpark summary from \(extra)")
    }
    // APP-F2b: the engine announces DSpark and the request carries the user's choice
    let dsparkReady = try JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"ready","dspark_supported":true}"#.utf8))
    try check(dsparkReady.dsparkSupported == true, "dspark_supported not decoded")
    try check(try JSONDecoder().decode(BridgeEvent.self, from: Data(#"{"type":"ready"}"#.utf8)).dsparkSupported == nil,
              "older engines must leave dspark_supported unknown")
    // FEAT-2026-11-25-INK2M-DFLASH2-APP: the engine's default (a DFlash2 drafter is installed) drives the
    // switch until the user chooses, and decides when the drafter is downloaded
    let dsparkDefaultReady = try JSONDecoder().decode(
      BridgeEvent.self, from: Data(#"{"type":"ready","dspark_supported":true,"dspark_default":true}"#.utf8))
    try check(dsparkDefaultReady.dsparkDefault == true, "dspark_default not decoded")
    try check(dsparkReady.dsparkDefault == nil, "older engines must leave dspark_default unknown")
    for (preference, engineDefault, expected) in [
      (Bool?.none, Bool?.none, false), (nil, false, false), (nil, true, true),
      (true, nil, true), (true, false, true), (true, true, true),
      (false, nil, false), (false, false, false), (false, true, false),
    ] {
      try check(StudioModel.dsparkSwitch(preference: preference, engineDefault: engineDefault) == expected,
                "switch for preference \(String(describing: preference)), default \(String(describing: engineDefault))")
    }
    let wanted = { (type: String?, supported: Bool?, engineDefault: Bool?, preference: Bool?) in
      StudioModel.dsparkDrafterWanted(modelType: type, supported: supported, engineDefault: engineDefault,
                                      preference: preference)
    }
    try check(wanted("inkling2_loop", true, false, nil), "untouched switch: the drafter is downloaded")
    try check(wanted("inkling2_loop", true, false, true), "switch on: the drafter is downloaded")
    try check(!wanted("inkling2_loop", true, false, false), "switch off: nothing is downloaded")
    try check(!wanted("inkling2_loop", true, true, nil), "drafter installed: nothing to download")
    try check(!wanted("inkling2_loop", true, nil, nil), "older engine: it could not use the drafter")
    try check(!wanted("inkling2_loop", nil, false, nil) && !wanted("inkling2_loop", false, false, nil),
              "no drafter support: nothing is downloaded")
    try check(!wanted("qwen3_5", true, false, nil) && !wanted(nil, true, false, nil),
              "only Inkling 2 Mini has this drafter")
    for enabled in [true, false] {
      var request = GenerationRequest(requestID: "r", conversationID: "c", messages: [], maxTokens: 8,
                                      temperature: 1, topK: 0, repetitionPenalty: 1)
      request.dspark = enabled
      let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(request)) as? [String: Any]
      try check(encoded?["dspark"] as? Bool == enabled, "dspark not encoded as \(enabled)")
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
    // Manual Tune scenarios; the automatic Tune is covered by tests/mtp-autotune-check.swift.
    preferences.set(false, forKey: StudioModel.mtpAutoTunePreferenceKey)
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
    try check(!relaunched.mtpEnabled && !relaunched.samplingLocked,
      "Legacy MTP intent activated before resolving a model")
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
    let types = ["lfm2", "lfm2_moe", "unknown", "qwen3_5", "qwen3_5_moe", "qwen4_exp"]
    let capabilities: [Bool?] = [nil, false, true]
    for type in types {
      for mtp in [false, true] {
        for flash in [false, true] {
          for mtpCapability in capabilities {
            for flashCapability in capabilities {
             for samplingCapability in capabilities {
             for flashSamplingCapability in capabilities {
              let model = StudioModel(
                conversationFileURL: file, isPreview: true, preferences: preferences)
              model.models = [local("qwen3.6-35b-a3b", type)]
              model.selectedModelName = "qwen3.6-35b-a3b"
              model.topP = 0.9
              model.presencePenalty = 0.4
              var ready: [String: Any] = ["type": "ready"]
              if let mtpCapability { ready["mtp_supported"] = mtpCapability }
              if let flashCapability { ready["dflash_supported"] = flashCapability }
              if let samplingCapability { ready["mtp_sampling_supported"] = samplingCapability }
              if let flashSamplingCapability { ready["dflash_sampling_supported"] = flashSamplingCapability }
              let data = try JSONSerialization.data(withJSONObject: ready)
              try model.samplingFixtureEvent(String(decoding: data, as: UTF8.self))
              model.samplingFixtureIntent(mtp: mtp, dflash: flash)
              let wantsMTP = mtp && (mtpCapability ?? ["qwen3_5", "qwen3_5_moe", "qwen4_exp"].contains(type))
              let wantsFlash = flash && (flashCapability ?? (type == "qwen3_5_moe"))
              // Only an engine announcing exact speculative sampling unlocks MTP or DFlash2.
              let locked = (wantsMTP && samplingCapability != true)
                || (wantsFlash && flashSamplingCapability != true)
              try check(
                model.mtpApplicable == wantsMTP && model.dflash2Applicable == wantsFlash,
                "Intent/capability mismatch for \(type), \(String(describing: mtpCapability)), \(String(describing: flashCapability))"
              )
              try check(
                model.samplingLocked == locked && model.samplingTemperature == (locked ? 0 : 0.73)
                  && model.samplingTopK == (locked ? 1 : 37)
                  && model.samplingRepetitionPenalty == (locked ? 1 : 1.12)
                  && model.samplingTopP == (locked ? 1 : 0.9)
                  && model.samplingPresencePenalty == (locked ? 0 : 0.4),
                "Displayed/request sampling disagrees with applicability")
              try check(
                model.temperature == 0.73 && model.topK == 37 && model.repetitionPenalty == 1.12,
                "Capability event changed saved sampling")
              try check(
                model.mtpActive == nil && model.dflashActive == nil,
                "Intent was incorrectly reported as observed engine activity")
              try check(
                model.tuneSampling == MLXL3Bridge.TuneSampling(
                  temperature: model.samplingTemperature, topK: model.samplingTopK, topP: model.samplingTopP,
                  repetitionPenalty: model.samplingRepetitionPenalty, presencePenalty: model.samplingPresencePenalty),
                "Tune would measure with other settings than chat")
             }
             }
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
    // Layout-only source formatting must not invalidate these wiring guards.
    // These remain structural smoke checks, not a Swift semantic parser.
    func containsWiring(_ source: String, _ required: String) -> Bool {
      source.filter { !$0.isWhitespace }.contains(required.filter { !$0.isWhitespace })
    }
    let binding = "get: { studio.samplingTopK }, set: { studio.topK = $0 }"
    try check(
      containsWiring("get: {\n studio.samplingTopK\n}, set: {\n studio.topK = $0\n}", binding),
      "Wiring guard rejects whitespace-only formatting")
    for mutation in [
      "get: { studio.topK }, set: { studio.topK = $0 }",
      "get: { studio.samplingTopK }, set: { studio.temperature = $0 }",
      "get: { studio.samplingTopK }",
    ] {
      try check(!containsWiring(mutation, binding), "Wiring guard accepted changed getter/setter")
    }
    let inspectorSource = try String(contentsOfFile: CommandLine.arguments[2], encoding: .utf8)
    for required in [
      "value: Binding(get: { studio.samplingTemperature }, set: { studio.temperature = $0 })",
      "get: { Double(studio.samplingTopK) }",
      "value: Binding(get: { studio.samplingRepetitionPenalty }, set: { studio.repetitionPenalty = $0 })",
      ".disabled(studio.samplingLocked)",
      "get: { studio.samplingTopP }",
      "set: { studio.topP = min(1, max(0.05, $0)) }",
      "get: { studio.samplingPresencePenalty }",
      "set: { studio.presencePenalty = $0 }",
      "if !studio.recommendedSamplingPresets.isEmpty {",
      "RecommendedSamplingMenu()",
      "Button(action: { studio.applySamplingPreset(preset) })",
      ".disabled(studio.isGenerating || studio.samplingLocked)",
      "let mode = studio.dflash2Applicable && studio.dflashSamplingSupported != true ? \"DFlash2\" : \"MTP\"",
      "if studio.mtpAvailable || studio.mtpRememberedEnabled || studio.mtpDownloading {",
      "if studio.dflash2Available || studio.dflash2Enabled || studio.dflashDownloading {",
      "DFlashSettingsContent()",
      "MTPSettingsContent().id(\"mtp-tuning\")",
    ] {
      try check(
        containsWiring(inspectorSource, required), "Missing inspector production wiring: " + required)
    }
    // The inspector card and the toolbar popover share DFlashSettingsContent.
    let dflashSource = try String(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])
      .deletingLastPathComponent().appendingPathComponent("DFlashSettingsContent.swift"), encoding: .utf8)
    for required in [
      "get: { studio.dflash2Enabled || studio.dflashDownloading }",
      "set: { studio.setDFlash2Enabled($0) }",
      "Preference saved; unavailable for this model",
      "MTPTuningControls(dflash: true).disabled(!studio.dflash2Available)",
      ".disabled(studio.isGenerating || (!(studio.dflash2Enabled || studio.dflashDownloading)",
      "DFlashSettingsContent()",
    ] {
      try check(containsWiring(dflashSource, required), "Missing shared DFlash2 production wiring: " + required)
    }
    let headerSource = try String(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])
      .deletingLastPathComponent().appendingPathComponent("StudioView.swift"), encoding: .utf8)
    try check(containsWiring(headerSource, "MTPToolbarControl()") && containsWiring(headerSource, "DFlashToolbarControl()"),
      "Header must offer the MTP and DFlash2 settings popovers")
    try check(!containsWiring(headerSource, "MTPTuningToolbarControl"),
      "Header must not keep the bare Tune button")
    let mtpSource = try String(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])
      .deletingLastPathComponent().appendingPathComponent("MTPSettingsContent.swift"), encoding: .utf8)
    for required in [
      "get: { studio.mtpRememberedEnabled }, set: { studio.setMTPEnabled($0) }",
      ".disabled(studio.isGenerating || !studio.engineState.isReady)",
      "MTPTuningControls()",
    ] {
      try check(containsWiring(mtpSource, required), "Missing shared MTP production wiring: " + required)
    }
    try check(!containsWiring(mtpSource, "Button(action: studio.tuneMTP)"),
      "MTP settings must not offer a duplicate tuning action")
    let tuningSource = try String(contentsOf: URL(fileURLWithPath: CommandLine.arguments[2])
      .deletingLastPathComponent().appendingPathComponent("MTPTuningControls.swift"), encoding: .utf8)
    for required in ["Button(action: tune)", ".disabled(!canTune)", "studio.tuneMTP()", "studio.tuneDFlash()"] {
      try check(containsWiring(tuningSource, required), "Missing tuning action or guard: " + required)
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
          && request["repetition_penalty"] as? Double == (greedy ? 1 : 1.12)
          && request["top_p"] as? Double == (greedy ? 1 : model.topP)
          && request["presence_penalty"] as? Double == (greedy ? 0 : model.presencePenalty),
        "Production bridge sampling mismatch")
      try check(
        model.temperature == 0.73 && model.topK == 37 && model.repetitionPenalty == 1.12,
        "Generation changed saved sampling")
    }
    for name in ["auto-dense", "lfm-dense", "lfm-moe", "plain", "auto-fail"] {
      let target = model.models.first { $0.name == name }!
      ModelMTPPreference.save(true, modelPath: target.path, preferences: preferences)
    }
    try await select("auto-dense")
    try check(
      model.mtpActive == true && model.samplingLocked, "Compatible Qwen did not prepare MTP")
    try await send(mtp: true, flash: false, greedy: true)
    try await checkInspector(greedy: true)
    for name in ["lfm-dense", "lfm-moe", "plain"] {
      try await select(name)
      try check(
        !model.mtpEnabled && model.mtpRememberedEnabled && !model.samplingLocked,
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
        && model.mtpRememberedEnabled,
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
    try check(model.mtpRememberedEnabled, "Another model's OFF erased the dense model's ON")
    try await send(mtp: true, flash: false, greedy: true)
    model.setMTPEnabled(false)
    try await wait("dense MTP OFF") { model.mtpActive == false }
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
    // Applying ordinary profiles must configure both engines OFF before a real send.
    let ordinary = GenerationProfile(name: "Ordinary", model: nil, systemPrompt: "", temperature: 0.73,
        topK: 37, repetitionPenalty: 1.12, contextLength: 0, responseSchema: "")
    try await select("auto-dense")
    model.setMTPEnabled(true)
    try await wait("profile MTP ready") { model.mtpActive == true }
    model.applyProfile(ordinary)
    model.draft = "profile check"
    try check(!model.canSend, "Profile sent before MTP configuration acknowledgement")
    try await wait("profile MTP off") { model.mtpActive == false && model.canSend }
    try await send(mtp: false, flash: false, greedy: false)
    try await select("dflash-good")
    model.setDFlash2Enabled(true)
    try await wait("profile DFlash ready") { model.dflashActive == true && !model.dflashDownloading }
    model.applyProfile(ordinary)
    model.draft = "profile check"
    try check(!model.canSend, "Profile sent before DFlash configuration acknowledgement")
    try await wait("profile DFlash off") { model.dflashActive == false && model.canSend }
    try await send(mtp: false, flash: false, greedy: false)

    // An engine announcing exact speculative sampling keeps MTP and the user's sampling.
    setenv("MLXL3_FIXTURE_MTP_SAMPLING", "1", 1)
    defer { unsetenv("MLXL3_FIXTURE_MTP_SAMPLING") }
    model.topP = 0.85
    model.presencePenalty = 0.5
    try await select("lfm-dense")
    try await select("auto-dense")
    model.setMTPEnabled(true)
    try await wait("sampled MTP ready") { model.mtpActive == true }
    try check(
      model.mtpSamplingSupported == true && model.mtpApplicable && !model.samplingLocked,
      "Sampling-capable engine kept MTP greedy")
    try await send(mtp: true, flash: false, greedy: false)
    try await checkInspector(greedy: false)
    // A sampled MTP Tune reports rows whose tokens differ from the baseline by design.
    try check(model.canTuneMTP, "Sampled MTP cannot be tuned")
    model.tuneMTP()
    try await wait("sampled MTP Tune") { !model.isTuningMTP }
    try check(
      model.mtpError == nil && model.mtpDepth == 2 && model.mtpTuneRows.count == 4
        && model.mtpTuneRows.allSatisfy { $0.sampled == true },
      "Sampled MTP Tune was rejected: \(model.mtpError ?? "no error"), depth \(model.mtpDepth)")
    // An engine announcing sampled DFlash2 keeps the user's sampling, for chat and for Tune.
    setenv("MLXL3_FIXTURE_DFLASH_SAMPLING", "1", 1)
    defer { unsetenv("MLXL3_FIXTURE_DFLASH_SAMPLING") }
    try await select("lfm-dense")
    try await select("dflash-good")
    model.setDFlash2Enabled(true)
    try await wait("sampled DFlash ready") { model.dflashActive == true && !model.dflashDownloading }
    try check(
      model.dflashSamplingSupported == true && model.dflash2Applicable && !model.samplingLocked,
      "Sampling-capable engine kept DFlash2 greedy")
    try await send(mtp: false, flash: true, greedy: false)
    try await checkInspector(greedy: false)
    try check(model.canTuneDFlash, "Sampled DFlash2 cannot be tuned")
    model.tuneDFlash()
    try await wait("sampled DFlash Tune") { !model.isTuningDFlash }
    try check(model.dflashDownloadError == nil && model.dflashMode == 2,
      "Sampled DFlash2 Tune was rejected: \(model.dflashDownloadError ?? "no error"), mode \(model.dflashMode)")
    let tuneRequest = try String(contentsOf: requestsURL, encoding: .utf8).split(separator: "\n")
      .map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
      .compactMap { $0["request"] as? [String: Any] }.last { $0["type"] as? String == "tune_dflash" }!
    try check(
      tuneRequest["temperature"] as? Double == 0.73 && tuneRequest["top_k"] as? Int == 37
        && tuneRequest["repetition_penalty"] as? Double == 1.12
        && tuneRequest["top_p"] as? Double == model.topP
        && tuneRequest["presence_penalty"] as? Double == model.presencePenalty,
      "Tune DFlash2 did not measure with the chat sampling settings")
    try check(
      model.dflashTuneRows.count == 4 && model.dflashTuneRows.allSatisfy { $0.sampled == true }
        && model.dflashTuneRows[1].nearTies == [MTPNearTie(prompt: 0, position: 12, gap: 0.04)],
      "Sampled rows or near ties were not decoded")
    model.setDFlash2Enabled(false)
    model.draft = "after sampled DFlash"
    try await wait("sampled DFlash OFF") { model.dflashActive == false && model.canSend }
    try check(model.persistNow(), "Could not persist top-p/presence")
    let sampledPersisted = try ConversationStore.load(from: file)!
    try check(
      sampledPersisted.topP == 0.85 && sampledPersisted.presencePenalty == 0.5,
      "Workspace lost top-p/presence")
    let restored = StudioModel(conversationFileURL: file, isPreview: true, preferences: preferences)
    try check(restored.topP == 0.85 && restored.presencePenalty == 0.5, "Relaunch lost top-p/presence")
    // Snapshots written before top-p/presence still decode with neutral values.
    let legacy = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as! [String: Any]
    var stripped = legacy
    stripped.removeValue(forKey: "topP")
    stripped.removeValue(forKey: "presencePenalty")
    try JSONSerialization.data(withJSONObject: stripped).write(to: file)
    let legacyRelaunch = StudioModel(conversationFileURL: file, isPreview: true, preferences: preferences)
    try check(
      legacyRelaunch.topP == 1 && legacyRelaunch.presencePenalty == 0
        && legacyRelaunch.temperature == 0.73,
      "Legacy snapshot did not decode with neutral top-p/presence")
    // Recommended settings: exactly the catalog (with its alternative
    // quantizations) and the values published on each official model card.
    // Decision models (d1) answer in one pass and never sample, so they carry no preset.
    let decisionCatalog = Set(ModelLibrary.decisionModels.map(\.id))
    let catalog = Set(ModelLibrary.recommendedModels.flatMap { [$0.id] + $0.choices.map(\.id) })
      .subtracting(decisionCatalog)
    try check(!decisionCatalog.isEmpty && decisionCatalog.isDisjoint(with: SamplingPresets.byRepository.keys),
      "Decision models must not get sampling presets")
    try check(catalog == Set(SamplingPresets.byRepository.keys), "Presets do not cover exactly the catalog")
    func values(_ preset: SamplingPreset) -> [Double] {
      [preset.temperature, Double(preset.topK), preset.topP, preset.repetitionPenalty, preset.presencePenalty]
    }
    // Independent oracle: [temperature, top_k, top_p, repetition, presence].
    let qwen36: [[Double]] = [[1.0, 20, 0.95, 1, 1.5], [0.6, 20, 0.95, 1, 0], [0.7, 20, 0.8, 1, 1.5]]
    let qwen38: [[Double]] = [[1.0, 20, 0.95, 1, 0], [0.7, 20, 0.8, 1, 1.5]]
    let lfm3: [[Double]] = [[0.1, 50, 1, 1.1, 0], [1.0, 20, 0.95, 1, 0]]
    let oracle: [String: [[Double]]] = [
      "0xzknw/LFM3-1.7B-Heretic-NX-EXL3-8bpw": [[0.1, 50, 1, 1.1, 0]],
      "0xzknw/LFM3-1.7B-EXL3-8bpw": lfm3, "0xzknw/LFM3-1.7B-EXL3-4.49bpw": lfm3,
      "0xzknw/LFM3-1.7B-EXL3-4bpw": lfm3,
      "0xzknw/LFM3-VL-2B-EXL3-8bpw": [[0.1, 50, 1, 1.1, 0]],
      "0xzknw/LFM3-VL-2B-EXL3-4.49bpw": [[0.1, 50, 1, 1.1, 0]],
      "0xzknw/LFM3-VL-2B-EXL3-4bpw": [[0.1, 50, 1, 1.1, 0]],
      "0xzknw/LFM2.5-24B-A2B-EXL3-4bpw": [[0.1, 50, 1, 1.05, 0]],
      "UnstableLlama/Qwen3.6-35B-A3B-exl3-2.49bpw": qwen36,
      "turboderp/Qwen3.8-27B-exl3": qwen38, "turboderp/Qwen3.8-Flash-Next-exl3": qwen38,
      "0xzknw/Inkling-2-Mini-EXL3-3.33bpw": [[1.0, 0, 1, 1, 0]],
    ]
    for (repository, presets) in SamplingPresets.byRepository {
      try check(presets.map(values) == oracle[repository], "Preset values differ for " + repository)
      try check(
        presets.allSatisfy(\.valid) && Set(presets.map(\.id)).count == presets.count
          && presets.allSatisfy { !$0.french.isEmpty && !$0.english.isEmpty },
        "Invalid preset table for " + repository)
      let stem = String(repository.split(separator: "/").last!)
      // Engine download names: <repo name>-<variant folder or 8-char id>.
      for suffix in ["-0123abcd", "-2.00bpw", "--8bpw", ""] {
        try check(
          SamplingPresets.repository(name: stem + suffix, path: "/m/" + stem + suffix) == repository
            && SamplingPresets.repository(name: "renamed", path: "/m/" + (stem + suffix).lowercased()) == repository,
          "Catalog download not matched: " + stem + suffix)
      }
      for other in [stem + "x-1", "x" + stem + "-1", String(stem.dropLast()), stem + "_1"] {
        try check(SamplingPresets.presets(name: other, path: "/m/" + other).isEmpty, "Foreign model matched " + other)
      }
    }
    // Installs named by older multi-variant catalog repos (`<repo>--<variant>`).
    for (legacy, repository) in [
      ("LFM3-1.7B-EXL3--8bpw", "0xzknw/LFM3-1.7B-EXL3-8bpw"),
      ("LFM3-1.7B-EXL3--4.49bpw", "0xzknw/LFM3-1.7B-EXL3-4.49bpw"),
      ("LFM3-1.7B-Heretic-NX-EXL3--8bpw", "0xzknw/LFM3-1.7B-Heretic-NX-EXL3-8bpw"),
      ("LFM3-VL-2B-EXL3--4bpw", "0xzknw/LFM3-VL-2B-EXL3-4bpw"),
    ] {
      try check(SamplingPresets.repository(name: legacy, path: "/m/" + legacy) == repository,
        "Legacy catalog install not matched: " + legacy)
    }
    for other in ["plain", "qwen3.6-35b-a3b", "Qwen3.8-27B-1", "LFM3-1.7B-EXL3-4.4-1", "my-LFM3-1.7B-EXL3-8bpw-1",
                  "LFM3-1.7B-EXL3--16bpw", "LFM3-1.7B-EXL3"] {
      try check(SamplingPresets.presets(name: other, path: "/m/" + other).isEmpty, "Foreign model matched " + other)
    }
    let presetModel = StudioModel(conversationFileURL: file, isPreview: true, preferences: preferences)
    presetModel.models = [local("Qwen3.8-27B-exl3-2.00bpw", "qwen3_5"), local("plain", "qwen3_5")]
    presetModel.selectedModelName = "plain"
    try check(presetModel.recommendedSamplingPresets.isEmpty, "Non-catalog model offered presets")
    presetModel.selectedModelName = "Qwen3.8-27B-exl3-2.00bpw"
    let offered = presetModel.recommendedSamplingPresets
    try check(offered.map(values) == qwen38, "Selected catalog model offered the wrong presets")
    for preset in offered {
      presetModel.applySamplingPreset(preset)
      try check(
        [presetModel.temperature, Double(presetModel.topK), presetModel.topP, presetModel.repetitionPenalty,
         presetModel.presencePenalty] == values(preset)
          && presetModel.isSamplingPresetActive(preset)
          && offered.filter(presetModel.isSamplingPresetActive) == [preset],
        "Preset was not applied exactly: " + preset.id)
    }
    let before = values(offered[1])
    presetModel.applySamplingPreset(SamplingPresets.byRepository["0xzknw/LFM2.5-24B-A2B-EXL3-4bpw"]![0])
    try check(
      [presetModel.temperature, Double(presetModel.topK), presetModel.topP, presetModel.repetitionPenalty,
       presetModel.presencePenalty] == before,
      "A preset of another model was applied")
    presetModel.topK = 21
    try check(offered.allSatisfy { !presetModel.isSamplingPresetActive($0) }, "Edited settings still marked as a preset")
    print(
      "Speculative sampling checks passed: \(checks) assertions, 1944 capability/intent states, sampled MTP, relaunch, Qwen/LFM dense/MoE, OFF/return, DFlash, persisted settings and real bridge requests"
    )
  }
}
