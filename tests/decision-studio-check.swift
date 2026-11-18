import AVFoundation
import AppKit
import Foundation

// Decision mode: request drafts, model discovery, the decide-bridge lifecycle (fake engine),
// audio normalization and the decide-serve launch. Never loads a model.
@main struct DecisionStudioCheck {
  @MainActor static func main() async throws {
    let arguments = CommandLine.arguments
    precondition(arguments.count == 2, "usage: decision-studio-check <fake-decision-engine.py>")
    let fake = URL(fileURLWithPath: arguments[1])
    let root = FileManager.default.temporaryDirectory.appending(path: "mlxl3-decision-check-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    var count = 0
    func check(_ value: @autoclosure () -> Bool, _ message: String, line: UInt = #line) {
      precondition(value(), "\(message) (line \(line))")
      count += 1
    }
    func wait(_ message: String, line: UInt = #line, _ predicate: @MainActor () -> Bool) async throws {
      for _ in 0..<500 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      preconditionFailure("timed out: \(message) (line \(line))")
    }
    func throwsMessage(_ body: () throws -> Void) -> String? {
      do { try body(); return nil } catch { return error.localizedDescription }
    }

    // Drafts: JSON shape, option order, validation messages.
    let example = DecisionDraft.example
    let text = String(decoding: try example.request(), as: UTF8.self)
    let object = try JSONSerialization.jsonObject(with: Data(text.utf8)) as! [String: Any]
    check(object["state"] as? String == example.state, "text state travels as a string")
    let questions = object["questions"] as! [String: Any]
    check(Set(questions.keys) == ["refund", "team", "urgency"], "every question is named")
    check((questions["team"] as! [String: Any])["type"] as? String == "choice", "choice type")
    let billing = text.range(of: "\"billing\"")!.lowerBound, shipping = text.range(of: "\"shipping\"")!.lowerBound
    let technical = text.range(of: "\"technical\"")!.lowerBound
    check(billing < shipping && shipping < technical, "options keep their listed order")
    check(text.range(of: "\"refund\"")!.lowerBound < text.range(of: "\"team\"")!.lowerBound
      && text.range(of: "\"team\"")!.lowerBound < text.range(of: "\"urgency\"")!.lowerBound, "questions keep their order")
    check((questions["urgency"] as! [String: Any])["criteria"] as? [String] == ["low", "medium", "high"], "score levels")
    check(object["media"] == nil, "no media key without media")

    var draft = example
    draft.stateIsJSON = true
    draft.state = "{\"a\": [1, 2]}"
    let jsonState = try JSONSerialization.jsonObject(with: try draft.request()) as! [String: Any]
    check(jsonState["state"] is [String: Any], "JSON state is an object")
    draft.state = "{not json"
    check(throwsMessage { _ = try draft.request() } != nil, "invalid JSON state refused")
    draft = example
    draft.questions[1].name = "refund"
    check(throwsMessage { _ = try draft.request() } != nil, "duplicate names refused")
    draft = example
    draft.questions.append(DecisionQuestion(name: "where", kind: .box, instructions: "Where is it?"))
    check(throwsMessage { _ = try draft.request() } != nil, "a box needs exactly one picture")
    draft.media = [DecisionMediaItem(kind: .image, path: "/tmp/x.png", label: "x")]
    let boxed = try JSONSerialization.jsonObject(with: try draft.request()) as! [String: Any]
    check((boxed["media"] as? [[String: String]])?.first?["type"] == "image", "media list carries the picture")
    draft = example
    draft.questions[2].options = [DecisionOption(name: "only")]
    check(throwsMessage { _ = try draft.request() } != nil, "a score needs 2 to 10 levels")
    draft = example
    draft.questions = [DecisionQuestion(name: "n", kind: .estimate, instructions: "How many?", low: "5", high: "1")]
    check(throwsMessage { _ = try draft.request() } != nil, "an estimate's range runs low to high")
    draft.questions[0].low = "0"; draft.questions[0].high = "1"; draft.questions[0].step = ""
    let estimate = (try JSONSerialization.jsonObject(with: try draft.request()) as! [String: Any])["questions"] as! [String: Any]
    check(((estimate["n"] as! [String: Any])["range"] as? [Double]) == [0, 1] && (estimate["n"] as! [String: Any])["step"] == nil,
      "an empty step reads on 128 values")
    draft = example
    draft.questions[0].instructions = "  "
    check(throwsMessage { _ = try draft.request() } != nil, "empty instructions refused")
    draft.questions[0].spokenPath = "/tmp/q.wav"
    let spoken = (try JSONSerialization.jsonObject(with: try draft.request()) as! [String: Any])["questions"] as! [String: Any]
    check(((spoken["refund"] as! [String: Any])["instructions"] as? [String: String]) == ["type": "audio", "path": "/tmp/q.wav"],
      "a spoken question sends its clip")
    draft = DecisionDraft()
    draft.questions = example.questions
    check(throwsMessage { _ = try draft.request() } != nil, "a request needs a state or media")

    // Discovery: only folders recording d1_attention with an EXL3 descriptor.
    func modelFolder(_ name: String, d1: Bool, type: String = "lfm3_vl") throws -> URL {
      let folder = root.appending(path: name)
      try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
      var config: [String: Any] = ["model_type": type]
      if d1 { config["d1_attention"] = "set" }
      try JSONSerialization.data(withJSONObject: config).write(to: folder.appending(path: "config.json"))
      try Data("{\"quant_method\":\"exl3\"}".utf8).write(to: folder.appending(path: "quantization_config.json"))
      return folder
    }
    let d1 = try modelFolder("d1-2B", d1: true)
    let omni = try modelFolder("d1-Omni-2B", d1: true, type: "lfm3_omni")
    let chat = try modelFolder("LFM3-VL-2B", d1: false)
    let broken = try modelFolder("d1-broken", d1: true)
    check(DecisionModelEntry.inspect(path: chat.path, custom: true) == nil, "a chat model is not a decision model")
    check(DecisionModelEntry.inspect(path: omni.path, custom: true)?.audio == true, "Omni takes audio")
    check(DecisionModelEntry.inspect(path: d1.path, custom: true)?.audio == false, "d1-2B takes no audio")
    let suite = "mlxl3.decision.check.\(UUID().uuidString)"
    let prefs = UserDefaults(suiteName: suite)!
    defer { prefs.removePersistentDomain(forName: suite) }
    let studio = DecisionStudio(preferences: prefs)
    studio.executable = { fake }
    check(!studio.addFolder(chat, registered: []), "adding a chat folder is refused")
    check(studio.error != nil, "the refusal says why")
    let registered = LocalModel(name: "registered-d1", path: d1.path, modelType: "lfm3_vl", format: "EXL3", bits: 8,
      sizeBytes: 1, modules: 142, addedAt: "", size: "1 B")
    let registeredChat = LocalModel(name: "chat", path: chat.path, modelType: "lfm3_vl", format: "EXL3", bits: 8,
      sizeBytes: 1, modules: 142, addedAt: "", size: "1 B")
    studio.refresh(registered: [registered, registeredChat])
    check(studio.models.map(\.name) == ["registered-d1"], "registered decision models are listed, chat models are not")
    check(studio.addFolder(omni, registered: [registered]), "an Omni folder is added")
    check(studio.models.count == 2 && studio.selectedPath == omni.path, "the added folder is selected")
    check(prefs.stringArray(forKey: "decision.folders") == [omni.path], "added folders persist")
    let reopened = DecisionStudio(preferences: prefs)
    reopened.refresh(registered: [])
    check(reopened.models.map(\.path) == [omni.path] && reopened.selectedPath == omni.path, "folders and choice survive a restart")

    // Bridge lifecycle against the fake engine.
    let launches = root.appending(path: "launches.jsonl")
    setenv("MLXL3_FAKE_DECISION_LOG", launches.path, 1)
    studio.draft = example
    studio.run()
    try await wait("first answer") { studio.outcome != nil || studio.error != nil }
    check(studio.error == nil, "first decision succeeds: \(studio.error ?? "")")
    check(studio.state == .ready && studio.capabilities?.audio == true, "the Omni bridge announces audio")
    let outcome = studio.outcome!
    check(outcome.order == ["refund", "team", "urgency"], "answers keep the question order")
    check((outcome.answers["refund"] as? [String: Any])?["noul"] as? Double == 0.875, "noul answer parsed")
    check(((outcome.answers["team"] as? [String: Any])?["order_seen"] as? [String]) == ["billing", "shipping", "technical"],
      "the engine received the options in order")
    check(outcome.inputTokens > 24 && outcome.elapsedMilliseconds == 12.5, "usage parsed")
    let firstLaunch = try String(contentsOf: launches, encoding: .utf8).split(separator: "\n")
    check(firstLaunch.count == 1 && firstLaunch[0].contains("decide-bridge") && firstLaunch[0].contains(omni.path), "one bridge launch")
    studio.run()
    try await wait("second answer") { !studio.running }
    let launchCount = try String(contentsOf: launches, encoding: .utf8).split(separator: "\n").count
    check(launchCount == 1, "the loaded bridge is reused")

    studio.draft.questions.append(DecisionQuestion(name: "fail", kind: .noul, instructions: "x"))
    studio.run()
    try await wait("error answer") { !studio.running }
    check(studio.error?.contains("unsupported") == true && studio.state == .ready, "an engine error is shown and the bridge stays up")

    studio.draft.questions[3].name = "crash"
    studio.run()
    try await wait("crash") { !studio.running }
    check(studio.error != nil, "a crashed bridge reports an error")
    try await wait("failed state") { if case .failed = studio.state { true } else { false } }
    check(!studio.active, "a crashed bridge is gone")
    studio.draft = example
    studio.run()
    try await wait("recovery") { !studio.running && studio.outcome != nil && studio.error == nil }
    check(studio.state == .ready, "the next decision relaunches the bridge")

    // A model that cannot load fails with its stderr; switching models unloads the old one.
    check(studio.addFolder(broken, registered: []), "a broken folder is still listed")
    check(!studio.active, "changing the model unloads the bridge")
    studio.run()
    try await wait("broken load") { !studio.running }
    check(studio.error?.contains("safetensors") == true, "load failure shows the engine's message: \(studio.error ?? "")")

    // Audio needs an Omni model: d1-2B refuses before sending.
    check(studio.addFolder(d1, registered: []), "back to d1-2B")
    studio.draft.media = [DecisionMediaItem(kind: .audio, path: "/tmp/a.wav", label: "a")]
    studio.run()
    try await wait("audio refusal") { !studio.running }
    check(studio.error?.contains("Omni") == true, "audio is refused on d1-2B: \(studio.error ?? "")")
    studio.unload()
    check(!studio.active && studio.state == .idle, "unload stops the bridge")

    // Audio import: any Core Audio file becomes 16-bit PCM WAV at its own rate; 16-bit WAV passes.
    let aiff = root.appending(path: "tone.aiff")
    do {
      let format = AVAudioFormat(standardFormatWithSampleRate: 22_050, channels: 2)!
      let file = try AVAudioFile(forWriting: aiff, settings: format.settings)
      let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 22_050)!
      buffer.frameLength = 22_050
      for c in 0..<2 { for i in 0..<22_050 { buffer.floatChannelData![c][i] = sin(Float(i) * 0.05) * 0.5 } }
      try file.write(from: buffer)
    }
    let wav = try DecisionMediaImport.normalizedAudio(aiff)
    check(wav.pathExtension == "wav" && wav != aiff, "AIFF is converted")
    let header = try Data(contentsOf: wav)
    check(header.prefix(4) == Data("RIFF".utf8) && header[8..<12] == Data("WAVE".utf8), "a RIFF/WAVE file")
    let converted = try AVAudioFile(forReading: wav)
    check(converted.fileFormat.sampleRate == 22_050 && converted.fileFormat.channelCount == 2
      && (converted.fileFormat.settings[AVLinearPCMBitDepthKey] as? Int) == 16, "rate and channels kept, 16-bit PCM")
    check(abs(Int(converted.length) - 22_050) <= 1, "every frame kept")
    let passed = try DecisionMediaImport.normalizedAudio(wav)
    check(passed == wav, "a 16-bit WAV passes through")
    try? FileManager.default.removeItem(at: wav)
    let image = try DecisionMediaImport.prepare(URL(fileURLWithPath: "/tmp/missing.png"))
    check(image.kind == .image, "PNG pictures are passed by path")

    // Server: decide-serve with the key on stdin, ready port, stop.
    studio.draft = example
    studio.startServer(port: 18_731, network: false)
    try await wait("server ready") { studio.server.running }
    check(studio.server.port == 18_731 && studio.server.apiKey?.isEmpty == false, "server port and key")
    let serve = try String(contentsOf: launches, encoding: .utf8).split(separator: "\n").last!
    check(serve.contains("decide-serve") && serve.contains(d1.path) && serve.contains("--api-key-stdin")
      && serve.contains("127.0.0.1") && !serve.contains("--context-length"), "decide-serve arguments")
    let body = try studio.serverBody()
    let parsedBody = try JSONSerialization.jsonObject(with: Data(body.utf8))
    check(parsedBody is [String: Any], "server body is JSON")
    studio.server.stop()
    try await wait("server stopped") { !studio.server.active }
    unsetenv("MLXL3_FAKE_DECISION_LOG")
    print("decision studio check passed: \(count) checks")
  }
}
