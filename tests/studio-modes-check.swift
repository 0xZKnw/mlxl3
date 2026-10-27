import AppKit
import Foundation
import SwiftUI

extension StudioModel {
  func modesRegistryForCheck() throws -> URL { try prepareAgentRegistry() }
  var modesBridgeRunning: Bool { bridge.isRunning }

  static func checkAgentGuards(root: URL, prefs: UserDefaults, model: LocalModel) async throws
    -> Int
  {
    var count = 0
    func check(_ value: @autoclosure () -> Bool, _ message: String) {
      precondition(value(), message)
      count += 1
    }
    func wait(_ predicate: @MainActor () -> Bool) async throws {
      for _ in 0..<400 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      preconditionFailure("Agent guard timed out")
    }
    func launches() throws -> [[String: Any]] {
      try String(contentsOf: root.appendingPathComponent("mode-launches.jsonl"), encoding: .utf8)
        .split(separator: "\n").map {
          try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any]
        }
    }
    func readyAgent(_ name: String) async throws -> StudioModel {
      let fixture = StudioModel(
        conversationFileURL: root.appendingPathComponent(name + ".json"), preferences: prefs)
      fixture.models = [model]
      fixture.selectModel(model.name)
      try await wait { fixture.engineState.isReady }
      fixture.selectMode(.agent)
      try await wait {
        fixture.engineState.isReady && fixture.mcpToolCount == 1 && !fixture.mcpUpdating
      }
      return fixture
    }
    let cooperative = try await readyAgent("cooperative")
    cooperative.draft = "hang-cooperative"
    cooperative.send()
    try await wait {
      cooperative.currentConversation?.messages.last?.parts.last?.text.contains("32 tokens") == true
    }
    let pending = cooperative.currentConversation!.messages.last!
    cooperative.stopGeneration()
    try await wait { !cooperative.isGenerating }
    check(
      cooperative.engineState.isReady && pending.error != nil,
      "Cooperative Stop preserves the loaded model and partial response")
    cooperative.draft = "Resume after Stop"
    cooperative.send()
    try await wait { !cooperative.isGenerating }
    check(
      cooperative.currentConversation!.messages.last!.error == nil,
      "A fresh request succeeds after cancellation")
    cooperative.stopServices()
    await cooperative.bridge.stopAndWait()

    let ignored = try await readyAgent("ignored")
    ignored.draft = "hang-ignore"
    ignored.send()
    try await wait { ignored.currentConversation?.messages.last?.thinking.isEmpty == false }
    let ignoredMessage = ignored.currentConversation!.messages.last!
    let ignoredPID = (try launches().last!["pid"] as! NSNumber).int32Value
    let stopStarted = ContinuousClock.now
    ignored.stopGeneration()
    try await wait { !ignored.isGenerating }
    check(
      ignoredMessage.error != nil && !ignored.bridge.isRunning,
      "Stop falls back when SIGUSR1 and SIGTERM are ignored")
    ignored.selectModel(model.name)
    try await wait { ignored.engineState.isReady && !ignored.mcpUpdating }
    check(ContinuousClock.now - stopStarted < .seconds(7), "Ignored Stop and reload are bounded")
    check(
      kill(ignoredPID, 0) == -1 && errno == ESRCH,
      "Reload joins the previous engine before loading another model")
    check(ignored.engineState.isReady, "Model can be loaded again after forced shutdown")
    ignored.stopServices()
    await ignored.bridge.stopAndWait()

    for deadline in [120, 600] {
      let guarded = try await readyAgent("deadline-\(deadline)")
      guarded.draft = "hang-cooperative"
      guarded.send()
      try await wait { guarded.currentConversation?.messages.last?.thinking.isEmpty == false }
      let start = guarded.generationStartedAt!
      guarded.generationLastActivity = deadline == 120 ? start : start + .seconds(599)
      let last = guarded.generationLastActivity
      let stale = try JSONDecoder().decode(
        BridgeEvent.self,
        from: Data(#"{"type":"generation_status","request_id":"stale","text":"still busy"}"#.utf8))
      guarded.handle(stale)
      check(guarded.generationLastActivity == last, "Stale requests cannot keep the watchdog alive")
      guarded.checkAgentGenerationDeadline(at: start + .seconds(deadline - 1))
      check(guarded.cancellationFallback == nil, "Deadline is not triggered early")
      guarded.checkAgentGenerationDeadline(at: start + .seconds(deadline))
      try await wait { !guarded.isGenerating }
      check(
        guarded.currentConversation!.messages.last!.error?.contains(
          deadline == 120 ? "2 minutes" : "10 minutes") == true,
        "Deadline stops the real request and explains why")
      check(
        guarded.generationWatchdog == nil && guarded.cancellationFallback == nil,
        "Completion cleans up every watchdog")
      guarded.stopServices()
      await guarded.bridge.stopAndWait()
    }
    let limited = try await readyAgent("budget")
    limited.draft = "budget-limit"
    limited.send()
    try await wait { !limited.isGenerating }
    check(
      limited.currentConversation!.messages.last!.error?.contains("4 096") == true,
      "Budget exhaustion keeps the partial output and reports truncation")
    let limitedRequest =
      try JSONSerialization.jsonObject(
        with: Data(limited.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
    check(
      limitedRequest["max_tokens"] as? Int == 4096, "Agent sends a finite production token budget")
    limited.stopServices()
    await limited.bridge.stopAndWait()
    return count
  }
}

@main struct StudioModesCheck {
  @MainActor static func main() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-modes-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
    setenv("MLXL3_HOME", root.path, 1)
    let suite = "mlxl3.modes.\(UUID())"
    let history = root.appendingPathComponent("history.json")
    let prefs = UserDefaults(suiteName: suite)!
    defer { prefs.removePersistentDomain(forName: suite) }
    let studio = StudioModel(conversationFileURL: history, preferences: prefs)
    defer { studio.stopServices() }
    var count = 0
    func check(_ value: @autoclosure () -> Bool, _ message: String) {
      precondition(value(), message)
      count += 1
    }
    func wait(_ predicate: @MainActor () -> Bool) async throws {
      for _ in 0..<400 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      preconditionFailure("Mode fixture timed out")
    }
    let model = LocalModel(
      name: "modes", path: "/tmp/fixture-model", modelType: "fixture", format: "EXL3", bits: 2.5,
      sizeBytes: 100, modules: 1, addedAt: "", size: "100 B")
    studio.models = [model]
    studio.selectModel(model.name)
    try await wait { studio.engineState.isReady }
    let chat = studio.selectedConversationID!
    check(
      studio.selectedMode == .chat && studio.currentConversation?.agent == false,
      "Old conversation defaults to Chat")
    studio.selectMode(.agent)
    try await wait { studio.engineState.isReady && studio.mcpToolCount == 1 && !studio.mcpUpdating }
    let agent = studio.selectedConversationID!
    let workspace = studio.agentWorkspaceURL!
    check(
      agent != chat && studio.currentConversation?.agent == true,
      "Agent gets a separate conversation")
    let registry = try studio.modesRegistryForCheck()
    let config =
      try JSONSerialization.jsonObject(
        with: Data(
          contentsOf: registry.deletingLastPathComponent().appendingPathComponent("mcp.json")))
      as! [String: Any]
    let servers = config["mcpServers"] as! [String: [String: Any]]
    check(
      Set(servers.keys) == ["exa", "sandbox"] && servers["exa"]?["enabled"] as? Bool == false,
      "No user or remote MCP in Agent")
    check(
      (servers["sandbox"]?["args"] as? [String]) == ["--agent-tools", workspace.path],
      "Production helper receives the task folder")
    check(
      !workspace.path.hasPrefix(registry.deletingLastPathComponent().path),
      "Commands cannot write their configuration")
    func launches() throws -> [[String: Any]] {
      try String(contentsOf: root.appendingPathComponent("mode-launches.jsonl"), encoding: .utf8)
        .split(separator: "\n").map {
          try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any]
        }
    }
    let initialLaunches = try launches().count
    let actualRegistry = try launches().last?["registry"] as? String
    check(
      actualRegistry == registry.path, "Agent actually starts the bridge with its isolated registry"
    )
    studio.newConversation()
    try await wait { !studio.mcpUpdating }
    let second = studio.selectedConversationID!
    let currentLaunches = try launches().count
    check(
      second != agent && currentLaunches == initialLaunches,
      "New agent tasks reuse the resident model")
    studio.selectConversation(agent)
    try await wait { !studio.mcpUpdating }
    check(studio.agentWorkspaceURL == workspace, "Returning to a task restores its folder")
    studio.responseSchema = "{\"type\":\"object\"}"
    studio.draft = "Create a file"
    check(studio.canSend, "Agent ready after tool handshake")
    studio.send()
    try await wait { !studio.isGenerating }
    let request =
      try JSONSerialization.jsonObject(
        with: Data(studio.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
    check(
      request["mcp_enabled"] as? Bool == true && request["response_schema"] == nil,
      "Agent routes tools independently of Chat schema/MCP settings")
    check(
      studio.labSamples.count == 1 && studio.labSamples[0].model == "modes"
        && studio.labSamples[0].mode == "agent", "Completion records real model and mode")
    check(
      studio.labSamples[0].stats.decodeTps == 30 && studio.labSamples[0].stats.prefillTps == 120,
      "Lab retains metrics without recomputation")
    let messageID = studio.currentConversation!.messages.last!.id
    check(
      studio.branchConversation(at: messageID) && studio.currentConversation?.agent == true,
      "Agent branch retains mode")
    try await wait { !studio.mcpUpdating }
    check(studio.persistNow(), "Workspace persisted")
    let restored = StudioModel(conversationFileURL: history, isPreview: true, preferences: prefs)
    check(
      restored.selectedMode == .agent && restored.currentConversation?.agent == true
        && restored.labSamples.count == 1, "Mode and Lab survive relaunch")
    for ttft in [0.0, 1, -1, .infinity, .nan] {
      for prefill in [0.0, 1, -1, .infinity, .nan] {
        for decode in [0.0, 1, -1, .infinity, .nan] {
          for prompt in [-1, 0, 1] {
            for generated in [-1, 0, 1] {
              let stats = GenerationStats(
                ttftSeconds: ttft, prefillTps: prefill, decodeTps: decode,
                promptTokens: prompt, generatedTokens: generated, peakMemoryGB: nil,
                modelSizeGB: nil,
                cachedPromptTokens: nil, evaluatedPromptTokens: nil)
              let sample = LabSample(
                model: "fixture", modelPath: "", bits: nil, mode: "chat", runtime: "", system: "",
                hardware: "", context: 0, temperature: 0, topK: 1, mtp: false, dflash: false,
                stats: stats)
              let expected =
                generated == 1 && prompt >= 0
                && [ttft, prefill, decode].allSatisfy { $0 == 0 || $0 == 1 }
              check(sample.valid == expected, "Lab excludes nonfinite/negative/empty measurements")
            }
          }
        }
      }
    }
    for conversation in studio.conversations
    where conversation.agent && conversation.id != studio.selectedConversationID {
      studio.deleteConversation(conversation.id)
    }
    studio.deleteConversation(studio.selectedConversationID!)
    try await wait { studio.engineState.isReady && !studio.mcpUpdating }
    check(
      studio.selectedMode == .chat && (try? launches().last?["registry"] as? String) == nil,
      "Deleting the last Agent task reloads the actual Chat registry")
    studio.selectMode(.chat)
    try await wait { studio.engineState.isReady }
    check(
      studio.selectedConversationID == chat && studio.responseSchema.contains("object"),
      "Chat history and settings preserved")
    studio.selectMode(.lab)
    studio.draft = "not a chat"
    check(!studio.canSend, "Lab cannot trigger inference")
    studio.startServer(port: 0)
    check(!studio.localServer.active && studio.featureError != nil, "Invalid server port refused")
    // Use a disposable high port and verify the endpoint, not only a ready label.
    let port = 39000 + Int(getpid() % 10000)
    studio.startServer(port: port)
    try await wait { studio.localServer.running || studio.localServer.error != nil }
    check(
      studio.localServer.running && !studio.canSend && !studio.modesBridgeRunning,
      "Server owns the model exclusively")
    let endpoint = URL(string: "http://127.0.0.1:\(port)/v1/models")!
    let (_, refused) = try await URLSession.shared.data(from: endpoint)
    check(
      (refused as? HTTPURLResponse)?.statusCode == 401, "Server rejects unauthenticated clients")
    var authenticated = URLRequest(url: endpoint)
    authenticated.setValue(
      "Bearer \(studio.localServer.apiKey!)", forHTTPHeaderField: "Authorization")
    let (data, response) = try await URLSession.shared.data(for: authenticated)
    check(
      (response as? HTTPURLResponse)?.statusCode == 200 && !data.isEmpty,
      "Actual local server endpoint responds")
    studio.selectMode(.chat)
    try await wait {
      studio.selectedMode == .chat && studio.engineState.isReady && !studio.localServer.active
    }
    check(!studio.localServer.running, "Returning to Chat stops and joins Server")
    studio.localServer.start(model: "serve-fail", port: port, context: 0)
    try await wait { !studio.localServer.active }
    check(
      studio.localServer.error?.contains("fixture bind failure") == true,
      "Startup failure remains visible")
    studio.localServer.start(model: "serve-silent", port: port, context: 0)
    studio.localServer.stop()
    try await wait { !studio.localServer.active }
    check(!studio.localServer.busy, "Silent startup can be cancelled")
    _ = NSApplication.shared
    check(StudioBrand.logoURL != nil && StudioBrand.logo != nil, "Real MLXL3 app artwork loads")
    let logoWindow = NSWindow(
      contentRect: NSRect(x: 0, y: 0, width: 128, height: 128),
      styleMask: .borderless, backing: .buffered, defer: false)
    logoWindow.isReleasedWhenClosed = false
    defer { logoWindow.close() }
    let logoHost = NSHostingView(
      rootView: MonogramMark(size: 96).padding(16).background(StudioTheme.canvas))
    logoWindow.contentView = logoHost
    logoHost.frame = NSRect(x: 0, y: 0, width: 128, height: 128)
    try await Task.sleep(for: .milliseconds(250))
    logoHost.layoutSubtreeIfNeeded()
    guard let logoBitmap = logoHost.bitmapImageRepForCachingDisplay(in: logoHost.bounds) else {
      preconditionFailure("Logo bitmap missing")
    }
    logoHost.cacheDisplay(in: logoHost.bounds, to: logoBitmap)
    if CommandLine.arguments.count > 2 {
      try logoBitmap.representation(using: .png, properties: [:])!.write(
        to: URL(fileURLWithPath: CommandLine.arguments[2]).deletingLastPathComponent()
          .appendingPathComponent("logo-mask-pixels.png"))
    }
    let scale = Double(logoBitmap.pixelsWide) / 128
    func color(_ x: Int, _ y: Int) -> NSColor {
      logoBitmap.colorAt(x: Int(Double(x) * scale), y: Int(Double(y) * scale))!.usingColorSpace(
        .deviceRGB)!
    }
    let background = color(2, 2)
    for point in [(18, 18), (109, 18), (18, 109), (109, 109), (64, 18), (64, 109)] {
      let pixel = color(point.0, point.1)
      if abs(pixel.redComponent - background.redComponent) >= 0.015 {
        FileHandle.standardError.write(
          Data("Logo pixel \(point): \(pixel), baseline \(background)\n".utf8))
      }
      check(
        abs(pixel.redComponent - background.redComponent) < 0.015
          && abs(pixel.greenComponent - background.greenComponent) < 0.015
          && abs(pixel.blueComponent - background.blueComponent) < 0.015,
        "Logo has no rectangular background")
    }
    var visible = 0
    for y in stride(from: 16, to: 112, by: 2) {
      for x in stride(from: 16, to: 112, by: 2)
      where color(x, y).redComponent > background.redComponent + 0.3 { visible += 1 }
    }
    check(visible > 100, "Actual logo remains visible after background masking")
    if CommandLine.arguments.count > 2 {
      _ = NSApplication.shared
      let preview = StudioModel.uiPreview()
      preview.selectMode(.agent)
      let window = NSWindow(
        contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800), styleMask: .borderless,
        backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      let host = NSHostingView(
        rootView: StudioView().environmentObject(preview).preferredColorScheme(.dark))
      window.contentView = host
      host.frame = NSRect(x: 0, y: 0, width: 1200, height: 800)
      defer { window.close() }
      try await Task.sleep(for: .milliseconds(250))
      host.layoutSubtreeIfNeeded()
      guard let image = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
        preconditionFailure("Agent UI did not render")
      }
      host.cacheDisplay(in: host.bounds, to: image)
      let png = image.representation(using: .png, properties: [:])!
      try png.write(to: URL(fileURLWithPath: CommandLine.arguments[2]))
      check(
        image.pixelsWide > 0 && image.pixelsHigh > 0 && png.count > 10_000,
        "Agent workspace renders nonempty UI")
    }
    count += try await StudioModel.checkAgentGuards(root: root, prefs: prefs, model: model)
    print(
      "Studio modes checks passed: \(count), navigation, migration, isolated Agent registry, model reuse, request wiring, Lab persistence and real server lifecycle"
    )
  }
}
