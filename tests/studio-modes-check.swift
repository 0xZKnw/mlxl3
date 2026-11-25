import AppKit
import Foundation
import SwiftUI
import Vision

extension StudioModel {
  func modesRegistryForCheck() throws -> URL { try prepareAgentRegistry() }
  var modesBridgeRunning: Bool { bridge.isRunning }
  func modesStatusForCheck(_ requestID: String?) throws {
    var event: [String: Any] = ["type": "mcp_status", "mcp_servers": 1, "mcp_tools": 1,
      "mcp_errors": [:] as [String: String]]
    if let requestID { event["request_id"] = requestID }
    handle(try JSONDecoder().decode(BridgeEvent.self, from: JSONSerialization.data(withJSONObject: event)))
  }

  static func checkAgentGuards(root: URL, prefs: UserDefaults, model: LocalModel) async throws
    -> Int
  {
    var count = 0
    func check(_ value: @autoclosure () -> Bool, _ message: String) {
      precondition(value(), message)
      count += 1
    }
    func wait(line: UInt = #line, _ predicate: @MainActor () -> Bool) async throws {
      for _ in 0..<400 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      preconditionFailure("Agent guard timed out at line \(line)")
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
    try await wait { !cooperative.isGenerating && !cooperative.agentStudioBusy }
    check(
      cooperative.engineState.isReady && pending.error != nil,
      "Cooperative Stop preserves the loaded model and partial response")
    check(cooperative.labUsage.summary(for: .allTime).models.isEmpty,
      "Cancelling without engine counters does not fabricate usage")
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

    for deadline in [300] {
      let guarded = try await readyAgent("deadline-\(deadline)")
      guarded.draft = "hang-cooperative"
      guarded.send()
      try await wait { guarded.currentConversation?.messages.last?.thinking.isEmpty == false }
      let start = guarded.generationStartedAt!
      guarded.generationLastActivity = start
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
          "5 minutes") == true,
        "Deadline stops the real request and explains why")
      check(
        guarded.generationWatchdog == nil && guarded.cancellationFallback == nil,
        "Completion cleans up every watchdog")
      guarded.stopServices()
      await guarded.bridge.stopAndWait()
    }
    // No total time limit (it used to stop at 10 minutes): an answer that keeps streaming
    // runs as long as it needs; only silence stops it.
    let active = try await readyAgent("deadline-active")
    active.draft = "hang-cooperative"
    active.send()
    try await wait { active.currentConversation?.messages.last?.thinking.isEmpty == false }
    let activeStart = active.generationStartedAt!
    active.generationLastActivity = activeStart + .seconds(3_599)
    active.checkAgentGenerationDeadline(at: activeStart + .seconds(3_600))
    try await Task.sleep(for: .milliseconds(200))
    check(
      active.isGenerating && active.cancellationFallback == nil
        && active.currentConversation!.messages.last!.error == nil,
      "An Agent answer still streaming after an hour is not stopped")
    active.checkAgentGenerationDeadline(at: activeStart + .seconds(3_599 + 300))
    try await wait { !active.isGenerating }
    check(
      active.currentConversation!.messages.last!.error?.contains("5 minutes") == true,
      "Five silent minutes still stop it")
    active.stopServices()
    await active.bridge.stopAndWait()
    let limited = try await readyAgent("budget")
    limited.draft = "budget-limit"
    limited.send()
    try await wait { !limited.isGenerating }
    check(
      limited.currentConversation!.messages.last!.error?.contains("32 768") == true,
      "Budget exhaustion keeps the partial output and reports truncation")
    let limitedRequest =
      try JSONSerialization.jsonObject(
        with: Data(limited.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
    // Finite, but no longer the 4,096 tokens a reasoning model spent before acting.
    check(
      limitedRequest["max_tokens"] as? Int == 32_768 && limitedRequest["tool_rounds"] as? Int == 24,
      "Agent sends a finite production token budget and its tool-round limit: \(limitedRequest["max_tokens"] ?? "nil") \(limitedRequest["tool_rounds"] ?? "nil")")
    limited.stopServices()
    await limited.bridge.stopAndWait()
    // AGENT-2026-11-23-BOUCLE: an engine announcing `agent_loop_supported` gets no total
    // budget and the long round limit; reasoning is never cut (no `reasoning_budget`).
    let marker = root.appendingPathComponent("agent-loop-engine")
    FileManager.default.createFile(atPath: marker.path, contents: Data())
    defer { try? FileManager.default.removeItem(at: marker) }
    let looping = try await readyAgent("agent-loop")
    looping.draft = "Write the whole game"
    looping.send()
    try await wait { !looping.isGenerating && !looping.agentStudioBusy }
    let loopRequest =
      try JSONSerialization.jsonObject(
        with: Data(looping.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
    check(
      loopRequest["max_tokens"] as? Int == -1 && loopRequest["tool_rounds"] as? Int == 256
        && loopRequest["reasoning_budget"] == nil
        && looping.currentConversation!.messages.last!.error == nil,
      "A newer engine's Agent turn has no total budget, 256 rounds and no reasoning cut: \(loopRequest["max_tokens"] ?? "nil") \(loopRequest["tool_rounds"] ?? "nil") \(loopRequest["reasoning_budget"] ?? "nil")")
    let turnsBefore = looping.currentConversation!.messages.count
    looping.draft = "budget-limit"
    looping.send()
    try await wait {
      looping.currentConversation!.messages.count == turnsBefore + 2 && !looping.isGenerating
        && !looping.agentStudioBusy
    }
    let loopError = looping.currentConversation!.messages.last!.error ?? ""
    check(
      loopError.contains("Limite de sortie") || loopError.contains("Output limit"),
      "A newer engine's output limit is reported without the old 32,768 total: \(loopError)")
    check(
      limitedRequest["reasoning_budget"] == nil,
      "An older engine is not sent a reasoning budget")
    looping.selectMode(.chat)
    try await wait { !looping.mcpUpdating && looping.engineState.isReady }
    let chatBefore = looping.currentConversation?.messages.count ?? 0
    looping.draft = "Plain chat"
    looping.send()
    try await wait {
      (looping.currentConversation?.messages.count ?? 0) >= 2
        && looping.currentConversation!.messages.count != chatBefore && !looping.isGenerating
    }
    let chatRequest =
      try JSONSerialization.jsonObject(
        with: Data(looping.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
    check(
      chatRequest["max_tokens"] as? Int == -1 && chatRequest["tool_rounds"] == nil
        && chatRequest["reasoning_budget"] == nil,
      "Chat keeps its request without Agent budgets")
    // Deliver the same engine completion again after it has already closed the request.
    let completedStats = try JSONEncoder().encode(looping.currentConversation!.messages.last!.stats!)
    let replay: [String: Any] = ["type": "complete", "request_id": chatRequest["request_id"] as! String,
      "stats": try JSONSerialization.jsonObject(with: completedStats)]
    looping.handle(try JSONDecoder().decode(BridgeEvent.self, from: JSONSerialization.data(withJSONObject: replay)))
    check(looping.labUsage.summary(for: .allTime).totals == LabUsageTotals(
      inputTokens: 36, outputTokens: 18, generations: 3, cachedInputTokens: nil,
      decode: LabDecodeUsage(average: 30, maximum: 30, count: 3))
      && looping.labSamples.map(\.mode) == ["agent", "agent", "chat"],
      "Chat and Agent share measured usage and a repeated completion cannot count it twice")
    let native: [ReasoningEffort] = [.none, .minimal, .low, .medium, .high, .max]
    check(looping.reasoningLevels == native, "Chat offers the distinct levels of the installed model and engine")
    for level in native {
      looping.newConversation()
      looping.setReasoningLevel(level)
      looping.draft = "Native effort " + level.rawValue
      looping.send()
      try await wait { !looping.isGenerating }
      let request = try JSONSerialization.jsonObject(with: Data(looping.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
      check(request["reasoning_effort"] as? String == level.rawValue && request["reasoning_budget"] == nil
              && request["max_tokens"] as? Int == -1, "Every level sends only native effort and allows a complete answer")
    }
    looping.selectMode(.agent)
    try await wait { !looping.mcpUpdating && looping.engineState.isReady }
    check(looping.reasoningLevels == native, "Agent and Chat use the same model controls")
    looping.draft = "Agent after chat"
    looping.send()
    try await wait { !looping.isGenerating }
    let agentRequest = try JSONSerialization.jsonObject(with: Data(looping.currentConversation!.messages.last!.content.utf8)) as! [String: Any]
    check(agentRequest["reasoning_effort"] as? String == "max" && agentRequest["reasoning_budget"] == nil
            && agentRequest["max_tokens"] as? Int == -1, "Agent preserves the natural effort")
    looping.selectMode(.chat)
    try await wait { !looping.mcpUpdating && looping.engineState.isReady }
    check(looping.effectiveReasoningChoice == .max, "Returning to Chat restores the saved native choice")
    looping.stopServices()
    await looping.bridge.stopAndWait()
    return count
  }
}

@main struct StudioModesCheck {
  // A local midnight after the 25-hour change in New York exercises both civil-day boundaries.
  @MainActor static func checkLabUsage(root: URL, prefs: UserDefaults,
    check: (Bool, String) -> Void) throws {
    let shares = [0, 0.00003, 0.001, 0.2856, 0.999, 0.9999, 1]
    check(shares.map { LabWorkspaceView.formattedShare($0, locale: Locale(identifier: "en_US")) }
      == ["0%", "< 0.1%", "0.1%", "28.6%", "99.9%", "> 99.9%", "100%"],
      "Small and nearly complete portions stay distinguishable from exactly zero and one")
    let iso = ISO8601DateFormatter()
    var calendar = Calendar(identifier: .gregorian)
    calendar.timeZone = TimeZone(identifier: "America/New_York")!
    let now = iso.date(from: "2026-11-02T05:00:00Z")!
    func sample(_ model: String, _ date: Date, _ input: Int, _ output: Int,
      cached: Int? = nil, evaluated: Int? = nil, decode: Double = 3) -> LabSample {
      LabSample(date: date, model: model, modelPath: "", bits: nil, mode: "chat", runtime: "",
        system: "", hardware: "", context: 8192, temperature: 0, topK: 1, mtp: false, dflash: false,
        stats: GenerationStats(ttftSeconds: 1, prefillTps: 5, decodeTps: decode,
          promptTokens: input, generatedTokens: output, peakMemoryGB: nil, modelSizeGB: nil,
          cachedPromptTokens: cached, evaluatedPromptTokens: evaluated, contextUsed: 8192))
    }
    let a = sample("A", now, 10, 6, cached: 7, evaluated: 3)
    let b = sample("B", now.addingTimeInterval(-1), 12, 4, evaluated: 4)
    let seventh = sample("A", iso.date(from: "2026-10-27T04:00:00Z")!, 4, 2, cached: 0)
    let eighth = sample("B", seventh.date.addingTimeInterval(-1), 7, 1)
    let thirtieth = sample("B", iso.date(from: "2026-10-04T04:00:00Z")!, 5, 1)
    let older = sample("C", thirtieth.date.addingTimeInterval(-1), 14, 4)
    let ledger = LabUsageLedger(samples: [older, a, b, seventh, eighth, thirtieth, a], calendar: calendar)
    func stats(_ period: LabUsagePeriod) -> LabUsageSummary {
      ledger.summary(for: period, now: now, calendar: calendar)
    }
    check(stats(.today).totals == LabUsageTotals(inputTokens: 10, outputTokens: 6,
      generations: 1, cachedInputTokens: 7, decode: LabDecodeUsage(average: 3, maximum: 3, count: 1)), "Today excludes the preceding second and counts reused input once")
    check(stats(.sevenDays).totals == LabUsageTotals(inputTokens: 26, outputTokens: 12,
      generations: 3, cachedInputTokens: 15, decode: LabDecodeUsage(average: 3, maximum: 3, count: 3)), "Seven local days cross the DST transition and include day seven")
    check(stats(.thirtyDays).totals.totalTokens == 52 && stats(.thirtyDays).totals.cachedInputTokens == nil,
      "Thirty days include the first midnight, exclude the preceding second and preserve unknown cache counts")
    check(stats(.allTime).models.map(\.model) == ["B", "A", "C"]
      && stats(.allTime).models.map(\.totals.totalTokens) == [30, 22, 18]
      && stats(.allTime).totals.generations == 6, "All-time totals group by model and deduplicate measurement IDs")
    check(LabUsagePeriod.allCases.map { stats($0).activeDays } == [1, 3, 5, 6],
      "Count days with recorded usage once, including calendar boundaries")
    check(LabUsagePeriod.allCases.map { stats($0).previousPeriodTokens } == [16, 8, 18, nil],
      "Compare with the preceding disjoint day, week or month across the fall clock change")
    let peak = stats(.sevenDays).busiestDay
    check(peak?.tokens == 16 && peak?.date == iso.date(from: "2026-11-02T00:00:00Z")
      && peak?.formatted(locale: Locale(identifier: "en_US")) == "Nov 2, 2026",
      "Two equal peaks choose the most recent civil day and its localized date label")
    let multiModel = sample("C", seventh.date, 6, 7)
    let combined = LabUsageLedger(samples: [older, a, b, seventh, multiModel], calendar: calendar)
      .summary(for: .allTime, now: now, calendar: calendar)
    check(combined.activeDays == 4 && combined.busiestDay?.tokens == 19
      && combined.busiestDay?.date == iso.date(from: "2026-10-27T00:00:00Z"),
      "Activity and the largest daily volume group different models by civil date")
    let empty = LabUsageLedger().summary(for: .today, now: now, calendar: calendar)
    let yesterdayOnly = LabUsageLedger(samples: [b], calendar: calendar)
      .summary(for: .today, now: now, calendar: calendar)
    check(empty.activeDays == 0 && empty.busiestDay == nil && empty.previousPeriodTokens == nil
      && yesterdayOnly.activeDays == 0 && yesterdayOnly.busiestDay == nil && yesterdayOnly.previousPeriodTokens == 16,
      "Empty periods have no peak, and an available comparison does not require current activity")
    let previousWeekStart = iso.date(from: "2026-10-20T04:00:00Z")!
    let previousMonthStart = iso.date(from: "2026-09-04T04:00:00Z")!
    let boundarySamples = [sample("A", previousWeekStart.addingTimeInterval(-1), 4, 1),
      sample("A", previousWeekStart, 8, 1), sample("B", previousMonthStart.addingTimeInterval(-1), 16, 1),
      sample("B", previousMonthStart, 32, 1)]
    let boundaries = LabUsageLedger(samples: boundarySamples, calendar: calendar)
    check(boundaries.summary(for: .sevenDays, now: now, calendar: calendar).previousPeriodTokens == 9
      && boundaries.summary(for: .thirtyDays, now: now, calendar: calendar).previousPeriodTokens == 33,
      "Previous periods include their first local midnight and exclude the preceding second")
    let tie = LabUsageLedger(samples: [b, a], calendar: calendar).summary(for: .allTime)
    check(tie.models.map(\.model) == ["A", "B"], "Equal usage has a deterministic model ordering")
    let future = sample("A", iso.date(from: "2026-11-03T05:00:00Z")!, 1, 1)
    check(!LabUsagePeriod.today.contains(b, now: now, calendar: calendar)
      && !LabUsagePeriod.sevenDays.contains(future, now: now, calendar: calendar)
      && LabUsagePeriod.thirtyDays.contains(thirtieth, now: now, calendar: calendar),
      "Detail filters use the same inclusive local-day boundaries as summaries")
    var dated = b
    dated.localDay = LabUsagePeriod.day(b.date, calendar: calendar)
    var utc = calendar
    utc.timeZone = TimeZone(secondsFromGMT: 0)!
    check(!LabUsagePeriod.today.contains(dated, now: now, calendar: utc)
      && LabUsageLedger(samples: [dated], calendar: utc).summary(for: .today, now: now, calendar: utc).models.isEmpty,
      "Changing time zones keeps a recorded local day in both totals and details")
    let inconsistent = sample("A", now, 4, 3, cached: 4, evaluated: 2)
    check(LabUsageLedger(samples: [inconsistent]).summary(for: .allTime).totals.cachedInputTokens == nil,
      "Inconsistent cache counters do not inflate measured input")
    let huge = sample("A", now, .max, 1)
    check(LabUsageLedger(samples: [huge, a]).summary(for: .allTime).totals.totalTokens == .max,
      "Corrupt or extreme counters cannot wrap token totals")

    let extremeActivity = LabUsageLedger(samples: [sample("A", b.date, .max, 1),
      sample("B", b.date, 1, 1), huge, a], calendar: calendar)
    check(extremeActivity.summary(for: .allTime).busiestDay?.tokens == .max
      && extremeActivity.summary(for: .today, now: now, calendar: calendar).previousPeriodTokens == .max,
      "Extreme daily and comparison totals saturate instead of wrapping")

    let invalid = [0.0, -2, .infinity, -.infinity, .nan].map { sample("A", now, 2, 3, decode: $0) }
    let noInterval = [sample("A", now, 3, 0, decode: 200), sample("A", now, 4, 1, decode: 300)]
    let unmeasured = LabUsageLedger(samples: invalid + noInterval).summary(for: .allTime).totals
    check(unmeasured.decode == nil && unmeasured.generations == 7 && unmeasured.totalTokens == 33,
      "A missing decode interval or a nonpositive/nonfinite rate leaves measured token usage intact")
    let measured = [sample("A", now, 100, 2, decode: 10), sample("A", now, 1, 12, decode: 30),
      sample("A", now, 500, 3, decode: 50), sample("A", b.date, 1, 2, decode: 90),
      sample("B", now, 2, 6, decode: 25), sample("A", older.date, 3, 2, decode: 200)]
    let speeds = LabUsageLedger(samples: measured + [measured[1]] + invalid + noInterval, calendar: calendar)
    let today = speeds.summary(for: .today, now: now, calendar: calendar)
    let week = speeds.summary(for: .sevenDays, now: now, calendar: calendar)
    let month = speeds.summary(for: .thirtyDays, now: now, calendar: calendar)
    let all = speeds.summary(for: .allTime)
    check(today.models.first?.totals.decode == LabDecodeUsage(average: 30, maximum: 50, count: 3)
      && week.models.first?.totals.decode == LabDecodeUsage(average: 45, maximum: 90, count: 4),
      "Per-model means weight measured generations equally across days with different sample counts")
    check(week.models.last?.totals.decode == LabDecodeUsage(average: 25, maximum: 25, count: 1)
      && week.totals.decode?.count == 5 && abs((week.totals.decode?.average ?? 0) - 41) < 1e-12
      && month.models.first?.totals.decode == week.models.first?.totals.decode
      && all.models.first?.totals.decode?.maximum == 200 && all.models.first?.totals.decode?.count == 5,
      "Decode maxima, means and sample counts respect the selected period and model")
    let extremes = LabUsageLedger(samples: [sample("A", now, 0, 2, decode: .greatestFiniteMagnitude),
      sample("A", now, 0, 2, decode: .greatestFiniteMagnitude), sample("A", now, 0, 2, decode: .leastNonzeroMagnitude)])
    check(extremes.valid && extremes.summary(for: .allTime).totals.decode?.average.isFinite == true,
      "All positive finite decode speeds remain persistable without an overflowing sum")

    // Legacy snapshots retained only the detailed list. The migration must archive it before cropping.
    let file = root.appendingPathComponent("legacy-lab.json")
    let samples = (0..<1001).map { _ in sample("historic", now, 3, 2) }
    var legacy = WorkspaceSnapshot(selectedConversationID: nil, selectedModelName: nil, conversations: [],
      temperature: 0, topK: 1, repetitionPenalty: 1, systemPrompt: "")
    legacy.labSamples = samples + [samples[0]]
    try ConversationStore.write(legacy, to: file)
    let migrated = StudioModel(conversationFileURL: file, isPreview: true, preferences: prefs)
    let expected = LabUsageTotals(inputTokens: 3003, outputTokens: 2002, generations: 1001, cachedInputTokens: nil,
      decode: LabDecodeUsage(average: 3, maximum: 3, count: 1001))
    check(migrated.labUsage.summary(for: .allTime).totals == expected
      && migrated.labSamples.count == 1000 && migrated.labSamples.first?.id == samples[1].id,
      "Migration preserves old totals beyond the detail cap and ignores duplicate IDs without needing a conversation")
    check(migrated.persistNow(), "Migrated usage saves through the workspace writer")
    let reopened = StudioModel(conversationFileURL: file, isPreview: true, preferences: prefs)
    check(reopened.labUsage.summary(for: .allTime).totals == expected && reopened.labSamples.count == 1000,
      "Reopening the migrated ledger does not replay the recent samples or lose older usage")

    // Exact wire shape of the original token-only ledger, including an archived model with no details.
    let day = LabUsagePeriod.day(now, calendar: .current)!
    let oldLedger = """
      {"days":{"\(day)":{"historic":{"inputTokens":3003,"outputTokens":2002,"generations":1001}},
        "\(day - 61)":{"archived":{"inputTokens":80,"outputTokens":40,"generations":5}}}}
      """
    legacy.labUsage = try JSONDecoder().decode(LabUsageLedger.self, from: Data(oldLedger.utf8))
    legacy.labSamples = [samples[0], samples[1], samples[0]]
    let oldFile = root.appendingPathComponent("token-only-lab.json")
    try ConversationStore.write(legacy, to: oldFile)
    let filled = StudioModel(conversationFileURL: oldFile, isPreview: true, preferences: prefs)
    let recovered = filled.labUsage.summary(for: .allTime)
    check(recovered.models.map(\.model) == ["historic", "archived"] && recovered.totals.totalTokens == 5125
      && recovered.totals.generations == 1006 && recovered.models[1].totals.decode == nil
      && recovered.models[0].totals.decode == LabDecodeUsage(average: 3, maximum: 3, count: 2),
      "Backfilling a token-only ledger preserves every historical total and uses only available, unique rates")
    check(recovered.activeDays == 2 && recovered.busiestDay?.tokens == 5005
      && recovered.busiestDay?.date == Date(timeIntervalSince1970: Double(day) * 86_400),
      "Archived daily usage contributes without any retained individual generation details")
    check(filled.persistNow(), "Backfilled rates save atomically with existing usage")
    let again = StudioModel(conversationFileURL: oldFile, isPreview: true, preferences: prefs)
    check(again.labUsage.summary(for: .allTime).totals == recovered.totals,
      "A second launch does not backfill the same two speeds a second time")
  }

  @MainActor static func main() {
    let app = NSApplication.shared
    app.setActivationPolicy(.accessory)
    // SwiftUI's native buttons use AppKit's event loop while tracking a click.
    Task { @MainActor in
      do {
        try await runChecks()
        exit(EXIT_SUCCESS)
      } catch {
        FileHandle.standardError.write(Data("\(error)\n".utf8))
        exit(EXIT_FAILURE)
      }
    }
    app.run()
    exit(EXIT_FAILURE)
  }

  @MainActor private static func runChecks() async throws {
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
    func wait(line: UInt = #line, _ predicate: @MainActor () -> Bool) async throws {
      for _ in 0..<400 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      preconditionFailure("Mode fixture timed out at line \(line): mode=\(studio.selectedMode), engine=\(studio.engineState), MCP updating=\(studio.mcpUpdating), tools=\(studio.mcpToolCount), server=\(studio.localServer.log), error=\(studio.featureError ?? "none")")
    }
    let modelFolder = root.appendingPathComponent("model")
    try FileManager.default.createDirectory(at: modelFolder, withIntermediateDirectories: true)
    try #"{% set effort_map = {"none":0, "minimal":0.1, "low":0.2, "medium":0.7, "high":0.9, "xhigh":0.99, "max":0.99} %}{{ reasoning_effort }}"#
      .write(to: modelFolder.appendingPathComponent("chat_template.jinja"), atomically: true, encoding: .utf8)
    let model = LocalModel(
      name: "modes", path: modelFolder.path, modelType: "fixture", format: "EXL3", bits: 2.5,
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
      Set(servers.keys) == ["exa", "sandbox"] && servers["exa"]?["enabled"] as? Bool == true,
      "No user or remote MCP in Agent; web search on with Internet, the default")
    check(
      (servers["sandbox"]?["args"] as? [String]) == [
        "--agent-tools", workspace.path,
        workspace.deletingLastPathComponent().appendingPathComponent("studio/project.json").path,
      ],
      "Production helper receives the task folder and its permission envelope")
    let permissionsFile = workspace.deletingLastPathComponent().appendingPathComponent(
      "studio/project.json")
    let project = AgentProjectStore(workspace: workspace)
    check(
      FileManager.default.fileExists(atPath: permissionsFile.path)
        && !permissionsFile.path.hasPrefix(workspace.path + "/"),
      "Helper permissions exist outside the model-writable task folder")
    let initialState = try JSONDecoder().decode(
      AgentProjectState.self, from: Data(contentsOf: permissionsFile))
    check(initialState.permissions == AgentPermissions(), "New task permissions use the defaults")
    var restricted = AgentPermissions()
    restricted.write = false
    restricted.run = false
    restricted.commandSeconds = 1
    restricted.internet = false
    try project.setPermissions(restricted)
    _ = try studio.modesRegistryForCheck()
    let restrictedState = try JSONDecoder().decode(
      AgentProjectState.self, from: Data(contentsOf: permissionsFile))
    check(
      restrictedState.permissions == restricted,
      "Refreshing the registry preserves denied tools and the bounded command deadline")
    check(
      !workspace.path.hasPrefix(registry.deletingLastPathComponent().path),
      "Commands cannot write their configuration")
    // Web search follows the task's Internet setting; the sandbox entry is unchanged.
    let offlineServers =
      (try JSONSerialization.jsonObject(
        with: Data(contentsOf: registry.deletingLastPathComponent().appendingPathComponent("mcp.json")))
        as! [String: Any])["mcpServers"] as! [String: [String: Any]]
    check(offlineServers["exa"]?["enabled"] as? Bool == false, "Internet off disables web search")
    var online = restricted
    online.internet = true
    try project.setPermissions(online)
    let onlineServers =
      (try JSONSerialization.jsonObject(
        with: Data(contentsOf: try studio.modesRegistryForCheck().deletingLastPathComponent().appendingPathComponent("mcp.json")))
        as! [String: Any])["mcpServers"] as! [String: [String: Any]]
    check(
      onlineServers["exa"]?["enabled"] as? Bool == true && onlineServers["sandbox"]?["args"] as? [String] == servers["sandbox"]?["args"] as? [String],
      "Internet enables web search for this task only")
    try project.setPermissions(restricted)
    _ = try studio.modesRegistryForCheck()
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
    let secondWorkspace = studio.agentWorkspaceURL!
    _ = try studio.modesRegistryForCheck()
    let secondConfig =
      try JSONSerialization.jsonObject(
        with: Data(
          contentsOf: registry.deletingLastPathComponent().appendingPathComponent("mcp.json")))
      as! [String: Any]
    let secondServers = secondConfig["mcpServers"] as! [String: [String: Any]]
    check(
      secondWorkspace != workspace
        && (secondServers["sandbox"]?["args"] as? [String]) == [
          "--agent-tools", secondWorkspace.path,
          secondWorkspace.deletingLastPathComponent().appendingPathComponent("studio/project.json")
            .path,
        ],
      "Switching tasks replaces both helper paths without reusing another task's permissions")
    let secondState = try AgentProjectStore(workspace: secondWorkspace).state()
    check(
      secondState.permissions == AgentPermissions(),
      "A new task does not inherit denied tools from the previous task")
    try project.setPermissions(AgentPermissions())
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
    let sample = studio.labSamples[0]
    check(sample.id == studio.currentConversation?.messages.last?.id,
      "An Agent measurement identifies the generation that produced it")
    check(studio.labUsage.summary(for: .today).totals == LabUsageTotals(
      inputTokens: 12, outputTokens: 6, generations: 1, cachedInputTokens: nil,
      decode: LabDecodeUsage(average: 30, maximum: 30, count: 1)),
      "Agent completion uses actual prompt and generated counts")
    try checkLabUsage(root: root, prefs: prefs) { result, message in check(result, message) }
    checkLabResponseMeasurements { check($0, $1) }
    let messageID = studio.currentConversation!.messages.last!.id
    try await wait { !studio.agentStudioBusy }
    check(
      studio.branchConversation(at: messageID) && studio.currentConversation?.agent == true,
      "Agent branch retains mode")
    try await wait { !studio.mcpUpdating }
    check(studio.persistNow(), "Workspace persisted")
    let restored = StudioModel(conversationFileURL: history, isPreview: true, preferences: prefs)
    check(
      restored.selectedMode == .agent && restored.currentConversation?.agent == true
        && restored.labSamples.count == 1 && restored.labUsage.summary(for: .allTime).totals.totalTokens == 18,
      "Branching copies messages without double counting them on save or relaunch")
    studio.regenerate(messageID)
    let regeneratedBranch = studio.selectedConversationID!
    check(studio.mcpUpdating && !studio.isGenerating,
      "Regeneration must wait for delayed MCP handshake")
    try studio.modesStatusForCheck("obsolete-workspace-handshake")
    check(studio.mcpUpdating && !studio.isGenerating,
      "A stale workspace handshake released regeneration")
    try await wait { studio.currentConversation?.messages.last?.isStreaming == false
      && studio.currentConversation?.messages.last?.role == .assistant && !studio.agentStudioBusy }
    let regenerated = studio.currentConversation!.messages.last!
    check(studio.selectedConversationID == regeneratedBranch && regenerated.error == nil
      && regenerated.content.contains("Create a file"), "Agent regeneration never produced its response")
    check(studio.currentConversation!.messages.count == 2, "Regeneration sent more than once")
    check(studio.labUsage.summary(for: .allTime).totals.generations == 2
      && studio.labUsage.summary(for: .allTime).totals.totalTokens == 36,
      "Regenerating a copied message contributes the new generation once")
    // Allow an extra status/ready polling interval: no second request may appear.
    try await Task.sleep(for: .milliseconds(300))
    check(studio.currentConversation!.messages.count == 2, "Late handshake duplicated regeneration")
    studio.regenerate(regenerated.id)
    let abandonedBranch = studio.selectedConversationID!
    studio.selectConversation(regeneratedBranch)
    try await wait { !studio.mcpUpdating }
    try await Task.sleep(for: .milliseconds(300))
    check(studio.conversations.first(where: { $0.id == abandonedBranch })?.messages.isEmpty == true,
      "Navigating away did not cancel pending regeneration")
    studio.regenerate(regenerated.id)
    studio.stopGeneration()
    try await wait { !studio.mcpUpdating }
    try await Task.sleep(for: .milliseconds(300))
    check(studio.currentConversation?.messages.isEmpty == true,
      "Stop did not cancel regeneration waiting for tools")
    studio.selectConversation(regeneratedBranch)
    try await wait { !studio.mcpUpdating }
    studio.regenerate(regenerated.id)
    try studio.modesStatusForCheck(nil)
    try await Task.sleep(for: .milliseconds(100))
    check(studio.currentConversation?.messages.isEmpty == true && studio.featureError != nil,
      "An uncorrelated legacy handshake silently auto-sent a regeneration")
    try await wait { !studio.mcpUpdating }
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
              let usage = LabUsageLedger(samples: [sample]).summary(for: .allTime).totals
              let usableTokens = prompt >= 0 && generated >= 0 && (prompt == 1 || generated == 1)
              check(usage.generations == (usableTokens ? 1 : 0)
                && usage.totalTokens == (usableTokens ? prompt + generated : 0),
                "Ledger counts measured tokens independently of speed availability")
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
    check(studio.labUsage.summary(for: .allTime).totals.generations == 2,
      "Cancelled regenerations and deleting Agent conversations leave measured usage intact")
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
    studio.startServer(port: 8000, host: "example.com")
    check(!studio.localServer.active, "Unsupported server host refused")
    studio.startServer(port: 8000, context: -1)
    check(!studio.localServer.active, "Negative server context refused")
    studio.startServer(port: 8000, context: Int(Int32.max))
    check(!studio.localServer.active, "Context above the loaded model limit refused")
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
    studio.startServer(port: port, host: "0.0.0.0", context: 1024)
    try await wait { studio.localServer.running || studio.localServer.error != nil }
    check(studio.localServer.running && studio.localServer.log.contains("0.0.0.0")
          && studio.localServer.log.contains("1024"), "Network host and context reach the server")
    var networkRequest = URLRequest(url: endpoint)
    networkRequest.setValue("Bearer \(studio.localServer.apiKey!)", forHTTPHeaderField: "Authorization")
    let (_, networkResponse) = try await URLSession.shared.data(for: networkRequest)
    check((networkResponse as? HTTPURLResponse)?.statusCode == 200, "Network-mode server accepts its generated key")
    let (_, networkRefused) = try await URLSession.shared.data(from: endpoint)
    check((networkRefused as? HTTPURLResponse)?.statusCode == 401, "Network-mode server requires a key")
    studio.selectMode(.chat)
    try await wait { studio.selectedMode == .chat && studio.engineState.isReady && !studio.localServer.active }
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
      rootView: MonogramMark(size: 96).padding(16).background(StudioTheme.canvas)
        .studioAppearance(.dark))
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
    try await checkLabPeriodControl(root: root, prefs: prefs) { check($0, $1) }
    print(
      "Studio modes checks passed: \(count), navigation, migration, isolated Agent registry, model reuse, request wiring, Lab persistence and real server lifecycle"
    )
  }
}

extension StudioModesCheck {
  @MainActor static func checkLabPeriodControl(root: URL, prefs: UserDefaults,
    check: (Bool, String) -> Void) async throws {
    let now = Date()
    func row(_ model: String, _ daysAgo: Int, _ input: Int, _ output: Int, _ decode: Double,
             firstToken: Double? = nil) -> LabSample {
      var stats = GenerationStats(ttftSeconds: 1, prefillTps: 5, decodeTps: decode, promptTokens: input,
        generatedTokens: output, peakMemoryGB: nil, modelSizeGB: nil, cachedPromptTokens: nil, evaluatedPromptTokens: nil)
      stats.endToEndTTFTSeconds = firstToken
      return LabSample(date: Calendar.current.date(byAdding: .day, value: -daysAgo, to: now)!,
        model: model, modelPath: "", bits: nil, mode: "chat", runtime: "", system: "", hardware: "",
        context: 4096, temperature: 0, topK: 1, mtp: false, dflash: false, stats: stats)
    }
    let active = UUID()
    let chats = [ConversationSnapshot(id: UUID(), title: "Another chat", messages: [], createdAt: now),
                 ConversationSnapshot(id: active, title: "Active chat", messages: [], createdAt: now)]
    var snapshot = WorkspaceSnapshot(selectedConversationID: active, selectedModelName: nil, conversations: chats,
      temperature: 0, topK: 1, repetitionPenalty: 1, systemPrompt: "")
    snapshot.labSamples = [row("Atlas", 0, 100, 29, 26.5, firstToken: 2),
                           row("Birch", 0, 70, 15, 42.5, firstToken: 6),
                           row("Birch", 14, 192, 10, 67.5, firstToken: 9),
                           row("Atlas", 45, 316, 35, 92.5), row("Cedar", 45, 10, 0, 0)]
    let file = root.appendingPathComponent("lab-period-switch.json")
    try ConversationStore.write(snapshot, to: file)
    let fixture = StudioModel(conversationFileURL: file, isPreview: true, preferences: prefs)
    let previousLanguage = fixture.language
    fixture.setLanguage(.en)
    defer { fixture.setLanguage(previousLanguage) }
    fixture.draft = "Continue this draft after checking usage"
    fixture.selectMode(.lab)
    var host = NSHostingView(rootView: StudioView().environmentObject(fixture)
      .defaultAppStorage(prefs).preferredColorScheme(.light))
    let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1160, height: 660),
                          styleMask: .borderless, backing: .buffered, defer: false)
    window.isReleasedWhenClosed = false
    window.contentView = host
    window.setFrameOrigin(NSPoint(x: -6000, y: -6000))
    window.orderBack(nil)
    defer { window.close() }
    try await Task.sleep(for: .milliseconds(150))
    host.layoutSubtreeIfNeeded()

    func periodControl(in view: NSView) -> NSSegmentedControl? {
      if let picker = view as? NSSegmentedControl, picker.segmentCount == 4 { return picker }
      for child in view.subviews {
        if let picker = periodControl(in: child) { return picker }
      }
      return nil
    }
    guard let picker = periodControl(in: host) else { preconditionFailure("Missing Lab period control") }
    // SwiftUI text is drawn into the hosting view. Reading that view's own bitmap
    // exercises the displayed value without Screen Recording or Accessibility access.
    typealias Cell = (text: String, box: CGRect)
    func displayedCells() async throws -> [Cell] {
      try await Task.sleep(for: .milliseconds(120))
      host.layoutSubtreeIfNeeded()
      guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
        preconditionFailure("No Lab bitmap")
      }
      host.cacheDisplay(in: host.bounds, to: bitmap)
      guard let image = bitmap.cgImage else { preconditionFailure("No Lab pixels") }
      let read = VNRecognizeTextRequest()
      read.recognitionLevel = .accurate
      read.usesLanguageCorrection = false
      read.recognitionLanguages = ["en-US"]
      read.minimumTextHeight = 0.014
      let stages = try read.supportedComputeStageDevices
      precondition(!stages.isEmpty, "No OCR compute stages")
      for (stage, devices) in stages {
        guard let cpu = devices.first(where: { if case .cpu = $0 { true } else { false } }) else {
          preconditionFailure("No CPU OCR device for the Lab test")
        }
        read.setComputeDevice(cpu, for: stage)
      }
      do {
        try VNImageRequestHandler(cgImage: image).perform([read])
      } catch {
        // Vision can transiently fail to prepare its CPU compute operation on the first image.
        FileHandle.standardError.write(Data("Retrying Lab OCR after \(error)\n".utf8))
        try await Task.sleep(for: .milliseconds(300))
        try VNImageRequestHandler(cgImage: image).perform([read])
      }
      return (read.results ?? []).compactMap { observation in
        observation.topCandidates(1).first.map { (text: $0.string, box: observation.boundingBox) }
      }
    }

    func clickSidebar(_ word: String, in cells: [Cell]) {
      let sidebar = cells.filter { $0.box.midX * host.bounds.width < fixture.sidebarWidth.points }
      guard let item = sidebar.first(where: { cell in
        cell.text.lowercased().split(whereSeparator: { !$0.isLetter }).contains(Substring(word.lowercased()))
      }) else {
        preconditionFailure("Missing Lab sidebar action '\(word)', read \(sidebar.map(\.text))")
      }
      click(item)
    }

    func click(_ item: Cell) {
      // Vision and window event coordinates both have their origin at the bottom left.
      let point = NSPoint(x: item.box.midX * host.bounds.width, y: item.box.midY * host.bounds.height)
      func event(_ type: NSEvent.EventType) -> NSEvent {
        NSEvent.mouseEvent(with: type, location: point, modifierFlags: [],
          timestamp: ProcessInfo.processInfo.systemUptime, windowNumber: window.windowNumber,
          context: nil, eventNumber: 0, clickCount: 1, pressure: 1)!
      }
      NSApp.postEvent(event(.leftMouseUp), atStart: false)
      window.sendEvent(event(.leftMouseDown))
      if let up = NSApp.nextEvent(matching: .leftMouseUp, until: .distantPast, inMode: .default, dequeue: true) {
        window.sendEvent(up)
      }
    }

    @discardableResult
    func expectDisplayed(_ total: String, model: String, tokens: String, average: String, maximum: String) async throws -> [Cell] {
      let cells = try await displayedCells()
      check(cells.contains { $0.text == total }, "Expected the rendered Lab total \(total), read \(cells.map(\.text))")
      // These three values share the model name's baseline. Their visual order
      // distinguishes totals, means, and maxima even when two rates coincide.
      guard let name = cells.first(where: { $0.text == model }) else {
        preconditionFailure("Missing rendered model \(model)")
      }
      let row = cells.filter {
        $0.box.midX > name.box.midX && abs($0.box.midY - name.box.midY) * host.bounds.height < 9
      }.sorted { $0.box.midX < $1.box.midX }
      let numbers = row.flatMap { $0.text.split(separator: " ") }
      check(Array(numbers.suffix(3)) == [Substring(tokens), Substring(average), Substring(maximum)],
            "Expected \(tokens) tokens, then rates \(average) and \(maximum) for \(model), read \(numbers)")
      return cells
    }
    func expectText(_ text: String, in cells: [Cell]) {
      let words = cells.map { $0.text.lowercased() }.joined(separator: " ")
      check(words.contains(text.lowercased()), "Expected '\(text)' in rendered Lab, read \(cells.map(\.text))")
    }
    let month = try await expectDisplayed("416", model: "Birch", tokens: "287", average: "55.0", maximum: "67.5")
    expectText("18 tokens", in: month)
    check(month.contains { $0.text.range(of: #"(?i)\b6\s*s\b"#, options: .regularExpression) != nil },
      "Expected 6 s for the 30-day median first text, read \(month.map(\.text))")
    expectText("2 active days", in: month)
    expectText("214 tokens", in: month)
    expectText("361 tokens", in: month)
    let periods = [(L("Aujourd’hui", "Today"), "214", "Birch", "85", "42.5", "42.5"),
                   (L("Depuis le début", "All time"), "777", "Atlas", "480", "59.5", "92.5")]
    for (title, total, model, tokens, average, maximum) in periods {
      guard let segment = (0..<picker.segmentCount).first(where: { picker.label(forSegment: $0) == title }) else {
        preconditionFailure("Missing Lab segment \(title)")
      }
      picker.selectedSegment = segment
      check(picker.sendAction(picker.action, to: picker.target), "Lab period action reaches its target")
      let result = try await expectDisplayed(total, model: model, tokens: tokens, average: average, maximum: maximum)
      if title == "Today" {
        expectText("22 tokens", in: result)
        check(result.contains { $0.text.range(of: #"(?i)\b4\s*s\b"#, options: .regularExpression) != nil },
          "Expected 4 s for today's median first text, read \(result.map(\.text))")
        expectText("1 active day", in: result)
      } else {
        expectText("361 tokens", in: result)
        expectText("3 active days", in: result)
        check(!result.contains { $0.text.localizedCaseInsensitiveContains("previous") },
          "All-time usage has no preceding period")
      }
    }
    window.setContentSize(NSSize(width: 820, height: 660))
    let compact = try await expectDisplayed("777", model: "Atlas", tokens: "480", average: "59.5", maximum: "92.5")

    clickSidebar("Generations", in: compact)
    let history = try await displayedCells()
    check(history.contains { $0.text.contains("Export") } && history.contains { $0.text.contains("92.5") },
      "Generations exposes its export action and the measured decode rate from 45 days ago")
    // "This" uniquely identifies this sidebar entry and avoids OCR's Latin/Cyrillic Mac homoglyphs.
    clickSidebar("This", in: history)
    let hardware = try await displayedCells()
    check(hardware.contains { $0.text.contains("CPU cores") } && hardware.contains { $0.text.contains("Unified memory") },
      "This Mac shows processor and memory details")
    clickSidebar("Usage", in: hardware)
    let usage = try await expectDisplayed("777", model: "Atlas", tokens: "480", average: "59.5", maximum: "92.5")
    guard let restored = periodControl(in: host) else { preconditionFailure("Missing period control after navigation") }
    check(restored.selectedSegment >= 0 && restored.label(forSegment: restored.selectedSegment) == "All time",
      "Navigating between Lab pages preserves the selected period")
    clickSidebar("Back", in: usage)
    try await Task.sleep(for: .milliseconds(120))
    check(fixture.selectedMode == .chat && fixture.selectedConversationID == active
      && fixture.draft == "Continue this draft after checking usage",
      "Lab Back returns to the original Studio conversation with its unsent draft")

    fixture.showAppSettings = true
    fixture.selectMode(.lab)
    check(!fixture.canOpenLab && fixture.selectedMode == .chat, "Opening Lab while Settings is visible is refused")
    fixture.showAppSettings = false
    fixture.selectMode(.lab)
    check(fixture.selectedMode == .lab, "Lab can open again after Settings closes")
    fixture.selectMode(.chat)
    check(fixture.selectedMode == .chat && fixture.selectedConversationID == active
      && fixture.draft == "Continue this draft after checking usage",
      "Reselecting the previous mode also restores the original chat and draft")

    // A historic generation still belongs to All time when the selected period is empty.
    snapshot.labSamples = [row("Earlier", 90, 80, 7, 19, firstToken: 8)]
    let earlierFile = root.appendingPathComponent("lab-historic-only.json")
    try ConversationStore.write(snapshot, to: earlierFile)
    let earlier = StudioModel(conversationFileURL: earlierFile, isPreview: true, preferences: prefs)
    earlier.selectMode(.lab)
    host = NSHostingView(rootView: StudioView().environmentObject(earlier)
      .defaultAppStorage(prefs).preferredColorScheme(.light))
    window.contentView = host
    let empty = try await displayedCells()
    // Vision occasionally reads a small zero as an uppercase O.
    check(empty.contains { $0.text.range(of: #"(?i)\b[0o] generations\b"#, options: .regularExpression) != nil },
      "Empty period shows zero generations, read \(empty.map(\.text))")
    expectText("No generations recorded in this period", in: empty)
    guard let otherPeriod = empty.first(where: { $0.text.localizedCaseInsensitiveContains("Show all time") }) else {
      preconditionFailure("Expected an action to view the archived usage, read \(empty.map(\.text))")
    }
    click(otherPeriod)
    let historic = try await displayedCells()
    check(historic.contains { $0.text == "87" } && historic.contains { $0.text == "Earlier" },
      "Show all time opens the recorded history from an empty period")


  }
}

extension StudioModesCheck {
  static func checkLabResponseMeasurements(check: (Bool, String) -> Void) {
    var utc = Calendar(identifier: .gregorian)
    utc.timeZone = TimeZone(secondsFromGMT: 0)!
    let now = Date(timeIntervalSince1970: 100 * 86_400)
    func sample(_ endToEnd: Double?, daysAgo: Int = 0, model: String = "Lab") -> LabSample {
      var stats = GenerationStats(ttftSeconds: 0.01, prefillTps: 260, decodeTps: 40,
        promptTokens: 5, generatedTokens: 1, peakMemoryGB: nil, modelSizeGB: nil,
        cachedPromptTokens: nil, evaluatedPromptTokens: nil)
      stats.endToEndTTFTSeconds = endToEnd
      return LabSample(date: now.addingTimeInterval(-Double(daysAgo) * 86_400),
        model: model, modelPath: "", bits: nil, mode: "chat", runtime: "", system: "",
        hardware: "", context: 8192, temperature: 0, topK: 1, mtp: false, dflash: false, stats: stats)
    }
    func summary(_ samples: [LabSample], _ period: LabUsagePeriod = .today) -> LabResponseMeasurements {
      LabResponseMeasurements(samples: samples, period: period, now: now, calendar: utc)
    }

    let fast = sample(0.25)
    let rows = [sample(7), fast, sample(1.5), fast, sample(nil), sample(.infinity), sample(.nan),
                sample(0), sample(-2), sample(20, model: ""), sample(5, daysAgo: 6), sample(22, daysAgo: 7)]
    let today = summary(rows)
    check(today.medianFirstTokenSeconds == 1.5 && today.measuredCount == 3 && today.retainedCount == 8,
      "First-text median uses one finite, positive end-to-end measurement per valid turn and counts missing coverage")
    let week = summary(rows, .sevenDays)
    check(week.medianFirstTokenSeconds == 3.25 && week.measuredCount == 4 && week.retainedCount == 9,
      "An even-sized sample uses the midpoint and the seven-day window includes day six but excludes day seven")
    let onlyEngine = summary([sample(nil)])
    check(onlyEngine.medianFirstTokenSeconds == nil && onlyEngine.measuredCount == 0 && onlyEngine.retainedCount == 1,
      "A valid engine first-token time does not supply an end-to-end measurement")
    var moved = fast
    moved.localDay = 99
    let noneToday = summary([moved])
    check(noneToday.medianFirstTokenSeconds == nil && noneToday.retainedCount == 0
      && summary([moved], .allTime).medianFirstTokenSeconds == 0.25,
      "First-text samples use the recorded civil day even if the time zone changes")
    let huge = Double.greatestFiniteMagnitude, tiny = Double.leastNonzeroMagnitude
    check(summary([sample(huge), sample(huge)]).medianFirstTokenSeconds == huge
      && summary([sample(tiny), sample(2 * tiny)]).medianFirstTokenSeconds == 2 * tiny,
      "The midpoint stays finite for huge delays and preserves subnormal precision for tiny delays")
  }
}
