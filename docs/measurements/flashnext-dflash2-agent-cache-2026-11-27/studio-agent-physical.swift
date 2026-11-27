import Foundation

// This extension is compiled with StudioModel.swift to observe the same bridge events that Studio handles.
extension StudioModel {
    fileprivate func observeInventoryTrial(_ observe: @escaping @MainActor (BridgeEvent) -> Void) {
        let handle = bridge.onEvent
        bridge.onEvent = { event in
            handle?(event)
            observe(event)
        }
    }

    fileprivate func stopInventoryTrial() async {
        stopServices()
        await bridge.waitForRetiringProcesses()
    }
}

@main struct StudioAgentPhysical {
    private struct TrialError: LocalizedError {
        let errorDescription: String?
        init(_ message: String) { errorDescription = message }
    }

    private struct Registry: Decodable { let models: [String: LocalModel] }

    private static let records = [("stock.txt", "17"), ("returns.txt", "5"), ("dispatch.txt", "4")]
    private static let question = "Calculate the units currently in stock. Read stock.txt, returns.txt, then dispatch.txt via sandbox.read with one file per call in this order. Add the returned units to stock and subtract the dispatched units. When all three records have been read, answer only TOTAL=<number>."

    @MainActor static func main() async {
        let args = CommandLine.arguments
        // The real sandbox server starts this executable as a child, just as it starts Studio.
        if args.count == 10, args[1] == "--agent-js" { AgentJavaScript.supervisorMode(arguments: args) }
        if args.count == 4, args[1] == "--agent-js-kernel" { AgentJavaScript.kernelMode(arguments: args) }
        if (3...5).contains(args.count), args[1] == "--agent-tools" {
            AgentSandbox.serve(root: args[2], permissionsPath: args.count >= 4 ? args[3] : nil,
                               longCalls: args.count == 5 && args[4] == "long-calls")
        }
        guard args.count == 6 else {
            print("Usage: studio-agent-physical <engine> <registry> <model-name> <draft> <report.json>")
            exit(EXIT_FAILURE)
        }
        let output = URL(fileURLWithPath: args[5])
        var report: [String: Any] = ["status": "failed", "source": "StudioModel.send", "model": args[3]]
        do {
            let registry = try JSONDecoder().decode(Registry.self, from: Data(contentsOf: URL(fileURLWithPath: args[2])))
            guard let model = registry.models[args[3]], model.modelType == "qwen4_exp" else {
                throw TrialError("Registry must contain a Flash Next checkpoint")
            }
            try await run(engine: args[1], model: model, draft: args[4], report: &report)
            report["status"] = "complete"
        } catch {
            report["error"] = error.localizedDescription
        }
        do {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try (data + Data([10])).write(to: output, options: .atomic)
            print(String(decoding: data, as: UTF8.self))
        } catch {
            print("Could not save physical Agent result: \(error.localizedDescription)")
            exit(EXIT_FAILURE)
        }
        if report["status"] as? String != "complete" { exit(EXIT_FAILURE) }
    }

    @MainActor private static func wait(
        _ label: String, seconds: Double, studio: StudioModel, until ready: () -> Bool
    ) async throws {
        let deadline = ProcessInfo.processInfo.systemUptime + seconds
        while ProcessInfo.processInfo.systemUptime < deadline {
            if case let .failed(message) = studio.engineState { throw TrialError("\(label): \(message)") }
            if ready() { return }
            try await Task.sleep(for: .milliseconds(50))
        }
        throw TrialError("\(label) timed out: \(studio.sendBlockReason ?? studio.engineState.label)")
    }

    @MainActor private static func run(
        engine: String, model: LocalModel, draft: String, report: inout [String: Any]
    ) async throws {
        let files = FileManager.default
        let root = files.temporaryDirectory.resolvingSymlinksInPath()
            .appendingPathComponent("mlxl3-studio-dflash-\(UUID())", isDirectory: true)
        try files.createDirectory(at: root, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        defer { try? files.removeItem(at: root) }
        setenv("CFFIXED_USER_HOME", root.path, 1)
        setenv("MLXL3_HOME", root.path, 1)
        setenv("MLXL3_EXECUTABLE", URL(fileURLWithPath: engine).path, 1)
        let suite = "io.mlxl3.agent-physical.\(UUID())"
        guard let preferences = UserDefaults(suiteName: suite) else { throw TrialError("Cannot isolate preferences") }
        defer { preferences.removePersistentDomain(forName: suite) }
        preferences.set("en", forKey: "studio.language")
        preferences.set([model.name: 4096], forKey: "studio.contextLengths")
        preferences.set(true, forKey: "studio.dflash2Enabled")
        let draftPath = URL(fileURLWithPath: draft).path
        preferences.set(draftPath, forKey: "studio.dflashDraftPath")
        preferences.set(draftPath, forKey: "studio.dflashDraft.\(model.path)")

        let studio = StudioModel(conversationFileURL: root.appendingPathComponent("history.json"),
                                 preferences: preferences, unified: true)
        do {
            try await task(studio, model: model, report: &report)
        } catch {
            await studio.stopInventoryTrial()
            throw error
        }
        await studio.stopInventoryTrial()
    }

    @MainActor private static func task(
        _ studio: StudioModel, model: LocalModel, report: inout [String: Any]
    ) async throws {
        var ready: BridgeEvent?
        var mode: BridgeEvent?
        var finish: BridgeEvent?
        studio.observeInventoryTrial { event in
            switch event.type {
            case "ready": ready = event
            case "generation_mode": mode = event
            case "complete": finish = event
            default: break
            }
        }
        studio.selectMode(.agent)
        guard let workspace = studio.agentWorkspaceURL, let store = studio.agentStore else {
            throw TrialError("The real Agent task has no workspace")
        }
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        for (path, units) in records {
            try Data("units=\(units)\n".utf8).write(to: workspace.appendingPathComponent(path))
        }
        try store.ensureState()
        var permissions = AgentPermissions()
        permissions.write = false
        permissions.run = false
        permissions.internet = false
        permissions.internetHosts = []
        try store.setPermissions(permissions)
        studio.models = [model]
        studio.selectModel(model.name)
        studio.systemPrompt = "You are an inventory assistant working with local records. Use relative file paths and report the result."
        studio.draft = question
        try await wait("Agent and resident draft readiness", seconds: 180, studio: studio) {
            studio.dflashDownloadError != nil || (studio.canSend && studio.dflashActive == true)
        }
        if let problem = studio.dflashDownloadError { throw TrialError(problem) }
        report["capability"] = ready?.flashDFlashLivePromptCacheSupported ?? false
        report["context"] = ready?.contextLimit
        report["engine_revision"] = ready?.runtimeCommit
        report["mlx_version"] = ready?.mlxVersion
        guard ready?.flashDFlashLivePromptCacheSupported == true, ready?.contextLimit == 4096 else {
            throw TrialError("Engine did not announce Flash DFlash2 Agent live cache at context 4096")
        }
        studio.setReasoningLevel(.none)
        studio.setDFlashMode(2)
        let started = ProcessInfo.processInfo.systemUptime
        studio.send()
        try await wait("Agent generation", seconds: 900, studio: studio) {
            studio.currentConversation?.messages.last?.role == .assistant
                && !studio.isGenerating && !studio.agentStudioBusy
        }
        report["wall_seconds"] = ProcessInfo.processInfo.systemUptime - started
        report["dflash_mode"] = mode?.dflashMode
        report["dflash_width"] = mode?.dflashProposals
        report["stop_reason"] = finish?.stopReason
        guard let message = studio.currentConversation?.messages.last else {
            throw TrialError("Assistant message was not rendered")
        }
        report["answer"] = message.content
        report["tool_activities"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(message.toolActivities))
        guard message.error == nil, let stats = message.stats else {
            throw TrialError(message.error ?? "Assistant message has no native generation stats")
        }
        report["stats"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(stats))
        guard mode?.dflashActive == true, mode?.dflashMode == 2, mode?.dflashProposals == 2, mode?.mtpActive == false,
              (stats.dflashAcceptedTokens ?? 0) > 0,
              (stats.cachedPromptTokens ?? 0) > 0, (stats.evaluatedPromptTokens ?? 0) > 0,
              (stats.memory?.processLifetimePeakBytes ?? 0) > 0,
              finish?.stopReason == "eos", finish?.contextFull == false,
              stats.toolRounds == records.count, message.toolActivities.count == records.count,
              message.content.trimmingCharacters(in: .whitespacesAndNewlines) == "TOTAL=18" else {
            throw TrialError("Agent answer, stop, draft, cache, memory or tool telemetry is invalid")
        }
        for (activity, (path, units)) in zip(message.toolActivities, records) {
            let arguments = try JSONSerialization.jsonObject(with: Data((activity.arguments ?? "{}").utf8))
                as? [String: Any]
            guard activity.toolName == "sandbox.read", activity.state == .complete,
                  arguments?["file_path"] as? String == path,
                  activity.result?.contains("units=\(units)") == true else {
                throw TrialError("Agent did not read and return \(path) in order")
            }
        }
    }
}
