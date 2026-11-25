import Foundation

@MainActor
enum ChatTimelineCheck {
    static func run() throws {
        func check(_ condition: Bool, _ label: String) throws {
            if !condition { throw MCPPreferenceCheck.Failure(message: label) }
        }
        let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
        message.processing("Préparation du contexte")
        message.processing("Lecture · 500 tokens")
        try check(message.parts.count == 1, "Progress updates must not append blocks")
        message.append("Je cherche.", phase: "thinking")
        let firstThinkingID = message.parts.last!.id
        message.startTool(id: "exa-1", serverName: "exa", toolName: "web_search_exa")
        message.finishTool(id: "exa-1", result: "Documentation", isError: false)
        message.processing("Lecture des résultats MCP")
        message.append("Je compare.", phase: "thinking")
        message.append(" Voici.", phase: "thinking")
        message.append("La réponse.", phase: "answer")
        message.finish(stats: nil, fallbackAnswer: nil, cacheContext: "full context")
        try check(message.parts.map(\.kind) == [.processing, .thinking, .tool, .processing, .thinking, .answer],
                  "Tool rounds must remain chronological")
        try check(message.parts[1].id == firstThinkingID && message.parts[1].text == "Je cherche.",
                  "Later reasoning mutated the initial block")
        try check(message.parts[4].text == "Je compare. Voici.", "Streaming text lost")
        try check(message.parts.filter { $0.kind == .processing }.allSatisfy { $0.elapsed != nil },
                  "Processing never completed")
        let encoded = try JSONEncoder().encode(message.snapshot)
        let snapshot = try JSONDecoder().decode(ChatMessageSnapshot.self, from: encoded)
        let restored = ChatMessage(snapshot: snapshot)!
        try check(restored.parts.map(\.id) == message.parts.map(\.id), "Timeline not persisted")
        try check(restored.toolActivities[0].state == .complete, "Tool state lost on relaunch")
        var legacy = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
        legacy.removeValue(forKey: "parts")
        let oldSnapshot = try JSONDecoder().decode(ChatMessageSnapshot.self, from: JSONSerialization.data(withJSONObject: legacy))
        try check(ChatMessage(snapshot: oldSnapshot)!.content == message.content, "Legacy chat migration failed")
        // The code a tool ran is kept with it; conversations saved before decode without it.
        let agent = ChatMessage(role: .assistant, content: "", isStreaming: true)
        agent.startTool(id: "js-1", serverName: "sandbox", toolName: "javascript", arguments: "const x = 1;\nreturn x")
        agent.finishTool(id: "js-1", result: "Result:\n1", isError: false)
        let agentData = try JSONEncoder().encode(agent.snapshot)
        let agentRestored = ChatMessage(snapshot: try JSONDecoder().decode(ChatMessageSnapshot.self, from: agentData))!
        try check(agentRestored.toolActivities[0].arguments == "const x = 1;\nreturn x", "Tool code lost on relaunch")
        try check(restored.toolActivities[0].arguments == nil, "A tool without arguments gained some")
        let oldActivity = try JSONDecoder().decode(
            ToolActivity.self,
            from: Data(#"{"id":"a","serverName":"exa","toolName":"t","state":"complete","result":"r"}"#.utf8))
        try check(oldActivity.arguments == nil && oldActivity.result == "r", "Saved tool activities must still decode")
        func shown(_ json: String) throws -> String? {
            try JSONDecoder().decode(
                BridgeEvent.self,
                from: Data((#"{"type":"tool_start","request_id":"r","tool_call_id":"c","tool_name":"t""# + json + "}").utf8)
            ).arguments?.text
        }
        try check(try shown(#","arguments":{"code":"const x = 1;\nreturn x"}"#) == "const x = 1;\nreturn x", "JavaScript code shown as written")
        try check(try shown(#","arguments":{"path":"a.txt","limit":2}"#) == #"{"limit":2,"path":"a.txt"}"#, "Other arguments shown as sorted JSON")
        try check(try shown(#","arguments":{"code":"x","path":"a"}"#) == #"{"code":"x","path":"a"}"#, "Code beside other arguments stays JSON")
        try check(try shown(#","arguments":"{\"code\":\"…""#) == #"{"code":"…"#, "Truncated arguments shown as text")
        try check(try shown("") == nil && (try shown(#","arguments":null"#)) == nil, "Older engines send no arguments")
        let interrupted = ChatMessage(role: .assistant, content: "", isStreaming: true)
        interrupted.processing("Traitement")
        interrupted.fail("Arrêtée")
        try check(interrupted.parts[0].elapsed != nil && !interrupted.isStreaming, "Cancellation left a live spinner")
        try checkActivityClocks(check)
        try checkSegments(check)
        try checkToolPresentation(check)
    }

    /// Reasoning and tool parts keep how long they took; answers carry no clock.
    private static func checkActivityClocks(_ check: (Bool, String) throws -> Void) throws {
        let message = ChatMessage(role: .assistant, content: "", isStreaming: true)
        message.append("Je ", phase: "thinking")
        message.append("cherche.", phase: "thinking")
        try check(message.parts.count == 1 && message.parts[0].elapsed == nil, "Continuing reasoning closed its clock")
        message.startTool(id: "t1", serverName: "sandbox", toolName: "run", arguments: #"{"command":"true"}"#)
        try check(message.parts[0].elapsed != nil, "A tool call did not close the reasoning clock")
        try check(message.parts[1].startedAt != nil && message.parts[1].elapsed == nil, "A running tool has no live clock")
        message.finishTool(id: "t1", result: "Exit status: 0", isError: false)
        try check(message.parts[1].elapsed != nil, "A finished tool kept a running clock")
        message.append("Pense encore.", phase: "thinking")
        message.processing("Lecture des résultats MCP")
        try check(message.parts[2].elapsed != nil, "Context reading did not close the reasoning clock")
        message.append("Réponse.", phase: "answer")
        try check(message.parts.last?.startedAt == nil, "An answer part carries a clock")
        message.startTool(id: "t2", serverName: "sandbox", toolName: "run", arguments: #"{"command":"sleep 9"}"#)
        message.fail("Arrêtée")
        try check(message.toolActivities.last?.state == .failed && message.parts.last?.elapsed != nil,
                  "An interrupted tool kept a running clock")
        let finished = ChatMessage(role: .assistant, content: "", isStreaming: true)
        finished.append("Bref.", phase: "thinking")
        finished.finish(stats: nil, fallbackAnswer: nil, cacheContext: nil)
        try check(finished.parts[0].elapsed != nil, "Finishing did not close the reasoning clock")
        let data = try JSONEncoder().encode(message.snapshot)
        let restored = ChatMessage(snapshot: try JSONDecoder().decode(ChatMessageSnapshot.self, from: data))!
        try check(restored.parts.map(\.elapsed) == message.parts.map(\.elapsed), "Durations not persisted")
    }

    /// Activity parts group into one segment per run; answers stand alone; ids are stable.
    private static func checkSegments(_ check: (Bool, String) throws -> Void) throws {
        let kinds: [AssistantPart.Kind] = [.processing, .thinking, .tool, .processing, .thinking, .answer]
        let parts = kinds.map { AssistantPart(kind: $0) }
        let segments = AssistantSegment.split(parts)
        try check(segments.map(\.parts.count) == [5, 1] && segments.map(\.isActivity) == [true, false],
                  "Activity parts not grouped before the answer")
        try check(segments[0].id == parts[0].id && segments[1].id == parts[5].id, "Segment ids not taken from their first part")
        let mixed = AssistantSegment.split([.thinking, .answer, .tool, .answer, .answer].map { AssistantPart(kind: $0) })
        try check(mixed.map(\.parts.count) == [1, 1, 1, 1, 1] && mixed.map(\.isActivity) == [true, false, true, false, false],
                  "Answers must split activity runs and stand alone")
        try check(AssistantSegment.split([]).isEmpty, "No parts must give no segment")
        let growing = parts.prefix(2).map { $0 }
        try check(AssistantSegment.split(growing)[0].id == AssistantSegment.split(growing + [AssistantPart(kind: .tool)])[0].id,
                  "A growing segment changed identity (rows would lose their state)")
        // Deterministic sweep over every sequence of up to five parts: the split is a
        // partition in order, and no two activity segments are adjacent.
        let all: [AssistantPart.Kind] = [.thinking, .answer, .tool, .processing]
        var sequences: [[AssistantPart.Kind]] = [[]]
        for _ in 0..<5 { sequences += sequences.filter { $0.count == sequences.last!.count }.flatMap { s in all.map { s + [$0] } } }
        for sequence in sequences {
            let input = sequence.map { AssistantPart(kind: $0) }
            let output = AssistantSegment.split(input)
            try check(output.flatMap(\.parts).map(\.id) == input.map(\.id), "Split lost or reordered parts: \(sequence)")
            try check(output.allSatisfy { $0.isActivity ? $0.parts.allSatisfy { $0.kind != .answer } : $0.parts.count == 1 },
                      "Mixed segment: \(sequence)")
            try check(!zip(output, output.dropFirst()).contains { $0.isActivity && $1.isActivity }, "Adjacent activity segments: \(sequence)")
        }
    }

    private static func checkToolPresentation(_ check: (Bool, String) throws -> Void) throws {
        defer { AppLocalization.set(.fr) }
        func tool(_ name: String, server: String? = "sandbox", _ arguments: String?, _ state: ToolActivity.State = .complete,
                  result: String? = nil) -> ToolCallPresentation {
            ToolCallPresentation(ToolActivity(id: "x", serverName: server, toolName: name, state: state, result: result, arguments: arguments))
        }
        AppLocalization.set(.en)
        let script = tool("javascript", "// Inventory\n\n  const files = list();\nreturn files")
        try check(script.kind == .javascript && script.detail == "const files = list();" && script.monospaced,
                  "JavaScript must show its first statement")
        try check(script.title == "Ran JavaScript" && !script.detailCoversArguments && script.source == nil,
                  "JavaScript title or source wrong")
        try check(tool("javascript", #"{"code":"return 1"}"#).detail == "return 1", "JavaScript code inside JSON not read")
        try check(tool("javascript", "// only a comment").detail == nil, "A comment is not a statement")

        let read = tool("sandbox.read_file", #"{"path":"src/stats.pl"}"#)
        try check(read.kind == .readFile && read.title == "Read" && read.detail == "src/stats.pl", "File read wrong: \(read)")
        try check(read.isPath && read.monospaced && read.detailCoversArguments && read.outcome == nil, "File read layout wrong")
        try check(tool("read_file", #"{"path":"a"}"#, .running).title == "Reading", "Running tense wrong")
        try check(tool("read_file", #"{"path":"a"}"#, .failed).title == "Read file", "Failed title wrong")
        try check(tool("read_file", #"{"path":"a"}"#, .failed).outcome == "Failed", "Failed outcome missing")
        try check(tool("write_file", #"{"content":"x\ny","path":"b.txt"}"#).detail == "b.txt", "Write must name its file")
        try check(!tool("write_file", #"{"content":"x\ny","path":"b.txt"}"#).detailCoversArguments, "Written content must stay visible")
        try check(tool("list_files", "{}").detail == nil && tool("list_files", "{}").detailCoversArguments, "Listing has no object")

        // Native tools (AGENT-2026-11-25-NATIVE-TOOLS): an action and its object, like the others.
        let nativeRead = tool("sandbox.read", #"{"file_path":"src/app.py","offset":40}"#)
        try check(nativeRead.kind == .readFile && nativeRead.title == "Read" && nativeRead.detail == "src/app.py" && nativeRead.isPath,
                  "Native read wrong: \(nativeRead)")
        let nativeEdit = tool("edit", #"{"file_path":"a.py","old_string":"x = 1\ny = 2","new_string":"x = 3"}"#)
        try check(nativeEdit.kind == .editFile && nativeEdit.title == "Edited" && nativeEdit.detail == "a.py" && nativeEdit.isPath
                  && !nativeEdit.detailCoversArguments, "Native edit wrong: \(nativeEdit)")
        try check(tool("edit", #"{"file_path":"a.py"}"#, .running).title == "Editing", "Edit running tense wrong")
        try check(tool("write", #"{"file_path":"b.py","content":"x\ny"}"#).kind == .writeFile
                  && tool("write", #"{"file_path":"b.py","content":"x\ny"}"#).detail == "b.py", "Native write must name its file")
        let nativeGrep = tool("grep", #"{"path":"src","pattern":"def \\w+","include":"*.py"}"#)
        try check(nativeGrep.kind == .search && nativeGrep.title == "Searched" && nativeGrep.detail == "def \\w+" && nativeGrep.monospaced
                  && !nativeGrep.isPath, "Search must show its pattern, not its folder: \(nativeGrep)")
        let nativeGlob = tool("glob", #"{"pattern":"**/*.swift"}"#)
        try check(nativeGlob.kind == .findFiles && nativeGlob.title == "Found files" && nativeGlob.detail == "**/*.swift" && !nativeGlob.isPath,
                  "File search wrong: \(nativeGlob)")
        let nativeBash = tool("bash", #"{"command":"sh test.sh"}"#, result: "FAIL\n[exit code: 1]")
        try check(nativeBash.kind == .command && nativeBash.detail == "sh test.sh" && nativeBash.outcome == "Exit 1",
                  "Native command must show its exit code: \(nativeBash)")
        try check(tool("bash", #"{"command":"true"}"#, result: "[exit code: 0]").outcome == nil, "A passing native command shows no outcome")
        try check(tool("bash", #"{"command":"yes"}"#, result: "y\n[stopped by the time or output limit] [exit code: 143]").outcome == "Stopped",
                  "A stopped native command must say so")
        try check(tool("bash", #"{"command":"echo '[exit code: 9]'; true"}"#, result: "[exit code: 9]\n[exit code: 0]").outcome == nil,
                  "Only the last line is the command's status")
        let nativeJob = tool("job_output", #"{"job_id":3,"wait":true}"#)
        try check(nativeJob.kind == .job && nativeJob.title == "Background job" && nativeJob.detail == "#3", "Job call wrong: \(nativeJob)")
        try check(tool("job_kill", #"{"job_id":3}"#).title == "Stopped background job" && tool("job_list", "{}").detail == nil, "Job titles wrong")

        let failedRun = tool("run", #"{"command":"sh test.sh"}"#, .failed, result: "FAIL\nExit status: 1\n")
        try check(failedRun.kind == .command && failedRun.detail == "sh test.sh" && failedRun.outcome == "Exit 1",
                  "Failed command must show its exit status: \(failedRun)")
        try check(tool("run", #"{"command":"true"}"#, result: "ok\nExit status: 0").outcome == nil, "A passing command shows no outcome")
        try check(tool("run", #"{"command":"yes"}"#, .failed, result: "y\nCommand stopped: time/output limit or parent exited.").outcome == "Stopped",
                  "A stopped command must say so")
        try check(tool("run", #"{"command":"x"}"#, .failed, result: "Exit status: banana").outcome == "Failed", "Unparsed status must fall back")

        let search = tool("web_search_exa", server: "exa", #"{"numResults":5,"query":"mlx\n  prompt cache"}"#)
        try check(search.kind == .mcp && search.title == "Web search" && search.detail == "“mlx prompt cache”",
                  "MCP search wrong: \(search)")
        try check(search.source == "exa" && search.symbol == "magnifyingglass" && !search.detailCoversArguments, "MCP source wrong")
        try check(tool("exa.web_search_exa", server: "exa", nil) == tool("web_search_exa", server: "exa", nil), "Server prefix not stripped")
        let filesystem = tool("fs.read_file", server: "fs", #"{"path":"notes.md"}"#)
        try check(filesystem.kind == .readFile && filesystem.detail == "notes.md", "A prefixed MCP file tool must read as a file read")
        try check(tool("getPullRequest", server: "github", #"{"number":4}"#).title == "Get pull request", "camelCase not split")
        try check(tool("github", server: "github", nil).title == "Github", "A name made only of the server must stay readable")
        try check(tool("fetch_url", server: "web", #"{"url":"https://a.b/c"}"#).symbol == "globe", "Fetch tools use the globe")
        try check(tool("run_job", server: "my-sandbox", nil).source == nil, "Sandbox servers are not named")
        try check(tool("search", server: nil, nil).source == nil, "No server, no source")
        let long = tool("search", server: "s", "{\"query\":\"" + String(repeating: "é", count: 400) + "\"}")
        try check(long.detail?.count == ToolCallPresentation.maximumDetail + 2 && long.detail?.hasSuffix("…”") == true,
                  "Long queries must be clipped: \(long.detail?.count ?? -1)")
        let truncated = tool("search", server: "s", #"{"query":"abc…"#)
        try check(truncated.detail == #"{"query":"abc…"# && !truncated.detailCoversArguments, "Truncated arguments must still show")
        try check(tool("invalid_tool_call", server: nil, nil).kind == .unreadable
                  && tool("invalid_tool_call", server: nil, nil).title == "Unreadable tool call", "Unreadable call wrong")

        func input(_ name: String, server: String? = "sandbox", _ arguments: String?) -> (title: String, text: String)? {
            let activity = ToolActivity(id: "x", serverName: server, toolName: name, state: .complete, result: nil, arguments: arguments)
            return ToolCallPresentation(activity).input(of: activity)
        }
        try check(input("read_file", #"{"path":"a"}"#) == nil, "A single argument shown on the line must not repeat")
        try check(input("list_files", "{}") == nil && input("run", nil) == nil, "No arguments, no input section")
        try check(input("write", #"{"content":"line 1\nline 2","file_path":"b"}"#).map { [$0.title, $0.text] } == ["Content", "line 1\nline 2"],
                  "A native write must show its content as text")
        try check(input("edit", #"{"file_path":"a","old_string":"x = 1\ny = \"2\"","new_string":"x = 3"}"#).map { [$0.title, $0.text] }
                  == ["Change", "- x = 1\n- y = \"2\"\n+ x = 3"], "An edit must show what it replaces and with what")
        try check(input("read", #"{"file_path":"a"}"#) == nil, "A read shown on its line has no input section")
        try check(input("write_file", #"{"content":"line 1\nline 2","path":"b"}"#).map { [$0.title, $0.text] } == ["Content", "line 1\nline 2"],
                  "Written content must read as text, not escaped JSON")
        try check(input("javascript", "const a = 1;\nreturn a").map { [$0.title, $0.text] } == ["Code", "const a = 1;\nreturn a"],
                  "JavaScript must show as written")
        try check(input("javascript", #"{"code":"return 2"}"#)?.text == "return 2", "JavaScript inside JSON must show its code")
        try check(input("javascript", #"{"code":"write('a.py', text)","text":"x = 1\n"}"#)?.text == "write('a.py', text)\n\n— text —\nx = 1\n",
                  "A JavaScript text argument must follow its code")
        try check(input("javascript", #"{"code":"x","text":"t","path":"a"}"#)?.text == #"{"code":"x","text":"t","path":"a"}"#,
                  "Code and text beside other arguments stay as sent")
        try check(input("search", server: "exa", #"{"numResults":5,"query":"a/b"}"#)?.text == "{\n  \"numResults\" : 5,\n  \"query\" : \"a/b\"\n}",
                  "Other arguments must be indented JSON with sorted keys and plain slashes")
        try check(input("search", server: "exa", #"{"query":"abc…"#)?.text == #"{"query":"abc…"#, "Truncated arguments must show as received")

        AppLocalization.set(.fr)
        try check(tool("read_file", #"{"path":"a"}"#).title == "Fichier lu", "French done title wrong")
        try check(tool("run", #"{"command":"x"}"#, .failed, result: "Exit status: 2").outcome == "Code 2", "French exit wrong")
        try check(tool("javascript", "x", .running).title == "Exécution du JavaScript", "French running title wrong")
    }
}
