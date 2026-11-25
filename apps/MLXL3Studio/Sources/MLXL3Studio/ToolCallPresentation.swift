import Foundation

/// How a tool call reads in the transcript: an action and its object ("Read src/stats.pl",
/// "Ran sh test.sh") instead of the raw function name, its server and a status badge.
/// Pure: built from the saved `ToolActivity`, so restored conversations read the same.
struct ToolCallPresentation: Equatable {
    enum Kind: Equatable { case javascript, command, readFile, writeFile, editFile, search, findFiles, job, listFiles, unreadable, mcp }

    let kind: Kind
    /// SF Symbol of the action.
    let symbol: String
    /// The action, in the tense of the call's state.
    let title: String
    /// What the action applies to (path, command, query, first line of code); nil when none.
    let detail: String?
    /// The detail is code, a command or a path.
    let monospaced: Bool
    /// The detail is a path: truncated in the middle so the file name stays visible.
    let isPath: Bool
    /// MCP server named after the action; nil for the Agent sandbox and unnamed servers.
    let source: String?
    /// The detail already shows every argument: the expanded view need not repeat them.
    let detailCoversArguments: Bool
    /// Short outcome shown beside a finished call ("Exit 1", "Failed").
    let outcome: String?

    static let maximumDetail = 160

    init(_ activity: ToolActivity) {
        let name = Self.bareName(activity.toolName, server: activity.serverName)
        let kind = Self.kind(of: name)
        let state = activity.state
        self.kind = kind
        let arguments = Self.arguments(activity.arguments)
        let code = kind == .javascript ? (arguments.object?["code"] as? String ?? arguments.text) : nil

        var detail: String?
        var key: String?
        switch kind {
        case .javascript:
            detail = code.flatMap(Self.firstCodeLine)
        case .unreadable, .listFiles:
            detail = nil
        case .search, .findFiles:
            // What is looked for, not where: `path` only narrows the search.
            if let pattern = arguments.object?["pattern"] as? String, !Self.oneLine(pattern).isEmpty {
                (key, detail) = ("pattern", Self.clipped(Self.oneLine(pattern)))
            } else {
                (key, detail) = Self.primaryArgument(arguments.object, fallback: arguments.text)
            }
        case .job:
            detail = (arguments.object?["job_id"]).flatMap { $0 is NSNull ? nil : "#\($0)" }
        default:
            (key, detail) = Self.primaryArgument(arguments.object, fallback: arguments.text)
        }
        if let value = detail, Self.quotedKeys.contains(key ?? "") {
            detail = "“\(value)”"
        }
        self.detail = detail
        isPath = ["path", "file", "filename", "file_path", "directory", "dir"].contains(key ?? "")
            || kind == .readFile || kind == .writeFile || kind == .editFile
        monospaced = kind == .javascript || kind == .command || kind == .search || kind == .findFiles || isPath
        let covered: Bool = {
            guard let object = arguments.object else { return arguments.text == nil }
            guard object.count == 1, let only = object.values.first as? String else { return object.isEmpty }
            return Self.oneLine(only).count <= Self.maximumDetail && !only.contains("\n")
        }()
        detailCoversArguments = kind != .javascript && covered

        let server = activity.serverName.map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
        source = kind == .mcp && !server.isEmpty && !server.lowercased().contains("sandbox") ? server : nil

        let running = state == .running
        let failed = state == .failed
        switch kind {
        case .javascript:
            symbol = "curlybraces"
            title = running ? L("Exécution du JavaScript", "Running JavaScript")
                : failed ? "JavaScript" : L("JavaScript exécuté", "Ran JavaScript")
        case .command:
            symbol = "terminal"
            title = running ? L("Exécution", "Running")
                : failed ? L("Commande", "Command") : L("Commande exécutée", "Ran")
        case .readFile:
            symbol = "doc.text"
            title = running ? L("Lecture", "Reading") : failed ? L("Lecture", "Read file") : L("Fichier lu", "Read")
        case .writeFile:
            symbol = "square.and.pencil"
            title = running ? L("Écriture", "Writing") : failed ? L("Écriture", "Write file") : L("Fichier écrit", "Wrote")
        case .editFile:
            symbol = "pencil.line"
            title = running ? L("Modification", "Editing") : failed ? L("Modification", "Edit file") : L("Fichier modifié", "Edited")
        case .search:
            symbol = "magnifyingglass"
            title = running ? L("Recherche", "Searching") : failed ? L("Recherche", "Search") : L("Recherche faite", "Searched")
        case .findFiles:
            symbol = "folder"
            title = running ? L("Recherche de fichiers", "Finding files")
                : failed ? L("Recherche de fichiers", "Find files") : L("Fichiers trouvés", "Found files")
        case .job:
            symbol = "clock.arrow.circlepath"
            title = name == "job_kill" ? L("Tâche de fond arrêtée", "Stopped background job")
                : name == "job_list" ? L("Tâches de fond", "Background jobs") : L("Tâche de fond", "Background job")
        case .listFiles:
            symbol = "folder"
            title = running ? L("Liste des fichiers", "Listing files")
                : failed ? L("Liste des fichiers", "List files") : L("Fichiers listés", "Listed files")
        case .unreadable:
            symbol = "exclamationmark.triangle"
            title = L("Appel d’outil illisible", "Unreadable tool call")
        case .mcp:
            let lowered = name.lowercased()
            symbol = lowered.contains("search") ? "magnifyingglass"
                : ["fetch", "url", "browse", "crawl", "web"].contains(where: lowered.contains) ? "globe"
                : "wrench.and.screwdriver"
            title = Self.humanized(name, server: activity.serverName)
        }
        outcome = state == .running ? nil : Self.outcome(kind: kind, failed: failed, result: activity.result)
    }

    /// The arguments worth showing under the line: JavaScript as written, a written file's
    /// content as text, other JSON indented; nil when the line already shows the only argument.
    func input(of activity: ToolActivity) -> (title: String, text: String)? {
        guard let text = activity.arguments, !text.isEmpty, !detailCoversArguments else { return nil }
        let object = Self.arguments(text).object
        switch kind {
        case .javascript:
            if let code = object?["code"] as? String, object?.count == 1 { return (L("Code", "Code"), code) }
            // The `text` argument (a file's content) follows the code that uses it.
            if let code = object?["code"] as? String, let content = object?["text"] as? String, object?.count == 2 {
                return (L("Code", "Code"), code + "\n\n— text —\n" + content)
            }
            return (L("Code", "Code"), text)
        case .writeFile:
            if let content = object?["content"] as? String { return (L("Contenu", "Content"), content) }
        case .editFile:
            // The replaced text, then its replacement, as written: no JSON escapes to read through.
            if let old = object?["old_string"] as? String, let new = object?["new_string"] as? String {
                return (L("Modification", "Change"), "- " + old.replacingOccurrences(of: "\n", with: "\n- ")
                    + "\n+ " + new.replacingOccurrences(of: "\n", with: "\n+ "))
            }
        default:
            break
        }
        if let object,
           let pretty = try? JSONSerialization.data(
               withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]),
           let indented = String(data: pretty, encoding: .utf8) {
            return (L("Entrée", "Input"), indented)
        }
        return (L("Entrée", "Input"), text)
    }

    // MARK: - Names

    /// `exa.web_search_exa` → `web_search_exa`: the engine may prefix the public name with its server.
    static func bareName(_ name: String, server: String?) -> String {
        if let server, !server.isEmpty, name.hasPrefix(server + "."), name.count > server.count + 1 {
            return String(name.dropFirst(server.count + 1))
        }
        if let dot = name.lastIndex(of: "."), name.index(after: dot) < name.endIndex,
           ["sandbox", "workspace"].contains(name[..<dot].lowercased()) {
            return String(name[name.index(after: dot)...])
        }
        return name
    }

    static func kind(of name: String) -> Kind {
        switch name {
        case "javascript": .javascript
        case "run", "run_command", "shell", "bash": .command
        case "read_file", "read_text_file", "read": .readFile
        case "write_file", "write": .writeFile
        case "edit": .editFile
        case "grep": .search
        case "glob": .findFiles
        case "job_output", "job_list", "job_kill": .job
        case "list_files", "list_directory": .listFiles
        case "invalid_tool_call": .unreadable
        default: .mcp
        }
    }

    /// `web_search_exa` from server `exa` → "Web search"; camelCase and dashes split too.
    static func humanized(_ name: String, server: String?) -> String {
        var words: [String] = []
        var current = ""
        var previous: Character?
        for character in name {
            if character == "_" || character == "-" || character == "." || character == " " {
                if !current.isEmpty { words.append(current) }
                current = ""
            } else if character.isUppercase, let before = previous, before.isLowercase {
                words.append(current)
                current = String(character)
            } else {
                current.append(character)
            }
            previous = character
        }
        if !current.isEmpty { words.append(current) }
        let serverWords = Set((server ?? "").lowercased().split(whereSeparator: { "_-. ".contains($0) }).map(String.init))
        let kept = words.filter { !serverWords.contains($0.lowercased()) }
        let sentence = (kept.isEmpty ? words : kept).map { $0.lowercased() }.joined(separator: " ")
        guard !sentence.isEmpty else { return name.isEmpty ? L("Outil", "Tool") : name }
        return sentence.prefix(1).uppercased() + sentence.dropFirst()
    }

    // MARK: - Arguments

    struct Arguments {
        let object: [String: Any]?
        /// The arguments when they are not a JSON object (JavaScript code, a truncated prefix).
        let text: String?
    }

    static func arguments(_ text: String?) -> Arguments {
        guard let text, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return Arguments(object: nil, text: nil)
        }
        if text.first == "{", let data = text.data(using: .utf8),
           let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return Arguments(object: object, text: nil)
        }
        return Arguments(object: nil, text: text)
    }

    /// Keys whose value is the object of the action, most specific first.
    static let primaryKeys = [
        "path", "file_path", "file", "filename", "command", "cmd", "query", "q", "search", "url", "uri",
        "pattern", "name", "prompt", "text", "input", "directory", "dir",
    ]
    static let quotedKeys: Set<String> = ["query", "q", "search", "prompt", "text", "input"]

    static func primaryArgument(_ object: [String: Any]?, fallback: String?) -> (key: String?, value: String?) {
        guard let object else { return (nil, fallback.map(oneLine).map(clipped)) }
        for key in primaryKeys {
            if let value = object[key] as? String, !oneLine(value).isEmpty { return (key, clipped(oneLine(value))) }
        }
        for key in object.keys.sorted() {
            if let value = object[key] as? String, !oneLine(value).isEmpty { return (key, clipped(oneLine(value))) }
        }
        return (nil, nil)
    }

    /// The first line of code that says something: blank lines and `//` comments are skipped.
    static func firstCodeLine(_ code: String) -> String? {
        for line in code.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("//") { continue }
            return clipped(oneLine(trimmed))
        }
        return nil
    }

    static func oneLine(_ text: String) -> String {
        text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).joined(separator: " ")
    }

    static func clipped(_ text: String) -> String {
        text.count > maximumDetail ? String(text.prefix(maximumDetail - 1)) + "…" : text
    }

    // MARK: - Outcome

    /// A command's exit status when it failed ("Exit 1"), otherwise "Failed" for a failed call.
    static func outcome(kind: Kind, failed: Bool, result: String?) -> String? {
        if kind == .command, let result {
            let last = result.split(whereSeparator: \.isNewline).last.map { String($0).trimmingCharacters(in: .whitespaces) } ?? ""
            if last.hasPrefix("Exit status: "), let code = Int(last.dropFirst("Exit status: ".count)), code != 0 {
                return L("Code \(code)", "Exit \(code)")
            }
            if last.hasPrefix("Command stopped") { return L("Arrêtée", "Stopped") }
            // The native `bash` tool ends its result with `[exit code: N]`.
            if last.hasSuffix("]"), let mark = last.range(of: "[exit code: ", options: .backwards),
               let code = Int(last[mark.upperBound...].dropLast()), code != 0 {
                return last.contains("[stopped by") ? L("Arrêtée", "Stopped") : L("Code \(code)", "Exit \(code)")
            }
        }
        return failed ? L("Échec", "Failed") : nil
    }
}
