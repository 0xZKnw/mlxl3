import AppKit
import SwiftMath
import SwiftUI

private enum MarkdownBlockKind {
    case paragraph(String)
    case heading(level: Int, text: String)
    case unordered(indent: Int, text: String)
    case ordered(indent: Int, number: String, text: String)
    case quote(String)
    case code(language: String?, text: String)
    case math(String)
    case table(header: [String], alignments: [MarkdownTableAlignment], rows: [[String]])
    case rule
}

private enum MarkdownTableAlignment {
    case leading
    case center
    case trailing
}

private struct MarkdownBlock: Identifiable {
    let id: Int
    let kind: MarkdownBlockKind
}

private struct OpenMathBlock {
    let opener: String
    let closer: String
    var lines: [String]
}

private enum MarkdownParser {
    static func parse(_ source: String) -> [MarkdownBlock] {
        var kinds: [MarkdownBlockKind] = []
        var paragraph: [String] = []
        var codeLines: [String]?
        var codeLanguage: String?
        var mathBlock: OpenMathBlock?

        func flushParagraph() {
            guard !paragraph.isEmpty else { return }
            kinds.append(.paragraph(paragraph.joined(separator: " ")))
            paragraph.removeAll(keepingCapacity: true)
        }

        let lines = source.components(separatedBy: .newlines)
        var lineIndex = 0
        while lineIndex < lines.count {
            let rawLine = lines[lineIndex]
            let trimmed = rawLine.trimmingCharacters(in: .whitespaces)

            if codeLines != nil {
                if trimmed.hasPrefix("```") {
                    kinds.append(.code(language: codeLanguage, text: codeLines!.joined(separator: "\n")))
                    codeLines = nil
                    codeLanguage = nil
                } else {
                    codeLines?.append(rawLine)
                }
                lineIndex += 1
                continue
            }

            if var openMath = mathBlock {
                if let closerRange = trimmed.range(of: openMath.closer) {
                    let prefix = String(trimmed[..<closerRange.lowerBound])
                    if !prefix.isEmpty {
                        openMath.lines.append(prefix)
                    }
                    kinds.append(.math(openMath.lines.joined(separator: "\n")))
                    mathBlock = nil

                    let suffix = String(trimmed[closerRange.upperBound...])
                        .trimmingCharacters(in: .whitespaces)
                    if !suffix.isEmpty {
                        paragraph.append(suffix)
                    }
                } else {
                    openMath.lines.append(rawLine)
                    mathBlock = openMath
                }
                lineIndex += 1
                continue
            }

            if lineIndex + 1 < lines.count,
               let table = table(
                   headerLine: rawLine,
                   delimiterLine: lines[lineIndex + 1]
               ) {
                flushParagraph()
                var rows: [[String]] = []
                lineIndex += 2
                while lineIndex < lines.count,
                      let row = tableRow(from: lines[lineIndex]),
                      !lines[lineIndex].trimmingCharacters(in: .whitespaces).isEmpty {
                    rows.append(normalized(row, count: table.header.count))
                    lineIndex += 1
                }
                kinds.append(
                    .table(
                        header: table.header,
                        alignments: table.alignments,
                        rows: rows
                    )
                )
                continue
            }

            if trimmed.hasPrefix("```") {
                flushParagraph()
                let language = String(trimmed.dropFirst(3)).trimmingCharacters(in: .whitespaces)
                codeLanguage = language.isEmpty ? nil : language
                codeLines = []
                lineIndex += 1
                continue
            }
            if let opened = openMath(from: trimmed) {
                flushParagraph()
                if let closerRange = opened.remainder.range(of: opened.closer) {
                    let formula = opened.remainder[..<closerRange.lowerBound]
                        .trimmingCharacters(in: .whitespaces)
                    if !formula.isEmpty {
                        kinds.append(.math(formula))
                    }
                    let suffix = opened.remainder[closerRange.upperBound...]
                        .trimmingCharacters(in: .whitespaces)
                    if !suffix.isEmpty {
                        paragraph.append(suffix)
                    }
                } else {
                    let firstLine = opened.remainder.trimmingCharacters(in: .whitespaces)
                    mathBlock = OpenMathBlock(
                        opener: opened.opener,
                        closer: opened.closer,
                        lines: firstLine.isEmpty ? [] : [firstLine]
                    )
                }
                lineIndex += 1
                continue
            }
            if trimmed.isEmpty {
                flushParagraph()
                lineIndex += 1
                continue
            }
            if isRule(trimmed) {
                flushParagraph()
                kinds.append(.rule)
                lineIndex += 1
                continue
            }
            if let heading = heading(from: trimmed) {
                flushParagraph()
                kinds.append(.heading(level: heading.level, text: heading.text))
                lineIndex += 1
                continue
            }
            if let listItem = unorderedItem(from: rawLine) {
                flushParagraph()
                kinds.append(.unordered(indent: listItem.indent, text: listItem.text))
                lineIndex += 1
                continue
            }
            if let listItem = orderedItem(from: rawLine) {
                flushParagraph()
                kinds.append(
                    .ordered(
                        indent: listItem.indent,
                        number: listItem.number,
                        text: listItem.text
                    )
                )
                lineIndex += 1
                continue
            }
            if trimmed.hasPrefix(">") {
                flushParagraph()
                let quote = trimmed.dropFirst().trimmingCharacters(in: .whitespaces)
                kinds.append(.quote(quote))
                lineIndex += 1
                continue
            }
            paragraph.append(trimmed)
            lineIndex += 1
        }

        if let codeLines {
            kinds.append(.code(language: codeLanguage, text: codeLines.joined(separator: "\n")))
        }
        if let mathBlock {
            let raw = ([mathBlock.opener] + mathBlock.lines).joined(separator: "\n")
            kinds.append(.paragraph(raw))
        }
        flushParagraph()
        return kinds.enumerated().map { MarkdownBlock(id: $0.offset, kind: $0.element) }
    }

    private static func openMath(
        from line: String
    ) -> (opener: String, closer: String, remainder: String)? {
        if line.hasPrefix("$$") {
            return ("$$", "$$", String(line.dropFirst(2)))
        }
        if line.hasPrefix(#"\["#) {
            return (#"\["#, #"\]"#, String(line.dropFirst(2)))
        }
        return nil
    }

    private static func isRule(_ line: String) -> Bool {
        let compact = line.filter { !$0.isWhitespace }
        guard compact.count >= 3, let first = compact.first, first == "-" || first == "*" else {
            return false
        }
        return compact.allSatisfy { $0 == first }
    }

    private static func heading(from line: String) -> (level: Int, text: String)? {
        let level = line.prefix(while: { $0 == "#" }).count
        guard (1...6).contains(level) else { return nil }
        let remainder = line.dropFirst(level)
        guard remainder.first?.isWhitespace == true else { return nil }
        return (level, remainder.trimmingCharacters(in: .whitespaces))
    }

    private static func indentation(of line: String) -> (count: Int, remainder: Substring) {
        let whitespace = line.prefix(while: { $0 == " " || $0 == "\t" })
        let spaces = whitespace.reduce(into: 0) { count, character in
            count += character == "\t" ? 4 : 1
        }
        return (spaces / 2, line.dropFirst(whitespace.count))
    }

    private static func unorderedItem(from line: String) -> (indent: Int, text: String)? {
        let parsed = indentation(of: line)
        guard parsed.remainder.count >= 2,
              let marker = parsed.remainder.first,
              ["-", "*", "+"].contains(marker),
              parsed.remainder.dropFirst().first?.isWhitespace == true
        else { return nil }
        return (
            parsed.count,
            parsed.remainder.dropFirst().trimmingCharacters(in: .whitespaces)
        )
    }

    private static func orderedItem(
        from line: String
    ) -> (indent: Int, number: String, text: String)? {
        let parsed = indentation(of: line)
        let digits = parsed.remainder.prefix(while: \.isNumber)
        guard !digits.isEmpty else { return nil }
        let afterDigits = parsed.remainder.dropFirst(digits.count)
        guard afterDigits.hasPrefix(". ") else { return nil }
        return (
            parsed.count,
            String(digits),
            afterDigits.dropFirst(2).trimmingCharacters(in: .whitespaces)
        )
    }

    private static func table(
        headerLine: String,
        delimiterLine: String
    ) -> (header: [String], alignments: [MarkdownTableAlignment])? {
        guard let header = tableRow(from: headerLine),
              let delimiters = tableRow(from: delimiterLine),
              header.count == delimiters.count,
              header.count >= 2
        else { return nil }

        var alignments: [MarkdownTableAlignment] = []
        for rawDelimiter in delimiters {
            let delimiter = rawDelimiter.trimmingCharacters(in: .whitespaces)
            let core = delimiter.trimmingCharacters(in: CharacterSet(charactersIn: ":"))
            guard core.count >= 3, core.allSatisfy({ $0 == "-" }) else { return nil }
            if delimiter.hasPrefix(":"), delimiter.hasSuffix(":") {
                alignments.append(.center)
            } else if delimiter.hasSuffix(":") {
                alignments.append(.trailing)
            } else {
                alignments.append(.leading)
            }
        }
        return (normalized(header, count: delimiters.count), alignments)
    }

    private static func tableRow(from line: String) -> [String]? {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.contains("|") else { return nil }

        var cells: [String] = []
        var current = ""
        var escaped = false
        var insideCode = false
        for character in trimmed {
            if escaped {
                current.append(character)
                escaped = false
            } else if character == "\\" {
                current.append(character)
                escaped = true
            } else if character == "`" {
                insideCode.toggle()
                current.append(character)
            } else if character == "|", !insideCode {
                cells.append(current.trimmingCharacters(in: .whitespaces))
                current = ""
            } else {
                current.append(character)
            }
        }
        cells.append(current.trimmingCharacters(in: .whitespaces))

        if trimmed.hasPrefix("|") { cells.removeFirst() }
        if trimmed.hasSuffix("|") { cells.removeLast() }
        return cells.count >= 2 ? cells : nil
    }

    private static func normalized(_ cells: [String], count: Int) -> [String] {
        Array((cells + Array(repeating: "", count: max(0, count - cells.count))).prefix(count))
    }
}

struct StreamingTextChunk: Identifiable, Equatable {
    let id: Int
    let source: String
}

enum StreamingTextChunker {
    static let targetCharacters = 6_000
    static let hardLimitCharacters = 16_000

    static func chunks(_ source: String) -> [StreamingTextChunk] {
        guard source.count > targetCharacters else {
            return [StreamingTextChunk(id: 0, source: source)]
        }

        let lines = source.split(separator: "\n", omittingEmptySubsequences: false)
        var chunks: [StreamingTextChunk] = []
        var current = ""
        var byteOffset = 0
        var insideCodeFence = false

        func flush() {
            guard !current.isEmpty else { return }
            chunks.append(StreamingTextChunk(id: byteOffset, source: current))
            byteOffset += current.utf8.count
            current = ""
        }

        for (index, line) in lines.enumerated() {
            let renderedLine = String(line) + (index == lines.count - 1 ? "" : "\n")
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            current += renderedLine

            if trimmed.hasPrefix("```") {
                insideCodeFence.toggle()
            }
            let isBlankBoundary = trimmed.isEmpty
            if !insideCodeFence,
               (current.count >= hardLimitCharacters
                || (current.count >= targetCharacters && isBlankBoundary)) {
                flush()
            }
        }
        flush()
        return chunks.isEmpty ? [StreamingTextChunk(id: 0, source: source)] : chunks
    }
}

struct MarkdownResponseView: View {
    private let source: String
    private let streaming: Bool

    init(_ source: String, streaming: Bool = false) {
        self.source = source
        self.streaming = streaming
    }

    var body: some View {
        let chunks = StreamingTextChunker.chunks(source)
        VStack(alignment: .leading, spacing: 10) {
            ForEach(chunks) { chunk in
                MarkdownChunkView(
                    source: chunk.source,
                    streaming: streaming && chunk.id == chunks.last?.id
                )
                .equatable()
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
    }
}

private struct MarkdownChunkView: View, Equatable {
    let source: String
    let streaming: Bool

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.streaming == rhs.streaming
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(MarkdownParser.parse(source)) { block in
                blockView(block.kind)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func blockView(_ block: MarkdownBlockKind) -> some View {
        switch block {
        case let .paragraph(text):
            animatedText(text, size: 15, weight: .regular)
                .lineSpacing(5)
        case let .heading(level, text):
            animatedText(
                text,
                size: headingSize(level),
                weight: level <= 2 ? .bold : .semibold
            )
            .tracking(level <= 2 ? -0.3 : -0.1)
            .padding(.top, level <= 2 ? 8 : 4)
        case let .unordered(indent, text):
            HStack(alignment: .firstTextBaseline, spacing: 9) {
                Circle()
                    .fill(Color.white.opacity(indent == 0 ? 0.72 : 0.42))
                    .frame(width: indent == 0 ? 4.5 : 3.5, height: indent == 0 ? 4.5 : 3.5)
                animatedText(text, size: 15, weight: .regular)
                    .lineSpacing(4)
            }
            .padding(.leading, CGFloat(indent) * 18 + 3)
        case let .ordered(indent, number, text):
            HStack(alignment: .firstTextBaseline, spacing: 9) {
                Text(number + ".")
                    .font(.system(size: 12, weight: .bold, design: .rounded))
                    .foregroundStyle(Color.white.opacity(0.48))
                    .frame(minWidth: 18, alignment: .trailing)
                animatedText(text, size: 15, weight: .regular)
                    .lineSpacing(4)
            }
            .padding(.leading, CGFloat(indent) * 18)
        case let .quote(text):
            HStack(spacing: 12) {
                Capsule()
                    .fill(Color.white.opacity(0.22))
                    .frame(width: 2)
                animatedText(text, size: 13.5, weight: .regular)
                    .italic()
                    .foregroundStyle(Color.white.opacity(0.68))
            }
            .padding(.vertical, 2)
        case let .code(language, text):
            CodeBlockView(language: language, source: text, streaming: streaming)
        case let .math(latex):
            LatexBlockView(latex: latex, streaming: streaming)
        case let .table(header, alignments, rows):
            MarkdownTableView(
                header: header,
                alignments: alignments,
                rows: rows,
                streaming: streaming
            )
        case .rule:
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [.clear, Color.white.opacity(0.16), .clear],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .frame(height: 1)
                .padding(.vertical, 6)
        }
    }

    private func animatedText(
        _ source: String,
        size: CGFloat,
        weight: Font.Weight
    ) -> some View {
        SmoothTokenText(
            content: InlineLatexText.render(source, mathSize: size),
            source: source,
            streaming: streaming
        )
        .font(.system(size: size, weight: weight))
        .foregroundStyle(Color.white.opacity(0.92))
    }

    private func headingSize(_ level: Int) -> CGFloat {
        switch level {
        case 1: 23
        case 2: 19
        case 3: 16.5
        default: 14.5
        }
    }

}

private struct CodeTextChunk: Identifiable, Equatable {
    let id: Int
    let source: String
}

private enum CodeTextChunker {
    static let targetCharacters = 2_048

    static func chunks(_ source: String) -> [CodeTextChunk] {
        guard source.count > targetCharacters else {
            return [CodeTextChunk(id: 0, source: source)]
        }

        let lines = source.split(separator: "\n", omittingEmptySubsequences: false)
        var chunks: [CodeTextChunk] = []
        var current = ""
        var sourceOffset = 0

        func flush() {
            guard !current.isEmpty else { return }
            chunks.append(CodeTextChunk(id: sourceOffset, source: current))
            sourceOffset += current.utf8.count
            current = ""
        }

        for (index, line) in lines.enumerated() {
            var remainder = String(line) + (index == lines.count - 1 ? "" : "\n")
            if current.count + remainder.count > targetCharacters, !current.isEmpty {
                flush()
            }
            while remainder.count > targetCharacters {
                let end = remainder.index(remainder.startIndex, offsetBy: targetCharacters)
                current = String(remainder[..<end])
                flush()
                remainder = String(remainder[end...])
            }
            current += remainder
        }
        flush()
        return chunks.isEmpty ? [CodeTextChunk(id: 0, source: source)] : chunks
    }
}

@MainActor
private enum SyntaxHighlighter {
    private static let base = NSColor(srgbRed: 0.82, green: 0.85, blue: 0.89, alpha: 1)
    private static let comment = NSColor(srgbRed: 0.43, green: 0.49, blue: 0.54, alpha: 1)
    private static let string = NSColor(srgbRed: 0.62, green: 0.83, blue: 0.65, alpha: 1)
    private static let keyword = NSColor(srgbRed: 0.48, green: 0.73, blue: 0.98, alpha: 1)
    private static let type = NSColor(srgbRed: 0.74, green: 0.67, blue: 0.94, alpha: 1)
    private static let number = NSColor(srgbRed: 0.94, green: 0.72, blue: 0.43, alpha: 1)
    private static let function = NSColor(srgbRed: 0.48, green: 0.84, blue: 0.86, alpha: 1)
    private static var lexerCache: [String: NSRegularExpression] = [:]
    private static var markupRegexCache: [(NSRegularExpression, NSColor)]?

    static func highlight(_ source: String, language rawLanguage: String?) -> AttributedString {
        let result = NSMutableAttributedString(
            string: source,
            attributes: [.foregroundColor: base]
        )
        guard !source.isEmpty else { return AttributedString(result) }

        let language = normalized(rawLanguage)
        if ["html", "xml", "svg", "vue", "svelte"].contains(language) {
            highlightMarkup(result, source: source)
            return AttributedString(result)
        }

        guard let regex = lexer(for: language) else { return AttributedString(result) }

        let fullRange = NSRange(source.startIndex..<source.endIndex, in: source)
        for match in regex.matches(in: source, range: fullRange) {
            let colors = [comment, string, keyword, type, number, function]
            for group in 1...colors.count where match.range(at: group).location != NSNotFound {
                result.addAttribute(
                    .foregroundColor,
                    value: colors[group - 1],
                    range: match.range(at: group)
                )
                break
            }
        }
        return AttributedString(result)
    }

    private static func highlightMarkup(_ result: NSMutableAttributedString, source: String) {
        if markupRegexCache == nil {
            markupRegexCache = [
                (#"<!--[\s\S]*?-->"#, comment),
                (#"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"#, string),
                (#"</?\s*[A-Za-z][A-Za-z0-9:_-]*"#, keyword),
                (#"\b[A-Za-z_:][A-Za-z0-9:._-]*(?=\s*=)"#, function),
                (#"<!DOCTYPE[^>]*>|</?|/?>"#, type),
                (#"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);"#, number),
            ].compactMap { pattern, color in
                guard let regex = try? NSRegularExpression(
                    pattern: pattern,
                    options: [.anchorsMatchLines]
                ) else { return nil }
                return (regex, color)
            }
        }
        for (regex, color) in markupRegexCache ?? [] {
            apply(regex, color: color, to: result, source: source)
        }
    }

    private static func apply(
        _ regex: NSRegularExpression,
        color: NSColor,
        to result: NSMutableAttributedString,
        source: String
    ) {
        let range = NSRange(source.startIndex..<source.endIndex, in: source)
        for match in regex.matches(in: source, range: range) {
            result.addAttribute(.foregroundColor, value: color, range: match.range)
        }
    }

    private static func lexer(for language: String) -> NSRegularExpression? {
        let key = lexerKey(for: language)
        if let cached = lexerCache[key] { return cached }
        let definition = definition(for: language)
        let pattern = "(\(definition.comments))|(\(definition.strings))|\\b(\(alternation(definition.keywords)))\\b|\\b(\(alternation(definition.types)))\\b|(\\b(?:0[xX][0-9A-Fa-f](?:_?[0-9A-Fa-f])*|0[bB][01](?:_?[01])*|(?:\\d(?:_?\\d)*)?(?:\\.\\d(?:_?\\d)*)|\\d(?:_?\\d)*(?:[eE][+-]?\\d+)?)[fFdDuUlL]*\\b)|(\\b[A-Za-z_$][A-Za-z0-9_$]*(?=\\s*\\())"
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.anchorsMatchLines]
        ) else { return nil }
        lexerCache[key] = regex
        return regex
    }

    private static func lexerKey(for language: String) -> String {
        switch language {
        case "python", "swift", "javascript", "typescript", "bash", "json", "sql", "yaml", "toml", "markdown":
            language
        default:
            "generic"
        }
    }

    private static func alternation(_ values: [String]) -> String {
        guard !values.isEmpty else { return "(?!)" }
        return "(?:" + values.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|") + ")"
    }

    private static func normalized(_ rawLanguage: String?) -> String {
        let language = rawLanguage?.lowercased().trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return switch language {
        case "js", "jsx": "javascript"
        case "ts", "tsx": "typescript"
        case "py": "python"
        case "rs": "rust"
        case "sh", "zsh", "shell": "bash"
        case "yml": "yaml"
        case "md": "markdown"
        case "c++", "cc", "hpp": "cpp"
        case "cs": "csharp"
        default: language
        }
    }

    private static func definition(for language: String) -> SyntaxDefinition {
        switch language {
        case "python":
            SyntaxDefinition(
                comments: #"#.*$"#,
                strings: #"\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"#,
                keywords: ["and", "as", "assert", "async", "await", "break", "case", "class", "continue", "def", "del", "elif", "else", "except", "False", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "match", "None", "nonlocal", "not", "or", "pass", "raise", "return", "True", "try", "while", "with", "yield"],
                types: ["bool", "bytes", "dict", "float", "frozenset", "int", "list", "object", "set", "str", "tuple", "type"]
            )
        case "swift":
            SyntaxDefinition(
                comments: #"//.*$|/\*[\s\S]*?\*/"#,
                strings: #"\"\"\"[\s\S]*?\"\"\"|\"(?:\\.|[^\"\\])*\""#,
                keywords: ["actor", "as", "async", "await", "break", "case", "catch", "class", "continue", "default", "defer", "do", "else", "enum", "extension", "fallthrough", "false", "fileprivate", "for", "func", "guard", "if", "import", "in", "init", "inout", "internal", "is", "let", "nil", "nonisolated", "open", "private", "protocol", "public", "repeat", "return", "self", "some", "static", "struct", "subscript", "super", "switch", "throw", "throws", "true", "try", "typealias", "var", "where", "while"],
                types: ["Any", "Array", "Bool", "Character", "Data", "Dictionary", "Double", "Error", "Float", "Int", "Never", "Result", "Set", "String", "URL", "UUID", "Void"]
            )
        case "javascript", "typescript":
            SyntaxDefinition(
                comments: #"//.*$|/\*[\s\S]*?\*/"#,
                strings: #"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`"#,
                keywords: ["async", "await", "break", "case", "catch", "class", "const", "continue", "debugger", "default", "delete", "do", "else", "export", "extends", "false", "finally", "for", "from", "function", "get", "if", "implements", "import", "in", "instanceof", "interface", "let", "new", "null", "of", "package", "private", "protected", "public", "return", "set", "static", "super", "switch", "this", "throw", "true", "try", "typeof", "undefined", "var", "void", "while", "with", "yield"],
                types: ["Array", "BigInt", "Boolean", "Date", "Error", "Map", "Number", "Object", "Promise", "Record", "Set", "String", "Symbol", "unknown", "never"]
            )
        case "bash":
            SyntaxDefinition(
                comments: #"#.*$"#,
                strings: #"\"(?:\\.|[^\"\\])*\"|'[^']*'"#,
                keywords: ["case", "do", "done", "elif", "else", "esac", "export", "fi", "for", "function", "if", "in", "local", "readonly", "return", "select", "then", "time", "until", "while"],
                types: ["false", "true"]
            )
        case "json":
            SyntaxDefinition(
                comments: #"(?!)"#,
                strings: #"\"(?:\\.|[^\"\\])*\""#,
                keywords: ["false", "null", "true"],
                types: []
            )
        case "sql":
            SyntaxDefinition(
                comments: #"--.*$|/\*[\s\S]*?\*/"#,
                strings: #"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\""#,
                keywords: ["ADD", "ALTER", "AND", "AS", "ASC", "BEGIN", "BETWEEN", "BY", "CASE", "CREATE", "DELETE", "DESC", "DISTINCT", "DROP", "ELSE", "END", "EXISTS", "FROM", "FULL", "GROUP", "HAVING", "IN", "INDEX", "INNER", "INSERT", "INTO", "IS", "JOIN", "LEFT", "LIKE", "LIMIT", "NOT", "NULL", "ON", "OR", "ORDER", "OUTER", "RETURNING", "RIGHT", "SELECT", "SET", "TABLE", "THEN", "UNION", "UNIQUE", "UPDATE", "VALUES", "WHEN", "WHERE", "WITH"],
                types: ["BIGINT", "BOOLEAN", "CHAR", "DATE", "DECIMAL", "FLOAT", "INTEGER", "JSON", "NUMERIC", "REAL", "TEXT", "TIMESTAMP", "VARCHAR"]
            )
        case "yaml", "toml", "markdown":
            SyntaxDefinition(
                comments: #"#.*$|<!--[^>]*-->"#,
                strings: #"\"(?:\\.|[^\"\\])*\"|'[^']*'"#,
                keywords: ["false", "null", "true", "yes", "no"],
                types: []
            )
        default:
            SyntaxDefinition(
                comments: #"//.*$|/\*[\s\S]*?\*/"#,
                strings: #"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`"#,
                keywords: ["abstract", "as", "async", "await", "break", "case", "catch", "class", "const", "continue", "default", "defer", "do", "else", "enum", "export", "extends", "false", "final", "finally", "for", "func", "function", "if", "implements", "import", "in", "interface", "let", "match", "new", "nil", "null", "package", "private", "protected", "public", "return", "static", "struct", "super", "switch", "this", "throw", "throws", "true", "try", "type", "var", "void", "where", "while", "yield"],
                types: ["bool", "boolean", "byte", "char", "double", "float", "int", "long", "short", "string", "uint", "usize"]
            )
        }
    }
}

private struct SyntaxDefinition {
    let comments: String
    let strings: String
    let keywords: [String]
    let types: [String]
}

private struct CodeBlockView: View {
    let language: String?
    let source: String
    let streaming: Bool
    @State private var copied = false

    var body: some View {
        let chunks = CodeTextChunker.chunks(source)
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Text(language ?? "code")
                    .font(.system(size: 10, weight: .medium, design: .monospaced))
                    .foregroundStyle(StudioTheme.quiet)
                Spacer()
                Button(action: copySource) {
                    Label(copied ? "Copié" : "Copier", systemImage: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10, weight: .medium))
                        .foregroundStyle(copied ? StudioTheme.accent : StudioTheme.secondary)
                        .padding(.horizontal, 9)
                        .frame(height: 25)
                }
                .buttonStyle(StudioControlStyle())
                .help("Copier tout le bloc")
            }
            VStack(alignment: .leading, spacing: 0) {
                ForEach(chunks) { chunk in
                    CodeTextChunkView(
                        source: chunk.source,
                        language: language,
                        streaming: streaming && chunk.id == chunks.last?.id
                    )
                    .equatable()
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(13)
        .background(Color.white.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.075), lineWidth: 0.7)
        }
    }

    private func copySource() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(source, forType: .string)
        copied = true
        Task { @MainActor in
            try? await Task.sleep(for: .seconds(1.4))
            copied = false
        }
    }
}

private struct CodeTextChunkView: View, Equatable {
    let source: String
    let language: String?
    let streaming: Bool

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.language == rhs.language && lhs.streaming == rhs.streaming
    }

    var body: some View {
        Text(SyntaxHighlighter.highlight(source, language: language))
            .font(.system(size: 12, weight: .regular, design: .monospaced))
            .opacity(streaming ? 0.94 : 1)
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct MarkdownTableView: View {
    let header: [String]
    let alignments: [MarkdownTableAlignment]
    let rows: [[String]]
    let streaming: Bool

    var body: some View {
        ScrollView(.horizontal) {
            Grid(horizontalSpacing: 0, verticalSpacing: 0) {
                GridRow {
                    ForEach(header.indices, id: \.self) { column in
                        cell(header[column], column: column, header: true)
                    }
                }
                ForEach(rows.indices, id: \.self) { row in
                    GridRow {
                        ForEach(header.indices, id: \.self) { column in
                            cell(rows[row][column], column: column, header: false)
                        }
                    }
                }
            }
            .background(Color.white.opacity(0.018), in: RoundedRectangle(cornerRadius: 11))
            .overlay {
                RoundedRectangle(cornerRadius: 11)
                    .stroke(Color.white.opacity(0.10), lineWidth: 0.7)
            }
            .clipShape(RoundedRectangle(cornerRadius: 11))
        }
        .scrollIndicators(.never)
    }

    private func cell(_ source: String, column: Int, header isHeader: Bool) -> some View {
        SmoothTokenText(
            content: InlineLatexText.render(source, mathSize: 12.5),
            source: source,
            streaming: streaming
        )
        .font(.system(size: 12.5, weight: isHeader ? .semibold : .regular, design: .rounded))
        .foregroundStyle(Color.white.opacity(isHeader ? 0.92 : 0.76))
        .frame(minWidth: 112, maxWidth: 260, alignment: frameAlignment(column))
        .padding(.horizontal, 12)
        .padding(.vertical, isHeader ? 10 : 9)
        .background(isHeader ? Color.white.opacity(0.055) : Color.clear)
        .overlay(alignment: .trailing) {
            if column < header.count - 1 {
                Rectangle().fill(Color.white.opacity(0.07)).frame(width: 0.7)
            }
        }
        .overlay(alignment: .bottom) {
            Rectangle().fill(Color.white.opacity(0.07)).frame(height: 0.7)
        }
    }

    private func frameAlignment(_ column: Int) -> Alignment {
        guard alignments.indices.contains(column) else { return .leading }
        switch alignments[column] {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }
}

@MainActor
private enum InlineLatexText {
    static func render(_ source: String, mathSize: CGFloat) -> Text {
        InlineMathParser.parse(source).reduce(Text("")) { result, piece in
            switch piece {
            case let .prose(text):
                return Text("\(result)\(markdownText(text))")
            case let .math(latex, raw, mode):
                guard let image = LatexImageCache.image(
                    latex: latex,
                    fontSize: mode == .display ? mathSize * 1.22 : mathSize,
                    mode: mode
                ) else {
                    return Text("\(result)\(Text(raw))")
                }
                let offset = mode == .display ? -2.4 : -1.2
                let formula = Text(Image(nsImage: image)).baselineOffset(offset)
                return Text("\(result)\(formula)")
            }
        }
    }

    private static func markdownText(_ source: String) -> Text {
        let attributed = (try? AttributedString(
            markdown: source,
            options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        )) ?? AttributedString(source)
        return Text(attributed)
    }
}

private enum InlineMathPiece {
    case prose(String)
    case math(latex: String, raw: String, mode: MTMathUILabelMode)
}

private enum InlineMathParser {
    static func parse(_ source: String) -> [InlineMathPiece] {
        let characters = Array(source)
        var pieces: [InlineMathPiece] = []
        var textStart = 0
        var index = 0

        func appendProse(until end: Int) {
            guard end > textStart else { return }
            pieces.append(.prose(String(characters[textStart..<end])))
        }

        func appendMath(
            from start: Int,
            contentStart: Int,
            closeStart: Int,
            end: Int,
            mode: MTMathUILabelMode
        ) {
            appendProse(until: start)
            pieces.append(
                .math(
                    latex: String(characters[contentStart..<closeStart]),
                    raw: String(characters[start..<end]),
                    mode: mode
                )
            )
            textStart = end
            index = end
        }

        while index < characters.count {
            if characters[index] == "`" {
                index = indexAfterCodeSpan(startingAt: index, in: characters)
                continue
            }
            if matches(#"\("#, at: index, in: characters),
               let close = indexOf(#"\)"#, after: index + 2, in: characters) {
                appendMath(
                    from: index,
                    contentStart: index + 2,
                    closeStart: close,
                    end: close + 2,
                    mode: .text
                )
                continue
            }
            if matches(#"\["#, at: index, in: characters),
               let close = indexOf(#"\]"#, after: index + 2, in: characters) {
                appendMath(
                    from: index,
                    contentStart: index + 2,
                    closeStart: close,
                    end: close + 2,
                    mode: .display
                )
                continue
            }
            if characters[index] == "$", !isEscaped(index, in: characters) {
                let delimiterLength = index + 1 < characters.count && characters[index + 1] == "$" ? 2 : 1
                if let close = dollarCloser(
                    after: index + delimiterLength,
                    length: delimiterLength,
                    in: characters
                ) {
                    appendMath(
                        from: index,
                        contentStart: index + delimiterLength,
                        closeStart: close,
                        end: close + delimiterLength,
                        mode: delimiterLength == 2 ? .display : .text
                    )
                    continue
                }
            }
            index += 1
        }

        appendProse(until: characters.count)
        return pieces
    }

    private static func matches(_ needle: String, at index: Int, in characters: [Character]) -> Bool {
        let pattern = Array(needle)
        guard index + pattern.count <= characters.count else { return false }
        return Array(characters[index..<(index + pattern.count)]) == pattern
    }

    private static func indexOf(
        _ needle: String,
        after start: Int,
        in characters: [Character]
    ) -> Int? {
        var index = start
        while index < characters.count {
            if matches(needle, at: index, in: characters), !isEscaped(index, in: characters) {
                return index
            }
            index += 1
        }
        return nil
    }

    private static func dollarCloser(
        after start: Int,
        length: Int,
        in characters: [Character]
    ) -> Int? {
        var index = start
        while index < characters.count {
            guard characters[index] == "$", !isEscaped(index, in: characters) else {
                index += 1
                continue
            }
            if length == 1 {
                let besideDollar = index + 1 < characters.count && characters[index + 1] == "$"
                if !besideDollar {
                    return index
                }
            } else if index + 1 < characters.count && characters[index + 1] == "$" {
                return index
            }
            index += 1
        }
        return nil
    }

    private static func indexAfterCodeSpan(startingAt start: Int, in characters: [Character]) -> Int {
        var index = start + 1
        while index < characters.count {
            if characters[index] == "`", !isEscaped(index, in: characters) {
                return index + 1
            }
            index += 1
        }
        return start + 1
    }

    private static func isEscaped(_ index: Int, in characters: [Character]) -> Bool {
        guard index > 0 else { return false }
        var backslashes = 0
        var cursor = index - 1
        while characters[cursor] == "\\" {
            backslashes += 1
            guard cursor > 0 else { break }
            cursor -= 1
        }
        return backslashes.isMultiple(of: 2) == false
    }
}

@MainActor
private enum LatexImageCache {
    private static let cache = NSCache<NSString, NSImage>()

    static func image(
        latex: String,
        fontSize: CGFloat,
        mode: MTMathUILabelMode
    ) -> NSImage? {
        let modeKey = mode == .display ? "display" : "text"
        let key = "\(modeKey)|\(fontSize)|\(latex)" as NSString
        if let cached = cache.object(forKey: key) {
            return cached
        }

        let renderer = MTMathImage(
            latex: latex,
            fontSize: fontSize,
            textColor: NSColor(white: 0.94, alpha: 1),
            labelMode: mode,
            textAlignment: .center
        )
        let (_, rendered) = renderer.asImage()
        guard let rendered else { return nil }
        cache.setObject(rendered, forKey: key)
        return rendered
    }
}

private struct LatexBlockView: View {
    let latex: String
    let streaming: Bool
    @State private var visible = false

    var body: some View {
        Group {
            if let image = LatexImageCache.image(
                latex: latex,
                fontSize: 18,
                mode: .display
            ) {
                ScrollView(.horizontal) {
                    Image(nsImage: image)
                        .padding(.horizontal, 16)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
                .scrollIndicators(.never)
            } else {
                Text("$$\n\(latex)\n$$")
                    .font(.system(size: 12, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.68))
            }
        }
        .padding(.vertical, 7)
        .opacity(streaming ? (visible ? 1 : 0.25) : 1)
        .scaleEffect(streaming ? (visible ? 1 : 0.985) : 1, anchor: .leading)
        .onAppear {
            guard streaming else { return }
            withAnimation(.easeOut(duration: 0.16)) {
                visible = true
            }
        }
    }
}

private struct SmoothTokenText: View {
    let content: Text
    let source: String
    let streaming: Bool

    var body: some View {
        content
    }
}
