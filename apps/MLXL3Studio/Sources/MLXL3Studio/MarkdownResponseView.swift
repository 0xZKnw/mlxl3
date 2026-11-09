import AppKit
import SwiftMath
import SwiftUI

private enum MarkdownBlockKind: Sendable {
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

private enum MarkdownTableAlignment: Sendable {
    case leading
    case center
    case trailing
}

private struct MarkdownBlock: Identifiable, Sendable {
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
        var codeFence = ""
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
                if trimmed.hasPrefix(codeFence), trimmed.dropFirst(codeFence.count).allSatisfy({ $0 == codeFence.first || $0.isWhitespace }) {
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

            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                flushParagraph()
                codeFence = String(trimmed.prefix(while: { $0 == trimmed.first }))
                let language = String(trimmed.dropFirst(codeFence.count)).trimmingCharacters(in: .whitespaces)
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

struct StreamingTextChunk: Identifiable, Equatable, Sendable {
    let id: Int
    let source: String
    // IDs include repeated table headers; only this offset indexes raw source.
    var sourceOffset = 0
    var tableLines: [String] = []
}

enum StreamingTextChunker {
    static let targetCharacters = 6_000
    static let hardLimitCharacters = 16_000

    final class Cache {
        private let markdown: Bool
        private var previousSource = ""
        private var previousChunks: [StreamingTextChunk] = []

        init(markdown: Bool = true) { self.markdown = markdown }

        private func split(_ source: String, continuingTable: [String] = []) -> [StreamingTextChunk]
        {
            if markdown {
                return StreamingTextChunker.chunks(source, continuingTable: continuingTable)
            }
            return CodeTextChunker.chunks(source).map {
                StreamingTextChunk(id: $0.id, source: $0.source, sourceOffset: $0.id)
            }
        }

        func chunks(_ source: String) -> [StreamingTextChunk] {
            if source.utf8.count == previousSource.utf8.count,
                source.utf8.elementsEqual(previousSource.utf8), !previousChunks.isEmpty
            {
                return previousChunks
            }
            defer { previousSource = source }
            // ponytail: exact append check still scans bytes; an explicit append
            // API is only needed if that check, rather than parsing, dominates.
            guard previousChunks.count > 2, source.utf8.starts(with: previousSource.utf8) else {
                previousChunks = split(source)
                return previousChunks
            }
            // A full chunk of lookbehind keeps partial fences, headers and
            // Unicode graphemes out of the frozen prefix.
            let stableCount = previousChunks.count - 2
            let restart = previousChunks[stableCount]
            let boundary = source.utf8.index(source.utf8.startIndex, offsetBy: restart.sourceOffset)
            let changed = split(String(source[boundary...]), continuingTable: restart.tableLines)
            previousChunks.removeLast(2)
            previousChunks += changed.map { chunk in
                StreamingTextChunk(
                    id: restart.id + chunk.id, source: chunk.source,
                    sourceOffset: restart.sourceOffset + chunk.sourceOffset,
                    tableLines: chunk.tableLines
                )
            }
            return previousChunks
        }
    }

    static func chunks(_ source: String, continuingTable: [String] = []) -> [StreamingTextChunk] {
        let prefix = continuingTable.count == 2 && continuingTable[1].contains("---") ? continuingTable.joined() : ""
        let rendered = prefix + source
        guard let threshold = rendered.index(rendered.startIndex, offsetBy: targetCharacters, limitedBy: rendered.endIndex),
              threshold < rendered.endIndex else {
            return [StreamingTextChunk(id: 0, source: rendered, tableLines: continuingTable)]
        }

        let lines = source.split(separator: "\n", omittingEmptySubsequences: false)
        var chunks: [StreamingTextChunk] = []
        var current = prefix
        var byteOffset = 0
        var insideCodeFence = false
        var fence = ""
        var tableLines = continuingTable
        var currentCharacters = prefix.count
        var sourceOffset = 0
        var chunkSourceOffset = 0
        var chunkTableLines = continuingTable

        func flush() {
            guard !current.isEmpty else { return }
            chunks.append(StreamingTextChunk(id: byteOffset, source: current,
                                             sourceOffset: chunkSourceOffset, tableLines: chunkTableLines))
            byteOffset += current.utf8.count
            chunkSourceOffset = sourceOffset
            current = ""
            currentCharacters = 0
            chunkTableLines = tableLines
        }

        for (index, line) in lines.enumerated() {
            let renderedLine = String(line) + (index == lines.count - 1 ? "" : "\n")
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            current += renderedLine
            currentCharacters += renderedLine.count
            sourceOffset += renderedLine.utf8.count

            if !insideCodeFence && (trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~")) {
                fence = String(trimmed.prefix(while: { $0 == trimmed.first }))
                insideCodeFence = true
            } else if insideCodeFence && trimmed.hasPrefix(fence) && trimmed.dropFirst(fence.count).allSatisfy({ $0 == fence.first || $0.isWhitespace }) {
                insideCodeFence = false
            }
            if !insideCodeFence {
                if trimmed.contains("|") {
                    if tableLines.count < 2 { tableLines.append(renderedLine) }
                } else { tableLines = [] }
            }
            let isBlankBoundary = trimmed.isEmpty
            if !insideCodeFence,
               (currentCharacters >= hardLimitCharacters
                || (currentCharacters >= targetCharacters && isBlankBoundary)) {
                flush()
                if tableLines.count == 2, tableLines[1].contains("---") {
                    current = tableLines.joined()
                    currentCharacters = current.count
                }
            }
        }
        flush()
        return chunks.isEmpty ? [StreamingTextChunk(id: 0, source: source)] : chunks
    }
}

private struct PreparedMarkdownChunk: Identifiable, Sendable, Equatable {
    let chunk: StreamingTextChunk
    let blocks: [MarkdownBlock]
    var id: Int { chunk.id }

    static func == (lhs: Self, rhs: Self) -> Bool { lhs.chunk == rhs.chunk }
}

private actor MarkdownPreparation {
    private let chunkCache = StreamingTextChunker.Cache()
    private var previous: [PreparedMarkdownChunk] = []

    func prepare(_ source: String) -> [PreparedMarkdownChunk]? {
        guard !Task.isCancelled else { return nil }
        let chunks = chunkCache.chunks(source)
        let old = Dictionary(uniqueKeysWithValues: previous.map { ($0.id, $0) })
        var prepared: [PreparedMarkdownChunk] = []
        for chunk in chunks {
            guard !Task.isCancelled else { return nil }
            if let cached = old[chunk.id], cached.chunk.source.utf8.elementsEqual(chunk.source.utf8) {
                prepared.append(cached)
            } else {
                prepared.append(PreparedMarkdownChunk(chunk: chunk, blocks: MarkdownParser.parse(chunk.source)))
            }
        }
        previous = prepared
        return prepared
    }
}

@MainActor private final class MarkdownRenderModel: ObservableObject {
    @Published private(set) var chunks: [PreparedMarkdownChunk] = []
    private let preparation = MarkdownPreparation()
    private var pendingSource: String?
    private var requestedSource = ""
    private var renderTask: Task<Void, Never>?
    private var generation = UUID()

    func request(_ source: String) {
        requestedSource = source
        pendingSource = source
        guard renderTask == nil else { return }
        let ticket = generation
        renderTask = Task { [weak self, preparation] in
            while let source = self?.pendingSource {
                self?.pendingSource = nil
                guard let prepared = await preparation.prepare(source),
                      !Task.isCancelled, let self, generation == ticket else { return }
                // A prepared prefix is still correct while tokens arrive. Keep
                // displaying progress even when parsing takes longer than one
                // delta, then prepare only the newest pending snapshot.
                if requestedSource.utf8.starts(with: source.utf8) { chunks = prepared }
            }
            if self?.generation == ticket { self?.renderTask = nil }
        }
    }

    func cancel() {
        generation = UUID()
        renderTask?.cancel()
        renderTask = nil
        pendingSource = nil
    }
}

struct MarkdownResponseView: View, Equatable {
    private let source: String
    private let streaming: Bool
    private let textScale: CGFloat
    private let htmlPreviewEnabled: Bool
    @StateObject private var renderer = MarkdownRenderModel()

    init(_ source: String, streaming: Bool = false, textScale: CGFloat = 1, htmlPreviewEnabled: Bool = true) {
        self.source = source
        self.streaming = streaming
        self.textScale = textScale
        self.htmlPreviewEnabled = htmlPreviewEnabled && !streaming
    }

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.streaming == rhs.streaming && lhs.textScale == rhs.textScale && lhs.htmlPreviewEnabled == rhs.htmlPreviewEnabled
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if HTMLPreview.isDocument(source) {
                CodeBlockView(language: "html", source: source, streaming: streaming,
                              textScale: textScale, htmlPreviewEnabled: htmlPreviewEnabled).equatable()
            } else {
            ForEach(renderer.chunks) { chunk in
                MarkdownChunkView(
                    prepared: chunk,
                    streaming: streaming && chunk.id == renderer.chunks.last?.id,
                    textScale: textScale,
                    htmlPreviewEnabled: htmlPreviewEnabled
                )
                .equatable()
            }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .textSelection(.enabled)
        .onAppear { renderer.request(source) }
        .onChange(of: source) { renderer.request(source) }
        .onDisappear { renderer.cancel() }
    }
}

private struct MarkdownChunkView: View, Equatable {
    let prepared: PreparedMarkdownChunk
    let streaming: Bool
    let textScale: CGFloat
    let htmlPreviewEnabled: Bool

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.prepared.chunk.source == rhs.prepared.chunk.source && lhs.streaming == rhs.streaming && lhs.textScale == rhs.textScale && lhs.htmlPreviewEnabled == rhs.htmlPreviewEnabled
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(prepared.blocks) { block in
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
                .lineSpacing(5 * textScale)
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
                    .fill(StudioTheme.contrast.opacity(indent == 0 ? 0.72 : 0.42))
                    .frame(width: indent == 0 ? 4.5 : 3.5, height: indent == 0 ? 4.5 : 3.5)
                animatedText(text, size: 15, weight: .regular)
                    .lineSpacing(4 * textScale)
            }
            .padding(.leading, CGFloat(indent) * 18 + 3)
        case let .ordered(indent, number, text):
            HStack(alignment: .firstTextBaseline, spacing: 9) {
                Text(number + ".")
                    .font(.system(size: 12 * textScale, weight: .bold, design: .rounded))
                    .foregroundStyle(StudioTheme.contrast.opacity(0.48))
                    .frame(minWidth: 18, alignment: .trailing)
                animatedText(text, size: 15, weight: .regular)
                    .lineSpacing(4 * textScale)
            }
            .padding(.leading, CGFloat(indent) * 18)
        case let .quote(text):
            HStack(spacing: 12) {
                Capsule()
                    .fill(StudioTheme.contrast.opacity(0.22))
                    .frame(width: 2)
                animatedText(text, size: 13.5, weight: .regular)
                    .italic()
                    .foregroundStyle(StudioTheme.contrast.opacity(0.68))
            }
            .padding(.vertical, 2)
        case let .code(language, text):
            CodeBlockView(language: language, source: text, streaming: streaming, textScale: textScale, htmlPreviewEnabled: htmlPreviewEnabled)
                .equatable()
        case let .math(latex):
            LatexBlockView(latex: latex, streaming: streaming, textScale: textScale)
        case let .table(header, alignments, rows):
            MarkdownTableView(
                header: header,
                alignments: alignments,
                rows: rows,
                streaming: streaming,
                textScale: textScale
            )
        case .rule:
            Rectangle()
                .fill(
                    LinearGradient(
                        colors: [.clear, StudioTheme.contrast.opacity(0.16), .clear],
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
            content: InlineLatexText.render(source, mathSize: size * textScale),
            source: source,
            streaming: streaming
        )
        .font(.system(size: size * textScale, weight: weight))
        .foregroundStyle(StudioTheme.contrast.opacity(0.92))
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

struct CodeTextChunk: Identifiable, Equatable {
    let id: Int
    let source: String
    var lexicalPrefix = ""
    var lexicalSuffix = ""
}

enum CodeTextChunker {
    static let targetCharacters = 2_048
    // Carry multiline lexemes across layout chunks without building one huge
    // attributed string. This is display highlighting, not a compiler lexer.
    private static let lexemes = try! NSRegularExpression(pattern: #"<!--[\s\S]*?(?:-->|\z)|/\*[\s\S]*?(?:\*/|\z)|//[^\n]*|#[^\n]*|\"\"\"[\s\S]*?(?:\"\"\"|\z)|'''[\s\S]*?(?:'''|\z)|\"(?:\\.|[^\"\\])*(?:\"|\z)|'(?:\\.|[^'\\])*(?:'|\z)|`(?:\\.|[^`\\])*(?:`|\z)"#)

    final class Cache {
        private var previousSource = ""
        private var previousChunks: [CodeTextChunk] = []

        func highlightedChunks(_ source: String) -> [CodeTextChunk] {
            if source.utf8.count == previousSource.utf8.count,
               source.utf8.elementsEqual(previousSource.utf8), !previousChunks.isEmpty { return previousChunks }
            defer { previousSource = source }
            // ponytail: appended escapes use the original full scan; use an incremental lexer if they dominate.
            guard previousChunks.count > 2, previousSource.utf8.last != 92,
                  source.utf8.starts(with: previousSource.utf8),
                  !source.utf8.suffix(source.utf8.count - previousSource.utf8.count).contains(92) else {
                previousChunks = CodeTextChunker.highlightedChunks(source)
                return previousChunks
            }
            // Keep a full chunk of lookbehind for delimiters split across appends.
            let stableCount = previousChunks.count - 2
            let offset = previousChunks[stableCount].id
            let boundary = source.utf8.index(source.utf8.startIndex, offsetBy: offset)
            var prefix = previousChunks[stableCount].lexicalPrefix
            if ["\"", "'", "`"].contains(prefix) {
                let backslashes = source.utf8[..<boundary].reversed().prefix(while: { $0 == 92 }).count
                if !backslashes.isMultiple(of: 2) { prefix += "\\" }
            }
            let tail = String(source[boundary...])
            let changed = CodeTextChunker.highlightedChunks(tail, lexicalPrefix: prefix)
            if !prefix.isEmpty, changed.first?.lexicalPrefix != previousChunks[stableCount].lexicalPrefix {
                previousChunks = CodeTextChunker.highlightedChunks(source)
                return previousChunks
            }
            previousChunks.removeLast(2)
            previousChunks += changed.map {
                CodeTextChunk(id: offset + $0.id, source: $0.source,
                              lexicalPrefix: $0.lexicalPrefix, lexicalSuffix: $0.lexicalSuffix)
            }
            return previousChunks
        }
    }

    static func highlightedChunks(_ source: String, lexicalPrefix: String = "") -> [CodeTextChunk] {
        var output = chunks(source)
        let lexicalSource = lexicalPrefix + source
        let text = lexicalSource as NSString
        let matches = lexemes.matches(in: lexicalSource, range: NSRange(location: 0, length: text.length))
        var offset = lexicalPrefix.utf16.count
        var matchIndex = 0
        for index in output.indices {
            let end = offset + output[index].source.utf16.count
            while matchIndex < matches.count && NSMaxRange(matches[matchIndex].range) <= offset { matchIndex += 1 }
            var probe = matchIndex
            while probe < matches.count && matches[probe].range.location < end {
                let range = matches[probe].range
                let head = text.substring(with: NSRange(location: range.location, length: min(4, range.length)))
                let pair: (String, String)?
                if head.hasPrefix("<!--") { pair = ("<!--", "-->") }
                else if head.hasPrefix("/*") { pair = ("/*", "*/") }
                else if head.hasPrefix("\"\"\"") { pair = ("\"\"\"", "\"\"\"") }
                else if head.hasPrefix("'''") { pair = ("'''", "'''") }
                else if head.hasPrefix("//") { pair = ("//", "\n") }
                else if head.hasPrefix("#") { pair = ("#", "\n") }
                else if let first = head.first { pair = (String(first), String(first)) }
                else { pair = nil }
                if let pair {
                    if range.location < offset { output[index].lexicalPrefix = pair.0 }
                    if NSMaxRange(range) >= end { output[index].lexicalSuffix = pair.1 }
                }
                probe += 1
            }
            offset = end
        }
        return output
    }

    static func chunks(_ source: String) -> [CodeTextChunk] {
        guard let threshold = source.index(source.startIndex, offsetBy: targetCharacters, limitedBy: source.endIndex),
              threshold < source.endIndex else {
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
            let rendered = String(line) + (index == lines.count - 1 ? "" : "\n")
            var remainder = rendered[...]
            if !current.isEmpty,
               let limit = remainder.index(remainder.startIndex, offsetBy: targetCharacters - current.count, limitedBy: remainder.endIndex),
               limit < remainder.endIndex {
                flush()
            }
            while let end = remainder.index(remainder.startIndex, offsetBy: targetCharacters, limitedBy: remainder.endIndex), end < remainder.endIndex {
                current = String(remainder[..<end])
                flush()
                remainder = remainder[end...]
            }
            current += String(remainder)
        }
        flush()
        return chunks.isEmpty ? [CodeTextChunk(id: 0, source: source)] : chunks
    }
}

@MainActor
enum MarkdownRegressionCheck {
    static func run() {
        for fence in ["```", "````", "~~~"] {
            let blocks = MarkdownParser.parse("\(fence)html\n<div>test</div>\n\(fence)")
            precondition(blocks.count == 1)
            guard case let .code(language, text) = blocks[0].kind else { preconditionFailure("Fence lost") }
            precondition(language == "html" && text == "<div>test</div>")
        }
        let source = String(repeating: "x", count: 300_000)
        let start = Date()
        let chunks = CodeTextChunker.chunks(source)
        precondition(chunks.map(\.source).joined() == source)
        precondition(chunks.allSatisfy { $0.source.count <= 2048 })
        let blocks = MarkdownParser.parse("```html\n" + source)
        precondition(blocks.count == 1)
        precondition(Date().timeIntervalSince(start) < 3, "Large code processing regressed")
        let table = "| A | B |\n| --- | --- |\n" + String(repeating: "| hello | world |\n", count: 3000)
        for chunk in StreamingTextChunker.chunks(table) {
            guard case .table = MarkdownParser.parse(chunk.source).first?.kind else { preconditionFailure("Table header lost") }
        }
        func mathCount(_ text: String) -> Int {
            InlineMathParser.parse(text).filter { if case .math = $0 { true } else { false } }.count
        }
        precondition(mathCount("Costs $5 and $10; `price $x$`") == 0)
        precondition(mathCount(#"Use $x^2$ and \(y\); ``code ` $z$``"#) == 2)
        let commentChunks = CodeTextChunker.highlightedChunks("/*\n" + String(repeating: "comment\n", count: 1000) + "*/\nlet x = 1")
        precondition(commentChunks[1].lexicalPrefix == "/*" && commentChunks[1].lexicalSuffix == "*/")
        print("Markdown checks passed: fences, 300k code, long tables")
    }
}

@MainActor
enum SyntaxHighlighter {
    private static let base = StudioTheme.adaptiveColor(
        light: NSColor(srgbRed: 0.10, green: 0.13, blue: 0.17, alpha: 1),
        dark: NSColor(srgbRed: 0.82, green: 0.85, blue: 0.89, alpha: 1))
    private static let comment = StudioTheme.adaptiveColor(
        light: NSColor(srgbRed: 0.32, green: 0.36, blue: 0.39, alpha: 1),
        dark: NSColor(srgbRed: 0.43, green: 0.49, blue: 0.54, alpha: 1))
    private static let string = StudioTheme.adaptiveColor(
        light: NSColor(srgbRed: 0.15, green: 0.39, blue: 0.20, alpha: 1),
        dark: NSColor(srgbRed: 0.62, green: 0.83, blue: 0.65, alpha: 1))
    private static let keyword = StudioTheme.adaptiveColor(
        light: NSColor(srgbRed: 0.08, green: 0.30, blue: 0.65, alpha: 1),
        dark: NSColor(srgbRed: 0.48, green: 0.73, blue: 0.98, alpha: 1))
    private static let type = StudioTheme.adaptiveColor(
        light: NSColor(srgbRed: 0.43, green: 0.22, blue: 0.62, alpha: 1),
        dark: NSColor(srgbRed: 0.74, green: 0.67, blue: 0.94, alpha: 1))
    private static let number = StudioTheme.adaptiveColor(
        light: NSColor(srgbRed: 0.55, green: 0.29, blue: 0.05, alpha: 1),
        dark: NSColor(srgbRed: 0.94, green: 0.72, blue: 0.43, alpha: 1))
    private static let function = StudioTheme.adaptiveColor(
        light: NSColor(srgbRed: 0.05, green: 0.36, blue: 0.40, alpha: 1),
        dark: NSColor(srgbRed: 0.48, green: 0.84, blue: 0.86, alpha: 1))
    private static var lexerCache: [String: NSRegularExpression] = [:]
    private static var markupRegexCache: [(NSRegularExpression, NSColor)]?

    struct Token: Sendable, Equatable {
        let range: NSRange
        /// Zero is base; 1...6 match the existing lexer groups.
        let color: Int
    }

    static func color(_ token: Int) -> NSColor {
        [base, comment, string, keyword, type, number, function,
         NSColor.systemGreen, NSColor.systemRed, NSColor.systemTeal][token]
    }

    /// Pure lexical output. No AppKit objects or mutable UI caches cross actors.
    nonisolated static func tokens(_ source: String, language raw: String, diff: Bool) -> [Token] {
        let full = NSRange(location: 0, length: source.utf16.count)
        guard full.length > 0 else { return [] }
        var colors = [UInt8](repeating: 0, count: full.length)
        func paint(_ range: NSRange, _ color: Int) {
            guard range.location != NSNotFound else { return }
            for index in range.location..<NSMaxRange(range) { colors[index] = UInt8(color) }
        }
        let language = normalized(raw)
        if diff {
            var offset = 0
            for line in source.components(separatedBy: "\n") {
                let count = line.utf16.count
                let color = line.hasPrefix("+") && !line.hasPrefix("+++") ? 7
                    : line.hasPrefix("-") && !line.hasPrefix("---") ? 8
                    : line.hasPrefix("@@") ? 9 : 0
                paint(NSRange(location: offset, length: count), color)
                offset += count + 1
            }
        } else if ["html", "xml", "svg", "vue", "svelte"].contains(language) {
            for (pattern, color) in markupPatterns {
                guard !Task.isCancelled,
                      let regex = try? NSRegularExpression(pattern: pattern, options: [.anchorsMatchLines]) else { continue }
                for match in regex.matches(in: source, range: full) { paint(match.range, color) }
            }
        } else if let regex = makeLexer(for: language) {
            for match in regex.matches(in: source, range: full) {
                if Task.isCancelled { return [] }
                for group in 1...6 where match.range(at: group).location != NSNotFound {
                    paint(match.range(at: group), group)
                    break
                }
            }
        }
        var runs: [Token] = []
        var start = 0
        for index in 1...colors.count where index == colors.count || colors[index] != colors[start] {
            runs.append(Token(range: NSRange(location: start, length: index - start), color: Int(colors[start])))
            start = index
        }
        return runs
    }

    nonisolated private static var markupPatterns: [(String, Int)] {
        [
            (#"<!--[\s\S]*?-->"#, 1),
            (#"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"#, 2),
            (#"</?\s*[A-Za-z][A-Za-z0-9:_-]*"#, 3),
            (#"\b[A-Za-z_:][A-Za-z0-9:._-]*(?=\s*=)"#, 6),
            (#"<!DOCTYPE[^>]*>|</?|/?>"#, 4),
            (#"&(?:[A-Za-z][A-Za-z0-9]+|#\d+|#x[0-9A-Fa-f]+);"#, 5),
        ]
    }

    static func highlight(_ text: String, language rawLanguage: String?, prefix: String = "", suffix: String = "") -> AttributedString {
        let source = prefix + text + suffix
        let result = NSMutableAttributedString(
            string: source,
            attributes: [.foregroundColor: base]
        )
        guard !source.isEmpty else { return AttributedString(result) }

        let language = normalized(rawLanguage)
        if ["html", "xml", "svg", "vue", "svelte"].contains(language) {
            highlightMarkup(result, source: source)
            return AttributedString(result.attributedSubstring(from: NSRange(location: prefix.utf16.count, length: text.utf16.count)))
        }

        guard let regex = lexer(for: language) else { return AttributedString(text) }

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
        return AttributedString(result.attributedSubstring(from: NSRange(location: prefix.utf16.count, length: text.utf16.count)))
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
        let regex = makeLexer(for: language)
        lexerCache[key] = regex
        return regex
    }

    nonisolated private static func makeLexer(for language: String) -> NSRegularExpression? {
        let definition = definition(for: language)
        let pattern = "(\(definition.comments))|(\(definition.strings))|\\b(\(alternation(definition.keywords)))\\b|\\b(\(alternation(definition.types)))\\b|(\\b(?:0[xX][0-9A-Fa-f](?:_?[0-9A-Fa-f])*|0[bB][01](?:_?[01])*|(?:\\d(?:_?\\d)*)?(?:\\.\\d(?:_?\\d)*)|\\d(?:_?\\d)*(?:[eE][+-]?\\d+)?)[fFdDuUlL]*\\b)|(\\b[A-Za-z_$][A-Za-z0-9_$]*(?=\\s*\\())"
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.anchorsMatchLines]
        ) else { return nil }
        return regex
    }

    nonisolated private static func lexerKey(for language: String) -> String {
        switch language {
        case "python", "swift", "javascript", "typescript", "bash", "json", "sql", "yaml", "toml", "markdown":
            language
        default:
            "generic"
        }
    }

    nonisolated private static func alternation(_ values: [String]) -> String {
        guard !values.isEmpty else { return "(?!)" }
        return "(?:" + values.map(NSRegularExpression.escapedPattern(for:)).joined(separator: "|") + ")"
    }

    nonisolated private static func normalized(_ rawLanguage: String?) -> String {
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

    nonisolated private static func definition(for language: String) -> SyntaxDefinition {
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

private struct SyntaxDefinition: Sendable {
    let comments: String
    let strings: String
    let keywords: [String]
    let types: [String]
}

struct CodeBlockView: View, Equatable {
    let language: String?
    let source: String
    let streaming: Bool
    var textScale: CGFloat = 1
    var htmlPreviewEnabled = false
    @EnvironmentObject private var studio: StudioModel
    @State private var showingPreview = false
    @State private var copied = false
    @State private var codeScroll = StreamingScrollState()
    @State private var chunkCache = CodeTextChunker.Cache()

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.language == rhs.language && lhs.streaming == rhs.streaming
            && lhs.textScale == rhs.textScale && lhs.htmlPreviewEnabled == rhs.htmlPreviewEnabled
    }

    private var canPreview: Bool {
        !streaming && HTMLPreview.canRun(language: language, source: source, completed: htmlPreviewEnabled)
    }
    private var lineCount: Int { source.utf8.reduce(1) { $1 == 10 ? $0 + 1 : $0 } }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                Image(systemName: "chevron.left.forwardslash.chevron.right")
                    .foregroundStyle(StudioTheme.quiet)
                Text(language?.uppercased() ?? "CODE")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .lineLimit(1).truncationMode(.middle)
                Spacer(minLength: 8)
                if streaming {
                    Text(L("Écriture…", "Writing…"))
                        .font(.system(size: 10)).foregroundStyle(StudioTheme.quiet)
                }
                if canPreview {
                    HStack(spacing: 2) {
                        modeButton("Code", icon: "chevron.left.forwardslash.chevron.right", preview: false)
                        modeButton(L("Aperçu", "Preview"), icon: "play", preview: true)
                    }
                    .padding(3).background(Color.white.opacity(0.04), in: RoundedRectangle(cornerRadius: 8))
                }
                Button(action: copySource) {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .foregroundStyle(copied ? StudioTheme.accent : StudioTheme.secondary)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(StudioControlStyle())
                .accessibilityIdentifier("code.copy")
                .accessibilityLabel(copied ? L("Copié", "Copied") : L("Copier le code", "Copy code"))
                .help(L("Copier tout le bloc", "Copy entire block"))
            }
            .padding(.horizontal, 14).frame(height: 48)
            .background(Color.white.opacity(0.025))
            Rectangle().fill(StudioTheme.edge).frame(height: 1)
            if showingPreview && canPreview && studio.studioDocument == nil {
                InlineHTMLPreview(source: source, workspace: studio.selectedMode == .agent ? studio.agentWorkspaceURL : nil)
                    .frame(height: 440)
            } else if !streaming {
                StudioCodeEditor(text: .constant(source), language: language ?? "text", editable: false,
                                  documentID: "inline-code", fontSize: 12.5 * textScale)
                     .frame(height: min(440, max(76, CGFloat(lineCount) * 19 * textScale + 34)))
            } else {
                streamingCode.padding(16)
            }
        }
        .background(StudioTheme.canvas)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12).stroke(StudioTheme.edge, lineWidth: 1)
                .allowsHitTesting(false)
        }
        .onChange(of: canPreview) { _, allowed in if !allowed { showingPreview = false } }
    }

    private func modeButton(_ label: String, icon: String, preview: Bool) -> some View {
        Button { showingPreview = preview } label: {
            Label(label, systemImage: icon)
                .font(.system(size: 10, weight: .medium))
                .padding(.horizontal, 9).frame(height: 26)
                .foregroundStyle(showingPreview == preview ? StudioTheme.ink : StudioTheme.quiet)
                .background(showingPreview == preview ? Color.white.opacity(0.085) : .clear,
                            in: RoundedRectangle(cornerRadius: 6))
        }
        .buttonStyle(.plain)
        .accessibilityIdentifier(preview ? "code.preview" : "code.source")
        .accessibilityAddTraits(showingPreview == preview ? .isSelected : [])
        .help(preview ? L("Aperçu isolé avec JavaScript et imports CDN", "Isolated preview with JavaScript and CDN imports") : L("Afficher le code source", "Show source code"))
    }

    @ViewBuilder private var streamingCode: some View {
        let chunks = chunkCache.highlightedChunks(source)
        if chunks.count > 8 {
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(chunks) { chunk in
                        CodeTextChunkView(
                            source: chunk.source, language: language,
                            prefix: chunk.lexicalPrefix, suffix: chunk.lexicalSuffix,
                            streaming: chunk.id == chunks.last?.id, textScale: textScale
                        ).equatable()
                    }
                }
            }
            .frame(height: 408)
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .defaultScrollAnchor(codeScroll.shouldFollow ? .bottom : nil, for: .sizeChanges)
            .onScrollPhaseChange { _, phase in codeScroll.phaseChanged(phase) }
            .onScrollGeometryChange(for: Bool.self) { g in
                g.contentSize.height - g.visibleRect.maxY < 60
            } action: { _, bottom in
                codeScroll.proximityChanged(bottom)
            }
        } else {
            VStack(alignment: .leading, spacing: 0) {
                ForEach(chunks) { chunk in
                    CodeTextChunkView(
                        source: chunk.source, language: language,
                        prefix: chunk.lexicalPrefix, suffix: chunk.lexicalSuffix,
                        streaming: chunk.id == chunks.last?.id, textScale: textScale
                    ).equatable()
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
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
    let prefix: String
    let suffix: String
    let streaming: Bool
    let textScale: CGFloat

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.language == rhs.language && lhs.prefix == rhs.prefix && lhs.suffix == rhs.suffix && lhs.streaming == rhs.streaming && lhs.textScale == rhs.textScale
    }

    var body: some View {
        SmoothTokenText(content: Text(SyntaxHighlighter.highlight(source, language: language, prefix: prefix, suffix: suffix)), source: source, streaming: streaming)
            .font(.system(size: 12 * textScale, weight: .regular, design: .monospaced))
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
    let textScale: CGFloat

    var body: some View {
        ScrollView(.horizontal) {
            VStack(alignment: .leading, spacing: 0) {
                row(header, header: true)
                ForEach(rows.indices, id: \.self) { index in
                    row(rows[index], header: false)
                }
            }
            .overlay {
                HStack(spacing: 0) {
                    ForEach(header.indices, id: \.self) { column in
                        Color.clear
                            .containerRelativeFrame(.horizontal) { [count = header.count, scale = textScale] width, _ in
                                max(260 * scale + 24, width / CGFloat(max(1, count)))
                            }
                            .overlay(alignment: .trailing) {
                                if column < header.count - 1 {
                                    Rectangle().fill(StudioTheme.contrast.opacity(0.07)).frame(width: 0.7)
                                }
                            }
                    }
                }
                .allowsHitTesting(false)
            }
            .background(StudioTheme.contrast.opacity(0.018), in: RoundedRectangle(cornerRadius: 11))
            .overlay {
                RoundedRectangle(cornerRadius: 11)
                    .stroke(StudioTheme.contrast.opacity(0.10), lineWidth: 0.7)
            }
            .clipShape(RoundedRectangle(cornerRadius: 11))
        }
        .scrollIndicators(.never)
    }

    private func row(_ values: [String], header isHeader: Bool) -> some View {
        HStack(alignment: .top, spacing: 0) {
            ForEach(header.indices, id: \.self) { column in
                cell(values[column], column: column, header: isHeader)
            }
        }
        .background(StudioTheme.contrast.opacity(isHeader ? 0.055 : 0))
        .overlay(alignment: .bottom) {
            Rectangle().fill(StudioTheme.contrast.opacity(0.07)).frame(height: 0.7)
        }
    }

    private func cell(_ source: String, column: Int, header isHeader: Bool) -> some View {
        SmoothTokenText(
            content: InlineLatexText.render(source, mathSize: 12.5 * textScale, htmlLineBreaks: true),
            source: source,
            streaming: streaming
        )
        .font(.system(size: 12.5 * textScale, weight: isHeader ? .semibold : .regular, design: .rounded))
        .foregroundStyle(StudioTheme.contrast.opacity(isHeader ? 0.92 : 0.76))
        .containerRelativeFrame(.horizontal, alignment: frameAlignment(column)) { [count = header.count, scale = textScale] width, _ in
            max(260 * scale, width / CGFloat(max(1, count)) - 24)
        }
        .fixedSize(horizontal: false, vertical: true)
        .padding(.horizontal, 12)
        .padding(.vertical, isHeader ? 10 : 9)
    }

    private func frameAlignment(_ column: Int) -> Alignment {
        guard alignments.indices.contains(column) else { return .topLeading }
        switch alignments[column] {
        case .leading: return .topLeading
        case .center: return .top
        case .trailing: return .topTrailing
        }
    }
}

@MainActor
private enum InlineLatexText {
    static func render(_ source: String, mathSize: CGFloat, htmlLineBreaks: Bool = false) -> Text {
        InlineMathParser.parse(source, htmlLineBreaks: htmlLineBreaks).reduce(Text("")) { result, piece in
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
                let formula = Text(Image(nsImage: image).renderingMode(.template)).baselineOffset(offset)
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
    static func parse(_ source: String, htmlLineBreaks: Bool = false) -> [InlineMathPiece] {
        let characters = Array(source)
        var pieces: [InlineMathPiece] = []
        var textStart = 0
        var index = 0
        var breaks: [Range<Int>] = []

        func appendProse(until end: Int) {
            guard end > textStart else { return }
            var prose = ""
            var start = textStart
            for range in breaks {
                prose += String(characters[start..<range.lowerBound]) + "\n"
                start = range.upperBound
            }
            prose += String(characters[start..<end])
            breaks.removeAll(keepingCapacity: true)
            pieces.append(.prose(prose))
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
            if matches(#"\("#, at: index, in: characters), !isEscaped(index, in: characters),
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
            if matches(#"\["#, at: index, in: characters), !isEscaped(index, in: characters),
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
                if delimiterLength == 1 && (index + 1 == characters.count || characters[index + 1].isWhitespace) {
                    index += 1
                    continue
                }
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
            if htmlLineBreaks, characters[index] == "<", !isEscaped(index, in: characters),
               index + 3 <= characters.count,
               String(characters[index..<(index + 3)]).lowercased() == "<br" {
                var end = index + 3
                while end < characters.count && (characters[end] == " " || characters[end] == "\t") {
                    end += 1
                }
                if end < characters.count && characters[end] == "/" { end += 1 }
                if end < characters.count && characters[end] == ">" {
                    breaks.append(index..<(end + 1))
                    index = end + 1
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
            if length == 1 && (characters[index] == "`" || characters[index] == "\n") { return nil }
            guard characters[index] == "$", !isEscaped(index, in: characters) else {
                index += 1
                continue
            }
            if length == 1 {
                let besideDollar = index + 1 < characters.count && characters[index + 1] == "$"
                let beforeSpace = index == start || characters[index - 1].isWhitespace
                let afterDigit = index + 1 < characters.count && characters[index + 1].isNumber
                if !besideDollar && !beforeSpace && !afterDigit {
                    return index
                }
                return nil
            } else if index + 1 < characters.count && characters[index + 1] == "$" {
                return index
            }
            index += 1
        }
        return nil
    }

    private static func indexAfterCodeSpan(startingAt start: Int, in characters: [Character]) -> Int {
        var index = start + 1
        while index < characters.count && characters[index] == "`" { index += 1 }
        let length = index - start
        while index < characters.count {
            if characters[index] == "`", !isEscaped(index, in: characters) {
                let run = index
                while index < characters.count && characters[index] == "`" { index += 1 }
                if index - run == length { return index }
                continue
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
    private static let cache: NSCache<NSString, NSImage> = {
        let cache = NSCache<NSString, NSImage>()
        cache.countLimit = 128
        cache.totalCostLimit = 32 * 1024 * 1024
        return cache
    }()

    static func image(
        latex: String,
        fontSize: CGFloat,
        mode: MTMathUILabelMode
    ) -> NSImage? {
        guard latex.count <= 4096 else { return nil }
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
        rendered.isTemplate = true
        cache.setObject(rendered, forKey: key, cost: Int(rendered.size.width * rendered.size.height * 4))
        return rendered
    }
}

private struct LatexBlockView: View {
    let latex: String
    let streaming: Bool
    let textScale: CGFloat
    @State private var visible = false

    var body: some View {
        Group {
            if let image = LatexImageCache.image(
                latex: latex,
                fontSize: 18 * textScale,
                mode: .display
            ) {
                ScrollView(.horizontal) {
                    Image(nsImage: image)
                        .renderingMode(.template)
                        .foregroundStyle(StudioTheme.ink)
                        .accessibilityLabel(latex)
                        .padding(.horizontal, 16)
                        .frame(maxWidth: .infinity, alignment: .center)
                }
                .scrollIndicators(.never)
            } else {
                Text("$$\n\(latex)\n$$")
                    .font(.system(size: 12 * textScale, design: .monospaced))
                    .foregroundStyle(StudioTheme.contrast.opacity(0.68))
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

private struct TokenFadeState {
  struct Batch {
    var count: Int
    let started: Double

    func opacity(at time: Double) -> Double {
      if time >= started + TokenFadeState.duration { return 1 }
      return 0.15 + 0.85 * max(0, (time - started) / TokenFadeState.duration)
    }
  }

  static let duration = 0.12
  private(set) var batches: [Batch] = []
  private var previousSource = ""
  private var previousCount = 0

  mutating func update(_ source: String, at time: Double, animated: Bool) {
    let count = source.count
    defer {
      previousSource = source
      previousCount = count
    }
    guard animated, source.utf8.starts(with: previousSource.utf8) else {
      batches.removeAll(keepingCapacity: true)
      return
    }
    batches.removeAll { time >= $0.started + Self.duration }
    let added = min(64, max(0, count - previousCount))
    if added > 0 { batches.append(Batch(count: added, started: time)) }
    var excess = batches.reduce(0) { $0 + $1.count } - 64
    while excess > 0 {
      let removed = min(excess, batches[0].count)
      batches[0].count -= removed
      excess -= removed
      if batches[0].count == 0 { batches.removeFirst() }
    }
  }
}

struct SmoothTokenText: View {
  let content: Text
  let source: String
  let streaming: Bool
  @Environment(\.accessibilityReduceMotion) private var reduceMotion
  @State private var fade = TokenFadeState()
  @State private var time: Double = 0

  private var animates: Bool { streaming && !reduceMotion && source.utf8.count <= 16_000 }

  var body: some View {
    Group {
      if animates {
        content.textRenderer(TokenFadeRenderer(batches: fade.batches, time: time))
      } else {
        content
      }
    }
    .fixedSize(horizontal: false, vertical: true)
    .task(id: source) {
      guard animates else { fade = TokenFadeState(); return }
      let now = ProcessInfo.processInfo.systemUptime
      var transaction = Transaction(animation: nil)
      transaction.disablesAnimations = true
      withTransaction(transaction) {
        fade.update(source, at: now, animated: true)
        time = now
      }
      guard !fade.batches.isEmpty else { return }
      await Task.yield()
      guard !Task.isCancelled else { return }
      // Advancing one clock preserves the age of earlier fragments.
      withAnimation(.linear(duration: TokenFadeState.duration)) {
        time = now + TokenFadeState.duration
      }
    }
  }
}

private struct TokenFadeRenderer: TextRenderer {
  let batches: [TokenFadeState.Batch]
  var time: Double
  var animatableData: Double {
    get { time }
    set { time = newValue }
  }
  func draw(layout: Text.Layout, in context: inout GraphicsContext) {
    guard batches.contains(where: { $0.opacity(at: time) < 1 }) else {
      for line in layout { context.draw(line) }
      return
    }
    let total = layout.reduce(0) { $0 + $1.reduce(0) { $0 + $1.count } }
    let fading = batches.reduce(0) { $0 + $1.count }
    var opaque = max(0, total - fading)
    var skipped = max(0, fading - total)
    var batch = 0
    while batch < batches.count, skipped >= batches[batch].count {
      skipped -= batches[batch].count
      batch += 1
    }
    var remaining = batch < batches.count ? batches[batch].count - skipped : 0
    for line in layout {
      for run in line {
        var start = run.startIndex
        while start < run.endIndex {
          let count = min(run.endIndex - start, opaque > 0 ? opaque : remaining)
          var copy = context
          if opaque > 0 {
            opaque -= count
          } else {
            copy.opacity *= batches[batch].opacity(at: time)
            remaining -= count
          }
          copy.draw(run[start..<start + count])
          start += count
          if opaque == 0, remaining == 0, batch < batches.count - 1 {
            batch += 1
            remaining = batches[batch].count
          }
        }
      }
    }
  }
}

@MainActor
enum GeneratedHTML {
    static func documents(_ source: String) -> [String] {
        if HTMLPreview.canRun(language: nil, source: source, completed: true) { return [source] }
        return Array(MarkdownParser.parse(source).compactMap { block in
            if case let .code(language, text) = block.kind, HTMLPreview.canRun(language: language, source: text, completed: true) { return text }
            return nil
        }.prefix(8))
    }
}
