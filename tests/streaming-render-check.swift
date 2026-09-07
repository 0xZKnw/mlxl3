// Run from the repository root in zsh; set MLXL3_RENDER_BENCHMARK=1 for timings:
// test_dir="$(mktemp -d -t mlxl3-render-check)"
// CLANG_MODULE_CACHE_PATH="$test_dir/modules" swift build --disable-sandbox --package-path apps/MLXL3Studio
// sources=(apps/MLXL3Studio/Sources/MLXL3Studio/*.swift)
// sources=("${(@)sources:#*/MLXL3StudioApp.swift}")
// swiftc -O -swift-version 6 -parse-as-library -module-cache-path "$test_dir/modules" -I apps/MLXL3Studio/.build/arm64-apple-macosx/debug/Modules "${sources[@]}" apps/MLXL3Studio/.build/arm64-apple-macosx/debug/SwiftMath.build/*.swift.o tests/streaming-render-check.swift -o "$test_dir/check"
// "$test_dir/check"
import AppKit
import SwiftUI

@MainActor private final class StateProbe {
    var value = ""
    var update: (() -> Void)?
}

private struct EquatableStateProbe: View, Equatable {
    let probe: StateProbe
    @State private var expanded = false
    @State private var copied = false
    @State private var followsBottom = true
    nonisolated static func == (lhs: Self, rhs: Self) -> Bool { true }
    var body: some View {
        probe.value = "\(expanded)/\(copied)/\(followsBottom)/" + L("fr", "en")
        probe.update = { expanded.toggle(); copied.toggle(); followsBottom.toggle() }
        return Text(probe.value)
    }
}

@main struct StreamingRenderCheck {
    @MainActor static func main() {
        let probe = StateProbe()
        let host = NSHostingView(rootView: EquatableStateProbe(probe: probe).equatable().id(AppLanguage.fr))
        host.frame = NSRect(x: 0, y: 0, width: 200, height: 100)
        host.layoutSubtreeIfNeeded()
        precondition(probe.value == "false/false/true/fr")
        probe.update?()
        host.layoutSubtreeIfNeeded()
        precondition(probe.value == "true/true/false/fr", "Equatable view suppressed local state changes")
        AppLocalization.set(.en)
        host.rootView = EquatableStateProbe(probe: probe).equatable().id(AppLanguage.en)
        host.layoutSubtreeIfNeeded()
        precondition(probe.value == "false/false/true/en", "Language identity did not refresh the view")
        AppLocalization.set(.fr)
        MarkdownRegressionCheck.run()
        for grapheme in ["a", "é", "e\u{301}", "👩‍💻", "🇫🇷", "\r\n"] {
            let target = CodeTextChunker.targetCharacters
            for count in [target - 1, target, target + 1] {
                let source = String(repeating: grapheme, count: count)
                let chunks = CodeTextChunker.chunks(source)
                precondition(chunks.count == (count + target - 1) / target)
                precondition(chunks.allSatisfy { $0.source.count <= target })
                precondition(chunks.map(\.source).joined().utf8.elementsEqual(source.utf8))
            }
            for prefixCount in [target - 2, target - 1] {
                let prefix = String(repeating: grapheme, count: prefixCount) + "\n"
                let source = prefix + grapheme + grapheme
                let chunks = CodeTextChunker.chunks(source)
                precondition(chunks.count == 2 && chunks[0].source == prefix && chunks[1].source == grapheme + grapheme)
            }
            let boundary = String(repeating: grapheme, count: StreamingTextChunker.targetCharacters - 2) + "\n\n"
            precondition(StreamingTextChunker.chunks(boundary).count == 1)
            let extended = StreamingTextChunker.chunks(boundary + grapheme)
            precondition(extended.count == 2 && extended[0].source == boundary && extended[1].source == grapheme)
        }
        let cache = CodeTextChunker.Cache()
        precondition(cache.highlightedChunks("") == CodeTextChunker.highlightedChunks(""))
        let lexemes = [
            ("/*", "*/"), ("<!--", "-->"), ("\"\"\"", "\"\"\""), ("'''", "'''"),
            ("\"", "\""), ("'", "'"), ("`", "`"), ("//", "\n"), ("#", "\n"),
        ]
        var growing = ""
        var fragmentIndex = 0
        for (opener, closer) in lexemes {
            growing = ""
            let escaped = "é🙂e\u{301}" + String(repeating: "x", count: 255) + "\\\\x\\\" "
            let text = "\nlet value = 42\n" + opener + String(repeating: escaped, count: 32) + closer + "\n"
            var remainder = text[...]
            while !remainder.isEmpty {
                let length = [1, 2, 3, 61, 257][fragmentIndex % 5]
                let end = remainder.index(remainder.startIndex, offsetBy: length, limitedBy: remainder.endIndex) ?? remainder.endIndex
                growing += remainder[..<end]
                remainder = remainder[end...]
                let actual = cache.highlightedChunks(growing)
                let expected = CodeTextChunker.highlightedChunks(growing)
                if actual != expected {
                    let detail = "Incremental lexemes diverged at fragment \(fragmentIndex): \(actual.count) / \(expected.count) chunks\n"
                    FileHandle.standardError.write(Data(detail.utf8))
                    for (a, b) in zip(actual, expected) where a != b {
                        FileHandle.standardError.write(Data("actual: id=\(a.id), bytes=\(a.source.utf8.count), prefix=\(a.lexicalPrefix.debugDescription), suffix=\(a.lexicalSuffix.debugDescription)\nexpected: id=\(b.id), bytes=\(b.source.utf8.count), prefix=\(b.lexicalPrefix.debugDescription), suffix=\(b.lexicalSuffix.debugDescription)\n".utf8))
                        break
                    }
                    preconditionFailure("Incremental lexemes diverged")
                }
                fragmentIndex += 1
            }
        }
        for replacement in ["", "x", growing, "//" + growing.dropFirst(2), String(growing.prefix(5_000)), growing.precomposedStringWithCanonicalMapping, growing.decomposedStringWithCanonicalMapping] {
            precondition(cache.highlightedChunks(replacement) == CodeTextChunker.highlightedChunks(replacement), "Replacement was treated as an append")
        }
        for quote in ["\"", "'", "`"] {
            for escape in [["\\", "\n"], ["\\\n"]] {
                var partial = quote + String(repeating: "x", count: 9_000)
                precondition(cache.highlightedChunks(partial) == CodeTextChunker.highlightedChunks(partial))
                for fragment in escape + ["value", quote] {
                    partial += fragment
                    precondition(cache.highlightedChunks(partial) == CodeTextChunker.highlightedChunks(partial), "Escaped newline changed previous chunks")
                }
            }
        }
        var combined = String(repeating: "x", count: 6_000)
        for fragment in ["e", "\u{301}", "👩", "\u{200D}", "💻", "\r", "\n"] {
            combined += fragment
            precondition(cache.highlightedChunks(combined) == CodeTextChunker.highlightedChunks(combined), "Unicode fragment changed a stable boundary")
        }
        print("Incremental code checks passed: \(fragmentIndex) Unicode/escaped/multiline fragments and replacements")

        let source = String(repeating: "A sentence with words and punctuation.\n\n", count: 6_720)
        let frozen = MarkdownResponseView(source)
        precondition(frozen == MarkdownResponseView(source))
        precondition(frozen != MarkdownResponseView(source, streaming: true))
        precondition(MarkdownResponseView("a") != MarkdownResponseView("b"))

        guard ProcessInfo.processInfo.environment["MLXL3_RENDER_BENCHMARK"] == "1" else { return }

        // Isolate the work avoided by an unchanged view; this is not a frame-rate benchmark.
        var processed = 0
        let start = Date()
        for _ in 0..<100 { processed += StreamingTextChunker.chunks(source).count }
        let full = Date().timeIntervalSince(start)
        let gatedStart = Date()
        for _ in 0..<100 where frozen != MarkdownResponseView(source) {
            processed += StreamingTextChunker.chunks(source).count
        }
        let gated = Date().timeIntervalSince(gatedStart)
        precondition(processed > 0)
        print(String(format: "Frozen 269 KB text, 100 updates: rechunk %.3f ms; equality gate %.3f ms", full * 1_000, gated * 1_000))

        for bytes in [65_536, 262_144, 1_048_576] {
            let code = String(repeating: "let value = 42 // comment\n", count: bytes / 26)
            let incremental = CodeTextChunker.Cache()
            _ = incremental.highlightedChunks(code)
            var monolithicSeconds = 0.0
            var incrementalSeconds = 0.0
            var current = code
            for _ in 0..<30 {
                current += "let next = 42\n"
                let start = Date()
                let expected = CodeTextChunker.highlightedChunks(current)
                monolithicSeconds += Date().timeIntervalSince(start)
                let incrementalStart = Date()
                let actual = incremental.highlightedChunks(current)
                incrementalSeconds += Date().timeIntervalSince(incrementalStart)
                precondition(actual == expected)
            }
            print(String(format: "Active code %d bytes, preparation ms/update: monolithic %.3f; incremental %.3f", bytes, monolithicSeconds * 1_000 / 30, incrementalSeconds * 1_000 / 30))
        }
    }
}
