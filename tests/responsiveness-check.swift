import AppKit
import Foundation
import SwiftUI

@main struct ResponsivenessCheck {
    @MainActor static func main() async throws {
        var checks = 0
        func check(_ value: Bool, _ label: String) {
            precondition(value, label)
            checks += 1
        }
        let project = UUID()
        var documents = (0..<20).map {
            LibraryDocument(projectID: project, fileName: "\($0).txt", originalPath: "/\($0)", text: $0 == 17 ? "alpha beta gamma" : "alpha")
        }
        let worker = DocumentSearchWorker()
        let first = await worker.search("alpha beta gamma", documents: documents, limit: 5)
        check(first.map(\.fileName) == ["17.txt", "0.txt", "1.txt", "2.txt", "3.txt"], "top-k score and stable ties")
        let initialBuilds = await worker.buildCount
        let again = await worker.search("alpha", documents: documents, limit: 99)
        let reusedBuilds = await worker.buildCount
        check(initialBuilds == 20 && reusedBuilds == 20 && again.count == 8, "cache reuse and hard result cap")
        documents[0].text = "delta"
        documents[1].fileName = "renamed.txt"
        let changed = await worker.search("delta", documents: documents)
        let changedBuilds = await worker.buildCount
        check(changedBuilds == 22 && changed.first?.fileName == "0.txt", "content and metadata invalidation")
        let empty = await worker.search("delta", documents: [])
        check(empty.isEmpty, "removed documents cannot leak excerpts")
        check(DocumentSearch.search("alpha", documents: documents, limit: -1).isEmpty, "negative limit")
        check(DocumentSearch.search("a !", documents: documents).isEmpty, "no meaningful terms")
        let pdf = LibraryDocument(projectID: project, fileName: "book.pdf", originalPath: "", text: "[Page 2]\nÉtoile alpha\n[Page 3]\nbeta")
        let pdfResults = DocumentSearch.search("etoile beta", documents: [pdf])
        check(pdfResults.map(\.page) == [2, 3], "PDF boundaries and Unicode folding")
        let long = LibraryDocument(projectID: project, fileName: "long.txt", originalPath: "", text: String(repeating: "alpha ", count: 1000))
        check(DocumentSearch.search("alpha", documents: [long], limit: 8).allSatisfy { $0.text.count <= 1200 }, "excerpt bound")

        // Independent oracle is a full rebuild from the actual edited string.
        for source in ["", "a\nb\n", "😀\né\n終", "a\r\nb", "no newline"] {
            let boundaries = Array(source.indices) + [source.endIndex]
            for lower in boundaries.indices {
                for upper in lower..<boundaries.count {
                    let range = NSRange(boundaries[lower]..<boundaries[upper], in: source)
                    for replacement in ["", "\n", "😀\nx\n", "z"] {
                        let changed = (source as NSString).replacingCharacters(in: range, with: replacement)
                        let incremental = SourceLineOffsets.replacing(SourceLineOffsets.build(source), range: range, replacement: replacement)
                        check(incremental == SourceLineOffsets.build(changed), "incremental UTF-16 offsets")
                    }
                }
            }
        }
        let message = ChatMessage(role: .assistant, content: "before")
        var conversation = Conversation(messages: [message])
        let saved = conversation.snapshot
        message.append(" after", phase: nil)
        let updated = conversation.snapshot
        check(saved.messages[0].content == "before" && updated.messages[0].content == "before after", "immutable cached snapshots")
        message.sources = pdfResults
        message.turnContext = "new context"
        check(conversation.snapshot.messages[0].sources == pdfResults && conversation.snapshot.messages[0].turnContext == "new context", "source/context mutation invalidates cache")
        conversation.messages = [ChatMessage(id: message.id, role: .assistant, content: "replacement")]
        check(conversation.snapshot.messages[0].content == "replacement", "same UUID different object invalidates cache")
        var copy = conversation
        copy.messages.append(ChatMessage(role: .user, content: "copy only"))
        check(conversation.snapshot.messages.count == 1 && copy.snapshot.messages.count == 2,
              "value copies have independent membership caches")
        let shared = conversation.messages[0]
        shared.append(" shared edit", phase: nil)
        check(copy.snapshot.messages[0].content.hasSuffix("shared edit") &&
              conversation.snapshot.messages[0].content.hasSuffix("shared edit"),
              "multiple snapshot observers see the same message mutation")
        conversation.updatedAt = Date(timeIntervalSince1970: 1234)
        check(Conversation(snapshot: conversation.snapshot).updatedAt == conversation.updatedAt, "activity date persists")

        let attachment = ChatAttachment(fileName: "missing.txt", text: "retained content", fileBytes: 16)
        let draft = ConversationDraft(conversationID: conversation.id, text: "unsent", attachments: [attachment])
        var workspace = WorkspaceSnapshot(selectedConversationID: conversation.id, selectedModelName: nil,
            conversations: [conversation.snapshot], temperature: 0, topK: 1, repetitionPenalty: 1, systemPrompt: "")
        workspace.drafts = [draft]
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-responsive-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let file = root.appendingPathComponent("history.json")
        try ConversationStore.write(workspace, to: file)
        check(try ConversationStore.load(from: file)?.drafts?.first?.attachments == [attachment], "draft restore uses embedded text")
        let previous = try Data(contentsOf: file)
        try ConversationStore.write(workspace, to: file)
        check(try Data(contentsOf: file.appendingPathExtension("backup")) == previous, "exact previous bytes backed up")
        try Data("broken".utf8).write(to: file)
        do { try ConversationStore.write(workspace, to: file); preconditionFailure("corrupt history overwritten") }
        catch { check(try Data(contentsOf: file) == Data("broken".utf8), "corrupt old history preserved") }
        workspace.drafts = [ConversationDraft(conversationID: UUID(), text: "orphan", attachments: [])]
        do { try ConversationStore.validate(workspace); preconditionFailure("orphan draft accepted") }
        catch { checks += 1 }
        workspace.drafts = [ConversationDraft(conversationID: conversation.id, text: String(repeating: "x", count: 65537), attachments: [])]
        do { try ConversationStore.validate(workspace); preconditionFailure("oversized draft accepted") }
        catch { checks += 1 }

        _ = NSApplication.shared
        let editor = NSTextView()
        editor.allowsUndo = true
        let coordinator = StudioCodeEditor.Coordinator(StudioCodeEditor(text: .constant(""), language: "swift", editable: true, documentID: "test"))
        for length in [200 * 1024, 256 * 1024, 2 * 1024 * 1024] {
            let text = "let emoji = \"😀\"\n" + String(repeating: "x", count: length)
            editor.string = text
            editor.setSelectedRange(NSRange(location: 4, length: 5))
            coordinator.resetLines(editor)
            coordinator.style(editor)
            await coordinator.waitForStyling()
            check(editor.string == text && editor.selectedRange() == NSRange(location: 4, length: 5), "style preserves characters/selection at size boundary")
        }
        editor.string = "abc"
        editor.setSelectedRange(NSRange(location: 3, length: 0))
        editor.insertText("def", replacementRange: editor.selectedRange())
        coordinator.style(editor)
        await coordinator.waitForStyling()
        editor.undoManager?.undo()
        check(editor.string == "abc", "recoloring preserves undo")
        editor.setMarkedText("é", selectedRange: NSRange(location: 1, length: 0), replacementRange: NSRange(location: 0, length: 0))
        let marked = editor.markedRange(), markedText = editor.string
        coordinator.style(editor)
        await coordinator.waitForStyling()
        check(editor.markedRange() == marked && editor.string == markedText, "marked IME text untouched")
        editor.unmarkText()
        let server = LocalServer()
        server.start(model: "unused", port: 8000, context: 1024,
                     executable: URL(fileURLWithPath: "/not-an-engine"))
        check(server.active && server.busy, "resolution is a visible active operation")
        server.stop()
        for _ in 0..<10 { await Task.yield() }
        check(!server.active && !server.busy && server.error == nil,
              "Stop invalidates an unresolved launch without starting a child")
        server.start(model: "unused", port: 1, context: 1024,
                     executable: URL(fileURLWithPath: "/not-an-engine"))
        for _ in 0..<100 where server.active { try await Task.sleep(for: .milliseconds(10)) }
        check(!server.active && !server.busy && server.error != nil,
              "invalid port releases resolution state and reports failure")
        // Compare detached lexical runs to the existing production highlighter,
        // including multiline state and non-BMP ranges. Not a duplicated lexer.
        for (language, text) in [
            ("swift", "/* 😀\n comment */\nlet x = \"\"\"\ntext\n\"\"\"\n"),
            ("python", "# comment\nx = '''multi\nline'''\nreturn 123"),
            ("html", "<!-- 😀 -->\n<div class=\"x\">&amp;</div>"),
            ("javascript", "const x = `multi\nline`; // end")
        ] {
            let tokens = await Task.detached { SyntaxHighlighter.tokens(text, language: language, diff: false) }.value
            let expected = NSAttributedString(SyntaxHighlighter.highlight(text, language: language))
            check(tokens.reduce(0) { $0 + $1.range.length } == expected.length, "lexer spans cover UTF-16 buffer")
            for token in tokens {
                for index in token.range.location..<NSMaxRange(token.range) {
                    let color = expected.attribute(.foregroundColor, at: index, effectiveRange: nil) as? NSColor
                    check(color == SyntaxHighlighter.color(token.color), "background lexer agrees with old color oracle")
                }
            }
        }
        editor.string = "let old = 123"
        coordinator.resetLines(editor)
        coordinator.style(editor)
        editor.string = "// new 😀"
        coordinator.resetLines(editor)
        coordinator.style(editor)
        await coordinator.waitForStyling()
        check(editor.string == "// new 😀" && (editor.textStorage?.attribute(.foregroundColor, at: 0, effectiveRange: nil) as? NSColor) == SyntaxHighlighter.color(1),
              "old lexical job cannot overwrite a new document")
        print("Responsiveness: \(checks) checks passed")
    }
}
