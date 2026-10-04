import AppKit
import CoreText
import PDFKit
import SwiftUI

private struct CheckFailure: Error { let message: String }

@main struct FileImportCheck {
  static func require(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
    if try !condition() { throw CheckFailure(message: message) }
  }

  static func rejects(_ expected: AttachmentImportError, _ operation: () throws -> Void) throws {
    do {
      try operation()
    } catch let error as AttachmentImportError {
      try require(
        error == expected,
        "Expected \(expected), got \(error)")
      return
    }
    throw CheckFailure(message: "Expected rejection: \(expected)")
  }

  static func pdf(at url: URL, pages: [String?], locked: Bool = false) throws {
    var box = CGRect(x: 0, y: 0, width: 612, height: 792)
    let options: CFDictionary? =
      locked
      ? [kCGPDFContextUserPassword: "test-password", kCGPDFContextOwnerPassword: "test-owner"]
        as CFDictionary
      : nil
    guard let context = CGContext(url as CFURL, mediaBox: &box, options) else {
      throw CheckFailure(message: "Cannot create PDF fixture")
    }
    for text in pages {
      context.beginPDFPage(nil)
      if let text {
        context.textPosition = CGPoint(x: 40, y: 700)
        let line = CTLineCreateWithAttributedString(
          NSAttributedString(
            string: text, attributes: [.font: NSFont.systemFont(ofSize: 14)]
          ) as CFAttributedString)
        CTLineDraw(line, context)
      }
      context.endPDFPage()
    }
    context.closePDF()
  }

  @MainActor static func main() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "mlxl3-import-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func file(_ name: String, _ data: Data) throws -> URL {
      let url = root.appendingPathComponent(name)
      try data.write(to: url)
      return url
    }
    let originalText = "Projet CEDAR-42\nBudget : 73 euros.\nÉté 👋\n"
    let txt = try file("notes.txt", Data(originalText.utf8))
    let attachment = try ChatAttachmentImporter.read(txt)
    try require(
      attachment.text == originalText && attachment.fileBytes == originalText.utf8.count,
      "UTF-8 changed")
    for (name, data) in [
      ("bom.md", Data([0xEF, 0xBB, 0xBF]) + Data(originalText.utf8)),
      ("le.txt", Data([0xFF, 0xFE]) + originalText.data(using: .utf16LittleEndian)!),
      ("be.txt", Data([0xFE, 0xFF]) + originalText.data(using: .utf16BigEndian)!),
      ("source.rs", Data("fn main() {}\n".utf8)),
      ("config.yaml", Data("value: 73\n".utf8)),
      ("data.csv", Data("name,value\nCEDAR,73\n".utf8)),
      ("README", Data(originalText.utf8)),
    ] {
      let parsed = try ChatAttachmentImporter.read(file(name, data))
      try require(
        !parsed.text.isEmpty && !parsed.text.hasPrefix("\u{FEFF}"), "BOM or format failed: \(name)")
    }
    try rejects(.invalidText) {
      _ = try ChatAttachmentImporter.read(file("bad.txt", Data([0xC3, 0x28])))
    }
    try rejects(.invalidText) {
      _ = try ChatAttachmentImporter.read(file("binary.txt", Data([65, 0, 66])))
    }
    for bytes: [UInt8] in [[0xFF, 0xFE, 65], [0xFF, 0xFE, 0, 0xD8]] {
      try rejects(.invalidText) { _ = try ChatAttachmentImporter.decodeText(Data(bytes)) }
    }
    try rejects(.empty) {
      _ = try ChatAttachmentImporter.read(file("empty.txt", Data(" \t\n".utf8)))
    }
    try rejects(.unsupported) {
      _ = try ChatAttachmentImporter.read(file("image.png", Data("not an image".utf8)))
    }
    try rejects(.notAFile) { _ = try ChatAttachmentImporter.read(root) }
    try rejects(.notAFile) {
      _ = try ChatAttachmentImporter.read(URL(string: "https://example.invalid/document.txt")!)
    }
    do {
      _ = try ChatAttachmentImporter.read(root.appendingPathComponent("missing.txt"))
      throw CheckFailure(message: "Missing file was accepted")
    } catch is CocoaError {}
    let large = try file(
      "large.txt", Data(repeating: 65, count: ChatAttachmentImporter.maxFileBytes + 1))
    try rejects(.fileTooLarge) { _ = try ChatAttachmentImporter.read(large) }
    let tooMuchText = try file(
      "text-limit.txt", Data(repeating: 65, count: ChatAttachmentImporter.maxTextBytes + 1))
    try rejects(.textTooLarge) { _ = try ChatAttachmentImporter.read(tooMuchText) }
    let exact = try ChatAttachmentImporter.read(
      file("exact.txt", Data(repeating: 65, count: ChatAttachmentImporter.maxTextBytes)))
    try require(
      exact.text.utf8.count == ChatAttachmentImporter.maxTextBytes, "Exact text limit rejected")

    let document = root.appendingPathComponent("report.pdf")
    try pdf(at: document, pages: ["Budget CEDAR-42: 73 euros", nil, "Conclusion: accepted"])
    let extracted = try ChatAttachmentImporter.read(document)
    try require(
      extracted.text.contains("CEDAR-42") && extracted.text.contains("73 euros"),
      "PDF text not extracted")
    try require(
      extracted.text.contains("[Page 1]") && extracted.text.contains("[Page 3]"),
      "PDF page references lost")
    let scan = root.appendingPathComponent("scan.pdf")
    try pdf(at: scan, pages: [nil])
    try rejects(.empty) { _ = try ChatAttachmentImporter.read(scan) }
    let locked = root.appendingPathComponent("locked.pdf")
    try pdf(at: locked, pages: ["secret"], locked: true)
    try rejects(.lockedPDF) { _ = try ChatAttachmentImporter.read(locked) }
    try rejects(.invalidPDF) {
      _ = try ChatAttachmentImporter.read(file("broken.pdf", Data("broken".utf8)))
    }
    let batch = ChatAttachmentImporter.load([txt, scan, document, txt], existing: [])
    try require(
      batch.attachments.count == 2 && batch.errors.count == 1,
      "Partial import or deduplication failed")

    // Finite exhaustive admission check on the production function: all counts 0...9
    // and sizes around the individual and cumulative byte boundaries.
    var checked = 0
    let sizes = [
      0, 1, ChatAttachmentImporter.maxTextBytes - 1, ChatAttachmentImporter.maxTextBytes,
      ChatAttachmentImporter.maxTextBytes + 1,
    ]
    for count in 0...9 {
      for used in [
        0, ChatAttachmentImporter.maxTotalTextBytes - 1, ChatAttachmentImporter.maxTotalTextBytes,
      ] {
        let previous = (0..<count).map { index in
          ChatAttachment(
            fileName: "previous-\(index)",
            text: index == 0 ? String(repeating: "a", count: used) : "", fileBytes: 0)
        }
        let actualUsed = count == 0 ? 0 : used
        for size in sizes {
          let candidate = ChatAttachment(
            fileName: "candidate", text: String(repeating: "b", count: size), fileBytes: size)
          let shouldAccept = count < 8 && size <= 262_144 && actualUsed + size <= 524_288
          do {
            let combined = try ChatAttachmentImporter.adding(candidate, to: previous)
            try require(shouldAccept && combined.count == count + 1, "Admission violated limits")
          } catch is AttachmentImportError {
            try require(!shouldAccept, "Valid boundary rejected")
          }
          checked += 1
        }
      }
    }
    try require(
      try ChatAttachmentImporter.adding(attachment, to: [attachment]) == [attachment],
      "Duplicate changed draft")
    let eight = try (0..<8).map { try file("file-\($0).txt", Data("item \($0)".utf8)) }
    let full = ChatAttachmentImporter.load(eight, existing: [])
    try require(full.attachments.count == 8 && full.errors.isEmpty, "Eight files not accepted")
    let duplicateAtLimit = ChatAttachmentImporter.load([eight[0]], existing: full.attachments)
    try require(
      duplicateAtLimit.attachments.count == 8 && duplicateAtLimit.errors.isEmpty,
      "Duplicate at file limit failed")
    let ninth = ChatAttachmentImporter.load([document], existing: full.attachments)
    try require(ninth.attachments.count == 8 && ninth.errors.count == 1, "Ninth file not rejected")
    try require(
      ChatAttachment.promptContent("question", attachments: []) == "question",
      "Unattached prompt changed")

    // Sampled round-trip checks use an independent seeded generator and multiple encodings.
    var seed: UInt64 = 73
    let alphabet = Array("abcXYZ09 \t\n\ré🦀中")
    for iteration in 0..<128 {
      var value = ""
      for _ in 0..<(iteration + 1) {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1
        value.append(alphabet[Int((seed >> 32) % UInt64(alphabet.count))])
      }
      for data in [Data(value.utf8), Data([0xFF, 0xFE]) + value.data(using: .utf16LittleEndian)!] {
        try require(
          try ChatAttachmentImporter.decodeText(data) == value, "Seeded encoding round trip failed")
      }
    }

    let suite = "io.mlxl3.file-import-check." + UUID().uuidString
    let prefs = UserDefaults(suiteName: suite)!
    defer { prefs.removePersistentDomain(forName: suite) }
    setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
    let history = root.appendingPathComponent("history.json")
    let studio = StudioModel(conversationFileURL: history, preferences: prefs)
    defer { studio.ejectModel() }
    let first = studio.selectedConversationID!
    await studio.importChatFiles([txt, document])
    try require(
      studio.pendingAttachments.count == 2 && !studio.canSend,
      "Import requires a ready engine before sending")
    studio.newConversation()
    let second = studio.selectedConversationID!
    try require(studio.pendingAttachments.isEmpty, "Attachments leaked into a new conversation")
    await studio.importChatFiles([txt])
    studio.selectConversation(first)
    try require(studio.pendingAttachments.count == 2, "Attachment draft lost on switching")
    studio.removeChatAttachment(studio.pendingAttachments[1].id)
    try require(studio.pendingAttachments.count == 1, "Removal failed")
    await studio.importChatFiles([scan])
    try require(
      studio.attachmentImportError != nil && studio.pendingAttachments.count == 1,
      "Failed import erased draft")
    studio.dismissAttachmentImportError()
    try require(studio.attachmentImportError == nil, "Error dismissal failed")

    studio.models = [
      LocalModel(
        name: "file-import", path: root.path, modelType: "fixture", format: "EXL3", bits: 3,
        sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
    ]
    studio.selectModel("file-import")
    for _ in 0..<250 where !studio.engineState.isReady {
      try await Task.sleep(for: .milliseconds(20))
    }
    try require(studio.engineState.isReady && studio.canSend, "Attachment-only send unavailable")
    studio.send()
    try require(
      studio.pendingAttachments.isEmpty && !studio.canSend, "Send did not consume attachment draft")
    for _ in 0..<250 where studio.isGenerating { try await Task.sleep(for: .milliseconds(20)) }
    let user = studio.currentConversation!.messages[0]
    let reply = studio.currentConversation!.messages[1].content
    let sent = try JSONSerialization.jsonObject(with: Data(reply.utf8)) as! [[String: Any]]
    try require(
      (sent.last?["content"] as? String)?.contains(originalText) == true,
      "Production bridge omitted attachment text")
    try require(
      user.attachments == [attachment] || user.attachments.first?.text == originalText,
      "User history lost attachment")
    try require(!user.content.contains(originalText), "Document text exposed as user message")
    try require(studio.persistNow(), "History not saved")
    try FileManager.default.removeItem(at: txt)
    let reopened = StudioModel(conversationFileURL: history, isPreview: true, preferences: prefs)
    try require(
      reopened.currentConversation?.messages[0].attachments.first?.text == originalText,
      "Saved attachment depends on original file")
    let old = ChatMessage(role: .user, content: "legacy message")
    let legacy = try JSONDecoder().decode(
      ChatMessageSnapshot.self, from: JSONEncoder().encode(old.snapshot))
    try require(
      ChatMessage(snapshot: legacy)?.attachments.isEmpty == true,
      "Legacy history migration failed")

    studio.draft = "What was the budget?"
    studio.send()
    for _ in 0..<250 where studio.isGenerating { try await Task.sleep(for: .milliseconds(20)) }
    let followUp =
      try JSONSerialization.jsonObject(
        with: Data(studio.currentConversation!.messages.last!.content.utf8)) as! [[String: Any]]
    try require(
      (followUp.first?["content"] as? String)?.contains("CEDAR-42") == true,
      "Follow-up lost attached context")
    studio.selectConversation(second)
    try require(studio.pendingAttachments.count == 1, "Send modified another conversation's draft")
    studio.deleteConversation(second)
    try require(studio.pendingAttachments.isEmpty, "Deletion retained pending attachments")

    // Switching during extraction must deliver to the captured destination.
    studio.newConversation()
    let destination = studio.selectedConversationID!
    let importing = Task { await studio.importChatFiles([document]) }
    for _ in 0..<500 where !studio.isImportingFiles && studio.pendingAttachments.isEmpty {
      await Task.yield()
    }
    try require(
      studio.isImportingFiles || !studio.pendingAttachments.isEmpty, "Import never started")
    if studio.isImportingFiles { try require(!studio.canSend, "Send allowed during extraction") }
    studio.newConversation()
    await importing.value
    try require(studio.pendingAttachments.isEmpty, "Async import delivered to wrong conversation")
    studio.selectConversation(destination)
    try require(studio.pendingAttachments.count == 1, "Async import lost its destination")
    let deleting = Task { await studio.importChatFiles([document]) }
    for _ in 0..<500 where !studio.isImportingFiles { await Task.yield() }
    studio.deleteConversation(destination)
    await deleting.value
    try require(studio.attachmentDrafts[destination] == nil, "Import resurrected a deleted draft")

    // Render the production composer and message timeline with real AppKit.
    _ = NSApplication.shared
    let host = NSHostingView(
      rootView: VStack {
        ComposerView()
        MessagesView(messages: [user])
      }.environmentObject(reopened))
    host.frame = CGRect(x: 0, y: 0, width: 900, height: 600)
    host.layoutSubtreeIfNeeded()
    try require(host.fittingSize.height > 0, "Attachment UI failed to render")
    print(
      "File import checks passed: text/PDF, errors, \(checked) finite boundary cases, 256 encoding samples, drafts, bridge, history, async lifecycle and UI rendering"
    )
  }
}
