import AppKit
import Foundation
import SwiftUI

private struct ImageCheckFailure: Error { let message: String }

@main struct ImageInputCheck {
  @MainActor static func main() async throws {
    var checks = 0
    func check(_ condition: @autoclosure () throws -> Bool, _ message: String) throws {
      guard try condition() else { throw ImageCheckFailure(message: message) }
      checks += 1
    }
    func rejects(_ expected: AttachmentImportError, _ operation: () throws -> Void) throws {
      do { try operation() }
      catch let error as AttachmentImportError {
        try check(error == expected, "Expected \(expected), got \(error)")
        return
      }
      throw ImageCheckFailure(message: "Expected rejection: \(expected)")
    }
    func wait(_ predicate: @MainActor () -> Bool) async throws {
      for _ in 0..<400 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(20))
      }
      throw ImageCheckFailure(message: "Image fixture timed out")
    }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("mlxl3-images-\(UUID())")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }
    func file(_ name: String, _ data: Data) throws -> URL {
      let url = root.appendingPathComponent(name)
      try data.write(to: url)
      return url
    }
    let fixtures = URL(fileURLWithPath: CommandLine.arguments[2])
    var images: [ChatAttachment] = []
    for (ext, mime) in [("png", "image/png"), ("jpeg", "image/jpeg"), ("webp", "image/webp")] {
      let data = try Data(contentsOf: fixtures.appendingPathComponent("red.\(ext)"))
      let url = try file("red.\(ext)", data)
      try rejects(.unsupported) { _ = try ChatAttachmentImporter.read(url) }
      let parsed = try ChatAttachmentImporter.read(url, allowImages: true)
      try check(parsed.text.isEmpty && parsed.image?.data == data && parsed.image?.mimeType == mime,
                "Image bytes/MIME changed: \(ext)")
      try check(NSImage(data: parsed.image!.data) != nil, "Image preview failed: \(ext)")
      images.append(parsed)
    }
    let image = images[0]
    try check(try ChatAttachmentImporter.read(file("png-named-jpeg.jpeg", image.image!.data), allowImages: true).image?.mimeType == "image/png", "MIME trusted the extension instead of image bytes")
    try rejects(.unsupportedImage) {
      _ = try ChatAttachmentImporter.read(fixtures.appendingPathComponent("red.gif"), allowImages: true)
    }
    try rejects(.imageResolution) {
      _ = try ChatAttachmentImporter.read(fixtures.appendingPathComponent("oversized.png"), allowImages: true)
    }
    try rejects(.invalidImage) {
      _ = try ChatAttachmentImporter.read(file("bad.png", Data("broken".utf8)), allowImages: true)
    }
    try rejects(.invalidImage) {
      _ = try ChatAttachmentImporter.read(file("truncated.png", image.image!.data.prefix(20)), allowImages: true)
    }
    try rejects(.imagesTooLarge) {
      _ = try ChatAttachmentImporter.read(file("large.png", Data(repeating: 0, count: 8 * 1024 * 1024 + 1)), allowImages: true)
    }
    // Admission boundaries use synthetic payloads; decoder correctness uses the real fixtures above.
    for used in [0, 1, 4 * 1024 * 1024, 8 * 1024 * 1024 - 1] {
      for extra in [0, 1, 8 * 1024 * 1024 - used, 8 * 1024 * 1024 - used + 1] {
        let existing = used == 0 ? [] : [ChatAttachment(fileName: "first.png", text: "", fileBytes: used,
          image: ChatImage(mimeType: "image/png", data: Data(repeating: 1, count: used)))]
        let next = ChatAttachment(fileName: "next.png", text: "", fileBytes: extra,
          image: ChatImage(mimeType: "image/png", data: Data(repeating: 2, count: extra)))
        if extra == 0 { try rejects(.invalidImage) { _ = try ChatAttachmentImporter.adding(next, to: existing) } }
        else if used + extra > 8 * 1024 * 1024 {
          try rejects(.imagesTooLarge) { _ = try ChatAttachmentImporter.adding(next, to: existing) }
        } else { try check(try ChatAttachmentImporter.adding(next, to: existing).count == existing.count + 1, "Exact image budget rejected") }
      }
    }
    try check(try ChatAttachmentImporter.adding(image, to: [image]).count == 1, "Duplicate image retained")
    var different = image
    different.image = images[1].image
    try check(try ChatAttachmentImporter.adding(different, to: [image]).count == 2, "Different pixels were discarded")
    let documentURL = try file("notes.txt", Data("Budget CEDAR-42: 73 euros.".utf8))
    let document = try ChatAttachmentImporter.read(documentURL)
    let text = ChatAttachment.promptContent("Question", attachments: [image, document])
    try check(text.contains(document.text) && !text.contains("base64") && !text.contains("red.png"), "Image leaked into text content")
    let encoder = JSONEncoder(), decoder = JSONDecoder()
    try check(try decoder.decode(ChatAttachment.self, from: encoder.encode(image)) == image, "Image Codable round trip failed")
    var legacy = try JSONSerialization.jsonObject(with: encoder.encode(document)) as! [String: Any]
    legacy.removeValue(forKey: "image")
    try check(try decoder.decode(ChatAttachment.self, from: JSONSerialization.data(withJSONObject: legacy)).image == nil, "Old attachment migration failed")
    let draft = ConversationDraft(conversationID: UUID(), text: "Hi", attachments: [image])
    try check(draft.valid && draft.byteCount == 2 + image.image!.data.count, "Draft image budget omitted bytes")

    setenv("MLXL3_EXECUTABLE", CommandLine.arguments[1], 1)
    setenv("MLXL3_HOME", root.path, 1)
    let suite = "mlxl3.images.\(UUID())", history = root.appendingPathComponent("history.json")
    let preferences = UserDefaults(suiteName: suite)!
    defer { preferences.removePersistentDomain(forName: suite) }
    let studio = StudioModel(conversationFileURL: history, preferences: preferences)
    defer { studio.ejectModel() }
    studio.models = [ ("image-import", "lfm3_vl"), ("file-import", "lfm3"), ("image-old", "lfm3_vl") ].map {
      LocalModel(name: $0.0, path: root.path, modelType: $0.1, format: "EXL3", bits: 8,
        sizeBytes: 1, modules: 1, addedAt: "", size: "1 B")
    }
    try check(!studio.canImportImages, "Images enabled before READY")
    studio.selectModel("image-import")
    try await wait { studio.engineState.isReady }
    try check(studio.canImportImages, "Vision capability ignored")
    await studio.importChatFiles([root.appendingPathComponent("red.png"), documentURL])
    try check(studio.pendingAttachments.count == 2 && studio.canSend, "Mixed image/document import failed")
    studio.send()
    try await wait { !studio.isGenerating }
    let user = studio.currentConversation!.messages[0]
    let messages = try JSONSerialization.jsonObject(with: Data(studio.currentConversation!.messages.last!.content.utf8)) as! [[String: Any]]
    try check(messages.last?["images"] as? [String] == [image.image!.dataURL], "Real bridge omitted image data URL")
    try check((messages.last?["content"] as? String)?.contains(document.text) == true, "Mixed document was omitted")
    try check(!(messages.last?["content"] as! String).contains("base64"), "Image encoded in text prompt")
    try check(studio.persistNow(), "Image history save failed")
    try FileManager.default.removeItem(at: root.appendingPathComponent("red.png"))
    let reopened = StudioModel(conversationFileURL: history, isPreview: true, preferences: preferences)
    try check(reopened.currentConversation?.messages[0].attachments[0].image == image.image, "History depends on removed source image")
    studio.draft = "And its color?"
    studio.send()
    try await wait { !studio.isGenerating }
    let followUp = try JSONSerialization.jsonObject(with: Data(studio.currentConversation!.messages.last!.content.utf8)) as! [[String: Any]]
    try check(followUp.first?["images"] as? [String] == [image.image!.dataURL], "Follow-up lost historical image")

    studio.selectModel("file-import")
    try await wait { studio.engineState.isReady }
    studio.draft = "continue"
    try check(!studio.canImportImages && !studio.canSend && studio.sendBlockReason != nil, "Text model silently accepts image history")
    studio.newConversation()
    await studio.importChatFiles([root.appendingPathComponent("red.jpeg")])
    try check(studio.pendingAttachments.isEmpty && studio.attachmentImportError != nil, "Text model imported image")
    studio.dismissAttachmentImportError()
    studio.selectModel("image-old")
    try await wait { studio.engineState.isReady }
    try check(!studio.canImportImages, "Old engine without capability enabled images")
    studio.selectModel("image-import")
    try await wait { studio.engineState.isReady }
    await studio.importChatFiles([root.appendingPathComponent("red.jpeg")])
    try check(studio.draft.isEmpty && studio.canSend, "Image-only send unavailable")
    studio.selectModel("file-import")
    try await wait { studio.engineState.isReady }
    try check(!studio.canSend && studio.pendingAttachments.count == 1, "Model switch discarded or silently sent pending image")
    studio.selectModel("image-import")
    try await wait { studio.engineState.isReady }
    try check(studio.canSend && studio.pendingAttachments.count == 1, "Vision reload lost pending image")
    studio.send()
    try await wait { !studio.isGenerating }
    try check(studio.currentConversation!.messages[0].attachments[0].image == images[1].image, "Image-only send lost attachment")
    let id = studio.selectedConversationID!
    studio.attachmentDrafts[id] = (0..<8).map { n in
      ChatAttachment(fileName: "\(n).png", text: "", fileBytes: image.fileBytes, image: image.image)
    }
    try check(!studio.canSend && studio.sendBlockReason != nil, "Ninth image across turns passed engine limit")
    studio.attachmentDrafts[id] = nil

    _ = NSApplication.shared
    let host = NSHostingView(rootView: VStack {
      ComposerView()
      MessagesView(messages: [user])
    }.environmentObject(reopened))
    for width in [680.0, 1040.0] {
      host.frame = CGRect(x: 0, y: 0, width: width, height: 600)
      host.layoutSubtreeIfNeeded()
      try check(host.fittingSize.height > 0, "Image UI failed to render")
      if let destination = ProcessInfo.processInfo.environment["MLXL3_IMAGE_CHECK_OUTPUT"],
        let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let png = bitmap.representation(using: .png, properties: [:])!
        try png.write(to: URL(fileURLWithPath: destination).appendingPathComponent("lfm3-image-chat-\(Int(width)).png"))
      }
    }
    print("Image input checks passed: \(checks) checks, PNG/JPEG/WebP, boundaries, bridge, history, model changes and two UI widths")
  }
}
