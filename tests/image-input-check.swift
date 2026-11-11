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
    _ = NSApplication.shared
    let board = NSPasteboard(name: NSPasteboard.Name("mlxl3-test-\(UUID())"))
    defer { board.releaseGlobally() }
    board.clearContents()
    board.setString("ordinary text", forType: .string)
    try check(ComposerPasteboard.image(from: board) == nil, "Text paste was intercepted")
    board.clearContents()
    board.setData(image.image!.data, forType: .png)
    guard case .data(let clipboardPNG) = ComposerPasteboard.image(from: board) else {
      throw ImageCheckFailure(message: "PNG clipboard data was not detected")
    }
    try check(try ChatAttachmentImporter.readPastedImage(clipboardPNG).image == image.image, "PNG clipboard data changed")
    let tiff = NSBitmapImageRep(data: image.image!.data)!.representation(using: .tiff, properties: [:])!
    board.clearContents()
    board.setData(tiff, forType: .tiff)
    guard case .data(let clipboardTIFF) = ComposerPasteboard.image(from: board) else {
      throw ImageCheckFailure(message: "TIFF clipboard data was not detected")
    }
    let converted = try ChatAttachmentImporter.readPastedImage(clipboardTIFF)
    try check(converted.image?.mimeType == "image/png" && NSImage(data: converted.image!.data) != nil,
              "Clipboard TIFF was not converted to a valid PNG")
    board.clearContents()
    board.writeObjects([root.appendingPathComponent("red.png") as NSURL])
    guard case .files(let urls) = ComposerPasteboard.image(from: board) else {
      throw ImageCheckFailure(message: "Copied image file was not detected")
    }
    try check(urls == [root.appendingPathComponent("red.png")], "Copied file URL changed")
    try rejects(.invalidImage) { _ = try ChatAttachmentImporter.readPastedImage(Data("broken".utf8)) }
    try rejects(.imageResolution) { _ = try ChatAttachmentImporter.readPastedImage(Data(contentsOf: fixtures.appendingPathComponent("oversized.png"))) }
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
    await studio.importPastedImage(clipboardPNG)
    try check(studio.pendingAttachments.isEmpty && studio.attachmentImportError != nil, "Clipboard image bypassed model capability")
    studio.dismissAttachmentImportError()
    try check(studio.sendBlockReason == nil && !studio.canAttemptSend, "Empty composer showed an unloaded-model warning")
    studio.send()
    try check(studio.sendBlockReason == nil, "Empty send showed an unloaded-model warning")
    studio.draft = "A question"
    try check(studio.sendBlockReason == nil && studio.canAttemptSend, "Unloaded warning appeared before send")
    studio.send()
    try check(studio.sendBlockReason != nil && studio.draft == "A question" && studio.currentConversation!.messages.isEmpty,
              "Unloaded send did not show warning or preserve the draft")
    studio.draft = ""
    try check(studio.sendBlockReason == nil, "Unloaded warning remained after clearing the draft")
    studio.draft = "A question"
    try check(studio.sendBlockReason == nil, "Cleared warning reappeared without another send")
    studio.newConversation()
    try check(studio.sendBlockReason == nil, "Unloaded warning leaked into a new conversation")
    studio.draft = "A question"
    studio.send()
    studio.selectModel("image-import")
    try await wait { studio.engineState.isReady }
    try check(studio.sendBlockReason == nil && !studio.attemptedSendWithoutModel, "Warning survived model load")
    studio.draft = ""
    try check(studio.canImportImages, "Vision capability ignored")
    ComposerPasteboard.importImage(.data(clipboardPNG), into: studio)
    try await wait { !studio.pendingAttachments.isEmpty || studio.attachmentImportError != nil }
    try check(studio.pendingAttachments.first?.image == image.image, "Clipboard image was not attached")
    await studio.importChatFiles([documentURL])
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
    let additional = try (0..<8).map { try file("\($0).png", image.image!.data) }
    await studio.importChatFiles(additional)
    try check(!studio.canSend && studio.sendBlockReason != nil, "Ninth image across turns passed engine limit")
    for attachment in studio.pendingAttachments { studio.removeChatAttachment(attachment.id) }
    await studio.importChatFiles([root.appendingPathComponent("red.jpeg")])

    _ = NSApplication.shared
    let sentPreview = NSHostingView(rootView: ChatAttachmentView(attachment: image))
    let draftPreview = NSHostingView(rootView: ChatAttachmentView(attachment: image, onRemove: {}))
    try check(abs(sentPreview.fittingSize.width - sentPreview.fittingSize.height) < 1, "Sent image has extra bubble padding")
    try check(abs(draftPreview.fittingSize.width - draftPreview.fittingSize.height) < 1, "Draft image has extra bubble padding")
    for (width, height) in [(120, 40), (40, 120)] {
      let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
        bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
        colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
      let png = bitmap.representation(using: .png, properties: [:])!
      let rectangular = ChatAttachment(fileName: "rectangle.png", text: "", fileBytes: png.count,
        image: ChatImage(mimeType: "image/png", data: png))
      for draft in [false, true] {
        let view = NSHostingView(rootView: ChatAttachmentView(attachment: rectangular, onRemove: draft ? {} : nil))
        let size = view.fittingSize
        try check(abs(size.width / size.height - CGFloat(width) / CGFloat(height)) < 0.01,
                  "Image preview added margins for \(width)x\(height), draft=\(draft)")
      }
    }
    for width in [680.0, 1040.0] {
      let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: width, height: 600),
                            styleMask: .borderless, backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      defer { window.close() }
      let host = NSHostingView(rootView: VStack {
        MessagesView(messages: [user])
        ComposerView()
      }.foregroundStyle(StudioTheme.ink).background(StudioTheme.canvas(for: .midnightBlue))
        .environmentObject(studio).studioAppearance(.midnightBlue))
      window.contentView = host
      host.frame = CGRect(x: 0, y: 0, width: width, height: 600)
      try await Task.sleep(for: .milliseconds(250))
      host.layoutSubtreeIfNeeded()
      try check(host.fittingSize.height > 0, "Image UI failed to render")
      guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else {
        throw ImageCheckFailure(message: "Image bitmap missing")
      }
      host.cacheDisplay(in: host.bounds, to: bitmap)
      let png = bitmap.representation(using: .png, properties: [:])!
      let pixels = NSBitmapImageRep(data: png)!
      var redPixels = 0
      for y in stride(from: 0, to: pixels.pixelsHigh, by: 8) {
        for x in stride(from: 0, to: pixels.pixelsWide, by: 8) {
          if let pixel = pixels.colorAt(x: x, y: y)?.usingColorSpace(.sRGB),
            pixel.redComponent > 0.8 && pixel.redComponent > pixel.greenComponent + 0.5
              && pixel.redComponent > pixel.blueComponent + 0.5 {
            redPixels += 1
          }
        }
      }
      if let destination = ProcessInfo.processInfo.environment["MLXL3_IMAGE_CHECK_OUTPUT"] {
        try png.write(to: URL(fileURLWithPath: destination).appendingPathComponent("lfm3-image-chat-\(Int(width)).png"))
      }
      try check(redPixels > 100, "Image pixels are not visible inline at \(width): \(redPixels) red samples")
    }
    var transfer = ModelDownloadProgress(total: 12_000_000_000)
    transfer.update(completed: 4_000_000_000, total: 12_000_000_000, at: 1)
    transfer.update(completed: 4_100_000_000, total: 12_000_000_000, at: 2)
    for (name, status, message) in [("transfer", ModelDownloadStatus.transferring, Optional<String>.none),
                                    ("paused", .paused, "Téléchargement suspendu."),
                                    ("failed", .failed, "Connexion interrompue.")] {
      let card = DownloadProgressCard(repository: "owner/LFM3-VL-2B-EXL3-very-long-model-name", progress: transfer,
        status: status, message: message, pause: {}, resume: status == .transferring ? nil : {})
      let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 330, height: 220),
                            styleMask: .borderless, backing: .buffered, defer: false)
      window.isReleasedWhenClosed = false
      defer { window.close() }
      let host = NSHostingView(rootView: VStack(alignment: .leading, spacing: 16) {
        Text("Téléchargements").font(.system(size: 14, weight: .semibold))
        card
      }.padding(18).frame(width: 330, height: 220, alignment: .topLeading)
        .foregroundStyle(StudioTheme.ink).background(StudioTheme.canvas(for: .midnightBlue))
        .studioAppearance(.midnightBlue))
      window.contentView = host
      host.frame = CGRect(x: 0, y: 0, width: 330, height: 220)
      try await Task.sleep(for: .milliseconds(100))
      host.layoutSubtreeIfNeeded()
      try check(host.fittingSize.width <= 330 && host.fittingSize.height <= 220, "Download \(name) overflowed its popover")
      if let destination = ProcessInfo.processInfo.environment["MLXL3_IMAGE_CHECK_OUTPUT"],
         let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) {
        host.cacheDisplay(in: host.bounds, to: bitmap)
        try bitmap.representation(using: .png, properties: [:])!.write(
          to: URL(fileURLWithPath: destination).appendingPathComponent("download-\(name).png"))
      }
    }
    print("Image input checks passed: \(checks) checks, PNG/JPEG/WebP, boundaries, bridge, history, model changes and two UI widths")
  }
}
