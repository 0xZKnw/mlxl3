import Foundation
import PDFKit
import UniformTypeIdentifiers

struct ChatAttachment: Codable, Identifiable, Sendable, Equatable {
  var id = UUID()
  let fileName: String
  let text: String
  let fileBytes: Int

  static func promptContent(_ message: String, attachments: [ChatAttachment]) -> String {
    guard !attachments.isEmpty else { return message }
    let documents = attachments.map {
      "--- File: \(String(reflecting: $0.fileName)) ---\n\($0.text)\n--- End of file ---"
    }.joined(separator: "\n\n")
    return message + "\n\nAttached documents (reference material):\n\n" + documents
  }
}

enum AttachmentImportError: Error, LocalizedError, Equatable {
  case unsupported, notAFile, fileTooLarge, textTooLarge, empty, invalidText, invalidPDF, lockedPDF
  case tooManyFiles, totalTooLarge

  var errorDescription: String? {
    switch self {
    case .unsupported:
      L(
        "Format non pris en charge. Choisissez un PDF ou un fichier texte (TXT, Markdown, CSV, JSON, code…).",
        "Unsupported format. Choose a PDF or a text file (TXT, Markdown, CSV, JSON, source code…).")
    case .notAFile:
      L("Choisissez un fichier local, pas un dossier.", "Choose a local file, not a folder.")
    case .fileTooLarge: L("Le fichier dépasse 20 Mio.", "The file exceeds 20 MiB.")
    case .textTooLarge:
      L(
        "Le texte extrait dépasse 256 Kio. Divisez le document.",
        "Extracted text exceeds 256 KiB. Split the document.")
    case .empty:
      L(
        "Ce fichier ne contient pas de texte lisible. Les PDF scannés nécessitent une reconnaissance de texte préalable.",
        "This file has no readable text. Scanned PDFs need text recognition first.")
    case .invalidText:
      L(
        "Encodage non pris en charge ou contenu binaire. Utilisez UTF-8 ou UTF-16 avec BOM.",
        "Unsupported encoding or binary content. Use UTF-8 or UTF-16 with a BOM.")
    case .invalidPDF: L("Le PDF est illisible ou endommagé.", "The PDF is unreadable or damaged.")
    case .lockedPDF:
      L(
        "Le PDF est protégé par un mot de passe. Déverrouillez-le avant de l’importer.",
        "The PDF is password protected. Unlock it before importing.")
    case .tooManyFiles:
      L(
        "Vous pouvez joindre jusqu’à 8 fichiers par message.",
        "You can attach up to 8 files per message.")
    case .totalTooLarge:
      L(
        "Le texte des pièces jointes dépasse 512 Kio au total. Retirez un fichier ou divisez les documents.",
        "Attachment text exceeds 512 KiB in total. Remove a file or split the documents.")
    }
  }
}

enum ChatAttachmentImporter {
  static let maxFiles = 8
  static let maxFileBytes = 20 * 1_024 * 1_024
  static let maxTextBytes = 256 * 1_024
  static let maxTotalTextBytes = 512 * 1_024
  // Some plain-text formats have no registered UTType on older macOS SDKs.
  private static let textExtensions: Set<String> = [
    "txt", "md", "markdown", "csv", "tsv", "json", "jsonl", "xml", "yaml", "yml",
    "toml", "log", "ini", "cfg", "conf", "rs", "py", "swift", "js", "ts", "tsx",
    "jsx", "c", "h", "cpp", "hpp", "css", "html", "sh", "zsh", "sql", "go",
    "java", "kt", "rb", "tex", "ipynb", "bend", "vue",
  ]

  static var contentTypes: [UTType] {
    [.pdf, .plainText, .text, .sourceCode, .json, .xml, .commaSeparatedText]
      + textExtensions.sorted().compactMap { UTType(filenameExtension: $0) }
  }

  struct Batch: Sendable {
    let attachments: [ChatAttachment]
    let errors: [String]
  }

  static func load(_ urls: [URL], existing: [ChatAttachment]) -> Batch {
    var attachments = existing
    var errors: [String] = []
    for url in urls {
      if Task.isCancelled { break }
      do {
        try Task.checkCancellation()
        attachments = try adding(read(url), to: attachments)
      } catch {
        errors.append("\(url.lastPathComponent): \(error.localizedDescription)")
      }
    }
    return Batch(attachments: attachments, errors: errors)
  }

  static func adding(_ attachment: ChatAttachment, to existing: [ChatAttachment]) throws
    -> [ChatAttachment]
  {
    if existing.contains(where: { $0.fileName == attachment.fileName && $0.text == attachment.text }
    ) {
      return existing
    }
    guard existing.count < maxFiles else { throw AttachmentImportError.tooManyFiles }
    guard attachment.text.utf8.count <= maxTextBytes else {
      throw AttachmentImportError.textTooLarge
    }
    let used = existing.reduce(0) { $0 + $1.text.utf8.count }
    guard attachment.text.utf8.count <= maxTotalTextBytes - used else {
      throw AttachmentImportError.totalTooLarge
    }
    return existing + [attachment]
  }

  static func read(_ url: URL) throws -> ChatAttachment {
    guard url.isFileURL else { throw AttachmentImportError.notAFile }
    let scoped = url.startAccessingSecurityScopedResource()
    defer { if scoped { url.stopAccessingSecurityScopedResource() } }
    let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
    guard values.isRegularFile == true else { throw AttachmentImportError.notAFile }
    guard (values.fileSize ?? 0) <= maxFileBytes else { throw AttachmentImportError.fileTooLarge }
    let ext = url.pathExtension.lowercased()
    let type = UTType(filenameExtension: ext)
    guard
      ext == "pdf" || ext.isEmpty || textExtensions.contains(ext)
        || type?.conforms(to: .text) == true || type?.conforms(to: .sourceCode) == true
    else {
      throw AttachmentImportError.unsupported
    }

    // Bound the actual read as well as the initial stat: a file can grow while importing.
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var data = Data()
    while let chunk = try handle.read(upToCount: min(64 * 1_024, maxFileBytes + 1 - data.count)),
      !chunk.isEmpty
    {
      try Task.checkCancellation()
      data.append(chunk)
      guard data.count <= maxFileBytes else { throw AttachmentImportError.fileTooLarge }
    }

    let text: String
    if ext == "pdf" {
      guard let pdf = PDFDocument(data: data) else { throw AttachmentImportError.invalidPDF }
      guard !pdf.isLocked else { throw AttachmentImportError.lockedPDF }
      var pages: [String] = []
      var bytes = 0
      for index in 0..<pdf.pageCount {
        try Task.checkCancellation()
        guard let page = pdf.page(at: index)?.string,
          !page.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { continue }
        let section = "[Page \(index + 1)]\n" + page
        bytes += section.utf8.count + (pages.isEmpty ? 0 : 2)
        guard bytes <= maxTextBytes else { throw AttachmentImportError.textTooLarge }
        pages.append(section)
      }
      text = pages.joined(separator: "\n\n")
    } else {
      text = try decodeText(data)
    }
    guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
      throw AttachmentImportError.empty
    }
    guard text.utf8.count <= maxTextBytes else { throw AttachmentImportError.textTooLarge }
    return ChatAttachment(fileName: url.lastPathComponent, text: text, fileBytes: data.count)
  }

  static func decodeText(_ data: Data) throws -> String {
    let payload: Data
    let encoding: String.Encoding
    if data.starts(with: [0xFF, 0xFE]) {
      payload = data.dropFirst(2)
      encoding = .utf16LittleEndian
    } else if data.starts(with: [0xFE, 0xFF]) {
      payload = data.dropFirst(2)
      encoding = .utf16BigEndian
    } else {
      payload = data.starts(with: [0xEF, 0xBB, 0xBF]) ? data.dropFirst(3) : data
      encoding = .utf8
    }
    // Foundation may repair malformed UTF-16. Reject any lossy decode instead.
    guard let text = String(data: payload, encoding: encoding),
      text.data(using: encoding) == payload,
      !text.unicodeScalars.contains(where: {
        ($0.value < 32 && $0.value != 9 && $0.value != 10 && $0.value != 13) || $0.value == 127
      })
    else { throw AttachmentImportError.invalidText }
    return text
  }
}
