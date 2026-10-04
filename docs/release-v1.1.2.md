# MLXL3 Desktop v1.1.2 — file attachments

Use **Attach** in the chat composer, **Command-Shift-O**, or drag files onto
the composer to add PDFs and text documents. TXT, Markdown, CSV, JSON and
source code are supported. Click a file to preview its extracted text; remove
it with the cross before sending. Sending files without a question asks the
model to analyze them.

Document text is included in the model's conversation context and saved in
history. Follow-up messages keep that context, even after moving or deleting
the original files. Attachment drafts stay with their own conversation.
PDF text includes page numbers. Scanned PDFs need text recognition first;
password-protected PDFs must be unlocked before import. Images, Office files
and other binary formats are not supported by this text-only import.

Each message accepts up to eight files: 20 MiB per source file, 256 KiB of
extracted text per file and 512 KiB of extracted text in total. Invalid and
oversized documents produce explicit errors without discarding valid imports
or silently truncating their text. The selected model's context limit also
applies.

Build 17 targets Apple Silicon and macOS 26.2 or newer. The DMG contains the
SwiftUI app, Rust/Metal engine and MLX runtime; model weights are downloaded
separately. Signing remains ad-hoc, without Apple notarization.
