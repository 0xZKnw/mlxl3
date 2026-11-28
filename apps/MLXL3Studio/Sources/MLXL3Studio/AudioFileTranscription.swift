import AVFoundation
import CoreMedia
import Foundation
import Speech
import UniformTypeIdentifiers

/// macOS decodes the selected file and transcribes it on-device. The conversation stores text only.
enum AudioFileTranscription {
    static let maxFileBytes = 64 * 1_024 * 1_024
    static let maxDuration: TimeInterval = 300
    private static let extensions = ["m4a", "wav", "mp3"]
    static let contentTypes = extensions.compactMap { UTType(filenameExtension: $0) }

    typealias Recognize = @Sendable (AVAudioFile, Locale) async throws -> String

    static func supports(_ url: URL) -> Bool {
        extensions.contains(url.pathExtension.lowercased())
    }

    static func read(_ url: URL, locale: Locale, recognize: Recognize) async throws -> ChatAttachment {
        guard url.isFileURL else { throw AttachmentImportError.notAFile }
        guard supports(url) else { throw AttachmentImportError.unsupported }
        try Task.checkCancellation()
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        guard values.isRegularFile == true, let fileBytes = values.fileSize else {
            throw AttachmentImportError.notAFile
        }
        guard fileBytes <= maxFileBytes else { throw AttachmentImportError.audioTooLarge }
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: url)
        } catch {
            throw AttachmentImportError.invalidAudio
        }
        let format = file.processingFormat
        guard file.length > 0, format.sampleRate.isFinite, format.sampleRate > 0,
              (1...8).contains(format.channelCount)
        else {
            throw AttachmentImportError.invalidAudio
        }
        guard Double(file.length) / format.sampleRate <= maxDuration else {
            throw AttachmentImportError.audioTooLong
        }
        let text = try await recognize(file, locale).trimmingCharacters(in: .whitespacesAndNewlines)
        try Task.checkCancellation()
        guard !text.isEmpty else { throw AttachmentImportError.noSpeech }
        guard text.utf8.count <= ChatAttachmentImporter.maxTextBytes else {
            throw AttachmentImportError.textTooLarge
        }
        return ChatAttachment(fileName: url.lastPathComponent, text: text, fileBytes: fileBytes)
    }

    static func recognize(_ file: AVAudioFile, locale: Locale) async throws -> String {
        guard let supported = await DictationTranscriber.supportedLocale(equivalentTo: locale)
        else { throw AttachmentImportError.speechLanguageUnavailable }
        try Task.checkCancellation()
        let transcriber = DictationTranscriber(locale: supported, preset: .longDictation)
        do {
            if let installation = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                try Task.checkCancellation()
                try await installation.downloadAndInstall()
            }
        } catch {
            try Task.checkCancellation()
            throw AttachmentImportError.speechAssetsUnavailable
        }
        try Task.checkCancellation()
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        return try await withTaskCancellationHandler {
            try await withThrowingTaskGroup(of: String?.self) { group in
                group.addTask { try await transcript(from: transcriber) }
                group.addTask {
                    guard let last = try await analyzer.analyzeSequence(from: file) else {
                        throw AttachmentImportError.invalidAudio
                    }
                    try await analyzer.finalizeAndFinish(through: last)
                    return nil
                }
                do {
                    var text = ""
                    for try await result in group {
                        if let result { text = result }
                    }
                    return text
                } catch {
                    group.cancelAll()
                    await analyzer.cancelAndFinishNow()
                    throw error
                }
            }
        } onCancel: {
            Task { await analyzer.cancelAndFinishNow() }
        }
    }

    private static func transcript(from transcriber: DictationTranscriber) async throws -> String {
        var lines: [String] = []
        var bytes = 0
        for try await result in transcriber.results {
            try Task.checkCancellation()
            guard result.isFinal else { continue }
            let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            let seconds = result.range.start.seconds
            let line: String
            if seconds.isFinite, (0...maxDuration).contains(seconds) {
                let elapsed = Int(seconds)
                line = String(format: "[%02d:%02d] ", elapsed / 60, elapsed % 60) + text
            } else {
                line = text
            }
            bytes += line.utf8.count + (lines.isEmpty ? 0 : 1)
            guard bytes <= ChatAttachmentImporter.maxTextBytes else { throw AttachmentImportError.textTooLarge }
            lines.append(line)
        }
        return lines.joined(separator: "\n")
    }
}
