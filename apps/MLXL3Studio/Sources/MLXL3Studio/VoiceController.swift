import AVFoundation
import Speech
import SwiftUI

nonisolated func requestSpeechAuthorization(
    _ request: @Sendable (@escaping @Sendable (SFSpeechRecognizerAuthorizationStatus) -> Void) -> Void
) async -> SFSpeechRecognizerAuthorizationStatus {
    await withCheckedContinuation { continuation in
        request { @Sendable status in continuation.resume(returning: status) }
    }
}

private struct DictationDraft {
    var confirmed: String

    mutating func receive(_ phrase: String, isFinal: Bool) -> String {
        guard !phrase.isEmpty else { return confirmed }
        let needsSpace =
            confirmed.last.map { !$0.isWhitespace } == true
            && phrase.first.map { !$0.isWhitespace && !",.…)".contains($0) } == true
        let draft = confirmed + (needsSpace ? " " : "") + phrase
        if isFinal { confirmed = draft }
        return draft
    }
}

// The audio tap can race stop(). Serialize converter access and close the stream once.
private final class SpeechAudioSink: @unchecked Sendable {
    private let lock = NSLock()
    private let converter: AVAudioConverter
    private var pendingInput: AVAudioPCMBuffer?
    private var output: AsyncThrowingStream<AnalyzerInput, Error>.Continuation?

    init(converter: AVAudioConverter, output: AsyncThrowingStream<AnalyzerInput, Error>.Continuation) {
        self.converter = converter
        self.output = output
        converter.primeMethod = .none
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard let output else { return }
        let format = converter.outputFormat
        let frames = ceil(Double(buffer.frameLength) * format.sampleRate / converter.inputFormat.sampleRate)
        guard frames > 0 else { return }
        guard buffer.format == converter.inputFormat, frames < Double(UInt32.max),
            let converted = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(frames))
        else {
            output.finish(throwing: CocoaError(.formatting))
            self.output = nil
            return
        }
        pendingInput = buffer
        defer { pendingInput = nil }
        var error: NSError?
        let status = converter.convert(to: converted, error: &error) { _, inputStatus in
            guard let input = self.pendingInput else {
                inputStatus.pointee = .noDataNow
                return nil
            }
            self.pendingInput = nil
            inputStatus.pointee = .haveData
            return input
        }
        if status == .error {
            output.finish(throwing: error ?? CocoaError(.formatting))
            self.output = nil
        } else if converted.frameLength > 0 {
            output.yield(AnalyzerInput(buffer: converted))
        }
    }

    func finish() {
        lock.lock()
        defer { lock.unlock() }
        output?.finish()
        output = nil
    }
}

@MainActor
final class VoiceController: ObservableObject {
    @Published private(set) var recording = false
    @Published private(set) var starting = false
    @Published var error: String?
    private let engine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    private var sink: SpeechAudioSink?
    private var analyzer: SpeechAnalyzer?
    private var recognition: Task<Void, Never>?
    private var timeout: Task<Void, Never>?
    private var session = UUID()
    private var hasTap = false

    func start(locale: String, draft: String, update: @escaping @MainActor (String) -> Void) async {
        guard !recording, !starting else { return }
        error = nil
        starting = true
        let token = UUID()
        session = token
        defer { if session == token { starting = false } }
        let microphone = await AVCaptureDevice.requestAccess(for: .audio)
        guard session == token else { return }
        guard microphone else {
            error = L(
                "Accès au microphone refusé dans Réglages Système.", "Microphone access denied in System Settings.")
            return
        }
        let authorization = await requestSpeechAuthorization(SFSpeechRecognizer.requestAuthorization)
        guard session == token else { return }
        guard authorization == .authorized else {
            error = L("Reconnaissance vocale non autorisée.", "Speech recognition is not authorized.")
            return
        }
        let supported = await DictationTranscriber.supportedLocale(equivalentTo: Locale(identifier: locale))
        guard session == token else { return }
        guard let supported else {
            error = L(
                "La dictée locale n’est pas disponible pour cette langue sur ce Mac.",
                "On-device dictation is unavailable for this language on this Mac.")
            return
        }
        let transcriber = DictationTranscriber(locale: supported, preset: .progressiveLongDictation)
        do {
            if let install = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                guard session == token else { return }
                try await install.downloadAndInstall()
            }
        } catch {
            guard session == token else { return }
            self.error = error.localizedDescription
            return
        }
        guard session == token else { return }
        let format = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber])
        guard session == token else { return }
        let input = engine.inputNode
        let hardwareFormat = input.outputFormat(forBus: 0)
        guard hardwareFormat.sampleRate > 0, hardwareFormat.channelCount > 0,
            let format, let converter = AVAudioConverter(from: hardwareFormat, to: format)
        else {
            error = L("Microphone indisponible.", "Microphone unavailable.")
            return
        }
        synthesizer.stopSpeaking(at: .immediate)
        let (audio, output) = AsyncThrowingStream<AnalyzerInput, Error>.makeStream()
        let sink = SpeechAudioSink(converter: converter, output: output)
        let analyzer = SpeechAnalyzer(modules: [transcriber])
        self.sink = sink
        self.analyzer = analyzer
        recognition = Task { [weak self] in
            var transcript = DictationDraft(confirmed: draft)
            do {
                for try await result in transcriber.results {
                    guard self?.session == token else { return }
                    update(transcript.receive(String(result.text.characters), isFinal: result.isFinal))
                }
                if self?.session == token { self?.stop() }
            } catch {
                guard self?.session == token else { return }
                self?.stop()
                self?.error = error.localizedDescription
            }
        }
        do {
            try await analyzer.start(inputSequence: audio)
            guard session == token else { return }
            input.installTap(onBus: 0, bufferSize: 1024, format: hardwareFormat) { @Sendable buffer, _ in
                sink.append(buffer)
            }
            hasTap = true
            try engine.start()
            recording = true
        } catch {
            guard session == token else { return }
            stop()
            self.error = error.localizedDescription
            return
        }
        timeout = Task { [weak self] in
            try? await Task.sleep(for: .seconds(55))
            guard !Task.isCancelled, self?.session == token else { return }
            self?.stop()
        }
    }

    func stop() {
        session = UUID()
        starting = false
        recording = false
        timeout?.cancel()
        timeout = nil
        engine.stop()
        if hasTap {
            engine.inputNode.removeTap(onBus: 0)
            hasTap = false
        }
        sink?.finish()
        sink = nil
        recognition?.cancel()
        recognition = nil
        if let analyzer { Task { await analyzer.cancelAndFinishNow() } }
        analyzer = nil
    }

    func speak(_ text: String, locale: String) {
        stop()
        synthesizer.stopSpeaking(at: .immediate)
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: locale)
        synthesizer.speak(utterance)
    }

    func stopSpeaking() { synthesizer.stopSpeaking(at: .immediate) }
}
