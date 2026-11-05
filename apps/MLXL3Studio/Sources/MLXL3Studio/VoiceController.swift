import AVFoundation
import Speech
import SwiftUI

@MainActor
final class VoiceController: ObservableObject {
    @Published private(set) var recording = false
    @Published private(set) var starting = false
    @Published var error: String?
    private let engine = AVAudioEngine()
    private let synthesizer = AVSpeechSynthesizer()
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var recognition: SFSpeechRecognitionTask?
    private var timeout: Task<Void, Never>?
    private var session = UUID()
    private var hasTap = false

    func start(locale: String, update: @escaping @MainActor (String) -> Void) async {
        guard !recording, !starting else { return }
        error = nil; starting = true
        let token = UUID(); session = token
        defer { if session == token { starting = false } }
        let microphone = await AVCaptureDevice.requestAccess(for: .audio)
        guard session == token else { return }
        guard microphone else { error = L("Accès au microphone refusé dans Réglages Système.", "Microphone access denied in System Settings."); return }
        let authorization = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
        }
        guard session == token else { return }
        guard authorization == .authorized else { error = L("Reconnaissance vocale non autorisée.", "Speech recognition is not authorized."); return }
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: locale)), recognizer.isAvailable, recognizer.supportsOnDeviceRecognition else {
            error = L("La dictée locale n’est pas disponible pour cette langue sur ce Mac.", "On-device dictation is unavailable for this language on this Mac."); return
        }
        synthesizer.stopSpeaking(at: .immediate)
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        self.request = request
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { stop(); error = L("Microphone indisponible.", "Microphone unavailable."); return }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in request.append(buffer) }
        hasTap = true
        recognition = recognizer.recognitionTask(with: request) { [weak self] result, failure in
            let text = result?.bestTranscription.formattedString
            let final = result?.isFinal == true
            let detail = failure?.localizedDescription
            Task { @MainActor in
                guard let self, self.session == token else { return }
                if let text { update(text) }
                if final || detail != nil { self.stop(); self.error = detail }
            }
        }
        do { try engine.start(); recording = true }
        catch { stop(); self.error = error.localizedDescription }
        timeout = Task { [weak self] in
            try? await Task.sleep(for: .seconds(55))
            guard !Task.isCancelled, self?.session == token else { return }
            self?.stop()
        }
    }

    func stop() {
        session = UUID(); starting = false; recording = false
        timeout?.cancel(); timeout = nil
        engine.stop()
        if hasTap { engine.inputNode.removeTap(onBus: 0); hasTap = false }
        request?.endAudio(); request = nil
        recognition?.cancel(); recognition = nil
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
