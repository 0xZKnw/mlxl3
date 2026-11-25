import AVFoundation
import Foundation
import Speech

@main struct VoiceCallbackCheck {
    @MainActor static func main() async throws {
        let statuses: [SFSpeechRecognizerAuthorizationStatus] = [.authorized, .denied, .restricted, .notDetermined]
        for expected in statuses {
            for _ in 0..<25 {
                let received = await requestSpeechAuthorization { callback in
                    DispatchQueue.global().async { callback(expected) }
                }
                precondition(received == expected, "Speech authorization callback changed its status")
            }
        }
        checkPhrases()
        // The sink yields the converted buffers themselves here (production wraps them in
        // `AnalyzerInput`), so the check reads them without `AnalyzerInput.buffer`, deprecated in OS 27.
        let (audio, output) = AsyncThrowingStream<AVAudioPCMBuffer, Error>.makeStream()
        let initialFormat = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!
        let convertedFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16, sampleRate: 8_000, channels: 1, interleaved: false)!
        let sink = SpeechAudioSink(
            converter: AVAudioConverter(from: initialFormat, to: convertedFormat)!, output: output) { $0 }
        let initialBuffer = AVAudioPCMBuffer(pcmFormat: initialFormat, frameCapacity: 1024)!
        initialBuffer.frameLength = 1024
        sink.append(initialBuffer)
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            if index % 9 == 0 { sink.finish() }
            guard let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
                let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1024)
            else {
                preconditionFailure("Could not construct a test audio buffer")
            }
            buffer.frameLength = 1024
            sink.append(buffer)
        }
        sink.finish()
        sink.append(initialBuffer)
        var iterator = audio.makeAsyncIterator()
        let first = try await iterator.next()
        precondition(first?.format == convertedFormat && first!.frameLength > 0)
        while let converted = try await iterator.next() {
            precondition(converted.format == convertedFormat && converted.frameLength > 0)
        }

        let (invalidAudio, invalidOutput) = AsyncThrowingStream<AVAudioPCMBuffer, Error>.makeStream()
        let invalidSink = SpeechAudioSink(
            converter: AVAudioConverter(from: convertedFormat, to: initialFormat)!, output: invalidOutput) { $0 }
        invalidSink.append(initialBuffer)
        var invalidIterator = invalidAudio.makeAsyncIterator()
        do {
            _ = try await invalidIterator.next()
            preconditionFailure("A changed hardware format must report an error")
        } catch let error as CocoaError { precondition(error.code == .formatting) }
        print(
            "Voice checks passed: phrase revisions across pauses, 100 authorization callbacks, audio conversion, concurrent stop and format error"
        )
    }

    private static func checkPhrases() {
        var draft = DictationDraft(confirmed: "Déjà écrit")
        precondition(draft.receive("bon", isFinal: false) == "Déjà écrit bon")
        precondition(draft.receive("bonjour.", isFinal: true) == "Déjà écrit bonjour.")
        precondition(draft.receive("", isFinal: false) == "Déjà écrit bonjour.")
        precondition(draft.receive("Je", isFinal: false) == "Déjà écrit bonjour. Je")
        precondition(draft.receive("Je reprends.", isFinal: true) == "Déjà écrit bonjour. Je reprends.")
        precondition(draft.receive(" Je reprends.", isFinal: true) == "Déjà écrit bonjour. Je reprends. Je reprends.")
        precondition(draft.receive("", isFinal: true) == "Déjà écrit bonjour. Je reprends. Je reprends.")

        var fresh = DictationDraft(confirmed: "")
        precondition(fresh.receive("test", isFinal: false) == "test")
        precondition(fresh.receive("", isFinal: false) == "")
        precondition(fresh.receive("texte", isFinal: true) == "texte")
        precondition(fresh.receive(".", isFinal: true) == "texte.")

        // English and most languages: no space before closing or high punctuation.
        var english = DictationDraft(confirmed: "Is it ready")
        precondition(english.receive("?", isFinal: true) == "Is it ready?")
        precondition(english.receive("Yes", isFinal: true) == "Is it ready? Yes")
        precondition(english.receive("!", isFinal: true) == "Is it ready? Yes!")
        var note = DictationDraft(confirmed: "Note")
        precondition(note.receive(":", isFinal: true) == "Note:")
        precondition(note.receive("done", isFinal: true) == "Note: done")
        precondition(note.receive(";", isFinal: true) == "Note: done;")
        var bracket = DictationDraft(confirmed: "[draft")
        precondition(bracket.receive("]", isFinal: true) == "[draft]")

        // French typography: a space before `!?;:`, none before `,` or `.`.
        var french = DictationDraft(confirmed: "Bonjour", spacesBeforeHighPunctuation: true)
        precondition(french.receive("!", isFinal: true) == "Bonjour !")
        precondition(french.receive("Ça va", isFinal: false) == "Bonjour ! Ça va")
        precondition(french.receive("Ça va", isFinal: true) == "Bonjour ! Ça va")
        precondition(french.receive("?", isFinal: true) == "Bonjour ! Ça va ?")
        precondition(french.receive("Oui", isFinal: true) == "Bonjour ! Ça va ? Oui")
        precondition(french.receive(",", isFinal: true) == "Bonjour ! Ça va ? Oui,")
        precondition(french.receive("voici", isFinal: true) == "Bonjour ! Ça va ? Oui, voici")
        precondition(french.receive(":", isFinal: true) == "Bonjour ! Ça va ? Oui, voici :")
    }
}
