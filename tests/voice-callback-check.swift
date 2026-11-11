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
        let sink = SpeechAudioSink(SFSpeechAudioBufferRecognitionRequest())
        let initialFormat = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!
        let initialBuffer = AVAudioPCMBuffer(pcmFormat: initialFormat, frameCapacity: 32)!
        initialBuffer.frameLength = 32
        sink.append(initialBuffer)
        DispatchQueue.concurrentPerform(iterations: 64) { index in
            if index % 9 == 0 { sink.finish() }
            guard let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1),
                  let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 32) else {
                preconditionFailure("Could not construct a test audio buffer")
            }
            buffer.frameLength = 32
            sink.append(buffer)
        }
        sink.finish()
        print("Voice callback checks passed: 100 background authorization callbacks; concurrent audio append/finish")
    }
}
