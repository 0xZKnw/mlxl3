import SwiftUI

struct MessageActionsView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var message: ChatMessage
    var body: some View {
        HStack(spacing: 2) {
            if message.role == .user {
                action(L("Modifier dans une nouvelle branche", "Edit in a new branch"), "pencil") { studio.editMessage(message.id) }
                    .disabled(studio.isGenerating || message.isStreaming)
            } else {
                action(L("Régénérer dans une nouvelle branche", "Regenerate in a new branch"), "arrow.clockwise") { studio.regenerate(message.id) }
                    .disabled(!studio.engineState.isReady || message.isStreaming)
                action(L("Lire la réponse", "Read answer aloud"), "speaker.wave.2") { studio.voice.speak(message.content, locale: studio.voiceLocale) }
                    .disabled(message.isStreaming || message.content.isEmpty)
                action(L("Arrêter la lecture", "Stop reading"), "speaker.slash") { studio.voice.stopSpeaking() }
            }
            action(L("Créer une branche ici", "Branch from here"), "arrow.triangle.branch") { studio.branchConversation(at: message.id) }
                .disabled(studio.isGenerating || message.isStreaming)
        }
    }

    /// One quiet icon button; the name is its tooltip and its accessibility label.
    private func action(_ title: String, _ symbol: String, perform: @escaping () -> Void) -> some View {
        Button(action: perform) {
            Image(systemName: symbol)
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(StudioTheme.quiet)
                .frame(width: 26, height: 26)
        }
        .buttonStyle(StudioControlStyle())
        .help(title)
        .accessibilityLabel(title)
    }
}

struct ComposerVoiceControl: View {
    @ObservedObject var studio: StudioModel
    @ObservedObject var voice: VoiceController
    var body: some View {
        HStack {
            Button {
                if voice.recording || voice.starting { voice.stop() } else { studio.startDictation() }
            } label: {
                Label(voice.recording || voice.starting ? L("Arrêter la dictée", "Stop dictation") : L("Dicter", "Dictate"), systemImage: voice.recording ? "mic.fill" : "mic")
                    .labelStyle(.iconOnly).frame(width: 28, height: 28)
                    .foregroundStyle(voice.recording ? Color.red : StudioTheme.secondary)
            }.disabled(studio.isGenerating).help(L("Dictée sur ce Mac uniquement", "On-device dictation only"))
            if voice.starting { ProgressView().controlSize(.mini) }
            if let error = voice.error { Text(error).foregroundStyle(.orange).lineLimit(2).frame(maxWidth: 180) }
        }.buttonStyle(.plain).font(.system(size: 13)).foregroundStyle(StudioTheme.secondary)
        .onDisappear { voice.stop() }
    }
}
