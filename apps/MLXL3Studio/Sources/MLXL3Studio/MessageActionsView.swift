import SwiftUI

struct MessageActionsView: View {
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var message: ChatMessage
    var body: some View {
        HStack(spacing: 12) {
            if message.role == .user {
                Button { studio.editMessage(message.id) } label: { Label(L("Modifier dans une nouvelle branche", "Edit in a new branch"), systemImage: "pencil") }
                    .disabled(studio.isGenerating || message.isStreaming)
            } else {
                Button { studio.regenerate(message.id) } label: { Label(L("Régénérer dans une nouvelle branche", "Regenerate in a new branch"), systemImage: "arrow.clockwise") }
                    .disabled(!studio.engineState.isReady || message.isStreaming)
                Button { studio.voice.speak(message.content, locale: studio.voiceLocale) } label: { Label(L("Lire la réponse", "Read answer aloud"), systemImage: "speaker.wave.2") }
                    .disabled(message.isStreaming || message.content.isEmpty)
                Button { studio.voice.stopSpeaking() } label: { Label(L("Arrêter la lecture", "Stop reading"), systemImage: "speaker.slash") }
            }
            Button { studio.branchConversation(at: message.id) } label: { Label(L("Créer une branche ici", "Branch from here"), systemImage: "arrow.triangle.branch") }
                .disabled(studio.isGenerating || message.isStreaming)
        }.labelStyle(.iconOnly).buttonStyle(.plain).font(.caption).foregroundStyle(StudioTheme.secondary)
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
                Label(voice.recording ? L("Arrêter la dictée", "Stop dictation") : L("Dicter", "Dictate"), systemImage: voice.recording ? "mic.fill" : "mic")
            }.disabled(studio.isGenerating).help(L("Dictée sur ce Mac uniquement", "On-device dictation only"))
            if voice.starting { ProgressView().controlSize(.mini) }
            if let error = voice.error { Text(error).foregroundStyle(.orange).lineLimit(2) }
            Spacer()
            Button { studio.exportMarkdown() } label: { Label(L("Exporter cette conversation en Markdown", "Export this conversation as Markdown"), systemImage: "square.and.arrow.up").labelStyle(.iconOnly) }
        }.buttonStyle(.plain).font(.caption).foregroundStyle(StudioTheme.secondary).padding(.horizontal, 18).padding(.top, 10)
        .onDisappear { voice.stop() }
    }
}
