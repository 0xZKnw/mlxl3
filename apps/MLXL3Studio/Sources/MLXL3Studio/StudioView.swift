import SwiftUI

struct StudioView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        ZStack {
            StudioTheme.canvas
            RadialGradient(
                colors: [Color(red: 0.10, green: 0.16, blue: 0.22).opacity(0.28), .clear],
                center: UnitPoint(x: 0.72, y: 0.08),
                startRadius: 20,
                endRadius: 620
            )

            HStack(spacing: 0) {
                SidebarView()
                    .frame(width: 250)

                Rectangle()
                    .fill(Color.white.opacity(0.075))
                    .frame(width: 1)

                ChatWorkspaceView()
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                if studio.showInspector {
                    Rectangle()
                        .fill(Color.white.opacity(0.075))
                        .frame(width: 1)
                    GenerationInspector()
                        .frame(width: 294)
                        .transition(.move(edge: .trailing).combined(with: .opacity))
                }
            }
        }
        .ignoresSafeArea()
        .frame(minWidth: 980, minHeight: 650)
        .animation(.snappy(duration: 0.28), value: studio.showInspector)
    }
}

private struct SidebarView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 11) {
                LogoMark(size: 32)
                VStack(alignment: .leading, spacing: -1) {
                    Text("MLXL3")
                        .font(.system(size: 16, weight: .bold, design: .rounded))
                        .tracking(0.7)
                    Text("DESKTOP")
                        .font(.system(size: 8, weight: .semibold, design: .rounded))
                        .tracking(1.8)
                        .foregroundStyle(StudioTheme.quiet)
                }
                Spacer()
            }
            .padding(.top, 44)
            .padding(.horizontal, 18)

            Button(action: studio.newConversation) {
                HStack(spacing: 9) {
                    Image(systemName: "plus")
                        .font(.system(size: 12, weight: .bold))
                    Text("Nouvelle conversation")
                        .font(.system(size: 13, weight: .semibold))
                    Spacer()
                    Text("⌘N")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .foregroundStyle(StudioTheme.quiet)
                }
                .padding(.horizontal, 13)
                .frame(height: 40)
            }
            .buttonStyle(PrimaryGlassButtonStyle())
            .padding(.top, 24)
            .padding(.horizontal, 14)

            Text("CONVERSATIONS")
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .tracking(1.4)
                .foregroundStyle(StudioTheme.quiet)
                .padding(.horizontal, 19)
                .padding(.top, 27)
                .padding(.bottom, 9)

            ScrollView {
                LazyVStack(spacing: 3) {
                    ForEach(studio.conversations) { conversation in
                        ConversationRow(
                            conversation: conversation,
                            selected: conversation.id == studio.selectedConversationID
                        )
                        .onTapGesture { studio.selectedConversationID = conversation.id }
                        .contextMenu {
                            Button("Supprimer", role: .destructive) {
                                studio.deleteConversation(conversation.id)
                            }
                        }
                    }
                }
                .padding(.horizontal, 9)
            }

            Spacer(minLength: 12)

            if let model = studio.selectedModel {
                VStack(alignment: .leading, spacing: 9) {
                    HStack {
                        StatusDot(state: studio.engineState)
                        Text(studio.engineState.label)
                            .font(.system(size: 10, weight: .semibold))
                            .foregroundStyle(StudioTheme.secondary)
                        Spacer()
                    }
                    Text(model.name)
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(1)
                    HStack(spacing: 6) {
                        Text(model.format)
                        Text("·")
                        Text(model.bits.map { String(format: "%.2f BPW", $0) } ?? "BPW —")
                        Text("·")
                        Text(model.size)
                    }
                    .font(.system(size: 9, weight: .medium, design: .rounded))
                    .foregroundStyle(StudioTheme.quiet)
                }
                .padding(13)
                .premiumGlass(radius: 15, tint: Color.white.opacity(0.025))
                .padding(.horizontal, 12)
                .padding(.bottom, 14)
            }
        }
        .background(StudioTheme.sidebar.opacity(0.93))
    }
}

private struct ConversationRow: View {
    let conversation: Conversation
    let selected: Bool

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: conversation.messages.isEmpty ? "sparkles" : "bubble.left")
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(selected ? Color.white : StudioTheme.quiet)
                .frame(width: 16)
            Text(conversation.title)
                .font(.system(size: 12, weight: selected ? .semibold : .regular))
                .foregroundStyle(selected ? Color.white : StudioTheme.secondary)
                .lineLimit(1)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 10)
        .frame(height: 36)
        .background(
            selected ? Color.white.opacity(0.085) : Color.clear,
            in: RoundedRectangle(cornerRadius: 10, style: .continuous)
        )
        .contentShape(Rectangle())
    }
}

private struct ChatWorkspaceView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceHeader()

            if let conversation = studio.currentConversation, !conversation.messages.isEmpty {
                MessagesView(messages: conversation.messages)
            } else {
                WelcomeView()
            }

            ComposerView()
                .padding(.horizontal, 34)
                .padding(.bottom, 24)
        }
    }
}

private struct WorkspaceHeader: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        HStack(spacing: 12) {
            Menu {
                ForEach(studio.models) { model in
                    Button {
                        studio.selectModel(model.name)
                    } label: {
                        if model.name == studio.selectedModelName {
                            Label(model.name, systemImage: "checkmark")
                        } else {
                            Text(model.name)
                        }
                    }
                }
            } label: {
                HStack(spacing: 9) {
                    StatusDot(state: studio.engineState)
                    Text(studio.selectedModelName ?? "Choisir un modèle")
                        .font(.system(size: 12, weight: .semibold))
                        .lineLimit(1)
                    Image(systemName: "chevron.down")
                        .font(.system(size: 9, weight: .bold))
                        .foregroundStyle(StudioTheme.quiet)
                }
                .padding(.horizontal, 13)
                .frame(height: 36)
            }
            .buttonStyle(GlassPillButtonStyle())
            .disabled(studio.models.isEmpty || studio.isGenerating)

            Text(studio.engineState.label)
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(StudioTheme.quiet)
                .lineLimit(1)

            Spacer()

            Button(action: studio.ejectModel) {
                Image(systemName: "eject.fill")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 34, height: 34)
            }
            .buttonStyle(RoundGlassButtonStyle())
            .disabled(!studio.canEject)
            .help("Éjecter le modèle et libérer la mémoire Metal")

            Button {
                studio.showInspector.toggle()
            } label: {
                Image(systemName: "slider.horizontal.3")
                    .font(.system(size: 12, weight: .semibold))
                    .frame(width: 34, height: 34)
            }
            .buttonStyle(RoundGlassButtonStyle())
            .help("Réglages de génération")
        }
        .padding(.top, 35)
        .padding(.horizontal, 24)
        .padding(.bottom, 13)
    }
}

private struct WelcomeView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        VStack(spacing: 0) {
            Spacer()
            ZStack {
                Circle()
                    .fill(Color.white.opacity(0.05))
                    .frame(width: 116, height: 116)
                    .blur(radius: 18)
                LogoMark(size: 64)
            }
            Text("Intelligence locale, sans compromis.")
                .font(.system(size: 25, weight: .semibold, design: .rounded))
                .tracking(-0.45)
                .padding(.top, 24)
            Text("Discute avec tes modèles EXL3 directement sur Apple Metal.\nPrivé, rapide, entièrement sur ce Mac.")
                .font(.system(size: 12.5, weight: .regular))
                .foregroundStyle(StudioTheme.secondary)
                .multilineTextAlignment(.center)
                .lineSpacing(4)
                .padding(.top, 11)

            if case let .failed(message) = studio.engineState {
                Label(message, systemImage: "exclamationmark.triangle.fill")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(Color(red: 1, green: 0.56, blue: 0.56))
                    .padding(.horizontal, 14)
                    .padding(.vertical, 10)
                    .background(Color.red.opacity(0.08), in: Capsule())
                    .padding(.top, 22)
            }
            Spacer()
            Spacer().frame(height: 48)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
