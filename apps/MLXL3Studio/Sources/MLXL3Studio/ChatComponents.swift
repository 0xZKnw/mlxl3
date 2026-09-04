import SwiftUI

struct MessagesView: View {
    let messages: [ChatMessage]

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 27) {
                    ForEach(messages) { message in
                        MessageView(message: message)
                            .id(message.id)
                    }
                    Color.clear
                        .frame(height: 4)
                        .id("conversation-bottom")
                }
                .padding(.horizontal, 48)
                .padding(.top, 28)
                .padding(.bottom, 26)
                .frame(maxWidth: 920)
                .frame(maxWidth: .infinity)
            }
            .scrollIndicators(.never)
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .onChange(of: messages.last?.id) {
                Task { @MainActor in
                    await Task.yield()
                    var transaction = Transaction(animation: nil)
                    transaction.disablesAnimations = true
                    withTransaction(transaction) {
                        proxy.scrollTo("conversation-bottom", anchor: .bottom)
                    }
                }
            }
            .overlay {
                if let lastMessage = messages.last {
                    StreamingScrollFollower(message: lastMessage) {
                        var transaction = Transaction(animation: nil)
                        transaction.disablesAnimations = true
                        withTransaction(transaction) {
                            proxy.scrollTo("conversation-bottom", anchor: .bottom)
                        }
                    }
                    .frame(width: 0, height: 0)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct StreamingScrollFollower: View {
    @ObservedObject var message: ChatMessage
    let scrollToBottom: @MainActor () -> Void
    @State private var pendingScroll: Task<Void, Never>?

    var body: some View {
        Color.clear
            .onChange(of: message.streamRevision) {
                guard pendingScroll == nil else { return }
                pendingScroll = Task { @MainActor in
                    try? await Task.sleep(for: .milliseconds(250))
                    guard !Task.isCancelled else { return }
                    scrollToBottom()
                    pendingScroll = nil
                }
            }
            .onDisappear {
                pendingScroll?.cancel()
                pendingScroll = nil
            }
    }
}

private struct MessageView: View {
    @ObservedObject var message: ChatMessage

    var body: some View {
        if message.role == .user {
            HStack {
                Spacer(minLength: 120)
                Text(message.content)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(Color.white.opacity(0.94))
                    .textSelection(.enabled)
                    .padding(.horizontal, 17)
                    .padding(.vertical, 12)
                    .premiumGlass(radius: 18, tint: Color.white.opacity(0.045))
            }
        } else {
            AssistantMessageView(message: message)
        }
    }
}

private struct AssistantMessageView: View {
    @ObservedObject var message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 13) {
            LogoMark(size: 27)
                .padding(.top, 1)

            VStack(alignment: .leading, spacing: 13) {
                HStack(spacing: 7) {
                    Text("MLXL3")
                        .font(.system(size: 11, weight: .bold, design: .rounded))
                        .tracking(0.55)
                    if message.isStreaming {
                        StreamingIndicator()
                    }
                }

                if !message.thinking.isEmpty {
                    ThinkingBlock(
                        text: message.thinking,
                        streaming: message.isStreaming && message.content.isEmpty
                    )
                } else if message.isStreaming && message.content.isEmpty {
                    ThinkingPlaceholder()
                }

                if !message.toolActivities.isEmpty {
                    VStack(spacing: 7) {
                        ForEach(message.toolActivities) { activity in
                            MCPToolActivityRow(activity: activity)
                        }
                    }
                }

                if !message.content.isEmpty {
                    MarkdownResponseView(
                        message.content,
                        streaming: message.isStreaming
                    )
                }

                if let error = message.error {
                    Label(error, systemImage: "exclamationmark.circle.fill")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(Color(red: 1.0, green: 0.48, blue: 0.50))
                }

                if let stats = message.stats {
                    StatsRow(stats: stats)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

}

private struct MCPToolActivityRow: View {
    let activity: ToolActivity
    @State private var expanded = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            if let result = activity.result, !result.isEmpty {
                Text(result)
                    .font(.system(size: 10.5, weight: .regular, design: .monospaced))
                    .foregroundStyle(Color.white.opacity(0.58))
                    .lineLimit(10)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 8)
            }
        } label: {
            HStack(spacing: 8) {
                stateIcon
                VStack(alignment: .leading, spacing: 1) {
                    Text(activity.toolName)
                        .font(.system(size: 10.5, weight: .semibold, design: .rounded))
                        .lineLimit(1)
                    if let server = activity.serverName {
                        Text("MCP · \(server)")
                            .font(.system(size: 8, weight: .bold, design: .rounded))
                            .tracking(0.65)
                            .foregroundStyle(StudioTheme.quiet)
                    }
                }
                Spacer()
                Text(stateLabel)
                    .font(.system(size: 8, weight: .bold, design: .rounded))
                    .tracking(0.5)
                    .foregroundStyle(stateColor)
            }
        }
        .tint(Color.white.opacity(0.42))
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(Color.white.opacity(0.025), in: RoundedRectangle(cornerRadius: 11))
        .overlay {
            RoundedRectangle(cornerRadius: 11)
                .stroke(Color.white.opacity(0.07), lineWidth: 0.6)
        }
    }

    @ViewBuilder
    private var stateIcon: some View {
        if activity.state == .running {
            ProgressView()
                .controlSize(.mini)
                .tint(StudioTheme.accent)
        } else {
            Image(systemName: activity.state == .complete ? "checkmark.circle.fill" : "xmark.circle.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(stateColor)
        }
    }

    private var stateLabel: String {
        switch activity.state {
        case .running: "EXÉCUTION"
        case .complete: "TERMINÉ"
        case .failed: "ERREUR"
        }
    }

    private var stateColor: Color {
        activity.state == .failed
            ? Color(red: 1, green: 0.48, blue: 0.50)
            : StudioTheme.accent.opacity(0.82)
    }
}

private struct StreamingIndicator: View {
    @State private var pulse = false

    var body: some View {
        Circle()
            .fill(StudioTheme.accent)
            .frame(width: 5, height: 5)
            .opacity(pulse ? 0.3 : 1)
            .scaleEffect(pulse ? 0.72 : 1)
            .animation(.easeInOut(duration: 0.72).repeatForever(), value: pulse)
            .onAppear { pulse = true }
    }
}

private struct ThinkingPlaceholder: View {
    @State private var shimmer = false

    var body: some View {
        HStack(spacing: 9) {
            Image(systemName: "ellipsis")
                .font(.system(size: 10, weight: .bold))
            Text("RAISONNEMENT")
                .font(.system(size: 9, weight: .bold, design: .rounded))
                .tracking(1.25)
            HStack(spacing: 3) {
                ForEach(0..<3, id: \.self) { index in
                    Circle()
                        .frame(width: 3, height: 3)
                        .opacity(shimmer ? 0.25 + Double(index) * 0.28 : 0.9 - Double(index) * 0.25)
                }
            }
        }
        .foregroundStyle(StudioTheme.thinking)
        .onAppear { withAnimation(.easeInOut(duration: 0.8).repeatForever()) { shimmer = true } }
    }
}

private struct ThinkingBlock: View {
    let text: String
    let streaming: Bool
    @State private var expanded = true

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            ScrollView {
                let chunks = StreamingTextChunker.chunks(text)
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(chunks) { chunk in
                        ThinkingTextChunk(
                            source: chunk.source,
                            streaming: streaming && chunk.id == chunks.last?.id
                        )
                        .equatable()
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 10)
            }
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .defaultScrollAnchor(.bottom, for: .sizeChanges)
            .frame(maxHeight: 180)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "ellipsis")
                    .font(.system(size: 10, weight: .bold))
                Text(streaming ? "RAISONNEMENT EN COURS" : "RAISONNEMENT")
                    .font(.system(size: 9, weight: .bold, design: .rounded))
                    .tracking(1.15)
                if streaming { StreamingIndicator() }
            }
            .foregroundStyle(StudioTheme.thinking)
        }
        .tint(Color.white.opacity(0.52))
        .padding(.horizontal, 14)
        .padding(.vertical, 12)
        .premiumGlass(radius: 14, tint: Color.white.opacity(0.018))
    }
}

private struct ThinkingTextChunk: View, Equatable {
    let source: String
    let streaming: Bool

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.streaming == rhs.streaming
    }

    var body: some View {
        Text(source)
            .font(.system(size: 11.5, weight: .regular, design: .monospaced))
            .foregroundStyle(Color.white.opacity(streaming ? 0.55 : 0.57))
            .lineSpacing(3)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct StatsRow: View {
    let stats: GenerationStats

    var body: some View {
        HStack(spacing: 7) {
            MetricChip(icon: "bolt.fill", value: String(format: "%.1f tok/s", stats.decodeTps), label: "decode")
            MetricChip(icon: "arrow.right.to.line", value: String(format: "%.1f tok/s", stats.prefillTps), label: "prefill")
            MetricChip(icon: "timer", value: String(format: "%.0f ms", stats.ttftSeconds * 1000), label: "TTFT")
            MetricChip(icon: "memorychip", value: String(format: "%.2f GB", stats.peakMemoryGB), label: "pic")
            MetricChip(
                icon: "externaldrive.badge.checkmark",
                value: "\(stats.cachedPromptTokens ?? 0)/\(stats.evaluatedPromptTokens ?? stats.promptTokens)",
                label: String(format: "cache %.0f%%", stats.cacheHitPercent)
            )
        }
        .padding(.top, 3)
    }
}

private struct MetricChip: View {
    let icon: String
    let value: String
    let label: String

    var body: some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 8, weight: .semibold))
            Text(value)
                .foregroundStyle(Color.white.opacity(0.72))
            Text(label)
                .foregroundStyle(StudioTheme.quiet)
        }
        .font(.system(size: 9, weight: .medium, design: .rounded))
        .padding(.horizontal, 8)
        .frame(height: 24)
        .background(Color.white.opacity(0.04), in: Capsule())
        .overlay { Capsule().stroke(Color.white.opacity(0.07), lineWidth: 0.6) }
    }
}

struct ComposerView: View {
    @EnvironmentObject private var studio: StudioModel
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .bottom, spacing: 12) {
                TextField("Écris un message…", text: $studio.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 14))
                    .lineLimit(1...6)
                    .focused($focused)
                    .padding(.vertical, 5)
                    .frame(minHeight: 34, alignment: .topLeading)
                    .contentShape(Rectangle())
                    .disabled(studio.isGenerating)
                    .onSubmit { studio.send() }
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.control) {
                            studio.draft.append("\n")
                        } else {
                            studio.send()
                        }
                        return .handled
                    }

                if studio.isGenerating {
                    Button(action: studio.stopGeneration) {
                        Image(systemName: "stop.fill")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(Color.black)
                            .frame(width: 31, height: 31)
                    }
                    .buttonStyle(RoundGlassButtonStyle(bright: true))
                    .help("Arrêter")
                } else {
                    Button(action: studio.send) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(studio.canSend ? Color.black : Color.white.opacity(0.35))
                            .frame(width: 31, height: 31)
                    }
                    .buttonStyle(RoundGlassButtonStyle(bright: true))
                    .disabled(!studio.canSend)
                }
            }
            .padding(.horizontal, 15)
            .padding(.top, 12)
            .padding(.bottom, 10)

            HStack(spacing: 7) {
                Label("LOCAL", systemImage: "lock.fill")
                Text("·")
                Text("EXL3")
                Text("·")
                Text("APPLE METAL")
                Spacer()
                Text("↩ ENVOYER  ·  ⌃↩ NOUVELLE LIGNE")
            }
            .font(.system(size: 8.5, weight: .semibold, design: .rounded))
            .tracking(0.65)
            .foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 16)
            .padding(.bottom, 10)
        }
        .contentShape(RoundedRectangle(cornerRadius: 23, style: .continuous))
        .onTapGesture {
            guard !studio.isGenerating else { return }
            focused = true
        }
        .premiumGlass(radius: 23, tint: Color.white.opacity(0.035))
        .frame(maxWidth: 790)
        .frame(maxWidth: .infinity)
        .onAppear { focused = true }
        .onChange(of: studio.isGenerating) {
            if !studio.isGenerating {
                focused = true
            }
        }
    }
}
