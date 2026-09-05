import SwiftUI

struct MessagesView: View {
    let messages: [ChatMessage]
    @State private var followsBottom = true
    @State private var userIsScrolling = false

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 38) {
                    ForEach(messages) { message in
                        MessageView(message: message)
                            .id(message.id)
                    }
                    Color.clear
                        .frame(height: 4)
                        .id("conversation-bottom")
                }
                .padding(.horizontal, 28)
                .padding(.top, 38)
                .padding(.bottom, 26)
                .frame(maxWidth: 820)
                .frame(maxWidth: .infinity)
            }
            .scrollIndicators(.never)
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .onScrollPhaseChange { _, phase in
                userIsScrolling = phase == .interacting || phase == .decelerating
            }
            .onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.contentSize.height - geometry.visibleRect.maxY < 100
            } action: { _, nearBottom in
                if userIsScrolling || nearBottom { followsBottom = nearBottom }
            }
            .onChange(of: messages.last?.id) {
                followsBottom = true
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
                        guard followsBottom else { return }
                        var transaction = Transaction(animation: nil)
                        transaction.disablesAnimations = true
                        withTransaction(transaction) {
                            proxy.scrollTo("conversation-bottom", anchor: .bottom)
                        }
                    }
                    .frame(width: 0, height: 0)
                }
            }
            .overlay(alignment: .bottom) {
                if !followsBottom {
                    Button {
                        followsBottom = true
                        proxy.scrollTo("conversation-bottom", anchor: .bottom)
                    } label: {
                        Label(L("Dernier message", "Latest message"), systemImage: "arrow.down")
                            .font(.system(size: 11, weight: .medium))
                            .padding(.horizontal, 14).padding(.vertical, 9)
                    }
                    .buttonStyle(GlassPillButtonStyle())
                    .padding(.bottom, 12)
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
          HStack(alignment: .top) {
            Spacer(minLength: 64)
            VStack(alignment: .trailing, spacing: 9) {
                Text(L("Vous", "You"))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(StudioTheme.quiet)
                Text(message.content)
                    .font(.system(size: 15, weight: .medium))
                    .foregroundStyle(StudioTheme.ink)
                    .lineSpacing(5)
                    .textSelection(.enabled)
                    .padding(.horizontal, 17)
                    .padding(.vertical, 13)
                    .background(Color.white.opacity(0.055), in: RoundedRectangle(cornerRadius: 12))
            }
          }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .padding(.bottom, 4)
        } else {
            AssistantMessageView(message: message)
        }
    }
}

private struct AssistantMessageView: View {
    @ObservedObject var message: ChatMessage

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 13) {
                HStack(spacing: 7) {
                    MonogramMark(size: 15)
                    Text("MLXL3")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(StudioTheme.secondary)
                    if message.isStreaming {
                        StreamingIndicator()
                    }
                }

                if message.parts.isEmpty && message.isStreaming {
                    ThinkingPlaceholder()
                }
                ForEach(message.parts) { part in
                    let live = message.isStreaming && message.parts.last?.id == part.id
                    switch part.kind {
                    case .thinking:
                        ThinkingBlock(text: part.text, streaming: live)
                    case .answer:
                        MarkdownResponseView(part.text, streaming: live)
                    case .tool:
                        if let activity = message.toolActivities.first(where: { $0.id == part.toolID }) {
                            MCPToolActivityRow(activity: activity)
                        }
                    case .processing:
                        PrefillActivityRow(part: part, live: live)
                    }
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

private struct PrefillActivityRow: View {
    let part: AssistantPart
    let live: Bool

    var body: some View {
        if live {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                row(at: context.date)
            }
        } else {
            row(at: .now)
        }
    }

    private func row(at date: Date) -> some View {
            HStack(spacing: 8) {
                if live {
                    ProgressView().controlSize(.mini)
                } else {
                    Image(systemName: part.interrupted == true ? "stop" : "checkmark")
                        .font(.system(size: 10))
                }
                Text(live ? localizedProgress(part.text) : (part.interrupted == true ? L("Traitement interrompu", "Processing interrupted") : L("Contexte traité", "Context processed")))
                    .lineLimit(2)
                Spacer(minLength: 8)
                let elapsed = part.elapsed ?? date.timeIntervalSince(part.startedAt ?? date)
                Text("\(max(0, Int(elapsed))) s").monospacedDigit()
            }
            .font(.system(size: 11))
            .foregroundStyle(StudioTheme.secondary)
            .padding(.vertical, 4)
            .accessibilityElement(children: .combine)
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
        case .running: L("EXÉCUTION", "RUNNING")
        case .complete: L("TERMINÉ", "DONE")
        case .failed: L("ERREUR", "ERROR")
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
            Text(L("Réflexion en cours", "Thinking"))
                .font(.system(size: 11, weight: .medium))
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
                Text(streaming ? L("Réflexion en cours", "Thinking") : L("Réflexion", "Thinking"))
                    .font(.system(size: 11, weight: .medium))
                if streaming { StreamingIndicator() }
            }
            .foregroundStyle(StudioTheme.thinking)
        }
        .tint(Color.white.opacity(0.52))
        .padding(.leading, 14)
        .padding(.vertical, 5)
        .overlay(alignment: .leading) {
            Rectangle().fill(StudioTheme.edge).frame(width: 1)
        }
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
            .foregroundStyle(StudioTheme.secondary)
            .lineSpacing(3)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
    }
}

private struct StatsRow: View {
    let stats: GenerationStats

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 132), alignment: .leading)], alignment: .leading, spacing: 6) {
            MetricChip(icon: "bolt.fill", value: String(format: "%.1f tok/s", stats.decodeTps), label: "decode")
            MetricChip(icon: "arrow.right.to.line", value: String(format: "%.1f tok/s", stats.prefillTps), label: "prefill")
            MetricChip(icon: "timer", value: String(format: "%.0f ms", stats.ttftSeconds * 1000), label: "TTFT")
            MetricChip(icon: "memorychip", value: String(format: "%.2f GB", stats.peakMemoryGB), label: L("pic", "peak"))
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
        .monospacedDigit()
    }
}

struct ComposerView: View {
    @EnvironmentObject private var studio: StudioModel
    @FocusState private var focused: Bool

    var body: some View {
        VStack(spacing: 0) {
            HStack(alignment: .bottom, spacing: 12) {
                TextField(L("Écrivez un message…", "Write a message…"), text: $studio.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15))
                    .lineLimit(1...6)
                    .focused($focused)
                    .padding(.vertical, 5)
                    .frame(minHeight: 48, alignment: .topLeading)
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
                    .help(L("Arrêter", "Stop"))
                    .accessibilityLabel(L("Arrêter la génération", "Stop generation"))
                } else {
                    Button(action: studio.send) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(studio.canSend ? Color.black : Color.white.opacity(0.35))
                            .frame(width: 31, height: 31)
                    }
                    .buttonStyle(RoundGlassButtonStyle(bright: true))
                    .disabled(!studio.canSend)
                    .help(L("Envoyer le message", "Send message"))
                    .accessibilityLabel(L("Envoyer le message", "Send message"))
                }
            }
            .padding(.horizontal, 18)
            .padding(.top, 17)
            .padding(.bottom, 14)

            ComposerFooterView()
        }
        .contentShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .onTapGesture {
            guard !studio.isGenerating else { return }
            focused = true
        }
        .background(Color(red: 0.105, green: 0.108, blue: 0.108), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(focused ? 0.22 : 0.10), lineWidth: 0.7)
                .allowsHitTesting(false)
        }
        .frame(maxWidth: 790)
        .frame(maxWidth: .infinity)
        .onAppear { focused = true }
        .onChange(of: studio.draft) {
            if !studio.isGenerating && !studio.draft.isEmpty { focused = true }
        }
        .onChange(of: studio.isGenerating) {
            if !studio.isGenerating {
                focused = true
            }
        }
    }
}


private struct ComposerFooterView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
            HStack(spacing: 7) {
                Toggle(isOn: Binding(get: { studio.mcpEnabled }, set: { studio.setMCPEnabled($0) })) {
                    HStack(spacing: 5) {
                        Image(systemName: "network")
                        Text("MCP")
                    }
                }
                .toggleStyle(.switch)
                .controlSize(.mini)
                .fixedSize()
                .tint(StudioTheme.accent)
                .disabled(studio.isGenerating || studio.mcpUpdating)
                .accessibilityLabel(L("Activer les outils MCP", "Enable MCP tools"))
                .help(L("Exa et les serveurs MCP configurés. Les requêtes peuvent quitter ce Mac. Choix mémorisé.", "Exa and configured MCP servers. Requests may leave this Mac. Your choice is saved."))
                if studio.mcpUpdating {
                    ProgressView().controlSize(.mini).help(L("Connexion MCP…", "Connecting MCP…"))
                } else if studio.mcpEnabled && !studio.mcpErrors.isEmpty {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .help(studio.mcpErrors.values.sorted().joined(separator: "\n"))
                }
                Rectangle().fill(StudioTheme.edge).frame(width: 1, height: 12)
                HStack(spacing: 6) {
                    StatusDot(state: studio.engineState)
                    Text(studio.isGenerating ? L("Génération en cours", "Generating") : studio.engineState.label)
                }
                Spacer()
                Text(L("↵ Envoyer     ⌃↵ Nouvelle ligne", "↵ Send     ⌃↵ New line"))
                Button { studio.showInspector.toggle() } label: {
                    Label(studio.contextLabel, systemImage: "chart.bar")
                        .monospacedDigit()
                }
                .buttonStyle(.plain)
                .help(L("Contexte : prompt, outils et génération du dernier envoi. Brouillon non inclus. Cliquer pour régler la limite.", "Context: prompt, tools and generation from the last send. Draft excluded. Click to adjust the limit."))
                .accessibilityLabel(L("Contexte utilisé", "Context used") + " " + studio.contextLabel)
            }
            .font(.system(size: 10, weight: .regular))
            .foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 18)
            .padding(.bottom, 13)
    }
}
