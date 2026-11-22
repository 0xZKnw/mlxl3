import SwiftUI
import UniformTypeIdentifiers

struct StreamingScrollState {
  var followsBottom = true
  private(set) var userIsScrolling = false
  var shouldFollow: Bool { followsBottom && !userIsScrolling }

  mutating func phaseChanged(_ phase: ScrollPhase) {
    userIsScrolling = phase == .tracking || phase == .interacting || phase == .decelerating
  }

  mutating func proximityChanged(_ nearBottom: Bool) {
    if userIsScrolling || nearBottom { followsBottom = nearBottom }
  }

  /// A followed transcript more than a couple of points from its scroll end (or past it)
  /// is re-pinned; layout rounding stays inside this tolerance.
  static func needsRepin(gap: CGFloat) -> Bool { abs(gap) > 2 }
  /// Delays (ms) between the re-pin checks that follow a content or viewport size change.
  static let repinChecks = [0, 16, 33, 66, 133]
}

private struct BottomMetrics: Equatable {
  var height: CGFloat = 0
  var gap: CGFloat = 0
  var container = CGSize.zero

  /// Changes when the content or the viewport is resized, never when the reader scrolls.
  struct Layout: Hashable {
    let height: CGFloat
    let width: CGFloat
    let viewport: CGFloat
  }
  var layout: Layout { Layout(height: height, width: container.width, viewport: container.height) }
}

struct MessagesView: View {
  /// Recent messages are measured eagerly. A `LazyVStack` re-estimates rows that leave its
  /// window (their Markdown renders again from empty), which moved the transcript height by
  /// thousands of points while a reply streamed and made the followed view jump up and down.
  static let measuredTail = 12

  @Environment(\.conversationDensity) private var density
  @Environment(\.conversationReadingWidth) private var readingWidth
  let messages: [ChatMessage]
  @State private var scroll = StreamingScrollState()
  @State private var metrics = BottomMetrics()
  @Environment(\.accessibilityReduceMotion) private var reduceMotion

  var body: some View {
    let split = max(0, messages.count - Self.measuredTail)
    ScrollViewReader { proxy in
      ScrollView {
        VStack(spacing: density.spacing) {
          if split > 0 {
            LazyVStack(spacing: density.spacing) {
              ForEach(messages.prefix(split)) { message in
                MessageView(message: message)
                  .id(message.id)
              }
            }
          }
          ForEach(messages.suffix(from: split)) { message in
            MessageView(message: message)
              .transition(
                reduceMotion || message.id == messages.last?.id
                  ? .identity : .offset(y: 7).combined(with: .opacity))
              .id(message.id)
          }
          // The marker carries the 26 pt bottom margin, so pinning it reaches the scroll end,
          // where the native size-change anchor engages.
          Color.clear
            .frame(height: 30)
            .id("conversation-bottom")
        }
        .padding(.horizontal, 28)
        .padding(.top, 38)
        .frame(maxWidth: readingWidth.messageMaximum)
        .frame(maxWidth: .infinity)
      }
      .scrollIndicators(.never)
      .mask(StudioEdgeFade(top: 26, bottom: 16))
      .defaultScrollAnchor(.bottom, for: .initialOffset)
      // Growth keeps the bottom pinned natively. Animated `scrollTo` calls restarted every
      // 50 ms batch: each one showed the target for a frame, then slid from the old offset.
      .defaultScrollAnchor(scroll.shouldFollow ? .bottom : nil, for: .sizeChanges)
      .onScrollPhaseChange { _, phase in
        scroll.phaseChanged(phase)
      }
      .onScrollGeometryChange(for: Bool.self) { geometry in
        geometry.contentSize.height - geometry.visibleRect.maxY < 100
      } action: { _, nearBottom in
        scroll.proximityChanged(nearBottom)
      }
      .onScrollGeometryChange(for: BottomMetrics.self) { geometry in
        BottomMetrics(
          height: geometry.contentSize.height.rounded(),
          gap: (geometry.contentSize.height - geometry.visibleRect.maxY).rounded(),
          container: CGSize(
            width: geometry.containerSize.width.rounded(),
            height: geometry.containerSize.height.rounded()))
      } action: { _, value in
        metrics = value
      }
      .task(id: metrics.layout) {
        // Safety net when growth or a resize moved the end and the anchor did not absorb it:
        // re-pin without animation. A reader's own scroll never starts it. The anchor and a
        // reflow can settle a few frames after the change (a resize anchored to the old
        // viewport height), so the gap is checked again for a short while.
        guard metrics.height > 0 else { return }
        for delay in StreamingScrollState.repinChecks {
          if delay == 0 { await Task.yield() } else { try? await Task.sleep(for: .milliseconds(delay)) }
          guard !Task.isCancelled else { return }
          if scroll.shouldFollow, StreamingScrollState.needsRepin(gap: metrics.gap) { pin(proxy) }
        }
      }
      .task(id: messages.last?.id) {
        scroll.followsBottom = true
        await Task.yield()
        guard !Task.isCancelled else { return }
        pin(proxy)
      }
      .overlay(alignment: .bottom) {
        if !scroll.followsBottom {
          Button {
            scroll.followsBottom = true
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

  private func pin(_ proxy: ScrollViewProxy) {
    var transaction = Transaction(animation: nil)
    transaction.disablesAnimations = true
    withTransaction(transaction) {
      proxy.scrollTo("conversation-bottom", anchor: .bottom)
    }
  }
}

private struct MessageView: View {
    @Environment(\.conversationTextScale) private var textScale
    @Environment(\.conversationFont) private var conversationFont
    @Environment(\.messageBubbleStyle) private var bubbleStyle
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var message: ChatMessage

    var body: some View {
        if message.role == .user {
          HStack(alignment: .top) {
            Spacer(minLength: 64)
            VStack(alignment: .trailing, spacing: 9) {
                HStack(spacing: 9) {
                    Text(L("Vous", "You"))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(StudioTheme.quiet)
                }
                ForEach(message.attachments) { attachment in
                    ChatAttachmentView(attachment: attachment)
                }
                if !message.content.isEmpty {
                  Text(message.content)
                    .font(.system(size: 15 * textScale, weight: .medium, design: conversationFont.design))
                    .foregroundStyle(StudioTheme.ink)
                    .lineSpacing(5 * textScale)
                    .textSelection(.enabled)
                    .padding(.horizontal, 17)
                    .padding(.vertical, 13)
                    .background(bubbleStyle == .tinted ? studio.accentColor.opacity(0.18) : StudioTheme.contrast.opacity(bubbleStyle == .subtle ? 0.055 : 0), in: RoundedRectangle(cornerRadius: 14, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(StudioTheme.edge, lineWidth: bubbleStyle == .outlined ? 1 : bubbleStyle == .subtle ? 0.5 : 0)
                    }
                }
                HStack { MessageCopyButton(message: message); MessageActionsView(message: message) }
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
    @Environment(\.conversationTextScale) private var textScale
    @ObservedObject var message: ChatMessage
    @State private var citedSource: DocumentExcerpt?

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
                    Spacer(minLength: 8)
                }

                if message.parts.isEmpty && message.isStreaming {
                    ThinkingPlaceholder()
                }
                ForEach(message.parts) { part in
                    let live = message.isStreaming && message.parts.last?.id == part.id
                    switch part.kind {
                    case .thinking:
                        ThinkingBlock(text: part.text, streaming: live, textScale: textScale)
                            .equatable()
                    case .answer:
                        MarkdownResponseView(part.text, streaming: live,
                                             textScale: textScale, htmlPreviewEnabled: !message.isStreaming && message.error == nil)
                            .equatable()
                            .environment(\.openURL, OpenURLAction { url in
                                if url.scheme == "mlxl3-source" {
                                    citedSource = message.sources.first { $0.id.lowercased() == url.host?.lowercased() }
                                    return .handled
                                }
                                return ["http", "https"].contains(url.scheme ?? "") ? .systemAction : .discarded
                            })
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

                if !message.sources.isEmpty {
                    ScrollView(.horizontal) {
                        HStack {
                            ForEach(message.sources) { source in
                                Button(source.label) { citedSource = source }.buttonStyle(.bordered).font(.caption)
                            }
                        }
                    }
                }
                if let stats = message.stats {
                    StatsRow(stats: stats)
                }
                if !message.content.isEmpty {
                    HStack { MessageCopyButton(message: message); MessageActionsView(message: message) }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .sheet(item: $citedSource) { DocumentSourceView(source: $0) }
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
    @Environment(\.conversationTextScale) private var textScale
    let activity: ToolActivity
    @State private var expanded = false
    @State private var fullOutput = false

    @State private var fullArguments = false

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            // What ran (the program of a JavaScript call), then what came back.
            if let arguments = activity.arguments, !arguments.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    Text(activity.toolName == "javascript" ? L("Code exécuté", "Code run") : L("Arguments", "Arguments"))
                        .font(.system(size: 8, weight: .bold, design: .rounded))
                        .tracking(0.65)
                        .foregroundStyle(StudioTheme.quiet)
                    if fullArguments {
                        ScrollView([.vertical, .horizontal]) {
                            Text(arguments).textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }.frame(maxHeight: 240)
                    } else {
                        Text(arguments).lineLimit(12).textSelection(.enabled)
                    }
                    HStack {
                        if arguments.split(separator: "\n", omittingEmptySubsequences: false).count > 12 {
                            Button(fullArguments ? L("Réduire", "Show less") : L("Voir tout le code", "Show all code")) {
                                fullArguments.toggle()
                            }
                        }
                        Button(L("Copier le code", "Copy code")) {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(arguments, forType: .string)
                        }
                    }.buttonStyle(.borderless)
                }
                .font(.system(size: 10.5 * textScale, weight: .regular, design: .monospaced))
                .foregroundStyle(StudioTheme.contrast.opacity(0.58))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 8)
                .accessibilityIdentifier("tool.arguments")
            }
            if let result = activity.result, !result.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    if fullOutput {
                        ScrollView([.vertical, .horizontal]) {
                            Text(result).textSelection(.enabled)
                                .fixedSize(horizontal: false, vertical: true)
                        }.frame(maxHeight: 240)
                    } else {
                        Text(result).lineLimit(10).textSelection(.enabled)
                    }
                    HStack {
                        Button(fullOutput ? L("Réduire", "Show less") : L("Voir la sortie complète", "Show full output")) {
                            fullOutput.toggle()
                        }
                        Button(L("Copier", "Copy")) {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(result, forType: .string)
                        }
                    }.buttonStyle(.borderless)
                }
                .font(.system(size: 10.5 * textScale, weight: .regular, design: .monospaced))
                .foregroundStyle(StudioTheme.contrast.opacity(0.58))
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 8)
            }
        } label: {
            HStack(spacing: 8) {
                stateIcon
                VStack(alignment: .leading, spacing: 1) {
                    Text(activity.toolName == "invalid_tool_call" ? L("Appel d’outil illisible", "Unreadable tool call") : activity.toolName)
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
        .tint(StudioTheme.contrast.opacity(0.42))
        .padding(.horizontal, 12)
        .padding(.vertical, 9)
        .background(StudioTheme.contrast.opacity(0.025), in: RoundedRectangle(cornerRadius: 11))
        .overlay {
            RoundedRectangle(cornerRadius: 11)
                .stroke(StudioTheme.contrast.opacity(0.07), lineWidth: 0.6)
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
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var pulse = false

    var body: some View {
        Circle()
            .fill(StudioTheme.accent)
            .frame(width: 5, height: 5)
            // Scope the endless animation to opacity/scale. A view-wide
            // `.animation(_:value:)` started from `onAppear` also captured the
            // insertion layout, so the dot's position looped forever between
            // its first frame and its real slot (a "bouncing ball").
            .animation(reduceMotion ? nil : .easeInOut(duration: 0.72).repeatForever()) { dot in
                dot
                    .opacity(!reduceMotion && pulse ? 0.3 : 1)
                    .scaleEffect(!reduceMotion && pulse ? 0.72 : 1)
            }
            .onAppear { pulse = !reduceMotion }
            .onChange(of: reduceMotion) { _, value in pulse = !value }
    }
}

private struct ThinkingPlaceholder: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
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
                        // Only opacity repeats, never layout (see StreamingIndicator).
                        .animation(dotAnimation) { dot in dot.opacity(dotOpacity(index)) }
                }
            }
        }
        .foregroundStyle(StudioTheme.thinking)
        .onAppear { shimmer = !reduceMotion }
        .onChange(of: reduceMotion) { _, value in shimmer = !value }
    }

    // Typed helpers keep this view within the hosted compiler's type-check budget.
    private var dotAnimation: Animation? {
        reduceMotion ? nil : .easeInOut(duration: 0.8).repeatForever()
    }

    private func dotOpacity(_ index: Int) -> Double {
        let position = Double(index)
        return !reduceMotion && shimmer ? 0.25 + position * 0.28 : 0.9 - position * 0.25
    }
}

private struct ThinkingBlock: View, Equatable {
    let text: String
    let streaming: Bool
    let textScale: CGFloat
    @State private var expanded = true
    @State private var chunkCache = StreamingTextChunker.Cache(markdown: false)
    @State private var scroll = StreamingScrollState()

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.text == rhs.text && lhs.streaming == rhs.streaming && lhs.textScale == rhs.textScale
    }

    var body: some View {
        DisclosureGroup(isExpanded: $expanded) {
            ScrollView {
                let chunks = chunkCache.chunks(text)
                VStack(alignment: .leading, spacing: 0) {
                    ForEach(chunks) { chunk in
                        ThinkingTextChunk(
                            source: chunk.source,
                            streaming: streaming && chunk.id == chunks.last?.id,
                            textScale: textScale
                        )
                        .equatable()
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 10)
            }
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .defaultScrollAnchor(scroll.shouldFollow ? .bottom : nil, for: .sizeChanges)
            .onScrollPhaseChange { _, phase in scroll.phaseChanged(phase) }
            .onScrollGeometryChange(for: Bool.self) { geometry in
                geometry.contentSize.height - geometry.visibleRect.maxY < 60
            } action: { _, nearBottom in
                scroll.proximityChanged(nearBottom)
            }
            .frame(maxHeight: 180)
        } label: {
            HStack(spacing: 8) {
                Text(streaming ? L("Réflexion en cours", "Thinking") : L("Réflexion", "Thinking"))
                    .font(.system(size: 11, weight: .medium))
                if streaming { StreamingIndicator() }
            }
            .foregroundStyle(StudioTheme.thinking)
        }
        .tint(StudioTheme.contrast.opacity(0.52))
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
    let textScale: CGFloat

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.streaming == rhs.streaming && lhs.textScale == rhs.textScale
    }

    var body: some View {
        SmoothTokenText(content: Text(source), source: source, streaming: streaming)
            .font(.system(size: 11.5 * textScale, weight: .regular, design: .monospaced))
            .foregroundStyle(StudioTheme.secondary)
            .lineSpacing(3 * textScale)
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
            MetricChip(icon: "timer", value: String(format: "%.0f ms", stats.ttftSeconds * 1000), label: L("TTFT moteur", "engine TTFT"))
            if let elapsed = stats.elapsedSeconds {
                MetricChip(icon: "clock", value: String(format: "%.1f s", elapsed), label: L("total · outils inclus", "total · incl. tools"))
            }
            if let memory = stats.memory {
                MetricChip(icon: "memorychip", value: (memory.nativeMetalAllocatedBytes ?? memory.mlxPeakBytes).map { String(format: "%.2f GB", Double($0) / 1e9) } ?? "—", label: memory.nativeMetalAllocatedBytes == nil ? L("pic MLX", "MLX peak") : L("Metal alloué", "Metal allocated"))
                if let footprint = memory.processFootprintBytes {
                    MetricChip(icon: "memorychip", value: String(format: "%.2f GB", Double(footprint) / 1e9), label: L("RAM moteur", "engine RAM"))
                }
            } else if let size = stats.modelSizeGB {
                MetricChip(icon: "externaldrive", value: String(format: "%.2f GB", size), label: L("poids disque", "disk weights"))
            } else if let legacy = stats.peakMemoryGB {
                MetricChip(icon: "memorychip", value: String(format: "%.2f GB", legacy), label: L("ancienne mesure", "legacy metric"))
            }
            if let accepted = stats.dflashAcceptedTokens, let proposed = stats.dflashProposedTokens, proposed > 0 {
                MetricChip(icon: "checkmark.circle", value: "\(accepted)/\(proposed)", label: String(format: "DFlash %.0f%%", 100 * Double(accepted) / Double(proposed)))
            }
            if let accepted = stats.mtpAcceptedTokens, let proposed = stats.mtpProposedTokens, proposed > 0 {
                MetricChip(icon: "checkmark.circle", value: "\(accepted)/\(proposed)", label: String(format: "MTP %.0f%%", 100 * Double(accepted) / Double(proposed)))
            }
            if let dspark = stats.dsparkAcceptance {
                MetricChip(icon: "checkmark.circle", value: "\(dspark.accepted)/\(dspark.proposed)",
                           label: String(format: L("DSpark %.0f%% · %.2f tok/bloc", "DSpark %.0f%% · %.2f tok/block"),
                                         100 * Double(dspark.accepted) / Double(dspark.proposed), dspark.tokensPerBlock))
            }
            if let blocks = stats.dflashBlocks, blocks > 0, let seconds = stats.dflashBlockSeconds {
                MetricChip(icon: "square.stack", value: String(format: "%.1f ms", seconds * 1000 / Double(blocks)), label: L("par bloc", "per block"))
            }
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
                .foregroundStyle(StudioTheme.contrast.opacity(0.72))
            Text(label)
                .foregroundStyle(StudioTheme.quiet)
        }
        .font(.system(size: 9, weight: .medium, design: .rounded))
        .padding(.horizontal, 8)
        .frame(height: 24)
        .monospacedDigit()
    }
}

struct ChatAttachmentView: View {
    let attachment: ChatAttachment
    var onRemove: (() -> Void)? = nil
    @State private var showPreview = false

    var body: some View {
        HStack(spacing: 8) {
            Button { showPreview = true } label: {
                if let data = attachment.image?.data, let image = NSImage(data: data),
                   image.size.width > 0, image.size.height > 0 {
                    let maxWidth: CGFloat = onRemove == nil ? 240 : 96
                    let maxHeight: CGFloat = onRemove == nil ? 180 : 72
                    let scale = min(maxWidth / image.size.width, maxHeight / image.size.height)
                    Image(nsImage: image).renderingMode(.original).resizable().scaledToFit()
                        .frame(width: image.size.width * scale, height: image.size.height * scale)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    Label(attachment.fileName, systemImage: attachment.fileName.lowercased().hasSuffix(".pdf") ? "doc.richtext" : "doc.text")
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
            }
            .buttonStyle(.plain)
            .accessibilityLabel(attachment.fileName)
            .help(attachment.image != nil ? L("Afficher l’image", "Preview image") : L("Afficher le texte extrait", "Preview extracted text"))
            .popover(isPresented: $showPreview) {
                VStack(alignment: .leading, spacing: 12) {
                    Text(attachment.fileName).font(.headline)
                    if let data = attachment.image?.data, let image = NSImage(data: data) {
                        Image(nsImage: image).renderingMode(.original).resizable().scaledToFit()
                            .accessibilityLabel(attachment.fileName)
                    } else {
                        ScrollView {
                            Text(attachment.text)
                                .font(.system(size: 12, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .padding(18)
                .frame(width: 500, height: 360)
            }
            if attachment.image == nil, let onRemove {
                Button(action: onRemove) {
                    Image(systemName: "xmark").font(.system(size: 9, weight: .semibold))
                }
                .buttonStyle(.plain)
                .accessibilityLabel(L("Retirer \(attachment.fileName)", "Remove \(attachment.fileName)"))
                .help(L("Retirer le fichier", "Remove file"))
            }
        }
        .font(.system(size: 11))
        .foregroundStyle(StudioTheme.secondary)
        .padding(.horizontal, attachment.image == nil ? 11 : 0)
        .padding(.vertical, attachment.image == nil ? 8 : 0)
        .background(StudioTheme.contrast.opacity(attachment.image == nil ? 0.06 : 0), in: RoundedRectangle(cornerRadius: 8))
        .overlay(alignment: .topTrailing) {
            if attachment.image != nil, let onRemove {
                Button(action: onRemove) {
                    Image(systemName: "xmark").font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(.white).frame(width: 22, height: 22)
                        .background(.black.opacity(0.65), in: Circle())
                }
                .buttonStyle(.plain).padding(4)
                .accessibilityLabel(L("Retirer \(attachment.fileName)", "Remove \(attachment.fileName)"))
                .help(L("Retirer le fichier", "Remove file"))
            }
        }
        .frame(maxWidth: 300)
    }
}

struct ComposerView: View {
    @Environment(\.conversationReadingWidth) private var readingWidth
    @EnvironmentObject private var studio: StudioModel
    @FocusState private var focused: Bool
    @State private var dropTargeted = false

    var body: some View {
        VStack(spacing: 0) {
            if !studio.pendingAttachments.isEmpty {
                ScrollView(.horizontal) {
                    HStack(spacing: 8) {
                        ForEach(studio.pendingAttachments) { attachment in
                            ChatAttachmentView(attachment: attachment) {
                                studio.removeChatAttachment(attachment.id)
                            }
                            .disabled(studio.composerBusy || studio.isImportingFiles)
                        }
                    }
                }
                .scrollIndicators(.never)
                .padding(.horizontal, 18)
                .padding(.top, 14)
            }
            if studio.isImportingFiles {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.mini)
                    Text(L("Lecture des fichiers…", "Reading files…"))
                    Spacer()
                }
                .font(.system(size: 11))
                .foregroundStyle(StudioTheme.secondary)
                .padding(.horizontal, 18)
                .padding(.top, 12)
            }
            if let error = studio.attachmentImportError {
                HStack(alignment: .top, spacing: 8) {
                    Label(error, systemImage: "exclamationmark.triangle")
                        .textSelection(.enabled)
                    Spacer(minLength: 0)
                    Button(action: studio.dismissAttachmentImportError) { Image(systemName: "xmark") }
                        .buttonStyle(.plain)
                        .accessibilityLabel(L("Fermer l’erreur d’import", "Dismiss import error"))
                }
                .font(.system(size: 11))
                .foregroundStyle(.orange)
                .padding(.horizontal, 18)
                .padding(.top, 12)
            }
            HStack(alignment: .bottom, spacing: 12) {
                TextField(studio.selectedMode == .agent ? L("Décrivez votre tâche…", "Describe your task…") : L("Écrivez un message…", "Write a message…"), text: $studio.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 15))
                    .lineLimit(1...6)
                    .focused($focused)
                    .padding(.vertical, 5)
                    .frame(minHeight: 38, alignment: .topLeading)
                    .contentShape(Rectangle())
                    .background(ComposerPasteMonitor(focused: focused, studio: studio))
                    .onPasteCommand(of: [.image]) { _ in
                        guard focused, let image = ComposerPasteboard.image(from: .general) else { return }
                        ComposerPasteboard.importImage(image, into: studio)
                    }
                    .disabled(studio.composerBusy)
                    .onSubmit { studio.send() }
                    .onKeyPress(.return, phases: .down) { press in
                        if press.modifiers.contains(.control) {
                            if let editor = NSApp.keyWindow?.firstResponder as? NSTextView {
                                editor.insertText("\n", replacementRange: editor.selectedRange())
                            }
                        } else {
                            studio.send()
                        }
                        return .handled
                    }

            }
            .padding(.horizontal, 18)
            .padding(.top, 15)
            .padding(.bottom, 3)

            if let reason = studio.sendBlockReason {
                Text(reason).font(.caption).foregroundStyle(StudioTheme.secondary)
                    .padding(.horizontal, 18).padding(.bottom, 6)
                    .accessibilityLabel(reason)
            }
            ComposerFooterView()
        }
        .contentShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .onTapGesture {
            guard !studio.composerBusy else { return }
            focused = true
        }
        .background {
            // Two shadows: a tight contact shadow and a wide ambient lift.
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(StudioTheme.composer)
                .shadow(color: .black.opacity(0.08), radius: 2, y: 1)
                .shadow(color: .black.opacity(focused ? 0.16 : 0.11), radius: focused ? 22 : 16, y: focused ? 10 : 7)
        }
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(
                    dropTargeted ? AnyShapeStyle(studio.accentColor)
                        : focused ? AnyShapeStyle(studio.accentColor.opacity(0.45))
                        : AnyShapeStyle(StudioTheme.contrast.opacity(0.12)),
                    lineWidth: dropTargeted ? 2 : 1)
                .allowsHitTesting(false)
        }
        .animation(.easeOut(duration: 0.18), value: focused)
        .dropDestination(for: URL.self) { urls, _ in
            guard studio.canImportFiles, !urls.isEmpty else { return false }
            Task { await studio.importChatFiles(urls) }
            return true
        } isTargeted: { dropTargeted = $0 }
        .frame(maxWidth: readingWidth.composerMaximum)
        .frame(maxWidth: .infinity)
        .onAppear { focused = true }
        .onChange(of: studio.draft) {
            if !studio.composerBusy && !studio.draft.isEmpty { focused = true }
        }
        .onChange(of: studio.composerBusy) {
            if !studio.composerBusy {
                focused = true
            }
        }
    }
}

enum ComposerPasteboard {
    enum Image {
        case files([URL])
        case data(Data)
    }

    static func image(from board: NSPasteboard) -> Image? {
        let urls = (board.readObjects(forClasses: [NSURL.self], options: [.urlReadingFileURLsOnly: true]) as? [URL] ?? [])
            .filter { UTType(filenameExtension: $0.pathExtension)?.conforms(to: .image) == true }
        if !urls.isEmpty { return .files(urls) }
        for type in [NSPasteboard.PasteboardType.png, .init(UTType.jpeg.identifier), .init(UTType.webP.identifier), .tiff] {
            if let data = board.data(forType: type) { return .data(data) }
        }
        return nil
    }

    @MainActor static func importImage(_ image: Image, into studio: StudioModel) {
        let conversationID = studio.selectedConversationID
        Task {
            guard studio.selectedConversationID == conversationID else { return }
            switch image {
            case .files(let urls): await studio.importChatFiles(urls)
            case .data(let data): await studio.importPastedImage(data)
            }
        }
    }
}

private struct ComposerPasteMonitor: NSViewRepresentable {
    let focused: Bool
    let studio: StudioModel

    func makeNSView(context: Context) -> PasteMonitorView { PasteMonitorView() }
    func updateNSView(_ view: PasteMonitorView, context: Context) {
        view.focused = focused
        view.studio = studio
    }

    final class PasteMonitorView: NSView {
        var focused = false
        weak var studio: StudioModel?
        private var monitor: Any?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            if let monitor { NSEvent.removeMonitor(monitor); self.monitor = nil }
            guard window != nil else { return }
            monitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
                let handled = MainActor.assumeIsolated {
                    guard let self, self.focused, self.window?.isKeyWindow == true,
                          self.window?.firstResponder is NSTextView,
                          event.charactersIgnoringModifiers?.lowercased() == "v",
                          event.modifierFlags.intersection([.command, .shift, .option, .control]) == .command,
                          let studio = self.studio, studio.canImportFiles else { return false }
                    guard let image = ComposerPasteboard.image(from: .general) else { return false }
                    ComposerPasteboard.importImage(image, into: studio)
                    return true
                }
                return handled ? nil : event
            }
        }

        isolated deinit { if let monitor { NSEvent.removeMonitor(monitor) } }
    }
}


/// Reasoning depth of models whose chat template accepts `reasoning_effort` (Inkling 2,
/// Qwen3.8…): only the levels that template names, kept per model.
private struct ReasoningMenu: View {
    @EnvironmentObject private var studio: StudioModel
    let support: ReasoningSupport
    let level: ReasoningEffort

    var body: some View {
        Menu {
            Picker(L("Raisonnement", "Reasoning"), selection: Binding(
                get: { level }, set: { studio.setReasoningLevel($0) })) {
                ForEach(support.levels) { option in
                    Label {
                        Text(option.title)
                        Text(option == support.defaultLevel
                             ? L("\(option.detail) · par défaut", "\(option.detail) · default")
                             : option.detail)
                    } icon: {
                        Image(systemName: option.symbol)
                    }
                    .tag(option)
                }
            }
            .pickerStyle(.inline)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: level.symbol)
                    .foregroundStyle(level == .none ? StudioTheme.quiet : studio.accentColor)
                Text(level.label)
            }
            .font(.system(size: 12, weight: .medium))
            .padding(.horizontal, 9)
            .frame(height: 28)
            .background(StudioTheme.panel, in: Capsule())
            .contentTransition(.symbolEffect(.replace))
        }
        .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
        .fixedSize()
        .disabled(studio.isGenerating)
        .accessibilityIdentifier("composer.reasoning")
        .accessibilityLabel(L("Raisonnement", "Reasoning"))
        .accessibilityValue(level.label)
        .help(L("Profondeur de raisonnement du modèle", "Model reasoning depth"))
    }
}

private struct ComposerFooterView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
            HStack(spacing: 7) {
                Menu {
                    Button(action: studio.chooseChatFiles) {
                        Label(L("Joindre des fichiers…", "Attach files…"), systemImage: "paperclip")
                    }
                    .disabled(!studio.canImportFiles)
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                    if studio.selectedMode != .agent {
                        Divider()
                        Toggle(L("Activer les outils MCP", "Enable MCP tools"), isOn: Binding(
                            get: { studio.mcpEnabled }, set: { studio.setMCPEnabled($0) }))
                        .disabled(studio.isGenerating || studio.mcpUpdating)
                        .help(L("Les serveurs configurés peuvent envoyer des données hors de ce Mac.", "Configured servers may send data outside this Mac."))
                        Button(L("Configurer les outils…", "Configure tools…")) {
                            studio.showMCPManager = true
                        }.disabled(studio.isGenerating || studio.mcpUpdating)
                    }
                    Divider()
                    Button(action: studio.exportMarkdown) {
                        Label(L("Exporter la conversation…", "Export conversation…"), systemImage: "square.and.arrow.up")
                    }
                } label: {
                    Label(L("Outils", "Tools"), systemImage: "plus")
                        .font(.system(size: 12, weight: .medium))
                        .padding(.horizontal, 9)
                        .frame(height: 28)
                        .background(StudioTheme.panel, in: Capsule())
                }
                .menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden)
                .fixedSize()
                .accessibilityLabel(L("Fichiers et outils", "Files and tools"))
                .accessibilityValue(studio.mcpEnabled ? L("Outils activés", "Tools enabled") : L("Outils désactivés", "Tools disabled"))
                .help(L("Fichiers et outils (⌘⇧O pour joindre un fichier)", "Files and tools (⌘⇧O to attach a file)"))
                if studio.reasoningControlAvailable, let support = studio.reasoningSupport,
                   let level = studio.reasoningLevel {
                    ReasoningMenu(support: support, level: level)
                }
                if studio.mcpUpdating {
                    ProgressView().controlSize(.mini).help(L("Connexion MCP…", "Connecting MCP…"))
                } else if studio.mcpEnabled && !studio.mcpErrors.isEmpty {
                    Image(systemName: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                        .help(studio.mcpErrors.values.sorted().joined(separator: "\n"))
                }
                Spacer()
                ComposerVoiceControl(studio: studio, voice: studio.voice)
                if studio.composerBusy || studio.preparingDocuments {
                    Button(action: studio.stopGeneration) {
                        Image(systemName: "stop.fill")
                            .font(.system(size: 10, weight: .bold))
                            .foregroundStyle(StudioTheme.onAccent)
                            .frame(width: 31, height: 31)
                    }
                    .buttonStyle(RoundGlassButtonStyle(bright: true))
                    .help(L("Arrêter", "Stop"))
                    .accessibilityLabel(L("Arrêter la génération", "Stop generation"))
                } else {
                    Button(action: studio.send) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 12, weight: .bold))
                            .foregroundStyle(StudioTheme.onAccent)
                            .frame(width: 31, height: 31)
                    }
                    .buttonStyle(RoundGlassButtonStyle(bright: true))
                    .disabled(!studio.canAttemptSend)
                    .help(L("Envoyer (Retour). Nouvelle ligne : Ctrl+Retour.", "Send (Return). New line: Control+Return."))
                    .accessibilityLabel(L("Envoyer le message", "Send message"))
                }
            }
            .font(.system(size: 13, weight: .regular))
            .foregroundStyle(StudioTheme.quiet)
            .padding(.horizontal, 18)
            .padding(.bottom, 10)
    }
}
