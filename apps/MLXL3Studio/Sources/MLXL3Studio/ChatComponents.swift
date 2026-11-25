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
  var agentWorkspace: URL? = nil
  var savingAgentTurn = false
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
              ForEach(messages.enumerated().prefix(split), id: \.element.id) { index, message in
                row(message, at: index, isLast: false)
                  .id(message.id)
              }
            }
          }
          ForEach(messages.enumerated().dropFirst(split), id: \.element.id) { index, message in
            row(message, at: index, isLast: index == messages.count - 1)
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

  private func row(_ message: ChatMessage, at index: Int, isLast: Bool) -> some View {
    // Read the boundary in O(1) even for long or imported transcripts.
    let start = agentWorkspace != nil && message.role == .assistant && index > 0
      && messages[index - 1].role == .user ? messages[index - 1].id : nil
    return MessageView(message: message, isLast: isLast, agentWorkspace: agentWorkspace,
      startID: start, savingAgentTurn: isLast && savingAgentTurn)
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
    @Environment(\.studioStyle) private var style
    @Environment(\.conversationTextScale) private var textScale
    @Environment(\.conversationFont) private var conversationFont
    @Environment(\.messageBubbleStyle) private var bubbleStyle
    @EnvironmentObject private var studio: StudioModel
    @ObservedObject var message: ChatMessage
    var isLast = false
    var agentWorkspace: URL? = nil
    var startID: UUID? = nil
    var savingAgentTurn = false
    @State private var hovered = false

    var body: some View {
        if message.role == .user {
          HStack(alignment: .top) {
            Spacer(minLength: 64)
            VStack(alignment: .trailing, spacing: style.isGraphite ? 8 : 9) {
                // Graphite: the bubble already says who wrote it.
                if !style.isGraphite {
                    HStack(spacing: 9) {
                        Text(L("Vous", "You"))
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(StudioTheme.quiet)
                    }
                }
                ForEach(message.attachments) { attachment in
                    ChatAttachmentView(attachment: attachment)
                }
                if !message.content.isEmpty && style.isGraphite {
                  Text(message.content)
                    .font(.system(size: 15 * textScale, weight: .regular, design: conversationFont.design))
                    .foregroundStyle(StudioTheme.ink)
                    .lineSpacing(5 * textScale)
                    .textSelection(.enabled)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 11)
                    .background(bubbleStyle == .tinted ? studio.accentColor.opacity(0.16) : StudioTheme.contrast.opacity(bubbleStyle == .subtle ? 0.065 : 0), in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .strokeBorder(StudioTheme.edge, lineWidth: bubbleStyle == .outlined ? 1 : 0)
                    }
                } else if !message.content.isEmpty {
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
            }
            // Beside the bubble on hover, outside the layout: a row of identical buttons
            // under every question was noise, and reserving its height left a gap.
            .overlay(alignment: .bottomLeading) {
                MessageActionBar(message: message)
                    .alignmentGuide(.leading) { $0[.trailing] + 8 }
                    .opacity(hovered ? 1 : 0)
                    .animation(.easeOut(duration: 0.12), value: hovered)
            }
          }
            .frame(maxWidth: .infinity, alignment: .trailing)
            .padding(.bottom, 4)
            .contentShape(Rectangle())
            .onHover { hovered = $0 }
        } else {
            AssistantMessageView(message: message, isLast: isLast, agentWorkspace: agentWorkspace, startID: startID, savingAgentTurn: savingAgentTurn)
        }
    }
}

private struct AssistantMessageView: View {
    @EnvironmentObject private var studio: StudioModel
    @Environment(\.studioStyle) private var style
    @Environment(\.conversationTextScale) private var textScale
    @ObservedObject var message: ChatMessage
    var isLast = false
    var agentWorkspace: URL? = nil
    var startID: UUID? = nil
    var savingAgentTurn = false
    @State private var citedSource: DocumentExcerpt?
    @State private var turnOutcome: Result<AgentTurnResult?, Error>?
    @State private var hovered = false

    var body: some View {
        HStack(alignment: .top, spacing: 0) {
            VStack(alignment: .leading, spacing: 13) {
                // Graphite: the answer speaks for itself, no mark and name above each reply.
                if !style.isGraphite {
                    HStack(spacing: 7) {
                        MonogramMark(size: 15)
                        Text("MLXL3")
                            .font(.system(size: 11, weight: .medium))
                            .foregroundStyle(StudioTheme.secondary)
                        Spacer(minLength: 8)
                    }
                }

                if message.parts.isEmpty && message.isStreaming {
                    PendingActivityRow(label: L("Préparation de la réponse", "Preparing response"))
                }
                ForEach(AssistantSegment.split(message.parts)) { segment in
                    if segment.isActivity {
                        // Reasoning and tool lines read as one log, tighter than paragraphs.
                        VStack(alignment: .leading, spacing: 3) {
                            ForEach(segment.parts) { part in activity(part) }
                            // A finished tool before the next round's first event: the run goes on.
                            if segment.parts.last?.id == message.parts.last?.id, awaitingNextStep {
                                PendingActivityRow(label: L("Préparation de l’étape suivante", "Preparing the next step"))
                            }
                        }
                    } else if let part = segment.parts.first {
                        MarkdownResponseView(part.text, streaming: isLive(part),
                                             textScale: textScale, htmlPreviewEnabled: !message.isStreaming && message.error == nil)
                            .equatable()
                            .environment(\.openURL, OpenURLAction { url in
                                if url.scheme == "mlxl3-source" {
                                    citedSource = message.sources.first { $0.id.lowercased() == url.host?.lowercased() }
                                    return .handled
                                }
                                return ["http", "https"].contains(url.scheme ?? "") ? .systemAction : .discarded
                            })
                    }
                }

                if let error = message.error {
                    Label(error, systemImage: "exclamationmark.circle")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(StudioTheme.failure)
                }

                if !message.sources.isEmpty {
                    ScrollView(.horizontal) {
                        HStack(spacing: 6) {
                            ForEach(message.sources) { source in
                                Button { citedSource = source } label: {
                                    Label(source.label, systemImage: "doc.text")
                                        .font(.system(size: 11.5, weight: .medium))
                                        .lineLimit(1)
                                        .padding(.horizontal, 10)
                                        .frame(height: 26)
                                        .overlay {
                                            Capsule().strokeBorder(StudioTheme.edge, lineWidth: 0.5)
                                        }
                                }
                                .buttonStyle(StudioControlStyle(cornerRadius: 13))
                            }
                        }
                    }
                    .scrollIndicators(.never)
                }
                if let agentWorkspace, startID != nil, !message.isStreaming, !savingAgentTurn {
                    switch turnOutcome {
                    case .success(let result?) where !result.files.isEmpty || !result.commands.isEmpty:
                        AgentTurnResultRow(workspace: agentWorkspace, result: result)
                    case .failure(let error):
                        Label(L("Résultat du tour indisponible", "Turn result unavailable"), systemImage: "exclamationmark.circle")
                            .font(.caption).foregroundStyle(StudioTheme.secondary)
                            .help(error.localizedDescription).accessibilityHint(error.localizedDescription)
                    default:
                        EmptyView()
                    }
                }
                if let stats = message.stats {
                    StatsRow(stats: stats)
                }
                if !message.content.isEmpty {
                    // The latest reply keeps its actions; older ones show them on hover.
                    MessageActionBar(message: message)
                        .opacity(isLast || hovered ? 1 : 0)
                        .animation(.easeOut(duration: 0.12), value: hovered)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .contentShape(Rectangle())
        .onHover { hovered = $0 }
        .sheet(item: $citedSource) { DocumentSourceView(source: $0) }
        .background {
            if let agentWorkspace, let startID, !message.isStreaming, !savingAgentTurn {
                // Loading cannot add an empty row to the transcript or move a plain answer.
                Color.clear.allowsHitTesting(false).accessibilityHidden(true)
                  .task(id: agentWorkspace.appendingPathComponent(startID.uuidString)) {
                    let store = studio.agentStore(for: agentWorkspace)
                    let assistantID = message.id
                    let outcome = await Task.detached(priority: .utility) {
                        Result { try store.turnResult(user: startID, assistant: assistantID) }
                    }.value
                    if !Task.isCancelled { turnOutcome = outcome }
                }
            }
        }
    }

    private func isLive(_ part: AssistantPart) -> Bool {
        message.isStreaming && message.parts.last?.id == part.id
    }

    private var awaitingNextStep: Bool {
        guard message.isStreaming, let last = message.parts.last, last.kind == .tool else { return false }
        return message.toolActivities.first(where: { $0.id == last.toolID })?.state != .running
    }

    @ViewBuilder
    private func activity(_ part: AssistantPart) -> some View {
        switch part.kind {
        case .thinking:
            ThinkingBlock(text: part.text, streaming: isLive(part), elapsed: part.elapsed, textScale: textScale)
                .equatable()
        case .tool:
            if let activity = message.toolActivities.first(where: { $0.id == part.toolID }) {
                ToolCallRow(activity: activity, elapsed: part.elapsed)
            }
        case .processing:
            PrefillActivityRow(part: part, live: isLive(part))
        case .answer:
            EmptyView()
        }
    }
}

/// Consecutive reasoning, tool and context parts form one activity segment; each answer
/// part stands alone. A segment is identified by its first part, so it keeps its identity
/// (and the rows inside keep their expanded state) while it grows.
struct AssistantSegment: Identifiable {
    let parts: [AssistantPart]
    var id: UUID { parts[0].id }
    var isActivity: Bool { parts[0].kind != .answer }

    static func split(_ parts: [AssistantPart]) -> [AssistantSegment] {
        var segments: [AssistantSegment] = []
        var run: [AssistantPart] = []
        for part in parts {
            if part.kind == .answer {
                if !run.isEmpty { segments.append(AssistantSegment(parts: run)); run = [] }
                segments.append(AssistantSegment(parts: [part]))
            } else {
                run.append(part)
            }
        }
        if !run.isEmpty { segments.append(AssistantSegment(parts: run)) }
        return segments
    }
}

/// Copy, regenerate, read aloud and branch: one row of quiet icon buttons.
struct MessageActionBar: View {
    @ObservedObject var message: ChatMessage

    var body: some View {
        HStack(spacing: 2) {
            MessageCopyButton(message: message)
            MessageActionsView(message: message)
        }
        // The first icon's glyph lines up with the text above it.
        .padding(.leading, message.role == .assistant ? -6 : 0)
    }
}

// MARK: - Activity column

/// Reasoning, tool calls and context reading share one column: an icon, what happened and
/// its object, then quiet metadata. Nothing is boxed; detail opens under the line.
private enum ActivityMetrics {
    static let iconWidth: CGFloat = 16
    static let spacing: CGFloat = 8
    static var indent: CGFloat { iconWidth + spacing }

    /// "4 s", "1 min 12 s".
    static func duration(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        return total < 60 ? "\(total) s" : "\(total / 60) min \(total % 60) s"
    }
}

private struct ActivityIcon: View {
    @Environment(\.conversationTextScale) private var textScale
    let symbol: String
    var color: Color = StudioTheme.quiet

    var body: some View {
        Image(systemName: symbol)
            .font(.system(size: 11 * textScale, weight: .medium))
            .foregroundStyle(color)
            .frame(width: ActivityMetrics.iconWidth)
            .accessibilityHidden(true)
    }
}

private struct ActivityChevron: View {
    let expanded: Bool
    let visible: Bool

    var body: some View {
        Image(systemName: "chevron.right")
            .font(.system(size: 8.5, weight: .bold))
            .foregroundStyle(StudioTheme.quiet)
            .rotationEffect(.degrees(expanded ? 90 : 0))
            .opacity(visible || expanded ? 1 : 0)
            .accessibilityHidden(true)
    }
}

/// The clickable line of an expandable activity: a hover wash that hugs the text.
private struct ActivityLineStyle: ButtonStyle {
    let hovered: Bool

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .background(
                StudioTheme.contrast.opacity(configuration.isPressed ? 0.075 : hovered ? 0.045 : 0),
                in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
            .padding(.horizontal, -6)
    }
}

/// Waiting with nothing to show yet: the words say what for, the sweep that it is ongoing.
private struct PendingActivityRow: View {
    @Environment(\.conversationTextScale) private var textScale
    let label: String

    var body: some View {
        HStack(alignment: .firstTextBaseline, spacing: ActivityMetrics.spacing) {
            ActivityIcon(symbol: "circle.dotted")
            StudioShimmerText(text: label, font: .system(size: 13 * textScale, weight: .medium))
        }
        .padding(.vertical, 3)
        .accessibilityElement(children: .combine)
    }
}

/// Context reading (prefill). Live: what is read and for how long. Done: kept only when it
/// took a noticeable time or was interrupted; "Context processed · 0 s" said nothing.
private struct PrefillActivityRow: View {
    @Environment(\.conversationTextScale) private var textScale
    let part: AssistantPart
    let live: Bool

    var body: some View {
        if live {
            TimelineView(.periodic(from: .now, by: 1)) { context in
                row(at: context.date)
            }
        } else if part.interrupted == true || (part.elapsed ?? 0) >= 1 {
            row(at: .now)
        }
    }

    private func row(at date: Date) -> some View {
        let elapsed = part.elapsed ?? date.timeIntervalSince(part.startedAt ?? date)
        let interrupted = part.interrupted == true
        return HStack(alignment: .firstTextBaseline, spacing: ActivityMetrics.spacing) {
            ActivityIcon(symbol: interrupted ? "stop" : "doc.text.magnifyingglass")
            StudioShimmerText(
                text: live ? localizedProgress(part.text)
                    : interrupted ? L("Lecture du contexte interrompue", "Context reading interrupted")
                    : L("Contexte lu", "Read the context"),
                font: .system(size: 13 * textScale, weight: .medium), active: live)
                .lineLimit(2)
            Text(ActivityMetrics.duration(elapsed))
                .font(.system(size: 12 * textScale))
                .monospacedDigit()
                .foregroundStyle(StudioTheme.quiet)
        }
        .padding(.vertical, 3)
        .accessibilityElement(children: .combine)
    }
}

/// One tool call as a line: "Read src/stats.pl", "Ran sh test.sh · Exit 1", "Web search
/// “mlx cache” · exa". Running calls sweep their title; the arguments and the result open
/// underneath on click.
private struct ToolCallRow: View {
    @Environment(\.conversationTextScale) private var textScale
    let activity: ToolActivity
    let elapsed: Double?
    @State private var expanded = false
    @State private var hovered = false

    var body: some View {
        let shown = ToolCallPresentation(activity)
        let input = shown.input(of: activity)
        let output = activity.result.flatMap { $0.isEmpty ? nil : $0 }
        let expandable = input != nil || output != nil
        VStack(alignment: .leading, spacing: 6) {
            Button {
                withAnimation(.easeOut(duration: 0.16)) { expanded.toggle() }
            } label: {
                line(shown, expandable: expandable)
            }
            .buttonStyle(ActivityLineStyle(hovered: hovered && expandable))
            .allowsHitTesting(expandable)
            .onHover { hovered = $0 }
            .accessibilityLabel(accessibilityText(shown))
            .accessibilityHint(expandable ? L("Afficher le détail", "Show details") : "")
            if expanded {
                ToolCallDetails(activity: activity, input: input, output: output)
                    .padding(.leading, ActivityMetrics.indent)
                    .transition(.opacity)
            }
        }
    }

    private func line(_ shown: ToolCallPresentation, expandable: Bool) -> some View {
        let failed = activity.state == .failed
        return HStack(alignment: .firstTextBaseline, spacing: ActivityMetrics.spacing) {
            ActivityIcon(symbol: shown.symbol, color: failed ? StudioTheme.failure : StudioTheme.quiet)
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                StudioShimmerText(text: shown.title, font: .system(size: 13 * textScale, weight: .medium),
                                  active: activity.state == .running)
                    .layoutPriority(1)
                if let detail = shown.detail {
                    Text(detail)
                        .font(shown.monospaced ? .system(size: 12 * textScale, design: .monospaced)
                              : .system(size: 13 * textScale))
                        .foregroundStyle(StudioTheme.quiet)
                        .lineLimit(1)
                        .truncationMode(shown.isPath ? .middle : .tail)
                }
                if let source = shown.source {
                    Text("· \(source)")
                        .font(.system(size: 12 * textScale))
                        .foregroundStyle(StudioTheme.quiet)
                        .lineLimit(1)
                        .layoutPriority(1)
                }
                if let outcome = shown.outcome {
                    Text(outcome)
                        .font(.system(size: 12 * textScale, weight: .medium))
                        .foregroundStyle(failed ? StudioTheme.failure : StudioTheme.quiet)
                        .fixedSize()
                } else if activity.state != .running, let elapsed, elapsed >= 1 {
                    Text(ActivityMetrics.duration(elapsed))
                        .font(.system(size: 12 * textScale))
                        .monospacedDigit()
                        .foregroundStyle(StudioTheme.quiet)
                        .fixedSize()
                }
                if expandable {
                    ActivityChevron(expanded: expanded, visible: hovered)
                }
            }
        }
    }

    private func accessibilityText(_ shown: ToolCallPresentation) -> String {
        [shown.title, shown.detail, shown.source, shown.outcome ?? (activity.state == .running ? L("en cours", "running") : nil)]
            .compactMap { $0 }.joined(separator: ", ")
    }
}

/// What a tool was given and what it returned, in one quiet panel.
private struct ToolCallDetails: View {
    @Environment(\.conversationTextScale) private var textScale
    let activity: ToolActivity
    let input: (title: String, text: String)?
    let output: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            if let input {
                ToolCallSection(title: input.title, text: input.text, lines: 14)
                    .accessibilityIdentifier("tool.arguments")
            }
            if input != nil && output != nil {
                Rectangle().fill(StudioTheme.edge).frame(height: 0.5)
            }
            if let output {
                ToolCallSection(title: activity.state == .failed ? L("Erreur", "Error") : L("Résultat", "Output"),
                                text: output, lines: 12)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(StudioTheme.contrast.opacity(0.03), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 9, style: .continuous).strokeBorder(StudioTheme.edge, lineWidth: 0.5)
        }
    }
}

private struct ToolCallSection: View {
    @Environment(\.conversationTextScale) private var textScale
    let title: String
    let text: String
    let lines: Int
    @State private var full = false
    @State private var copied = false

    var body: some View {
        let long = text.split(separator: "\n", omittingEmptySubsequences: false).count > lines || text.count > lines * 120
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 4) {
                Text(title)
                    .font(.system(size: 11 * textScale, weight: .medium))
                    .foregroundStyle(StudioTheme.quiet)
                Spacer(minLength: 8)
                if long {
                    Button(full ? L("Réduire", "Show less") : L("Tout afficher", "Show all")) { full.toggle() }
                        .font(.system(size: 11 * textScale, weight: .medium))
                        .padding(.horizontal, 7).frame(height: 22)
                        .buttonStyle(StudioControlStyle())
                }
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(text, forType: .string)
                    copied = true
                } label: {
                    Image(systemName: copied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10.5, weight: .medium))
                        .frame(width: 22, height: 22)
                }
                .buttonStyle(StudioControlStyle())
                .help(L("Copier", "Copy"))
                .accessibilityLabel(L("Copier", "Copy"))
            }
            Group {
                if full {
                    ScrollView([.vertical, .horizontal]) {
                        Text(text).fixedSize(horizontal: false, vertical: true)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 300)
                } else {
                    Text(text).lineLimit(lines)
                }
            }
            .font(.system(size: 11.5 * textScale, design: .monospaced))
            .foregroundStyle(StudioTheme.secondary)
            .lineSpacing(2)
            .textSelection(.enabled)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.horizontal, 12)
        .padding(.top, 8)
        .padding(.bottom, 11)
        .task(id: copied) {
            guard copied else { return }
            try? await Task.sleep(for: .seconds(1.2))
            guard !Task.isCancelled else { return }
            copied = false
        }
    }
}

/// Reasoning: open while it streams, folded to "Thought for 4 s" once the model moves on.
/// A click keeps it the way the reader chose.
private struct ThinkingBlock: View, Equatable {
    let text: String
    let streaming: Bool
    let elapsed: Double?
    let textScale: CGFloat
    /// nil follows the stream; a click pins it.
    @State private var chosen: Bool?
    @State private var hovered = false
    @State private var chunkCache = StreamingTextChunker.Cache(markdown: false)
    @State private var scroll = StreamingScrollState()

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.text == rhs.text && lhs.streaming == rhs.streaming && lhs.elapsed == rhs.elapsed
            && lhs.textScale == rhs.textScale
    }

    private var title: String {
        if streaming { return L("Réflexion en cours", "Thinking") }
        if let elapsed, elapsed >= 1 {
            let duration = ActivityMetrics.duration(elapsed)
            return L("Réflexion · \(duration)", "Thought for \(duration)")
        }
        return L("Réflexion", "Thought")
    }

    var body: some View {
        let expanded = chosen ?? streaming
        VStack(alignment: .leading, spacing: 4) {
            Button {
                withAnimation(.easeOut(duration: 0.16)) { chosen = !expanded }
            } label: {
                HStack(alignment: .firstTextBaseline, spacing: ActivityMetrics.spacing) {
                    ActivityIcon(symbol: "brain")
                    StudioShimmerText(text: title, font: .system(size: 13 * textScale, weight: .medium),
                                      active: streaming)
                    ActivityChevron(expanded: expanded, visible: hovered)
                }
            }
            .buttonStyle(ActivityLineStyle(hovered: hovered))
            .onHover { hovered = $0 }
            .accessibilityLabel(title)
            .accessibilityValue(expanded ? L("développé", "expanded") : L("replié", "collapsed"))
            if expanded {
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
                    .padding(.vertical, 4)
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
                .mask(StudioEdgeFade(top: 6, bottom: 6))
                .padding(.leading, ActivityMetrics.indent)
                // The thread under the icon ties the text to its line.
                .overlay(alignment: .leading) {
                    Rectangle().fill(StudioTheme.edge)
                        .frame(width: 1)
                        .padding(.leading, ActivityMetrics.iconWidth / 2 - 0.5)
                        .padding(.vertical, 2)
                }
                .transition(.opacity)
            }
        }
    }
}

private struct ThinkingTextChunk: View, Equatable {
    @Environment(\.studioStyle) private var style
    let source: String
    let streaming: Bool
    let textScale: CGFloat

    nonisolated static func == (lhs: Self, rhs: Self) -> Bool {
        lhs.source == rhs.source && lhs.streaming == rhs.streaming && lhs.textScale == rhs.textScale
    }

    var body: some View {
        SmoothTokenText(content: Text(source), source: source, streaming: streaming)
            // Graphite: proportional type, de-emphasised; mono belongs to code.
            .font(style.isGraphite ? .system(size: 13 * textScale, weight: .regular)
                  : .system(size: 11.5 * textScale, weight: .regular, design: .monospaced))
            .foregroundStyle(style.isGraphite ? StudioTheme.contrast.opacity(0.62) : StudioTheme.secondary)
            .lineSpacing((style.isGraphite ? 4 : 3) * textScale)
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
                           label: String(format: L("DFlash2 %.0f%% · %.2f tok/bloc", "DFlash2 %.0f%% · %.2f tok/block"),
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
    @Environment(\.studioStyle) private var style
    let icon: String
    let value: String
    let label: String

    var body: some View {
        if style.isGraphite {
            // Numbers, not an icon per chip.
            HStack(spacing: 5) {
                Text(value).foregroundStyle(StudioTheme.secondary)
                Text(label).foregroundStyle(StudioTheme.quiet)
            }
            .font(.system(size: 11))
            .frame(height: 22)
            .monospacedDigit()
            .accessibilityElement(children: .combine)
        } else {
            chip
        }
    }

    private var chip: some View {
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
    @Environment(\.studioStyle) private var style
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
                TextField(studio.canSteerActiveAgent ? L("Précisez la tâche en cours…", "Add an instruction…")
                    : studio.selectedMode == .agent ? L("Décrivez votre tâche…", "Describe your task…")
                    : L("Écrivez un message…", "Write a message…"), text: $studio.draft, axis: .vertical)
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
                    .disabled(studio.composerBusy && !studio.canSteerActiveAgent)
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
            guard !studio.composerBusy || studio.canSteerActiveAgent else { return }
            focused = true
        }
        .background {
            // Two shadows: a tight contact shadow and a wide ambient lift (softer in Graphite).
            let graphite = style.isGraphite
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .fill(StudioTheme.composer)
                .shadow(color: .black.opacity(graphite ? 0.06 : 0.08), radius: graphite ? 1 : 2, y: 1)
                .shadow(color: .black.opacity(focused ? (graphite ? 0.12 : 0.16) : (graphite ? 0.08 : 0.11)),
                        radius: focused ? (graphite ? 18 : 22) : (graphite ? 14 : 16), y: focused ? (graphite ? 8 : 10) : (graphite ? 6 : 7))
        }
        .overlay {
            // Graphite keeps the focus ring neutral: the accent is not decoration.
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .strokeBorder(
                    dropTargeted ? AnyShapeStyle(studio.accentColor)
                        : style.isGraphite ? AnyShapeStyle(StudioTheme.contrast.opacity(focused ? 0.18 : 0.10))
                        : focused ? AnyShapeStyle(studio.accentColor.opacity(0.45))
                        : AnyShapeStyle(StudioTheme.contrast.opacity(0.12)),
                    lineWidth: dropTargeted ? 2 : style.isGraphite ? 0.5 : 1)
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


/// The model’s native thinking switch or effort levels.
private struct ReasoningMenu: View {
    @EnvironmentObject private var studio: StudioModel
    let support: ReasoningSupport
    let level: ReasoningEffort

    private var currentLabel: String {
        if studio.effectiveReasoningChoice == nil { return L("Réflexion auto", "Auto thinking") }
        return support.title(for: level, inComposer: true)
    }

    var body: some View {
        Menu {
            Picker(L("Raisonnement", "Reasoning"), selection: Binding(
                get: { studio.effectiveReasoningChoice }, set: { choice in
                    if let choice { studio.setReasoningLevel(choice) } else { studio.resetReasoningLevel() }
                })) {
                Text(L("Par défaut du modèle", "Model default")).tag(nil as ReasoningEffort?)
                ForEach(studio.reasoningLevels) { option in
                    Label(support.title(for: option), systemImage: option.symbol)
                    .tag(Optional(option))
                }
            }
            .pickerStyle(.inline)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: level.symbol)
                    .foregroundStyle(level == .none ? StudioTheme.quiet : studio.accentColor)
                Text(currentLabel)
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
        .accessibilityValue(currentLabel)
        .help(L("Profondeur de raisonnement du modèle", "Model reasoning depth"))
    }
}

/// Agent: the task's folder and its Internet switch as icons beside Tools, in the
/// composer itself rather than in a second bar under it.
private struct ComposerAgentControls: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
        let internet = studio.agentPermissions.internet
        HStack(spacing: 2) {
            if let folder = studio.agentDisplayFolder {
                Button {
                    if let workspace = studio.agentWorkspaceURL {
                        studio.studioDocument = StudioDocument(workspace: workspace, showPermissions: true)
                    }
                } label: {
                    Label(folder.lastPathComponent, systemImage: "folder")
                        .font(.system(size: 12, weight: .medium)).lineLimit(1).truncationMode(.middle)
                        .frame(maxWidth: 140).fixedSize(horizontal: true, vertical: false)
                        .padding(.horizontal, 7).frame(height: 28)
                }
                .buttonStyle(WorkspaceControlStyle())
                .disabled(studio.isGenerating || studio.agentStudioBusy)
                .help(folder.path + "\n" + L("Voir les autorisations de cette tâche", "View permissions for this task"))
                .accessibilityLabel(L("Autorisations du dossier ", "Permissions for folder ") + folder.lastPathComponent)
                .accessibilityIdentifier("agent.composer.folder")
            } else {
                Button(action: studio.openAgentWorkspace) {
                    Image(systemName: "folder")
                        .foregroundStyle(StudioTheme.secondary)
                        .frame(width: 28, height: 28)
                }
                .buttonStyle(WorkspaceControlStyle())
                .help(L("Ouvrir le dossier de la tâche dans le Finder", "Open the task’s folder in Finder"))
                .accessibilityLabel(L("Dossier de la tâche", "Task folder"))
            }

            Button { studio.setAgentInternet(!internet) } label: {
                Image(systemName: internet ? "globe" : "network.slash")
                    .foregroundStyle(internet ? StudioTheme.ink : StudioTheme.quiet)
                    .frame(width: 28, height: 28)
            }
            .buttonStyle(WorkspaceControlStyle(selected: internet))
            .disabled(studio.isGenerating || studio.agentStudioBusy)
            .help(internet
                ? L("Internet activé pour cette tâche : commandes (pip, npm, git clone), fetch et recherche web. Cliquer pour le couper.",
                    "Internet is on for this task: commands (pip, npm, git clone), fetch and web search. Click to turn it off.")
                : L("Internet coupé pour cette tâche. Cliquer pour l’activer.", "Internet is off for this task. Click to turn it on."))
            .accessibilityLabel("Internet")
            .accessibilityValue(internet ? L("activé", "on") : L("coupé", "off"))
            .accessibilityIdentifier("agent.composer.internet")
        }
        .font(.system(size: 13))
    }
}

private struct ComposerFooterView: View {
    @EnvironmentObject private var studio: StudioModel

    var body: some View {
            HStack(spacing: 7) {
                Menu {
                    Button(action: studio.chooseChatFiles) {
                        Label(studio.selectedMode == .agent
                              ? L("Joindre des fichiers à lire…", "Attach files to read…")
                              : L("Joindre des fichiers…", "Attach files…"), systemImage: "paperclip")
                    }
                    .disabled(!studio.canImportFiles)
                    .keyboardShortcut("o", modifiers: [.command, .shift])
                    if studio.selectedMode == .agent {
                        Button(action: studio.chooseAgentFolder) {
                            Label(L("Travailler dans un dossier…", "Work in a folder…"), systemImage: "folder.badge.plus")
                        }
                        .disabled(studio.isGenerating || studio.agentStudioBusy)
                    } else {
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
                if studio.selectedMode == .agent {
                    ComposerAgentControls()
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
                    if studio.canSteerActiveAgent && !studio.draft.isEmpty {
                        Button(action: studio.send) {
                            Image(systemName: "arrow.turn.down.right")
                                .frame(width: 31, height: 31)
                        }
                        .buttonStyle(RoundGlassButtonStyle(bright: false))
                        .disabled(!studio.canQueueAgentGuidance)
                        .help(L("Transmettre la précision (Retour)", "Send instruction (Return)"))
                        .accessibilityLabel(L("Transmettre la précision", "Send instruction"))
                        .accessibilityIdentifier("agent.sendGuidance")
                    }
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
