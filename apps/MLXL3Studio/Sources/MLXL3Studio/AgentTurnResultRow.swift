import SwiftUI

/// The transcript links to the snapshots of this exchange, even after later edits.
struct AgentTurnResultRow: View {
  let workspace: URL
  let result: AgentTurnResult
  @EnvironmentObject private var studio: StudioModel

  var body: some View {
    VStack(alignment: .leading, spacing: 6) {
      if !result.files.isEmpty {
        ViewThatFits(in: .horizontal) {
          HStack(spacing: 10) { fileActions(result) }
          VStack(alignment: .leading, spacing: 4) { fileActions(result) }
        }
      }
      commandLabels(result.commands.tests, tests: true)
      commandLabels(result.commands.commands, tests: false)
    }
    .font(.system(size: 12, weight: .medium))
    .foregroundStyle(StudioTheme.secondary)
  }

  @ViewBuilder private func fileActions(_ result: AgentTurnResult) -> some View {
    let count = result.files.count
    if let path = result.newDocument {
      Button {
        studio.studioDocument = StudioDocument(workspace: workspace,
          file: workspace.appendingPathComponent(path), turnResult: result)
      } label: {
        Label(URL(fileURLWithPath: path).lastPathComponent, systemImage: "doc")
          .lineLimit(1).truncationMode(.middle)
      }.buttonStyle(.studio).help(L("Ouvrir le document de ce tour", "Open this turn’s document"))
        .accessibilityLabel(L("Ouvrir \(path)", "Open \(path)"))
        .accessibilityIdentifier("agent.turn.document")
    }
    Button {
      studio.studioDocument = StudioDocument(workspace: workspace, turnResult: result)
    } label: {
      Label(count == 1 ? L("Examiner 1 fichier", "Review 1 file")
          : L("Examiner \(count) fichiers", "Review \(count) files"), systemImage: "plus.forwardslash.minus")
    }.buttonStyle(.studio).accessibilityIdentifier("agent.turn.review")
      .accessibilityHint(L("Afficher les modifications de cette réponse", "Show the changes made during this response"))
  }

  @ViewBuilder private func commandLabels(_ counts: [String: Int], tests: Bool) -> some View {
    if !counts.isEmpty {
      let singular = counts.count == 1 && counts.values.first == 1
      let title = tests
        ? (singular ? L("Commande de test", "Test command") : L("Commandes de test", "Test commands"))
        : (singular ? L("Commande", "Command") : L("Commandes", "Commands"))
      let items: [(AgentCommandReceipt.Outcome, String)] = [
        (.failure, L("en échec", "failed")),
        (.interrupted, L(counts["interrupted"] == 1 ? "interrompue" : "interrompues", "interrupted")),
        (.unconfirmed, L("sans résultat confirmé", "unconfirmed")),
        (.success, L(counts["success"] == 1 ? "réussie" : "réussies", "succeeded")),
      ]
      let detail = items.compactMap { outcome, label in counts[outcome.rawValue].map { "\($0) \(label)" } }
        .joined(separator: " · ")
      Label("\(title) · \(detail)", systemImage: tests ? "checklist" : "terminal")
        .foregroundStyle((counts["failure"] ?? 0) > 0 ? StudioTheme.failure : StudioTheme.secondary)
    }
  }
}
