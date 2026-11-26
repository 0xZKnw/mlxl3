import SwiftUI

/// The transcript links to the snapshots of this exchange, even after later edits.
struct AgentTurnResultRow: View {
  let workspace: URL
  let result: AgentTurnResult
  @EnvironmentObject private var studio: StudioModel

  var body: some View {
    // Only the files: how many commands failed or succeeded is not shown under an answer
    // (each command and its result is already in the turn's steps).
    ViewThatFits(in: .horizontal) {
      HStack(spacing: 10) { fileActions(result) }
      VStack(alignment: .leading, spacing: 4) { fileActions(result) }
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
}
